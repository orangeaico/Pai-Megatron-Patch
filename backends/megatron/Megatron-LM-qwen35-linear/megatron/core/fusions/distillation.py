# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""Knowledge Distillation with Tensor Parallelism for Large Language Models.

This module provides efficient knowledge distillation functionality for large language models
with support for tensor parallelism and memory-efficient processing. It adapts Megatron-LM's
tensor-parallel vocabulary sharding to work with apple/ml-cross-entropy's cross-entropy
computation for scalable distillation training.

The module implements both traditional knowledge distillation and an efficient chunked approach
that processes sequences in memory-friendly chunks while maintaining mathematical equivalence.
All computations are compatible with Megatron-LM's distributed training paradigms.

Key Features:
    - Tensor-parallel vocabulary processing for large vocabularies
    - Memory-efficient chunked sequence processing
    - Support for both standard and temperature-scaled distillation
    - Automatic teacher data generation for testing
    - Debug utilities for monitoring loss computation

Dependencies:
    pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"

References:
    - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
    - Repository: https://github.com/apple/ml-cross-entropy

Example:
    Basic usage for knowledge distillation:
    
    >>> # Standard distillation loss computation
    >>> kl_loss, teacher_data = distillation_loss(
    ...     embeddings=student_embeddings,
    ...     classifier_weight=vocab_projection_weights,
    ...     labels=target_labels,
    ...     vocab_size=50000,
    ...     teacher_data=teacher_logits_data,
    ...     temperature=3.0
    ... )
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from contextlib import contextmanager
import time
import torch
import torch.nn.functional as F

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_context_parallel_world_size,
    get_context_parallel_rank,
)
from megatron.core.tensor_parallel import gather_from_tensor_model_parallel_region
from megatron.core.fusions.cce_loss import cce_per_token_loss

# Constants
DEFAULT_TEMPERATURE = 1.0
DEFAULT_CHUNK_SIZE = 8192
DEFAULT_IGNORE_INDEX = -100
DEFAULT_NUM_TEACHER_TOKENS = 50


def _get_rank_for_logging() -> int:
    """Best-effort retrieval of the distributed rank for gated logging."""
    if not torch.distributed.is_available():
        return 0
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def _debug_print(message: str, enabled: bool) -> None:
    """Print from rank0 only when debug logging is enabled."""
    if not enabled or not message:
        return
    if _get_rank_for_logging() == 0:
        print(message)


@contextmanager
def _timed_section(name: str, enabled: bool, sync_cuda: bool = False):
    """Context manager that prints elapsed time when debug mode is active."""
    if not enabled:
        yield
        return

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        stream = torch.cuda.current_stream()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        if sync_cuda:
            torch.cuda.synchronize()
        start_event.record(stream)
        try:
            yield
        finally:
            end_event.record(stream)
            end_event.synchronize()
            elapsed_ms = start_event.elapsed_time(end_event)
            if sync_cuda:
                torch.cuda.synchronize()
            _debug_print(f"[distill-timer] {name}: {elapsed_ms:.3f} ms", enabled)
    else:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            _debug_print(f"[distill-timer] {name}: {elapsed_ms:.3f} ms", enabled)


@dataclass(frozen=True)
class TeacherBatchTensors:
    """Preprocessed teacher payload for fast lookups."""
    positions: torch.Tensor           # [N] global token positions (sorted)
    row_ptr: torch.Tensor             # [N + 1] prefix sums into flattened indices/values
    indices: torch.Tensor             # [nnz] teacher vocab indices (global or local depending on caller)
    values: torch.Tensor              # [nnz] teacher logits


def distillation_loss(
    *,
    embeddings: torch.Tensor,
    classifier_weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    impl: str = "cce",
    reduction: str = "none", 
    shift: bool = True,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
    teacher_data: Optional[List[Dict[str, Any]]] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    debug: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Compute knowledge distillation loss using Cross-Entropy with Efficient implementation.
    
    This function computes both the standard cross-entropy loss and KL divergence loss
    for knowledge distillation. It supports tensor parallelism for distributed training
    and processes sequences in chunks to manage memory usage.
    
    Args:
        embeddings: Input embeddings tensor of shape [B, T, H] or [T, B, H] or [B, T_local, H] with sequence parallelism
        classifier_weight: Weight matrix of shape [V, H] (or [V_local, H] with vocab parallelism)
        labels: Target labels of shape [B, T_global] (global with respect to sequence parallelism)
        vocab_size: Total vocabulary size
        impl: Implementation type for CCE loss computation (default: "cce")
        reduction: Reduction method for loss (default: "none")
        shift: Whether to shift labels for next-token prediction (default: True)
        ignore_index: Index to ignore in loss computation (default: -100)
        teacher_data: List of teacher data dictionaries, one per batch element.
                     Each dict contains 'positions', 'indices', 'values' keys and may
                     optionally include a precomputed 'row_ptr' tensor for fast lookup.
        temperature: Temperature parameter for softmax (default: 1.0)
        chunk_size: Sequence chunk size for memory-efficient processing (default: 1024)
        debug: Enable debug output (default: False)
        
    Returns:
        - kl_losses: Per-token KL divergence losses of shape [B, T]
        
    Raises:
        ValueError: If teacher_data is None and cannot be generated
    """
    # Compute standard cross-entropy loss with log-sum-exp values
    with _timed_section("cce_per_token_loss", debug, sync_cuda=True):
        cross_entropy_loss, log_sum_exp_values = cce_per_token_loss(
            embeddings=embeddings,
            classifier_weight=classifier_weight,
            labels=labels,
            vocab_size=vocab_size,
            impl=impl,
            reduction=reduction,
            shift=int(shift),
            ignore_index=ignore_index,
            return_lse=True,
            temperature=temperature,
        )
    # Transpose embeddings to batch-first format: [B, S, H]
    batch_first_embeddings = embeddings.transpose(0, 1).contiguous()
    
    # Get tensor parallelism configuration
    tensor_parallel_world_size = get_tensor_model_parallel_world_size()
    is_tensor_parallel = tensor_parallel_world_size > 1
    
    device = batch_first_embeddings.device
    batch_size, sequence_length, hidden_size = batch_first_embeddings.shape

    # Initialize KL loss tensor
    kl_loss_tensor = torch.zeros(
        (batch_size, sequence_length), 
        device=device, 
        dtype=torch.float32
    )
    if is_tensor_parallel:
        classifier_weight = classifier_weight.t()
        classifier_weight = gather_from_tensor_model_parallel_region(classifier_weight)
        classifier_weight = classifier_weight.t().contiguous()
    # Process each batch element separately
    for batch_idx in range(batch_size):
        with _timed_section(f"prepare_teacher_batch[{batch_idx}]", debug):
            teacher_batch = _prepare_teacher_batch_tensors(
                teacher_entry=teacher_data[batch_idx],
                device=device,
            )
        with _timed_section(f"process_batch[{batch_idx}]", debug, sync_cuda=True):
            _process_batch_element_kl_loss(
                batch_idx=batch_idx,
                batch_embeddings=batch_first_embeddings[batch_idx],
                batch_labels=labels[batch_idx],
                batch_log_sum_exp=log_sum_exp_values[batch_idx].view(-1),
                batch_teacher_data=teacher_batch,
                classifier_weight=classifier_weight,
                kl_loss_tensor=kl_loss_tensor,
                temperature=temperature,
                chunk_size=chunk_size,
                debug=debug,
                ignore_index=ignore_index,
            )

    if debug:
        _print_debug_summary(kl_loss_tensor)

    return kl_loss_tensor


def _prepare_teacher_batch_tensors(
    *,
    teacher_entry: Dict[str, Any],
    device: torch.device,
) -> TeacherBatchTensors:
    """Convert teacher payload into ragged tensors on the target device."""
    empty_tensor_long = torch.empty(0, dtype=torch.long, device=device)
    empty_tensor_float = torch.empty(0, dtype=torch.float32, device=device)

    if teacher_entry is None:
        return TeacherBatchTensors(
            positions=empty_tensor_long,
            row_ptr=torch.zeros(1, dtype=torch.long, device=device),
            indices=empty_tensor_long,
            values=empty_tensor_float,
        )

    positions = teacher_entry.get('positions', [])
    indices_list = teacher_entry.get('indices', [])
    values_list = teacher_entry.get('values', [])
    row_ptr_tensor = teacher_entry.get('row_ptr')

    if row_ptr_tensor is not None:
        pos_tensor = positions.flatten()
        if pos_tensor.numel() == 0:
            return TeacherBatchTensors(
                positions=empty_tensor_long,
                row_ptr=torch.zeros(1, dtype=torch.long, device=device),
                indices=empty_tensor_long,
                values=empty_tensor_float,
            )
        return TeacherBatchTensors(
            positions=pos_tensor,
            row_ptr=row_ptr_tensor,
            indices=indices_list,
            values=values_list,
        )

    if not positions or not indices_list or not values_list:
        return TeacherBatchTensors(
            positions=empty_tensor_long,
            row_ptr=torch.zeros(1, dtype=torch.long, device=device),
            indices=empty_tensor_long,
            values=empty_tensor_float,
        )


def _process_batch_element_kl_loss(
    batch_idx: int,
    batch_embeddings: torch.Tensor,
    batch_labels: torch.Tensor,
    batch_log_sum_exp: torch.Tensor,
    batch_teacher_data: TeacherBatchTensors,
    classifier_weight: torch.Tensor,
    kl_loss_tensor: torch.Tensor,
    temperature: float,
    chunk_size: int,
    debug: bool,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> None:
    """
    Process KL loss computation for a single batch element.
    
    Args:
        batch_idx: Index of current batch element
        batch_embeddings: Embeddings for this batch element [T, H]
        batch_labels: Labels for this batch element [T]
        batch_log_sum_exp: Log-sum-exp values for this batch element [T]
        batch_teacher_data: Teacher data for this batch element
        classifier_weight: Classifier weight matrix
        kl_loss_tensor: Output tensor for KL losses
        temperature: Temperature parameter
        chunk_size: Processing chunk size
        ignore_index: Index to ignore
        debug: Enable debug output
    """
    # Nothing to do if no teacher payload for this batch.
    if batch_teacher_data.positions.numel() == 0:
        return

    sequence_length = batch_embeddings.size(0)
    
    # Process sequence in chunks for memory efficiency
    total_chunks = 0
    processed_chunks = 0
    extract_elapsed_ms = 0.0
    compute_elapsed_ms = 0.0

    for chunk_start in range(0, sequence_length, chunk_size):
        chunk_end = min(chunk_start + chunk_size, sequence_length)
        total_chunks += 1

        cuda_timing_enabled = debug and torch.cuda.is_available()
        if cuda_timing_enabled:
            stream = torch.cuda.current_stream()
            extract_start_event = torch.cuda.Event(enable_timing=True)
            extract_end_event = torch.cuda.Event(enable_timing=True)
            extract_start_event.record(stream)
        elif debug:
            extract_wall_start = time.perf_counter()

        chunk_data = _extract_chunk_teacher_data(
            teacher_batch=batch_teacher_data,
            batch_labels=batch_labels,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            temperature=temperature,
            debug=debug,
            ignore_index=ignore_index
        )
        if cuda_timing_enabled:
            extract_end_event.record(stream)
            extract_end_event.synchronize()
            extract_elapsed_ms += extract_start_event.elapsed_time(extract_end_event)
        elif debug:
            extract_elapsed_ms += (time.perf_counter() - extract_wall_start) * 1000.0

        if chunk_data is None:
            continue
        processed_chunks += 1
            
        if cuda_timing_enabled:
            compute_start_event = torch.cuda.Event(enable_timing=True)
            compute_end_event = torch.cuda.Event(enable_timing=True)
            compute_start_event.record(stream)
        elif debug:
            compute_wall_start = time.perf_counter()

        _compute_chunk_kl_losses(
            batch_idx=batch_idx,
            chunk_data=chunk_data,
            batch_embeddings=batch_embeddings,
            batch_log_sum_exp=batch_log_sum_exp,
            classifier_weight=classifier_weight,
            kl_loss_tensor=kl_loss_tensor,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            temperature=temperature,
            debug=debug,
        )
        if cuda_timing_enabled:
            compute_end_event.record(stream)
            compute_end_event.synchronize()
            compute_elapsed_ms += compute_start_event.elapsed_time(compute_end_event)
        elif debug:
            compute_elapsed_ms += (time.perf_counter() - compute_wall_start) * 1000.0

    if debug:
        _debug_print(
            "[distill-timer] batch "
            f"{batch_idx} chunk_loop: total={total_chunks}, processed={processed_chunks}, "
            f"extract={extract_elapsed_ms:.3f} ms, compute={compute_elapsed_ms:.3f} ms",
            debug,
        )


def _extract_chunk_teacher_data(
    teacher_batch: TeacherBatchTensors,
    batch_labels: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
    temperature: float,
    debug: bool,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Extract teacher tokens intersecting a given chunk using tensor layouts.
    """
    if teacher_batch.positions.numel() == 0:
        return None

    device = batch_labels.device
    local_positions = torch.arange(
        chunk_start, chunk_end, device=device, dtype=torch.long
    )
    if local_positions.numel() == 0:
        return None

    # Map local positions to global indices when context parallelism is active.
    cp_world_size = get_context_parallel_world_size()
    if cp_world_size > 1:
        cp_rank = get_context_parallel_rank()
        local_sequence_length = batch_labels.size(0)
        first_span_length = local_sequence_length // 2
        second_span_length = local_sequence_length - first_span_length

        if first_span_length == 0:
            global_positions = local_positions
        else:
            first_span_start = cp_rank * first_span_length
            second_span_start = (2 * cp_world_size - cp_rank - 1) * first_span_length

            in_first = local_positions < first_span_length
            in_second = (~in_first) & (
                (local_positions - first_span_length) < second_span_length
            )

            global_positions = torch.full_like(local_positions, -1)
            global_positions[in_first] = first_span_start + local_positions[in_first]

            second_offsets = local_positions[in_second] - first_span_length
            global_positions[in_second] = second_span_start + second_offsets
    else:
        global_positions = local_positions

    # Filter out ignored labels and positions without global mapping.
    valid_mask = (batch_labels[local_positions] != ignore_index) & (global_positions >= 0)
    if not torch.any(valid_mask):
        return None
    local_positions = local_positions[valid_mask]
    global_positions = global_positions[valid_mask]

    if global_positions.numel() == 0:
        return None

    # Locate teacher payload by searching into sorted positions tensor.
    search_idx = torch.searchsorted(teacher_batch.positions, global_positions)
    total_teacher_positions = teacher_batch.positions.size(0)
    valid_search = search_idx < total_teacher_positions
    matched_positions = torch.empty_like(search_idx, device=device)
    if torch.any(valid_search):
        matched_positions[valid_search] = teacher_batch.positions.index_select(
            0, search_idx[valid_search]
        )
    if torch.any(~valid_search):
        matched_positions[~valid_search] = -1
    has_match = valid_search & (matched_positions == global_positions)
    if not torch.any(has_match):
        return None

    local_positions = local_positions[has_match]
    matched_indices = search_idx[has_match]

    starts = teacher_batch.row_ptr[matched_indices]
    ends = teacher_batch.row_ptr[matched_indices + 1]
    lengths = ends - starts

    positive_length_mask = lengths > 0
    if not torch.any(positive_length_mask):
        return None

    local_positions = local_positions[positive_length_mask]
    matched_indices = matched_indices[positive_length_mask]
    starts = starts[positive_length_mask]
    lengths = lengths[positive_length_mask]
    global_positions = global_positions[has_match][positive_length_mask]

    num_positions = matched_indices.numel()
    total_tokens = int(lengths.sum().item())
    if total_tokens == 0:
        return None

    position_ids = torch.repeat_interleave(
        torch.arange(num_positions, device=device, dtype=torch.long),
        lengths,
    )
    lengths_cumsum = lengths.cumsum(0)
    relative_offsets = torch.arange(total_tokens, device=device, dtype=torch.long)
    relative_offsets = relative_offsets - torch.repeat_interleave(
        lengths_cumsum - lengths, lengths
    )
    flat_indices = torch.repeat_interleave(starts, lengths) + relative_offsets

    teacher_vocab_indices = teacher_batch.indices[flat_indices]
    teacher_logits = teacher_batch.values[flat_indices]

    total_tokens = int(teacher_vocab_indices.numel())
    if total_tokens == 0 or num_positions == 0:
        return None

    lengths_cumsum = lengths.cumsum(0)
    relative_offsets = torch.arange(total_tokens, device=device, dtype=torch.long)
    relative_offsets = relative_offsets - torch.repeat_interleave(
        lengths_cumsum - lengths, lengths
    )

    max_length = int(lengths.max().item())
    teacher_mask = torch.zeros(
        (num_positions, max_length),
        dtype=torch.bool,
        device=device,
    )
    teacher_mask[position_ids, relative_offsets] = True

    padded_indices = torch.full(
        (num_positions, max_length),
        -1,
        dtype=torch.long,
        device=device,
    )
    padded_indices[position_ids, relative_offsets] = teacher_vocab_indices

    scaled_teacher_logits = torch.full(
        (num_positions, max_length),
        float('-inf'),
        dtype=teacher_logits.dtype,
        device=device,
    )
    scaled_teacher_logits[position_ids, relative_offsets] = teacher_logits / temperature
    teacher_log_probs = F.log_softmax(scaled_teacher_logits, dim=-1)

    # Exclude non-finite teacher tokens (e.g., -inf) from any downstream KL terms.
    # This avoids 0 * (-inf) -> NaN when computing KL with masked zeros.
    finite_token_mask = torch.isfinite(teacher_log_probs)
    teacher_mask = teacher_mask & finite_token_mask

    teacher_log_probs = torch.where(
        teacher_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs)
    )

    if debug:
        curr_rank = torch.distributed.get_rank()
        print(
            f"Rank {curr_rank} Chunk [{chunk_start}:{chunk_end}] "
            f"valid positions: {num_positions}"
        )

    return {
        'positions': local_positions,
        'teacher_indices': padded_indices,
        'teacher_log_probs': teacher_log_probs,
        'teacher_mask': teacher_mask,
        'global_positions': global_positions,
    }


def _compute_chunk_kl_losses(
    batch_idx: int,
    chunk_data: Dict[str, torch.Tensor],
    batch_embeddings: torch.Tensor,
    batch_log_sum_exp: torch.Tensor,
    classifier_weight: torch.Tensor,
    kl_loss_tensor: torch.Tensor,
    chunk_start: int,
    chunk_end: int,
    temperature: float,
    debug: bool,
) -> None:
    """
    Compute KL divergence losses for a chunk of positions.
    
    Args:
        batch_idx: Batch element index
        chunk_data: Extracted chunk data
        batch_embeddings: Embeddings for the batch element
        batch_log_sum_exp: Log-sum-exp values for the batch element
        classifier_weight: Classifier weight matrix
        kl_loss_tensor: Output tensor for KL losses
        chunk_start: Start index of chunk
        chunk_end: End index of chunk
        temperature: Temperature parameter
        debug: Enable debug output
    """
    positions = chunk_data['positions']
    num_positions = positions.numel()
    if num_positions == 0:
        return

    padded_teacher_indices = chunk_data['teacher_indices']
    teacher_log_probs = chunk_data['teacher_log_probs']
    teacher_mask = chunk_data['teacher_mask']

    valid_teacher_indices = padded_teacher_indices[teacher_mask]
    if valid_teacher_indices.numel() == 0:
        return
    
    unique_teacher_indices, inverse = torch.unique(
        valid_teacher_indices, sorted=True, return_inverse=True
    )
    teacher_columns = torch.full_like(padded_teacher_indices, -1)
    teacher_columns[teacher_mask] = inverse

    chunk_embeddings = batch_embeddings[chunk_start:chunk_end]
    union_classifier_weights = classifier_weight.index_select(0, unique_teacher_indices)
    chunk_logits = chunk_embeddings @ union_classifier_weights.t().contiguous()

    positions_in_chunk_tensor = positions - chunk_start
    position_logits = chunk_logits[positions_in_chunk_tensor]

    teacher_columns_safe = torch.where(
        teacher_mask, teacher_columns, torch.zeros_like(teacher_columns)
    )
    gathered_student_logits = torch.take_along_dim(
        position_logits, teacher_columns_safe, dim=1
    )
    gathered_student_logits = torch.where(
        teacher_mask, gathered_student_logits, torch.zeros_like(gathered_student_logits)
    )

    position_log_sum_exp = batch_log_sum_exp[positions].unsqueeze(1)
    student_log_probs = torch.where(
        teacher_mask,
        gathered_student_logits / temperature - position_log_sum_exp,
        torch.zeros_like(gathered_student_logits),
    )

    teacher_probs = torch.where(
        teacher_mask,
        torch.exp(teacher_log_probs),
        torch.zeros_like(teacher_log_probs),
    )

    kl_per_token = teacher_probs * (teacher_log_probs - student_log_probs)
    kl_per_position = kl_per_token.sum(dim=1)
    
    # Apply temperature scaling
    kl_per_position_scaled = kl_per_position * (temperature ** 2)
    
    # Store results in output tensor
    kl_loss_tensor[batch_idx, positions] = kl_per_position_scaled
    
    if debug:
        _print_chunk_debug_info(
            batch_idx,
            positions,
            chunk_data.get('global_positions'),
            teacher_log_probs,
            student_log_probs,
            teacher_mask,
            position_log_sum_exp,
            kl_loss_tensor,
        )


def _print_chunk_debug_info(
    batch_idx: int,
    positions: torch.Tensor,
    global_positions: Optional[torch.Tensor],
    teacher_log_probs: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_mask: torch.Tensor,
    log_sum_exp_values: torch.Tensor,
    kl_losses: torch.Tensor,
) -> None:
    """Print debug information for chunk processing."""
    debug_position = 7365
    search_positions = global_positions if global_positions is not None else positions
    matches = torch.nonzero(search_positions == debug_position, as_tuple=False)
    if matches.numel() == 0:
        return

    position_index = int(matches[0].item())
    local_pos = int(positions[position_index].item())
    global_pos = int(search_positions[position_index].item())
    valid_mask = teacher_mask[position_index]
    teacher_probs = torch.exp(teacher_log_probs[position_index][valid_mask])
    student_logp = student_log_probs[position_index][valid_mask]
    print(
        f"batch {batch_idx} efficient pos local {local_pos} global {global_pos} "
        f"teacher_values_batch: {teacher_probs}"
    )
    print(
        f"batch {batch_idx} efficient pos {debug_position} "
        f"s_logpK: {student_logp} with logZ {log_sum_exp_values[position_index]}"
    )
    print(
        f"batch {batch_idx} efficient pos {debug_position} "
        f"kl_per_pos: {kl_losses[batch_idx, local_pos]}"
    )


def _print_debug_summary(
    kl_losses: torch.Tensor, 
) -> None:
    """Print debug summary information."""
    print(f"  KL losses: {kl_losses.sum()}")


def generate_fake_teacher_data(
    batch_size: int,
    seq_length: int,
    vocab_size: int,
    num_teacher_tokens_per_pos: int = 3,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> List[Dict[str, List]]:
    """
    Generate synthetic teacher data for testing distillation functionality.
    
    This function creates fake teacher data that mimics the structure expected
    by the distillation loss computation. Each batch element gets teacher data
    for all sequence positions.
    
    Args:
        batch_size: Number of batch elements
        seq_length: Length of each sequence
        vocab_size: Total vocabulary size
        num_teacher_tokens_per_pos: Number of teacher tokens per position (default: 3)
        device: Target device for tensors (default: CPU)
        dtype: Data type for teacher values (default: float32)
        
    Returns:
        List of teacher data dictionaries, one per batch element.
        Each dictionary contains:
        - 'positions': List of sequence positions
        - 'indices': List of teacher token indices for each position
        - 'values': List of teacher probability values for each position
    """
    if device is None:
        device = torch.device('cpu')
    
    teacher_data = []
    
    for batch_idx in range(batch_size):
        # Generate positions for all sequence positions
        sequence_positions = torch.arange(0, seq_length, device=device)
        
        batch_teacher_data = {
            'positions': [],
            'indices': [],
            'values': []
        }
        
        for position in sequence_positions:
            # Generate random vocabulary indices (sorted for consistency)
            random_teacher_indices, _ = torch.sort(
                torch.randint(0, vocab_size, (num_teacher_tokens_per_pos,), device=device)
            )
            
            # Generate random values that will be normalized to probabilities
            random_teacher_values = torch.rand(
                num_teacher_tokens_per_pos, device=device, dtype=dtype
            )
            
            batch_teacher_data['positions'].append(position)
            batch_teacher_data['indices'].append(random_teacher_indices)
            batch_teacher_data['values'].append(random_teacher_values)
        
        teacher_data.append(batch_teacher_data)
    
    return teacher_data


def traditional_distillation_loss(
    student_logits: torch.Tensor,
    teacher_data: List[Dict[str, Any]],
    labels: torch.Tensor,
    T: float = DEFAULT_TEMPERATURE,
    ignore_index: int = DEFAULT_IGNORE_INDEX,
) -> torch.Tensor:
    """
    Compute knowledge distillation loss using traditional approach.
    
    This function implements the standard Knowledge Distillation approach
    from Hinton et al. (2015), computing KL divergence between teacher
    and student distributions at specified temperature.
    
    The loss is computed as KL(P_teacher || Q_student) where:
    - P_teacher: Softmax over teacher-provided logits at temperature T
    - Q_student: Student's log-softmax over full vocabulary at temperature T,
                 gathered at teacher indices (normalization over full vocab)
    
    The KD term is scaled by T^2 as recommended in the literature.
    
    Args:
        student_logits: Student model logits of shape (B, T, V)
        teacher_data: List of teacher data dictionaries, one per batch element.
                     Each dict contains 'positions', 'indices', 'values' keys:
                     - positions: List of sequence positions with teacher data
                     - indices: List of teacher token indices for each position  
                     - values: List of teacher logit values for each position
        labels: Target labels of shape (B, T) with ignore_index for masked positions
        T: Temperature parameter for softmax (default: 1.0)
        ignore_index: Index value to ignore in loss computation (default: -100)
        
    Returns:
        Per-token KL divergence losses of shape [B, T]
        
    Raises:
        ValueError: If teacher_data length doesn't match batch size
    """
    device = student_logits.device
    batch_size, sequence_length, vocab_size = student_logits.shape
    if len(teacher_data) != batch_size:
        raise ValueError(
            f"teacher_data length {len(teacher_data)} must match batch size {batch_size}"
        )
    
    # Initialize per-position KL loss tensor
    kl_loss_tensor = torch.zeros(
        (batch_size, sequence_length), 
        device=device, 
        dtype=torch.float32
    )
    
    total_valid_positions = 0

    # Debug configuration
    debug_position_idx = 7365

    # Process each batch element independently
    for batch_idx in range(batch_size):
        batch_teacher_data = teacher_data[batch_idx]
        batch_label_sequence = labels[batch_idx]  # Shape: (T,)
        batch_student_logits = student_logits[batch_idx]  # Shape: (T, V)
        # Extract teacher data components
        teacher_positions = batch_teacher_data.get('positions', [])
        teacher_indices_list = batch_teacher_data.get('indices', [])
        teacher_values_list = batch_teacher_data.get('values', [])

        if (
            torch.is_tensor(teacher_positions)
            and torch.is_tensor(teacher_indices_list)
            and torch.is_tensor(teacher_values_list)
            and 'row_ptr' in batch_teacher_data
        ):
            row_ptr_tensor = batch_teacher_data['row_ptr']
            teacher_positions_tensor = teacher_positions.detach().cpu().view(-1)
            teacher_indices_tensor = teacher_indices_list.detach().cpu().view(-1)
            teacher_values_tensor = teacher_values_list.detach().cpu().view(-1)
            row_ptr_tensor = row_ptr_tensor.detach().cpu().view(-1)

            teacher_positions = teacher_positions_tensor.tolist()

            teacher_indices_list = []
            teacher_values_list = []
            for idx_pos in range(len(teacher_positions)):
                start = int(row_ptr_tensor[idx_pos].item())
                end = int(row_ptr_tensor[idx_pos + 1].item())
                teacher_indices_list.append(teacher_indices_tensor[start:end].tolist())
                teacher_values_list.append(teacher_values_tensor[start:end].tolist())

        if not teacher_positions or not teacher_indices_list or not teacher_values_list:
            continue  # Skip batch elements without teacher data
        
        batch_valid_positions = 0
        
        # Process each position with teacher data
        for teacher_position, teacher_token_indices, teacher_token_values in zip(
            teacher_positions, teacher_indices_list, teacher_values_list
        ):
            position_idx = int(
                teacher_position.item() if isinstance(teacher_position, torch.Tensor) 
                else teacher_position
            )

            if batch_label_sequence[position_idx] == ignore_index:
                continue
            
            # Convert teacher data to tensors on correct device
            teacher_indices_tensor = torch.as_tensor(
                teacher_token_indices, dtype=torch.long, device=device
            )
            teacher_values_tensor = torch.as_tensor(teacher_token_values, device=device)
            
            # Get student logits for this position
            student_logits_at_position = batch_student_logits[position_idx]  # Shape: (V,)
            
            # Compute student log-probabilities with temperature scaling
            # Normalization is over the full vocabulary
            student_log_probs_full_vocab = F.log_softmax(
                    student_logits_at_position / T, dim=-1, dtype=torch.float32
                ) # Shape: (V,)
              
            
            # Extract student log-probabilities for teacher token indices
            student_log_probs_teacher_tokens = student_log_probs_full_vocab[teacher_indices_tensor]  # Shape: (K,)
            
            # Compute teacher log-probabilities with temperature scaling
            # Normalization is over teacher indices only
            teacher_log_probs = F.log_softmax(
                teacher_values_tensor / T, dim=-1, dtype=torch.float32
            )  # Shape: (K,)

            # Mask out any non-finite teacher entries to avoid NaNs in KL
            finite_mask = torch.isfinite(teacher_log_probs)
            if not torch.all(finite_mask):
                # If all entries are non-finite, skip this position
                if not torch.any(finite_mask):
                    continue
                teacher_log_probs = teacher_log_probs[finite_mask]
                student_log_probs_teacher_tokens = student_log_probs_teacher_tokens[finite_mask]

            # Compute KL divergence: KL(P_teacher || Q_student)
            # KL = sum(p_teacher * (log p_teacher - log p_student))
            teacher_probs = teacher_log_probs.exp()
            kl_divergence_at_position = teacher_probs * (teacher_log_probs - student_log_probs_teacher_tokens)
            kl_divergence_scalar = kl_divergence_at_position.sum() * (T ** 2)  # Reduce to scalar

            # Store KL loss for this position
            kl_loss_tensor[batch_idx, position_idx] = kl_divergence_scalar

            # Debug output for specific position
            if position_idx == debug_position_idx:
                full_vocab_log_sum_exp = torch.logsumexp(
                    student_logits_at_position.float() / T, dim=-1
                )
                print(f"batch {batch_idx} traditional pos {debug_position_idx}  "
                      f"teacher_probs: {teacher_probs}")
                print(f"batch {batch_idx} traditional pos {debug_position_idx} "
                      f"student_log_probs_teacher: {student_log_probs_teacher_tokens} "
                      f"lse student: {full_vocab_log_sum_exp}")
                print(f"batch {batch_idx} traditional pos {debug_position_idx} "
                      f"kl loss {kl_divergence_scalar}")
            
            batch_valid_positions += 1
        
        print(f"batch {batch_idx} traditional valid positions: {batch_valid_positions}, "
              f"batch_kl_sum: {kl_loss_tensor[batch_idx].sum()}")
        total_valid_positions += batch_valid_positions

    print(f"KD loss total: {kl_loss_tensor.sum()}, Number of valid positions: {total_valid_positions}")
    return kl_loss_tensor
