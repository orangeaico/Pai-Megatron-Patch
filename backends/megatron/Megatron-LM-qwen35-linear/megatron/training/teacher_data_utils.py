"""Utilities for broadcasting teacher logits alongside Megatron batches.

This module provides helpers to pack sparse teacher logits for broadcast across
tensor-parallel ranks. To reduce per-iteration overhead, we add lightweight
buffer caching and pinned-memory staging so we can reuse device buffers across
steps and avoid repeated allocate→copy→free cycles.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

# -----------------------------------------------------------------------------
# Persistent buffer cache (per device, shape)
# -----------------------------------------------------------------------------

# key: (device_index, batch_size, max_positions, max_k)
_TEACHER_BUFFER_CACHE: Dict[Tuple[int, int, int, int], Dict[str, torch.Tensor]] = {}


def _normalize_device(device: Union[torch.device, int, str, None]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, int):
        return torch.device("cuda", device)
    if device is None:
        return torch.device("cuda", torch.cuda.current_device())
    # string inputs like "cuda" or "cuda:0"
    return torch.device(device)


def _device_index(device: Union[torch.device, int, str, None]) -> int:
    norm_device = _normalize_device(device)
    if norm_device.type == "cuda":
        return norm_device.index if norm_device.index is not None else torch.cuda.current_device()
    return -1


def _get_or_create_buffers(
    batch_size: int,
    max_positions: int,
    max_k: int,
    device: Union[torch.device, int, str, None],
) -> Dict[str, torch.Tensor]:
    device = _normalize_device(device)
    key = (_device_index(device), batch_size, max_positions, max_k)
    cached = _TEACHER_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached

    positions = torch.full(
        (batch_size, max_positions), _POS_PAD, dtype=torch.int64, device=device
    )
    indices = torch.full(
        (batch_size, max_positions, max_k), _IDX_PAD, dtype=torch.int64, device=device
    )
    values = torch.zeros(
        (batch_size, max_positions, max_k), dtype=torch.float32, device=device
    )
    counts = torch.zeros((batch_size, max_positions), dtype=torch.int64, device=device)

    buf = {"positions": positions, "indices": indices, "values": values, "counts": counts}
    _TEACHER_BUFFER_CACHE[key] = buf
    return buf


def _reset_buffers(buf: Dict[str, torch.Tensor]) -> None:
    # Ensure all entries are in a known state before writing new data.
    buf["positions"].fill_(_POS_PAD)
    buf["indices"].fill_(_IDX_PAD)
    buf["values"].zero_()
    buf["counts"].zero_()


def clear_teacher_buffer_cache() -> None:
    """Clear all cached teacher buffers (useful for long-running jobs)."""
    _TEACHER_BUFFER_CACHE.clear()


_POS_PAD = -1
_IDX_PAD = -1


def pack_teacher_batch(
    teacher_batch: Optional[Sequence[Optional[Dict[str, Any]]]],
    seq_length: int,
    device: Union[torch.device, int, str, None],
) -> Tuple[Optional[Dict[str, torch.Tensor]], Tuple[int, int]]:
    """Pad a batch of sparse teacher logits for tensor broadcast.

    Returns a dictionary of tensors and a tuple ``(max_positions, max_k)`` describing
    the padded dimensions. ``None`` signals that no teacher data is present.
    """
    device = _normalize_device(device)

    if not teacher_batch:
        return None, (-1, -1)

    batch_list: List[Dict[str, Any]] = []
    max_positions = 0
    max_k = 0
    for entry in teacher_batch:
        entry_dict = {
            "positions": [],
            "indices": [],
            "values": [],
        }
        if entry:
            positions_tensor = entry["positions"].detach().cpu().view(-1)
            row_ptr_tensor = entry["row_ptr"].detach().cpu().view(-1)
            indices_tensor = entry["indices"].detach().cpu().view(-1)
            values_tensor = entry["values"].detach().cpu().view(-1)

            if row_ptr_tensor.numel() != positions_tensor.numel() + 1:
                raise ValueError("teacher_data row_ptr must have len positions + 1")

            for idx_pos, pos_val in enumerate(positions_tensor.tolist()):
                pos_int = int(pos_val)
                if pos_int < 0 or pos_int >= seq_length:
                    raise ValueError(
                        f"teacher position {pos_int} out of range [0, {seq_length})"
                    )
                start = int(row_ptr_tensor[idx_pos].item())
                end = int(row_ptr_tensor[idx_pos + 1].item())
                idx_slice = indices_tensor[start:end].tolist()
                val_slice = values_tensor[start:end].tolist()
                if len(idx_slice) != len(val_slice):
                    raise ValueError(
                        "teacher_data indices/values length mismatch per position"
                    )
                entry_dict["positions"].append(pos_int)
                entry_dict["indices"].append([int(i) for i in idx_slice])
                entry_dict["values"].append([float(v) for v in val_slice])
                max_k = max(max_k, len(idx_slice))
            
        max_positions = max(max_positions, len(entry_dict["positions"]))
        batch_list.append(entry_dict)

    if max_positions == 0 or max_k == 0:
        return None, (-1, -1)

    batch_size = len(batch_list)

    # Stage into pinned host memory first to reduce small GPU copies, then copy
    # into a persistent GPU buffer in a single non-blocking transfer.
    positions_cpu = torch.full(
        (batch_size, max_positions), _POS_PAD, dtype=torch.int64, device="cpu", pin_memory=True
    )
    indices_cpu = torch.full(
        (batch_size, max_positions, max_k), _IDX_PAD, dtype=torch.int64, device="cpu", pin_memory=True
    )
    values_cpu = torch.zeros(
        (batch_size, max_positions, max_k), dtype=torch.float32, device="cpu", pin_memory=True
    )
    counts_cpu = torch.zeros((batch_size, max_positions), dtype=torch.int64, device="cpu", pin_memory=True)

    for b_idx, entry in enumerate(batch_list):
        for pos_idx, (pos, idxs, vals) in enumerate(
            zip(entry["positions"], entry["indices"], entry["values"])
        ):
            k = len(idxs)
            positions_cpu[b_idx, pos_idx] = pos
            counts_cpu[b_idx, pos_idx] = k
            if k:
                # Write into CPU staging; copy entire tensors once later.
                idx_tensor = torch.as_tensor(idxs, dtype=torch.int64)
                val_tensor = torch.as_tensor(vals, dtype=torch.float32)
                indices_cpu[b_idx, pos_idx, :k].copy_(idx_tensor)
                values_cpu[b_idx, pos_idx, :k].copy_(val_tensor)

    # Obtain (or create) persistent GPU buffers and copy staged data.
    gpu_buf = _get_or_create_buffers(batch_size, max_positions, max_k, device)
    # Overwrite fully to avoid stale leftovers when shapes are unchanged.
    gpu_buf["positions"].copy_(positions_cpu, non_blocking=True)
    gpu_buf["indices"].copy_(indices_cpu, non_blocking=True)
    gpu_buf["values"].copy_(values_cpu, non_blocking=True)
    gpu_buf["counts"].copy_(counts_cpu, non_blocking=True)

    return gpu_buf, (max_positions, max_k)


def allocate_teacher_tensors(
    batch_size: int,
    max_positions: int,
    max_k: int,
    device: Union[torch.device, int, str, None],
) -> Dict[str, torch.Tensor]:
    device = _normalize_device(device)
    buf = _get_or_create_buffers(batch_size, max_positions, max_k, device)
    # Reset before use to guarantee clean state when caller writes or receives.
    _reset_buffers(buf)
    return buf


def unpack_teacher_batch(
    packed: Optional[Dict[str, torch.Tensor]]
) -> Optional[List[Dict[str, torch.Tensor]]]:
    if packed is None:
        return None

    positions = packed["positions"]
    indices = packed["indices"]
    values = packed["values"]
    counts = packed["counts"]

    batch_size = positions.size(0)
    max_positions = positions.size(1)
    batch: List[Dict[str, torch.Tensor]] = []

    for b_idx in range(batch_size):
        counts_row = counts[b_idx]
        valid_mask = counts_row > 0
        num_valid = int(valid_mask.sum().item())

        if num_valid == 0:
            batch.append(
                {
                    "positions": torch.empty(0, dtype=torch.long, device=positions.device),
                    "row_ptr": torch.zeros(1, dtype=torch.long, device=positions.device),
                    "indices": torch.empty(0, dtype=torch.long, device=positions.device),
                    "values": torch.empty(0, dtype=torch.float32, device=positions.device),
                }
            )
            continue

        pos_indices = torch.nonzero(valid_mask, as_tuple=False).view(-1)
        positions_tensor = positions[b_idx, pos_indices].to(dtype=torch.long).contiguous()
        counts_tensor = counts_row[pos_indices].to(dtype=torch.long).contiguous()

        row_ptr = torch.zeros(num_valid + 1, dtype=torch.long, device=positions.device)
        row_ptr[1:] = torch.cumsum(counts_tensor, dim=0)
        total_tokens = int(row_ptr[-1].item())

        flat_indices = torch.empty(total_tokens, dtype=torch.long, device=positions.device)
        flat_values = torch.empty(total_tokens, dtype=torch.float32, device=positions.device)

        offset = 0
        for local_idx, pos_idx in enumerate(pos_indices.tolist()):
            count = int(counts_row[pos_idx].item())
            if count <= 0:
                continue
            slice_indices = indices[b_idx, pos_idx, :count].to(dtype=torch.long)
            slice_values = values[b_idx, pos_idx, :count].to(dtype=torch.float32)
            flat_indices[offset : offset + count] = slice_indices
            flat_values[offset : offset + count] = slice_values
            offset += count

        batch.append(
            {
                "positions": positions_tensor,
                "row_ptr": row_ptr,
                "indices": flat_indices,
                "values": flat_values,
            }
        )
    return batch


def has_teacher_data(packed: Optional[Dict[str, torch.Tensor]]) -> bool:
    if packed is None:
        return False
    counts = packed["counts"]
    return bool(counts.numel() and torch.any(counts > 0))
