# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""General utilities."""
import json
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from collections import defaultdict

import torch

from megatron.core.msc_utils import MultiStorageClientFeature, open_file

try:
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
except ImportError:
    try:
        from amp_C import multi_tensor_l2norm
        from apex.multi_tensor_apply import multi_tensor_applier
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. '
            'Falling back to local implementations of '
            'multi_tensor_applier and multi_tensor_l2norm'
        )

        from megatron.core.utils import (
            local_multi_tensor_l2_norm as multi_tensor_l2norm,
            local_multi_tensor_applier as multi_tensor_applier,
        )

from megatron.training import get_args, get_timers, get_adlr_autoresume
from megatron.core import mpu
from megatron.core.datasets.utils import get_blend_from_list
from megatron.core.tensor_parallel import param_is_not_tensor_parallel_duplicate
from megatron.core.utils import (
    get_batch_on_this_cp_rank,
    get_data_parallel_group_if_dtensor,
    to_local_if_dtensor,
    unwrap_model,
)
from megatron.legacy.model.module import param_is_not_shared
from megatron.training.teacher_data_utils import (
    allocate_teacher_tensors,
    pack_teacher_batch,
)
from megatron.core.fusions.distillation import generate_fake_teacher_data

def calc_params_l2_norm(model, force_create_fp32_copy=False):
    """Calculate l2 norm of parameters"""
    args = get_args()
    if not isinstance(model, list):
        model = [model]

    if getattr(args, 'use_megatron_fsdp', False):
        # All Megatron FSDP parameters are expected to be PyTorch DTensor.
        # params_data is a dict of device_mesh -> list of local tensors.
        params = []
        for model_chunk in model:
            model_chunk.stop_communication()
            for name, param in model_chunk.named_parameters():
                if not hasattr(param, "_local_tensor"):
                    raise RuntimeError(
                        f"Megatron FSDP requires parameters are PyTorch DTensor. "
                        f"Parameter {name} is not a DTensor."
                    )
                params.append(param)

        return calc_dtensor_params_l2_norm(params)

    # Seperate moe and dense params
    params_data = []
    moe_params_data = []
    sharded_params_data = []
    data_parallel_group = None

    for model_chunk in model:
        for param in model_chunk.parameters():
            data_parallel_group = get_data_parallel_group_if_dtensor(param, data_parallel_group)
            is_not_tp_duplicate = param_is_not_tensor_parallel_duplicate(param)
            if not is_not_tp_duplicate:
                continue
            assert is_not_tp_duplicate
            if not getattr(param, 'allreduce', True):
                assert param_is_not_shared(param)
                param = to_local_if_dtensor(param)
                if args.bf16:
                    if not force_create_fp32_copy and hasattr(param, 'main_param'):
                        if getattr(param, 'main_param_sharded', False):
                            if param.main_param is not None:
                                sharded_params_data.append(param.main_param)
                        else:
                            moe_params_data.append(param.main_param)
                    else:
                        # Fallback to original logic of making a fp32 copy of the
                        # parameter if `.main_param` attribute is not available.
                        moe_params_data.append(param.data.float())
                else:
                    moe_params_data.append(param.data)
            else:
                if param_is_not_shared(param):
                    param = to_local_if_dtensor(param)
                    if args.bf16:
                        if not force_create_fp32_copy and hasattr(param, 'main_param'):
                            if getattr(param, 'main_param_sharded', False):
                                if param.main_param is not None:
                                    sharded_params_data.append(param.main_param)
                            else:
                                params_data.append(param.main_param)
                        else:
                            # Fallback to original logic of making a fp32 copy of the
                            # parameter if `.main_param` attribute is not available.
                            params_data.append(param.data.float())
                    else:
                        params_data.append(param.data)

    # Calculate norm.
    dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
    if len(params_data) > 0:
        norm, _ = multi_tensor_applier(
            multi_tensor_l2norm, dummy_overflow_buf, [params_data], False  # no per-parameter norm.
        )
        norm_2 = norm * norm
    else:
        norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')

    if data_parallel_group is not None:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
        )

    # Add norm contribution from params with sharded main_params. These norms need to be
    # accumulated across the DP group since the main parameters are sharded because
    # of distributed optimizer.
    if len(sharded_params_data) > 0:
        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
        sharded_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [sharded_params_data],
            False,  # no per-parameter norm.
        )
        sharded_norm_2 = sharded_norm * sharded_norm
    else:
        sharded_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    # Sum over all DP groups, including CP since distributed optimizer state is
    # sharded jointly over DP+CP.
    torch.distributed.all_reduce(
        sharded_norm_2,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_data_parallel_group(with_context_parallel=True)
    )
    norm_2 += sharded_norm_2

    # Add norm contribution from expert layers in MoEs.
    if len(moe_params_data) > 0:
        moe_norm, _ = multi_tensor_applier(
            multi_tensor_l2norm,
            dummy_overflow_buf,
            [moe_params_data],
            False,  # no per-parameter norm.
        )
        moe_norm_2 = moe_norm * moe_norm

    # Account for MoE norm even if current rank doesn't have any expert params to prevent
    # hang in models with un-even numbers of MoE layers.
    # See details in https://gitlab-master.nvidia.com/ADLR/megatron-lm/-/issues/409
    else:
        moe_norm_2 = torch.zeros_like(norm_2)

    # Reduce norm across model parallel groups (dense and expert).
    # Dense params should sum across all model-parallel GPUs (tensor + pipeline).
    dense_reduce_group = mpu.get_model_parallel_group()
    ranks_in_dense_reduce_group = torch.distributed.get_process_group_ranks(dense_reduce_group)
    # Expert params should sum across all model-parallel GPUs (expert + tensor + pipeline).
    expert_reduce_group = mpu.get_expert_tensor_model_pipeline_parallel_group()
    ranks_in_expert_reduce_group = torch.distributed.get_process_group_ranks(expert_reduce_group)

    # If dense and expert reduce groups are the same, sum then reduce.
    if ranks_in_dense_reduce_group == ranks_in_expert_reduce_group:
        norm_2 += moe_norm_2
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
    # If dense and expert reduce groups are different, reduce then sum.
    else:
        torch.distributed.all_reduce(
            norm_2, op=torch.distributed.ReduceOp.SUM, group=dense_reduce_group
        )
        torch.distributed.all_reduce(
            moe_norm_2, op=torch.distributed.ReduceOp.SUM, group=expert_reduce_group
        )
        norm_2 += moe_norm_2

    return norm_2.item() ** 0.5


def calc_dtensor_params_l2_norm(params):
    """Calculate l2 norm of DTensor parameters."""
    params_data = defaultdict(list)
    for param in params:
        params_data[param._spec].append(param._local_tensor)

    total_norm_2 = torch.zeros((1,), dtype=torch.float32, device='cuda')
    dummy_overflow_buf = torch.zeros((1,), dtype=torch.int, device='cuda')
    for dtensor_spec, local_tensors in params_data.items():
        local_tensors = [t for t in local_tensors if t.numel() > 0]
        if len(local_tensors) == 0:
            norm = torch.zeros((1,), dtype=torch.float32, device='cuda')
        else:
            norm, _ = multi_tensor_applier(
                multi_tensor_l2norm, dummy_overflow_buf, [local_tensors], False  # no per-parameter norm.
            )
        norm_2 = norm * norm
        for pg, placement in zip(
            dtensor_spec.device_mesh.get_all_groups(),
            dtensor_spec.placements,
        ):
            if placement.is_shard():
                torch.distributed.all_reduce(
                    norm_2, op=torch.distributed.ReduceOp.SUM, group=pg
                )
            elif placement.is_replicate():
                # Replicated parameters are already summed across all ranks.
                pass
            else:
                raise RuntimeError(
                    f"Unsupported placement {placement} for Megatron FSDP."
                )
        total_norm_2 += norm_2

    return total_norm_2.item() ** 0.5


def average_losses_across_data_parallel_group(losses):
    """Reduce a tensor of losses across all GPUs."""
    averaged_losses = torch.cat([loss.clone().detach().view(1) for loss in losses])
    torch.distributed.all_reduce(averaged_losses, group=mpu.get_data_parallel_group())
    averaged_losses = averaged_losses / mpu.get_data_parallel_group().size()

    return averaged_losses


def reduce_max_stat_across_model_parallel_group(stat: float) -> float:
    """
    Ranks without an optimizer will have no grad_norm or num_zeros_in_grad stats.
    We need to ensure the logging and writer rank has those values.
    This function reduces a stat tensor across the model parallel group.

    We use an all_reduce max since the values have already been summed across optimizer ranks where possible
    """
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    else:
        return stat.item()


def logical_and_across_model_parallel_group(input: bool) -> bool:
    """
    This function gathers a bool value across the model parallel group
    """
    if input is True:
        input = 1
    else:
        input = 0
    input = torch.tensor([input], dtype=torch.int, device=torch.cuda.current_device())
    torch.distributed.all_reduce(
        input, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input.item())


def report_memory(name):
    """Simple GPU memory report."""
    mega_bytes = 1024.0 * 1024.0
    string = name + ' memory (MB)'
    string += ' | allocated: {}'.format(torch.cuda.memory_allocated() / mega_bytes)
    string += ' | max allocated: {}'.format(torch.cuda.max_memory_allocated() / mega_bytes)
    string += ' | reserved: {}'.format(torch.cuda.memory_reserved() / mega_bytes)
    string += ' | max reserved: {}'.format(torch.cuda.max_memory_reserved() / mega_bytes)
    if mpu.get_data_parallel_rank() == 0:
        print("[Rank {}] {}".format(torch.distributed.get_rank(), string), flush=True)
    # torch.cuda.reset_peak_memory_stats(torch.cuda.current_device())


def print_params_min_max_norm(optimizer, iteration):
    """Print min, max, and norm of all parameters."""
    index = 0
    rank = torch.distributed.get_rank()
    string = 'iteration, rank, index, tensor-model-parallel, min, max, norm\n'
    optimizer_ = optimizer.optimizer
    for param_group in optimizer_.param_groups:
        for param in param_group['params']:
            index += 1
            min_ = param.data.min()
            max_ = param.data.max()
            norm = torch.linalg.norm(param.data)
            string += '{:7d}, {:4d}, {:4d}, {:2d}, '.format(
                iteration, rank, index, int(param.tensor_model_parallel)
            )
            string += '{:.6E}, {:.6E}, {:.6E}\n'.format(min_, max_, norm)
    print(string, flush=True)


def check_adlr_autoresume_termination(iteration, model, optimizer, opt_param_scheduler):
    """Check for autoresume signal and exit if it is received."""
    from megatron.training.checkpointing import save_checkpoint

    args = get_args()
    autoresume = get_adlr_autoresume()
    # Add barrier to ensure consistnecy.
    torch.distributed.barrier()
    if autoresume.termination_requested():
        if args.save:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
        print_rank_0(">>> autoresume termination request found!")
        if torch.distributed.get_rank() == 0:
            autoresume.request_resume()
        print_rank_0(">>> training terminated. Returning")
        sys.exit(0)


def get_ltor_masks_and_position_ids(data,
                                    eod_token,
                                    pad_token,
                                    reset_position_ids,
                                    reset_attention_mask,
                                    eod_mask_loss,
                                    pad_mask_loss):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Loss mask.
    loss_mask = torch.ones(data.size(), dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0
    if pad_mask_loss:
        loss_mask[data == pad_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token] & position_ids[b, data[b] == pad_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


def print_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, print only on rank 0."""
    if rank is not None:
        if rank == 0:
            print(message, flush=True)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def warn_rank_0(message, rank=None):
    """If distributed is initialized or rank is specified, warn only on rank 0."""
    if rank is not None:
        if rank == 0:
            warnings.warn(message)
    elif torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            warnings.warn(message)
    else:
        warnings.warn(message)


def is_rank0():
    """Returns true if called in the rank0, false otherwise"""
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0


def is_last_rank():
    return torch.distributed.get_rank() == (torch.distributed.get_world_size() - 1)


def print_rank_last(message):
    """If distributed is initialized, print only on last rank."""
    if torch.distributed.is_initialized() and torch.distributed.get_backend() != 'fake':
        if is_last_rank():
            print(message, flush=True)
    else:
        print(message, flush=True)


def is_first_or_last_pipeline_stage(vp_stage):
    """Return True if on first or last pipeline stage, taking into account virtual
    pipeline parallelism."""
    ignore_virtual = True
    if vp_stage is not None:
        ignore_virtual = False
    return (
        mpu.is_pipeline_first_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
        or mpu.is_pipeline_last_stage(ignore_virtual=ignore_virtual, vp_stage=vp_stage)
    )


def get_device_arch_version():
    """Returns GPU arch version (8: Ampere, 9: Hopper, 10: Blackwell, ...)"""
    return torch.cuda.get_device_properties(torch.device("cuda:0")).major


def append_to_progress_log(string, barrier=True):
    """Append given string to progress log."""
    args = get_args()
    if args.save is None:
        return
    progress_log_filename = os.path.join(args.save, "progress.txt")
    if barrier:
        torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        with open_file(progress_log_filename, 'a') as f:
            job_id = os.getenv('SLURM_JOB_ID', '')
            num_gpus = args.world_size
            f.write(
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\tJob ID: {job_id}\t"
                f"# GPUs: {num_gpus}\t{string}\n"
            )


def get_blend_and_blend_per_split(args):
    """Get blend and blend_per_split from passed-in arguments."""
    use_data_path = args.data_path is not None or args.data_args_path is not None
    use_per_split_data_path = (
        any(
            elt is not None
            for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]
        )
        or args.per_split_data_args_path is not None
    )

    blend = None
    blend_per_split = None
    if use_data_path:
        if args.data_args_path is not None:
            assert args.data_path is None
            with open_file(args.data_args_path, 'r') as f:
                blend = get_blend_from_list(f.read().split())
        else:
            assert args.data_path is not None
            blend = get_blend_from_list(args.data_path)
    elif use_per_split_data_path:
        if args.per_split_data_args_path is not None:
            with open_file(args.per_split_data_args_path, 'r') as f:
                per_split_data_args = json.load(f)
                # Each element in blend_per_split should be a list of files (and optional
                # weights), so split string if needed.
                for split in ["train", "valid", "test"]:
                    if isinstance(per_split_data_args[split], str):
                        per_split_data_args[split] = per_split_data_args[split].split()

                blend_per_split = [
                    get_blend_from_list(per_split_data_args["train"]),
                    get_blend_from_list(per_split_data_args["valid"]),
                    get_blend_from_list(per_split_data_args["test"]),
                ]
        else:
            blend_per_split = [
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ]
    else:
        blend, blend_per_split = None, None

    return blend, blend_per_split


def get_batch_on_this_tp_rank(data_iterator, mtp_on_this_rank: bool = False):

    args = get_args()

    tp_rank = mpu.get_tensor_model_parallel_rank()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_src_rank = mpu.get_tensor_model_parallel_src_rank()

    use_variable_seq = getattr(args, 'variable_seq_lengths', False)

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                tp_src_rank,
                group=tp_group,
            )

    is_tp0 = tp_rank == 0

    def _broadcast_bool(value: bool) -> bool:
        if is_tp0:
            flag = torch.tensor(
                1 if value else 0, dtype=torch.int32, device=torch.cuda.current_device()
            )
        else:
            flag = torch.empty((), dtype=torch.int32, device=torch.cuda.current_device())
        _broadcast(flag)
        return bool(flag.item())

    teacher_header = (
        torch.empty(2, dtype=torch.int64, device=torch.cuda.current_device())
        if args.distillation_loss
        else None
    )
    teacher_tensors = None

    metadata = None

    if is_tp0:
        if args.pipeline_model_parallel_size == 1 or mtp_on_this_rank:
            mode = 0
        elif mpu.is_pipeline_first_stage():
            mode = 1
        elif mpu.is_pipeline_last_stage():
            mode = 2
        else:
            mode = 3
        mode_tensor = torch.tensor(mode, dtype=torch.int32, device=torch.cuda.current_device())
    else:
        mode_tensor = torch.empty((), dtype=torch.int32, device=torch.cuda.current_device())
    _broadcast(mode_tensor)
    mode = int(mode_tensor.item())

    dtype_encoding = {
        torch.float32: 0,
        torch.float16: 1,
        torch.bfloat16: 2,
        torch.float64: 3,
        torch.int64: 4,
        torch.int32: 5,
        torch.int16: 6,
        torch.int8: 7,
        torch.uint8: 8,
        torch.bool: 9,
    }
    dtype_decoding = {v: k for k, v in dtype_encoding.items()}

    def _encode_dtype(dtype: torch.dtype) -> int:
        if dtype not in dtype_encoding:
            raise RuntimeError(
                f"Unsupported dtype {dtype} encountered while sharing batch metadata across tensor parallel ranks."
            )
        return dtype_encoding[dtype]

    def _decode_dtype(idx: int) -> torch.dtype:
        if idx not in dtype_decoding:
            raise RuntimeError(
                f"Received unknown dtype id {idx} while reconstructing batch tensors on tensor parallel ranks."
            )
        return dtype_decoding[idx]

    def _exchange_batch_metadata(batch):
        metadata = {}
        keys = ('tokens', 'labels', 'loss_mask', 'position_ids', 'attention_mask')
        for key in keys:
            if is_tp0:
                tensor = batch[key]
                if tensor is None:
                    meta_tensor = torch.tensor([-1, -1], dtype=torch.int64, device=torch.cuda.current_device())
                else:
                    meta_tensor = torch.tensor(
                        [tensor.dim(), _encode_dtype(tensor.dtype)],
                        dtype=torch.int64,
                        device=torch.cuda.current_device(),
                    )
            else:
                meta_tensor = torch.empty(2, dtype=torch.int64, device=torch.cuda.current_device())

            torch.distributed.broadcast(meta_tensor, tp_src_rank, group=tp_group)

            ndims = int(meta_tensor[0].item())
            dtype_idx = int(meta_tensor[1].item())

            if ndims < 0:
                metadata[key] = (None, None)
                continue

            if ndims > 0:
                if is_tp0:
                    shape_tensor = torch.tensor(
                        batch[key].shape,
                        dtype=torch.int64,
                        device=torch.cuda.current_device(),
                    )
                else:
                    shape_tensor = torch.empty(ndims, dtype=torch.int64, device=torch.cuda.current_device())

                torch.distributed.broadcast(shape_tensor, tp_src_rank, group=tp_group)
                shape = tuple(int(dim) for dim in shape_tensor.tolist())
            else:
                shape = ()

            metadata[key] = (shape, _decode_dtype(dtype_idx))

        return metadata

    has_attention_mask = False
    has_max_seqlen = False
    has_local_cp_size = False
    seq_len = None

    if is_tp0:
        assert data_iterator is not None
        data = next(data_iterator)
        batch = {
            'tokens': data["tokens"].cuda(non_blocking=True),
            'labels': data["labels"].cuda(non_blocking=True),
            'loss_mask': data["loss_mask"].cuda(non_blocking=True),
            'attention_mask': (
                None
                if "attention_mask" not in data or data["attention_mask"] is None
                else data["attention_mask"].cuda(non_blocking=True)
            ),
            'position_ids': data["position_ids"].cuda(non_blocking=True),
            'cu_seqlens': (
                None
                if "cu_seqlens" not in data
                else data["cu_seqlens"].cuda(non_blocking=True)
            ),
            'cu_seqlens_padded': (
                None
                if "cu_seqlens_padded" not in data
                else data["cu_seqlens_padded"].cuda(non_blocking=True)
            ),
            'max_seqlen': (
                None
                if "max_seqlen" not in data
                else data["max_seqlen"].cuda(non_blocking=True)
            ),
            'local_cp_size': (
                None
                if "local_cp_size" not in data
                else data["local_cp_size"].cuda(non_blocking=True)
            ),
        }
        if args.distillation_loss:
            teacher_raw = data.get("teacher_data") if data is not None else None
            if teacher_raw is None and args.generate_fake_teacher_data:
                batch_size, seq_length = batch['tokens'].shape
                print_rank_0(
                    "Generating fake teacher data. This should only be used for debugging."
                )
                teacher_raw = generate_fake_teacher_data(
                    batch_size,
                    seq_length,
                    args.vocab_size,
                    10,
                    torch.cuda.current_device(),
                )
            packed, shape = pack_teacher_batch(
                teacher_raw, args.seq_length, torch.cuda.current_device()
            )
            if shape[0] < 0 or shape[1] < 0:
                teacher_header.fill_(-1)
            else:
                teacher_header[0] = shape[0]
                teacher_header[1] = shape[1]
            teacher_tensors = packed

        if use_variable_seq:
            metadata = _exchange_batch_metadata(batch)

        has_attention_mask = batch['attention_mask'] is not None
        has_max_seqlen = batch['max_seqlen'] is not None
        has_local_cp_size = batch['local_cp_size'] is not None

        if args.hybrid_context_parallel:
            seq_len = torch.tensor(
                batch['tokens'].shape[0],
                dtype=torch.int32,
                device=torch.cuda.current_device(),
            )
    else:
        if use_variable_seq:
            metadata = _exchange_batch_metadata(None)
        else:
            metadata = None

    has_attention_mask = _broadcast_bool(has_attention_mask)
    has_max_seqlen = _broadcast_bool(has_max_seqlen)
    has_local_cp_size = _broadcast_bool(has_local_cp_size)

    if args.hybrid_context_parallel:
        if not is_tp0:
            seq_len = torch.empty((), dtype=torch.int32, device=torch.cuda.current_device())
        _broadcast(seq_len)
        shape = seq_len.item()
    else:
        shape = (args.micro_batch_size, args.seq_length)

    if not is_tp0:
        if use_variable_seq:
            tokens_shape, tokens_dtype = metadata['tokens']
            tokens = (
                None
                if tokens_shape is None
                else torch.empty(
                    tokens_shape, dtype=tokens_dtype, device=torch.cuda.current_device()
                )
            )

            labels_shape, labels_dtype = metadata['labels']
            labels = (
                None
                if labels_shape is None
                else torch.empty(
                    labels_shape, dtype=labels_dtype, device=torch.cuda.current_device()
                )
            )

            loss_mask_shape, loss_mask_dtype = metadata['loss_mask']
            loss_mask = (
                None
                if loss_mask_shape is None
                else torch.empty(
                    loss_mask_shape, dtype=loss_mask_dtype, device=torch.cuda.current_device()
                )
            )

            attention_shape, attention_dtype = metadata['attention_mask']
            attention_mask = (
                None
                if attention_shape is None
                else torch.empty(
                    attention_shape, dtype=attention_dtype, device=torch.cuda.current_device()
                )
            )

            position_shape, position_dtype = metadata['position_ids']
            position_ids = (
                None
                if position_shape is None
                else torch.empty(
                    position_shape, dtype=position_dtype, device=torch.cuda.current_device()
                )
            )
        else:
            tokens = torch.empty(
                shape,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            labels = torch.empty(
                shape,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )
            loss_mask = torch.empty(
                shape,
                dtype=torch.float32,
                device=torch.cuda.current_device(),
            )
            if has_attention_mask:
                shape_attention_mask = (
                    (args.micro_batch_size, 1, args.seq_length, args.seq_length)
                    if not args.hybrid_context_parallel
                    else (1, 1, shape, shape)
                )
                attention_mask = torch.empty(
                    shape_attention_mask,
                    dtype=torch.bool,
                    device=torch.cuda.current_device(),
                )
            else:
                attention_mask = None
            position_ids = torch.empty(
                shape,
                dtype=torch.int64,
                device=torch.cuda.current_device(),
            )

        cu_seqlens = None
        cu_seqlens_padded = None
        max_seqlen = (
            torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
            if has_max_seqlen
            else None
        )
        local_cp_size = (
            torch.empty(1, dtype=torch.int32, device=torch.cuda.current_device())
            if has_local_cp_size
            else None
        )

    def _broadcast_cu_seqlens(cu_seqlens):
        dev = torch.cuda.current_device()
        n = 0 if cu_seqlens is None else int(cu_seqlens.numel())
        n_tensor = torch.tensor(n, dtype=torch.int64, device=dev)
        _broadcast(n_tensor)

        if n == 0:
            buf = torch.empty(0, dtype=torch.int32, device=dev)
        else:
            assert isinstance(cu_seqlens, torch.Tensor)
            assert cu_seqlens.dtype == torch.int32
            assert cu_seqlens.shape[0] == 1, "micro-batch-size must be 1 for packing"
            buf = cu_seqlens.to(device=dev, non_blocking=True).contiguous()
        _broadcast(buf)

    def _broadcast_cu_seqlens_recv():
        dev = torch.cuda.current_device()
        n = torch.empty((), dtype=torch.int64, device=dev)
        _broadcast(n)
        n = int(n.item())

        if n == 0:
            cu_seqlens = torch.empty(0, dtype=torch.int32, device=dev)
        else:
            cu_seqlens = torch.empty((args.micro_batch_size, n), dtype=torch.int32, device=dev)
        _broadcast(cu_seqlens)

        return cu_seqlens if n > 0 else None

    if is_tp0:
        if mode == 0:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
        elif mode == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])
        elif mode == 2:
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

        _broadcast_cu_seqlens(batch['cu_seqlens'])
        _broadcast_cu_seqlens(batch['cu_seqlens_padded'])
        _broadcast(batch['max_seqlen'])
        _broadcast(batch['local_cp_size'])

    else:
        if mode == 1:
            labels = None
            loss_mask = None
        elif mode == 2:
            tokens = None
            position_ids = None

        if mode == 0:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)
        elif mode == 1:
            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)
        elif mode == 2:
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        cu_seqlens = _broadcast_cu_seqlens_recv()
        cu_seqlens_padded = _broadcast_cu_seqlens_recv()
        _broadcast(max_seqlen)
        _broadcast(local_cp_size)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'cu_seqlens': cu_seqlens,
            'cu_seqlens_padded': cu_seqlens_padded,
            'max_seqlen': max_seqlen,
            'local_cp_size': local_cp_size,
        }


    if args.distillation_loss:

        _broadcast(teacher_header)

        max_positions = int(teacher_header[0].item())
        max_k = int(teacher_header[1].item())
        if max_positions >= 0 and max_k >= 0:
            if teacher_tensors is None:
                teacher_tensors = allocate_teacher_tensors(
                    args.micro_batch_size,
                    max_positions,
                    max_k,
                    torch.cuda.current_device(),
                )
            _broadcast(teacher_tensors['positions'])
            _broadcast(teacher_tensors['counts'])
            _broadcast(teacher_tensors['indices'])
            _broadcast(teacher_tensors['values'])
        else:
            teacher_tensors = None

        if args.pipeline_model_parallel_size != 1:
            if mpu.is_pipeline_first_stage() or not mpu.is_pipeline_last_stage():
                teacher_tensors = None

    batch['teacher_data'] = teacher_tensors

    return batch


def update_use_dist_ckpt(args):
    args.use_dist_ckpt = args.ckpt_format != "torch"


def to_empty_if_meta_device(module: torch.nn.Module, *, device: torch.device, recurse=True):
    """Move tensors to device if not meta device; otherwise materialize with empty_like().

    Officially, torch suggests to_empty() for meta device materialization. Under the hood,
    torch.empty_like() is applied to all parameters or buffers (see _apply). This may
    accidently overwrite buffers with precomputed values during construction. Given the
    goal is to only materialize those tensors on meta device, this function checks the
    device first and only move the tensor to the destination if it is not on meta device.
   
    Args:
        module: The target module to apply this transformation.
        device: The desired device of the parameters
            and buffers in this module.
        recurse: Whether parameters and buffers of submodules should
            be recursively moved to the specified device.
    """

    def _empty_like_if_meta(tensor: torch.Tensor, *, device: torch.device):
        if tensor.device == torch.device("meta"):
            return torch.empty_like(tensor, device=device)
        else:
            return tensor.to(device)

    return module._apply(
        lambda t: _empty_like_if_meta(t, device=device), recurse=recurse
    )


def get_nvtx_range():
    """Create an NVTX range context manager."""
    try:
        from torch.cuda import nvtx

        @contextmanager
        def nvtx_range(msg, time=False):
            if time:
                timers = get_timers()
                timers(msg, log_level=0).start()
            try:
                nvtx.range_push(msg)
                yield
            finally:
                nvtx.range_pop()
                if time:
                    timers(msg, log_level=0).stop()

        return nvtx_range
    except:
        @contextmanager
        def dummy_range(msg):
            yield
        return dummy_range
