# Copyright (c) 2025 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import gc
import torch
import json
import logging
import argparse
import signal
import numpy as np

from typing import *

from collections import defaultdict
from functools import partial
from torch import distributed as dist
from safetensors.torch import save_file as safe_save_file
from huggingface_hub.serialization import split_torch_state_dict_into_shards

import megatron.training.checkpointing as mg_checkpointing

from general.synchronizer import BaseSynchronizer, ParamType

class ParamMergeError(ValueError):
    ...


def _patch_checkpoint_read_metadata_for_cpu():
    """Patch Megatron checkpoint metadata all-reduce to avoid CUDA-only tensors on CPU runs."""
    if torch.cuda.is_available():
        return

    original_fn = mg_checkpointing.read_metadata
    if getattr(original_fn, "_cpu_safe_patch", False):
        return

    def _cpu_safe_read_metadata(tracker_filename):
        iteration = -1
        release = False

        with mg_checkpointing.open_file(tracker_filename, "r") as f:
            metastring = f.read().strip()
            try:
                iteration = int(metastring)
            except ValueError:
                release = metastring == "release"
                if not release:
                    mg_checkpointing.print_rank_0(
                        f"ERROR: Invalid metadata file {tracker_filename}. Exiting"
                    )
                    raise SystemExit(1)
                iteration = 0

        assert iteration > -1 or release, f"error parsing metadata file {tracker_filename}"

        if torch.distributed.is_initialized():
            # Megatron backend uses CUDA unconditionally here; use CPU tensor for CPU-only runs.
            iters = torch.tensor([iteration], dtype=torch.long, device="cpu")
            torch.distributed.all_reduce(iters, op=torch.distributed.ReduceOp.MAX)
            max_iter = int(iters[0].item())
            if iteration != max_iter:
                rank = torch.distributed.get_rank()
                print(
                    "WARNING: on rank {} found iteration {} in the metadata while max "
                    "iteration across the ranks is {}, replacing it with max iteration.".format(
                        rank, iteration, max_iter
                    ),
                    flush=True,
                )
        else:
            max_iter = iteration

        return max_iter, release

    _cpu_safe_read_metadata._cpu_safe_patch = True
    mg_checkpointing.read_metadata = _cpu_safe_read_metadata


def _patch_checkpoint_torch_load_for_cpu():
    """Patch torch.load in checkpointing path to reduce peak CPU memory usage."""
    if torch.cuda.is_available():
        return None

    # Allowlist globals required by Megatron checkpoints for weights_only=True.
    safe_globals = [
        argparse.Namespace,
        signal.Signals,
        np.dtype,
        np.ndarray,
        np.core.multiarray._reconstruct,
    ]
    try:
        from megatron.core.transformer.enums import AttnBackend
        safe_globals.append(AttnBackend)
    except Exception:
        pass
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_globals)

    original_torch_load = mg_checkpointing.torch.load

    def _cpu_friendly_torch_load(*args, **kwargs):
        kwargs.setdefault("map_location", "cpu")
        kwargs.setdefault("weights_only", True)
        kwargs.setdefault("mmap", True)
        return original_torch_load(*args, **kwargs)

    mg_checkpointing.torch.load = _cpu_friendly_torch_load
    return original_torch_load


class _LazyTensor:
    """Lazily materialized tensor view used to reduce peak CPU conversion memory."""

    def __init__(
        self,
        shape: Union[torch.Size, Tuple[int, ...], List[int]],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        builder: Callable[[], torch.Tensor],
        name: str = "",
    ):
        self.shape = torch.Size(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self._builder = builder
        self._name = name

    def materialize(self) -> torch.Tensor:
        tensor = self._builder()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Lazy tensor builder must return torch.Tensor, got {type(tensor)}")
        if tensor.dtype != self.dtype:
            tensor = tensor.to(self.dtype)
        if tensor.device != self.device:
            tensor = tensor.to(self.device)
        if tensor.shape != self.shape:
            raise ValueError(
                f"Lazy tensor shape mismatch for '{self._name}': expected {self.shape}, got {tensor.shape}"
            )
        return tensor


class MG2HFSynchronizer(BaseSynchronizer):

    def __init__(self, load_dir, model_provider_func=None, skip_hf_initialization=False):
        super().__init__(load_dir, model_provider_func, skip_hf_initialization=skip_hf_initialization)
        if not self.dryrun:
            original_torch_load = None
            if not self.args.use_gpu:
                _patch_checkpoint_read_metadata_for_cpu()
                original_torch_load = _patch_checkpoint_torch_load_for_cpu()
            try:
                mg_checkpointing.load_checkpoint(
                    [self._mgmodel], 
                    None,
                    None, 
                    checkpointing_context=None,
                    skip_load_to_model_and_opt=False
                )
            finally:
                if original_torch_load is not None:
                    mg_checkpointing.torch.load = original_torch_load

        self.num_savers = self.args.num_hf_saver
        self.max_shard_size = self.args.max_shard_size
        # NOTE: mapping unique global id (0 ~ n-1) to local data
        self._local_params = dict()
        # NOTE: mapping global id to global metadata
        self._tensor_shape = dict()
        self._tensor_dtype = dict() # sharded shape (rather than hf param shape)
        # mapping rank to (tp, pp, etp, ep, dp, edp)
        # mapping rank to (tp, pp, etp, ep, dp, edp)
        # `all_gather_into_tensor` has backend-specific shape constraints under Gloo.
        # Use `all_gather` with explicit vectors so CPU conversion is backend-agnostic.
        local_rank_mapping = torch.tensor(
            [self.tp_rank, self.pp_rank, self.etp_rank, self.ep_rank, self.dp_rank, self.edp_rank],
            dtype=torch.int,
            device=self.device,
        )
        gathered_rank_mapping = [torch.empty_like(local_rank_mapping) for _ in range(self.world_size)]
        dist.all_gather(gathered_rank_mapping, local_rank_mapping)
        self._rank_mapping = torch.stack(gathered_rank_mapping, dim=0)
        # define the merge function type for each param
        try:
            self._merge_type: torch.Tensor = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)
        except:
            self._merge_type = None
        self._has_param: torch.Tensor = None # self._has_param[param_id].nonzero() ==> ranks that have this param

    def _all_gather_into_tensor_compat(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor):
        """Backend-agnostic all_gather_into_tensor (Gloo-safe for CPU conversion)."""
        if dist.get_backend() == "gloo":
            gathered = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
            dist.all_gather(gathered, input_tensor)
            output_tensor.view(-1).copy_(torch.cat([x.reshape(-1) for x in gathered], dim=0))
            return
        dist.all_gather_into_tensor(output_tensor, input_tensor)

    def _should_register_param(self, param_type: ParamType) -> bool:
        if param_type in [ParamType.MOE_COLUMN, ParamType.MOE_ROW, ParamType.MOE_GATE_UP, ParamType.MOE_DOWN]:
            # NOTE: only register on edp_rank 0
            return self.edp_rank == 0
        if param_type == ParamType.UNIQUE:
            return self.dp_rank == 0 and self.edp_rank == 0
        return self.dp_rank == 0

    def _materialize_local_param(self, param):
        if isinstance(param, _LazyTensor):
            return param.materialize()
        if not isinstance(param, torch.Tensor):
            raise TypeError(f"Unexpected local param type: {type(param)}")
        return param

    def copy_lazy(
        self,
        dst_tensor,
        shape: Union[torch.Size, Tuple[int, ...], List[int]],
        dtype: torch.dtype,
        builder: Callable[[], torch.Tensor],
        param_type: ParamType = ParamType.UNIQUE,
        name: str = "",
    ):
        param_id = self._hf_params_to_id[dst_tensor]
        if not self._should_register_param(param_type):
            return
        self._local_params[param_id] = _LazyTensor(shape, dtype, self.device, builder, name=name)
        self._merge_type[param_id] = param_type.value

    def _copy_impl(self, src_tensor, dst_tensor, param_type: ParamType=ParamType.UNIQUE):
        param_id = self._hf_params_to_id[dst_tensor]
        if not self._should_register_param(param_type):
            return
        self._local_params[param_id] = src_tensor
        self._merge_type[param_id] = param_type.value

    def set_preprocess_state(self, mg_model, hf_model):
        '''Set embedding params.'''
        self.copy(
            mg_model.embedding.word_embeddings.weight, 
            hf_model.embed_tokens.weight, 
            param_type=ParamType.COLUMN
        )

    def set_postprocess_state(self, mg_model, hf_model, is_mamba: bool=False):
        '''Set output layer & norm params.'''
        if is_mamba:
            self.copy(
                mg_model.decoder.final_norm.weight, 
                hf_model.model.norm.weight, 
            )
        else:
            self.copy(
                mg_model.decoder.final_layernorm.weight, 
                hf_model.norm.weight, 
            )
        if mg_model.share_embeddings_and_output_weights:
            output_layer_weight = mg_model.shared_embedding_or_output_weight() 
        else:
            output_layer_weight = mg_model.output_layer.weight
        # NOTE: hf_model refers to TextModel of VLM or Model of LLM and does not
        # contain lm_head, visit it by directly calling self._hfmodel
        self.copy(output_layer_weight, self._hfmodel.lm_head.weight, param_type=ParamType.COLUMN)

    def set_mla_selfattn_state(self, attn, hf_attn):
        # NOTE: MLA qkv_bias always False
        if self.args.q_lora_rank is None:
            self.copy(attn.linear_q_proj.weight, hf_attn.q_proj.weight, param_type=ParamType.COLUMN)
        else:
            self.copy(attn.linear_q_down_proj.weight, hf_attn.q_a_proj.weight, param_type=ParamType.COLUMN)
            self.copy(attn.linear_q_up_proj.weight, hf_attn.q_b_proj.weight, param_type=ParamType.COLUMN)
            if self.args.qk_layernorm:
                self.copy(
                    attn.linear_q_up_proj.layer_norm_weight,
                    hf_attn.q_a_layernorm.weight
                )

        self.copy(attn.linear_kv_down_proj.weight, hf_attn.kv_a_proj_with_mqa.weight, param_type=ParamType.COLUMN)
        self.copy(attn.linear_kv_up_proj.weight, hf_attn.kv_b_proj.weight, param_type=ParamType.COLUMN)
        if self.args.qk_layernorm:
            self.copy(
                attn.linear_kv_up_proj.layer_norm_weight,
                hf_attn.kv_a_layernorm.weight
            )

        self.copy(
            attn.linear_proj.weight,
            hf_attn.o_proj.weight,
            param_type=ParamType.ROW
        )

    def set_selfattn_state(self, attn, hf_attn):
        '''Set self-attention params.'''
        # Reshape loaded weights.
        tp = self.tp_size
        num_heads = self.args.num_attention_heads
        num_query_groups = (self.args.num_query_groups if self.args.group_query_attention else self.args.num_attention_heads)
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0
        # copy qk norm if indeed.
        if self.args.qk_layernorm:
            self.copy(attn.q_layernorm.weight, hf_attn.q_norm.weight)
            self.copy(attn.k_layernorm.weight, hf_attn.k_norm.weight)

        # Copy weights (re-order dimensions for Megatron).
        attn_proj_weight = attn.linear_qkv.weight.reshape(
            (num_query_groups // tp, (2 + num_querys_per_group)*dim, -1)
        )
        (
            q_proj_weight, 
            k_proj_weight, 
            v_proj_weight
        ) = torch.split(attn_proj_weight, [num_querys_per_group*dim, dim, dim], dim=1)

        self.copy(q_proj_weight, hf_attn.q_proj.weight, param_type=ParamType.QKV_W)
        self.copy(k_proj_weight, hf_attn.k_proj.weight, param_type=ParamType.QKV_W)
        self.copy(v_proj_weight, hf_attn.v_proj.weight, param_type=ParamType.QKV_W)

        self.copy(
            attn.linear_proj.weight,
            hf_attn.o_proj.weight,
            param_type=ParamType.ROW
        )

        # Copy bias
        if self.args.add_qkv_bias:
            attn_proj_bias = attn.linear_qkv.bias.reshape(
                (num_query_groups // tp, (2 + num_querys_per_group)*dim, -1)
            )
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                attn_proj_bias, 
                [num_querys_per_group*dim, dim, dim], 
                dim=1
            )
            self.copy(q_proj_bias, hf_attn.q_proj.bias, param_type=ParamType.QKV_B)
            self.copy(k_proj_bias, hf_attn.k_proj.bias, param_type=ParamType.QKV_B)
            self.copy(v_proj_bias, hf_attn.v_proj.bias, param_type=ParamType.QKV_B)

    def set_mlp_state(self, mlp, hf_mlp):
        '''Set MLP params.'''
        hidden_size = mlp.linear_fc1.weight.shape[-1]
        if not mlp.config.gated_linear_unit:
            self.copy(
                mlp.linear_fc1.weight, 
                hf_mlp.linear_fc1.weight,
                param_type=ParamType.COLUMN
            )
            if mlp.config.add_bias_linear:
                self.copy(
                    mlp.linear_fc1.bias, 
                    hf_mlp.linear_fc1.bias,
                    param_type=ParamType.COLUMN
                )

            self.copy(
                mlp.linear_fc2.weight,
                hf_mlp.linear_fc2.weight,
                param_type=ParamType.ROW
            )

            if mlp.config.add_bias_linear:
                self.copy(
                    mlp.linear_fc2.bias,
                    hf_mlp.linear_fc2.bias
                )
            return

        gate_proj_weight, up_proj_weight = mlp.linear_fc1.weight.reshape(2, -1, hidden_size)
        self.copy(
            gate_proj_weight, 
            hf_mlp.gate_proj.weight,
            param_type=ParamType.COLUMN
        )

        self.copy(
            up_proj_weight, 
            hf_mlp.up_proj.weight,
            param_type=ParamType.COLUMN
        )

        if mlp.config.add_bias_linear:
            gate_proj_bias, up_proj_bias = mlp.linear_fc1.bias.reshape(2, -1)
            self.copy(
                gate_proj_bias, 
                hf_mlp.gate_proj.bias,
                param_type=ParamType.COLUMN
            )

            self.copy(
                up_proj_bias, 
                hf_mlp.up_proj.bias,
                param_type=ParamType.COLUMN
            )

        self.copy(
            mlp.linear_fc2.weight,
            hf_mlp.down_proj.weight,
            param_type=ParamType.ROW
        )

        if mlp.config.add_bias_linear:
            self.copy(
                mlp.linear_fc2.bias,
                hf_mlp.down_proj.bias
            )

    def set_sequential_mlp_state(self, experts, hf_experts):
        '''Set MOE MLP params.'''
        experts = experts.local_experts
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            hidden_size = experts[mg_expert_id].linear_fc1.weight.shape[-1]
            (
                gate_proj_weight, 
                up_proj_weight
            ) = experts[mg_expert_id].linear_fc1.weight.reshape(2, -1, hidden_size)

            self.copy(
                gate_proj_weight, 
                hf_experts[hf_expert_id].gate_proj.weight,
                param_type=ParamType.MOE_COLUMN
            )

            self.copy(
                up_proj_weight, 
                hf_experts[hf_expert_id].up_proj.weight,
                param_type=ParamType.MOE_COLUMN
            )

            self.copy(
                experts[mg_expert_id].linear_fc2.weight,
                hf_experts[hf_expert_id].down_proj.weight,
                param_type=ParamType.MOE_ROW
            )

    def set_group_mlp_state(self, experts, hf_experts):
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            hidden_size = getattr(experts.linear_fc1, f'weight{mg_expert_id}').shape[-1]
            (
                gate_proj_weight, 
                up_proj_weight
            ) = getattr(experts.linear_fc1, f'weight{mg_expert_id}').reshape(2, -1, hidden_size)

            self.copy(
                gate_proj_weight, 
                hf_experts[hf_expert_id].gate_proj.weight,
                param_type=ParamType.MOE_COLUMN
            )

            self.copy(
                up_proj_weight, 
                hf_experts[hf_expert_id].up_proj.weight,
                param_type=ParamType.MOE_COLUMN
            )

            self.copy(
                getattr(experts.linear_fc2, f'weight{mg_expert_id}'),
                hf_experts[hf_expert_id].down_proj.weight,
                param_type=ParamType.MOE_ROW
            )

    def set_moe_layer_state(self, moe, hf_moe):
        # router
        self.copy(moe.router.weight, hf_moe.gate.weight)
        if moe.router.enable_expert_bias:
            self.copy(moe.router.expert_bias, hf_moe.gate.e_score_correction_bias)
        # experts
        if self.args.moe_grouped_gemm:
            # group gemm
            if self.args.moe_use_legacy_grouped_gemm:
                # weight1 and weight2, not impl
                raise NotImplementedError("Currently only TE GroupGEMM is implemented.")
            self.set_group_mlp_state(moe.experts, hf_moe.experts)
        else:
            # sequential
            self.set_sequential_mlp_state(moe.experts, hf_moe.experts)

        # shared experts
        if moe.shared_experts is not None:
            if moe.shared_experts.use_shared_expert_gate:
                self.copy(moe.shared_experts.gate_weight, hf_moe.shared_expert_gate.weight)

            hidden_size = moe.shared_experts.linear_fc1.weight.shape[-1]
            gate_proj_weight, up_proj_weight = moe.shared_experts.linear_fc1.weight.reshape(2, -1, hidden_size)

            try:
                hf_shared_expert = hf_moe.shared_experts
            except AttributeError:
                hf_shared_expert = hf_moe.shared_expert
            self.copy(
                gate_proj_weight, 
                hf_shared_expert.gate_proj.weight,
                param_type=ParamType.COLUMN
            )
            self.copy(
                up_proj_weight, 
                hf_shared_expert.up_proj.weight,
                param_type=ParamType.COLUMN
            )
            self.copy(
                moe.shared_experts.linear_fc2.weight,
                hf_shared_expert.down_proj.weight,
                param_type=ParamType.ROW
            )

    def set_layer_state(self, layer, hf_layer):
        '''Set transformer layer params.'''
        if self.args.multi_latent_attention:
            self.set_mla_selfattn_state(layer.self_attention, hf_layer.self_attn)
            self.copy(layer.input_layernorm.weight, hf_layer.input_layernorm.weight)
        else:
            self.set_selfattn_state(layer.self_attention, hf_layer.self_attn)
            self.copy(layer.self_attention.linear_qkv.layer_norm_weight, hf_layer.input_layernorm.weight)

        if hasattr(layer.mlp, 'router'):
            self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
            self.copy(layer.pre_mlp_layernorm.weight, hf_layer.post_attention_layernorm.weight)
        else:
            self.set_mlp_state(layer.mlp, hf_layer.mlp)
            self.copy(layer.mlp.linear_fc1.layer_norm_weight, hf_layer.post_attention_layernorm.weight)

    def _postprocess_output_tensor_before_save(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        return tensor

    def check_and_save(self, output_dir):
        export_state_dict = self._get_export_state_dict()
        sharded_info = split_torch_state_dict_into_shards(
            export_state_dict,
            max_shard_size=self.max_shard_size
        )

        global_shape = {self._hf_params_key_to_id[k]: v.shape for k, v in export_state_dict.items()}

        # select local bucket(s) for each rank
        n_savers = self.num_savers
        local_buckets = []
        max_n_local_buckets = 1
        if not sharded_info.is_sharded and self.rank == 0:
            local_buckets = list(sharded_info.filename_to_tensors.keys())
        else:
            n_buckets = len(sharded_info.filename_to_tensors)
            rank = self.rank
            n_local_buckets = n_buckets // n_savers
            remainder = n_buckets % n_savers
            max_n_local_buckets = n_local_buckets + 1
            if remainder == 0:
                start = rank * n_local_buckets
                max_n_local_buckets -= 1
            elif rank < remainder:
                n_local_buckets += 1
                start = rank * n_local_buckets
            else:
                start = rank * n_local_buckets + remainder

            local_buckets = list(sharded_info.filename_to_tensors.keys())[start:start + n_local_buckets]
            if rank == 0:
                index = {
                    "metadata": sharded_info.metadata,
                    "weight_map": sharded_info.tensor_to_filename,
                }
                os.makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
                    f.write(json.dumps(index, indent=2)) 
        
        self._collect_dist_info()
        stream_max_bytes = self._get_streaming_max_bytes()
        # In each iteration, all ranks save at most one local bucket
        for bucket_idx in range(max_n_local_buckets):
            required_keys = []
            if bucket_idx < len(local_buckets):
                bucket_name = local_buckets[bucket_idx]
                required_keys: List[str] = sharded_info.filename_to_tensors[bucket_name]

            output_data = {}
            key_chunks = self._split_required_keys_by_transfer_budget(required_keys, stream_max_bytes)
            local_n_chunks = torch.tensor([len(key_chunks)], dtype=torch.long, device=self.device)
            global_n_chunks = local_n_chunks.clone()
            dist.all_reduce(global_n_chunks, op=dist.ReduceOp.MAX)
            for chunk_idx in range(int(global_n_chunks.item())):
                required_keys_chunk = key_chunks[chunk_idx] if chunk_idx < len(key_chunks) else []
                # build send/recv op across all ranks
                data, buffers, send_param_ids, recv_param_ids, ops = self._build_p2p_ops(required_keys_chunk)
                # run data sync
                if self.debug:
                    logging.info(
                        f"[Iters {bucket_idx}.{chunk_idx} RANK {self.rank}] starts synchronizing "
                        "parameters with other ranks..."
                    )
                if len(ops) > 0:
                    reqs = dist.batch_isend_irecv(ops)
                    if self.debug:
                        for op in ops:
                            size_mib = op.tensor.numel() * op.tensor.dtype.itemsize / 2 ** 20
                            if op.op == dist.isend:
                                logging.info(
                                    f"[Iters {bucket_idx}.{chunk_idx} RANK {self.rank}] "
                                    f"({self.rank} -> {op.peer}) with {size_mib} MiB."
                                )
                            else:
                                logging.info(
                                    f"[Iters {bucket_idx}.{chunk_idx} RANK {self.rank}] "
                                    f"({op.peer} -> {self.rank}) with {size_mib} MiB."
                                )
                    for req in reqs:
                        req.wait()
                if self.debug:
                    logging.info(f"[Iters {bucket_idx}.{chunk_idx} RANK {self.rank}] finishes synchronizing")

                for remote_rank, param_ids in recv_param_ids.items():
                    unpacked = self._unpack_from_buffer(buffers[remote_rank], param_ids)
                    for param_id, tensor in zip(param_ids, unpacked):
                        data[param_id][remote_rank] = tensor

                # apply merge function on the results
                # data: Dict[param_id, Dict[rank_id, tensor]]
                for param_id, data_dict in data.items():
                    param_type = ParamType(int(self._merge_type[param_id]))
                    key = self._id_to_hf_params_key[param_id] # for debugging
                    if param_type == ParamType.NULL:
                        raise ValueError(f"ParamType.NULL found on {key}.")
                    try:
                        data[param_id] = self._merge_data(param_type, data_dict)
                    except ParamMergeError as e:
                        raise ValueError(f"Merge Error on key {key}: {e}")
                    if data[param_id].shape != global_shape[param_id]:
                        raise ValueError(
                            f"Unexpected shape on {key}. Expected: {global_shape[param_id]}, "
                            f"but {data[param_id].shape}"
                        )

                for key in required_keys_chunk:
                    param_id = self._hf_params_key_to_id[key]
                    if param_id not in data:
                        raise ValueError(
                            f"Missing merged tensor for key `{key}` (param_id={param_id}). "
                            "Likely missing sync mapping or missing MCore parameter registration."
                        )
                    output_data[key] = self._postprocess_output_tensor_before_save(
                        key, data[param_id]
                    )

                # aggressively release intermediate tensors between chunks.
                del data, buffers, send_param_ids, recv_param_ids, ops
                if not self.args.use_gpu:
                    gc.collect()

            # save safetensor files
            if bucket_idx < len(local_buckets):
                if not self.dryrun:
                    for key in required_keys:
                        # NOTE: we save extra copy for each tied parameter.
                        if self._id_to_hf_params_key[self._hf_params_key_to_id[key]] != key:
                            output_data[key] = output_data[key].clone()
                    safe_save_file(
                        output_data,
                        os.path.join(output_dir, local_buckets[bucket_idx]),
                        metadata={"format": "pt"},
                    )
                print(f"[Iters {bucket_idx} RANK {self.rank}] {local_buckets[bucket_idx]} is saved.")

            if self.debug:
                logging.debug(f"[Iters {bucket_idx} RANK {self.rank}] joined")
                dist.barrier()

    def _get_export_state_dict(self):
        return self._hfmodel.state_dict()

    def _get_streaming_max_bytes(self):
        # 0/negative disables streaming chunking and preserves previous behavior.
        max_mb = float(os.environ.get("M2H_STREAM_MAX_MB", "512"))
        if max_mb <= 0:
            return None
        return int(max_mb * (1024 ** 2))

    def _split_required_keys_by_transfer_budget(
        self,
        required_keys: List[str],
        max_bytes: Optional[int],
    ) -> List[List[str]]:
        if len(required_keys) == 0:
            return [[]]
        if max_bytes is None:
            return [required_keys]

        keys_by_param_id = defaultdict(list)
        ordered_param_ids = []
        seen = set()
        for key in required_keys:
            param_id = self._hf_params_key_to_id[key]
            keys_by_param_id[param_id].append(key)
            if param_id not in seen:
                seen.add(param_id)
                ordered_param_ids.append(param_id)

        chunks_param_ids = []
        current_chunk = []
        current_bytes = 0
        for param_id in ordered_param_ids:
            if param_id in self._tensor_shape and param_id in self._tensor_dtype:
                shape = self._tensor_shape[param_id]
                dtype = self._tensor_dtype[param_id]
                param_bytes = int(shape.numel() * dtype.itemsize)
            else:
                # Unknown size: isolate this parameter so it does not blow up a chunk.
                param_bytes = max_bytes

            if current_chunk and current_bytes + param_bytes > max_bytes:
                chunks_param_ids.append(current_chunk)
                current_chunk = []
                current_bytes = 0

            current_chunk.append(param_id)
            current_bytes += param_bytes

        if current_chunk:
            chunks_param_ids.append(current_chunk)

        chunks = []
        for chunk_param_ids in chunks_param_ids:
            chunk_keys = []
            for param_id in chunk_param_ids:
                chunk_keys.extend(keys_by_param_id[param_id])
            chunks.append(chunk_keys)
        return chunks

    def _collect_dist_info(self):
        # Collect following metadatas:
        # param_id --> source_rank
        self._has_param = torch.zeros(
            [self.world_size, len(self._hf_params_key_to_id)], dtype=torch.bool, device=self.device
        )
        for param_id in self._local_params.keys():
            self._has_param[self.rank][param_id] = True
        self._all_gather_into_tensor_compat(self._has_param, self._has_param[self.rank])
        self._has_param = self._has_param.T        
        # param_id --> tensor_shape  Dict[int, Tuple[int, ...]]
        # param_id --> tensor_dtype  Dict[int, dtype]
        for param_id, param in self._local_params.items():
            if isinstance(param, _LazyTensor):
                self._tensor_shape[param_id] = param.shape
                self._tensor_dtype[param_id] = param.dtype
            else:
                if param is None:
                    raise ValueError(
                        f"Local parameter `{self._id_to_hf_params_key.get(param_id, param_id)}` "
                        "was registered as None. Mapping should skip unset tensors."
                    )
                self._tensor_shape[param_id] = param.shape
                self._tensor_dtype[param_id] = param.dtype

        # collect across ranks
        tensor_shapes = [None] * self.world_size
        tensor_dtypes = [None] * self.world_size
        dist.all_gather_object(tensor_shapes, self._tensor_shape)
        dist.all_gather_object(tensor_dtypes, self._tensor_dtype)
        # NOTE: merge them together
        for rank, remote_shape in enumerate(tensor_shapes):
            if rank == self.rank:
                continue
            for remote_key, shape in remote_shape.items():
                if remote_key not in self._tensor_shape:
                    self._tensor_shape[remote_key] = shape
                elif shape != self._tensor_shape[remote_key]:
                    raise ValueError(
                        f"Find mismatched shape on local rank {self.rank} and remote rank {rank}, local shape: {self._tensor_shape[remote_key]}; remote shape: {shape}"
                    )

        for rank, remote_dtype in enumerate(tensor_dtypes):
            if rank == self.rank:
                continue
            for remote_key, dtype in remote_dtype.items():
                if remote_key not in self._tensor_dtype:
                    self._tensor_dtype[remote_key] = dtype
                elif dtype != self._tensor_dtype[remote_key]:
                    raise ValueError(
                        f"Find mismatched dtype on local rank {self.rank} and remote rank {rank}, local shape: {self._tensor_dtype[remote_key]}; remote shape: {dtype}"
                    )
        
        # merge_type
        global_merge_type = torch.zeros([self.world_size, self.hf_size], dtype=self._merge_type.dtype, device=self.device)
        self._all_gather_into_tensor_compat(global_merge_type, self._merge_type)
        for remote_rank_id, remote_merge_type in enumerate(global_merge_type):
            if self.debug:
                and_mask = torch.logical_and(remote_merge_type > 0, self._merge_type > 0)
                if (self._merge_type[and_mask] != remote_merge_type[and_mask]).any():
                    param_id = -1
                    for param_id in range(self.hf_size):
                        if (
                            self._merge_type[param_id] > 0 and 
                            remote_merge_type[param_id] > 0 and 
                            self._merge_type[param_id] != remote_merge_type[param_id]
                        ):
                            break
                    key = self._id_to_hf_params_key[param_id]
                    raise ValueError(f"Find mismatched merge_type between local rank {self.rank} and remote rank {remote_rank_id} on key {key}")
            self._merge_type[remote_merge_type > 0] = remote_merge_type[remote_merge_type > 0]

    def _unpack_from_buffer(self, buffer: torch.Tensor, param_ids: List[int]) -> List[torch.Tensor]:
        start = 0
        datas = []
        for param_id in param_ids:
            shape = self._tensor_shape[param_id]
            dtype = self._tensor_dtype[param_id]

            offset = shape.numel() * dtype.itemsize
            datas.append(buffer[start:start + offset].view(dtype).view(shape).clone())
            start += offset
        
        if start != buffer.numel():
            raise ValueError(f"Expect {start} bytes from remote, but got {buffer.numel()} bytes!")
        return datas

    def _pack_into_byte_buffer(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([t.flatten().view(torch.uint8) for t in tensors])

    def _build_p2p_ops(self, required_keys: List[str]):
        required_ids = torch.zeros(
            [self.world_size, self.hf_size], dtype=torch.bool, device=self.device
        )
        for k in required_keys:
            required_ids[self.rank][self._hf_params_key_to_id[k]] = True
        self._all_gather_into_tensor_compat(required_ids, required_ids[self.rank])

        send_ops = []
        if self.debug:
            # (param_id, src_rank, dst_rank)
            send_recv_pattern = torch.zeros([self.hf_size, self.world_size, self.world_size], dtype=torch.int, device=self.device)
        send_param_ids = defaultdict(list)
        datas = defaultdict(list)
        # NOTE: for rank i to rank j, send params by ascending order
        # NOTE: to avoid hangs observed in multi-nodes scenarios, we merge tensors sent to same remote rank into a large uint8 tensor
        for param_id, has_data in enumerate(self._has_param[:, self.rank]):
            if not has_data:
                continue
            # group by remote_id
            for remote_rank, should_send in enumerate(required_ids[:, param_id]):
                if not should_send or remote_rank == self.rank:
                    continue
                # NOTE: for each receiver, send param in ascending order by id
                data = self._materialize_local_param(self._local_params[param_id])
                if data.device != self.device:
                    logging.warning(f"Find unexpected device {data.device} on key {self._id_to_hf_params_key[param_id]}, moving to {self.device}")
                    data = data.to(self.device)
                if data.dtype != self._tensor_dtype[param_id]:
                    raise ValueError(f"Get mismatched data type on key {self._id_to_hf_params_key[param_id]}")
                datas[remote_rank].append(data)
                send_param_ids[remote_rank].append(param_id)
                if self.debug:
                    send_recv_pattern[param_id, self.rank, remote_rank] += 1

        for remote_rank, raw_data in datas.items():
            if len(raw_data) > 0:
                send_ops.append(dist.P2POp(
                    dist.isend,
                    self._pack_into_byte_buffer(raw_data), 
                    peer=remote_rank,
                    tag=self.rank * self.world_size + remote_rank # (sender_rank, receiver_rank) ignored in NCCL
                ))

        recv_ops = []
        collected_data = defaultdict(dict)
        buffer_size = [0] * self.world_size
        recv_param_ids = defaultdict(list)
        # NOTE: for rank i to rank j, recv params by ascending order
        for param_id, is_required in enumerate(required_ids[self.rank]):
            if not is_required:
                continue
            for remote_rank, has_data in enumerate(self._has_param[param_id]):
                if not has_data:
                    continue
                if remote_rank == self.rank:
                    collected_data[param_id][remote_rank] = self._materialize_local_param(self._local_params[param_id])
                else:
                    recv_param_ids[remote_rank].append(param_id)
                    shape = self._tensor_shape[param_id]
                    dtype = self._tensor_dtype[param_id]
                    buffer_size[remote_rank] += shape.numel() * dtype.itemsize
                    if self.debug:
                        send_recv_pattern[param_id, remote_rank, self.rank] -= 1
        
        buffers = [None] * self.world_size
        for remote_rank, rank_size in enumerate(buffer_size):
            if rank_size == 0:
                continue
            buffers[remote_rank] = torch.empty(rank_size, dtype=torch.uint8, device=self.device)
            recv_ops.append(dist.P2POp(
                dist.irecv,
                buffers[remote_rank], 
                peer=remote_rank,
                tag=remote_rank * self.world_size + self.rank # (sender_rank, receiver_rank) ignored in NCCL
            ))

        if self.debug:
            dist.all_reduce(send_recv_pattern)
            if send_recv_pattern.sum() != 0:
                for param_id, pattern_per_param in enumerate(send_recv_pattern):
                    if pattern_per_param.sum() != 0:
                        raise ValueError(f"Mismatched send/recv ops detected on key {self._id_to_hf_params_key[param_id]}: {pattern_per_param}.")
                raise ValueError("Mismatched send/recv ops detected.")
            logging.debug(f"[RANK {self.rank}] {len(send_ops)} send op & {len(recv_ops)} recv op.")

        return collected_data, buffers, send_param_ids, recv_param_ids, (send_ops + recv_ops)

    def _merge_data(self, merge_type: ParamType, tensor_dict) -> torch.Tensor:
        """Merge ShardedTensor collected across the group on this rank.
        If DP or EDP is larger than 1, tensor_dict could contains multiple data
        copies and an data deduplication isrequired.

        Args:
            merge_type (ParamType): Themerge policy be applied on the 
            ShardedTensor 
            tensor_dict (Dict[int, torch.Tensor]): The collected dict of 
            ShardedTensors, mapping global_rank to tensor data.

        Returns:
            torch.Tensor: The merged tensor
        """

        def deduplicate_and_sort(tensor_dict, rank_group: int):
            tensors = dict()
            for global_rank, data in tensor_dict.items():
                rank = self._rank_mapping[global_rank][rank_group]
                tensors[int(rank)] = data
            return [item[1] for item in sorted(tensors.items())]
        
        # tensor_dict: Dict[remote_rank, torch.Tensor]
        def merge_along_axis(axis, tensor_dict, is_expert: bool = False):
            global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device) # (N, )
            ranks = self._rank_mapping.index_select(0, global_ranks) # (N, 6)
            # NOTE: for most cases, data will be from the same pp_rank, filter data from smallest pp_rank to fix tie-embedding
            tensor_dict = {k: v for k, v in tensor_dict.items() if self._rank_mapping[k, 1] == ranks[:, 1].min()}
            if self.debug:
                if is_expert and ranks[:, 5].any():
                    raise ParamMergeError("Unexpected expert parameter data from non-zero expert dp rank")
                if not is_expert and ranks[:, 4].any():
                    raise ParamMergeError("Unexpected parameter data from non-zero dp rank")
                if is_expert and ranks[:, 3].max() != ranks[:, 3].min():
                    raise ParamMergeError("Unexpected expert parameter data from multiple ep ranks")

            if is_expert:
                tensors = deduplicate_and_sort(tensor_dict, 2)
            else:
                tensors = deduplicate_and_sort(tensor_dict, 0)
            return torch.cat(tensors, dim=axis)
        
        def no_merge_func(tensor_dict):
            return list(tensor_dict.values())[0]
        
        def merge_qkv(is_bias, tensor_dict):
            res = merge_along_axis(0, tensor_dict)
            if is_bias:
                assert res.shape[-1] == 1
                return res.flatten()
            return res.flatten(0, 1)

        def merge_mamba_conv1d(tensor_dict):
            global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device)
            ranks = self._rank_mapping.index_select(0, global_ranks)  # (N, 6)
            # For tied tensors, keep one PP stage consistently (same as merge_along_axis()).
            tensor_dict = {
                k: v
                for k, v in tensor_dict.items()
                if self._rank_mapping[k, 1] == ranks[:, 1].min()
            }

            if self.debug and ranks[:, 4].any():
                raise ParamMergeError("Unexpected parameter data from non-zero dp rank")

            q_local = self.args.mamba_num_groups * self.args.mamba_state_dim // self.tp_size
            k_local = q_local
            v_local = self.args.mamba_num_heads * self.args.mamba_head_dim // self.tp_size
            expected_local = q_local + k_local + v_local

            q_tensors, k_tensors, v_tensors = [], [], []
            for tensor in deduplicate_and_sort(tensor_dict, 0):
                if tensor.shape[0] != expected_local:
                    raise ParamMergeError(
                        f"Unexpected local Mamba conv1d split size: "
                        f"expected dim0={expected_local}, got {tensor.shape[0]}"
                    )
                q, k, v = torch.split(tensor, [q_local, k_local, v_local], dim=0)
                q_tensors.append(q)
                k_tensors.append(k)
                v_tensors.append(v)

            return torch.cat(
                [
                    torch.cat(q_tensors, dim=0),
                    torch.cat(k_tensors, dim=0),
                    torch.cat(v_tensors, dim=0),
                ],
                dim=0,
            )

        def merge_linear_conv1d(tensor_dict):
            global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device)
            ranks = self._rank_mapping.index_select(0, global_ranks)  # (N, 6)
            # For tied tensors, keep one PP stage consistently (same as merge_along_axis()).
            tensor_dict = {
                k: v
                for k, v in tensor_dict.items()
                if self._rank_mapping[k, 1] == ranks[:, 1].min()
            }

            if self.debug and ranks[:, 4].any():
                raise ParamMergeError("Unexpected parameter data from non-zero dp rank")

            q_local = self.args.linear_num_key_heads * self.args.linear_key_head_dim // self.tp_size
            k_local = q_local
            v_local = (
                self.args.linear_num_value_heads * self.args.linear_value_head_dim // self.tp_size
            )
            expected_local = q_local + k_local + v_local

            q_tensors, k_tensors, v_tensors = [], [], []
            for tensor in deduplicate_and_sort(tensor_dict, 0):
                if tensor.shape[0] != expected_local:
                    raise ParamMergeError(
                        f"Unexpected local linear conv1d split size: "
                        f"expected dim0={expected_local}, got {tensor.shape[0]}"
                    )
                q, k, v = torch.split(tensor, [q_local, k_local, v_local], dim=0)
                q_tensors.append(q)
                k_tensors.append(k)
                v_tensors.append(v)

            return torch.cat(
                [
                    torch.cat(q_tensors, dim=0),
                    torch.cat(k_tensors, dim=0),
                    torch.cat(v_tensors, dim=0),
                ],
                dim=0,
            )

        def merge_qgkv(is_bias, tensor_dict):
            res = merge_along_axis(0, tensor_dict)
            if is_bias:
                assert res.shape[-1] == 1
                return res.transpose(0, 1).flatten()
            return res.transpose(0, 1).flatten(0, 2)

        def merge_moe_gate_up_tensor(tensor_dict):
            global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device) # (N, )
            ranks = self._rank_mapping.index_select(0, global_ranks) # (N, 6)
            etp_tensors = {}
            for etp_rank in range(ranks[:, 2].max().item() + 1):
                sub_tensor_dict = {}
                for global_rank, cor_ranks in zip(global_ranks, ranks):
                    if cor_ranks[2] != etp_rank: # same ETP rank
                        continue
                    sub_tensor_dict[global_rank.item()] = tensor_dict[global_rank.item()]
                etp_tensors[etp_rank] = torch.cat(deduplicate_and_sort(sub_tensor_dict, 3), dim=0)
            res = torch.cat([item[1] for item in sorted(etp_tensors.items())], dim=2)
            if res.dim() == 3:
                # Packed expert gate_up tensors are already [num_experts, 2*ffn, hidden].
                return res.contiguous()
            if res.dim() == 4:
                return res.flatten(1, 2).permute(0, 2, 1).contiguous()
            raise ParamMergeError(
                f"Unexpected tensor rank for MOE_GATE_UP merge: {res.dim()}."
            )

        def merge_moe_down_tensor(tensor_dict):
            global_ranks = torch.tensor(list(tensor_dict.keys()), dtype=torch.long, device=self.device) # (N, )
            ranks = self._rank_mapping.index_select(0, global_ranks) # (N, 6)
            etp_tensors = {}
            for etp_rank in range(ranks[:, 2].max().item() + 1):
                sub_tensor_dict = {}
                for global_rank, cor_ranks in zip(global_ranks, ranks):
                    if cor_ranks[2] != etp_rank: # same ETP rank
                        continue
                    sub_tensor_dict[global_rank.item()] = tensor_dict[global_rank.item()]
                etp_tensors[etp_rank] = torch.cat(deduplicate_and_sort(sub_tensor_dict, 3), dim=0)
            res = torch.cat([item[1] for item in sorted(etp_tensors.items())], dim=1)
            return res.permute(0, 2, 1).contiguous()

        merge_func_mapping = {
            ParamType.MOE_COLUMN: partial(merge_along_axis, 0, is_expert=True),
            ParamType.MOE_ROW: partial(merge_along_axis, 1, is_expert=True),
            ParamType.COLUMN: partial(merge_along_axis, 0),
            ParamType.ROW: partial(merge_along_axis, 1),
            ParamType.QKV_W: partial(merge_qkv, False),
            ParamType.QKV_B: partial(merge_qkv, True),
            ParamType.UNIQUE: no_merge_func,
            ParamType.QGKV_W: partial(merge_qgkv, False),
            ParamType.MOE_GATE_UP: merge_moe_gate_up_tensor,
            ParamType.MOE_DOWN: merge_moe_down_tensor,
            ParamType.MAMBA_CONV1D: merge_mamba_conv1d,
            ParamType.LINEAR_CONV1D: merge_linear_conv1d,
        }
        return merge_func_mapping[merge_type](tensor_dict)
