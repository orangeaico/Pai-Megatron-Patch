# Copyright (c) 2026 Alibaba PAI Team.
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
import json
import logging
import os
from types import SimpleNamespace

import torch
from transformers import AutoConfig

from general.m2h_synchronizer import MG2HFSynchronizer as _MG2HFSynchronizer
from general.synchronizer import ParamType


class MG2HFSynchronizer(_MG2HFSynchronizer):
    """MCore -> HF synchronizer for Qwen3.5 text + MTP."""

    _SAFETENSORS_DTYPE_TO_TORCH = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F64": torch.float64,
        "I8": torch.int8,
        "U8": torch.uint8,
        "I16": torch.int16,
        "I32": torch.int32,
        "I64": torch.int64,
        "BOOL": torch.bool,
    }

    def __init__(self, load_dir, model_provider_func=None):
        self._extra_hf_state = {}
        self._reference_hf_dtypes = {}
        super().__init__(load_dir, model_provider_func)
        rebuild_mapping = self._drop_visual_tower_if_present()
        self._reference_hf_dtypes = self._load_reference_hf_dtype_map()
        if self.rank == 0 and self._reference_hf_dtypes:
            logging.info(
                "Loaded %d reference HF dtypes for export casting.",
                len(self._reference_hf_dtypes),
            )
        if not hasattr(self._hfmodel, "mtp"):
            self._extra_hf_state = self._load_external_mtp_state_dict()
            if self.rank == 0 and self._extra_hf_state:
                logging.info(
                    "HF runtime model has no native `mtp` module; using mtp.* key metadata "
                    "from reference HF checkpoint for export."
                )
        if rebuild_mapping or hasattr(self._hfmodel, "mtp") or self._extra_hf_state:
            self.build_hf_mapping()
            self._merge_type = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)
        self._validate_parallel_args()
        self._validate_mtp_args()
        self.layout = self.get_hybrid_layout()
        if len(self.layout) != self.args.num_layers:
            raise ValueError(
                "Invalid hybrid layout length: "
                f"len(layout)={len(self.layout)} but num_layers={self.args.num_layers}."
            )

    def _get_mapping_state_dict(self):
        state_dict = self._hfmodel.state_dict(keep_vars=True)
        if not self._extra_hf_state:
            return self._canonicalize_state_dict_keys(state_dict)
        merged = dict(state_dict)
        merged.update(self._extra_hf_state)
        return self._canonicalize_state_dict_keys(merged)

    def _get_export_state_dict(self):
        state_dict = self._hfmodel.state_dict()
        if not self._extra_hf_state:
            return self._canonicalize_state_dict_keys(state_dict)
        merged = dict(state_dict)
        merged.update(self._extra_hf_state)
        return self._canonicalize_state_dict_keys(merged)

    @staticmethod
    def _canonicalize_hf_key(key: str) -> str:
        if (
            key.startswith("model.")
            and not key.startswith("model.language_model.")
            and not key.startswith("model.visual.")
        ):
            return "model.language_model." + key[len("model."):]
        return key

    def _canonicalize_state_dict_keys(self, state_dict):
        return {self._canonicalize_hf_key(key): value for key, value in state_dict.items()}

    def _load_reference_hf_dtype_map(self):
        hf_ref_dir = getattr(self.args, "hf_dir", None)
        if not hf_ref_dir:
            return {}

        index_path = os.path.join(hf_ref_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            if self.rank == 0:
                logging.warning("HF reference index not found at %s for dtype casting.", index_path)
            return {}

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        if not weight_map:
            return {}

        keys_by_shard = {}
        for key, filename in weight_map.items():
            keys_by_shard.setdefault(filename, []).append(key)

        shard_headers = {}
        for filename in keys_by_shard:
            shard_path = os.path.join(hf_ref_dir, filename)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(f"Missing shard `{filename}` referenced by {index_path}.")
            shard_headers[filename] = self._load_safetensors_header(shard_path)

        ref_dtypes = {}
        for filename, keys in keys_by_shard.items():
            header = shard_headers[filename]
            for key in keys:
                tensor_meta = header.get(key)
                if tensor_meta is None:
                    raise KeyError(f"Key `{key}` was not found in shard header `{filename}`.")
                ref_dtypes[self._canonicalize_hf_key(key)] = self._safetensors_dtype_to_torch(
                    tensor_meta["dtype"]
                )
        return ref_dtypes

    def _postprocess_output_tensor_before_save(self, key: str, tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = self._reference_hf_dtypes.get(key)
        if target_dtype is not None and tensor.dtype != target_dtype:
            tensor = tensor.to(dtype=target_dtype)
        return tensor

    @classmethod
    def _safetensors_dtype_to_torch(cls, dtype_name: str) -> torch.dtype:
        if dtype_name not in cls._SAFETENSORS_DTYPE_TO_TORCH:
            raise ValueError(f"Unsupported safetensors dtype `{dtype_name}`.")
        return cls._SAFETENSORS_DTYPE_TO_TORCH[dtype_name]

    @staticmethod
    def _load_safetensors_header(shard_path: str):
        with open(shard_path, "rb") as f:
            header_len = int.from_bytes(f.read(8), byteorder="little")
            header = json.loads(f.read(header_len))
        return header

    def _load_external_mtp_state_dict(self):
        hf_ref_dir = getattr(self.args, "hf_dir", None)
        if not hf_ref_dir:
            return {}

        index_path = os.path.join(hf_ref_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            if self.rank == 0:
                logging.warning(
                    "HF reference index is missing at %s; cannot preload mtp.* metadata.",
                    index_path,
                )
            return {}

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        mtp_weight_map = {
            key: filename for key, filename in weight_map.items() if key.startswith("mtp.")
        }
        if not mtp_weight_map:
            if self.rank == 0:
                logging.warning("No mtp.* keys found in %s.", index_path)
            return {}

        keys_by_shard = {}
        for key, filename in mtp_weight_map.items():
            keys_by_shard.setdefault(filename, []).append(key)

        shard_headers = {}
        for filename in keys_by_shard:
            shard_path = os.path.join(hf_ref_dir, filename)
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Missing shard `{filename}` referenced by {index_path}."
                )
            shard_headers[filename] = self._load_safetensors_header(shard_path)

        mtp_state = {}
        for filename, keys in keys_by_shard.items():
            header = shard_headers[filename]
            for key in keys:
                tensor_meta = header.get(key)
                if tensor_meta is None:
                    raise KeyError(
                        f"Key `{key}` was not found in shard header `{filename}`."
                    )
                dtype = self._safetensors_dtype_to_torch(tensor_meta["dtype"])
                shape = tuple(tensor_meta["shape"])
                mtp_state[key] = torch.empty(shape, dtype=dtype, device="meta")

        return mtp_state

    def _mtp_state_tensor(self, key: str):
        if key not in self._extra_hf_state:
            raise KeyError(f"Missing mtp key in export state: {key}")
        return self._extra_hf_state[key]

    def _build_mtp_proxy_from_state_dict(self):
        def ref(key):
            return SimpleNamespace(weight=self._mtp_state_tensor(key))

        layer_prefix = "mtp.layers.0"
        self_attn = SimpleNamespace(
            q_proj=ref(f"{layer_prefix}.self_attn.q_proj.weight"),
            k_proj=ref(f"{layer_prefix}.self_attn.k_proj.weight"),
            v_proj=ref(f"{layer_prefix}.self_attn.v_proj.weight"),
            o_proj=ref(f"{layer_prefix}.self_attn.o_proj.weight"),
            q_norm=ref(f"{layer_prefix}.self_attn.q_norm.weight"),
            k_norm=ref(f"{layer_prefix}.self_attn.k_norm.weight"),
        )

        experts = []
        for expert_id in range(self.args.num_experts):
            experts.append(
                SimpleNamespace(
                    gate_proj=ref(f"{layer_prefix}.mlp.experts.{expert_id}.gate_proj.weight"),
                    up_proj=ref(f"{layer_prefix}.mlp.experts.{expert_id}.up_proj.weight"),
                    down_proj=ref(f"{layer_prefix}.mlp.experts.{expert_id}.down_proj.weight"),
                )
            )

        shared_expert_gate_key = f"{layer_prefix}.mlp.shared_expert_gate.weight"
        shared_expert_gate = (
            ref(shared_expert_gate_key)
            if shared_expert_gate_key in self._extra_hf_state
            else None
        )

        mlp = SimpleNamespace(
            gate=ref(f"{layer_prefix}.mlp.gate.weight"),
            experts=experts,
            shared_expert=SimpleNamespace(
                gate_proj=ref(f"{layer_prefix}.mlp.shared_expert.gate_proj.weight"),
                up_proj=ref(f"{layer_prefix}.mlp.shared_expert.up_proj.weight"),
                down_proj=ref(f"{layer_prefix}.mlp.shared_expert.down_proj.weight"),
            ),
            shared_expert_gate=shared_expert_gate,
        )

        layer = SimpleNamespace(
            self_attn=self_attn,
            input_layernorm=ref(f"{layer_prefix}.input_layernorm.weight"),
            post_attention_layernorm=ref(f"{layer_prefix}.post_attention_layernorm.weight"),
            mlp=mlp,
        )

        return SimpleNamespace(
            pre_fc_norm_embedding=ref("mtp.pre_fc_norm_embedding.weight"),
            pre_fc_norm_hidden=ref("mtp.pre_fc_norm_hidden.weight"),
            fc=ref("mtp.fc.weight"),
            layers=[layer],
            norm=ref("mtp.norm.weight"),
        )

    def _drop_visual_tower_if_present(self) -> bool:
        dropped = False
        for root in [self._hfmodel, getattr(self._hfmodel, "model", None)]:
            if root is None:
                continue
            for name in ("visual", "vision_model", "vision_tower"):
                if hasattr(root, name):
                    delattr(root, name)
                    dropped = True
        if dropped and self.rank == 0:
            logging.info("Dropped HF vision tower for text-only qwen3.5 conversion.")
        return dropped

    def _validate_parallel_args(self):
        args = self.args
        if args.tensor_model_parallel_size <= 0:
            raise ValueError("tensor_model_parallel_size must be > 0.")
        if args.pipeline_model_parallel_size <= 0:
            raise ValueError("pipeline_model_parallel_size must be > 0.")
        if args.expert_model_parallel_size <= 0:
            raise ValueError("expert_model_parallel_size must be > 0.")
        if args.expert_tensor_parallel_size <= 0:
            raise ValueError("expert_tensor_parallel_size must be > 0.")

        if getattr(args, "num_layers_per_virtual_pipeline_stage", None) is not None:
            raise ValueError(
                "LAYERS_PER_VP / --num-layers-per-virtual-pipeline-stage is not supported "
                "for Qwen3.5-35B-A3B conversion."
            )

        if args.num_query_groups % args.tensor_model_parallel_size != 0:
            raise ValueError(
                "num_query_groups must be divisible by tensor_model_parallel_size. "
                f"Got num_query_groups={args.num_query_groups}, tp={args.tensor_model_parallel_size}."
            )
        mamba_num_groups = getattr(args, "mamba_num_groups", None)
        if mamba_num_groups is not None and mamba_num_groups % args.tensor_model_parallel_size != 0:
            raise ValueError(
                "mamba_num_groups must be divisible by tensor_model_parallel_size. "
                f"Got mamba_num_groups={mamba_num_groups}, tp={args.tensor_model_parallel_size}."
            )
        mamba_num_heads = getattr(args, "mamba_num_heads", None)
        if mamba_num_heads is not None and mamba_num_heads % args.tensor_model_parallel_size != 0:
            raise ValueError(
                "mamba_num_heads must be divisible by tensor_model_parallel_size. "
                f"Got mamba_num_heads={mamba_num_heads}, tp={args.tensor_model_parallel_size}."
            )
        if args.num_experts % args.expert_model_parallel_size != 0:
            raise ValueError(
                "num_experts must be divisible by expert_model_parallel_size. "
                f"Got num_experts={args.num_experts}, ep={args.expert_model_parallel_size}."
            )

    def _validate_mtp_args(self):
        if self.args.mtp_num_layers != 1:
            raise ValueError(
                "Qwen3.5-35B-A3B conversion requires --mtp-num-layers 1. "
                f"Found {self.args.mtp_num_layers}."
            )

    def _get_text_config(self):
        config = AutoConfig.from_pretrained(self.load_dir, trust_remote_code=True)
        return getattr(config, "text_config", config)

    def get_hybrid_layout(self) -> str:
        if self.args.hybrid_override_pattern:
            return self.args.hybrid_override_pattern

        text_config = self._get_text_config()
        layer_types = getattr(text_config, "layer_types", None)
        if not layer_types:
            raise ValueError("Cannot infer hybrid layout: `text_config.layer_types` is missing.")

        pattern_map = {
            "linear_attention": "M-",
            "full_attention": "*-",
        }
        layout = []
        for layer_type in layer_types:
            if layer_type not in pattern_map:
                raise ValueError(f"Unsupported layer type in text_config.layer_types: {layer_type}")
            layout.append(pattern_map[layer_type])
        return "".join(layout)

    def _get_hf_text_model(self, hf_model):
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
            return hf_model.model.language_model
        if hasattr(hf_model, "model"):
            return hf_model.model
        return hf_model

    @staticmethod
    def _get_layer_norm_weight(module, module_name):
        if hasattr(module, "layer_norm_weight"):
            return module.layer_norm_weight
        raise AttributeError(
            f"{module_name} does not expose `layer_norm_weight`. "
            "Use qwen3_next fused LayerNormLinear-compatible spec for conversion."
        )

    def sync_params(self, mg_model=None, hf_model=None):
        if mg_model is None:
            mg_model = self._mgmodel
        if hf_model is None:
            hf_model = self._hfmodel
        hf_text_model = self._get_hf_text_model(hf_model)

        if mg_model.pre_process:
            self.set_preprocess_state(mg_model=mg_model, hf_model=hf_text_model)

        if mg_model.post_process:
            self.set_postprocess_state(mg_model=mg_model, hf_text_model=hf_text_model)

        for mg_layer_id, global_mg_layer_id in self._build_pipeline_parallel_mapping().items():
            hf_layer_id = global_mg_layer_id // 2
            if (
                self.tp_rank == 0
                and self.ep_rank == 0
                and self.etp_rank == 0
                and global_mg_layer_id % 2 == 0
            ):
                logging.info(f"Converting layer {hf_layer_id}")

            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_text_model.layers[hf_layer_id]

            if global_mg_layer_id % 2 == 0:
                if hasattr(hf_layer, "linear_attn"):
                    self.set_mamba_layer_state(layer.mixer, hf_layer.linear_attn)
                    self.copy(
                        self._get_layer_norm_weight(layer.mixer.in_proj, "mixer.in_proj"),
                        hf_layer.input_layernorm.weight,
                    )
                elif hasattr(hf_layer, "self_attn"):
                    self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
                    self.copy(
                        self._get_layer_norm_weight(
                            layer.self_attention.linear_qgkv, "self_attention.linear_qgkv"
                        ),
                        hf_layer.input_layernorm.weight,
                    )
                else:
                    raise ValueError(
                        f"Layer {hf_layer_id} has neither linear_attn nor self_attn in HF model."
                    )
            else:
                self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
                self.copy(
                    layer.pre_mlp_layernorm.weight,
                    hf_layer.post_attention_layernorm.weight,
                )

        self.sync_mtp(mg_model=mg_model, hf_model=hf_model)

    def set_postprocess_state(self, mg_model, hf_text_model):
        self.copy(mg_model.decoder.final_norm.weight, hf_text_model.norm.weight)
        if mg_model.share_embeddings_and_output_weights:
            output_layer_weight = mg_model.shared_embedding_or_output_weight()
        else:
            output_layer_weight = mg_model.output_layer.weight
        self.copy(output_layer_weight, self._hfmodel.lm_head.weight, param_type=ParamType.COLUMN)

    def set_moe_layer_state(self, moe, hf_moe):
        self.copy(moe.router.weight, hf_moe.gate.weight)
        if moe.router.enable_expert_bias:
            self.copy(moe.router.expert_bias, hf_moe.gate.e_score_correction_bias)

        is_packed_expert = hasattr(hf_moe.experts, "gate_up_proj") and hasattr(
            hf_moe.experts, "down_proj"
        )
        if is_packed_expert:
            if self.args.moe_grouped_gemm:
                self.set_packed_group_mlp_state(moe.experts, hf_moe.experts)
            else:
                self.set_packed_sequential_mlp_state(moe.experts, hf_moe.experts)
        elif self.args.moe_grouped_gemm:
            if self.args.moe_use_legacy_grouped_gemm:
                raise NotImplementedError("Currently only TE GroupGEMM is implemented.")
            self.set_group_mlp_state(moe.experts, hf_moe.experts)
        else:
            self.set_sequential_mlp_state(moe.experts, hf_moe.experts)

        if moe.shared_experts is not None:
            hf_shared_expert_gate = getattr(hf_moe, "shared_expert_gate", None)
            gate_weight = getattr(moe.shared_experts, "gate_weight", None)
            if hf_shared_expert_gate is not None:
                if not isinstance(gate_weight, torch.Tensor):
                    raise ValueError(
                        "HF layer has `mlp.shared_expert_gate.weight` but MCore shared expert "
                        "gate is missing. Ensure model is built with `--moe-shared-expert-gate` "
                        "and re-run HF->MCore conversion."
                    )
                self.copy(gate_weight, hf_shared_expert_gate.weight)

            hidden_size = moe.shared_experts.linear_fc1.weight.shape[-1]
            gate_proj_weight, up_proj_weight = moe.shared_experts.linear_fc1.weight.reshape(
                2, -1, hidden_size
            )
            try:
                hf_shared_expert = hf_moe.shared_experts
            except AttributeError:
                hf_shared_expert = hf_moe.shared_expert
            self.copy(gate_proj_weight, hf_shared_expert.gate_proj.weight, param_type=ParamType.COLUMN)
            self.copy(up_proj_weight, hf_shared_expert.up_proj.weight, param_type=ParamType.COLUMN)
            self.copy(moe.shared_experts.linear_fc2.weight, hf_shared_expert.down_proj.weight, param_type=ParamType.ROW)

    def set_packed_group_mlp_state(self, experts, hf_experts):
        expert_ids = sorted(self._build_expert_parallel_mapping().keys())
        if len(expert_ids) == 0:
            return

        first_gate_up = getattr(experts.linear_fc1, f"weight{expert_ids[0]}")
        first_down = getattr(experts.linear_fc2, f"weight{expert_ids[0]}").transpose(0, 1)

        gate_up_shape = (len(expert_ids),) + tuple(first_gate_up.shape)
        down_shape = (len(expert_ids),) + tuple(first_down.shape)

        def _build_gate_up():
            return torch.stack(
                [getattr(experts.linear_fc1, f"weight{mg_expert_id}") for mg_expert_id in expert_ids],
                dim=0,
            )

        def _build_down():
            return torch.stack(
                [
                    getattr(experts.linear_fc2, f"weight{mg_expert_id}").transpose(0, 1)
                    for mg_expert_id in expert_ids
                ],
                dim=0,
            )

        self.copy_lazy(
            hf_experts.gate_up_proj,
            gate_up_shape,
            first_gate_up.dtype,
            _build_gate_up,
            param_type=ParamType.MOE_GATE_UP,
            name="packed_group_gate_up",
        )
        self.copy_lazy(
            hf_experts.down_proj,
            down_shape,
            first_down.dtype,
            _build_down,
            param_type=ParamType.MOE_DOWN,
            name="packed_group_down",
        )

    def set_packed_sequential_mlp_state(self, experts, hf_experts):
        local_experts = experts.local_experts
        expert_ids = sorted(self._build_expert_parallel_mapping().keys())
        if len(expert_ids) == 0:
            return

        first_gate_up = local_experts[expert_ids[0]].linear_fc1.weight
        first_down = local_experts[expert_ids[0]].linear_fc2.weight.transpose(0, 1)

        gate_up_shape = (len(expert_ids),) + tuple(first_gate_up.shape)
        down_shape = (len(expert_ids),) + tuple(first_down.shape)

        def _build_gate_up():
            return torch.stack(
                [local_experts[mg_expert_id].linear_fc1.weight for mg_expert_id in expert_ids],
                dim=0,
            )

        def _build_down():
            return torch.stack(
                [
                    local_experts[mg_expert_id].linear_fc2.weight.transpose(0, 1)
                    for mg_expert_id in expert_ids
                ],
                dim=0,
            )

        self.copy_lazy(
            hf_experts.gate_up_proj,
            gate_up_shape,
            first_gate_up.dtype,
            _build_gate_up,
            param_type=ParamType.MOE_GATE_UP,
            name="packed_seq_gate_up",
        )
        self.copy_lazy(
            hf_experts.down_proj,
            down_shape,
            first_down.dtype,
            _build_down,
            param_type=ParamType.MOE_DOWN,
            name="packed_seq_down",
        )

    def sync_mtp(self, mg_model, hf_model):
        if not hasattr(mg_model, "mtp"):
            raise ValueError(
                "MCore model does not contain `mtp`. "
                "Use the qwen3_5_text_mtp model provider with --mtp-num-layers 1."
            )

        mtp_layer = mg_model.mtp.layers[0]
        if hasattr(hf_model, "mtp"):
            hf_mtp = hf_model.mtp
        elif self._extra_hf_state:
            hf_mtp = self._build_mtp_proxy_from_state_dict()
        else:
            raise ValueError(
                "HF model does not contain `mtp` modules and no mtp.* reference "
                "metadata was loaded. Please provide --hf-dir pointing to the original "
                "HF checkpoint that contains mtp.* keys."
            )
        hf_mtp_layer = hf_mtp.layers[0]

        self.copy(mtp_layer.enorm.weight, hf_mtp.pre_fc_norm_embedding.weight)
        self.copy(mtp_layer.hnorm.weight, hf_mtp.pre_fc_norm_hidden.weight)
        self.copy(mtp_layer.eh_proj.weight, hf_mtp.fc.weight, param_type=ParamType.COLUMN)
        self.set_mtp_transformer_layer_state(mtp_layer.transformer_layer, hf_mtp_layer)
        self.copy(mtp_layer.final_layernorm.weight, hf_mtp.norm.weight)

    def set_mtp_transformer_layer_state(self, mtp_layer, hf_mtp_layer):
        self.set_gated_selfattn_state(mtp_layer.self_attention, hf_mtp_layer.self_attn)
        self.copy(
            self._get_layer_norm_weight(
                mtp_layer.self_attention.linear_qgkv, "mtp.self_attention.linear_qgkv"
            ),
            hf_mtp_layer.input_layernorm.weight,
        )
        self.set_moe_layer_state(mtp_layer.mlp, hf_mtp_layer.mlp)
        self.copy(
            mtp_layer.pre_mlp_layernorm.weight,
            hf_mtp_layer.post_attention_layernorm.weight,
        )

    def set_mamba_layer_state(self, mixer, hf_mixer):
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim,
        )
        split_size_list = [
            Dv * Nv // self.tp_size,
            Dv * Nv // self.tp_size,
            Dk * Nk // self.tp_size,
            Dk * Nk // self.tp_size,
            Nv // self.tp_size,
            Nv // self.tp_size,
        ]
        z, v, q, k, b, a = torch.split(mixer.in_proj.weight, split_size_list, dim=0)

        q_reshaped = q.reshape(Nk // self.tp_size, Dk, -1)
        k_reshaped = k.reshape(Nk // self.tp_size, Dk, -1)
        v_reshaped = v.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1)
        z_reshaped = z.reshape(Nk // self.tp_size, Dv * Nv // Nk, -1)
        b_reshaped = b.reshape(Nk // self.tp_size, Nv // Nk, -1)
        a_reshaped = a.reshape(Nk // self.tp_size, Nv // Nk, -1)

        if hasattr(hf_mixer, "in_proj_qkvz"):
            in_proj_qkvz_weight = torch.cat(
                [q_reshaped, k_reshaped, v_reshaped, z_reshaped], dim=1
            )
            self.copy(in_proj_qkvz_weight, hf_mixer.in_proj_qkvz.weight, param_type=ParamType.QKV_W)
            in_proj_ba_weight = torch.cat([b_reshaped, a_reshaped], dim=1)
            self.copy(in_proj_ba_weight, hf_mixer.in_proj_ba.weight, param_type=ParamType.QKV_W)
        else:
            in_proj_qkv_weight = torch.cat([q_reshaped, k_reshaped, v_reshaped], dim=1)
            self.copy(in_proj_qkv_weight, hf_mixer.in_proj_qkv.weight, param_type=ParamType.QKV_W)
            self.copy(z_reshaped, hf_mixer.in_proj_z.weight, param_type=ParamType.QKV_W)
            self.copy(b_reshaped, hf_mixer.in_proj_b.weight, param_type=ParamType.QKV_W)
            self.copy(a_reshaped, hf_mixer.in_proj_a.weight, param_type=ParamType.QKV_W)

        self.copy(mixer.dt_bias, hf_mixer.dt_bias, param_type=ParamType.COLUMN)
        self.copy(mixer.A_log, hf_mixer.A_log, param_type=ParamType.COLUMN)

        conv_v, conv_q, conv_k = torch.split(
            mixer.conv1d.weight,
            [
                Nv * Dv // self.tp_size,
                Nk * Dk // self.tp_size,
                Nk * Dk // self.tp_size,
            ],
            dim=0,
        )
        # Pack local shard as [Q || K || V], then let custom MAMBA_CONV1D merge
        # reassemble global HF ordering across TP ranks.
        conv_qkv_weight = torch.cat([conv_q, conv_k, conv_v], dim=0)
        self.copy(conv_qkv_weight, hf_mixer.conv1d.weight, param_type=ParamType.MAMBA_CONV1D)

        self.copy(mixer.norm.weight, hf_mixer.norm.weight, param_type=ParamType.UNIQUE)
        self.copy(mixer.out_proj.weight, hf_mixer.out_proj.weight, param_type=ParamType.ROW)

    def set_gated_selfattn_state(self, attn, hf_attn):
        tp = self.tp_size
        num_heads = self.args.num_attention_heads
        num_query_groups = (
            self.args.num_query_groups
            if self.args.group_query_attention
            else self.args.num_attention_heads
        )
        num_querys_per_group = num_heads // num_query_groups
        dim = self.args.kv_channels
        assert num_heads % num_querys_per_group == 0

        if self.args.qk_layernorm:
            self.copy(attn.q_layernorm.weight, hf_attn.q_norm.weight)
            self.copy(attn.k_layernorm.weight, hf_attn.k_norm.weight)

        attn_proj_weight = attn.linear_qgkv.weight.reshape(
            (num_query_groups // tp, (2 + num_querys_per_group * 2) * dim, -1)
        )
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            attn_proj_weight, [2 * num_querys_per_group * dim, dim, dim], dim=1
        )

        q_proj_weight = (
            q_proj_weight.reshape(num_query_groups // tp, 2, num_querys_per_group, dim, -1)
            .transpose(1, 2)
            .flatten(1, 3)
        )
        self.copy(q_proj_weight, hf_attn.q_proj.weight, param_type=ParamType.QKV_W)
        self.copy(k_proj_weight, hf_attn.k_proj.weight, param_type=ParamType.QKV_W)
        self.copy(v_proj_weight, hf_attn.v_proj.weight, param_type=ParamType.QKV_W)
        self.copy(attn.linear_proj.weight, hf_attn.o_proj.weight, param_type=ParamType.ROW)

        if self.args.add_qkv_bias:
            attn_proj_bias = attn.linear_qgkv.bias.reshape(
                (num_query_groups // tp, (2 + num_querys_per_group * 2) * dim, -1)
            )
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                attn_proj_bias,
                [2 * num_querys_per_group * dim, dim, dim],
                dim=1,
            )
            self.copy(q_proj_bias, hf_attn.q_proj.bias, param_type=ParamType.QKV_B)
            self.copy(k_proj_bias, hf_attn.k_proj.bias, param_type=ParamType.QKV_B)
            self.copy(v_proj_bias, hf_attn.v_proj.bias, param_type=ParamType.QKV_B)
