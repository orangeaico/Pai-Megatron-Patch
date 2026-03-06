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

from general.h2m_synchronizer import HF2MGSynchronizer as _HF2MGSynchronizer
from general.synchronizer import ParamType


class HF2MGSynchronizer(_HF2MGSynchronizer):
    """HF -> MCore synchronizer for Qwen3.5 text + MTP."""

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
        super().__init__(load_dir, model_provider_func)
        if self._drop_visual_tower_if_present():
            self.build_hf_mapping()
            if self.debug:
                self._visit = torch.zeros([self.hf_size], dtype=torch.int, device=self.device)
        self._reference_hf_dtypes = self._load_reference_hf_dtype_map()
        self._dtype_audit = os.getenv("QWEN35_DTYPE_AUDIT", "0") == "1"
        if self.rank == 0 and self._reference_hf_dtypes:
            logging.info(
                "Loaded %d reference HF dtypes for HF->MCore casting guard.",
                len(self._reference_hf_dtypes),
            )
        self._validate_parallel_args()
        self._validate_mtp_args()

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
        linear_num_key_heads = getattr(args, "linear_num_key_heads", None)
        if (
            linear_num_key_heads is not None
            and linear_num_key_heads % args.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                "linear_num_key_heads must be divisible by tensor_model_parallel_size. "
                f"Got linear_num_key_heads={linear_num_key_heads}, tp={args.tensor_model_parallel_size}."
            )
        linear_num_value_heads = getattr(args, "linear_num_value_heads", None)
        if (
            linear_num_value_heads is not None
            and linear_num_value_heads % args.tensor_model_parallel_size != 0
        ):
            raise ValueError(
                "linear_num_value_heads must be divisible by tensor_model_parallel_size. "
                f"Got linear_num_value_heads={linear_num_value_heads}, tp={args.tensor_model_parallel_size}."
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

    def _get_hf_text_model(self, hf_model):
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
            return hf_model.model.language_model
        if hasattr(hf_model, "model"):
            return hf_model.model
        return hf_model

    @staticmethod
    def _get_attn_qkv_module(attn, module_name):
        qkv = getattr(attn, "linear_qgkv", None)
        if qkv is None:
            qkv = getattr(attn, "linear_qkv", None)
        if qkv is None:
            raise AttributeError(
                f"{module_name} exposes neither `linear_qgkv` nor `linear_qkv`."
            )
        return qkv

    @classmethod
    def _resolve_input_layernorm_weight(cls, layer, module_name):
        attn = getattr(layer, "self_attention", None)
        if attn is not None:
            in_proj = getattr(attn, "in_proj", None)
            if in_proj is not None and hasattr(in_proj, "layer_norm_weight"):
                return in_proj.layer_norm_weight
            qkv = getattr(attn, "linear_qgkv", None)
            if qkv is None:
                qkv = getattr(attn, "linear_qkv", None)
            if qkv is not None and hasattr(qkv, "layer_norm_weight"):
                return qkv.layer_norm_weight
        input_ln = getattr(layer, "input_layernorm", None)
        if input_ln is not None and hasattr(input_ln, "weight"):
            return input_ln.weight
        raise AttributeError(
            f"Cannot resolve input layernorm target for {module_name}. "
            "Expected fused layernorm weight (`*.layer_norm_weight`) or "
            "`layer.input_layernorm.weight`."
        )

    @staticmethod
    def _resolve_pre_mlp_layernorm_weight(layer, module_name):
        pre_mlp_ln = getattr(layer, "pre_mlp_layernorm", None)
        if pre_mlp_ln is not None and hasattr(pre_mlp_ln, "weight"):
            return pre_mlp_ln.weight
        mlp = getattr(layer, "mlp", None)
        if mlp is not None:
            linear_fc1 = getattr(mlp, "linear_fc1", None)
            if linear_fc1 is not None and hasattr(linear_fc1, "layer_norm_weight"):
                return linear_fc1.layer_norm_weight
        raise AttributeError(
            f"Cannot resolve pre-MLP layernorm target for {module_name}. "
            "Expected `layer.pre_mlp_layernorm.weight` or "
            "`layer.mlp.linear_fc1.layer_norm_weight`."
        )

    @staticmethod
    def _resolve_linear_attn_norm_module(attn, module_name):
        norm = getattr(attn, "out_norm", None)
        if norm is None:
            norm = getattr(attn, "norm", None)
        if norm is None:
            raise AttributeError(
                f"{module_name} does not expose `out_norm` or `norm`."
            )
        return norm

    @staticmethod
    def _canonicalize_hf_key(key: str) -> str:
        if (
            key.startswith("model.")
            and not key.startswith("model.language_model.")
            and not key.startswith("model.visual.")
        ):
            return "model.language_model." + key[len("model."):]
        return key

    @classmethod
    def _safetensors_dtype_to_torch(cls, dtype_name: str) -> torch.dtype:
        if dtype_name not in cls._SAFETENSORS_DTYPE_TO_TORCH:
            raise ValueError(f"Unsupported safetensors dtype `{dtype_name}`.")
        return cls._SAFETENSORS_DTYPE_TO_TORCH[dtype_name]

    @staticmethod
    def _load_safetensors_header(shard_path: str):
        with open(shard_path, "rb") as f:
            header_len = int.from_bytes(f.read(8), byteorder="little")
            return json.loads(f.read(header_len))

    @staticmethod
    def _is_legacy_grouped_mlp(experts) -> bool:
        return hasattr(experts, "weight1") and hasattr(experts, "weight2")

    def _load_reference_hf_dtype_map(self):
        index_path = os.path.join(self.load_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            if self.rank == 0:
                logging.warning("HF source index not found at %s for dtype guard.", index_path)
            return {}

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})
        if not weight_map:
            return {}

        headers = {}
        ref_dtypes = {}
        for key, filename in weight_map.items():
            shard_path = os.path.join(self.load_dir, filename)
            if filename not in headers:
                if not os.path.exists(shard_path):
                    raise FileNotFoundError(f"Missing shard `{filename}` referenced by {index_path}.")
                headers[filename] = self._load_safetensors_header(shard_path)
            tensor_meta = headers[filename].get(key)
            if tensor_meta is None:
                raise KeyError(f"Key `{key}` was not found in shard header `{filename}`.")
            ref_dtypes[self._canonicalize_hf_key(key)] = self._safetensors_dtype_to_torch(
                tensor_meta["dtype"]
            )
        return ref_dtypes

    def _resolve_src_key_from_copy_input(self, src_tensor):
        if isinstance(src_tensor, str):
            key = src_tensor
        elif isinstance(src_tensor, (list, tuple)):
            return None
        else:
            key = self._hf_params_to_key.get(src_tensor)

        if key is None:
            return None
        if not self.args.untie_embeddings_and_output_weights and key == "lm_head.weight":
            key = "model.embed_tokens.weight"
        return self._canonicalize_hf_key(key)

    def _should_preserve_source_dtype(self, src_tensor):
        src_key = self._resolve_src_key_from_copy_input(src_tensor)
        if src_key is None:
            return False, None
        target_dtype = self._reference_hf_dtypes.get(src_key)
        return target_dtype == torch.float32, src_key

    def _copy_impl(self, src_tensor, dst_tensor, param_type: ParamType = ParamType.UNIQUE):
        tp_rank, tp_size = self.tp_rank, self.tp_size
        if param_type in [ParamType.MOE_COLUMN, ParamType.MOE_ROW, ParamType.MOE_GATE_UP, ParamType.MOE_DOWN]:
            tp_rank, tp_size = self.etp_rank, self.etp_size

        split_mapping = {
            ParamType.UNIQUE: lambda x: self.load_tensor(x),
            ParamType.COLUMN: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=0)[tp_rank],
            ParamType.ROW: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=1)[tp_rank],
            ParamType.GATE_UP: lambda x: torch.chunk(x, tp_size, dim=1)[tp_rank].flatten(0, 1),
            ParamType.QKV_W: lambda x: torch.chunk(x, tp_size, dim=0)[tp_rank].flatten(0, 1),
            ParamType.QKV_B: lambda x: torch.chunk(x, tp_size, dim=0)[tp_rank].flatten(),
            ParamType.MOE_COLUMN: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=0)[tp_rank],
            ParamType.MOE_ROW: lambda x: torch.chunk(self.load_tensor(x), tp_size, dim=1)[tp_rank],
            ParamType.MOE_DOWN: lambda x: torch.chunk(x, tp_size, dim=1)[tp_rank],
            ParamType.MOE_GATE_UP: lambda x: torch.chunk(x, tp_size, dim=1)[tp_rank].flatten(0, 1),
            ParamType.MERGED_LINEAR: lambda lst: torch.cat(
                [torch.chunk(x, tp_size, dim=0)[tp_rank].flatten(0, 1) for x in lst], dim=0
            ),
        }

        if self.dryrun:
            return dst_tensor.data.copy_(dst_tensor.clone())

        src_shard = split_mapping[param_type](src_tensor).to(device=dst_tensor.device)
        preserve_dtype, src_key = self._should_preserve_source_dtype(src_tensor)
        if preserve_dtype and src_shard.dtype != dst_tensor.dtype:
            if self._dtype_audit and self.rank == 0:
                logging.info(
                    "Preserving source dtype for `%s`: src=%s dst=%s",
                    src_key,
                    src_shard.dtype,
                    dst_tensor.dtype,
                )
            dst_tensor.data = src_shard.clone()
            return

        dst_tensor.data.copy_(src_shard)

    def _copy_linear_attn_norm_exact_fallback(self, hf_norm_weight, mg_norm_weight):
        if self.dryrun:
            return

        src = self.load_tensor(hf_norm_weight).to(device=mg_norm_weight.device)
        if src.shape != mg_norm_weight.shape:
            raise ValueError(
                "Shape mismatch when applying linear_attn.norm exact fallback: "
                f"hf={src.shape}, mcore={mg_norm_weight.shape}"
            )
        if torch.equal(src.float(), mg_norm_weight.float()):
            return

        key = self._resolve_src_key_from_copy_input(hf_norm_weight) or "<unknown-key>"
        max_delta = (src.float() - mg_norm_weight.float()).abs().max().item()
        logging.warning(
            "Applying linear_attn.norm exact fallback for `%s` (max_delta=%s).",
            key,
            max_delta,
        )
        mg_norm_weight.data = src.clone()

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
            hf_layer_id = global_mg_layer_id
            if self.tp_rank == 0 and self.ep_rank == 0 and self.etp_rank == 0:
                logging.info(f"Converting layer {hf_layer_id}")

            layer = mg_model.decoder.layers[mg_layer_id]
            hf_layer = hf_text_model.layers[hf_layer_id]
            if hasattr(hf_layer, "linear_attn"):
                self.set_linear_attn_layer_state(layer.self_attention, hf_layer.linear_attn)
            elif hasattr(hf_layer, "self_attn"):
                self.set_gated_selfattn_state(layer.self_attention, hf_layer.self_attn)
            else:
                raise ValueError(
                    f"Layer {hf_layer_id} has neither linear_attn nor self_attn in HF model."
                )
            self.copy(
                hf_layer.input_layernorm.weight,
                self._resolve_input_layernorm_weight(layer, f"decoder.layers.{hf_layer_id}.self_attention"),
            )

            self.set_moe_layer_state(layer.mlp, hf_layer.mlp)
            self.copy(
                hf_layer.post_attention_layernorm.weight,
                self._resolve_pre_mlp_layernorm_weight(layer, f"decoder.layers.{hf_layer_id}.mlp"),
            )

        self.sync_mtp(mg_model=mg_model, hf_model=hf_model)

    def set_postprocess_state(self, mg_model, hf_text_model):
        final_norm = getattr(mg_model.decoder, "final_norm", None)
        if final_norm is None:
            final_norm = getattr(mg_model.decoder, "final_layernorm", None)
        if final_norm is None or not hasattr(final_norm, "weight"):
            raise AttributeError("Cannot resolve decoder final norm weight for MCore model.")
        self.copy(hf_text_model.norm.weight, final_norm.weight)
        if mg_model.share_embeddings_and_output_weights:
            output_layer_weight = mg_model.shared_embedding_or_output_weight()
        else:
            output_layer_weight = mg_model.output_layer.weight
        self.copy(self._hfmodel.lm_head.weight, output_layer_weight, param_type=ParamType.COLUMN)

    def set_moe_layer_state(self, moe, hf_moe):
        self.copy(hf_moe.gate.weight, moe.router.weight)
        if moe.router.enable_expert_bias:
            self.copy(hf_moe.gate.e_score_correction_bias, moe.router.expert_bias)

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
            if hf_shared_expert_gate is not None:
                if not moe.shared_experts.use_shared_expert_gate:
                    raise ValueError(
                        "HF checkpoint contains `mlp.shared_expert_gate.weight` but MCore "
                        "model was built without shared expert gate. Add "
                        "`--moe-shared-expert-gate` and re-run conversion."
                    )
                self.copy(hf_shared_expert_gate.weight, moe.shared_experts.gate_weight)

            try:
                hf_shared_expert = hf_moe.shared_experts
            except AttributeError:
                hf_shared_expert = hf_moe.shared_expert
            self.set_mlp_state(moe.shared_experts, hf_shared_expert)

    def set_packed_group_mlp_state(self, experts, hf_experts):
        gate_up_proj = self.load_tensor(hf_experts.gate_up_proj)
        down_proj = self.load_tensor(hf_experts.down_proj)
        hidden_size = gate_up_proj.shape[-1]
        legacy_gate_up = None
        legacy_down = None
        if self._is_legacy_grouped_mlp(experts):
            legacy_gate_up = experts.weight1.view(
                experts.num_local_experts, experts.config.hidden_size, -1
            ).transpose(1, 2)
            legacy_down = experts.weight2.view(
                experts.num_local_experts, -1, experts.config.hidden_size
            ).transpose(1, 2)

        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            gate_up_weight = (
                legacy_gate_up[mg_expert_id]
                if legacy_gate_up is not None
                else getattr(experts.linear_fc1, f"weight{mg_expert_id}")
            )
            down_weight = (
                legacy_down[mg_expert_id]
                if legacy_down is not None
                else getattr(experts.linear_fc2, f"weight{mg_expert_id}")
            )
            self.copy(
                gate_up_proj[hf_expert_id].reshape(2, -1, hidden_size),
                gate_up_weight,
                param_type=ParamType.MOE_GATE_UP,
            )
            self.copy(
                down_proj[hf_expert_id],
                down_weight,
                param_type=ParamType.MOE_DOWN,
            )

    def set_packed_sequential_mlp_state(self, experts, hf_experts):
        gate_up_proj = self.load_tensor(hf_experts.gate_up_proj)
        down_proj = self.load_tensor(hf_experts.down_proj)
        hidden_size = gate_up_proj.shape[-1]
        local_experts = experts.local_experts

        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            self.copy(
                gate_up_proj[hf_expert_id].reshape(2, -1, hidden_size),
                local_experts[mg_expert_id].linear_fc1.weight,
                param_type=ParamType.MOE_GATE_UP,
            )
            self.copy(
                down_proj[hf_expert_id],
                local_experts[mg_expert_id].linear_fc2.weight,
                param_type=ParamType.MOE_DOWN,
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
        else:
            if self.rank == 0:
                logging.warning(
                    "HF model object has no `mtp`; loading MTP tensors directly from checkpoint keys."
                )
            hf_mtp = self._build_mtp_proxy_from_checkpoint_keys()
        hf_mtp_layer = hf_mtp.layers[0]

        self.copy(hf_mtp.pre_fc_norm_embedding.weight, mtp_layer.enorm.weight)
        self.copy(hf_mtp.pre_fc_norm_hidden.weight, mtp_layer.hnorm.weight)
        self.copy(hf_mtp.fc.weight, mtp_layer.eh_proj.weight, param_type=ParamType.COLUMN)
        self.set_mtp_transformer_layer_state(mtp_layer.transformer_layer, hf_mtp_layer)
        self.copy(hf_mtp.norm.weight, mtp_layer.final_layernorm.weight)

    def _build_mtp_proxy_from_checkpoint_keys(self):
        def _w(key):
            return SimpleNamespace(weight=key)

        experts = [
            SimpleNamespace(
                gate_proj=_w(f"mtp.layers.0.mlp.experts.{expert_id}.gate_proj.weight"),
                up_proj=_w(f"mtp.layers.0.mlp.experts.{expert_id}.up_proj.weight"),
                down_proj=_w(f"mtp.layers.0.mlp.experts.{expert_id}.down_proj.weight"),
            )
            for expert_id in range(self.args.num_experts)
        ]

        mtp_layer = SimpleNamespace(
            input_layernorm=_w("mtp.layers.0.input_layernorm.weight"),
            post_attention_layernorm=_w("mtp.layers.0.post_attention_layernorm.weight"),
            self_attn=SimpleNamespace(
                q_norm=_w("mtp.layers.0.self_attn.q_norm.weight"),
                k_norm=_w("mtp.layers.0.self_attn.k_norm.weight"),
                q_proj=_w("mtp.layers.0.self_attn.q_proj.weight"),
                k_proj=_w("mtp.layers.0.self_attn.k_proj.weight"),
                v_proj=_w("mtp.layers.0.self_attn.v_proj.weight"),
                o_proj=_w("mtp.layers.0.self_attn.o_proj.weight"),
            ),
            mlp=SimpleNamespace(
                gate=_w("mtp.layers.0.mlp.gate.weight"),
                experts=experts,
                shared_expert_gate=_w("mtp.layers.0.mlp.shared_expert_gate.weight"),
                shared_expert=SimpleNamespace(
                    gate_proj=_w("mtp.layers.0.mlp.shared_expert.gate_proj.weight"),
                    up_proj=_w("mtp.layers.0.mlp.shared_expert.up_proj.weight"),
                    down_proj=_w("mtp.layers.0.mlp.shared_expert.down_proj.weight"),
                ),
            ),
        )

        return SimpleNamespace(
            pre_fc_norm_embedding=_w("mtp.pre_fc_norm_embedding.weight"),
            pre_fc_norm_hidden=_w("mtp.pre_fc_norm_hidden.weight"),
            fc=_w("mtp.fc.weight"),
            norm=_w("mtp.norm.weight"),
            layers=[mtp_layer],
        )

    def set_mtp_transformer_layer_state(self, mtp_layer, hf_mtp_layer):
        self.set_gated_selfattn_state(mtp_layer.self_attention, hf_mtp_layer.self_attn)
        self.copy(
            hf_mtp_layer.input_layernorm.weight,
            self._resolve_input_layernorm_weight(mtp_layer, "mtp.transformer_layer.self_attention"),
        )
        self.set_mtp_moe_layer_state(mtp_layer.mlp, hf_mtp_layer.mlp)
        self.copy(
            hf_mtp_layer.post_attention_layernorm.weight,
            self._resolve_pre_mlp_layernorm_weight(mtp_layer, "mtp.transformer_layer.mlp"),
        )

    def set_mtp_moe_layer_state(self, moe, hf_moe):
        # Router and shared experts follow regular MoE mapping.
        self.copy(hf_moe.gate.weight, moe.router.weight)
        if moe.router.enable_expert_bias:
            self.copy(hf_moe.gate.e_score_correction_bias, moe.router.expert_bias)

        # MTP experts are saved as per-expert gate/up/down tensors.
        local_experts = moe.experts.local_experts
        for mg_expert_id, hf_expert_id in self._build_expert_parallel_mapping().items():
            hf_expert = hf_moe.experts[hf_expert_id]
            if self.dryrun:
                gate_up_proj_weight = local_experts[mg_expert_id].linear_fc1.weight
            else:
                gate_up_proj_weight = torch.stack(
                    [
                        self.load_tensor(hf_expert.gate_proj.weight),
                        self.load_tensor(hf_expert.up_proj.weight),
                    ]
                )

            self.copy(
                gate_up_proj_weight,
                local_experts[mg_expert_id].linear_fc1.weight,
                param_type=ParamType.MOE_GATE_UP,
            )
            self.copy(
                hf_expert.down_proj.weight,
                local_experts[mg_expert_id].linear_fc2.weight,
                param_type=ParamType.MOE_ROW,
            )

        if moe.shared_experts is not None:
            hf_shared_expert_gate = getattr(hf_moe, "shared_expert_gate", None)
            if hf_shared_expert_gate is not None:
                if not moe.shared_experts.use_shared_expert_gate:
                    raise ValueError(
                        "HF checkpoint contains `mlp.shared_expert_gate.weight` but MCore "
                        "model was built without shared expert gate. Add "
                        "`--moe-shared-expert-gate` and re-run conversion."
                    )
                self.copy(hf_shared_expert_gate.weight, moe.shared_experts.gate_weight)
            try:
                hf_shared_expert = hf_moe.shared_experts
            except AttributeError:
                hf_shared_expert = hf_moe.shared_expert
            self.set_mlp_state(moe.shared_experts, hf_shared_expert)

    def set_linear_attn_layer_state(self, attn, hf_mixer):
        # HF linear attention: [Q, K, V] + Z + B + A
        # MCore GatedDeltaNet in_proj order: [Q, K, V, Z, B, A]
        Nk, Nv, Dk, Dv = (
            hf_mixer.num_k_heads,
            hf_mixer.num_v_heads,
            hf_mixer.head_k_dim,
            hf_mixer.head_v_dim,
        )

        if hasattr(hf_mixer, "in_proj_qkvz"):
            q, k, v, z = torch.split(
                self.load_tensor(hf_mixer.in_proj_qkvz.weight).reshape(
                    Nk, 2 * Dk + 2 * Dv * Nv // Nk, -1
                ),
                [Dk, Dk, Dv * Nv // Nk, Dv * Nv // Nk],
                dim=1,
            )
            b, a = self.load_tensor(hf_mixer.in_proj_ba.weight).reshape(
                Nk, 2 * Nv // Nk, -1
            ).chunk(chunks=2, dim=1)
        else:
            q, k, v = torch.split(
                self.load_tensor(hf_mixer.in_proj_qkv.weight).reshape(
                    Nk, 2 * Dk + Dv * Nv // Nk, -1
                ),
                [Dk, Dk, Dv * Nv // Nk],
                dim=1,
            )
            z = self.load_tensor(hf_mixer.in_proj_z.weight).reshape(Nk, Dv * Nv // Nk, -1)
            b = self.load_tensor(hf_mixer.in_proj_b.weight).reshape(Nk, Nv // Nk, -1)
            a = self.load_tensor(hf_mixer.in_proj_a.weight).reshape(Nk, Nv // Nk, -1)

        self.copy([q, k, v, z, b, a], attn.in_proj.weight, param_type=ParamType.MERGED_LINEAR)

        self.copy(hf_mixer.dt_bias, attn.dt_bias, param_type=ParamType.COLUMN)
        self.copy(hf_mixer.A_log, attn.A_log, param_type=ParamType.COLUMN)

        conv_q, conv_k, conv_v = torch.split(
            self.load_tensor(hf_mixer.conv1d.weight),
            [Nk * Dk, Nk * Dk, Nv * Dv],
            dim=0,
        )
        self.copy(
            [
                conv_q.reshape(Nk, Dk, 1, -1),
                conv_k.reshape(Nk, Dk, 1, -1),
                conv_v.reshape(Nk, Dv * Nv // Nk, 1, -1),
            ],
            attn.conv1d.weight,
            param_type=ParamType.MERGED_LINEAR,
        )

        attn_norm = self._resolve_linear_attn_norm_module(attn, "linear_attention")
        self.copy(hf_mixer.norm.weight, attn_norm.weight)
        self._copy_linear_attn_norm_exact_fallback(hf_mixer.norm.weight, attn_norm.weight)
        self.copy(hf_mixer.out_proj.weight, attn.out_proj.weight, param_type=ParamType.ROW)

    def set_gated_selfattn_state(self, attn, hf_attn):
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
            self.copy(hf_attn.q_norm.weight, attn.q_layernorm.weight)
            self.copy(hf_attn.k_norm.weight, attn.k_layernorm.weight)

        qkv_module = self._get_attn_qkv_module(attn, "self_attention")
        if self.dryrun:
            attn_proj_weight = qkv_module.weight
        else:
            attn_proj_weight = torch.cat(
                [
                    self.load_tensor(hf_attn.q_proj.weight)
                    .reshape((num_query_groups, num_querys_per_group, 2, dim, -1))
                    .transpose(1, 2)
                    .flatten(1, 3),
                    self.load_tensor(hf_attn.k_proj.weight).reshape((num_query_groups, dim, -1)),
                    self.load_tensor(hf_attn.v_proj.weight).reshape((num_query_groups, dim, -1)),
                ],
                dim=1,
            )
        self.copy(
            attn_proj_weight,
            qkv_module.weight,
            param_type=ParamType.QKV_W,
        )
        self.copy(hf_attn.o_proj.weight, attn.linear_proj.weight, param_type=ParamType.ROW)

        if self.args.add_qkv_bias and getattr(qkv_module, "bias", None) is not None:
            if self.dryrun:
                attn_proj_bias = qkv_module.bias
            else:
                attn_proj_bias = torch.cat(
                    [
                        self.load_tensor(hf_attn.q_proj.bias).reshape(
                            (num_query_groups, num_querys_per_group * dim, -1)
                        ),
                        self.load_tensor(hf_attn.k_proj.bias).reshape((num_query_groups, dim, -1)),
                        self.load_tensor(hf_attn.v_proj.bias).reshape((num_query_groups, dim, -1)),
                    ],
                    dim=1,
                )
            self.copy(
                attn_proj_bias,
                qkv_module.bias,
                param_type=ParamType.QKV_B,
            )
