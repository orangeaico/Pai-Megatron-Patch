# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
from typing import Optional

import torch
from torch import nn

from megatron.core.extensions.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
    TENorm,
)
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider, LocalSpecProvider
from megatron.core.ssm.mamba_block import MambaStack, MambaStackSubmodules
from megatron.core.ssm.mamba_layer import MambaLayer, MambaLayerSubmodules
from megatron.core.ssm.mamba_mixer import MambaMixerSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules

from megatron_patch.model.qwen3_next.gated_attention import GatedSoftmaxAttention

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from megatron_patch.model.qwen3_next.gated_deltanet import GatedDeltaNetMixer


class LocalLayerNormColumnParallelLinear(ColumnParallelLinear):
    """CPU/local fallback for TE fused LayerNorm+ColumnParallelLinear.

    The synchronizers expect `layer_norm_weight` to exist on this module.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config,
        init_method,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation=False,
        embedding_activation_buffer=None,
        grad_output_buffer=None,
        is_expert=False,
        tp_comm_buffer_name=None,
        disable_grad_reduce=False,
        tp_group=None,
        normalization="RMSNorm",
        eps=1e-5,
        zero_centered_gamma=False,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name,
            disable_grad_reduce=disable_grad_reduce,
            tp_group=tp_group,
        )

        if config.use_cpu_initialization or not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device("cuda", torch.cuda.current_device())

        self.layer_norm_epsilon = eps
        self.normalization = normalization
        self.zero_centered_gamma = zero_centered_gamma
        self.layer_norm_weight = nn.Parameter(
            torch.ones(input_size, device=device, dtype=config.params_dtype)
        )
        if normalization == "LayerNorm":
            self.layer_norm_bias = nn.Parameter(
                torch.zeros(input_size, device=device, dtype=config.params_dtype)
            )
        else:
            self.register_parameter("layer_norm_bias", None)

    def _apply_input_norm(self, x):
        if self.normalization == "LayerNorm":
            mean = x.mean(dim=-1, keepdim=True)
            variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            y = (x - mean) * torch.rsqrt(variance + self.layer_norm_epsilon)
            if self.layer_norm_bias is None:
                return y * self.layer_norm_weight
            return y * self.layer_norm_weight + self.layer_norm_bias

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.layer_norm_epsilon) * self.layer_norm_weight

    def forward(self, input_, weight=None, runtime_gather_output=None):
        normed_input = self._apply_input_norm(input_)
        return super().forward(
            normed_input,
            weight=weight,
            runtime_gather_output=runtime_gather_output,
        )


class SharedExpertMLPCompat(SharedExpertMLP):
    """Compatibility wrapper for Megatron backend API differences.

    Some backends pass `gate` from MoELayer, some rely on ModuleSpec params.
    Accept both and default to config when absent.
    """

    def __init__(self, config, submodules, gate=None, pg_collection=None, **kwargs):
        if gate is None:
            gate = bool(getattr(config, "moe_shared_expert_gate", False))
        super().__init__(
            config=config,
            submodules=submodules,
            gate=gate,
            pg_collection=pg_collection,
        )


def get_moe_module_spec(
    use_te: Optional[bool] = True,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    if use_te is not None and use_te:
        if not HAVE_TE:
            backend = LocalSpecProvider()
        else:
            backend = TESpecProvider()
    else:
        backend = LocalSpecProvider()
    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        moe_use_legacy_grouped_gemm=moe_use_legacy_grouped_gemm,
    )


def _use_transformer_engine(args) -> bool:
    requested_impl = getattr(args, "transformer_impl", "transformer_engine")
    if requested_impl != "transformer_engine":
        return False
    if not HAVE_TE:
        return False
    return torch.cuda.is_available()


def get_qwen3_next_layer_spec(args):
    use_te = _use_transformer_engine(args)

    if use_te:
        backend: BackendSpecProvider = TESpecProvider()
        in_proj_cls = TELayerNormColumnParallelLinear
        row_linear_cls = TERowParallelLinear
        core_attention_cls = TEDotProductAttention
        qk_norm_cls = TENorm
        mlp_norm_cls = TENorm
    else:
        backend = LocalSpecProvider()
        in_proj_cls = LocalLayerNormColumnParallelLinear
        row_linear_cls = backend.row_parallel_linear()
        core_attention_cls = backend.core_attention()
        qk_norm_cls = backend.layer_norm(rms_norm=True, for_qk=True)
        mlp_norm_cls = backend.layer_norm(rms_norm=True)

    return ModuleSpec(
        module=MambaStack,
        submodules=MambaStackSubmodules(
            mamba_layer=ModuleSpec(
                module=MambaLayer,
                submodules=MambaLayerSubmodules(
                    mixer=ModuleSpec(
                        module=GatedDeltaNetMixer,
                        submodules=MambaMixerSubmodules(
                            in_proj=in_proj_cls,
                            out_proj=row_linear_cls,
                        ),
                    ),
                    mamba_bda=get_bias_dropout_add,
                ),
            ),
            attention_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    self_attention=ModuleSpec(
                        module=GatedSoftmaxAttention,
                        params={"attn_mask_type": AttnMaskType.causal},
                        submodules=SelfAttentionSubmodules(
                            linear_qkv=in_proj_cls,
                            core_attention=core_attention_cls,
                            linear_proj=row_linear_cls,
                            q_layernorm=qk_norm_cls,
                            k_layernorm=qk_norm_cls,
                        ),
                    ),
                    self_attn_bda=get_bias_dropout_add,
                ),
            ),
            mlp_layer=ModuleSpec(
                module=TransformerLayer,
                submodules=TransformerLayerSubmodules(
                    pre_mlp_layernorm=mlp_norm_cls,
                    mlp=get_moe_module_spec(
                        use_te=use_te,
                        num_experts=args.num_experts,
                        moe_grouped_gemm=args.moe_grouped_gemm,
                    ),
                    mlp_bda=get_bias_dropout_add,
                ),
            ),
        ),
    )


def get_moe_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = False,
    moe_use_legacy_grouped_gemm: Optional[bool] = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MoE"""
    del use_te_activation_func
    assert num_experts is not None

    linear_fc1 = backend.column_parallel_linear()
    linear_fc2 = backend.row_parallel_linear()
    activation_func = backend.activation_func()

    mlp = MLPSubmodules(
        linear_fc1=linear_fc1, linear_fc2=linear_fc2, activation_func=activation_func
    )

    expert_module, expert_submodule = backend.grouped_mlp_modules(
        moe_grouped_gemm is not None and moe_grouped_gemm,
        moe_use_legacy_grouped_gemm is not None and moe_use_legacy_grouped_gemm,
    )
    if expert_submodule is not None:
        expert_submodule.activation_func = activation_func

    experts = ModuleSpec(module=expert_module, submodules=expert_submodule)

    # shared experts spec
    # Do not hardcode gate in params; newer Megatron backends pass it explicitly.
    shared_experts = ModuleSpec(module=SharedExpertMLPCompat, submodules=mlp)

    # MoE module spec
    moe_module_spec = ModuleSpec(
        module=MoELayer, submodules=MoESubmodules(experts=experts, shared_experts=shared_experts)
    )
    return moe_module_spec
