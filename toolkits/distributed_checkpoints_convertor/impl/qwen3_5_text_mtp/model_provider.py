# Copyright (c) 2026 Alibaba PAI and Nvidia Megatron-LM Team.
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
from functools import partial

import torch
import torch._dynamo

from model_provider import model_provider as mcore_model_provider  # Megatron-LM-250908/model_provider.py

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.models.mamba import MambaModel
from megatron.core.transformer.multi_token_prediction import MultiTokenPredictionBlock
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.training import print_rank_0
from megatron.training.arguments import core_transformer_config_from_args

from megatron_patch.model.qwen3_next.layer_specs import get_qwen3_next_layer_spec
from megatron_patch.model.qwen3_next.transformer_config import Qwen3NextTransformerConfig

torch._dynamo.config.suppress_errors = True


def _use_transformer_engine(args) -> bool:
    return (
        getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine"
        and torch.cuda.is_available()
    )


def _get_mtp_transformer_layer_spec(args):
    stack_spec = get_qwen3_next_layer_spec(args)
    attention_layer = stack_spec.submodules.attention_layer
    mlp_layer = stack_spec.submodules.mlp_layer
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            self_attention=attention_layer.submodules.self_attention,
            self_attn_bda=attention_layer.submodules.self_attn_bda,
            pre_mlp_layernorm=mlp_layer.submodules.pre_mlp_layernorm,
            mlp=mlp_layer.submodules.mlp,
            mlp_bda=mlp_layer.submodules.mlp_bda,
        ),
    )


def _attach_mtp_if_needed(model, args, config, vp_stage=None):
    if args.mtp_num_layers is None:
        return
    if args.mtp_num_layers != 1:
        raise ValueError(
            "qwen3_5_text_mtp model provider only supports --mtp-num-layers 1. "
            f"Found {args.mtp_num_layers}."
        )

    mtp_transformer_layer_spec = _get_mtp_transformer_layer_spec(args)
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config,
        spec=mtp_transformer_layer_spec,
        use_transformer_engine=_use_transformer_engine(args),
        vp_stage=vp_stage,
    )

    if mtp_block_spec is None:
        return

    model.mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec, vp_stage=vp_stage)
    model.mtp_process = True


def _force_linear_attn_norm_fp32_for_m2h(model, args):
    """Keep linear-attention RMSNorm weights in FP32 for exact MCore->HF export."""
    if not getattr(args, "mcore2hf", False):
        return
    if not (getattr(args, "bf16", False) or getattr(args, "fp16", False)):
        return

    converted = 0
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        return

    for layer in layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            continue
        norm = getattr(mixer, "norm", None)
        weight = getattr(norm, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and weight.dtype != torch.float32:
            weight.data = weight.data.float()
            converted += 1

    if converted > 0:
        print_rank_0(
            f"Forced {converted} linear-attn norm weights to fp32 for exact MCore->HF export."
        )


def mamba_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
    **kwargs,
):
    print_rank_0("building Qwen3.5 MAMBA model ...")
    if config is None:
        config = core_transformer_config_from_args(args, Qwen3NextTransformerConfig)
    assert args.use_legacy_models is False, "Mamba only supported in MCore."

    model = MambaModel(
        config=config,
        mamba_stack_spec=get_qwen3_next_layer_spec(args),
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        hybrid_attention_ratio=args.hybrid_attention_ratio,
        hybrid_mlp_ratio=args.hybrid_mlp_ratio,
        hybrid_override_pattern=args.hybrid_override_pattern,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        pg_collection=pg_collection,
    )

    _force_linear_attn_norm_fp32_for_m2h(model=model, args=args)
    _attach_mtp_if_needed(model=model, args=args, config=config, vp_stage=vp_stage)
    return model


model_provider = partial(mcore_model_provider, mamba_builder)
