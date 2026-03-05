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
from gpt_builders import gpt_builder
from megatron.training import print_rank_0

torch._dynamo.config.suppress_errors = True


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
        attn = getattr(layer, "self_attention", None)
        if attn is None:
            attn = getattr(layer, "mixer", None)
        if attn is None:
            continue
        norm = getattr(attn, "out_norm", None)
        if norm is None:
            norm = getattr(attn, "norm", None)
        weight = getattr(norm, "weight", None)
        if isinstance(weight, torch.nn.Parameter) and weight.dtype != torch.float32:
            weight.data = weight.data.float()
            converted += 1

    if converted > 0:
        print_rank_0(
            f"Forced {converted} linear-attn norm weights to fp32 for exact MCore->HF export."
        )


def gpt_linear_builder(
    args,
    pre_process,
    post_process,
    vp_stage=None,
    config=None,
    pg_collection=None,
    **kwargs,
):
    print_rank_0("building Qwen3.5 GPT model with linear attention ...")
    model = gpt_builder(
        args=args,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
        config=config,
        pg_collection=pg_collection,
        **kwargs,
    )
    _force_linear_attn_norm_fp32_for_m2h(model=model, args=args)
    return model


model_provider = partial(mcore_model_provider, gpt_linear_builder)
