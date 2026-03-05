# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""
Thin wrapper that adapts Megatron-LM's tensor-parallel vocab sharding to
apple/ml-cross-entropy's `linear_cross_entropy` API.

Install:
  pip install "cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git"
Docs:
  https://github.com/apple/ml-cross-entropy
"""
from typing import Optional
import torch

from megatron.core.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_group,
)

try:
    # Upstream API
    from cut_cross_entropy import linear_cross_entropy, VocabParallelOptions  # type: ignore
except Exception as e:  # pragma: no cover
    linear_cross_entropy = None
    VocabParallelOptions = None
    _CCE_IMPORT_ERROR = e
else:
    _CCE_IMPORT_ERROR = None


def _maybe_build_vp_opts(vocab_size: int) -> Optional["VocabParallelOptions"]:
    """Create VocabParallelOptions when TP>1; otherwise return None."""
    tp_world = get_tensor_model_parallel_world_size()
    if tp_world <= 1:
        return None
    group = get_tensor_model_parallel_group()
    return VocabParallelOptions.from_vocab(vocab_size, group=group)
    # return VocabParallelOptions(start, end, group=group)


def cce_per_token_loss(
    *,
    embeddings: torch.Tensor,          # [B, T, H] or [T, B, H] or [B, T_local, H] with SP
    classifier_weight: torch.Tensor,   # [V, H] (or [V_local, H] with VP)
    labels: torch.Tensor,              # [B, T_global] (Megatron's labels are global wrt SP)
    vocab_size: int,
    impl: str = "cce",
    reduction: str = "none",
    shift: bool = True,
    ignore_index: int = -100,
    return_lse: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute (optionally shifted) per-token cross-entropy via CCE.
    Returns [B, T-1] if shift=True, else [B, T].
    """
    embeddings = embeddings.transpose(0, 1).contiguous()  # -> [B, S, H]
    if embeddings.size(0) != labels.size(0) or embeddings.size(1) != labels.size(1):
        raise RuntimeError(
            f"CCE shape mismatch after normalization: embeddings[...,:] has {embeddings.size()[:-1]} "
            f"but labels_local have {labels.size()} (expected embeddings.size()[:-1] == labels_local.size())."
        )
    if linear_cross_entropy is None:
        raise ImportError(
            "cut-cross-entropy is not installed but `use_linear_cross_entropy=True`.\n"
            "Install via: pip install \"cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git\"\n"
            f"Original import error: {_CCE_IMPORT_ERROR}"
        )

    # Build VP options (and assert shard size matches)
    vp_opts = _maybe_build_vp_opts(vocab_size)

    # CCE handles half/bfloat16 inputs and promotes to fp32 internally where needed.
    losses = linear_cross_entropy(
        embeddings,                      # [B, T, H]
        classifier_weight,               # [V/TP_SIZE, H]
        labels,                          # [B, T]
        impl=impl,
        reduction=reduction,
        shift=int(shift),
        ignore_index=ignore_index,
        vocab_parallel_options=vp_opts,
        return_lse=return_lse,
        softcap = temperature
    )

    return losses # return losses or (losses, lse)