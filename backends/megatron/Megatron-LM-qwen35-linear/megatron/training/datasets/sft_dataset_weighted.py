# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Dict, Optional
from math import gcd
import numpy as np
import torch

from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split
from megatron.training.datasets.sft_dataset import SFTDataset, SFTLowLevelDataset, IGNORE_INDEX

from megatron.core import parallel_state

DEBUG = True

class SFTDatasetWeightedMask(SFTDataset):
    """The dataset used during SFT"""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

    def _process_example(self, conversation_dict: Dict[str, Any]):        
        if not isinstance(conversation_dict, dict):
            raise ValueError(f"The sample must be a dict but got {type(conversation_dict)}")

        input_ids = conversation_dict.get("input_ids", [])
        labels = conversation_dict.get("labels", [])
        loss_mask_weighted = conversation_dict.get("loss_mask", [])

        assert len(input_ids) == len(labels) == len(loss_mask_weighted)
    
        return input_ids, labels, loss_mask_weighted

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        tokenizer = self.config.tokenizer
        max_seq_len = self.config.sequence_length

        conversation_list = self.dataset[int(self.indices[idx % len(self.indices)])]
        
        tokens, target, loss_mask_weighted = self._process_example(conversation_list)
        target = target[1:] + [IGNORE_INDEX]
        loss_mask_weighted = loss_mask_weighted[1:] + [0]

        original_seq_len = len(tokens)

        if original_seq_len > max_seq_len:
            if True:  # TODO: when too long to fit in context, truncate left to right
                tokens = tokens[: max_seq_len]
                target = target[: max_seq_len]
                loss_mask_weighted = loss_mask_weighted[: max_seq_len]
            else:  # right to left
                tokens = tokens[-(max_seq_len - 1) :]
                target = target[-(max_seq_len - 1) :]
                loss_mask_weighted = loss_mask_weighted[-(max_seq_len - 1) :]

        use_variable_seq_len = getattr(self.config, "variable_seq_lengths", False)

        if use_variable_seq_len:
            def _lcm(a: int, b: int) -> int:
                if a == 0 or b == 0:
                    return max(a, b)
                return abs(a * b) // gcd(a, b)

            base_seq_len = len(tokens) 
            required_multiple = 1

            cp_size = parallel_state.get_context_parallel_world_size()
            if cp_size > 1:
                required_multiple = _lcm(required_multiple, 2 * cp_size)

            tp_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                try:
                    tp_size = parallel_state.get_tensor_model_parallel_world_size()
                except AssertionError:
                    tp_size = 1
            if tp_size > 1:
                required_multiple = _lcm(required_multiple, cp_size * tp_size)

            padding_len = 0
            if required_multiple > 1:
                remainder = base_seq_len % required_multiple
                if remainder != 0:
                    padding_len = required_multiple - remainder

            final_seq_len = base_seq_len + padding_len
            if final_seq_len > max_seq_len:
                # Trim tokens until both constraints fit within max_seq_len.
                while final_seq_len > max_seq_len and tokens:
                    tokens.pop()
                    target.pop()
                    loss_mask_weighted.pop()
                    base_seq_len = len(tokens)
                    padding_len = 0
                    if required_multiple > 1:
                        remainder = base_seq_len % required_multiple
                        if remainder != 0:
                            padding_len = required_multiple - remainder
                    final_seq_len = base_seq_len + padding_len
                if final_seq_len > max_seq_len:
                    raise ValueError(
                        "Unable to satisfy tensor/context parallel padding within max_seq_len"
                    )
        else:
            num_tokens = len(tokens)
            padding_len = max_seq_len - num_tokens
            if padding_len < 0:
                raise ValueError("Sample longer than configured sequence length after truncation")

        tokens = np.array(
            tokens + [tokenizer.pad] * padding_len,
            dtype=np.int64,
        )
        target = np.array(
            target + [IGNORE_INDEX] * padding_len,
            dtype=np.int64,
        )
        loss_mask_weighted = np.array(
            loss_mask_weighted + [0.0] * padding_len,
            dtype=float,
        )

        tokens = torch.tensor(tokens).contiguous()
        target = torch.tensor(target).contiguous()
        loss_mask_weighted = torch.tensor(loss_mask_weighted).contiguous()

        loss_mask, position_ids, attention_mask = self._get_ltor_masks_and_position_ids(
            max_seq_len, target, tokenizer.pad, use_variable_seq_len
        )

        # Copy weighted mask values into loss_mask (preserve dtype)
        loss_mask = loss_mask_weighted.to(dtype=loss_mask.dtype)

        if DEBUG:
            curr_rank = torch.distributed.get_rank()
            S = target.shape[0]
            trainable = int((loss_mask > 0).sum().item())
            weighted_trainable = float(loss_mask.sum().item())

            # Prefer deriving padding from the attention mask (robust when pad token == eos, or packed data)
            pad_tokens = 0
            nonpad_lengths = [S]  # default: assume fully non-padded

            if attention_mask is not None and attention_mask.dim() == 4:
                # attention_mask: [B, 1, S, S]; a "valid" token row typically has any True in its row
                vis = attention_mask.squeeze(1).any(dim=-1)  # [B, S] boolean
                nonpad_lengths = vis.sum(dim=0).tolist()     # per-sample non-padded token counts
                pad_tokens = int((S - vis.sum(dim=0)).sum().item())
            else:
                # Fallback: count pad tokens by id (only if pad_token_id is defined)
                pad_id = tokenizer.pad
                pad_mask = (tokens == pad_id)
                nonpad_lengths = (S - pad_mask.sum(dim=0)).tolist()
                pad_tokens = int(pad_mask.sum().item())

            pad_ratio = pad_tokens / S

            print(
                f"[Rank {curr_rank}][DATA_DEBUG] "
                f"Index {idx} | Sample Len={S} | Nonpad Lengths={nonpad_lengths} | "
                f"Original Seq len={original_seq_len} | Truncated S={S} | trainable_tokens={trainable} ({trainable/S:.2%}) "
                f"| weighted trainable={weighted_trainable} ({weighted_trainable/S:.2%})| nonpad_len={nonpad_lengths} | pad_ratio={pad_ratio:.2%}"
            )

        if self.config.create_attention_mask:
            ret = {
                'tokens': tokens,
                'labels': target,
                'attention_mask': attention_mask,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }
        else:
            ret = {
                'tokens': tokens,
                'labels': target,
                'loss_mask': loss_mask,
                'position_ids': position_ids,
            }

        return ret
