"""Dataset utilities for JSON files containing sparse teacher logits payloads."""

from __future__ import annotations

import json
import os
from math import gcd
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from megatron.core import parallel_state
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.megatron_dataset import LowLevelDataset, MegatronDataset
from megatron.core.datasets.utils import Split


_LABEL_PAD_ID = -100
DEBUG = True


def _lcm(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return max(a, b)
    return abs(a * b) // gcd(a, b)


def _to_sequence(value: Any) -> List[Any]:
    """Normalize a JSON field into a Python list."""
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().view(-1).tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _pad_or_trim_tensor(
    data: Sequence[Any],
    length: int,
    pad_value: Any,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(data, dtype=dtype)
    if tensor.numel() < length:
        pad_size = length - tensor.numel()
        pad_tensor = torch.full((pad_size,), pad_value, dtype=dtype)
        tensor = torch.cat((tensor, pad_tensor), dim=0)
    elif tensor.numel() > length:
        tensor = tensor[:length]
    return tensor.contiguous()


def _normalize_teacher_payload(
    payload: Optional[Dict[str, Any]],
    seq_length: int,
) -> Optional[Dict[str, List[List[float]]]]:
    if payload is None:
        return None

    positions = _to_sequence(payload.get("positions"))
    indices = payload.get("indices", [])
    values = payload.get("values", [])

    if not positions:
        return None

    if len(indices) != len(values) or len(indices) != len(positions):
        raise ValueError(
            "teacher_logits must provide matching lengths for 'positions', 'indices', and 'values'"
        )

    normalized_positions: List[int] = []
    normalized_indices: List[List[int]] = []
    normalized_values: List[List[float]] = []
    dropped = 0

    for position, idx_list, val_list in zip(positions, indices, values):
        pos_int = int(position)
        if pos_int < 0 or pos_int >= seq_length:
            dropped += 1
            continue
        idxs = [int(idx) for idx in _to_sequence(idx_list)]
        vals = [float(val) for val in _to_sequence(val_list)]
        if len(idxs) != len(vals):
            raise ValueError("teacher logits indices/values length mismatch per position")
        normalized_positions.append(pos_int)
        normalized_indices.append(idxs)
        normalized_values.append(vals)
    
    if dropped and DEBUG:
        print(f"Dropped {dropped} teacher logits positions outside valid range [0, {seq_length})")

    if not normalized_positions:
        return None

    return {
        "positions": normalized_positions,
        "indices": normalized_indices,
        "values": normalized_values,
    }


def _convert_teacher_payload_to_tensors(
    payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Convert normalized teacher payload lists into torch tensors on CPU."""
    positions = torch.as_tensor(payload["positions"], dtype=torch.long).contiguous()

    indices_tensors = [
        torch.as_tensor(entry, dtype=torch.long).view(-1).contiguous()
        for entry in payload["indices"]
    ]
    values_tensors = [
        torch.as_tensor(entry, dtype=torch.float32).view(-1).contiguous()
        for entry in payload["values"]
    ]

    if not indices_tensors:
        row_ptr = torch.zeros(1, dtype=torch.long)
        flat_indices = torch.empty(0, dtype=torch.long)
        flat_values = torch.empty(0, dtype=torch.float32)
    else:
        lengths = torch.as_tensor(
            [tensor.numel() for tensor in indices_tensors], dtype=torch.long
        )
        row_ptr = torch.zeros(len(indices_tensors) + 1, dtype=torch.long)
        row_ptr[1:] = torch.cumsum(lengths, dim=0)
        flat_indices = torch.cat(indices_tensors, dim=0)
        flat_values = torch.cat(values_tensors, dim=0)

    return {
        "positions": positions,
        "row_ptr": row_ptr,
        "indices": flat_indices,
        "values": flat_values,
    }

class JsonTeacherLowLevelDataset:
    """Minimal iterable over JSON teacher records."""

    def __init__(self, dataset_path: str) -> None:
        if not os.path.isdir(dataset_path):
            raise ValueError(f"JSON teacher data directory '{dataset_path}' does not exist")

        file_paths = [
            os.path.join(dataset_path, entry)
            for entry in os.listdir(dataset_path)
            if entry.endswith(".json")
        ]

        if not file_paths:
            raise ValueError(f"No JSON files found in directory '{dataset_path}'")

        self.file_paths = file_paths

    def __len__(self) -> int:
        print(f"Number of JSON files found: {len(self.file_paths)}")
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_path = self.file_paths[idx]
        with open(file_path, "r", encoding="utf-8") as reader:
            record = json.load(reader)
        return record


class JsonTeacherDataset(MegatronDataset):
    """Dataset that reads one JSON file per training example."""

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: np.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: GPTDatasetConfig,
        *,
        label_pad_id: int = _LABEL_PAD_ID,
    ) -> None:
        super().__init__(dataset, dataset_path, indices, num_samples, index_split, config)

        if self.config.tokenizer is None:
            raise ValueError("JsonTeacherDataset requires a tokenizer in the config")

        self.seq_length = self.config.sequence_length
        self.label_pad_id = label_pad_id
        self.create_attention_mask = self.config.create_attention_mask
        self.use_variable_seq_len = getattr(self.config, "variable_seq_lengths", False)

        tokenizer = self.config.tokenizer

        pad_token = getattr(tokenizer, "pad", None)
        eod_token = getattr(tokenizer, "eod", None)

        if pad_token is None and eod_token is None:
            raise ValueError("Tokenizer must define either 'pad' or 'eod' token id")

        if pad_token is None:
            pad_token = eod_token
        if eod_token is None:
            eod_token = pad_token

        self.pad_token_id = int(pad_token)
        self.eod_token_id = int(eod_token)

        if self.create_attention_mask and not self.use_variable_seq_len:
            mask = torch.tril(torch.ones(self.seq_length, self.seq_length, dtype=torch.float32))
            mask = (mask < 0.5).unsqueeze(0)  # [1, S, S] causal mask (True denotes masked positions)
            self._attention_mask_template = mask
        else:
            self._attention_mask_template = None

        if not self.use_variable_seq_len:
            self._position_ids = torch.arange(self.seq_length, dtype=torch.long)
        else:
            self._position_ids = None

        self._cached_required_multiple: Optional[int] = None
        self.collate_fn = JsonTeacherCollator(
            create_attention_mask=self.create_attention_mask,
            pad_token_id=self.pad_token_id,
            label_pad_id=self.label_pad_id,
            use_variable_seq_len=self.use_variable_seq_len,
        )

        if len(self.indices) == 0:
            self.num_samples = 0

    def __len__(self) -> int:
        if self.num_samples is not None:
            return self.num_samples
        return len(self.indices)

    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        return len(low_level_dataset)

    @staticmethod
    def build_low_level_dataset(dataset_path: str, config: GPTDatasetConfig) -> LowLevelDataset:
        return JsonTeacherLowLevelDataset(dataset_path)

    def _required_sequence_multiple(self) -> int:
        if self._cached_required_multiple is not None:
            return self._cached_required_multiple

        required_multiple = 1

        try:
            cp_size = parallel_state.get_context_parallel_world_size()
        except (RuntimeError, ValueError, AttributeError, AssertionError):
            cp_size = 1
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

        self._cached_required_multiple = required_multiple
        return required_multiple

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        dataset_length = len(self.indices)
        if dataset_length == 0:
            raise IndexError("JsonTeacherDataset has no indices to sample")

        actual_index = int(self.indices[idx % dataset_length])
        record = self.dataset[actual_index]

        file_path = None
        if hasattr(self.dataset, "file_paths"):
            file_path = self.dataset.file_paths[actual_index]  # type: ignore[attr-defined]

        if "input_ids" not in record or "labels" not in record:
            location = f" '{file_path}'" if file_path else ""
            raise KeyError(
                f"Record{location} must contain 'input_ids' and 'labels' fields"
            )

        tokens_list = [int(x) for x in _to_sequence(record["input_ids"])]
        labels_list = [int(x) for x in _to_sequence(record["labels"])]

        original_seq_len = len(tokens_list)

        if len(tokens_list) != len(labels_list):
            raise ValueError(
                f"Record{f' {file_path}' if file_path else ''} contains mismatched "
                f"'input_ids' ({len(tokens_list)}) and 'labels' ({len(labels_list)}) lengths"
            )

        max_seq_len = self.seq_length
        if max_seq_len <= 0:
            raise ValueError("sequence_length must be a positive integer")

        # Trim to the configured maximum length before padding and adding EOS.
        if len(tokens_list) > max_seq_len - 1:
            tokens_list = tokens_list[:max_seq_len-1]
            labels_list = labels_list[:max_seq_len-1]

        active_len = len(tokens_list) + 1 # account for EOS added before shifting
        if active_len == 0:
            raise ValueError("Encountered empty record after trimming tokens to sequence length")

        padding_len = 0
        if self.use_variable_seq_len:
            required_multiple = self._required_sequence_multiple()
            final_len = active_len
            if required_multiple > 1:
                remainder = final_len % required_multiple
                if remainder != 0:
                    padding_len = required_multiple - remainder
                    final_len += padding_len
            # Ensure final length does not exceed max_seq_len by trimming tokens if needed.
            while final_len > max_seq_len and active_len > 0:
                active_len -= 1
                tokens_list = tokens_list[:active_len]
                labels_list = labels_list[:active_len]
                padding_len = 0
                final_len = active_len
                if required_multiple > 1 and active_len > 0:
                    remainder = final_len % required_multiple
                    if remainder != 0:
                        padding_len = required_multiple - remainder
                        final_len += padding_len
            if final_len > max_seq_len:
                raise ValueError(
                    "Unable to satisfy tensor/context parallel padding within max_seq_len"
                )
            if active_len == 0:
                raise ValueError(
                    "Unable to construct non-empty sample within sequence length constraints"
                )
        else:
            padding_len = max_seq_len - active_len
            final_len = max_seq_len
            if padding_len < 0:
                raise ValueError("Sample longer than configured sequence length after truncation")

        tokens_padded = tokens_list + [self.eod_token_id] + [self.pad_token_id] * (padding_len + 1)
        tokens = torch.tensor(tokens_padded, dtype=torch.long)
        tokens = tokens[:-1]

        labels_base = labels_list + [self.eod_token_id] + [self.label_pad_id] * (padding_len + 1)
        labels = torch.tensor(labels_base, dtype=torch.long)
        labels = labels[1:]

        if "loss_mask" in record:
            loss_mask = _pad_or_trim_tensor(
                record["loss_mask"],
                final_len,
                0.0,
                dtype=torch.float32,
            )
        else:
            loss_mask = (labels != self.label_pad_id).to(torch.float32)

        if self.create_attention_mask:
            if self._attention_mask_template is not None:
                attention_mask = self._attention_mask_template.clone()
            else:
                mask = torch.triu(torch.ones((final_len, final_len), dtype=torch.bool), diagonal=1)
                attention_mask = mask.unsqueeze(0)
        else:
            attention_mask = None

        if self._position_ids is not None:
            position_ids = self._position_ids.clone()
        else:
            position_ids = torch.arange(final_len, dtype=torch.long)

        teacher_raw = record.get("teacher_logits") or record.get("teacher_data")
        teacher_data = _normalize_teacher_payload(teacher_raw, final_len)
        if teacher_data is not None:
            teacher_data = _convert_teacher_payload_to_tensors(teacher_data)

        tokens = tokens.contiguous()
        labels = labels.contiguous()
        loss_mask = loss_mask.contiguous()
        position_ids = position_ids.contiguous()
        if attention_mask is not None:
            attention_mask = attention_mask.contiguous()

        if DEBUG:
            try:
                curr_rank = torch.distributed.get_rank()
            except (RuntimeError, ValueError, AttributeError):
                curr_rank = 0

            S = labels.size(0)
            trainable = int(loss_mask.sum().item())

            pad_tokens = int((tokens == self.pad_token_id).sum().item())
            nonpad_length = S - pad_tokens
            pad_ratio = pad_tokens / S if S > 0 else 0.0

            print(
                f"[Rank {curr_rank}][DATA_DEBUG] "
                f"Index {idx} | Sample Len={S} | Nonpad Length={nonpad_length} | "
                f"Original Seq len={original_seq_len} | Active Len={active_len} | "
                f"Padding={padding_len} | trainable_tokens={trainable} ({(trainable / S):.2%}) "
                f"| pad_ratio={pad_ratio:.2%}"
            )

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "teacher_data": teacher_data,
        }


class JsonTeacherCollator:
    """Collate function that stacks tensors and keeps teacher payload per sample."""

    def __init__(
        self,
        create_attention_mask: bool,
        pad_token_id: int,
        label_pad_id: int,
        use_variable_seq_len: bool,
    ) -> None:
        self.create_attention_mask = create_attention_mask
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id
        self.use_variable_seq_len = use_variable_seq_len

    @staticmethod
    def _pad_1d(tensor: torch.Tensor, target_len: int, pad_value: int | float) -> torch.Tensor:
        if tensor.size(0) == target_len:
            return tensor
        pad_shape = (target_len - tensor.size(0),)
        pad_tensor = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, pad_tensor], dim=0)

    def __call__(self, samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        sample_list = list(samples)
        if not sample_list:
            raise ValueError("JsonTeacherCollator received an empty batch")

        if not self.use_variable_seq_len:
            tokens = torch.stack([sample["tokens"] for sample in sample_list])
            labels = torch.stack([sample["labels"] for sample in sample_list])
            loss_mask = torch.stack([sample["loss_mask"] for sample in sample_list])
            position_ids = torch.stack([sample["position_ids"] for sample in sample_list])

            if self.create_attention_mask:
                attention = torch.stack([
                    sample["attention_mask"] for sample in sample_list
                ])
            else:
                attention = None
        else:
            seq_lengths = [sample["tokens"].size(0) for sample in sample_list]
            target_len = max(seq_lengths)

            tokens = torch.stack([
                self._pad_1d(sample["tokens"], target_len, self.pad_token_id)
                for sample in sample_list
            ])
            labels = torch.stack([
                self._pad_1d(sample["labels"], target_len, self.label_pad_id)
                for sample in sample_list
            ])
            loss_mask = torch.stack([
                self._pad_1d(sample["loss_mask"], target_len, 0.0)
                for sample in sample_list
            ])
            position_ids = torch.stack([
                self._pad_1d(sample["position_ids"], target_len, 0)
                for sample in sample_list
            ])

            if self.create_attention_mask:
                attention_list = []
                causal_base = torch.triu(
                    torch.ones((target_len, target_len), dtype=torch.bool), diagonal=1
                )
                for seq_len in seq_lengths:
                    mask = causal_base.clone()
                    if seq_len < target_len:
                        mask[seq_len:, :] = True
                    attention_list.append(mask.unsqueeze(0))
                attention = torch.stack(attention_list)
            else:
                attention = None

        teacher_list = [sample["teacher_data"] for sample in sample_list]

        return {
            "tokens": tokens,
            "labels": labels,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
            "attention_mask": attention,
            "teacher_data": teacher_list,
        }
