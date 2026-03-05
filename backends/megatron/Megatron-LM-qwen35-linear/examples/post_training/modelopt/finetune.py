# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.

"""Supervised Finetuning GPT."""
import contextlib
import itertools
import os
import sys
import time
from functools import partial
from typing import Any, Dict, Optional

import jsonlines

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import datasets
import torch
import transformers

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.post_training.arguments import add_modelopt_args
from megatron.post_training.model_provider import model_provider
from megatron.post_training.non_loss_data_func import report_draft_acceptance_length
from megatron.training import get_args, get_timers, get_tokenizer, pretrain
from megatron.training.utils import (
    average_losses_across_data_parallel_group,
    get_batch_on_this_cp_rank,
    get_ltor_masks_and_position_ids,
    print_rank_0,
    unwrap_model,
)

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)

# Memory profiling flags (from pretrain_gpt.py)
PHASE_LOGGER = False
DEBUG = False

# Memory profiling helper functions (from pretrain_gpt.py)
def _is_rank0():
    """Check if we're on rank 0."""
    return torch.distributed.is_initialized() and torch.distributed.get_rank() == 0 or \
           not torch.distributed.is_initialized()


def _bytes(n: int) -> str:
    """Convert bytes to a human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}PB"


def _barrier():
    """Synchronize all ranks."""
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _mem_stats(device=None):
    """Get CUDA memory statistics."""
    if device is None:
        device = torch.cuda.current_device()
    torch.cuda.synchronize(device)
    alloc    = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_a   = torch.cuda.max_memory_allocated(device)
    peak_r   = torch.cuda.max_memory_reserved(device)
    return {
        "allocated": _bytes(alloc),
        "reserved":  _bytes(reserved),
        "peak_allocated": _bytes(peak_a),
        "peak_reserved":  _bytes(peak_r),
    }


@contextlib.contextmanager
def mem_phase(name: str, do_barrier: bool = False):
    """Emit per-phase CUDA memory stats (allocated/reserved + peaks)."""
    if not PHASE_LOGGER or not torch.cuda.is_available():
        yield
        return
    device = torch.cuda.current_device()
    if do_barrier:
        _barrier()
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    try:
        yield
    finally:
        torch.cuda.synchronize(device)
        dt = time.time() - t0
        stats = _mem_stats(device)
        if _is_rank0():
            print(
                f"[MEM][{name}] dt={dt:.3f}s | "
                f"alloc={stats['allocated']} res={stats['reserved']} "
                f"peak_alloc={stats['peak_allocated']} peak_res={stats['peak_reserved']}",
                flush=True,
            )


def get_eos_id():
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer

    if hf_tokenizer.eos_token == "<|eot_id|>":
        return 128001
    if hf_tokenizer.eos_token == "<|eot|>":
        return 200001
    if hf_tokenizer.eos_token == "<|im_end|>":
        return 151643

    return hf_tokenizer.eos_token_id

def get_pad_id():
    tokenizer = get_tokenizer()
    hf_tokenizer = tokenizer._tokenizer
    return hf_tokenizer.pad_token_id


class SFTDataset(torch.utils.data.Dataset):

    hf_dataset_to_kwargs = {
        "Open-Orca/OpenOrca": {"split": "train"},
        "Open-Orca/SlimOrca": {"split": "train"},
        "nvidia/Daring-Anteater": {"split": "train"},
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": {"split": "train"},
        "HuggingFaceH4/ultrachat_200k": {"split": "train_sft"},
    }

    hf_dataset_to_conversation = {
        "Open-Orca/OpenOrca": lambda data: SFTDataset._to_conversation(
            data["question"], data["response"]
        ),
        "Open-Orca/SlimOrca": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "nvidia/Daring-Anteater": lambda data: SFTDataset._sharegpt_to_openai_conversations(data),
        "Magpie-Align/Magpie-Llama-3.1-Pro-MT-300K-Filtered": lambda data: SFTDataset._sharegpt_to_openai_conversations(
            data
        ),
    }

    hf_dataset_to_prompt_template = {
        "Open-Orca/OpenOrca": "{{ messages['question'] + ' ' + messages['response'] + ' ' }}",
    }

    def __init__(
        self,
        num_packed_samples: int,
        data_path: Optional[str],
        tokenizer: transformers.PreTrainedTokenizerBase,
        seq_length: int,
        hf_dataset: Optional[str] = None,
        num_shards: int = 1,
        shard_index: int = 0,
    ):
        """A simple dataset implementation for supervised fine-tuning.

        The raw data is processed and packed to an indexed dataset on the fly. Users
        specify the total number of packed samples and the dataloader (or sampler)
        access the packed dataset by indices. When the packed dataset length is smaller
        than the index, the packing process fetches the raw data in a cyclic fashion
        until the packed dataset has sufficient length.

        Args:
            data_path: Path to the json or jsonl file
            num_packed_samples: total number of packed samples (cyclic access)
            tokenizer: hf tokenizer
            seq_length: max sequence length
            hf_dataset: not supported yet
        """
        if not isinstance(tokenizer, transformers.PreTrainedTokenizerBase):
            raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

        self.num_packed_samples = num_packed_samples
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.hf_dataset = hf_dataset
        self.data_transformation = lambda data: data
        self.num_shards = num_shards
        self.shard_index = shard_index
        self.indexed_dataset = []
        self._raw_sample_index = 0

        # [WAR]: For DeepSeek-V3/R1 tokenizer, we modify the chat_template such that the <think>
        # tokens are preserved for supervised learning.
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace(
            REMOVE_THINK_CHAT_TEMPLATE, ""
        )

        if data_path is not None:
            if data_path.endswith(".json"):
                self._raw_samples = json.load(open(data_path))
            elif data_path.endswith(".jsonl"):
                with jsonlines.open(data_path, mode='r') as reader:
                    self._raw_samples = [obj for obj in reader]
            else:
                raise ValueError("data_path must be json or jsonl")
            print (f"Number of raw samples: {len(self._raw_samples)}")
            if DEBUG:
                print (f"Raw samples: {self._raw_samples[0]["conversations"][0]}")
        elif self.hf_dataset is not None:
            hf_dataset_kwargs = SFTDataset.hf_dataset_to_kwargs.get(
                self.hf_dataset, {"split": "train"}
            )
            self._raw_samples = datasets.load_dataset(self.hf_dataset, **hf_dataset_kwargs)
            self._raw_samples = self._raw_samples.shard(
                num_shards=self.num_shards, index=shard_index
            )

            print(
                "Rank {:3}/{:3} creates SFT data shard {:3}/{:3} with {:10} raw samples".format(
                    torch.distributed.get_rank(),
                    torch.distributed.get_world_size(),
                    self.shard_index,
                    self.num_shards,
                    len(self._raw_samples),
                ),
                flush=True,
            )

        else:
            raise ValueError("Either hf_dataset or data_path must be provided!")

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = SFTDataset.hf_dataset_to_prompt_template
        elif self.hf_dataset is not None:
            self.data_transformation = SFTDataset.hf_dataset_to_conversation.get(
                self.hf_dataset, lambda data: data
            )

        if self.tokenizer.chat_template is None:
            raise ValueError("No valid chat template!")

    def __len__(self):
        return self.num_packed_samples

    def __getitem__(self, idx):
        """Get the idx packed data.

        The packed data index is different from the raw data index where a packed sample
        of sequence-length may require concatenting multiple raw data. When all raw data
        are used up, the last packed data is throw away, and we have a packed dataset
        in memory. The packed data index may exceed the length of the packed dataset
        which will just wrap in a cyclic fashion.
        """
        idx = idx // self.num_shards

        while idx >= len(self.indexed_dataset):
            packed_samples = self._process_and_pack_example()
            if packed_samples is None:
                break
            else:
                self.indexed_dataset.append(packed_samples)
            if len(self.indexed_dataset) % 10000 == 0:
                print(
                    "Rank {:3}/{:3} requests {:10}/{:10} packed SFT sample".format(
                        torch.distributed.get_rank(),
                        torch.distributed.get_world_size(),
                        idx,
                        len(self.indexed_dataset),
                    ),
                    flush=True,
                )

        idx = idx % len(self.indexed_dataset)
        torch_sample = {}
        for key, val in self.indexed_dataset[idx].items():
            torch_sample[key] = torch.LongTensor(val)
        return torch_sample

    def _process_and_pack_example(self):
        """Process multiple raw data and pack them into fixed sequence length."""
        required_packed_tokens = self.seq_length + 1
        current_packed_samples = []
        current_packed_samples_token_count = 0

        while current_packed_samples_token_count < required_packed_tokens:
            if self._raw_sample_index >= len(self._raw_samples):
                return None
            raw_sample = self._raw_samples[self._raw_sample_index]
            self._raw_sample_index += 1
            processed_sample = self._process_example(raw_sample)
            if processed_sample is not None:
                current_packed_samples.append(processed_sample)
                current_packed_samples_token_count += processed_sample["token_count"]

        packed_samples = {}

        for key in ['input_ids', 'loss_mask']:
            packed_samples[key] = list(
                itertools.chain.from_iterable([obj[key] for obj in current_packed_samples])
            )

        for key in ['token_count']:
            packed_samples[key] = [obj[key] for obj in current_packed_samples]

        return packed_samples

    def _process_example(self, example: Dict[str, Any]):        
        if not isinstance(example, Dict):
            raise ValueError(f"The sample must be a Dict but got {type(example)}")

        example = self.data_transformation(example)

        conversations = example.get("conversations", None) or example.get("messages", None)
        if conversations is not None:
            msgs = conversations
            if len(msgs) < 2 or msgs[0]["role"] == "assistant":
                return None

            input_ids = []
            loss_mask = []

            # Tokenize message-by-message using the chat template so formatting stays consistent
            for i, m in enumerate(msgs):
                seg_ids = self.tokenizer.apply_chat_template(
                    [m], tokenize=True, add_generation_prompt=False
                )
                input_ids.extend(seg_ids)
                if m["role"] == "assistant":
                    loss_mask.extend([1] * len(seg_ids))
                else:
                    loss_mask.extend([0] * len(seg_ids))
        else:
            # Fallback: non-chat data → old behavior (train on all tokens)
            input_ids = self.tokenizer.apply_chat_template(example)
            loss_mask = [1] * len(input_ids)

        # Always add EOS between samples, but don’t train on it
        input_ids = input_ids + [get_eos_id()]
        loss_mask += [0]

        assert len(input_ids) == len(loss_mask)

        # Truncate to seq_length
        if len(input_ids) > self.seq_length:
            input_ids = input_ids[: self.seq_length]
            loss_mask = loss_mask[: self.seq_length]

        return {
            "input_ids": input_ids,
            "loss_mask":  loss_mask,
            "token_count": len(input_ids),
        }

    @classmethod
    def _to_conversation(cls, question, response):
        msg_question = {"role": "user", "content": question}
        msg_response = {"role": "assistant", "content": response}
        return {"conversations": [msg_question, msg_response]}

    @classmethod
    def _sharegpt_to_openai_conversations(cls, data):
        role_mapping = {
            "user": "user",
            "User": "user",
            "human": "user",
            "assistant": "assistant",
            "Assistant": "assistant",
            "gpt": "assistant",
            "system": "system",
            "System": "system",
        }
        processed_data = {"conversations": []}
        for msg in data["conversations"]:
            role = role_mapping[msg["from"]]
            content = msg["value"]
            processed_data["conversations"].append({"role": role, "content": content})
        return processed_data

    @classmethod
    def _special_to_openai_conversations(cls, data):
        processed_data = {"conversations": data["input"]["messages"]}
        return processed_data


def train_valid_test_sft_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples
            in train test and validation.
    """
    print_rank_0("> building train, validation, and test SFT datasets ...")
    args = get_args()
    tokenizer = get_tokenizer()

    if not isinstance(tokenizer._tokenizer, transformers.PreTrainedTokenizerBase):
        raise ValueError("SFTDataset only supports transformers.PreTrainedTokenizerBase!")

    if args.micro_batch_size > 1:
        raise ValueError("SFTDataloader only supports micro_batch_size=1.")

    kwargs = {
        "tokenizer": tokenizer._tokenizer,
        "seq_length": args.seq_length,
        # Optional kwargs
        "hf_dataset": args.finetune_hf_dataset,
        "num_shards": mpu.get_expert_data_parallel_world_size(),
        "shard_index": mpu.get_expert_data_parallel_rank(),
    }

    print ("Train data path: ", args.train_data_path)
    print ("Valid data path: ", args.valid_data_path)
    print ("Test data path: ", args.test_data_path)

    data_path = [
        args.train_data_path[0] if args.train_data_path else None,
        args.valid_data_path[0] if args.valid_data_path else None,
        args.test_data_path[0] if args.test_data_path else None,
    ]

    train_ds = SFTDataset(train_val_test_num_samples[0], data_path[0], **kwargs)
    valid_ds = SFTDataset(train_val_test_num_samples[1], data_path[1], **kwargs)
    test_ds = SFTDataset(train_val_test_num_samples[2], data_path[2], **kwargs)

    print_rank_0("> finished creating SFT datasets ...")

    return train_ds, valid_ds, test_ds


def get_batch(data_iterator):
    """Generate a batch."""
    # TODO: this is pretty hacky, find a better way
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    args = get_args()

    # Items and their type.
    keys = ["input_ids", "loss_mask"]
    datatype = torch.int64

    # Broadcast data since only TP rank-0 has the data_iterator.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack the data received.
    tokens_ = data_b["input_ids"]
    tokens = tokens_[:, 0 : 0 + args.seq_length].contiguous()
    labels = tokens_[:, 1 : 1 + args.seq_length].contiguous()
    answer_only_loss_mask = data_b["loss_mask"][:, 1 : 1 + args.seq_length].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens, get_eos_id(), get_pad_id(), args.reset_position_ids, args.reset_attention_mask, args.eod_mask_loss, True
    )
    loss_mask = loss_mask * answer_only_loss_mask.to(dtype=loss_mask.dtype)


    labels = labels.contiguous()
    loss_mask = loss_mask.contiguous()
    
    # Set labels to -100 where loss_mask is 0 (for ignore_index in CCE)
    labels = labels.clone()  # Clone to avoid modifying original tensor
    labels[loss_mask == 0] = -100

    batch = {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


def _mask_loss(output_tensor, loss_mask, mp_reduce=False):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()

    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()

    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    if mp_reduce and args.tensor_model_parallel_size > 1:
        # KD loss requires extra all-reduce to ensure same values across MP-TP partitions.
        loss = torch.sum(tensor_parallel.gather_from_tensor_model_parallel_region(loss.reshape(1)))

    return loss


def _allreduce_loss(loss):
    """Reduce loss for reporting purposes."""
    args = get_args()

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss * args.context_parallel_size, averaged_loss[0]


def loss_func(loss_mask: torch.Tensor, model: GPTModel, output_tensor: torch.Tensor):
    """Loss function (with KD Loss support).

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        model (GPTModel): The model (can be wrapped)
        output_tensor (Tensor): The tensor with the losses
    """
    args = get_args()

    # Unwrap for both Distillation and LANA
    model = unwrap_model(model)

    # Standard lm loss
    output_tensor = output_tensor.float()  # cache
    loss_lm = _mask_loss(output_tensor, loss_mask)
    loss_lm, loss_lm_avg = _allreduce_loss(loss_lm)
    loss, report = loss_lm, {'lm loss': loss_lm_avg}

    return loss, report

def _all_params(mod):
    if isinstance(mod, (list, tuple)):
        for m in mod: yield from _all_params(m)
    else:
        yield from mod.parameters()


def non_loss_data_func(model):
    report_draft_acceptance_length(model)

    if not DEBUG:
        return
    # Print every 50 steps
    # from megatron.training import get_args
    # args = get_args()
    # if (args.iteration % 50) != 0:
    #     return

    m = unwrap_model(model)
    params = [p for p in _all_params(m) if p is not None and p.requires_grad]
    if not params:
        return

    dev = params[0].device
    p2 = torch.zeros((), device=dev)
    for p in params:
        p2 += p.data.float().norm(2)**2
    torch.distributed.all_reduce(p2, op=torch.distributed.ReduceOp.SUM, group=mpu.get_data_parallel_group())

    if torch.distributed.get_rank() == 0:
        print(f"[debug] params_norm={p2.sqrt().item():.6f} ")
        

def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator: Input data iterator
        model: The GPT Model
    """
    timers = get_timers()

    # Get the batch.
    timers("batch-generator", log_level=2).start()
    with mem_phase("LOAD_BATCH", do_barrier=True):
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()

    if DEBUG and torch.distributed.get_rank() == 0:
        B, S = labels.shape
        trainable = int(loss_mask.sum().item())

        # Prefer deriving padding from the attention mask (robust when pad token == eos, or packed data)
        pad_tokens = 0
        nonpad_lengths = [S] * B  # default: assume fully non-padded

        if attention_mask is not None and attention_mask.dim() == 4:
            # attention_mask: [B, 1, S, S]; a "valid" token row typically has any True in its row
            vis = attention_mask.squeeze(1).any(dim=-1)  # [B, S] boolean
            nonpad_lengths = vis.sum(dim=1).tolist()     # per-sample non-padded token counts
            pad_tokens = int((S - vis.sum(dim=1)).sum().item())
        else:
            # Fallback: count pad tokens by id (only if pad_token_id is defined)
            pad_id = get_pad_id()
            pad_mask = (tokens == pad_id)
            nonpad_lengths = (S - pad_mask.sum(dim=1)).tolist()
            pad_tokens = int(pad_mask.sum().item())

        pad_ratio = pad_tokens / (B * S)

        print(
            "[debug] "
            f"S={S} | trainable_tokens={trainable} ({trainable/(B*S):.2%}) "
            f"| nonpad_len={nonpad_lengths} | pad_ratio={pad_ratio:.2%}"
        )
        
    # Forward pass with memory profiling
    with mem_phase("FORWARD", do_barrier=True):
        output_tensor = model(tokens, position_ids, attention_mask, labels=labels)

    return output_tensor, partial(loss_func, loss_mask, model)


if __name__ == "__main__":
    pretrain(
        train_valid_test_sft_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        extra_args_provider=add_modelopt_args,
        args_defaults={"tokenizer_type": "HuggingFaceTokenizer"},
        non_loss_data_func=non_loss_data_func,
    )
