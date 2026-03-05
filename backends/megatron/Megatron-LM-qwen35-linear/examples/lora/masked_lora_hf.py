#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
from typing import List, Dict

import torch
import torch.distributed as dist
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# ------------------------------------------------------------
# Env & safety
# ------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from transformers.utils import is_flash_attn_2_available
from flash_attn.losses.cross_entropy import CrossEntropyLoss as FlashCrossEntropyLoss

IGNORE_INDEX = -100


# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--train_file", type=str, required=True)  # jsonl
    p.add_argument("--eval_file", type=str, default=None)    # jsonl
    p.add_argument("--output_dir", type=str, default="qwen3coder30b-lora")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--per_device_train_bs", type=int, default=1)
    p.add_argument("--per_device_eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--no-use_qlora", dest="use_qlora", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--local_files_only", action="store_true", default=False)
    p.add_argument("--use_flash_attn", action="store_true", default=False)
    p.add_argument("--flash_attn_fused_ce", action="store_true", default=True)
    p.add_argument("--no-flash_attn_fused_ce", dest="flash_attn_fused_ce", action="store_false")
    p.add_argument("--left_truncate", action="store_true", default=False,
                   help="If a conversation exceeds max_seq_len, keep the most recent tokens (left-truncate).")
    return p.parse_args()


# ------------------------------------------------------------
# Tokenization helpers (ONE EXAMPLE per conversation)
# ------------------------------------------------------------
def _pick_msgs(ex: Dict) -> List[Dict]:
    if "messages" in ex:
        return ex["messages"]
    if "conversations" in ex:
        return ex["conversations"]
    raise ValueError("Each JSONL record must have 'messages' or 'conversations'.")


def tokenize_one_conversation(conv_msgs: List[Dict], tok, max_len: int, left_truncate: bool = False) -> Dict[str, List[int]]:
    """
    Build a single example from the full conversation:
      - input_ids: concatenated chat-template segments
      - labels: IGNORE_INDEX for non-assistant tokens; token ids for assistant tokens
      - attention_mask: 1 for real tokens
    (We leave padding to the collator to enforce a fixed length across the batch.)
    """
    input_ids: List[int] = []
    labels: List[int] = []

    for m in conv_msgs:
        seg_ids = tok.apply_chat_template([m], tokenize=True, add_generation_prompt=False)
        if not isinstance(seg_ids, list):
            seg_ids = list(seg_ids)

        # If adding this segment would exceed max_len, stop here
        total_len = len(input_ids) + len(seg_ids)
        if total_len > max_len:
            break
            
        input_ids.extend(seg_ids)
        if m.get("role") == "assistant":
            labels.extend(seg_ids)                         # learn on assistant spans
        else:
            labels.extend([IGNORE_INDEX] * len(seg_ids))   # mask user/system

    # Truncate to window (padding happens later in the collator)
    if len(input_ids) > max_len:
        if left_truncate:
            input_ids = input_ids[-max_len:]
            labels    = labels[-max_len:]
        else:
            input_ids = input_ids[:max_len]
            labels    = labels[:max_len]

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def build_single_example_dataset(json_path: str, tok, max_len: int, left_truncate: bool = False):
    """
    Load jsonl -> ONE row per conversation, tokenized & masked, **and padded/truncated
    to EXACTLY `max_len` inside the dataset** (so no custom collator is required).
    """
    raw = load_dataset("json", data_files=json_path, split="train")

    def fix_len(seq, fill):
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [fill] * (max_len - len(seq))

    def _map(ex):
        sample = tokenize_one_conversation(
            _pick_msgs(ex), tok, max_len=max_len, left_truncate=left_truncate
        )
        ids  = sample["input_ids"]
        labs = sample["labels"]
        attn = sample["attention_mask"]

        # Pad/trim to fixed length here
        ids_fixed  = fix_len(ids, tok.pad_token_id)
        labs_fixed = fix_len(labs, IGNORE_INDEX)
        attn_fixed = fix_len(attn, 0)

        return {
            "input_ids": ids_fixed,
            "labels": labs_fixed,
            "attention_mask": attn_fixed,
        }

    ds = raw.map(
        _map,
        remove_columns=raw.column_names,
        desc="Tokenizing + fixing length (assistant-only labels)",
    )
    return ds


# ------------------------------------------------------------
# Trainer subclass to log throughput
# ------------------------------------------------------------
class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accum_total_tokens = 0
        self._accum_labeled_tokens = 0
        self._accum_samples = 0

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        with torch.no_grad():
            attn = inputs.get("attention_mask")
            total_tokens = int(attn.sum().item()) if attn is not None else 0
            labels = inputs.get("labels")
            labeled_tokens = int((labels != IGNORE_INDEX).sum().item()) if labels is not None else 0
            input_ids = inputs.get("input_ids")
            micro_samples = int(input_ids.size(0)) if input_ids is not None else 0

        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._accum_total_tokens += total_tokens
        self._accum_labeled_tokens += labeled_tokens
        self._accum_samples += micro_samples
        return loss


class FlashAttnCETrainer(LoggingSFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  
        # ignore_index honored; inplace_backward saves memory
        self.flash_ce = FlashCrossEntropyLoss(
            ignore_index=-100, reduction="sum", inplace_backward=True
        )

    # IMPORTANT: accept the Trainer's extra kwarg
    def compute_loss(self, model, inputs, return_outputs: bool = False,
                     num_items_in_batch=None, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits                      # [B, T, V]
        # ensure dtype for labels
        if labels.dtype != torch.long:
            labels = labels.to(torch.long)

        B, T, V = logits.shape
        graph_zero = logits.sum() * 0.0              # scalar, requires_grad=True

        # ---- Fused Flash CE path (expects [N,V], [N]) ----
        logits_f = logits.reshape(B * T, V)
        labels_f = labels.reshape(B * T)
        keep = labels_f != IGNORE_INDEX
        if keep.any():
            # boolean indexing creates a copy; make it contiguous for Triton kernel
            loss_sum = self.flash_ce(
                logits_f[keep].contiguous(),
                labels_f[keep].contiguous()
            )                                     # sum over valid tokens
            denom = keep.sum().clamp_min(1)
            loss = loss_sum / denom
        else:
            loss = graph_zero
        return (loss, outputs) if return_outputs else loss

class PerfOnLogCallback(TrainerCallback):
    """
    Emit a second log line with throughput/memory metrics.
    """
    def __init__(self):
        super().__init__()
        self.trainer = None
        self._last_wall = None
        self._accum_tot = 0.0
        self._accum_lab = 0.0
        self._accum_sam = 0.0
        self._emitting = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.trainer = kwargs.get("trainer", getattr(self, "trainer", None))
        self._last_wall = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        tr = self.trainer
        if tr is None:
            return control

        self._accum_tot += float(tr._accum_total_tokens)
        self._accum_lab += float(tr._accum_labeled_tokens)
        self._accum_sam += float(tr._accum_samples)

        tr._accum_total_tokens = 0
        tr._accum_labeled_tokens = 0
        tr._accum_samples = 0
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self._emitting:
            return control
        tr = kwargs.get("trainer", self.trainer)
        if tr is None:
            return control

        now = time.perf_counter()
        elapsed = (now - self._last_wall) if self._last_wall else 0.0
        self._last_wall = now

        dev = tr.model.device
        def red(x: float):
            t = torch.tensor([float(x)], device=dev)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        tot = red(self._accum_tot);  lab = red(self._accum_lab);  sam = red(self._accum_sam)
        self._accum_tot = self._accum_lab = self._accum_sam = 0.0

        tps = (tot / max(elapsed, 1e-9))
        sps = (sam / max(elapsed, 1e-9))
        try:
            mem_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        except Exception:
            mem_gb = 0.0

        payload = {
            "throughput/step_time_sec": elapsed,
            "throughput/tokens_per_sec": tps,
            "throughput/samples_per_sec": sps,
            "counts/total_tokens_global": tot,
            "counts/labeled_tokens_global": lab,
            "pct/labeled_tokens": (lab / max(tot, 1.0)) if tot > 0 else 0.0,
            "gpu/max_mem_allocated_gb": mem_gb,
        }

        self._emitting = True
        try:
            tr.log(payload)
        finally:
            self._emitting = False

        return control


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True, local_files_only=args.local_files_only
    )
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Model (QLoRA optional)
    quant_cfg = None
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.use_qlora:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    if args.use_flash_attn and not is_flash_attn_2_available():
        raise RuntimeError(
            "flash-attn not available. Install a matching wheel for your CUDA/PyTorch."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        quantization_config=quant_cfg,
        torch_dtype=dtype if quant_cfg is None else None,
        device_map=None,
        attn_implementation="flash_attention_2" if args.use_flash_attn else "sdpa"
    )

    print(f"[ATTN] implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

    # Align PAD/BOS/EOS
    model.config.pad_token_id = tok.pad_token_id
    model.generation_config.pad_token_id = tok.pad_token_id
    model.config.bos_token_id = tok.bos_token_id
    model.generation_config.bos_token_id = tok.bos_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.generation_config.eos_token_id = tok.eos_token_id

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA config
    lora_cfg = LoraConfig(
        r=32, 
        lora_alpha=32, 
        lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", 
        task_type="CAUSAL_LM",
        use_rslora=False,
        loftq_config=None,
    )

    # Datasets: ONE example per conversation (unpadded; collator will fix length)
    train_ds = build_single_example_dataset(
        args.train_file, tok, max_len=args.max_seq_len, left_truncate=args.left_truncate
    )
    eval_ds = None
    if args.eval_file:
        eval_ds = build_single_example_dataset(
            args.eval_file, tok, max_len=args.max_seq_len, left_truncate=args.left_truncate
        )

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(f"[DATA] train examples: {len(train_ds):,}")
        if eval_ds is not None:
            print(f"[DATA] eval  examples: {len(eval_ds):,}")

    # Trainer config (explicit labels; no packing)
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="linear",
        # warmup_ratio=args.warmup_ratio,

        logging_strategy="steps",        
        logging_steps=args.logging_steps,
        logging_first_step=True,

        save_steps=args.save_steps,
        eval_strategy=("epoch" if eval_ds is not None else "no"),        
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=16,
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        save_total_limit=10,

        packing=False,
        completion_only_loss=False,   # we provide labels
        dataset_text_field=None,
        max_length=args.max_seq_len,  # not used by TRL here; harmless

        # save a checkpoint at the end of every epoch
        save_strategy="epoch"
    )

    if args.flash_attn_fused_ce and not is_flash_attn_2_available():
        raise RuntimeError(
            "flash-attn fused CE not available. Install a matching wheel for your CUDA/PyTorch."
        )
        
    if args.flash_attn_fused_ce:
        trainer_class = FlashAttnCETrainer
    else:
        trainer_class = LoggingSFTTrainer

    trainer = trainer_class(
        model=model,
        peft_config=lora_cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=cfg,
    )

    perf_cb = PerfOnLogCallback()
    perf_cb.trainer = trainer
    trainer.add_callback(perf_cb)

    # Train
    trainer.train()
    trainer.save_model()
    trainer.save_state()

    # Persist run args on master
    is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
