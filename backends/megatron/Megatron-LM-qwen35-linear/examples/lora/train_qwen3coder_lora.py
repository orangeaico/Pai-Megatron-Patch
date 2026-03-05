#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import torch
import torch.distributed as dist
from datasets import load_dataset, Features, Value
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

# ------------------------------------------------------------
# Args
# ------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)
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
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--no-use_qlora", dest="use_qlora", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--local_files_only", action="store_true", default=False)
    p.add_argument("--use_flash_attn", action="store_true", default=False)
    return p.parse_args()


def load_chat_jsonl(path):
    return load_dataset("json", data_files=path, split="train")


# ---- Subclass to count tokens per micro-step
class LoggingSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._accum_total_tokens = 0
        self._accum_labeled_tokens = 0
        self._accum_samples = 0
        self._step_start_time = None

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self._step_start_time is None:
            self._step_start_time = time.perf_counter()

        with torch.no_grad():
            attn = inputs.get("attention_mask")
            total_tokens = int(attn.sum().item()) if attn is not None else 0
            labels = inputs.get("labels")
            labeled_tokens = int((labels != -100).sum().item()) if labels is not None else 0
            input_ids = inputs.get("input_ids")
            micro_samples = int(input_ids.size(0)) if input_ids is not None else 0

        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        self._accum_total_tokens += total_tokens
        self._accum_labeled_tokens += labeled_tokens
        self._accum_samples += micro_samples
        return loss


class PerfOnLogCallback(TrainerCallback):
    """Compute throughput and attach it to the log payload that HF prints/records."""
    def __init__(self):
        super().__init__()
        self._last_log_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_log_time = time.perf_counter()

    def on_log(self, args, state, control, logs=None, **kwargs):
        trainer: LoggingSFTTrainer = getattr(self, "trainer", None)
        if trainer is None:
            return control

        now = time.perf_counter()
        elapsed = (now - self._last_log_time) if self._last_log_time else 0.0
        self._last_log_time = now

        # Reduce the accumulators across ranks to get global counts
        dev = trainer.model.device
        def red(x):
            t = torch.tensor([float(x)], device=dev)
            if dist.is_initialized():
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            return t.item()

        tot = red(trainer._accum_total_tokens)
        lab = red(trainer._accum_labeled_tokens)
        sam = red(trainer._accum_samples)

        # Compute per-second only if we have time and non-zero counts
        tps = (tot / elapsed) if elapsed > 0 else 0.0
        sps = (sam / elapsed) if elapsed > 0 else 0.0
        try:
            mem_gb = torch.cuda.max_memory_allocated(dev) / 1e9
        except Exception:
            mem_gb = 0.0

        # Attach to the current log payload so HF prints and records them
        if logs is not None:
            logs["throughput/step_time_sec"] = elapsed
            logs["throughput/tokens_per_sec"] = tps
            logs["throughput/samples_per_sec"] = sps
            logs["counts/total_tokens_global"] = tot
            logs["counts/labeled_tokens_global"] = lab
            logs["pct/labeled_tokens"] = (lab / max(tot, 1.0))
            logs["gpu/max_mem_allocated_gb"] = mem_gb

        # Reset accumulators after they’ve been logged
        trainer._accum_total_tokens = 0
        trainer._accum_labeled_tokens = 0
        trainer._accum_samples = 0
        trainer._step_start_time = None
        return control


def main():
    args = get_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    tok = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True, local_files_only=args.local_files_only
    )
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

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

    # ------------------------------------------------------------------
    # Train on ALL assistant messages, but ONLY if [prompt + completion + EOS]
    # already fits within max_seq_len. Longer pairs are discarded.
    # ------------------------------------------------------------------

    def expand_all_assistant_pairs_batched(batch):
        """Explode conversations into (prompt, completion) and DROP too-long pairs."""
        max_len = args.max_seq_len
        reserve_eos = 1  # TRL appends an EOS at the end of the sample
        out_prompts, out_completions = [], []
        msgs_batches = batch.get("messages", []) or batch.get("conversations", [])

        for msgs in msgs_batches:
            for i, m in enumerate(msgs):
                if m.get("role") != "assistant":
                    continue
                completion = (m.get("content") or "").strip()
                if not completion:
                    continue

                # Build prompt up to the assistant turn
                context = msgs[:i]
                prompt = tok.apply_chat_template(
                    context, tokenize=False, add_generation_prompt=True
                )

                # Tokenized lengths (no specials)
                prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                comp_ids   = tok(completion, add_special_tokens=False)["input_ids"]

                if len(comp_ids) == 0:
                    continue

                total_len = len(prompt_ids) + len(comp_ids) + reserve_eos
                if total_len <= max_len:            # <-- keep only if it fits
                    out_prompts.append(prompt)
                    out_completions.append(completion)
                else: 
                    break

        return {"prompt": out_prompts, "completion": out_completions}

    # Force Arrow to use large_string to avoid 32-bit offset overflow
    out_features = Features({"prompt": Value("large_string"), "completion": Value("large_string")})

    train_raw = load_chat_jsonl(args.train_file)
    train_pc = train_raw.map(
        expand_all_assistant_pairs_batched,
        batched=True,                                  # CHANGED: explode to rows
        remove_columns=train_raw.column_names,
        features=out_features,                         # CHANGED: large_string schema
        load_from_cache_file=False,
        desc="Expanding conversations into (prompt, completion) pairs (fit-only)",
    )
    # Safety filter (usually redundant now but harmless)
    train_pc = train_pc.filter(lambda ex: ex["completion"] and len(ex["completion"].strip()) > 0)

    eval_pc = None
    if args.eval_file:
        eval_raw = load_chat_jsonl(args.eval_file)
        eval_pc = eval_raw.map(
            expand_all_assistant_pairs_batched,
            batched=True,
            remove_columns=eval_raw.column_names,
            features=out_features,
            load_from_cache_file=False,
            desc="Expanding eval conversations (fit-only)",
        )
        eval_pc = eval_pc.filter(lambda ex: ex["completion"] and len(ex["completion"].strip()) > 0)

    # (Optional) quick visibility on how much data remained
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        print(f"[DATA] train examples after fit-only filter: {train_pc.num_rows:,}")
        if eval_pc is not None:
            print(f"[DATA] eval  examples after fit-only filter: {eval_pc.num_rows:,}")

    # ---- Trainer config
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
        eval_strategy=("epoch" if eval_pc is not None else "no"),
        eval_steps=(args.eval_steps if eval_pc is not None else None),
        bf16=args.bf16,
        fp16=not args.bf16,
        dataloader_num_workers=16,
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        save_total_limit=10,

        packing=False,
        completion_only_loss=True,
        dataset_text_field=None,
        max_length=args.max_seq_len,  # not used by TRL here; harmless

        # save a checkpoint at the end of every epoch
        save_strategy="epoch"
    )

    trainer = LoggingSFTTrainer(
        model=model,
        processing_class=tok,
        peft_config=lora_cfg,
        train_dataset=train_pc,
        eval_dataset=eval_pc,
        args=cfg,
    )
    trainer.add_callback(PerfOnLogCallback())

    trainer.train()
    trainer.save_model()
    trainer.save_state()

    is_master = (not dist.is_initialized()) or (dist.get_rank() == 0)
    if is_master:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)


if __name__ == "__main__":
    main()
