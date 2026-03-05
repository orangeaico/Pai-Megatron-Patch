#!/usr/bin/env python3

# Example to train a LoRA on the fused and quantized version of Qwen3-30B-A3B using Unsloth

import os
import json
import argparse
from typing import List, Dict

import torch

# Import unsloth before others
from unsloth import FastModel

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from qwen3_moe_fused.fast_lora import patch_Qwen3MoeFusedSparseMoeBlock_forward
from qwen3_moe_fused.lora import patch_lora_config
from qwen3_moe_fused.modular_qwen3_moe_fused import Qwen3MoeFusedForCausalLM
from qwen3_moe_fused.quantize.quantizer import patch_bnb_quantizer

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="bash99/Qwen3-30B-A3B-Instruct-2507-fused-bnb-4bit")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="qwen3-30b-a3b-unsloth-lora")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--per_device_train_bs", type=int, default=1)
    p.add_argument("--per_device_eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--no-use_qlora", dest="use_qlora", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--weight_decay", type=float, default=0.01)
    # LoRA hyperparams (defaults kept conservative for 30B A3B fused model)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=1)
    return p.parse_args()


def main():
    args = get_args()

    # Patches for fused Qwen3-MoE model and quantizer
    patch_bnb_quantizer()
    # We can set a smaller rank for MoE layers; rslora handles scaling
    patch_lora_config(
        rank_pattern={
            "q_proj": 32,
            "k_proj": 32,
            "v_proj": 32,
            "o_proj": 32,
            # "gate": 32,  # LoRA on routing gate is possible but unstable
            "gate_proj": 32,
            "up_proj": 32,
            "down_proj": 32,
        }
    )
    patch_Qwen3MoeFusedSparseMoeBlock_forward()

    # Load fused, bnb-4bit quantized model
    model, tokenizer = FastModel.from_pretrained(
        args.model_name,
        auto_model=Qwen3MoeFusedForCausalLM,
        device_map="cuda:0",
    )

    # Apply LoRA
    model = FastModel.get_peft_model(
        model,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            # "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        device_map="cuda:0",
    )
    
    # Ensure PAD token is set for padding
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_file_list = [args.train_file]
    eval_file_list = [args.eval_file] if args.eval_file else None
    print(f"[Train dataset] : {train_file_list}")
    print(f"[Eval dataset] : {eval_file_list}")

    # ---------------- Data loader: one example per conversation (assistant-only labels) ----------------
    IGNORE_INDEX = -100

    def _pick_msgs(ex: Dict) -> List[Dict]:
        if "messages" in ex and ex["messages"] is not None:
            return ex["messages"]
        if "conversations" in ex and ex["conversations"] is not None:
            return ex["conversations"]
        raise ValueError("Each record must have 'messages' or 'conversations'.")

    def _tokenize_one_conversation(conv_msgs: List[Dict], tok, max_len: int, left_truncate: bool = False):
        """
        Build a single example:
          - input_ids: concatenated chat-template segments (one per message)
          - labels: token ids only for assistant spans; IGNORE_INDEX elsewhere
          - attention_mask: 1 for real tokens, 0 for padding
        We clamp to max_len here, so no custom collator is needed.
        """
        input_ids: List[int] = []
        labels: List[int] = []

        for m in conv_msgs:
            role = m.get("role", "user")
            if role not in ("user", "assistant", "system"):
                continue

            content = m.get("content", "")
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)

            seg_ids = tok.apply_chat_template(
                [{"role": role, "content": content}],
                tokenize=True,
                add_generation_prompt=False,
            )
            if not isinstance(seg_ids, list):
                seg_ids = list(seg_ids)

            # stop if adding this would exceed the window
            if len(input_ids) + len(seg_ids) > max_len:
                break

            input_ids.extend(seg_ids)
            if role == "assistant":
                labels.extend(seg_ids)  # learn on assistant spans
            else:
                labels.extend([IGNORE_INDEX] * len(seg_ids))

        # clamp if too long
        if len(input_ids) > max_len:
            if left_truncate:
                input_ids[:] = input_ids[-max_len:]
                labels[:] = labels[-max_len:]
            else:
                input_ids[:] = input_ids[:max_len]
                labels[:] = labels[:max_len]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }

    def _map_record(ex):
        return _tokenize_one_conversation(_pick_msgs(ex), tokenizer, args.max_seq_len, left_truncate=False)

    # Load raw files (json or jsonl supported by datasets "json" loader)
    train_raw = load_dataset("json", data_files={"train": train_file_list})["train"]
    eval_raw = None
    if eval_file_list:
        eval_raw = load_dataset("json", data_files={"eval": eval_file_list})["eval"]

    # Tokenize + fix length inside the dataset
    train_combined_dataset = train_raw.map(
        _map_record,
        remove_columns=train_raw.column_names,
        desc="Tokenizing train (assistant-only labels, fixed length)",
    )
    train_combined_dataset = train_combined_dataset.filter(lambda x: len(x["input_ids"]) > 0)

    eval_combined_dataset = None
    if eval_raw is not None:
        eval_combined_dataset = eval_raw.map(
            _map_record,
            remove_columns=eval_raw.column_names,
            desc="Tokenizing eval (assistant-only labels, fixed length)",
        )
    eval_combined_dataset = eval_combined_dataset.filter(lambda x: len(x["input_ids"]) > 0) if eval_combined_dataset is not None else None

    print(f"[DATA] train examples: {len(train_combined_dataset):,}")
    if eval_combined_dataset is not None:
        print(f"[DATA] eval  examples: {len(eval_combined_dataset):,}")

    # ---------------- Train ----------------
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_combined_dataset,
        eval_dataset=eval_combined_dataset,
        args=SFTConfig(
            # we now pass tokenized tensors, not raw text
            dataset_text_field=None,
            packing=False,
            completion_only_loss=False,

            output_dir=args.output_dir,
            per_device_train_batch_size=args.per_device_train_bs,
            per_device_eval_batch_size=args.per_device_eval_bs,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=args.logging_steps,
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",

            # run eval at the end of every epoch (logs eval_loss)
            eval_strategy="epoch" if eval_combined_dataset is not None else "no",
            # save a checkpoint at the end of every epoch
            save_strategy="epoch",
            # Logging based on training steps
            logging_strategy="steps",
            # keep only the most recent N checkpoints
            save_total_limit=10,
            logging_first_step=True,
            bf16=args.bf16,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    main()
