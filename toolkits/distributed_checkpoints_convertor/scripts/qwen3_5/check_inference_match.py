#!/usr/bin/env python3
# Copyright (c) 2026 Alibaba PAI Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import gc
import os
import sys
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Explain tensor parallelism in one sentence.",
    "Write a Python function that checks if a number is prime.",
]


def _load_prompts(args: argparse.Namespace) -> List[str]:
    prompts: List[str] = []
    if args.prompt:
        prompts.extend(args.prompt)
    if args.prompt_file:
        if not os.path.exists(args.prompt_file):
            raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)
    return prompts


def _load_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_model(model_dir: str, dtype_name: str):
    resolved_dtype = _resolve_torch_dtype(dtype_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=resolved_dtype,
        low_cpu_mem_usage=True,
        device_map=None,
    )
    model.to("cpu")
    model.eval()
    return model


def _generate_with_logits(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
) -> List[Dict]:
    runs = []
    for prompt in prompts:
        encoded = tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to("cpu")
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to("cpu")

        generated = input_ids
        attn = attention_mask
        step_logits = []
        step_tokens = []

        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=generated,
                    attention_mask=attn,
                    use_cache=False,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :].detach().to(dtype=torch.float32, device="cpu")
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                step_logits.append(logits.squeeze(0).clone())
                step_tokens.append(int(next_token.item()))

                generated = torch.cat([generated, next_token], dim=1)
                attn = torch.cat([attn, torch.ones_like(next_token)], dim=1)

        runs.append(
            {
                "prompt": prompt,
                "input_ids": input_ids.squeeze(0).tolist(),
                "generated_ids": generated.squeeze(0).tolist(),
                "new_tokens": step_tokens,
                "step_logits": step_logits,
            }
        )
    return runs


def _max_rel_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().clamp_min(1e-12)
    return ((a - b).abs() / denom).max().item()


def _compare_runs(
    base_runs: List[Dict],
    roundtrip_runs: List[Dict],
    atol: float,
    rtol: float,
) -> int:
    if len(base_runs) != len(roundtrip_runs):
        print("FAILED: prompt-count mismatch")
        print(f"  base={len(base_runs)} roundtrip={len(roundtrip_runs)}")
        return 1

    failed = False
    for prompt_idx, (base, rt) in enumerate(zip(base_runs, roundtrip_runs)):
        if base["prompt"] != rt["prompt"]:
            print("FAILED: prompt mismatch between runs")
            print(f"  idx={prompt_idx}")
            return 1

        if base["new_tokens"] != rt["new_tokens"]:
            failed = True
            print("TOKEN_MISMATCH")
            print(f"  prompt_idx={prompt_idx}")
            print(f"  prompt={base['prompt']}")
            for step, (b_tok, r_tok) in enumerate(zip(base["new_tokens"], rt["new_tokens"])):
                if b_tok != r_tok:
                    print(f"  first_diff_step={step} base_token={b_tok} roundtrip_token={r_tok}")
                    break

        if len(base["step_logits"]) != len(rt["step_logits"]):
            failed = True
            print("FAILED: logits-step mismatch")
            print(
                f"  prompt_idx={prompt_idx} base_steps={len(base['step_logits'])} "
                f"roundtrip_steps={len(rt['step_logits'])}"
            )
            continue

        for step, (base_logits, rt_logits) in enumerate(zip(base["step_logits"], rt["step_logits"])):
            if base_logits.shape != rt_logits.shape:
                failed = True
                print("LOGITS_SHAPE_MISMATCH")
                print(
                    f"  prompt_idx={prompt_idx} step={step} "
                    f"base_shape={tuple(base_logits.shape)} roundtrip_shape={tuple(rt_logits.shape)}"
                )
                break

            if torch.allclose(base_logits, rt_logits, atol=atol, rtol=rtol):
                continue

            failed = True
            diff = (base_logits - rt_logits).abs()
            max_abs = diff.max().item()
            max_index = int(diff.argmax().item())
            max_rel = _max_rel_delta(base_logits, rt_logits)
            print("LOGITS_MISMATCH")
            print(f"  prompt_idx={prompt_idx} step={step}")
            print(f"  max_abs_delta={max_abs}")
            print(f"  max_rel_delta={max_rel}")
            print(f"  max_index={max_index}")
            print(f"  base_value={base_logits.view(-1)[max_index].item()}")
            print(f"  roundtrip_value={rt_logits.view(-1)[max_index].item()}")
            break

    if failed:
        print("FAILED: inference parity check failed")
        return 1

    print("PASSED: greedy tokens match exactly and logits match within tolerance.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "CPU-only inference parity checker for Qwen3.5 HF roundtrip checkpoints "
            "(greedy token exact match + logits tolerance)."
        )
    )
    parser.add_argument("--base-hf-dir", required=True, help="Path to base/original HF checkpoint")
    parser.add_argument(
        "--roundtrip-hf-dir", required=True, help="Path to roundtripped HF checkpoint"
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt string to evaluate (repeatable).",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="Optional text file with one prompt per line.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Greedy generation length per prompt.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=5e-3,
        help="Absolute tolerance for logits parity.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=5e-3,
        help="Relative tolerance for logits parity.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype used for model loading on CPU.",
    )
    args = parser.parse_args()

    prompts = _load_prompts(args)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("--atol/--rtol must be >= 0")

    print(f"Loaded {len(prompts)} prompt(s).")

    tokenizer = _load_tokenizer(args.base_hf_dir)

    print(f"Loading base model on CPU: {args.base_hf_dir}")
    base_model = _load_model(args.base_hf_dir, args.dtype)
    base_runs = _generate_with_logits(base_model, tokenizer, prompts, args.max_new_tokens)
    del base_model
    gc.collect()

    print(f"Loading roundtrip model on CPU: {args.roundtrip_hf_dir}")
    roundtrip_model = _load_model(args.roundtrip_hf_dir, args.dtype)
    roundtrip_runs = _generate_with_logits(roundtrip_model, tokenizer, prompts, args.max_new_tokens)
    del roundtrip_model
    gc.collect()

    return _compare_runs(base_runs, roundtrip_runs, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)
