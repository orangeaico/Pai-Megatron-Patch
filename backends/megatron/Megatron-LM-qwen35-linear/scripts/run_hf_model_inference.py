#!/usr/bin/env python3
"""
Run HF causal LM inference from a local base model path, with optional LoRA adapter.
- Supports single-prompt, interactive, and chat-template modes.
- Supports RoPE scaling flags.
- Can merge LoRA into the base model for faster inference.
"""

import os
import json
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Optional: PEFT for LoRA ---
try:
    from peft import PeftModel, AutoPeftModelForCausalLM
    _PEFT_AVAILABLE = True
except Exception:
    _PEFT_AVAILABLE = False


def _sanitize_adapter_config(adapter_dir: str):
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        # Fix legacy/null loftq_config that older peft can’t handle
        if "loftq_config" in cfg and cfg["loftq_config"] is None:
            cfg["loftq_config"] = {}
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)

# ------------------------------
# Model / Tokenizer Loading
# ------------------------------
def load_model_and_tokenizer(
    model_path: str,
    rope_scaling_type: str = None,
    rope_scaling_factor: float = None,
    original_max_position_embeddings: int = None,
    lora_path: str = None,
    merge_lora: bool = False,
    tokenizer_path: str = None,
):
    """
    Load base model and tokenizer from local disk, and (optionally) attach/merge a LoRA adapter.

    Returns: (model, tokenizer)
    """
    if lora_path and not _PEFT_AVAILABLE:
        raise RuntimeError(
            "peft is required for --lora-path. Install with: pip install peft"
        )

    # Validate paths
    if model_path and not os.path.exists(model_path):
        raise FileNotFoundError(f"Base model path does not exist: {model_path}")
    if lora_path and not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA adapter path does not exist: {lora_path}")
    if tokenizer_path and not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer path does not exist: {tokenizer_path}")

    # Tokenizer
    tok_src = tokenizer_path or model_path or lora_path
    if not tok_src:
        raise ValueError("Could not determine tokenizer source. Provide --model-path or --tokenizer-path.")
    print(f"Loading tokenizer from: {tok_src}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_src,
        trust_remote_code=True,
        local_files_only=True
    )

    # Prepare base kwargs
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": True,
        "local_files_only": True
    }

    # RoPE scaling (if requested)
    if rope_scaling_type and rope_scaling_factor:
        rope_scaling = {
            "type": rope_scaling_type,
            "factor": rope_scaling_factor
        }
        if original_max_position_embeddings:
            rope_scaling["original_max_position_embeddings"] = original_max_position_embeddings
        model_kwargs["rope_scaling"] = rope_scaling
        # model_kwargs["max_position_embeddings"] = 65000
        print(f"Using RoPE scaling: {rope_scaling}")

    # --- Load model (+ optional LoRA) ---
    if lora_path:
        # If adapter contains base reference, AutoPeft can load both in one go.
        # If you want to FORCE a specific base, pass model_path explicitly below.
        # Fallback to wrapping a separately loaded base with PeftModel.
        try:
            print(f"Attempting AutoPeft load from adapter: {lora_path}")
            autopeft_kwargs = dict(model_kwargs)
            if model_path:
                autopeft_kwargs["base_model_name_or_path"] = model_path
            model = AutoPeftModelForCausalLM.from_pretrained(lora_path, **autopeft_kwargs)
            print("Loaded via AutoPeftModelForCausalLM.")
        except Exception as e:
            print(f"AutoPeft path failed ({e}). Falling back to base+adapter attach.")
            if not model_path:
                raise ValueError("AutoPeft failed and no --model-path was provided to load a base model.")
            print(f"Loading base model from: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

            # Sanitize adapter config to avoid loftq None crash
            _sanitize_adapter_config(lora_path)

            print(f"Attaching LoRA adapter from: {lora_path}")
            model = PeftModel.from_pretrained(
                model,
                lora_path,
                is_trainable=False,
                device_map="auto",
                torch_dtype=torch.bfloat16,
            )

        if merge_lora:
            print("Merging LoRA weights into base model...")
            model = model.merge_and_unload()
            print("Merge complete.")
    else:
        # No adapter: load plain base
        if not model_path:
            raise ValueError("When --lora-path is not given, --model-path is required.")
        print(f"Loading base model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    # Tokenizer pad token safety
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Model & tokenizer ready.")
    return model, tokenizer


# ------------------------------
# Text Generation
# ------------------------------
def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    top_k: int = None,
    show_stats: bool = True,
):
    """
    Generate text and return (full_decoded, only_new_text, stats_dict).
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generation args
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
    }
    if top_k is not None:
        gen_kwargs["top_k"] = top_k

    # Generate
    with torch.no_grad():
        st = time.time()
        outputs = model.generate(**gen_kwargs)
        if show_stats:
            print(f"Generation time: {time.time() - st:.3f}s")
            # printing the dynamic generation_config can be noisy; keep for debugging
            print("Generation config:", model.generation_config)

    # Decode and slice new text (supports chat-style "<|im_start|>assistant")
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=False)

    # Try to locate an assistant tag (common in chat templates)
    tag = "<|im_start|>assistant"
    idx = full_text.find(tag)
    if idx != -1:
        only_new = full_text[idx + len(tag):].strip()
    else:
        # Fallback: naive slice new tokens relative to input length
        input_len = inputs["input_ids"].shape[1]
        new_ids = outputs[0][input_len:]
        only_new = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    stats = {
        "input_tokens": int(inputs["input_ids"].shape[1]),
        "output_tokens": int(outputs.shape[1]),
        "new_tokens": int(outputs.shape[1] - inputs["input_ids"].shape[1]),
    }
    return full_text, only_new, stats


# ------------------------------
# Modes
# ------------------------------
def run_single_inference(args):
    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
        lora_path=args.lora_path,
        merge_lora=args.merge_lora,
        tokenizer_path=args.tokenizer_path,
    )

    print(f"\nModel type: {type(model).__name__}")
    print(f"Device: {next(model.parameters()).device}")
    print(f"Mode: {'Sampling (non-deterministic)' if args.do_sample else 'Greedy (deterministic)'}")

    print(f"\nPrompt: {args.prompt}")
    print("-" * 80)

    print("Generating...")
    full_text, new_text, stats = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.do_sample,
        top_k=args.top_k,
        show_stats=True,
    )

    print("\nGenerated text:")
    print("=" * 80)
    print(full_text)
    print("=" * 80)

    print("\nNew tokens only:")
    print("-" * 80)
    print(new_text)
    print("-" * 80)

    print("\nToken counts:")
    print(f"  Input tokens: {stats['input_tokens']}")
    print(f"  Output tokens: {stats['output_tokens']}")
    print(f"  New tokens generated: {stats['new_tokens']}")


def run_interactive_mode(args):
    print("Running in interactive mode. Type 'quit' or 'exit' to stop.\n")

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
        lora_path=args.lora_path,
        merge_lora=args.merge_lora,
        tokenizer_path=args.tokenizer_path,
    )

    print("Ready for prompts.\n")
    while True:
        try:
            prompt = input("\nEnter prompt (or 'quit' to exit): ")
        except EOFError:
            break
        if prompt.lower() in ("quit", "exit"):
            break
        if not prompt.strip():
            continue

        print("\nGenerating...")
        _, new_text, _ = generate_text(
            model,
            tokenizer,
            prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.do_sample,
            top_k=args.top_k,
            show_stats=False,
        )

        print("\nGenerated continuation:")
        print("-" * 80)
        print(new_text)
        print("-" * 80)


def run_chat_mode(args):
    print("Running in chat mode...\n")

    model, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        original_max_position_embeddings=args.original_max_position_embeddings,
        lora_path=args.lora_path,
        merge_lora=args.merge_lora,
        tokenizer_path=args.tokenizer_path,
    )

    # Read chat prompt (expects a file with {"conversations": [...]})
    with open("scripts/inference_prompt.txt", "r") as f:
        chat_data = json.load(f)
    conversations = chat_data.get("conversations", [])

    # Apply chat template (enable_thinking can be toggled if your tokenizer supports it)
    prompt = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    os.makedirs("scripts", exist_ok=True)
    with open("scripts/prompt.txt", "w") as f:
        f.write(prompt)

    print("Chat prompt formatted:")
    print("-" * 80)
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("-" * 80)

    print("\nGenerating assistant response...")
    _, assistant_response, stats = generate_text(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.do_sample,
        top_k=args.top_k,
    )

    with open("scripts/inference_response.txt", "w") as f:
        f.write(assistant_response)

    print("\nAssistant response:")
    print("=" * 80)
    print(assistant_response)
    print("=" * 80)

    print("\nToken counts:")
    print(f"  Input tokens: {stats['input_tokens']}")
    print(f"  Output tokens: {stats['output_tokens']}")
    print(f"  New tokens generated: {stats['new_tokens']}")


# ------------------------------
# CLI
# ------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Run HF model inference from local path (+ optional LoRA).")
    p.add_argument("--model-path", type=str, default=None,
                   help="Local path to the base model directory")
    p.add_argument("--tokenizer-path", type=str, default=None,
                   help="Optional: load tokenizer from this path instead of model/adapter")
    p.add_argument("--lora-path", type=str, default=None,
                   help="Local path to a PEFT/LoRA adapter (folder with adapter_config.json)")
    p.add_argument("--merge-lora", action="store_true", default=False,
                   help="Merge LoRA weights into the base model for inference")
    p.add_argument("--prompt", type=str,
                   default="""Once upon a time, in a land far, far away,""",
                   help="Input prompt for text generation")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="Maximum number of new tokens to generate")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for sampling (only used if --do-sample is set)")
    p.add_argument("--top-p", type=float, default=1.0,
                   help="Top-p sampling (nucleus)")
    p.add_argument("--top-k", type=int, default=None,
                   help="Top-k sampling (e.g., 1 greedy, 50 diverse)")
    p.add_argument("--do-sample", action="store_true", default=False,
                   help="Use sampling instead of greedy decoding")
    p.add_argument("--interactive", action="store_true",
                   help="Run in interactive mode for multiple prompts")
    p.add_argument("--chat", action="store_true",
                   help="Run in chat mode with system/user/assistant messages")
    p.add_argument("--rope-scaling-type", type=str, default=None,
                   choices=["linear", "dynamic", "yarn", "longrope"],
                   help="RoPE scaling type")
    p.add_argument("--rope-scaling-factor", type=float, default=None,
                   help="RoPE scaling factor, e.g., 2.0, 4.0")
    p.add_argument("--original-max-position-embeddings", type=int, default=None,
                   help="Original max position embeddings (for RoPE scaling)")
    return p


def main():
    args = build_argparser().parse_args()

    if args.interactive:
        run_interactive_mode(args)
    elif args.chat:
        run_chat_mode(args)
    else:
        run_single_inference(args)


if __name__ == "__main__":
    main()
