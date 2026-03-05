#!/usr/bin/env python3
import argparse
from transformers import AutoTokenizer

def main():
    ap = argparse.ArgumentParser(description="Print Qwen/Qwen3 token for given token id(s).")
    ap.add_argument("ids", nargs="+", type=int, help="Token id(s) to inspect (e.g., 0 1 2 or 32000).")
    ap.add_argument("--model", "-m", default="Qwen/Qwen3-1.7B",
                    help="HF model id or local path (default: Qwen/Qwen3-1.7B).")
    ap.add_argument("--no-remote-code", action="store_true",
                    help="Disable trust_remote_code if you don't want to execute remote code.")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=not args.no_remote_code,
        use_fast=True
    )

    vocab_size = len(tokenizer)

    print(f"Loaded tokenizer from: {args.model}")
    print(f"Tokenizer size (len): {vocab_size}\n")

    for idx in args.ids:
        if idx < 0 or idx >= vocab_size:
            print(f"[id {idx}] ❌ out of range (0..{vocab_size-1})")
            continue

        # Token string from id
        tok = tokenizer.convert_ids_to_tokens(idx)

        # A human-ish rendering of the single token as text
        decoded = tokenizer.decode([idx], clean_up_tokenization_spaces=False, skip_special_tokens=False)

        print(f"[id {idx}]")
        print(f"  token (raw): {repr(tok)}")
        print(f"  token (decoded text): {repr(decoded)}")
        print("")

if __name__ == "__main__":
    main()
