#!/usr/bin/env python3
"""
Script to sort JSONL dataset by number of tokens after applying chat template and tokenization.
Uses Qwen3-1.7B tokenizer and prints length statistics.
"""

import json
import argparse
import os
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file and return list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dictionaries to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_single_item(item_with_index: Tuple[int, Dict], tokenizer_path: str) -> Tuple[int, Dict, int]:
    """
    Process a single item: apply chat template and tokenize.
    Returns (index, item, token_count).
    """
    index, item = item_with_index
    
    # Load tokenizer for this process (can't pickle tokenizer objects)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # Check if item has 'messages' field for chat format
    if 'messages' in item:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            item['messages'], 
            tokenize=False,
            add_generation_prompt=False
        )
    elif 'conversations' in item:
        # Apply chat template
        text = tokenizer.apply_chat_template(
            item['conversations'], 
            tokenize=False,
            add_generation_prompt=False
        )
    elif 'text' in item:
        # If no messages field, use text field directly
        text = item['text']
    else:
        # Try to find any text field
        possible_text_fields = ['content', 'prompt', 'input', 'output']
        text = None
        for field in possible_text_fields:
            if field in item:
                text = item[field]
                break
        if text is None:
            text = ""
    
    # Tokenize
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_count = len(tokens)
    
    return (index, item, token_count)


def apply_chat_template_and_tokenize(data: List[Dict], tokenizer_path: str, num_workers: int = 8) -> List[Tuple[Dict, int]]:
    """
    Apply chat template and tokenize each item in the dataset using multiprocessing.
    Returns list of (item, token_count) tuples.
    """
    # Create list of (index, item) tuples to preserve order
    indexed_data = list(enumerate(data))
    
    # Create partial function with tokenizer path
    process_func = partial(process_single_item, tokenizer_path=tokenizer_path)
    
    # Process in parallel
    print(f"Tokenizing with {num_workers} workers...")
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, indexed_data, chunksize=10),
            total=len(data),
            desc="Tokenizing"
        ))
    
    # Sort by index to preserve original order
    results.sort(key=lambda x: x[0])
    
    # Extract (item, token_count) tuples
    data_with_lengths = [(item, token_count) for _, item, token_count in results]
    
    return data_with_lengths


def calculate_and_print_stats(token_counts: List[int]):
    """Calculate and print statistics about token lengths."""
    if not token_counts:
        print("No data to calculate statistics.")
        return
    
    counts = np.array(token_counts)
    
    print("\n" + "="*50)
    print("Token Length Statistics")
    print("="*50)
    print(f"Total samples: {len(counts):,}")
    print(f"Min tokens: {np.min(counts):,}")
    print(f"Max tokens: {np.max(counts):,}")
    print(f"Mean tokens: {np.mean(counts):,.2f}")
    print(f"Median tokens: {np.median(counts):,.2f}")
    print(f"Std dev: {np.std(counts):,.2f}")
    
    # Count examples at max value
    max_count = np.max(counts)
    num_at_max = np.sum(counts == max_count)
    pct_at_max = (num_at_max / len(counts)) * 100
    print(f"\nExamples at max ({max_count:,} tokens): {num_at_max:,} ({pct_at_max:.2f}%)")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        print(f"  {p}th percentile: {np.percentile(counts, p):,.2f}")
    
    # Histogram buckets
    print("\nDistribution:")
    buckets = [(0, 8192), (8192, 16384), (16384, 24576), (24576, 32768), 
               (32768, 40960), (40960, 49152), (49152, 57344), (57344, 65536),
               (65536, float('inf'))]
    
    cumulative_count = 0
    print(f"  {'Bucket':15s}  {'Count':>8s} {'Pct':>6s}  {'Cumulative':>10s} {'Cum %':>6s}")
    print("  " + "-" * 55)
    
    for start, end in buckets:
        if end == float('inf'):
            count = np.sum(counts >= start)
            bucket_name = f"{start:,}+"
        else:
            count = np.sum((counts >= start) & (counts < end))
            bucket_name = f"{start:,}-{end:,}"
        
        percentage = (count / len(counts)) * 100
        cumulative_count += count
        cum_percentage = (cumulative_count / len(counts)) * 100
        
        print(f"  {bucket_name:15s}  {count:8,} {percentage:5.2f}%  {cumulative_count:10,} {cum_percentage:5.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Sort JSONL dataset by token count after applying chat template"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="/home/shared/megatron_dir/hf_models/Qwen3-1.7B/",
        help="Path to tokenizer model (default: /home/shared/megatron_dir/hf_models/Qwen3-1.7B/)"
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort in descending order (longest first)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for tokenization (default: 8)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data):,} samples")
    
    # Apply chat template and tokenize
    print("Applying chat template and tokenizing...")
    data_with_lengths = apply_chat_template_and_tokenize(data, args.tokenizer_path, args.workers)
    
    # Sort by token count
    print(f"Sorting by token count ({'descending' if args.descending else 'ascending'})...")
    data_with_lengths.sort(key=lambda x: x[1], reverse=args.descending)
    
    # Extract sorted data and token counts
    sorted_data = [item[0] for item in data_with_lengths]
    token_counts = [item[1] for item in data_with_lengths]
    
    # Generate output filename
    base_name = os.path.splitext(args.input_file)[0]
    output_file = f"{base_name}_sorted.jsonl"
    
    # Save sorted data
    print(f"Saving sorted data to {output_file}...")
    save_jsonl(sorted_data, output_file)
    
    # Print statistics
    calculate_and_print_stats(token_counts)
    
    print(f"\nDone! Sorted data saved to {output_file}")


if __name__ == "__main__":
    main()