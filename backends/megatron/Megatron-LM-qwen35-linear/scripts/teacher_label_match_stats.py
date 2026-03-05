#!/usr/bin/env python3
"""
Compute how often the teacher's top token matches the label, and how often
the gold label is present anywhere in the teacher's token list for each
position, aggregated across a directory of JSON files.

Expected JSON structure per file:
- input_ids: List[int]
- labels:    List[int]
- teacher_logits: {
    positions: List[int],           # positions in the sequence for which logits are provided
    values:    List[List[float]],   # logits for top-k tokens at each position
    indices:   List[List[int]]      # token IDs for those logits
  }

Usage:
  python3 teacher_label_match_stats.py --dir /path/to/jsons [--recursive] [--limit N]

Notes:
- Only positions with valid, non-negative labels and in-bounds indices are considered.
- The teacher "top token" is determined by the maximum finite logit at that position.
- "Label present in teacher tokens" checks if the gold label appears anywhere in the
  teacher's token indices list for that position.
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np


def argmax_finite(values: List[float]) -> int:
    """Return index of max among finite values; -1 if none are finite."""
    if not values:
        return -1
    arr = np.asarray(values, dtype=float)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return -1
    # Restrict to finite values to avoid NaN/-inf issues
    finite_indices = np.nonzero(finite_mask)[0]
    finite_vals = arr[finite_mask]
    rel_idx = int(np.argmax(finite_vals))
    return int(finite_indices[rel_idx])


def process_file(path: Path) -> Tuple[int, int, int, int, int]:
    """
    Process a single JSON file and return counts:
    - total_positions: total teacher positions in file
    - considered_positions: positions with in-bounds, non-negative labels
    - top_equals_label: count where teacher top token == label
    - label_in_teacher: count where label appears anywhere in teacher indices
    - skipped_positions: positions skipped due to invalid label/indexing
    """
    try:
        with path.open('r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return 0, 0, 0, 0, 0

    labels = data.get('labels')[1:]+[-100]
    tl = data.get('teacher_logits', {}) or {}
    positions = tl.get('positions') or []
    values = tl.get('values') or []
    indices = tl.get('indices') or []

    # Basic validation
    n = min(len(positions), len(values), len(indices))
    if n == 0 or labels is None:
        return 0, 0, 0, 0, 0

    total_positions = n
    considered_positions = 0
    top_equals_label = 0
    label_in_teacher = 0
    skipped_positions = 0

    labels_len = len(labels)

    for i in range(n):
        pos = positions[i]
        # Validate position
        if not isinstance(pos, int) or pos < 0 or pos >= labels_len:
            skipped_positions += 1
            continue

        label = labels[pos]
        # Skip masked/invalid labels (e.g., -100)
        if not isinstance(label, int) or label < 0:
            skipped_positions += 1
            continue

        logits_i = values[i]
        tokens_i = indices[i]
        if not isinstance(logits_i, list) or not isinstance(tokens_i, list) or len(logits_i) != len(tokens_i) or len(tokens_i) == 0:
            skipped_positions += 1
            continue

        considered_positions += 1

        # Find top token among finite logits only
        top_idx = argmax_finite(logits_i)
        top_token = tokens_i[top_idx] if top_idx >= 0 else None

        if top_token is not None and top_token == label:
            top_equals_label += 1

        # Check presence anywhere in teacher token list
        try:
            if label in tokens_i:
                label_in_teacher += 1
        except Exception:
            # In case tokens_i isn't a simple list of ints
            pass

    return total_positions, considered_positions, top_equals_label, label_in_teacher, skipped_positions


def main():
    parser = argparse.ArgumentParser(description="Count teacher top-token vs label matches across distillation JSONs")
    parser.add_argument("--dir", type=str, default="/home/shared/megatron_dir/data/distillation/qwen_480b_swe_bench", help="Directory containing JSON files")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files for quicker runs (0 = no limit)")
    parser.add_argument("--ext", type=str, default=".json", help="File extension to include (default: .json)")
    args = parser.parse_args()

    data_dir = Path(args.dir)
    if not data_dir.exists():
        print(f"Directory not found: {data_dir}")
        return

    if args.recursive:
        files = [p for p in data_dir.rglob(f"*{args.ext}") if p.is_file()]
    else:
        files = [p for p in data_dir.glob(f"*{args.ext}") if p.is_file()]

    if args.limit and args.limit > 0:
        files = files[: args.limit]

    if not files:
        print(f"No {args.ext} files found in {data_dir}")
        return

    print(f"Scanning {len(files)} file(s) under {data_dir} ...")

    grand_total_positions = 0
    grand_considered_positions = 0
    grand_top_equals_label = 0
    grand_label_in_teacher = 0
    grand_skipped_positions = 0

    for idx, path in enumerate(files, 1):
        if idx % 25 == 0:
            print(f"  Processed {idx}/{len(files)} files ...")

        t, c, top_eq, in_any, skipped = process_file(path)
        grand_total_positions += t
        grand_considered_positions += c
        grand_top_equals_label += top_eq
        grand_label_in_teacher += in_any
        grand_skipped_positions += skipped

    denom = grand_considered_positions if grand_considered_positions > 0 else 1
    top_match_pct = (grand_top_equals_label / denom) * 100.0
    in_any_pct = (grand_label_in_teacher / denom) * 100.0

    print("\n=== Teacher vs Label Match Summary ===")
    print(f"Files processed: {len(files)}")
    print(f"Teacher positions (raw): {grand_total_positions}")
    print(f"Positions considered (valid label): {grand_considered_positions}")
    print(f"Positions skipped (invalid/out-of-bounds): {grand_skipped_positions}")
    print(f"Top token == label: {grand_top_equals_label} ({top_match_pct:.2f}% of considered)")
    print(f"Label in teacher tokens: {grand_label_in_teacher} ({in_any_pct:.2f}% of considered)")


if __name__ == "__main__":
    main()

