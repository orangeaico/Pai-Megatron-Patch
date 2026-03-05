#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path

def split_jsonl_train_val(input_file, eval_ratio=5.0, seed=42, shuffle=False):
    """
    Split a JSONL file into training and validation sets.
    Randomly samples validation set but preserves original order unless shuffle=True.

    Args:
        input_file: Path to input JSONL file
        eval_ratio: Percentage of data for validation (default: 5.0)
        seed: Random seed for reproducibility (default: 42)
        shuffle: If True, shuffle the entire dataset before splitting (default: False)
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read all lines from the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle the entire dataset if requested
    if shuffle:
        random.shuffle(lines)
        print("Dataset shuffled before splitting")

    # Calculate split sizes
    total_samples = len(lines)
    val_size = int(total_samples * (eval_ratio / 100))
    train_size = total_samples - val_size
    
    print(f"Total samples: {total_samples}")
    print(f"Training samples: {train_size}")
    print(f"Validation samples: {val_size}")
    print(f"Validation ratio: {eval_ratio}%")
    
    # Create indices for all samples
    indices = list(range(total_samples))
    
    # Randomly sample validation indices
    val_indices = sorted(random.sample(indices, val_size))
    val_indices_set = set(val_indices)
    
    # Get train indices (all indices not in validation)
    train_indices = [i for i in indices if i not in val_indices_set]
    
    # Extract lines maintaining order
    train_lines = [lines[i] for i in train_indices]
    val_lines = [lines[i] for i in val_indices]
    
    # Create output filenames
    input_path = Path(input_file)
    base_name = input_path.stem
    output_dir = input_path.parent
    
    train_file = output_dir / f"{base_name}_train.jsonl"
    val_file = output_dir / f"{base_name}_val.jsonl"
    
    # Write training set
    with open(train_file, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line)
    
    # Write validation set
    with open(val_file, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line)
    
    print(f"\nTrain set saved to: {train_file}")
    print(f"Validation set saved to: {val_file}")

def main():
    parser = argparse.ArgumentParser(description='Split JSONL file into training and validation sets')
    parser.add_argument('input_file', type=str, help='Path to input JSONL file')
    parser.add_argument('--eval_ratio', type=float, default=5.0,
                        help='Percentage of data for validation (default: 5.0)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--shuffle', action='store_true',
                        help='Shuffle the entire dataset before splitting')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    # Validate eval_ratio
    if args.eval_ratio <= 0 or args.eval_ratio >= 100:
        print(f"Error: eval_ratio must be between 0 and 100 (got {args.eval_ratio})")
        return
    
    split_jsonl_train_val(args.input_file, args.eval_ratio, args.seed, args.shuffle)

if __name__ == "__main__":
    main()