#!/usr/bin/env python3
"""
Script to analyze weight matrices from a pickle file and calculate statistics.
"""

import torch
import pickle
import argparse
import numpy as np
from typing import Dict, Tuple, List

def calculate_tensor_stats(tensor: torch.Tensor) -> Dict[str, float]:
    """Calculate statistics for a tensor."""
    tensor_np = tensor.float().cpu().numpy()
    return {
        'min': float(np.min(tensor_np)),
        'max': float(np.max(tensor_np)),
        'mean': float(np.mean(np.abs(tensor_np))),  # Mean of absolute values
        'std': float(np.std(tensor_np))
    }

def analyze_weights(pkl_path: str) -> Dict[str, Dict[str, float]]:
    """Load pickle file and analyze all weight matrices."""
    print(f"Loading pickle file: {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        state_dict = pickle.load(f)
    
    print(f"\nAnalyzing {len(state_dict)} weight matrices...\n")
    
    stats_dict = {}
    
    for name, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            stats = calculate_tensor_stats(tensor)
            stats_dict[name] = stats
            
            print(f"{name}:")
            print(f"  Shape: {list(tensor.shape)}")
            print(f"  Min: {stats['min']:.6f}")
            print(f"  Max: {stats['max']:.6f}")
            print(f"  Mean (abs): {stats['mean']:.6f}")
            print(f"  Std: {stats['std']:.6f}")
            print()
    
    return stats_dict

def print_top_bottom_matrices(stats_dict: Dict[str, Dict[str, float]], n: int = 3):
    """Print top and bottom n matrices by mean value."""
    # Sort by mean value
    sorted_by_mean = sorted(stats_dict.items(), key=lambda x: x[1]['mean'])
    
    print(f"\n{'='*80}")
    print(f"TOP {n} MATRICES BY MEAN ABSOLUTE VALUE:")
    print(f"{'='*80}")
    
    for name, stats in sorted_by_mean[-n:][::-1]:
        print(f"\n{name}:")
        print(f"  Mean (abs): {stats['mean']:.6f}")
        print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Std: {stats['std']:.6f}")
    
    print(f"\n{'='*80}")
    print(f"BOTTOM {n} MATRICES BY MEAN ABSOLUTE VALUE:")
    print(f"{'='*80}")
    
    for name, stats in sorted_by_mean[:n]:
        print(f"\n{name}:")
        print(f"  Mean (abs): {stats['mean']:.6f}")
        print(f"  Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Std: {stats['std']:.6f}")
    
    print(f"\n{'='*80}")

def print_overall_statistics(stats_dict: Dict[str, Dict[str, float]]):
    """Print overall statistics across all matrices."""
    all_means = [stats['mean'] for stats in stats_dict.values()]
    all_stds = [stats['std'] for stats in stats_dict.values()]
    all_mins = [stats['min'] for stats in stats_dict.values()]
    all_maxs = [stats['max'] for stats in stats_dict.values()]
    
    print("\nOVERALL STATISTICS ACROSS ALL MATRICES:")
    print(f"{'='*80}")
    print(f"Average mean (abs) across all matrices: {np.mean(all_means):.6f}")
    print(f"Average std across all matrices: {np.mean(all_stds):.6f}")
    print(f"Global minimum value: {np.min(all_mins):.6f}")
    print(f"Global maximum value: {np.max(all_maxs):.6f}")
    print(f"Std of means across matrices: {np.std(all_means):.6f}")
    print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Analyze weight matrices from pickle file")
    parser.add_argument("pickle_file", type=str, help="Path to the pickle file")
    parser.add_argument("--top-n", type=int, default=3, 
                        help="Number of top/bottom matrices to show (default: 3)")
    parser.add_argument("--brief", action="store_true", 
                        help="Show only summary statistics")
    
    args = parser.parse_args()
    
    # Analyze weights
    if args.brief:
        print(f"Loading pickle file: {args.pickle_file}")
        with open(args.pickle_file, 'rb') as f:
            state_dict = pickle.load(f)
        
        print(f"\nAnalyzing {len(state_dict)} weight matrices...")
        stats_dict = {}
        
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                stats = calculate_tensor_stats(tensor)
                stats_dict[name] = stats
    else:
        stats_dict = analyze_weights(args.pickle_file)
    
    # Print overall statistics
    print_overall_statistics(stats_dict)
    
    # Print top and bottom matrices
    print_top_bottom_matrices(stats_dict, n=args.top_n)

if __name__ == "__main__":
    main()