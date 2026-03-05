#!/usr/bin/env python3
"""
Script to compare trajectory files between two sets of directories.
Finds common bug IDs and detects where their trajectories first differ.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
import argparse
import sys

def read_trajectory_file(file_path: str) -> dict:
    """Read and parse a trajectory JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_assistant_actions(trajectory_data: dict) -> List[str]:
    """Extract actions from assistant messages in the trajectory."""
    actions = []
    try:
        # Only look at the LAST trajectory item as they are cumulative
        trajectory = trajectory_data.get("trajectory", [])
        if trajectory:
            # Try last trajectory first
            last_trajectory = trajectory[-1]
            query = last_trajectory.get("query", [])
            for entry in query:
                if entry.get("role") == "assistant" and "action" in entry:
                    actions.append(entry["action"])
            
            # If no actions found in last trajectory, try second-to-last
            if not actions and len(trajectory) > 1:
                second_last_trajectory = trajectory[-2]
                query = second_last_trajectory.get("query", [])
                for entry in query:
                    if entry.get("role") == "assistant" and "action" in entry:
                        actions.append(entry["action"])
    except (KeyError, IndexError) as e:
        print(f"Error extracting actions: {e}")
    return actions

def find_first_difference_index(actions1: List[str], actions2: List[str]) -> Optional[int]:
    """
    Find the index of the first different action between two action sequences.
    Returns None if sequences are identical or one is a prefix of the other.
    """
    min_length = min(len(actions1), len(actions2))
    
    for i in range(min_length):
        if actions1[i] != actions2[i]:
            return i
    
    # If we've reached here, one sequence is a prefix of the other
    # Return the length of the shorter sequence if they differ in length
    if len(actions1) != len(actions2):
        return min_length
    
    # Sequences are identical
    return None

def collect_bug_actions(directories: List[str]) -> Dict[str, List[str]]:
    """
    Collect actions for each bug ID from the given directories.
    Returns a dictionary mapping bug_id to list of actions.
    """
    bug_actions = {}
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
        
        # Get all top-level subdirectories
        subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
        
        for subdir in subdirs:
            bug_id = subdir.name
            
            # Skip if we already have this bug ID
            if bug_id in bug_actions:
                continue
            
            # Find .traj files in the subdirectory
            traj_files = list(subdir.glob("*.traj"))
            
            if not traj_files:
                continue
            
            if len(traj_files) > 1:
                print(f"  Warning: Multiple .traj files found in {subdir}, using first one")
            
            traj_file = traj_files[0]
            
            try:
                traj_data = read_trajectory_file(traj_file)
                actions = extract_assistant_actions(traj_data)
                if actions:
                    bug_actions[bug_id] = actions
            except Exception as e:
                print(f"  Error processing {traj_file}: {e}")
    
    return bug_actions

def compare_directory_sets(dirs_set1: List[str], dirs_set2: List[str]) -> None:
    """Compare two sets of directories and analyze differences."""
    print("Collecting actions from first set of directories...")
    bug_actions_set1 = collect_bug_actions(dirs_set1)
    print(f"Found {len(bug_actions_set1)} bugs with actions in set 1")
    
    print("\nCollecting actions from second set of directories...")
    bug_actions_set2 = collect_bug_actions(dirs_set2)
    print(f"Found {len(bug_actions_set2)} bugs with actions in set 2")
    
    # Find common bug IDs
    common_bug_ids = set(bug_actions_set1.keys()) & set(bug_actions_set2.keys())
    print(f"\nFound {len(common_bug_ids)} common bug IDs between the two sets")
    
    if not common_bug_ids:
        print("No common bug IDs found between the two sets!")
        return
    
    # Analyze differences
    difference_indices = []
    identical_trajectories = 0
    one_is_prefix = 0
    
    # Detailed analysis per bug
    bug_differences = {}
    
    for bug_id in common_bug_ids:
        actions1 = bug_actions_set1[bug_id]
        actions2 = bug_actions_set2[bug_id]
        
        diff_index = find_first_difference_index(actions1, actions2)
        
        if diff_index is None:
            identical_trajectories += 1
        else:
            difference_indices.append(diff_index)
            if diff_index == min(len(actions1), len(actions2)):
                one_is_prefix += 1
            
            bug_differences[bug_id] = {
                'diff_index': diff_index,
                'len_actions1': len(actions1),
                'len_actions2': len(actions2),
                'is_prefix': diff_index == min(len(actions1), len(actions2))
            }
    
    # Print results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\nSummary:")
    print(f"Total common bugs: {len(common_bug_ids)}")
    print(f"Identical trajectories: {identical_trajectories} ({identical_trajectories/len(common_bug_ids)*100:.1f}%)")
    print(f"Different trajectories: {len(difference_indices)} ({len(difference_indices)/len(common_bug_ids)*100:.1f}%)")
    print(f"One trajectory is prefix of other: {one_is_prefix}")
    
    if difference_indices:
        print(f"\nStatistics on first difference index:")
        print(f"Min index: {min(difference_indices)}")
        print(f"Max index: {max(difference_indices)}")
        print(f"Average index: {sum(difference_indices)/len(difference_indices):.2f}")
        print(f"Median index: {sorted(difference_indices)[len(difference_indices)//2]}")
        
        # Histogram of difference indices
        print(f"\nHistogram of first difference indices:")
        index_buckets = defaultdict(int)
        bucket_size = 5
        
        for idx in difference_indices:
            bucket = (idx // bucket_size) * bucket_size
            index_buckets[bucket] += 1
        
        # Sort buckets and print
        sorted_buckets = sorted(index_buckets.items())
        for bucket_start, count in sorted_buckets:
            bucket_end = bucket_start + bucket_size - 1
            bar = '█' * min(count, 50)
            print(f"  [{bucket_start:3d}-{bucket_end:3d}]: {count:4d} bugs {bar}")
        
        # Bugs that differ at the very beginning
        early_diffs = [(bug_id, info) for bug_id, info in bug_differences.items() 
                       if info['diff_index'] <= 2]
        print(f"\nBugs that differ in first 3 actions: {len(early_diffs)}")
        if early_diffs:
            print("Examples:")
            for i, (bug_id, info) in enumerate(early_diffs[:10]):
                print(f"  {bug_id}: differs at index {info['diff_index']} " +
                      f"(lengths: {info['len_actions1']} vs {info['len_actions2']})")
        
        # Bugs that differ late
        late_diffs = [(bug_id, info['diff_index']) for bug_id, info in bug_differences.items() 
                      if info['diff_index'] >= 50]
        late_diffs.sort(key=lambda x: x[1], reverse=True)
        print(f"\nBugs that differ after 50+ actions: {len(late_diffs)}")
        if late_diffs:
            print("Top 10 latest differences:")
            for i, (bug_id, diff_idx) in enumerate(late_diffs[:10]):
                info = bug_differences[bug_id]
                print(f"  {bug_id}: differs at index {diff_idx} " +
                      f"(lengths: {info['len_actions1']} vs {info['len_actions2']})")
    
    # Unique bugs in each set
    unique_set1 = set(bug_actions_set1.keys()) - set(bug_actions_set2.keys())
    unique_set2 = set(bug_actions_set2.keys()) - set(bug_actions_set1.keys())
    
    print(f"\nUnique bugs in set 1 only: {len(unique_set1)}")
    print(f"Unique bugs in set 2 only: {len(unique_set2)}")
    
    # Distribution of action counts
    if difference_indices:
        print(f"\nDistribution of difference indices by decile:")
        sorted_indices = sorted(difference_indices)
        n = len(sorted_indices)
        for i in range(0, 10):
            start_pct = i * 10
            end_pct = (i + 1) * 10
            start_idx = int(n * start_pct / 100)
            end_idx = int(n * end_pct / 100) - 1
            if end_idx >= n:
                end_idx = n - 1
            if start_idx < n and end_idx < n:
                print(f"  {start_pct:2d}%-{end_pct:2d}%: {sorted_indices[start_idx]:3d} - {sorted_indices[end_idx]:3d}")

def main():
    parser = argparse.ArgumentParser(
        description='Compare trajectory files between two sets of directories.'
    )
    parser.add_argument(
        '--set1', 
        nargs='+',
        required=False,
        help='First set of directories to analyze'
    )
    parser.add_argument(
        '--set2',
        nargs='+', 
        required=False,
        help='Second set of directories to analyze'
    )
    
    args = parser.parse_args()
    base_path = "/home/shared/swe-agent_logs/shramana"
    args.set1 = [
        # f"{base_path}/20251010_204520_openai/Qwen3",
        # f"{base_path}/20251010_211809_openai/Qwen3",
        # f"{base_path}/20251010_215553_openai/Qwen3",
        # f"{base_path}/20251010_224248_openai/Qwen3",
        # f"{base_path}/20251011_003931_openai/Qwen3",
        # f"{base_path}/20251011_012852_openai/Qwen3",
        # f"{base_path}/20251011_015340_openai/Qwen3"
        "/home/shared/swe-agent_logs/shramana/20251013_162037_openai/Qwen3"
    ]

    args.set2 = [
         f"/home/shared/swe-agent_logs/shramana/20251013_164506_openai/Qwen3"
    ]
    
    print("Comparing trajectory sets...")
    print(f"Set 1: {len(args.set1)} directories")
    for d in args.set1:
        print(f"  - {d}")
    print(f"\nSet 2: {len(args.set2)} directories")
    for d in args.set2:
        print(f"  - {d}")
    
    compare_directory_sets(args.set1, args.set2)


if __name__ == "__main__":
    main()