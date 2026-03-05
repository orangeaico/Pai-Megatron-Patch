#!/usr/bin/env python3
"""
Script to analyze trajectory JSON files and generate statistics about repeated actions.
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
import glob

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

def detect_loops(actions: List[str]) -> Tuple[int, List[str], bool]:
    """
    Detect loops in action sequences.
    Returns the maximum loop length found, the loop pattern, and whether it's a terminal loop.
    """
    if not actions:
        return 0, [], False
    
    max_loop_length = 0
    loop_pattern = []
    is_terminal = False
    
    # Check for single element loops (same action repeating)
    for i in range(len(actions) - 1):
        if actions[i] == actions[i + 1]:
            # Count consecutive repetitions
            count = 2
            j = i + 2
            while j < len(actions) and actions[j] == actions[i]:
                count += 1
                j += 1
            if count > max_loop_length:
                max_loop_length = count
                loop_pattern = [actions[i]]
                # Check if this loop extends to the end
                is_terminal = (j == len(actions))
    
    # Check for two-element loops (alternating actions)
    for i in range(len(actions) - 3):
        if (actions[i] == actions[i + 2] and 
            actions[i + 1] == actions[i + 3] and 
            actions[i] != actions[i + 1]):
            # Count pattern repetitions
            pattern = [actions[i], actions[i + 1]]
            count = 2
            j = i + 4
            while j + 1 < len(actions) and actions[j] == pattern[0] and actions[j + 1] == pattern[1]:
                count += 1
                j += 2
            loop_length = count * 2  # Total actions in the loop
            if loop_length > max_loop_length:
                max_loop_length = loop_length
                loop_pattern = pattern
                # Check if this loop extends to the end
                is_terminal = (j >= len(actions) - 1)
    
    return max_loop_length, loop_pattern, is_terminal

def analyze_directories(directories: List[str]) -> None:
    """Analyze all trajectory files in the given directories."""
    all_bug_stats = {}
    all_actions = []
    action_counter = Counter()
    
    total_subdirs = 0
    total_traj_files = 0
    bugs_with_actions = 0
    bugs_without_actions = 0
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory {directory} does not exist, skipping...")
            continue
            
        print(f"\nProcessing directory: {directory}")
        
        # Get all top-level subdirectories
        subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
        print(f"  Found {len(subdirs)} subdirectories")
        total_subdirs += len(subdirs)
        
        dir_traj_count = 0
        dir_bugs_with_actions = 0
        
        for subdir in subdirs:
            bug_id = subdir.name
            
            # Find .traj files in the subdirectory
            traj_files = list(subdir.glob("*.traj"))
            
            if not traj_files:
                print(f"  No .traj file in {bug_id}")
                continue
            
            dir_traj_count += 1
            total_traj_files += 1
            
            if len(traj_files) > 1:
                print(f"  Warning: Multiple .traj files found in {subdir}, using first one")
            
            traj_file = traj_files[0]
            
            try:
                traj_data = read_trajectory_file(traj_file)
                actions = extract_assistant_actions(traj_data)
                
                if actions:
                    all_actions.extend(actions)
                    for action in actions:
                        action_counter[action] += 1
                    
                    loop_length, loop_pattern, is_terminal = detect_loops(actions)
                    unique_actions = len(set(actions))
                    repeated_actions = len(actions) - unique_actions
                    
                    all_bug_stats[bug_id] = {
                        'total_actions': len(actions),
                        'unique_actions': unique_actions,
                        'repeated_actions': repeated_actions,
                        'loop_length': loop_length,
                        'loop_pattern': loop_pattern,
                        'is_terminal_loop': is_terminal,
                        'actions': actions
                    }
                    
                    dir_bugs_with_actions += 1
                    bugs_with_actions += 1
                else:
                    bugs_without_actions += 1
                    print(f"  No actions found in {bug_id} - Path: {traj_file}")
                    
            except Exception as e:
                print(f"  Error processing {traj_file}: {e}")
                bugs_without_actions += 1
        
        print(f"  Processed {dir_traj_count} .traj files, {dir_bugs_with_actions} with actions")
    
    # Print statistics
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    # Summary of processing
    print(f"\nProcessing Summary:")
    print(f"Total subdirectories found: {total_subdirs}")
    print(f"Total .traj files found: {total_traj_files}")
    print(f"Bugs with actions: {bugs_with_actions}")
    print(f"Bugs without actions or errors: {bugs_without_actions}")
    
    # Overall statistics
    total_bugs = len(all_bug_stats)
    total_actions = len(all_actions)
    total_unique_actions = len(set(all_actions))
    total_repeated_actions = total_actions - total_unique_actions
    
    # Count bugs with 99 or more actions
    bugs_with_99_plus_actions = sum(1 for stats in all_bug_stats.values() if stats['total_actions'] >= 99)
    
    print(f"\nOverall Statistics:")
    print(f"Total bugs analyzed: {total_bugs}")
    print(f"Total actions across all bugs: {total_actions}")
    print(f"Total unique actions: {total_unique_actions}")
    print(f"Total repeated actions: {total_repeated_actions}")
    if total_actions > 0:
        print(f"Repetition rate: {total_repeated_actions/total_actions*100:.2f}%")
    print(f"Bugs with 99+ actions: {bugs_with_99_plus_actions} ({bugs_with_99_plus_actions/total_bugs*100:.1f}%)")
    
    # Most common actions
    print(f"\nTop 10 most common actions:")
    for action, count in action_counter.most_common(10):
        # Truncate long actions for display
        display_action = action if len(action) <= 80 else action[:77] + "..."
        print(f"  {display_action}: {count} occurrences")
    
    # Bugs with loops
    bugs_with_loops = [(bug_id, stats) for bug_id, stats in all_bug_stats.items() 
                       if stats['loop_length'] > 0]
    bugs_with_loops.sort(key=lambda x: x[1]['loop_length'], reverse=True)
    
    print(f"\nBugs with action loops (sorted by loop length):")
    print(f"Total bugs with loops: {len(bugs_with_loops)} out of {total_bugs} ({len(bugs_with_loops)/total_bugs*100:.1f}%)")
    
    if bugs_with_loops:
        for i, (bug_id, stats) in enumerate(bugs_with_loops):
            # Truncate long actions in pattern
            truncated_pattern = []
            for action in stats['loop_pattern']:
                truncated_action = action if len(action) <= 60 else action[:57] + "..."
                truncated_pattern.append(truncated_action)
            pattern_str = ' -> '.join(truncated_pattern) if len(truncated_pattern) > 1 else truncated_pattern[0]
            print(f"  {i+1}. {bug_id}: Loop length {stats['loop_length']} (pattern: {pattern_str})")
    else:
        print("  No bugs with loops detected.")
    
    # Histogram of repeated actions with buckets
    print(f"\nHistogram of bugs by number of repeated actions:")
    buckets = {
        '0': 0,
        '1-5': 0,
        '6-10': 0,
        '11-20': 0,
        '21-30': 0,
        '31-50': 0,
        '50+': 0
    }
    
    for bug_id, stats in all_bug_stats.items():
        repeated_actions = stats['repeated_actions']
        if repeated_actions == 0:
            buckets['0'] += 1
        elif repeated_actions <= 5:
            buckets['1-5'] += 1
        elif repeated_actions <= 10:
            buckets['6-10'] += 1
        elif repeated_actions <= 20:
            buckets['11-20'] += 1
        elif repeated_actions <= 30:
            buckets['21-30'] += 1
        elif repeated_actions <= 50:
            buckets['31-50'] += 1
        else:
            buckets['50+'] += 1
    
    # Print buckets in order
    bucket_order = ['0', '1-5', '6-10', '11-20', '21-30', '31-50', '50+']
    for bucket in bucket_order:
        count = buckets[bucket]
        bar = '█' * min(count, 50)
        print(f"  {bucket:>6} repeated actions: {count:4d} bugs {bar}")
    
    # Bugs with highest repetition rate
    print(f"\nTop 50 bugs by repetition rate:")
    bugs_with_rate = [(bug_id, stats['repeated_actions'] / stats['total_actions'] * 100 if stats['total_actions'] > 0 else 0) 
                     for bug_id, stats in all_bug_stats.items() if stats['total_actions'] > 0]
    bugs_with_rate.sort(key=lambda x: x[1], reverse=True)
    
    for i, (bug_id, rate) in enumerate(bugs_with_rate[:50]):
        stats = all_bug_stats[bug_id]
        print(f"  {i+1}. {bug_id}: {rate:.2f}% ({stats['repeated_actions']}/{stats['total_actions']} actions)")
    
    # Bugs with terminal loops
    print(f"\nBugs with terminal loops (loops that continue until end of trajectory):")
    terminal_loop_bugs = [(bug_id, stats) for bug_id, stats in all_bug_stats.items() 
                          if stats['is_terminal_loop'] and stats['loop_length'] > 0]
    terminal_loop_bugs.sort(key=lambda x: x[1]['loop_length'], reverse=True)
    
    print(f"Total bugs with terminal loops: {len(terminal_loop_bugs)}")
    
    if terminal_loop_bugs:
        print(f"\nDetailed list of bugs with terminal loops:")
        for i, (bug_id, stats) in enumerate(terminal_loop_bugs):
            # Truncate long actions in pattern
            truncated_pattern = []
            for action in stats['loop_pattern']:
                truncated_action = action if len(action) <= 60 else action[:57] + "..."
                truncated_pattern.append(truncated_action)
            pattern_str = ' -> '.join(truncated_pattern) if len(truncated_pattern) > 1 else truncated_pattern[0]
            
            print(f"  {i+1}. {bug_id}: Loop length {stats['loop_length']}, Total actions {stats['total_actions']}")
            print(f"      Pattern: {pattern_str}")


def main():
    # Static list of directories to analyze
    base_path = "/home/shared/swe-agent_logs/shramana"
    directories = [
        f"{base_path}/20251010_204520_openai/Qwen3",
        f"{base_path}/20251010_211809_openai/Qwen3",
        f"{base_path}/20251010_215553_openai/Qwen3",
        f"{base_path}/20251010_224248_openai/Qwen3",
        f"{base_path}/20251011_003931_openai/Qwen3",
        f"{base_path}/20251011_012852_openai/Qwen3",
        f"{base_path}/20251011_015340_openai/Qwen3"
    ]
    # directories = [
    #     f"{base_path}/20251013_162037_openai/Qwen3",
    # ]
    
    print("Analyzing trajectory files for action loops...")
    print(f"Directories to analyze: {len(directories)}")
    
    # Check which directories exist
    existing_dirs = []
    for d in directories:
        if os.path.exists(d):
            existing_dirs.append(d)
            subdirs_count = len([x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))])
            print(f"  ✓ {d} ({subdirs_count} subdirectories)")
        else:
            print(f"  ✗ {d} (NOT FOUND)")
    
    print(f"\nFound {len(existing_dirs)} existing directories")
    
    analyze_directories(directories)


if __name__ == "__main__":
    main()