import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List
from transformers import AutoTokenizer
import glob
import sys
from loss_mask_processors import (
    get_zero_loss_mask_indices as get_loss_mask_indices, 
    set_debug_mode,
    configure_loss_mask_processors
)

IGNORE_INDEX = -100

DATASET_WEIGHT_MULTIPLIER = 1

THOUGHT_WEIGHT = 1

# Tag-based loss mask values (priority: loss_mask > low_value_mask)
TAG_LOSS_MASK_VALUES = {
    "loss_mask": 0,
    "low_value_mask": 0.5,
    "medium_value_mask": 1,
    "high_value_mask": 3,
    "default": 1,
}

LOSS_MASK_TAG_KEYS = ["loss_mask", "low_value_mask", "medium_value_mask", "high_value_mask"]

def count_tokens(tokenizer, conversation_list: List[Dict[str, Any]]) -> int:
    """Count total tokens in a conversation without processing labels/loss_mask."""
    if not isinstance(conversation_list, list):
        raise ValueError(f"The sample must be a list but got {type(conversation_list)}")
    
    total_tokens = 0
    for m in conversation_list:
        msg_dict = {"role": m.get("role", ""), "content": m.get("content", "")}
        seg_ids = tokenizer.apply_chat_template(
            [msg_dict], tokenize=True, add_generation_prompt=False
        )
        total_tokens += len(seg_ids)
    
    return total_tokens


# The get_zero_loss_mask_indices function has been moved to loss_mask_processors.py
# and is imported as get_loss_mask_indices at the top of this file


def process_example_with_tags(tokenizer, conversation_list: List[Dict[str, Any]], stats: Dict[str, int]):
    """Process conversation and create input_ids, labels, and loss_mask based on update_tags."""
    if not isinstance(conversation_list, list):
        raise ValueError(f"The sample must be a list but got {type(conversation_list)}")

    input_ids = []
    labels = []
    loss_mask = []

    assistant_turn = 0
    # Tokenize message-by-message using the chat template
    for i, m in enumerate(conversation_list):
        # Extract fields
        role = m.get("role", "")
        content = m.get("content", "")
        thought = m.get("thought", "")
        update_tags = m.get("update_tags", [])
        tags = m.get("tags", [])
        
        # Create sub-dictionary with only role and content
        msg_dict = {"role": role, "content": content}
        
        seg_ids = tokenizer.apply_chat_template(
            [msg_dict], tokenize=True, add_generation_prompt=False
        )

        input_ids.extend(seg_ids)
        
        if role != "assistant":
            # Non-assistant messages
            labels.extend([IGNORE_INDEX] * len(seg_ids))
            loss_mask.extend([0] * len(seg_ids))
        else:
            assistant_turn += 1
            # Assistant messages
            labels.extend(seg_ids)
            
            # Process tags field for statistics
            if tags:
                if 'all_tags' not in stats:
                    stats['all_tags'] = {}
                for tag in tags:
                    # Handle tags with ## by splitting and using only the first part
                    tag_name = tag.split('##')[0] if '##' in tag else tag
                    stats['all_tags'][tag_name] = stats['all_tags'].get(tag_name, 0) + 1
            
            # Determine loss mask value based on tags (priority: loss_mask > low_value_mask)
            mask_value = TAG_LOSS_MASK_VALUES["default"] * DATASET_WEIGHT_MULTIPLIER  # Default value if no tags
            
            found_loss_mask_key = False
            for key in LOSS_MASK_TAG_KEYS:
                if key in update_tags:
                    mask_value = TAG_LOSS_MASK_VALUES[key] * DATASET_WEIGHT_MULTIPLIER
                    stats[key] = stats.get(key, 0) + 1
                    found_loss_mask_key = True
                    break
            if not found_loss_mask_key:
                mask_value = TAG_LOSS_MASK_VALUES["default"] * DATASET_WEIGHT_MULTIPLIER  # Default value if no tags
                stats["default"] = stats.get('default', 0) + 1
            
            # Create initial loss mask for this assistant turn
            turn_loss_mask = [mask_value] * len(seg_ids)

            if thought:                                
                # Create thought lookup tokens
                thought_dict = {"role": role, "content": thought}
                
                thought_seg_ids = tokenizer.apply_chat_template(
                    [thought_dict], tokenize=True, add_generation_prompt=False
                )
                
                # Remove special tokens at beginning and end. 
                # 3 beginning tokens: <im_start> assistant \n 
                # 2 ending tokens: <im_end> \n
                thought_seg_ids = thought_seg_ids[3:-2]                
                
                # Find thought tokens in original seg_ids
                thought_mask = [0] * len(seg_ids)                
                if thought_seg_ids:
                    # Look for the thought sequence in the original tokens
                    for j in range(len(seg_ids) - len(thought_seg_ids) + 1):
                        if seg_ids[j:j+len(thought_seg_ids)] == thought_seg_ids:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)):
                                thought_mask[j + k] = 1
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 1] == thought_seg_ids[:-1]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-1):
                                thought_mask[j + k] = 1                            
                            break
                        elif seg_ids[j:j+len(thought_seg_ids) - 2] == thought_seg_ids[:-2]:
                            # Mark these positions as thought tokens
                            for k in range(len(thought_seg_ids)-2):
                                thought_mask[j + k] = 1                            
                            break
            
                for thought_index, is_thought in enumerate(thought_mask):
                    if is_thought:
                        turn_loss_mask[thought_index] = turn_loss_mask[thought_index] * THOUGHT_WEIGHT 
                        
            # Get loss mask array from the new loss masking module
            loss_mask_array = get_loss_mask_indices(tokenizer, seg_ids, msg_dict, assistant_turn)
            
            # Track statistics for zero masking
            if loss_mask_array and len(loss_mask_array) == len(seg_ids):
                # Count indices that will be masked to 0
                masked_count = sum(1 for val in loss_mask_array if val == 0)
                
                if masked_count > 0:
                    if 'zero_masked_indices_count' not in stats:
                        stats['zero_masked_indices_count'] = 0
                    if 'zero_masked_messages_count' not in stats:
                        stats['zero_masked_messages_count'] = 0
                    
                    stats['zero_masked_indices_count'] += masked_count
                    stats['zero_masked_messages_count'] += 1
                
                # Apply the mask values from processors
                # Only apply mask values that are not -1 (which means no mask)
                for idx, mask_val in enumerate(loss_mask_array):
                    if mask_val != -1 and 0 <= idx < len(turn_loss_mask):
                        turn_loss_mask[idx] = mask_val * DATASET_WEIGHT_MULTIPLIER
            
            # Extend the overall loss mask
            loss_mask.extend(turn_loss_mask)
            

    assert len(input_ids) == len(labels) == len(loss_mask)

    return input_ids, labels, loss_mask


def extract_trajectory_query(traj_file: str, history_field: str = 'history') -> List[Dict[str, str]]:
    """Extract messages from trajectory file using specified history field."""
    with open(traj_file, 'r') as f:
        data = json.load(f)

    history = data.get(history_field, [])
    if not history:
        raise ValueError(f"No history found in {traj_file}")
    
    # Find the last assistant message index
    last_assistant_idx = -1
    for i in range(len(history) - 1, -1, -1):
        if history[i].get('role') == 'assistant':
            last_assistant_idx = i
            break
    
    if last_assistant_idx == -1:
        raise ValueError(f"No assistant messages found in history in {traj_file}")
    
    # Return all messages up to and including the last assistant message
    messages = history[:last_assistant_idx + 1]
    
    return messages


def main():
    parser = argparse.ArgumentParser(description='Create weighted SFT dataset from SWE-Agent logs')
    parser.add_argument(
        '--input_dir', 
        type=str, 
        default='/home/shared/swe-agent_logs/saurav/20251009_190531_openai/Qwen3',
        help='Input directory containing SWE-Agent logs'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        required=False,
        help='HuggingFace model path for tokenizer (required unless --no-tokenize is used)'
    )
    parser.add_argument(
        '--no-tokenize',
        action='store_true',
        help='Output raw messages without tokenization'
    )
    parser.add_argument(
        '--filter-file',
        type=str,
        default=None,
        help='Text file containing bug names to filter (one per line)'
    )
    parser.add_argument(
        '--filter-len',
        type=int,
        default=64000,
        help='Maximum token length for filtering examples (default: 64000)'
    )
    parser.add_argument(
        '--use-loss-mask-tags',
        action='store_true',
        help='Use update_tags field to determine loss mask values instead of action-based logic'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for loss mask processors'
    )
    parser.add_argument(
        '--loss-mask-processors',
        type=str,
        default=None,
        help='Comma-separated list of loss mask processors to use (e.g., "StrReplaceEditorProcessor,CommandGtProcessor,DiffProcessor"). '
             'Use "all" to enable all available processors. Default is None (no processors applied).'
    )
    parser.add_argument(
        '--history-field',
        type=str,
        default='history',
        help='Name of the field to extract messages from in trajectory files (default: history)'
    )

    args = parser.parse_args()
    
    # Validate that at least one processing mode is selected
    if not (args.no_tokenize or args.use_loss_mask_tags):
        parser.error("At least one of --no-tokenize or --use-loss-mask-tags must be specified")
    
    # Configure loss mask processors
    if args.use_loss_mask_tags:
        # Set debug mode if requested
        if args.debug:
            set_debug_mode(True)
        # Configure which processors to use
        configure_loss_mask_processors(args.loss_mask_processors)
    
    # Setup output file path based on processing mode
    if args.output_file is None:
        input_path = Path(args.input_dir)
        if args.no_tokenize:
            output_filename = "sft_dataset.jsonl"
        elif args.use_loss_mask_tags:
            output_filename = "sft_dataset_loss_mask.jsonl"
        args.output_file = str(input_path / output_filename)

    if args.filter_file is None:
        input_path = Path(args.input_dir)
        args.filter_file = str(input_path / f"resolved_submitted.txt")
    
    print ("=== CONFIGURATION ===")
    print (f"Input Directory: {args.input_dir}")
    print (f"Output File: {args.output_file}")
    print (f"Filter File: {args.filter_file}")
    
    if args.use_loss_mask_tags:
        processor_config = args.loss_mask_processors if args.loss_mask_processors else "None"
        print(f"Loss Mask Processors: {processor_config}")
    
    # Load tokenizer (needed for both modes now due to filtering)
    if not args.model_path:
        parser.error("--model_path is required for token counting")
    print(f"Loading tokenizer from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Load filter list if provided
    filter_bugs = None
    if args.filter_file and os.path.exists(args.filter_file):
        print(f"Loading filter file: {args.filter_file}")
        with open(args.filter_file, 'r') as f:
            filter_bugs = set(line.strip() for line in f if line.strip())
        print(f"Loaded {len(filter_bugs)} bugs to process from filter file")
    else:
        sys.exit(f"ERROR: Filter file not found: {args.filter_file}, exiting..")
    
    print(f"Processing input directory: {args.input_dir}")
    print(f"Writing output to: {args.output_file}")
    print(f"Filter length: {args.filter_len:,} tokens")
    
    # Create overflow output file path
    output_path = Path(args.output_file)
    overflow_file = output_path.parent / f"{output_path.stem}_{args.filter_len}+{output_path.suffix}"
    
    # Process the dataset
    with open(args.output_file, 'w', encoding='utf-8') as outfile, \
         open(overflow_file, 'w', encoding='utf-8') as overflow_outfile:
        
        processed_count = 0
        error_count = 0
        filtered_count = 0
        overflow_count = 0
        
        # Initialize statistics
        stats = {
            'total_input_ids': 0,
            'total_labels': 0,
            'total_loss_mask': 0,
            'loss_mask_distribution': {}  # Dynamic tracking of loss mask values
        }
        
        # Initialize length buckets for input_ids
        length_buckets = {
            '<64000': 0,
            '64000-70000': 0,
            '70000-80000': 0,
            '80000-90000': 0,
            '90000-98000': 0,
            '>=98000': 0
        }
        
        # Per-example statistics list
        example_stats = []
        
        # Scan subdirectories
        for subdir in os.listdir(args.input_dir):
            subdir_path = os.path.join(args.input_dir, subdir)
            
            if not os.path.isdir(subdir_path):
                continue
            
            # Skip if filter is active and bug is not in the filter list
            if filter_bugs is not None and subdir not in filter_bugs:
                continue
            
            # Find trajectory files
            traj_files = glob.glob(os.path.join(subdir_path, '*traj'))
            
            if not traj_files:
                print(f"No trajectory files found in {subdir}")
                continue
            
            # Process the first trajectory file found
            traj_file = traj_files[0]
            
            try:
                # Extract query from trajectory
                messages = extract_trajectory_query(traj_file, args.history_field)
                
                # Count tokens for filtering
                token_count = count_tokens(tokenizer, messages)
                
                if args.no_tokenize:
                    # Output raw messages without tokenization
                    # Filter to only include role and content fields
                    clean_messages = []
                    for msg in messages:
                        clean_msg = {
                            "role": msg.get("role", ""),
                            "content": msg.get("content", "")
                        }
                        clean_messages.append(clean_msg)
                    
                    output_entry = {
                        'messages': clean_messages,
                        'instance_id': subdir
                    }
                    
                    # Print per-example stats
                    print(f"Bug: {subdir} - Token count: {token_count:,}")
                    
                    # Update length buckets
                    if token_count < 64000:
                        length_buckets['<64000'] += 1
                    elif token_count < 70000:
                        length_buckets['64000-70000'] += 1
                    elif token_count < 80000:
                        length_buckets['70000-80000'] += 1
                    elif token_count < 90000:
                        length_buckets['80000-90000'] += 1
                    elif token_count < 98000:
                        length_buckets['90000-98000'] += 1
                    else:
                        length_buckets['>=98000'] += 1
                    
                    # Track example stats
                    example_stats.append({
                        'bug_id': subdir,
                        'input_ids_length': token_count
                    })
                else:
                    # Process the conversation with tokenization
                    if args.use_loss_mask_tags:
                        input_ids, labels, loss_mask = process_example_with_tags(tokenizer, messages, stats)
                    else:
                        # This should not happen due to validation, but keeping for clarity
                        raise ValueError("No valid processing mode selected")
                    
                    # Update statistics
                    stats['total_input_ids'] += len(input_ids)
                    stats['total_labels'] += len(labels)
                    stats['total_loss_mask'] += len(loss_mask)
                    
                    # Count loss mask values (only for assistant tokens, i.e., where labels != IGNORE_INDEX)
                    for i, mask_val in enumerate(loss_mask):
                        if labels[i] != IGNORE_INDEX:  # Only count assistant tokens
                            # Dynamically track all unique loss mask values
                            if mask_val not in stats['loss_mask_distribution']:
                                stats['loss_mask_distribution'][mask_val] = 0
                            stats['loss_mask_distribution'][mask_val] += 1
                    
                    # Track per-example stats
                    example_length = len(input_ids)
                    token_count = example_length  # For tokenize mode, use actual input_ids length
                    example_stats.append({
                        'bug_id': subdir,
                        'input_ids_length': example_length
                    })
                    
                    # Update length buckets
                    if example_length < 64000:
                        length_buckets['<64000'] += 1
                    elif example_length < 70000:
                        length_buckets['64000-70000'] += 1
                    elif example_length < 80000:
                        length_buckets['70000-80000'] += 1
                    elif example_length < 90000:
                        length_buckets['80000-90000'] += 1
                    elif example_length < 98000:
                        length_buckets['90000-98000'] += 1
                    else:
                        length_buckets['>=98000'] += 1
                    
                    # Print per-example stats
                    print(f"Bug: {subdir} - Input IDs length: {example_length:,}")
                    
                    # Create output entry
                    output_entry = {
                        'messages': {
                            'input_ids': input_ids,
                            'labels': labels,
                            'loss_mask': loss_mask
                        },
                        'instance_id': subdir
                    }
                
                # Write to appropriate output file based on token count
                if token_count <= args.filter_len:
                    outfile.write(json.dumps(output_entry) + '\n')
                    filtered_count += 1
                else:
                    overflow_outfile.write(json.dumps(output_entry) + '\n')
                    overflow_count += 1
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count} examples...")         
            except Exception as e:
                error_count += 1
                print(f"Error processing {subdir}/{os.path.basename(traj_file)}: {e}")
                continue
    
    print(f"\nProcessing complete!")
    print(f"Total bugs processed: {processed_count}")
    print(f"Bugs within filter length ({args.filter_len:,}): {filtered_count}")
    print(f"Bugs exceeding filter length: {overflow_count}")
    print(f"Errors encountered: {error_count}")
    print(f"Output written to: {args.output_file}")
    if overflow_count > 0:
        print(f"Overflow output written to: {overflow_file}")
    
    # Print statistics
    if not args.no_tokenize:
        print("\n=== TOKENIZATION STATISTICS ===")
        print(f"Total input_ids: {stats['total_input_ids']:,}")
        print(f"Total labels: {stats['total_labels']:,}")
        print(f"Total loss_mask: {stats['total_loss_mask']:,}")
        
        print("\n=== LOSS MASK DISTRIBUTION (Assistant tokens only) ===")
        # Calculate total assistant tokens from loss_mask_distribution
        if stats['loss_mask_distribution']:
            total_assistant_tokens = sum(stats['loss_mask_distribution'].values())
            if total_assistant_tokens > 0:
                # Sort loss mask values for consistent display
                sorted_mask_values = sorted(stats['loss_mask_distribution'].items())
                for mask_val, count in sorted_mask_values:
                    percentage = count / total_assistant_tokens * 100
                    # Add descriptive labels for known values
                    label = ""
                    if mask_val == 0:
                        label = " (loss_mask)"
                    elif mask_val == 0.5:
                        label = " (low_value_mask)"
                    elif mask_val == 1.0:
                        label = " (medium_value_mask/default)"
                    elif mask_val == 2.0:
                        label = " (high_value_mask)"
                    print(f"Loss mask {mask_val}{label}: {count:,} ({percentage:.1f}%)")
                print(f"Total assistant tokens: {total_assistant_tokens:,}")
        else:
            print("No assistant tokens found")
        
        # Print tag-based statistics if using tag mode
        if args.use_loss_mask_tags:
            print("\n=== TAG-BASED LOSS MASK STATISTICS ===")
            # Print statistics for each tag type found in update_tags
            tag_stats_found = False
            for key in LOSS_MASK_TAG_KEYS:
                if key in stats:
                    print(f"Messages with {key} tag: {stats[key]:,}")
                    tag_stats_found = True
            if 'default' in stats:
                print(f"Messages with default/no loss mask tags: {stats['default']:,}")
                tag_stats_found = True
            if not tag_stats_found:
                print("No tag-based statistics collected")
            
            # Print all tags found
            print("\n=== ALL TAGS FOUND ===")
            if 'all_tags' in stats and stats['all_tags']:
                # Sort tags by frequency (descending)
                sorted_tags = sorted(stats['all_tags'].items(), key=lambda x: x[1], reverse=True)
                for tag, count in sorted_tags:
                    print(f"{tag}: {count:,}")
            else:
                print("No tags found in the 'tags' field of messages")
            
            # Print zero masking statistics
            print("\n=== ZERO LOSS MASKING STATISTICS ===")
            if 'zero_masked_indices_count' in stats:
                print(f"Total indices masked to 0: {stats['zero_masked_indices_count']:,}")
                print(f"Messages with zero masking: {stats['zero_masked_messages_count']:,}")
                if stats['zero_masked_messages_count'] > 0:
                    avg_masked_per_message = stats['zero_masked_indices_count'] / stats['zero_masked_messages_count']
                    print(f"Average indices masked per message: {avg_masked_per_message:.1f}")
            else:
                print("No indices were masked to 0 by get_zero_loss_mask_indices()")
        
    
    print("\n=== TOKEN LENGTH DISTRIBUTION ===")
    for bucket, count in length_buckets.items():
        percentage = (count / processed_count * 100) if processed_count > 0 else 0
        print(f"{bucket}: {count} ({percentage:.1f}%)")
    
    print("\n=== PER-EXAMPLE STATISTICS ===")
    if example_stats:
        # Sort by length for better visualization
        sorted_examples = sorted(example_stats, key=lambda x: x['input_ids_length'], reverse=True)
        
        # Show top 10 longest examples
        print("\nTop 10 longest examples:")
        for i, ex in enumerate(sorted_examples[:10], 1):
            print(f"{i}. Bug: {ex['bug_id']} - Length: {ex['input_ids_length']:,}")
        
        # Calculate average and median
        lengths = [ex['input_ids_length'] for ex in example_stats]
        avg_length = sum(lengths) / len(lengths)
        sorted_lengths = sorted(lengths)
        median_length = sorted_lengths[len(lengths) // 2]
        
        print(f"\nAverage token length: {avg_length:,.1f}")
        print(f"Median token length: {median_length:,}")
        print(f"Min token length: {min(lengths):,}")
        print(f"Max token length: {max(lengths):,}")


if __name__ == "__main__":
    main()