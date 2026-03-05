#!/usr/bin/env python3
import pickle
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np

def format_bytes(size: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(size) < 1024.0:
            return f"{size:3.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"

def analyze_memory_snapshot(pickle_path: str):
    """Analyze PyTorch memory snapshot pickle file"""
    print(f"Loading memory snapshot from: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        snapshot = pickle.load(f)
    
    print("\nSnapshot Keys:", list(snapshot.keys()))
    print("\n" + "="*80)
    
    # Analyze allocator settings first
    if 'allocator_settings' in snapshot:
        print("\n=== ALLOCATOR SETTINGS ===")
        settings = snapshot['allocator_settings']
        if isinstance(settings, dict):
            for key, value in settings.items():
                print(f"  {key}: {value}")
    
    # Analyze segments
    if 'segments' in snapshot:
        print("\n=== MEMORY SEGMENTS ANALYSIS ===")
        segments = snapshot['segments']
        print(f"Total segments: {len(segments)}")
        
        # Collect active, inactive, and active_awaiting_free blocks from all segments
        active_blocks = []
        inactive_blocks = []
        active_awaiting_free_blocks = []
        for seg in segments:
            if isinstance(seg, dict):
                # Check if segment has blocks
                blocks = seg.get('blocks', [])
                for block in blocks:
                    if isinstance(block, dict):
                        # Extract first frame information
                        frames = block.get('frames', [])
                        block_info = {
                            'name': 'N/A',
                            'filename': 'N/A',
                            'size': block.get('size', 0),
                            'requested_size': block.get('requested_size', 0),
                            'address': block.get('address', 0),
                            'state': block.get('state', 'unknown')
                        }
                        
                        if frames and isinstance(frames[0], dict):
                            first_frame = frames[0]
                            block_info['name'] = first_frame.get('name', 'N/A')
                            block_info['filename'] = first_frame.get('filename', 'N/A')
                            block_info['line'] = first_frame.get('line', 0)
                        
                        if block.get('state') == 'active_allocated':
                            active_blocks.append(block_info)
                        elif block.get('state') == 'inactive':
                            inactive_blocks.append(block_info)
                        elif block.get('state') == 'active_awaiting_free':
                            active_awaiting_free_blocks.append(block_info)
        
        # Sort by size in descending order
        active_blocks.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"\n=== BLOCK-LEVEL FRAMES ANALYSIS ===")
        print(f"\n--- TOP 50 ACTIVE ALLOCATED BLOCKS (sorted by size) ---")
        print(f"Total active blocks found: {len(active_blocks)}")
        print(f"\n{'Rank':<6} {'Size':>15} {'Requested Size':>15} {'Frame Name':<40} {'Filename'}")
        print("-" * 120)
        
        for i, block in enumerate(active_blocks[:50], 1):
            print(f"{i:<6} {format_bytes(block['size']):>15} "
                  f"{format_bytes(block['requested_size']):>15} "
                  f"{block['name'][:39]:<40} "
                  f"{block['filename']}")
        
        # Group by unique frame name, filename, and line combination
        frame_totals = defaultdict(lambda: {'size': 0, 'requested_size': 0, 'count': 0})
        
        for block in active_blocks:
            key = (block['name'], block['filename'], block.get('line', 0))
            frame_totals[key]['size'] += block['size']
            frame_totals[key]['requested_size'] += block['requested_size']
            frame_totals[key]['count'] += 1
        
        # Convert to list and sort by total size
        frame_list = []
        for (name, filename, line), totals in frame_totals.items():
            frame_list.append({
                'name': name,
                'filename': filename,
                'line': line,
                'total_size': totals['size'],
                'total_requested_size': totals['requested_size'],
                'count': totals['count']
            })
        
        frame_list.sort(key=lambda x: x['total_size'], reverse=True)
        
        print(f"\n--- ACTIVE BLOCKS - MEMORY USAGE BY FRAME (sorted by total size) ---")
        print(f"Total unique frame combinations: {len(frame_list)}")
        print(f"\n{'Rank':<6} {'Total Size':>15} {'Total Req Size':>15} {'Count':>8} {'Frame Name':<40} {'Filename:Line'}")
        print("-" * 140)
        
        for i, frame in enumerate(frame_list, 1):
            print(f"{i:<6} {format_bytes(frame['total_size']):>15} "
                  f"{format_bytes(frame['total_requested_size']):>15} "
                  f"{frame['count']:>8} "
                  f"{frame['name'][:39]:<40} "
                  f"{frame['filename']}:{frame['line']}")
        
        # Now process inactive blocks
        inactive_blocks.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"\n--- TOP 50 INACTIVE BLOCKS (sorted by size) ---")
        print(f"Total inactive blocks found: {len(inactive_blocks)}")
        print(f"\n{'Rank':<6} {'Size':>15} {'Requested Size':>15} {'Frame Name':<40} {'Filename'}")
        print("-" * 120)
        
        for i, block in enumerate(inactive_blocks[:50], 1):
            print(f"{i:<6} {format_bytes(block['size']):>15} "
                  f"{format_bytes(block['requested_size']):>15} "
                  f"{block['name'][:39]:<40} "
                  f"{block['filename']}")
        
        # Group inactive blocks by unique frame name, filename, and line combination
        inactive_frame_totals = defaultdict(lambda: {'size': 0, 'requested_size': 0, 'count': 0})
        
        for block in inactive_blocks:
            key = (block['name'], block['filename'], block.get('line', 0))
            inactive_frame_totals[key]['size'] += block['size']
            inactive_frame_totals[key]['requested_size'] += block['requested_size']
            inactive_frame_totals[key]['count'] += 1
        
        # Convert to list and sort by total size
        inactive_frame_list = []
        for (name, filename, line), totals in inactive_frame_totals.items():
            inactive_frame_list.append({
                'name': name,
                'filename': filename,
                'line': line,
                'total_size': totals['size'],
                'total_requested_size': totals['requested_size'],
                'count': totals['count']
            })
        
        inactive_frame_list.sort(key=lambda x: x['total_size'], reverse=True)
        
        print(f"\n--- INACTIVE BLOCKS - MEMORY USAGE BY FRAME (sorted by total size) ---")
        print(f"Total unique frame combinations: {len(inactive_frame_list)}")
        print(f"\n{'Rank':<6} {'Total Size':>15} {'Total Req Size':>15} {'Count':>8} {'Frame Name':<40} {'Filename:Line'}")
        print("-" * 140)
        
        for i, frame in enumerate(inactive_frame_list, 1):
            print(f"{i:<6} {format_bytes(frame['total_size']):>15} "
                  f"{format_bytes(frame['total_requested_size']):>15} "
                  f"{frame['count']:>8} "
                  f"{frame['name'][:39]:<40} "
                  f"{frame['filename']}:{frame['line']}")
        
        # Now process active_awaiting_free blocks
        active_awaiting_free_blocks.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"\n--- TOP 50 ACTIVE AWAITING FREE BLOCKS (sorted by size) ---")
        print(f"Total active awaiting free blocks found: {len(active_awaiting_free_blocks)}")
        print(f"\n{'Rank':<6} {'Size':>15} {'Requested Size':>15} {'Frame Name':<40} {'Filename'}")
        print("-" * 120)
        
        for i, block in enumerate(active_awaiting_free_blocks[:50], 1):
            print(f"{i:<6} {format_bytes(block['size']):>15} "
                  f"{format_bytes(block['requested_size']):>15} "
                  f"{block['name'][:39]:<40} "
                  f"{block['filename']}")
        
        # Group active_awaiting_free blocks by unique frame name, filename, and line combination
        awaiting_frame_totals = defaultdict(lambda: {'size': 0, 'requested_size': 0, 'count': 0})
        
        for block in active_awaiting_free_blocks:
            key = (block['name'], block['filename'], block.get('line', 0))
            awaiting_frame_totals[key]['size'] += block['size']
            awaiting_frame_totals[key]['requested_size'] += block['requested_size']
            awaiting_frame_totals[key]['count'] += 1
        
        # Convert to list and sort by total size
        awaiting_frame_list = []
        for (name, filename, line), totals in awaiting_frame_totals.items():
            awaiting_frame_list.append({
                'name': name,
                'filename': filename,
                'line': line,
                'total_size': totals['size'],
                'total_requested_size': totals['requested_size'],
                'count': totals['count']
            })
        
        awaiting_frame_list.sort(key=lambda x: x['total_size'], reverse=True)
        
        print(f"\n--- ACTIVE AWAITING FREE BLOCKS - MEMORY USAGE BY FRAME (sorted by total size) ---")
        print(f"Total unique frame combinations: {len(awaiting_frame_list)}")
        print(f"\n{'Rank':<6} {'Total Size':>15} {'Total Req Size':>15} {'Count':>8} {'Frame Name':<40} {'Filename:Line'}")
        print("-" * 140)
        
        for i, frame in enumerate(awaiting_frame_list, 1):
            print(f"{i:<6} {format_bytes(frame['total_size']):>15} "
                  f"{format_bytes(frame['total_requested_size']):>15} "
                  f"{frame['count']:>8} "
                  f"{frame['name'][:39]:<40} "
                  f"{frame['filename']}:{frame['line']}")
        
        # Group segments by type
        segment_types = defaultdict(list)
        total_allocated = 0
        total_reserved = 0
        
        for seg in segments:
            if isinstance(seg, dict):
                seg_type = seg.get('category', 'unknown')
                segment_types[seg_type].append(seg)
                
                # Track memory sizes
                if 'allocated_size' in seg:
                    total_allocated += seg['allocated_size']
                if 'total_size' in seg:
                    total_reserved += seg['total_size']
        
        print(f"\nTotal Allocated Memory: {format_bytes(total_allocated)}")
        print(f"Total Reserved Memory: {format_bytes(total_reserved)}")
        print(f"Memory Fragmentation: {format_bytes(total_reserved - total_allocated)} ({(total_reserved - total_allocated) / total_reserved * 100:.1f}%)")
        
        print("\nMemory by Category:")
        category_stats = defaultdict(lambda: {'count': 0, 'allocated': 0, 'reserved': 0})
        
        for cat, segs in segment_types.items():
            for seg in segs:
                category_stats[cat]['count'] += 1
                category_stats[cat]['allocated'] += seg.get('allocated_size', 0)
                category_stats[cat]['reserved'] += seg.get('total_size', 0)
        
        # Sort by allocated memory
        sorted_categories = sorted(category_stats.items(), 
                                 key=lambda x: x[1]['allocated'], 
                                 reverse=True)
        
        for cat, stats in sorted_categories:
            print(f"  {cat:20s}: {stats['count']:5d} segments, "
                  f"Allocated: {format_bytes(stats['allocated']):>12s}, "
                  f"Reserved: {format_bytes(stats['reserved']):>12s}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze PyTorch memory snapshot")
    parser.add_argument("pickle_path", help="Path to the memory snapshot pickle file")
    args = parser.parse_args()
    
    analyze_memory_snapshot(args.pickle_path)