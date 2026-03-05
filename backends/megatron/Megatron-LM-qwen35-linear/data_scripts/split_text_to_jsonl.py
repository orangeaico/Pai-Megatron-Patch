#!/usr/bin/env python3
import argparse
import json
import os

def split_text_to_jsonl(input_file, separator="---CHUNK_END---", output_file=None):
    """
    Split a text file by separator and save as JSONL format.
    
    Args:
        input_file: Path to input text file
        separator: String to split the text by (default: "---CHUNK_END---")
        output_file: Path to output JSONL file (default: same as input with .jsonl extension)
    """
    # Determine output file path
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = base_name + ".jsonl"
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator
    chunks = content.split(separator)
    
    # Write to JSONL format
    with open(output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            # Strip whitespace and skip empty chunks
            chunk = chunk.strip()
            if chunk:
                json_obj = {"text": chunk}
                f.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    
    print(f"Processed {len([c for c in chunks if c.strip()])} chunks")
    print(f"Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Split text file by separator and save as JSONL')
    parser.add_argument('--input_file', type=str, 
                        default='/home/shared/qwen3-data-prep/extracted_data/all_chunks_concatenated.txt',
                        help='Path to input text file')
    parser.add_argument('--separator', type=str, 
                        default='---CHUNK_END---',
                        help='Separator string to split text')
    parser.add_argument('--output_file', type=str, 
                        default=None,
                        help='Path to output JSONL file (default: same as input with .jsonl extension)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist")
        return
    
    split_text_to_jsonl(args.input_file, args.separator, args.output_file)

if __name__ == "__main__":
    main()