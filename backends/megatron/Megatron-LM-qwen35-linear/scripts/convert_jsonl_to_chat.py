#!/usr/bin/env python3
"""
Script to convert JSONL file with chat data to the specified format.
Removes <|im_start|> and <|im_end|> tags and formats conversations.
"""

import json
import sys
import re

def clean_content(content):
    """Remove <|im_start|>, <|im_end|> tags and trailing newlines after <|im_end|>"""
    # Remove <|im_start|> and <|im_end|> tags
    content = content.replace('<|im_start|>', '').replace('<|im_end|>', '')
    # Remove single newline after <|im_end|> pattern (already removed but just in case)
    content = re.sub(r'\n$', '', content)
    return content.strip()

def parse_conversation(text):
    """Parse conversation text into role-content pairs"""
    conversation = []
    
    # Split by role markers (system, user, assistant)
    # Pattern to match role markers at the start of a line
    parts = re.split(r'^(system|user|assistant)\s*\n', text, flags=re.MULTILINE)
    
    # Remove empty first element if exists
    if parts and not parts[0].strip():
        parts = parts[1:]
    
    # Process pairs of (role, content)
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            role = parts[i].strip()
            content = clean_content(parts[i + 1])
            if role in ['system', 'user', 'assistant'] and content:
                conversation.append({'role': role, 'content': content})
    
    return conversation

def convert_jsonl_file(input_path, output_path):
    """Convert JSONL file to the specified chat format"""
    
    converted_count = 0
    error_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse JSON line
                data = json.loads(line.strip())
                
                # Extract text field
                if 'text' not in data:
                    print(f"Warning: Line {line_num} missing 'text' field, skipping")
                    error_count += 1
                    continue
                
                # Clean the text
                text = clean_content(data['text'])
                
                # Parse conversation
                conversation = parse_conversation(text)
                
                if conversation:
                    # Write the conversation wrapped in a dictionary with "conversations" key
                    output_dict = {"conversations": conversation}
                    json.dump(output_dict, outfile)
                    outfile.write('\n')
                    converted_count += 1
                else:
                    print(f"Warning: Line {line_num} produced empty conversation, skipping")
                    error_count += 1
                    
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} - Invalid JSON: {e}")
                error_count += 1
            except Exception as e:
                print(f"Error: Line {line_num} - {type(e).__name__}: {e}")
                error_count += 1
    
    print(f"\nConversion complete!")
    print(f"Successfully converted: {converted_count} conversations")
    print(f"Errors/skipped: {error_count} lines")
    print(f"Output written to: {output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_jsonl_to_chat.py <input_file> [output_file]")
        print("\nIf output_file is not specified, it defaults to <input_file>_converted.jsonl")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.jsonl', '_converted.jsonl')
    
    # Default input path if needed
    if input_file == 'train.jsonl':
        input_file = '/home/shared/megatron_dir/data/train.jsonl'
    
    print(f"Converting: {input_file}")
    print(f"Output to: {output_file}")
    
    convert_jsonl_file(input_file, output_file)

if __name__ == "__main__":
    main()