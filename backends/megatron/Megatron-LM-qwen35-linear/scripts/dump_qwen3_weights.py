#!/usr/bin/env python3
"""
Simple script to load Qwen3-1.7B from local directory and dump its weights to a pickle file.
"""

import torch
import pickle
import os
import argparse
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Dump Qwen3-1.7B weights to pickle file")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Local path to the model directory")
    parser.add_argument("--output-path", type=str, default="qwen3_weights.pkl",
                        help="Output pickle file path")
    
    args = parser.parse_args()
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist!")
        return
    
    # Load the model
    print(f"Loading model from local directory: {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True  # Ensure it only uses local files
    )
    
    # Get state dict
    print("Extracting state dict...")
    state_dict = model.state_dict()
    
    # Convert all tensors to the same format as reference (bfloat16)
    for key in state_dict:
        if state_dict[key].dtype != torch.bfloat16:
            state_dict[key] = state_dict[key].to(torch.bfloat16)
    
    print(f"\nModel has {len(state_dict)} parameters")
    print("Sample parameters:")
    for i, (key, value) in enumerate(list(state_dict.items())[:10]):
        print(f"  {key}: {value.shape} {value.dtype}")
    
    # Save to pickle file
    print(f"\nSaving to {args.output_path}...")
    with open(args.output_path, 'wb') as f:
        pickle.dump(state_dict, f)
    
    # Print file size
    file_size = os.path.getsize(args.output_path) / (1024 * 1024 * 1024)  # GB
    print(f"Saved successfully. File size: {file_size:.2f} GB")

if __name__ == "__main__":
    main()