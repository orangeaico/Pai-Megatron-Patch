from transformers import AutoTokenizer

# Input model directory (unfused) and output directory (fused)
model_id = "/workspace/data/Qwen3-Coder-30B-A3B-Instruct"
save_dir = "/workspace/data/Qwen3-Coder-30B-A3B-Instruct-fused"

# Use file-level conversion to avoid loading the full model into memory
from qwen3_moe_fused.convert import convert_model_to_fused

convert_model_to_fused(model_id, save_dir)

# Copy tokenizer files alongside the converted model
tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tok.save_pretrained(save_dir)