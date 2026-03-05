#!/bin/bash
# Qwen1-7B Megatron Deterministic Inference
# Based on the working Qwen2-0.5B script with adjusted parameters

set -e

echo "🚀 Qwen1-7B Megatron Deterministic Inference"
echo "============================================"

# Set distributed environment variables (required by Megatron)
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="localhost"
export MASTER_PORT="6001"

# Set PYTHONPATH
export PYTHONPATH=/workspace/megatron:$PYTHONPATH

# Navigate to megatron directory
cd /workspace/megatron

MODEL_PATH="/workspace/data/mega-models/Qwen3-1.7B"
TOKENIZER_STATIC_PATH="/workspace/data/mega-models/Qwen3-1.7B"

echo "✓ Environment configured"
echo "✓ Working directory: $(pwd)"
echo "✓ Model path: $MODEL_PATH"

# Verify model exists
if [ ! -f "$MODEL_PATH/config.json" ]; then
    echo "❌ ERROR: Model not found at $MODEL_PATH"
    echo "Please ensure the model is available"
    exit 1
fi

echo "✓ Model found"
echo ""
echo "🧪 Running inference..."

# Run inference with correct arguments for Qwen1-7B
python examples/inference/gpt/gpt_static_inference.py \
  --load $MODEL_PATH \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model $TOKENIZER_STATIC_PATH \
  --vocab-size 151936 \
  --hidden-size 2048 \
  --num-layers 28 \
  --num-attention-heads 16 \
  --ffn-hidden-size 6144 \
  --max-position-embeddings 40960 \
  --seq-length 2048 \
  --normalization RMSNorm \
  --norm-epsilon 1e-06 \
  --use-rotary-position-embeddings \
  --swiglu \
  --rotary-base 1000000 \
  --group-query-attention \
  --num-query-groups 8 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --bf16 \
  --disable-bias-linear \
  --qk-layernorm \
  --use-mcore-models \
  --prompts "<|im_start|>system
You are a helpful assistant<|im_end|>
<|im_start|>user
Who are you?<|im_end|>
<|im_start|>assistant
" \
  --temperature 0.1 \
  --top_k 10 \
  --num-tokens-to-generate 50 \
  --dist-ckpt-strictness ignore_all

echo ""
echo "🎯 Inference completed!"
echo "Check the output above for generated text"