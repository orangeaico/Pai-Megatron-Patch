#!/bin/bash
# Qwen3-Coder-30B-A3B Megatron Inference

set -e

echo "🚀 Qwen3-Coder-30B Megatron Static Inference"
echo "============================================"

# Distributed / runtime environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export MASTER_ADDR="localhost"
export MASTER_PORT="6001"

# Set PYTHONPATH
export PYTHONPATH=/workspace/megatron:$PYTHONPATH

# Navigate to megatron directory
cd /workspace/megatron

MODEL_PATH="/workspace/data/mega-models/Qwen3-Coder-30B-A3B-Instruct"
TOKENIZER_STATIC_PATH="/workspace/data/mega-models/Qwen3-Coder-30B-A3B-Instruct"

echo "✓ Environment configured"
echo "✓ Working directory: $(pwd)"
echo "✓ Model path: $MODEL_PATH"

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model directory not found at $MODEL_PATH"
    exit 1
fi

PROMPT=${PROMPT:-"<|im_start|>system
You are Qwen3-Coder-30B, a precise and efficient coding assistant.<|im_end|>
<|im_start|>user
Explain how to implement a parallel prefix sum in CUDA.<|im_end|>
<|im_start|>assistant
"}

NUM_TOKENS_TO_GENERATE=65536
TEMPERATURE=0.7
TOP_K=0
TOP_P=0
REPETITION_PENALTY=1.05

echo "🧪 Running inference..."

# Distributed setup for 2 GPUs
GPUS_PER_NODE=2
NUM_NODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

torchrun --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=$NUM_NODES \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  examples/inference/gpt/gpt_static_inference.py \
  --load "$MODEL_PATH" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_PATH" \
  --use-mcore-models \
  --seq-length 4096 \
  --max-position-embeddings 262144 \
  --hidden-size 2048 \
  --ffn-hidden-size 5472 \
  --num-layers 48 \
  --num-attention-heads 32 \
  --group-query-attention \
  --num-query-groups 4 \
  --kv-channels 128 \
  --normalization RMSNorm \
  --norm-epsilon 1e-06 \
  --position-embedding-type rope \
  --rotary-base 10000000 \
  --rotary-percent 1.0 \
  --rotary-seq-len-interpolation-factor 1 \
  --swiglu \
  --untie-embeddings-and-output-weights \
  --disable-bias-linear \
  --padded-vocab-size 151936 \
  --tensor-model-parallel-size 2 \
  --pipeline-model-parallel-size 1 \
  --micro-batch-size 1 \
  --no-gradient-accumulation-fusion \
  --empty-unused-memory-level 2 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 2 \
  --bf16 \
  --qk-layernorm \
  --moe-token-dispatcher-type alltoall \
  --moe-router-topk 8 \
  --moe-ffn-hidden-size 768 \
  --num-experts 128 \
  --num-tokens-to-generate "$NUM_TOKENS_TO_GENERATE" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --prompts "$PROMPT" \
  --dist-ckpt-strictness ignore_all \
  "$@"

echo ""
echo "🎯 Inference completed!"
echo "Check the output above for generated text"