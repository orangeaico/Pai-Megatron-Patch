#!/bin/bash
set -euo pipefail

# Performance / runtime environment defaults.
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_ROOT="${MEGATRON_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$MEGATRON_ROOT"

MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
BASE_DIR="${BASE_DIR:-/workspace/data}"
MODEL_PATH="${MODEL_PATH:-$BASE_DIR/mega-models/Qwen3.5-35B-A3B_torch_dist/torch_dist}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$BASE_DIR/mega-models/Qwen3.5-35B-A3B_torch_tp2_ep8}"

PROMPT=${PROMPT:-"<|im_start|>system
You are Qwen3.5-35B-A3B, a precise coding assistant.<|im_end|>
<|im_start|>user
Write a Python function for binary search with edge-case handling.<|im_end|>
<|im_start|>assistant
"}
NUM_TOKENS_TO_GENERATE="${NUM_TOKENS_TO_GENERATE:-256}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_K="${TOP_K:-0}"
TOP_P="${TOP_P:-0}"
DIST_CKPT_STRICTNESS="${DIST_CKPT_STRICTNESS:-log_all}"

# Distributed setup.
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
NUM_NODES="${NUM_NODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6001}"

# Model parallel setup from train_qwen3.5_35b_a3b.sh defaults.
TP_SIZE="${TP_SIZE:-1}"
PP_SIZE="${PP_SIZE:-1}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-1}"

# Model architecture from train_qwen3.5_35b_a3b.sh.
SEQ_LENGTH="${SEQ_LENGTH:-4096}"
MAX_POSITION_EMBEDDINGS="${MAX_POSITION_EMBEDDINGS:-262144}"
NUM_LAYERS="${NUM_LAYERS:-40}"
HIDDEN_SIZE="${HIDDEN_SIZE:-2048}"
NUM_HEADS="${NUM_HEADS:-16}"
NUM_QUERY_GROUPS="${NUM_QUERY_GROUPS:-2}"
KV_CHANNELS="${KV_CHANNELS:-256}"
PADDED_VOCAB_SIZE="${PADDED_VOCAB_SIZE:-248320}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
MOE_FFN_HIDDEN_SIZE="${MOE_FFN_HIDDEN_SIZE:-512}"
MOE_SHARED_EXPERT_INTERMEDIATE_SIZE="${MOE_SHARED_EXPERT_INTERMEDIATE_SIZE:-512}"
MOE_ROUTER_TOPK="${MOE_ROUTER_TOPK:-8}"

WORLD_SIZE=$((GPUS_PER_NODE * NUM_NODES))
MP_PRODUCT=$((TP_SIZE * PP_SIZE * CP_SIZE * EP_SIZE))
if (( WORLD_SIZE % MP_PRODUCT != 0 )); then
  echo "ERROR: WORLD_SIZE (${WORLD_SIZE}) must be divisible by TP*PP*CP*EP (${MP_PRODUCT})."
  exit 1
fi

if [[ ! -f "$MEGATRON_ROOT/examples/inference/gpt/gpt_static_inference.py" ]]; then
  echo "ERROR: gpt_static_inference.py not found. Set MEGATRON_ROOT or run from Megatron-LM checkout."
  exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "ERROR: model checkpoint path not found: $MODEL_PATH"
  exit 1
fi

if [[ ! -d "$TOKENIZER_PATH" ]]; then
  echo "ERROR: tokenizer path not found: $TOKENIZER_PATH"
  exit 1
fi

echo "Model name: $MODEL_NAME"
echo "Megatron root: $MEGATRON_ROOT"
echo "Model path: $MODEL_PATH"
echo "Tokenizer path: $TOKENIZER_PATH"
echo "World size: $WORLD_SIZE (GPUS_PER_NODE=$GPUS_PER_NODE, NUM_NODES=$NUM_NODES)"

torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NUM_NODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  examples/inference/gpt/gpt_static_inference.py \
  --load "$MODEL_PATH" \
  --auto-detect-ckpt-format \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_PATH" \
  --use-mcore-models \
  --micro-batch-size 1 \
  --bf16 \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$MAX_POSITION_EMBEDDINGS" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --num-attention-heads "$NUM_HEADS" \
  --group-query-attention \
  --num-query-groups "$NUM_QUERY_GROUPS" \
  --kv-channels "$KV_CHANNELS" \
  --qk-layernorm \
  --attention-output-gate \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --normalization RMSNorm \
  --norm-epsilon 1e-06 \
  --swiglu \
  --init-method-std 0.02 \
  --disable-bias-linear \
  --untie-embeddings-and-output-weights \
  --make-vocab-size-divisible-by 1 \
  --padded-vocab-size "$PADDED_VOCAB_SIZE" \
  --position-embedding-type mrope \
  --mrope-section 11 11 10 \
  --rotary-base 10000000 \
  --rotary-percent 0.25 \
  --enable-experimental \
  --experimental-attention-variant gated_delta_net \
  --linear-attention-freq 4 \
  --linear-conv-kernel-dim 4 \
  --linear-num-key-heads 16 \
  --linear-key-head-dim 128 \
  --linear-num-value-heads 32 \
  --linear-value-head-dim 128 \
  --num-experts "$NUM_EXPERTS" \
  --moe-layer-freq 1 \
  --moe-router-topk "$MOE_ROUTER_TOPK" \
  --moe-router-load-balancing-type aux_loss \
  --moe-aux-loss-coeff 0.001 \
  --moe-ffn-hidden-size "$MOE_FFN_HIDDEN_SIZE" \
  --moe-shared-expert-intermediate-size "$MOE_SHARED_EXPERT_INTERMEDIATE_SIZE" \
  --moe-shared-expert-gate \
  --moe-token-dispatcher-type alltoall \
  --moe-grouped-gemm \
  --moe-permute-fusion \
  --moe-router-dtype fp32 \
  --mtp-num-layers 1 \
  --tensor-model-parallel-size "$TP_SIZE" \
  --pipeline-model-parallel-size "$PP_SIZE" \
  --context-parallel-size "$CP_SIZE" \
  --expert-model-parallel-size "$EP_SIZE" \
  --expert-tensor-parallel-size "$EXPERT_TP_SIZE" \
  --sequence-parallel \
  --num-tokens-to-generate "$NUM_TOKENS_TO_GENERATE" \
  --temperature "$TEMPERATURE" \
  --top_k "$TOP_K" \
  --top_p "$TOP_P" \
  --prompts "$PROMPT" \
  --dist-ckpt-strictness "$DIST_CKPT_STRICTNESS" \
  "$@"
