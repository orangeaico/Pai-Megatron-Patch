#!/bin/bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Performance / env knobs (same style as your existing script)
# -----------------------------------------------------------------------------
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NVTE_ALLOW_NONDETERMINISTIC_ALGO:-1}
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

ENABLE_PROFILING=${ENABLE_PROFILING:-1}
ENABLE_NSYS_PROFILING=${ENABLE_NSYS_PROFILING:-0}

# -----------------------------------------------------------------------------
# User-editable section
# -----------------------------------------------------------------------------
TRAINING_MODE="sft"    # mock | cpt | sft
MODEL_NAME="Qwen3.5-35B-A3B"
TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")

BASE_DIR="${BASE_DIR:-/workspace/data}"
TOKENIZER_DIR="$BASE_DIR/mega-models/Qwen3.5-35B-A3B_torch_tp2_ep8"   # HF snapshot dir with tokenizer.json etc.
LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-$BASE_DIR/mega-models/Qwen3.5-35B-A3B_torch_dist/torch_dist}"

# Data paths
if [[ "$TRAINING_MODE" == "cpt" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/sft/pretraining/xarray_ctx4096_diff_16k_combined_text_document"
    VALID_DATA_PATH="$BASE_DIR/data/sft/pretraining/xarray_validation_ctx4096_text_document"
    TEST_DATA_PATH=$VALID_DATA_PATH

elif [[ "$TRAINING_MODE" == "sft" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/sft/onehop_tasks_26dec/sft_data/onehop_train_12556.jsonl"
    VALID_DATA_PATH="$BASE_DIR/data/sft/onehop_tasks_26dec/sft_data/onehop_train_12556.jsonl"
    TEST_DATA_PATH=$VALID_DATA_PATH 

elif [[ "$TRAINING_MODE" == "distillation" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/distillation/qwen_480b_swe_bench/"
    VALID_DATA_PATH="$BASE_DIR/data/distillation/qwen_480b_swe_bench_excluded/"
    TEST_DATA_PATH=$VALID_DATA_PATH
else
  echo "TRAINING_MODE must be one of: mock | cpt | sft"
  exit 1
fi

BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-$BASE_DIR/output/$TIMESTAMP}"
SAVE_CHECKPOINT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/checkpoints"
DATA_CACHE_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/data_cache"
TENSORBOARD_LOGS_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/tensorboard"
MEMORY_SNAPSHOT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/memory_snapshots/memory_snapshot.pickle"
LOG_DIR_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/logs"

mkdir -p "$SAVE_CHECKPOINT_PATH" "$DATA_CACHE_PATH" "$TENSORBOARD_LOGS_PATH" "$LOG_DIR_PATH" "$(dirname "$MEMORY_SNAPSHOT_PATH")"

echo "Mode: $TRAINING_MODE"
echo "Tokenizer: $TOKENIZER_DIR"
echo "Load: $LOAD_CHECKPOINT_PATH"
echo "Train: $TRAIN_DATA_PATH"
echo "Out: $BASE_OUTPUT_DIR"

# -----------------------------------------------------------------------------
# Distributed setup
# -----------------------------------------------------------------------------
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}

DISTRIBUTED_ARGS=(
  --nproc_per_node $GPUS_PER_NODE
  --nnodes $NUM_NODES
  --node_rank $NODE_RANK
  --master_addr $MASTER_ADDR
  --master_port $MASTER_PORT
)

PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"
if [[ ! -f "$PRETRAIN_SCRIPT_PATH" ]]; then
  echo "Run this from Megatron-LM repo root (pretrain_gpt.py not found)."
  exit 1
fi

# -----------------------------------------------------------------------------
# Model: Qwen3.5-35B-A3B (text_config) from HF config.json
# -----------------------------------------------------------------------------
# Parallelism (must satisfy: world_size = TP * PP * CP * EP * DP)
TP_SIZE=1
CP_SIZE=1
EP_SIZE=8
PP_SIZE=1
EXPERT_TP_SIZE=1

NUM_LAYERS=40
HIDDEN_SIZE=2048

# Qwen3.5 full-attn params: heads=16, kv_heads=2, head_dim=256
NUM_HEADS=16
NUM_QUERY_GROUPS=2
KV_CHANNELS=256

# Qwen3.5 vocab
VOCAB_SIZE=248320
PADDED_VOCAB_SIZE=248320

# Context
SEQ_LENGTH=${SEQ_LENGTH:-4096}           # practical training length; can raise later
MAX_POSITION_EMBEDDINGS=262144

MODEL_ARGS=(
  --use-mcore-models
  --num-layers $NUM_LAYERS
  --hidden-size $HIDDEN_SIZE
  --seq-length $SEQ_LENGTH
  --max-position-embeddings $MAX_POSITION_EMBEDDINGS

  --num-attention-heads $NUM_HEADS
  --group-query-attention
  --num-query-groups $NUM_QUERY_GROUPS
  --kv-channels $KV_CHANNELS
  --qk-layernorm

  --attention-output-gate
  --attention-dropout 0.0
  --hidden-dropout 0.0

  --normalization RMSNorm
  --norm-epsilon 1e-06
  --swiglu
  --init-method-std 0.02
  --disable-bias-linear
  --untie-embeddings-and-output-weights

  --make-vocab-size-divisible-by 1
  --padded-vocab-size $PADDED_VOCAB_SIZE

  # RoPE / mRoPE (Qwen3.5 rope_theta=1e7, partial_rotary_factor=0.25, mrope_section=[11,11,10])
  --position-embedding-type mrope
  --mrope-section 11 11 10
  --rotary-base 10000000
  --rotary-percent 0.25
)

# Hybrid attention schedule: linear attention + full attention every 4 layers
# (Qwen3.5 config uses full_attention_interval=4)
HYBRID_ATTN_ARGS=(
  --enable-experimental
  --experimental-attention-variant gated_delta_net
  --linear-attention-freq 4

  # linear attention params from HF config
  --linear-conv-kernel-dim 4
  --linear-num-key-heads 16
  --linear-key-head-dim 128
  --linear-num-value-heads 32
  --linear-value-head-dim 128
)

# MoE (Qwen3.5: 256 experts, topk=8, moe_intermediate=512, shared=512, aux=0.001)
MOE_ARGS=(
  --num-experts 256
  --moe-layer-freq 1
  --moe-router-topk 8
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff 0.001
  --moe-ffn-hidden-size 512
  --moe-shared-expert-intermediate-size 512
  --moe-shared-expert-gate

  # Common perf toggles (disable if your TE/deps don’t support them)
  --moe-token-dispatcher-type alltoall
  --moe-grouped-gemm
  --moe-permute-fusion
  --moe-router-dtype fp32
)

# --- MTP (Multi-Token Prediction) ---
# Qwen3.5 config indicates MTP-1 (mtp_num_hidden_layers=1).
# Megatron controls it via --mtp-num-layers and optional --mtp-loss-scaling-factor.
MTP_ARGS=(
  --mtp-num-layers 1
  --mtp-loss-scaling-factor 0.1   # default in Megatron; tune if you want
)

MODEL_PARALLEL_ARGS=(
  --tensor-model-parallel-size $TP_SIZE
  --pipeline-model-parallel-size $PP_SIZE
  --context-parallel-size $CP_SIZE
  --expert-model-parallel-size $EP_SIZE
  --expert-tensor-parallel-size $EXPERT_TP_SIZE
  --sequence-parallel
)

# -----------------------------------------------------------------------------
# Training args (load pretrained weights; do not restore optimizer/RNG)
# -----------------------------------------------------------------------------
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8

TRAINING_ARGS=(
  --micro-batch-size $MICRO_BATCH_SIZE
  --global-batch-size $GLOBAL_BATCH_SIZE

  # pick either train-iters or train-samples; using train-iters is simplest
  --train-iters ${TRAIN_ITERS:-10}

  --lr ${LR:-1.0e-4}
  --min-lr ${MIN_LR:-1.0e-5}
  --lr-decay-style cosine
  --lr-warmup-iters ${WARMUP_ITERS:-1}

  --adam-beta1 0.9
  --adam-beta2 0.95
  --weight-decay 0.01
  --clip-grad 1.0

  # Memory cleanup args
  --manual-gc
  --manual-gc-interval 5 

  --transformer-impl transformer_engine
  --enable-experimental
  --attention-backend flash
  --use-flash-attn
  --fused-linear-cross-entropy
  # --cross-entropy-loss-fusion
  # --cross-entropy-fusion-impl native
  
  # --recompute-granularity full
  # --recompute-method uniform
  # --recompute-num-layers 1
  --calculate-per-token-loss
  # --no-gradient-accumulation-fusion

  # data type arguments
  --bf16
  --use-distributed-optimizer
  --use-precision-aware-optimizer
  --overlap-grad-reduce
  --overlap-param-gather
  --main-params-dtype fp16
  --main-grads-dtype bf16
  --grad-reduce-in-bf16
  --exp-avg-dtype fp16
  --exp-avg-sq-dtype fp16

  --log-interval 10
  --eval-interval 200
  --eval-iters 5

  --save $SAVE_CHECKPOINT_PATH
  --save-interval ${SAVE_INTERVAL:-10}
  --load "$LOAD_CHECKPOINT_PATH"
  --no-save-optim
  --no-save-rng
  --no-load-optim
  --no-load-rng
  --finetune
  --ckpt-format torch
  --auto-detect-ckpt-format
  --dist-ckpt-strictness log_all
  --distributed-timeout-minutes 60
  # --ckpt-convert-format torch_dist
  # --ckpt-convert-save /workspace/data/mega-models/Qwen3.5-35B-A3B_torch_dist/
)

# -----------------------------------------------------------------------------
# Data args (mock / cpt / sft)
# -----------------------------------------------------------------------------
DATA_ARGS_LIST=(
  --vocab-size $VOCAB_SIZE
  --data-cache-path "$DATA_CACHE_PATH"
  --no-create-attention-mask-in-dataloader
  --num-workers ${NUM_WORKERS:-4}
)

if [[ "$TRAINING_MODE" == "mock" ]]; then
  DATA_ARGS_LIST+=(
    --mock-data
    --tokenizer-type NullTokenizer
    --split "99,1,0"
  )
elif [[ "$TRAINING_MODE" == "cpt" ]]; then
  DATA_ARGS_LIST+=(
    --train-data-path "$TRAIN_DATA_PATH"
    --valid-data-path "$VALID_DATA_PATH"
    --test-data-path "$TEST_DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_DIR"
  )
elif [[ "$TRAINING_MODE" == "sft" ]]; then
  DATA_ARGS_LIST+=(
    --train-data-path "$TRAIN_DATA_PATH"
    --valid-data-path "$VALID_DATA_PATH"
    --test-data-path "$TEST_DATA_PATH"
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "$TOKENIZER_DIR"
    --sft
    # --weighted-loss
    --variable-seq-lengths
  )
fi

# -----------------------------------------------------------------------------
# Optional profiling
# -----------------------------------------------------------------------------
EVAL_AND_LOGGING_ARGS=()
if [[ "$ENABLE_PROFILING" == 1 ]]; then
  EVAL_AND_LOGGING_ARGS+=(
    --profile
    --profile-step-start 0
    --profile-step-end 1
    --profile-ranks 0
    --use-pytorch-profiler
    --log-timers-to-tensorboard
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --record-memory-history
    --memory-snapshot-path "$MEMORY_SNAPSHOT_PATH"
    --timing-log-level 2
    --logging-level 10
    --timing-log-option all
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    # --dump-model-params-to-pickle
  )
fi

if [[ "$ENABLE_NSYS_PROFILING" == "1" ]]; then
  NSYS_PROFILE_COMMAND="nsys profile -o $LOG_DIR_PATH/nsys_run -t cuda,nvtx,osrt --sample=none --cpuctxsw=none"
else
  NSYS_PROFILE_COMMAND=""
fi

# -----------------------------------------------------------------------------
# Launch
# -----------------------------------------------------------------------------
set -x
$NSYS_PROFILE_COMMAND torchrun "${DISTRIBUTED_ARGS[@]}" \
  "$PRETRAIN_SCRIPT_PATH" \
  "${MODEL_ARGS[@]}" \
  "${HYBRID_ATTN_ARGS[@]}" \
  "${MOE_ARGS[@]}" \
  "${MTP_ARGS[@]}" \
  "${MODEL_PARALLEL_ARGS[@]}" \
  "${TRAINING_ARGS[@]}" \
  "${DATA_ARGS_LIST[@]}" \
  "${EVAL_AND_LOGGING_ARGS[@]}"
set +x
