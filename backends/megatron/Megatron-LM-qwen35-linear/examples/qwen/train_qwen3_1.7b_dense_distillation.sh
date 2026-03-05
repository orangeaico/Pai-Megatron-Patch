#!/bin/bash

#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0

MODEL_NAME="qwen3_1.7b"

LOAD_CHECKPOINT_PATH="/workspace/data/mega-models/Qwen3-1.7B"
TOKENIZER_ARG="/workspace/data/mega-models/Qwen3-1.7B" # Path to tokenizer model

JSON_TRAIN_DIR="/workspace/distillation_data"

SAVE_CHECKPOINT_PATH="output/$MODEL_NAME/checkpoints"
# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="output/$MODEL_NAME/benchmark_cache"
TENSORBOARD_LOGS_PATH="output/$MODEL_NAME/tensorboard_logs"
MEMORY_SNAPSHOT_PATH="output/$MODEL_NAME/memory_snapshots/memory_snapshot.pickle"

WANDB_API_KEY=''

# Create directories if they don't exist
mkdir -p "$(dirname "$SAVE_CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "$(dirname "$MEMORY_SNAPSHOT_PATH")"
mkdir -p "$DATA_CACHE_PATH"

# Distributed training setup
GPUS_PER_NODE=2
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters for Qwen3-1.7B
TP_SIZE=2
CP_SIZE=1
PP_SIZE=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=1
NUM_LAYERS=28
DTYPE="bf16"
SEQ_LENGTH=16384 # 65000
MAX_POSITION_EMBEDDINGS=40960 # 65000

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

MODEL_ARGS=(
    --use-mcore-models
    --num-layers $NUM_LAYERS
    --seq-length $SEQ_LENGTH
    --hidden-size 2048  
    --ffn-hidden-size 6144 
    --num-attention-heads 16  
    --group-query-attention
    --num-query-groups 8 
    --kv-channels 128 
    --qk-layernorm
    --normalization RMSNorm
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --make-vocab-size-divisible-by 1187
    --position-embedding-type rope
    --rotary-base 1000000  # Same as Qwen3 rope_theta
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    # --use-rope-scaling
    # --rope-scaling-factor 2
    --swiglu
    --norm-epsilon 1e-06
    --init-method-std 0.02  
    --disable-bias-linear
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 10
    --lr-decay-samples 10
    --exit-duration-in-mins 235

    # Learning rate args
    --lr-warmup-samples 0
    --lr 5.0e-5
    --min-lr 1.0e-7
    # --decoupled-lr 8.0e-4  # Adjusted for smaller model
    # --decoupled-min-lr 8.0e-5  # Adjusted for smaller model
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95

    # Regularization args
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.0
 
    # Memory cleanup args
    --manual-gc
    --manual-gc-interval 5  

    # Computation optimisation and recomputation args
    --transformer-impl transformer_engine
    --enable-experimental
    --use-flash-attn
    --fused-linear-cross-entropy
    # --cross-entropy-loss-fusion
    # --cross-entropy-fusion-impl native
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
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
)

# Conditional arguments based on DTYPE (FP8)
DTYPE_ARGS=()
if [[ "$DTYPE" == "fp8" ]]; then
    DTYPE_ARGS+=(
        "--fp8-format hybrid"
        "--fp8-amax-history-len 1024"
        "--fp8-amax-compute-algo max"
        "--fp8-param-gather"
    )
fi

# Model parallelism arguments
MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TP_SIZE
    --context-parallel-size $CP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
)

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=(
    "--distillation-loss"
    "--distillation-temperature 3.0"
    "--distillation-loss-alpha 0.5"
    "--tokenizer-type HuggingFaceTokenizer"
    "--tokenizer-model $TOKENIZER_ARG"
    "--data-path $JSON_TRAIN_DIR"
    "--data-cache-path ${DATA_CACHE_PATH}"
    "--split '95,5,0'"
    "--num-workers 0"
    "--vocab-size 151936"
)

CHECKPOINT_ARGS=(
    --finetune
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
    --distributed-timeout-minutes 60
    --no-load-optim
    --no-load-rng
    --no-save-optim
    --no-save-rng
    --load "$LOAD_CHECKPOINT_PATH"
    --save "$SAVE_CHECKPOINT_PATH"
    --save-interval 1000
)

EVAL_AND_LOGGING_ARGS=(
    --eval-iters 1
    --eval-interval 100
    # "--full-validation"
    --log-interval 1
    --log-throughput
    --profile
    --profile-step-start 2
    --profile-step-end 3
    --profile-ranks 0
    --use-pytorch-profiler
    --tensorboard-dir "$TENSORBOARD_LOGS_PATH"
    --log-timers-to-tensorboard
    --log-num-zeros-in-grad
    --log-params-norm
    --log-validation-ppl-to-tensorboard
    --log-memory-to-tensorboard
    --record-memory-history
    --memory-snapshot-path "$MEMORY_SNAPSHOT_PATH"
    # --dump-model-params-to-pickle
)

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-$MODEL_NAME}
        --wandb-exp-name ${WANDB_NAME:-$MODEL_NAME}
    )
fi



# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
torchrun ${DISTRIBUTED_ARGS[@]} \
    "$PRETRAIN_SCRIPT_PATH" \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]}

set +x
