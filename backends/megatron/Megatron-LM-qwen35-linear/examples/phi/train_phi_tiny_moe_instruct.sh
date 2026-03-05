#!/bin/bash

#export LOG_LEVEL=${LOG_LEVEL:-INFO}
#export NCCL_IB_TIMEOUT=${NCCL_IB_TIMEOUT:-19}
#export NVTE_FWD_LAYERNORM_SM_MARGIN=${NVTE_FWD_LAYERNORM_SM_MARGIN:-16}
#export NVTE_BWD_LAYERNORM_SM_MARGIN=${NVTE_BWD_LAYERNORM_SM_MARGIN:-16}
#export NCCL_P2P_NET_CHUNKSIZE=${NCCL_P2P_NET_CHUNKSIZE:-2097152}
#export NCCL_AVOID_RECORD_STREAMS=${NCCL_AVOID_RECORD_STREAMS:-1}

# Environment variables for performance tuning
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0

MODEL_NAME="phi_tiny_moe_instruct"

BASE_DIR="/workspace/data/"
LOAD_CHECKPOINT_PATH="$BASE_DIR/mega-models/Qwen3-Coder-30B-A3B-Instruct_cpu_nondist_mem_eff_save"

# TOKENIZER_ARG="$BASE_DIR/mega-models/Qwen3-Coder-30B-A3B-Instruct_cpu_nondist_mem_eff_save" # Path to tokenizer model, or "MOCK"
# DATA_ARG="$BASE_DIR/data/test_output.jsonl"

DATA_ARG="MOCK"
TOKENIZER_ARG="MOCK"

BASE_OUTPUT_DIR="$BASE_DIR/himanshu/output"
SAVE_CHECKPOINT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/checkpoints"
# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/benchmark_cache"
TENSORBOARD_LOGS_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/tensorboard_logs"
MEMORY_SNAPSHOT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/memory_snapshots/memory_snapshot.pickle"

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
EP_SIZE=2
EXPERT_TP_SIZE=1
PP_SIZE=1
LAYERS_PER_VP=1
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=2  
NUM_LAYERS=16  # Actual 32 layers
DTYPE="bf16"
SEQ_LENGTH=8192
MAX_POSITION_EMBEDDINGS=40960 

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
    --hidden-size 4096  
    --ffn-hidden-size 448 
    --num-attention-heads 16  
    --group-query-attention
    --num-query-groups 4 
    --kv-channels 128 
    --qk-layernorm
    --normalization RMSNorm
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --untie-embeddings-and-output-weights
    --position-embedding-type rope
    --rotary-base 1000000  # Same as Qwen3 rope_theta
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    --swiglu
    --norm-epsilon 1e-06
    --init-method-std 0.02 
    --disable-bias-linear
)

MOE_ARGS=(
    --num-experts 16 
    --moe-ffn-hidden-size 448
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 2  # num_experts_per_tok
    --moe-grouped-gemm
    --moe-aux-loss-coeff 1e-3  # router_aux_loss_coef from config
    --moe-token-dispatcher-type alltoall
    --moe-permute-fusion
    # --moe-expert-capacity-factor 1
    # --moe-router-dtype fp32
    # --moe-router-fusion # This is only supported in TransformerEngine 2.7.0 and above. Current installed TE is 2.2
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 300
    --lr-decay-samples 300
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
    # --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --context-parallel-size $CP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $EXPERT_TP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    # --num-layers-per-virtual-pipeline-stage $LAYERS_PER_VP  # interleaved PP; needs PP_SIZE>1
)

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=()
if [[ "$TOKENIZER_ARG" == "MOCK" ]] || [[ "$DATA_ARG" == "MOCK" ]] || [[ -z "$TOKENIZER_ARG" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--vocab-size 32064"  # Qwen3-1.7B vocab size
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
    )
else
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--data-path $DATA_ARG"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--data-cache-path ${DATA_CACHE_PATH}"
        "--split '99,1,0'"
        "--no-create-attention-mask-in-dataloader"
        "--no-mmap-bin-files"
        "--num-workers 1"
        # Note: --vocab-size might be inferred by HuggingFaceTokenizer or might need to be explicit.
        "--vocab-size 32064"  # Qwen3-1.7B vocab size
        # "--sft"
        # "--reset-position-ids"
        # "--reset-attention-mask"
        # "--eod-mask-loss"
        # "--no-check-for-nan-in-loss-and-grad"
    )
fi

CHECKPOINT_ARGS=(
    --finetune
    --auto-detect-ckpt-format
    --dist-ckpt-strictness log_all
    --distributed-timeout-minutes 60
    # --load "$LOAD_CHECKPOINT_PATH"
    --save "$SAVE_CHECKPOINT_PATH"
    --no-save-optim
    --no-save-rng
    --no-load-rng
    --no-load-optim
    --save-interval 50
    --exit-on-missing-checkpoint
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
    ${MOE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${DTYPE_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS_LIST[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${CHECKPOINT_ARGS[@]}

set +x
