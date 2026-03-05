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

# Set these to 1 to enable torch profiling and nvidia nsys profiling
ENABLE_PROFILING=0
ENABLE_NSYS_PROFILING=0

# CRITICAL - DOUBLE CHECK THIS VALUE
TRAINING_MODE="sft" # set from mock, cpt, sft or distillation

MODEL_NAME="Qwen3-Coder-30B-A3B-Instruct"

TIMESTAMP=$(date +"%Y_%m_%d_%H_%M_%S")
BASE_DIR="/workspace/data/"
LOAD_CHECKPOINT_PATH="$BASE_DIR/mega-models/Qwen3-Coder-30B-A3B-Instruct_torch_tp4_ep4"
TOKENIZER_ARG="$BASE_DIR/mega-models/Qwen3-Coder-30B-A3B-Instruct_torch_tp4_ep4" # Path to tokenizer model, or "MOCK"

echo "Training mode: $TRAINING_MODE"

if [[ "$TRAINING_MODE" == "cpt" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/sft/pretraining/xarray_ctx4096_diff_16k_combined_text_document"
    VALID_DATA_PATH="$BASE_DIR/data/sft/pretraining/xarray_validation_ctx4096_text_document"
    TEST_DATA_PATH=$VALID_DATA_PATH

elif [[ "$TRAINING_MODE" == "sft" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/sft/openhands/training_traj_sft_480b_30b_989.jsonl"
    VALID_DATA_PATH="$BASE_DIR/data/sft/openhands/validation_traj_sft_480b_30b_combined_24.jsonl"
    TEST_DATA_PATH=$VALID_DATA_PATH 

elif [[ "$TRAINING_MODE" == "distillation" ]]; then
    TRAIN_DATA_PATH="$BASE_DIR/data/distillation/qwen_480b_swe_bench/"
    VALID_DATA_PATH="$BASE_DIR/data/distillation/qwen_480b_swe_bench_excluded/"
    TEST_DATA_PATH=$VALID_DATA_PATH

elif [[ "$TRAINING_MODE" == "mock" ]]; then
    TRAIN_DATA_PATH="MOCK"
else
    echo "Training mode should be one of mock, cpt, sft or distillation. Invalid training mode: $TRAINING_MODE"
    exit 1
fi

BASE_OUTPUT_DIR="$BASE_DIR/himanshu/output/$TIMESTAMP"
SAVE_CHECKPOINT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/checkpoints"
# Data cache path (useful for both mock and real data)
DATA_CACHE_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/benchmark_cache"
TENSORBOARD_LOGS_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/tensorboard_logs"
MEMORY_SNAPSHOT_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/memory_snapshots/memory_snapshot.pickle"
LOG_DIR_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/logs"
CONVERSION_DIR_PATH="$BASE_OUTPUT_DIR/$MODEL_NAME/conversion"

echo "Timestamp: $TIMESTAMP"
echo "Load checkpoint path: $LOAD_CHECKPOINT_PATH"
echo "Tokenizer path: $TOKENIZER_ARG"
echo "TRAIN DATA PATH: $TRAIN_DATA_PATH"
echo "BASE OUTPUT DIR: $BASE_OUTPUT_DIR"

WANDB_API_KEY=''

# Create directories if they don't exist
mkdir -p "$(dirname "$SAVE_CHECKPOINT_PATH")"
mkdir -p "$(dirname "$TENSORBOARD_LOGS_PATH")"
mkdir -p "$(dirname "$MEMORY_SNAPSHOT_PATH")"
mkdir -p "$DATA_CACHE_PATH"
mkdir -p "$LOG_DIR_PATH"
mkdir -p "$CONVERSION_DIR_PATH"

# Distributed training setup
GPUS_PER_NODE=8
NUM_NODES=1
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Path to the pretrain_gpt.py script, assuming this script is run from the root of the Megatron-LM repository
PRETRAIN_SCRIPT_PATH="pretrain_gpt.py"

# Fixed model and training parameters for Qwen3-1.7B
TP_SIZE=4
CP_SIZE=2
EP_SIZE=4
EXPERT_TP_SIZE=1
PP_SIZE=1
LAYERS_PER_VP=1
MICRO_BATCH_SIZE=1 
GLOBAL_BATCH_SIZE=8
NUM_LAYERS=48  
DTYPE="bf16"
SEQ_LENGTH=65000
MAX_POSITION_EMBEDDINGS=262144 

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
    --ffn-hidden-size 5472 
    --num-attention-heads 32  
    --group-query-attention
    --num-query-groups 4 
    --kv-channels 128 
    --qk-layernorm
    --normalization RMSNorm
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --untie-embeddings-and-output-weights
    --make-vocab-size-divisible-by 1
    --padded-vocab-size 151936  
    --position-embedding-type rope
    --rotary-base 10000000  # Same as Qwen3 rope_theta
    --rotary-percent 1.0
    --rotary-seq-len-interpolation-factor 1
    --swiglu
    --norm-epsilon 1e-06
    --init-method-std 0.02 
    --disable-bias-linear
)

MOE_ARGS=(
    --num-experts 128 
    --moe-ffn-hidden-size 768
    --moe-router-load-balancing-type aux_loss
    --moe-router-topk 8  # num_experts_per_tok
    --moe-grouped-gemm
    --moe-aux-loss-coeff 0  # router_aux_loss_coef from config
    # required even if the model is not moe as there is a assertion check for variable seq length
    --moe-token-dispatcher-type alltoall # flex for --moe-enable-deepep
    --moe-permute-fusion
    --moe-router-dtype fp32
    # --moe-router-fusion # This is only supported in TransformerEngine 2.7.0 and above. Current installed TE is 2.2
    # --moe-router-force-load-balancing
    # --moe-enable-deepep
    # --overlap-moe-expert-parallel-comm
)

TRAINING_ARGS=(
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-samples 3968
    --lr-decay-samples 3968

    # Learning rate args
    --lr-warmup-samples 160
    --lr 1.0e-5 # 5.0e-5
    # --min-lr 1.0e-6 # 5.0e-6
    # --decoupled-lr 8.0e-4  # Adjusted for smaller model
    # --decoupled-min-lr 8.0e-5  # Adjusted for smaller model
    --lr-decay-style cosine
    --adam-beta1 0.9
    --adam-beta2 0.95

    # Regularization args
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --clip-grad 1.0
    --weight-decay 0.01
 
    # Memory cleanup args
    --manual-gc
    --manual-gc-interval 5  

    # Computation optimisation and recomputation args
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
    --sequence-parallel  # Always enable sequence parallelism with TP_SIZE=2
    --context-parallel-size $CP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --expert-tensor-parallel-size $EXPERT_TP_SIZE
    # --pipeline-model-parallel-size $PP_SIZE # Not explicitly set in llama script options, assume 1 if not multi-node PP
    # --num-layers-per-virtual-pipeline-stage $LAYERS_PER_VP  # interleaved PP; needs PP_SIZE>1
)

# Data arguments (conditional for mock vs real data)
DATA_ARGS_LIST=(
    "--vocab-size 151936"  # Qwen3-1.7B vocab size         
    "--no-mmap-bin-files"
    # "--data-cache-path ${DATA_CACHE_PATH}"
    # "--no-check-for-nan-in-loss-and-grad"
)
if [[ "$TRAINING_MODE" == "mock" ]]; then
    DATA_ARGS_LIST+=(
        "--mock-data"
        "--tokenizer-type NullTokenizer"
        "--tiktoken-pattern v2" 
        "--split '99,1,0'"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"                      
    )
elif [[ "$TRAINING_MODE" == "cpt" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"
        # "--trsft"
        # "--trsft-alpha 0.05"
        # "--reset-position-ids"
        # "--reset-attention-mask"
        # "--eod-mask-loss"        
    )
elif [[ "$TRAINING_MODE" == "sft" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"
        # "--data-path $TRAIN_DATA_PATH"
        # "--split '95,5,0'"  
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"               
        "--sft"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"        
        # "--trsft"
        # "--trsft-alpha 0.05"
        "--weighted-loss"
        "--variable-seq-lengths"
        # "--moe-token-dispatcher-type alltoall" # This needs to be set for variable seq lengths

        # "--reset-position-ids"
        # "--reset-attention-mask"
        # "--eod-mask-loss"
    )
elif [[ "$TRAINING_MODE" == "distillation" ]]; then
    # Settings for real data
    DATA_ARGS_LIST+=(
        "--train-data-path $TRAIN_DATA_PATH"
        "--valid-data-path $VALID_DATA_PATH"
        "--test-data-path $TEST_DATA_PATH"        
        "--tokenizer-type HuggingFaceTokenizer" 
        "--tokenizer-model $TOKENIZER_ARG"                
        "--sft"
        "--num-workers 1"
        "--no-create-attention-mask-in-dataloader"         
        "--distillation-loss"
        "--distillation-temperature 2.0"
        "--distillation-loss-alpha 0"      
    )
else
    echo "Training mode should be one of mock, cpt, sft or distillation. Invalid training mode: $TRAINING_MODE"
    exit 1
fi

CHECKPOINT_ARGS=(
    --finetune
    --ckpt-format torch
    --dist-ckpt-strictness log_all
    --distributed-timeout-minutes 60
    --load "$LOAD_CHECKPOINT_PATH"
    --save "$SAVE_CHECKPOINT_PATH"
    --no-save-optim
    --no-save-rng
    --no-load-rng
    --no-load-optim
    --save-interval 124
    --exit-on-missing-checkpoint
    # --ckpt-convert-format torch_dist
    # --ckpt-convert-save /workspace/data/himanshu/output/Qwen3-Coder-30B-A3B-Instruct/conversion/qwen3_30b_a3b_torch_dist/
)

EVAL_AND_LOGGING_ARGS=(
    --eval-iters 3
    --eval-interval 62
    # --full-validation
    --log-interval 1
    --log-throughput
    --log-num-zeros-in-grad
    --log-params-norm
)

if [[ "$ENABLE_PROFILING" == 1 ]]; then
    EVAL_AND_LOGGING_ARGS+=(
        --profile
        --profile-step-start 2
        --profile-step-end 3
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

if [ -n "${WANDB_API_KEY}" ]; then
    LOGGING_ARGS+=(
        --wandb-project ${WANDB_PROJECT:-$MODEL_NAME}
        --wandb-exp-name ${WANDB_NAME:-$MODEL_NAME}
    )
fi

if [[ "$ENABLE_NSYS_PROFILING" == 1 ]]; then
    NSYS_PROFILE_COMMAND="nsys profile -o $LOG_DIR_PATH/nsys_run -t cuda,nvtx,osrt --sample=none --cpuctxsw=none"
else
    NSYS_PROFILE_COMMAND=""
fi

# Ensure pretrain_gpt.py is found
if [ ! -f "$PRETRAIN_SCRIPT_PATH" ]; then
    echo "Error: pretrain_gpt.py not found at $PRETRAIN_SCRIPT_PATH"
    echo "Please ensure you are running this script from the root of the Megatron-LM repository, and pretrain_gpt.py is present."
    exit 1
fi

# Run the training command
$NSYS_PROFILE_COMMAND torchrun ${DISTRIBUTED_ARGS[@]} \
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
