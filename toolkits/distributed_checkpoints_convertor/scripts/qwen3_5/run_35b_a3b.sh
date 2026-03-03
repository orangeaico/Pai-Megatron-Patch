#!/bin/bash
set -euo pipefail

CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
CONVERTOR_DIR="$( dirname "$( dirname "${CURRENT_DIR}" )" )"
MEGATRON_PATCH_PATH="$( dirname "$( dirname "${CONVERTOR_DIR}" )" )"

# Backend selection:
# 1) explicit MEGATRON_BACKEND_DIR (absolute path or under backends/megatron)
# 2) auto-prefer Jan 20 2026 snapshot if present, otherwise fall back to 250908
if [ -n "${MEGATRON_BACKEND_DIR:-}" ]; then
    if [ -d "${MEGATRON_BACKEND_DIR}" ]; then
        SELECTED_MEGATRON_BACKEND="${MEGATRON_BACKEND_DIR}"
    elif [ -d "${MEGATRON_PATCH_PATH}/backends/megatron/${MEGATRON_BACKEND_DIR}" ]; then
        SELECTED_MEGATRON_BACKEND="${MEGATRON_PATCH_PATH}/backends/megatron/${MEGATRON_BACKEND_DIR}"
    else
        echo "MEGATRON_BACKEND_DIR=${MEGATRON_BACKEND_DIR} was set but not found."
        exit 1
    fi
else
    if [ -d "${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-260120" ]; then
        SELECTED_MEGATRON_BACKEND="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-260120"
    elif [ -d "${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250908" ]; then
        SELECTED_MEGATRON_BACKEND="${MEGATRON_PATCH_PATH}/backends/megatron/Megatron-LM-250908"
    else
        echo "No supported Megatron backend found. Checked Megatron-LM-260120 and Megatron-LM-250908."
        exit 1
    fi
fi

echo "Using Megatron backend: ${SELECTED_MEGATRON_BACKEND}"
export PYTHONPATH="${MEGATRON_PATCH_PATH}:${SELECTED_MEGATRON_BACKEND}:${CONVERTOR_DIR}/impl:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true # for PyTorch >= 2.6

MODEL_SIZE="${1:-A3B}" # reserved for compatibility
LOAD_DIR="${2:?LOAD_DIR is required}"
SAVE_DIR="${3:?SAVE_DIR is required}"
MG2HF="${4:?MG2HF is required}"
USE_CUDA="${5:?USE_CUDA is required}"
PR="${6:?PR is required}"
HF_DIR="${7:-}"

ensure_transformers_version() {
    local want_version="${1:-5.2.0}"
    echo "Installing transformers==${want_version} for this run..."
    python -m pip install -U "transformers==${want_version}"
    python - "${want_version}" <<'PY'
import sys
import transformers

want = sys.argv[1]
got = transformers.__version__
print(f"transformers runtime version: {got}")
if got != want:
    raise SystemExit(f"Expected transformers=={want}, but found {got}")
PY
}

ensure_transformers_version "${TRANSFORMERS_VERSION:-5.2.0}"

# Parallel layout is configurable through env vars:
#   TP_SIZE, PP_SIZE, EP_SIZE, EXPERT_TP_SIZE (or ETP_SIZE)

TP_SIZE="${TP_SIZE:-4}"
PP_SIZE="${PP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-4}"
EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-${ETP_SIZE:-1}}"

for v in "${TP_SIZE}" "${PP_SIZE}" "${EP_SIZE}" "${EXPERT_TP_SIZE}"; do
    if ! [[ "${v}" =~ ^[0-9]+$ ]] || [ "${v}" -le 0 ]; then
        echo "TP_SIZE/PP_SIZE/EP_SIZE/EXPERT_TP_SIZE must be positive integers."
        echo "Got TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, ETP=${EXPERT_TP_SIZE}"
        exit 1
    fi
done

if [ -n "${LAYERS_PER_VP:-}" ] || [ -n "${MP_VP:-}" ] || \
   [ -n "${NUM_LAYERS_PER_VP:-}" ] || [ -n "${NUM_LAYERS_PER_VIRTUAL_PIPELINE_STAGE:-}" ]; then
    echo "LAYERS_PER_VP / virtual pipeline stage layer splitting is not supported."
    exit 1
fi

copy_hf_artifacts() {
    local src="$1"
    local dst="$2"
    mkdir -p "${dst}"
    find -L "${src}" -maxdepth 1 -type f -name "*.json" -print0 | xargs -0 -r cp -t "${dst}"
    find -L "${src}" -maxdepth 1 -type f -name "merges.txt" -print0 | xargs -0 -r cp -t "${dst}"
    find -L "${src}" -maxdepth 1 -type f -name "tokenizer*" -print0 | xargs -0 -r cp -t "${dst}"
    find -L "${src}" -maxdepth 1 -type f -name "vocab.json" -print0 | xargs -0 -r cp -t "${dst}"
}

OTHER_ARGS=()
if [ "${MG2HF}" = true ]; then
    if [ -z "${HF_DIR}" ]; then
        echo "HF_DIR is required when MG2HF=true."
        exit 1
    fi
    CFG_ROOT="${HF_DIR}"
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${HF_DIR}"
        --hf-dir "${HF_DIR}"
        --mcore2hf
    )
    copy_hf_artifacts "${HF_DIR}" "${SAVE_DIR}"
else
    CFG_ROOT="${LOAD_DIR}"
    OTHER_ARGS+=(
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "${LOAD_DIR}"
    )
    copy_hf_artifacts "${LOAD_DIR}" "${SAVE_DIR}"
fi

TRANSFORMER_IMPL="${TRANSFORMER_IMPL:-transformer_engine}"
MOE_GROUPED_GEMM=true
MOE_TOKEN_DISPATCHER_TYPE="alltoall"

if [ "${USE_CUDA}" = true ]; then
    OTHER_ARGS+=(--use-gpu)
else
    export CUDA_VISIBLE_DEVICES=""
    # CPU conversion fallback:
    # - use local transformer implementation (TE requires CUDA)
    # - disable grouped-gemm kernels (CUDA-only)
    # - disable persistent LN (Torch local norm backend does not support it)
    # - use allgather token dispatcher (alltoall path allocates CUDA streams)
    TRANSFORMER_IMPL="local"
    MOE_GROUPED_GEMM=false
    MOE_TOKEN_DISPATCHER_TYPE="allgather"
    OTHER_ARGS+=(--use-cpu-initialization)
    OTHER_ARGS+=(--distributed-backend gloo)
    OTHER_ARGS+=(--no-persist-layer-norm)
    OTHER_ARGS+=(--no-ckpt-fully-parallel-save)
    OTHER_ARGS+=(--ckpt-format torch)
fi

if [ "${PR}" = fp16 ]; then
    OTHER_ARGS+=(--fp16)
elif [ "${PR}" = bf16 ]; then
    OTHER_ARGS+=(--bf16)
fi

if [ -n "${NUM_HF_SAVER:-}" ]; then
    OTHER_ARGS+=(--num-hf-saver "${NUM_HF_SAVER}")
fi
if [ -n "${MAX_SHARD_SIZE:-}" ]; then
    OTHER_ARGS+=(--max-shard-size "${MAX_SHARD_SIZE}")
fi

if [ ! -f "${CFG_ROOT}/config.json" ]; then
    echo "Cannot find config.json under ${CFG_ROOT}"
    exit 1
fi

eval "$(
python - "${CFG_ROOT}/config.json" <<'PY'
import json
import shlex
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

text = cfg.get("text_config", cfg)
layer_types = text.get("layer_types")
if not layer_types:
    raise ValueError("`text_config.layer_types` is required for qwen3.5 conversion.")
if int(text.get("num_hidden_layers", len(layer_types))) != len(layer_types):
    raise ValueError(
        "Mismatch between text_config.num_hidden_layers and len(layer_types): "
        f"{text.get('num_hidden_layers')} vs {len(layer_types)}"
    )

pattern_map = {
    "linear_attention": "M-",
    "full_attention": "*-",
}
hybrid_pattern = "".join(pattern_map[x] for x in layer_types)
mcore_num_layers = 2 * len(layer_types)
if len(hybrid_pattern) != mcore_num_layers:
    raise ValueError(
        "Hybrid override pattern length mismatch: "
        f"len(pattern)={len(hybrid_pattern)} vs mcore_num_layers={mcore_num_layers}"
    )
hybrid_attention_ratio = hybrid_pattern.count("*") / mcore_num_layers
hybrid_mlp_ratio = hybrid_pattern.count("-") / mcore_num_layers

mtp_num = text.get("mtp_num_hidden_layers")
if mtp_num != 1:
    raise ValueError(f"Expected text_config.mtp_num_hidden_layers == 1, got {mtp_num}.")

if text.get("mtp_use_dedicated_embeddings", False):
    raise ValueError("`mtp_use_dedicated_embeddings=true` is not supported in this converter.")

rope_cfg = text.get("rope_parameters", {}) or {}
rope_theta = rope_cfg.get("rope_theta")
if rope_theta is None:
    rope_theta = text.get("rope_theta", 1000000)

rotary_percent = rope_cfg.get("partial_rotary_factor", 1.0)

hidden_size = int(text["hidden_size"])
ffn_hidden_size = text.get("intermediate_size")
if ffn_hidden_size is None or int(ffn_hidden_size) <= 0:
    ffn_hidden_size = hidden_size * 5 // 2
else:
    ffn_hidden_size = int(ffn_hidden_size)

head_dim = text.get("head_dim")
if head_dim is None:
    head_dim = hidden_size // int(text["num_attention_heads"])

mamba_state_dim = int(text.get("linear_key_head_dim", head_dim))
mamba_head_dim = int(text.get("linear_value_head_dim", head_dim))
mamba_num_groups = int(text.get("linear_num_key_heads", text["num_attention_heads"]))
mamba_num_heads = int(text.get("linear_num_value_heads", text["num_attention_heads"]))

exports = {
    # In this converter path each HF layer maps to two MCore layers:
    # attention/mamba stage + MLP stage, so mcore_num_layers = 2 * hf_num_layers.
    "HF_NUM_HIDDEN_LAYERS": int(text["num_hidden_layers"]),
    "NUM_LAYERS": mcore_num_layers,
    "HIDDEN_SIZE": hidden_size,
    "FFN_HIDDEN_SIZE": int(ffn_hidden_size),
    "MOE_FFN_HIDDEN_SIZE": int(text["moe_intermediate_size"]),
    "NUM_ATTENTION_HEADS": int(text["num_attention_heads"]),
    "NUM_QUERY_GROUPS": int(text["num_key_value_heads"]),
    "KV_CHANNELS": int(head_dim),
    "MAX_POSITION_EMBEDDINGS": int(text["max_position_embeddings"]),
    "ROTARY_BASE": int(rope_theta),
    "ROTARY_PERCENT": float(rotary_percent),
    "NUM_EXPERTS": int(text["num_experts"]),
    "ROUTER_TOPK": int(text["num_experts_per_tok"]),
    "MOE_SHARED_EXPERT_SIZE": int(text["shared_expert_intermediate_size"]),
    "PADDED_VOCAB_SIZE": int(text.get("vocab_size", 0)),
    "MAMBA_STATE_DIM": mamba_state_dim,
    "MAMBA_HEAD_DIM": mamba_head_dim,
    "MAMBA_NUM_GROUPS": mamba_num_groups,
    "MAMBA_NUM_HEADS": mamba_num_heads,
    "HYBRID_ATTENTION_RATIO": hybrid_attention_ratio,
    "HYBRID_MLP_RATIO": hybrid_mlp_ratio,
    "HYBRID_OVERRIDE_PATTERN": hybrid_pattern,
}

for k, v in exports.items():
    print(f"export {k}={shlex.quote(str(v))}")
PY
)"

echo "Resolved model params from HF config:"
echo "  hf_num_hidden_layers=${HF_NUM_HIDDEN_LAYERS}"
echo "  mcore_num_layers=${NUM_LAYERS}"
echo "  hidden_size=${HIDDEN_SIZE} heads=${NUM_ATTENTION_HEADS} kv_groups=${NUM_QUERY_GROUPS}"
echo "  moe_experts=${NUM_EXPERTS} topk=${ROUTER_TOPK} moe_hidden=${MOE_FFN_HIDDEN_SIZE}"

NUM_NODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
if [ "${USE_CUDA}" = true ]; then
    readarray -t CUDA_DIAG < <(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
    print(torch.__version__)
    print(torch.version.cuda)
    print(int(torch.cuda.is_available()))
except Exception as e:
    print(0)
    print("IMPORT_ERROR")
    print(repr(e))
    print(0)
PY
)
    DETECTED_GPU_COUNT="${CUDA_DIAG[0]:-0}"
    TORCH_VERSION="${CUDA_DIAG[1]:-unknown}"
    TORCH_CUDA_VERSION="${CUDA_DIAG[2]:-unknown}"
    CUDA_AVAILABLE_FLAG="${CUDA_DIAG[3]:-0}"
    if ! [[ "${DETECTED_GPU_COUNT}" =~ ^[0-9]+$ ]] || [ "${DETECTED_GPU_COUNT}" -le 0 ]; then
        echo "USE_CUDA=true but no visible CUDA devices were detected."
        echo "Torch diagnostics: version=${TORCH_VERSION}, torch.version.cuda=${TORCH_CUDA_VERSION}, torch.cuda.is_available=${CUDA_AVAILABLE_FLAG}"
        echo "Env diagnostics: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}, NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
        if command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi -L output:"
            nvidia-smi -L || true
        else
            echo "nvidia-smi is not available in this container."
        fi
        echo "Please check CUDA_VISIBLE_DEVICES / container GPU allocation."
        echo "If running docker, ensure container is started with --gpus all and NVIDIA runtime."
        exit 1
    fi
    NPROC_PER_NODE="${NPROC_PER_NODE:-${KUBERNETES_CONTAINER_RESOURCE_GPU:-${DETECTED_GPU_COUNT}}}"
    if [ "${NPROC_PER_NODE}" -gt "${DETECTED_GPU_COUNT}" ]; then
        echo "NPROC_PER_NODE (${NPROC_PER_NODE}) exceeds visible GPU count (${DETECTED_GPU_COUNT})."
        echo "Set NPROC_PER_NODE to <= ${DETECTED_GPU_COUNT} (or unset it)."
        exit 1
    fi
else
    NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
fi

TOTAL_PROCESSES="$((NUM_NODES * NPROC_PER_NODE))"
MODEL_DIVISOR="$((TP_SIZE * PP_SIZE))"
EXPERT_DIVISOR="$((EXPERT_TP_SIZE * EP_SIZE * PP_SIZE))"
if [ "${TOTAL_PROCESSES}" -le 0 ] || \
   [ $((TOTAL_PROCESSES % MODEL_DIVISOR)) -ne 0 ] || \
   [ $((TOTAL_PROCESSES % EXPERT_DIVISOR)) -ne 0 ]; then
    echo "Invalid distributed layout for requested parallel sizes."
    echo "  total_processes=${TOTAL_PROCESSES} (nnodes=${NUM_NODES}, nproc_per_node=${NPROC_PER_NODE})"
    echo "  must be divisible by TP*PP=${MODEL_DIVISOR} and ETP*EP*PP=${EXPERT_DIVISOR}"
    echo "  current: TP=${TP_SIZE}, PP=${PP_SIZE}, EP=${EP_SIZE}, ETP=${EXPERT_TP_SIZE}"
    echo "For TP=2,PP=1,EP=2,ETP=1 you need at least NPROC_PER_NODE=2 on a single node."
    exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-6000}"

DISTRIBUTED_ARGS=(
    --nproc_per_node "${NPROC_PER_NODE}"
    --nnodes "${NUM_NODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
)

GPT_MODEL_ARGS=(
    --num-layers "${NUM_LAYERS}"
    --hidden-size "${HIDDEN_SIZE}"
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}"
    --moe-ffn-hidden-size "${MOE_FFN_HIDDEN_SIZE}"
    --normalization RMSNorm
    --swiglu
    --disable-bias-linear
    --num-attention-heads "${NUM_ATTENTION_HEADS}"
    --group-query-attention
    --num-query-groups "${NUM_QUERY_GROUPS}"
    --kv-channels "${KV_CHANNELS}"
    --qk-layernorm
    --seq-length 1
    --max-position-embeddings "${MAX_POSITION_EMBEDDINGS}"
    --attention-backend auto
    --position-embedding-type rope
    --rotary-base "${ROTARY_BASE}"
    --rotary-percent "${ROTARY_PERCENT}"
    --transformer-impl "${TRANSFORMER_IMPL}"
    --untie-embeddings-and-output-weights
    --moe-router-score-function softmax
    --moe-token-dispatcher-type "${MOE_TOKEN_DISPATCHER_TYPE}"
    --moe-router-topk "${ROUTER_TOPK}"
    --num-experts "${NUM_EXPERTS}"
    --moe-shared-expert-intermediate-size "${MOE_SHARED_EXPERT_SIZE}"
    --moe-shared-expert-gate
    --hybrid-attention-ratio "${HYBRID_ATTENTION_RATIO}"
    --hybrid-mlp-ratio "${HYBRID_MLP_RATIO}"
    --hybrid-override-pattern "${HYBRID_OVERRIDE_PATTERN}"
    --is-hybrid-model
    --mamba-state-dim "${MAMBA_STATE_DIM}"
    --mamba-head-dim "${MAMBA_HEAD_DIM}"
    --mamba-num-groups "${MAMBA_NUM_GROUPS}"
    --mamba-num-heads "${MAMBA_NUM_HEADS}"
    --mtp-num-layers 1
)

if [ "${MOE_GROUPED_GEMM}" = true ]; then
    GPT_MODEL_ARGS+=(--moe-grouped-gemm)
fi

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 1024
    --train-iters 500000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --lr 6.0e-5
    --lr-decay-style cosine
    --min-lr 6.0e-6
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size "${TP_SIZE}"
    --pipeline-model-parallel-size "${PP_SIZE}"
    --expert-model-parallel-size "${EP_SIZE}"
    --expert-tensor-parallel-size "${EXPERT_TP_SIZE}"
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
)

CONVERT_ARGS=(
    --model-type GPT
    --load-dir "${LOAD_DIR}"
    --save-dir "${SAVE_DIR}"
    --padded-vocab-size "${PADDED_VOCAB_SIZE}"
    --no-load-optim
    --no-load-rng
    --logging-level 20
    --synchronizer qwen3_5_text_mtp
    --pretrain-script qwen3_5_text_mtp.model_provider
    --debug
)

if [ "${DRYRUN:-false}" = true ]; then
    CONVERT_ARGS+=(--dryrun)
fi

cmd="torchrun ${DISTRIBUTED_ARGS[*]} impl/convert.py \
    ${GPT_MODEL_ARGS[*]} \
    ${TRAINING_ARGS[*]} \
    ${MODEL_PARALLEL_ARGS[*]} \
    ${EVAL_AND_LOGGING_ARGS[*]} \
    ${CONVERT_ARGS[*]} \
    ${OTHER_ARGS[*]}"

echo "${cmd}"
eval "${cmd}"
