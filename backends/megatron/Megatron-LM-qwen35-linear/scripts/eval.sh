#!/usr/bin/env bash
set -euo pipefail

# download base model from huggingface
BASE_MODEL_PATH="/home/shared/megatron_dir/base-models/Qwen3-1.7B"
# hf download Qwen/Qwen3-1.7B --local-dir $BASE_MODEL_PATH

TUNED_MODEL="/home/shared/megatron_dir/himanshu/output/hf_models_converted/qwen3_1.7b/300_iters"

MODEL=$BASE_MODEL_PATH

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

DTYPE="bfloat16"
MAX_MODEL_LEN=4096 
BATCH_SIZE="auto" # "auto" will use max possible batch size that fits in GPU memory      

TASKS="gsm8k" # "hellaswag,gsm8k,mmlu"
FEWSHOT=0 # no need since we are doing relative comparison
LIMIT=1000
SEED=0  

OUTDIR="eval_runs/$(date +%Y%m%d_%H%M%S)"

echo "Running eval on $MODEL"
echo "GPUs: $CUDA_VISIBLE_DEVICES | dtype=$DTYPE | max_len=$MAX_MODEL_LEN | batch=$BATCH_SIZE"

lm-eval \
  --model vllm \
  --model_args "pretrained=$MODEL,dtype=$DTYPE,max_model_len=$MAX_MODEL_LEN,tensor_parallel_size=2" \
  --gen_kwargs "temperature=0,top_p=0.05,top_k=0" \
  --tasks "$TASKS" \
  --num_fewshot "$FEWSHOT" \
  --batch_size "$BATCH_SIZE" \
  --limit "$LIMIT" \
  --output_path "$OUTDIR" \
  --seed "$SEED"

echo "Done. Results at: $OUTDIR/results.json"