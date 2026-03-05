mkdir -p /workspace/data/himanshu/output/

export TOKENIZERS_PARALLELISM=false
python /workspace/transformers-qwen3-moe-fused/fused_train_30b_a3b_unsloth.py \
  --model_name /workspace/data/Qwen3-Coder-30B-A3B-Instruct-fused \
  --train_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf.jsonl \
  --eval_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf_eval.jsonl \
  --output_dir /workspace/data/himanshu/output/qwen3coder30b-unsloth \
  --max_seq_len 65000 \
  --per_device_train_bs 1 \
  --per_device_eval_bs 1 \
  --grad_accum 8 \
  --epochs 4 \
  --logging_steps 1 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --use_qlora \
  --bf16 \
  --lora_r 32 \
  --lora_alpha 32 \