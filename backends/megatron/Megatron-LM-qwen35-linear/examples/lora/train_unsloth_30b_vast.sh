mkdir -p /workspace/data/himanshu/output/

python /workspace/Megatron-LM/examples/lora/train_qwen3coder_unsloth.py \
  --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
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