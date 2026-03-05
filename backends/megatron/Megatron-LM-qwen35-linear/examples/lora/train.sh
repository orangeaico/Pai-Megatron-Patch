
export TOKENIZERS_PARALLELISM=false

accelerate launch --config_file /home/himanshu/megatron_dir/Megatron-LM/examples/lora/accelerate_config.yaml \
/home/himanshu/megatron_dir/Megatron-LM/examples/lora/masked_lora_hf.py \
  --model_name /home/shared/megatron_dir/hf_models/Qwen3-1.7B \
  --train_file /home/shared/megatron_dir/data/sft/train_data_sft_480b_375_swe_bench_shuf.jsonl \
  --eval_file /home/shared/megatron_dir/data/sft/train_data_sft_480b_375_swe_bench_shuf_eval.jsonl \
  --output_dir /home/shared/megatron_dir/output/himanshu/qwen3coder30b-lora \
  --max_seq_len 32768 \
  --per_device_train_bs 1 \
  --per_device_eval_bs 1 \
  --grad_accum 8 \
  --epochs 4 \
  --lr 1e-4 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \
  --local_files_only \
  --use_flash_attn \