export TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/data/himanshu/output/

accelerate launch --config_file /workspace/Megatron-LM/examples/lora/accelerate_config_vast.yaml \
/workspace/Megatron-LM/examples/lora/masked_lora_hf.py \
  --model_name Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --train_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf.jsonl \
  --eval_file /workspace/data/data/sft/train_data_sft_480b_375_swe_bench_shuf_eval.jsonl \
  --output_dir /workspace/data/himanshu/output/qwen3coder30b-lora \
  --max_seq_len 65000 \
  --per_device_train_bs 1 \
  --grad_accum 8 \
  --epochs 4 \
  --lr 1e-4 \
  --use_qlora \
  --bf16 \
  --gradient_checkpointing \
  --use_flash_attn \