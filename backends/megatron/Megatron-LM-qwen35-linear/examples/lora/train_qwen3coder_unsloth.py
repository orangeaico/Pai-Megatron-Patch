# train_unsloth_only_user.py — reference pipeline, but ONLY your dataset
from unsloth import FastLanguageModel
from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig
import pandas as pd
import torch
import os, json
import argparse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="unsloth/Qwen3-1.7B")
    p.add_argument("--train_file", type=str, required=True)
    p.add_argument("--eval_file", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="qwen3coder30b-unsloth-lora")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--per_device_train_bs", type=int, default=1)
    p.add_argument("--per_device_eval_bs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--use_qlora", action="store_true", default=True)
    p.add_argument("--no-use_qlora", dest="use_qlora", action="store_false")
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--weight_decay", type=float, default=0.01)
    return p.parse_args()

args = get_args()

# ---------------- Model & LoRA (UNCHANGED) ----------------
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_len,
    load_in_4bit=args.use_qlora,
    load_in_8bit=not args.use_qlora,
    full_finetuning=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

train_file_list = [args.train_file]
eval_file_list = [args.eval_file]
print(f"[Train dataset] : {train_file_list}")
print(f"[Eval dataset] : {eval_file_list}")

train_data_ds = load_dataset("json", data_files={"train": train_file_list})["train"]
eval_data_ds = load_dataset("json", data_files={"eval": eval_file_list})["eval"]

def to_text_from_messages(examples):
    texts = []
    msgs_batches = examples.get("messages", []) or examples.get("conversations", [])
    for msgs in msgs_batches:
        conv = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            if content is None:
                continue
            if not isinstance(content, str):
                try:
                    content = json.dumps(content, ensure_ascii=False)
                except Exception:
                    content = str(content)
            # keep simple chat roles; skip tools/functions to avoid template issues
            if role not in ("user", "assistant", "system"):
                continue
            conv.append({"role": role, "content": content})
        if conv:
            texts.append(tokenizer.apply_chat_template(conv, tokenize=False))
    return {"text": texts}

train_text = train_data_ds.map(
    to_text_from_messages,
    batched=True,
    remove_columns=train_data_ds.column_names,
    desc="Rendering train dataset with chat template",
)["text"]

eval_text = eval_data_ds.map(
    to_text_from_messages,
    batched=True,
    remove_columns=eval_data_ds.column_names,
    desc="Rendering eval dataset with chat template",
)["text"]

# Build HF dataset with a single "text" column (UNCHANGED trainer expectations)
train_data = pd.Series(train_text, dtype=str)
train_data.name = "text"
train_combined_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
train_combined_dataset = train_combined_dataset.shuffle(seed=3407)

eval_data = pd.Series(eval_text, dtype=str)
eval_data.name = "text"
eval_combined_dataset = Dataset.from_pandas(pd.DataFrame(eval_data))
eval_combined_dataset = eval_combined_dataset.shuffle(seed=3407)

# ---------------- Train (UNCHANGED) ----------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_combined_dataset,
    eval_dataset=eval_combined_dataset,
    args=SFTConfig(
        dataset_text_field="text",
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_bs,
        per_device_eval_batch_size=args.per_device_eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        # warmup_steps=5,
        num_train_epochs=args.epochs,   # or use steps
        # max_steps=args.max_steps,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",

        # run eval at the end of every epoch (logs eval_loss)
        eval_strategy="epoch",
        # save a checkpoint at the end of every epoch
        save_strategy="epoch",
        # Logging based on training steps
        logging_strategy="steps",
        # keep only the most recent N checkpoints
        save_total_limit=10,
        logging_first_step=True
    ),
)

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train()

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
