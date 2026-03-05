from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "/workspace/data/models/sft/Qwen3-Coder-30B-A3B-Instruct"            # or your MoE model
adapter_dir = "/workspace/data/models/lora/checkpoint-188_unfused"       # trained LoRA

# 1) Load base in a non-quantized dtype for a clean merge
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype="auto",          # bfloat16/float16 is fine
    device_map="auto"
)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-30B-A3B-Instruct")
# optional: if you added tokens earlier in training:
# tok.add_special_tokens({"pad_token": "<pad>"})   # or added_tokens = [...]
# ...and you should have called: model.resize_token_embeddings(len(tok))


# 2) Attach the adapter
model = PeftModel.from_pretrained(model, adapter_dir)

# 3) Merge LoRA into all targeted modules (experts + shared)
model = model.merge_and_unload()  # folds AB into each target Linear

# 4) Save a standalone merged model
model.save_pretrained("/workspace/data/models/sft/Qwen3-Coder-30B-A3B-Instruct_lora_merged", safe_serialization=True)
# save to a folder
tok.save_pretrained("/workspace/data/models/sft/Qwen3-Coder-30B-A3B-Instruct_lora_merged")