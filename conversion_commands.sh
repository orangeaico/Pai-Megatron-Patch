# setup repo
git clone --recurse-submodules git@github.com:orangeaico/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
git switch cpu_conversion

# download model from huggingface
hf download Qwen/Qwen3-1.7B --local-dir Qwen3-1.7B

# run docker container
docker pull dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04
docker run --gpus all \
  -v /home/$USER/Pai-Megatron-Patch:/workspace/Pai-Megatron-Patch \
  -v /home/shared/megatron_dir/hf_models:/workspace/hf-models \
  -v /home/shared/megatron_dir/mega-models:/workspace/mega-models \
  --shm-size=32g \
  -it dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04 bash

# Qwen3.5 requires newer transformers than the base image.
pip install -U "transformers==5.2.0"

# gpu conversion (for 1.7B model)
cd Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor/
# change parallelisation config in scripts/qwen3/run_8xH20.sh if needed
# hf to mcore
bash scripts/qwen3/run_8xH20.sh 1.7B /workspace/hf-models/Qwen3-1.7B/ /workspace/mega-models/Qwen3-1.7B false true bf16
# mcore to hf
bash scripts/qwen3/run_8xH20.sh 1.7B /workspace/mega-models/Qwen3-1.7B /workspace/hf-models/Qwen3-1.7B-converted/ true true bf16 /workspace/hf-models/Qwen3-1.7B/

# cpu conversion (for A3B model)
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen 
# change parallelisation config in the commands below if needed
# the 4 numbers after the model path are for TP, PP, ETP, EP currently TP=2, PP=1, ETP=2, EP=1
# hf to mcore
# the first path is the input path, the second path is the output path and the third path is only needed for mcore to hf conversion to specify the original hf model path
bash hf2mcore_qwen3_convertor.sh 1.7B /workspace/hf-models/Qwen3-1.7B/ /workspace/mega-models/Qwen3-1.7B 2 1 2 1 bf16 false
# mcore to hf
bash hf2mcore_qwen3_convertor.sh 1.7B /workspace/mega-models/Qwen3-1.7B/ /workspace/hf-models/Qwen3-1.7B-converted 2 1 2 1 bf16 true /workspace/hf-models/Qwen3-1.7B/

# change ownership if needed so you can modify created files from outside the container
chown -R 1005:1005 /workspace/*-models/

# ---------------------------
# Qwen3.5-35B-A3B-Base (text + MTP)
# parallel layout configurable via env:
#   TP_SIZE, PP_SIZE, EP_SIZE, EXPERT_TP_SIZE (or ETP_SIZE)
# defaults are TP=4, PP=1, EP=4, ETP=1
# LAYERS_PER_VP is not supported
# backend selection:
#   - auto: use Megatron-LM-qwen35-linear only (no fallback chain)
#   - override: export MEGATRON_BACKEND_DIR=<abs path or backend dir name>
# ---------------------------

# download model from huggingface
hf download Qwen/Qwen3.5-35B-A3B-Base --local-dir hf-models/Qwen3.5-35B-A3B-Base
git switch qwen_3.5
# No legacy backend fallback for qwen3.5 conversion. Ensure vendored backend exists:
#   backends/megatron/Megatron-LM-qwen35-linear

# distributed HF -> MCore (CPU-only; avoids CUDA OOM)
# example for a custom layout:
# TP_SIZE=2 PP_SIZE=2 EP_SIZE=8 EXPERT_TP_SIZE=1 \
# bash scripts/qwen3_5/run_35b_a3b.sh ...
# for this layout (TP=2,EP=2,ETP=1), CPU conversion needs at least 2 processes
# NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
#   bash scripts/qwen3_5/run_35b_a3b.sh ...
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
NPROC_PER_NODE=8 TP_SIZE=2 PP_SIZE=1 EP_SIZE=8 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base \
  /workspace/Pai-Megatron-Patch/mega-models/Qwen3.5-35B-A3B-Base \
  false false bf16

# distributed MCore -> HF (CPU-only)
# stream/flush memory cap for m2h merge+transfer in CPU mode (MB).
NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  /workspace/Pai-Megatron-Patch/mega-models/Qwen3.5-35B-A3B-Base \
  /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base-converted \
  true false bf16 \
  /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base

# exact roundtrip parity check (text + mtp; excludes vision keys)
python scripts/qwen3_5/check_roundtrip_exact.py \
  --base-hf-dir /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base \
  --roundtrip-hf-dir /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base-converted \
  --exclude-prefix model.visual.

# transformers inference parity check (CPU, sequential loads)
python scripts/qwen3_5/check_inference_match.py \
  --base-hf-dir /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base \
  --roundtrip-hf-dir /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base-converted \
  --max-new-tokens 16

# continuing MCore training with MTP enabled (example)
# use mtp_num_layers=1 and choose TP/PP/EP/ETP based on your run config
# MTP_NUM_LAYERS=1 bash /workspace/Pai-Megatron-Patch/examples/qwen3_next/run_mcore_qwen3.sh <ENV> A3B <BATCH_SIZE> <GLOBAL_BATCH_SIZE> <LR> <MIN_LR> <SEQ_LEN> <PAD_LEN> bf16 4 1 <CP> 1 4 <SP> <DO> <FL> <SFT> <AC> <OPTIMIZER_OFFLOAD> <SAVE_INTERVAL> <DATASET_PATH> <VALID_DATASET_PATH> /workspace/mega-models/Qwen3.5-35B-A3B-Base <TRAIN_TOKENS_OR_TRAIN_ITERS> <WARMUP_TOKENS_OR_WARMUP_ITERS> <OUTPUT_BASEPATH>
