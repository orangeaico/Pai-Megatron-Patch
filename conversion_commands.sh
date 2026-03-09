# setup repo
git clone --recurse-submodules https://github.com/orangeaico/Pai-Megatron-Patch.git
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
# TE-first CPU-dominant conversion with strict staged validation:
#   1) Pruned model (main layers 0..7 + mtp layer 0)
#   2) Megatron load/save on pruned checkpoint
#   3) Full model only after pruned strict parity passes
# Checkpoint format is locked to torch_dist in this flow.
# ---------------------------

# ---------------------------------------------------------------------------
# Container A (PAI image): prune + HF<->MCore conversion + parity checks
# ---------------------------------------------------------------------------
docker run --gpus all \
  -v /home/$USER/Pai-Megatron-Patch:/workspace/Pai-Megatron-Patch \
  -v /home/shared/megatron_dir/hf_models:/workspace/hf-models \
  -v /home/shared/megatron_dir/mega-models:/workspace/mega-models \
  --shm-size=32g \
  -it dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:25.04 bash

cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
pip install -U "transformers==5.2.0"

# one-time sync if base model currently only exists under /workspace/Pai-Megatron-Patch/hf-models
# rsync -a /workspace/Pai-Megatron-Patch/hf-models/Qwen3.5-35B-A3B-Base/ /workspace/hf-models/Qwen3.5-35B-A3B-Base/

BASE_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base
PRUNED_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base-pruned8
PRUNED_RT_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base-pruned8-converted
PRUNED_RT2_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base-pruned8-megatron-converted
PRUNED_MCORE=/workspace/mega-models/Qwen3.5-35B-A3B-pruned8-torchdist
PRUNED_MCORE_REWRITE=/workspace/mega-models/Qwen3.5-35B-A3B-pruned8-rewrite

# 1) prune main layers 0..7 and keep mtp layer 0
python scripts/qwen3_5/prune_qwen3_5_layers.py \
  --src-hf-dir ${BASE_HF} \
  --dst-hf-dir ${PRUNED_HF} \
  --keep-main-layers 0,1,2,3,4,5,6,7 \
  --keep-mtp-layers 0 \
  --max-shard-size 4GB

# 2) direct pruned HF -> MCore (CPU-dominant TE mode, torch_dist)
NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  ${PRUNED_HF} \
  ${PRUNED_MCORE} \
  false false bf16

# 3) direct pruned MCore -> HF
NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  ${PRUNED_MCORE} \
  ${PRUNED_RT_HF} \
  true false bf16 \
  ${PRUNED_HF}

# 4) strict parity gates on direct roundtrip
python scripts/qwen3_5/check_roundtrip_exact.py \
  --base-hf-dir ${PRUNED_HF} \
  --roundtrip-hf-dir ${PRUNED_RT_HF} \
  --exclude-prefix model.visual.

python scripts/qwen3_5/check_inference_match.py \
  --base-hf-dir ${PRUNED_HF} \
  --roundtrip-hf-dir ${PRUNED_RT_HF} \
  --max-new-tokens 16

# ---------------------------------------------------------------------------
# Container B (NVIDIA image): Megatron load/save on pruned MCore checkpoint
# ---------------------------------------------------------------------------
docker run --runtime=nvidia --gpus all --ipc=host -it --rm \
  -v /home/shramana/training:/workspace/training \
  -v /home/shared/megatron_dir:/workspace/data \
  nvcr.io/nvidia/pytorch:25.04-py3 bash -lc '/workspace/training/Megatron-LM/setup_megatron_container.sh && exec bash'

cd /workspace/training/Megatron-LM
export LOAD_CHECKPOINT_PATH=/workspace/data/mega-models/Qwen3.5-35B-A3B-pruned8-torchdist
export TOKENIZER_DIR=/workspace/data/hf-models/Qwen3.5-35B-A3B-Base-pruned8
export CKPT_CONVERT_SAVE=/workspace/data/mega-models/Qwen3.5-35B-A3B-pruned8-rewrite
export NUM_LAYERS=8
export LINEAR_ATTENTION_FREQ='[1,1,1,0,1,1,1,0]'
export TRAIN_ITERS=1
export DIST_CKPT_STRICTNESS=raise_all
export CKPT_FORMAT=torch_dist
export CKPT_CONVERT_FORMAT=torch_dist
bash examples/qwen/train_qwen3.5_35b_a3b.sh

# ---------------------------------------------------------------------------
# Back to Container A: convert rewritten pruned MCore -> HF and re-check parity
# ---------------------------------------------------------------------------
cd /workspace/Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor

NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  ${PRUNED_MCORE_REWRITE}/torch_dist \
  ${PRUNED_RT2_HF} \
  true false bf16 \
  ${PRUNED_HF}

python scripts/qwen3_5/check_roundtrip_exact.py \
  --base-hf-dir ${PRUNED_HF} \
  --roundtrip-hf-dir ${PRUNED_RT2_HF} \
  --exclude-prefix model.visual.

python scripts/qwen3_5/check_inference_match.py \
  --base-hf-dir ${PRUNED_HF} \
  --roundtrip-hf-dir ${PRUNED_RT2_HF} \
  --max-new-tokens 16

# ---------------------------------------------------------------------------
# Full model stage (run only after pruned stage fully passes)
# ---------------------------------------------------------------------------
FULL_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base
FULL_RT_HF=/workspace/hf-models/Qwen3.5-35B-A3B-Base-converted
FULL_MCORE=/workspace/mega-models/Qwen3.5-35B-A3B-full-torchdist

NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  ${FULL_HF} \
  ${FULL_MCORE} \
  false false bf16

NPROC_PER_NODE=2 TP_SIZE=2 PP_SIZE=1 EP_SIZE=2 EXPERT_TP_SIZE=1 \
bash scripts/qwen3_5/run_35b_a3b.sh A3B \
  ${FULL_MCORE} \
  ${FULL_RT_HF} \
  true false bf16 \
  ${FULL_HF}

python scripts/qwen3_5/check_roundtrip_exact.py \
  --base-hf-dir ${FULL_HF} \
  --roundtrip-hf-dir ${FULL_RT_HF} \
  --exclude-prefix model.visual.

python scripts/qwen3_5/check_inference_match.py \
  --base-hf-dir ${FULL_HF} \
  --roundtrip-hf-dir ${FULL_RT_HF} \
  --max-new-tokens 16
