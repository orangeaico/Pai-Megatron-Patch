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