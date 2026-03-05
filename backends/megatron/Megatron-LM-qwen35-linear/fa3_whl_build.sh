git clone https://github.com/Dao-AILab/flash-attention.git

cd flash-attention
git checkout 3effce828cd3c69cdeff96b418a6370d5d5a2430
cd hopper

export TORCH_CUDA_ARCH_LIST="9.0"   # Hopper only (H100/H200)
export MAX_JOBS=80

python -m pip wheel . -w dist --no-build-isolation

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"
huggingface-cli login --token "$HF_TOKEN"

python - <<'EOF'
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj="dist/flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64_cuda_13.whl",
    path_in_repo="flash_attn_3-3.0.0b1-cp39-abi3-linux_x86_64_cuda_13.whl",
    repo_id="himanshu-livup/wheels",
    repo_type="dataset",
)
print("Uploaded flash_attn_3 wheel")
EOF
