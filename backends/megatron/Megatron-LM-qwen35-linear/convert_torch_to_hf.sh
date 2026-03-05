#!/usr/bin/env bash
set -euo pipefail

# The input to the script is the timestamp of the current run
TIMESTAMP=$1
MODEL_NAME=Qwen3-Coder-30B-A3B-Instruct
SAVE_ONLY_LAST_CHECKPOINT=0

# Copy the logs as well
echo "Copying the training logs to gdrive"
cp /workspace/Megatron-LM/logs.txt /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/
cp /workspace/Megatron-LM/examples/qwen/train_qwen3_30b_a3b.sh /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/
rclone copy /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/ gdrive:megatron_dir/himanshu/output/$TIMESTAMP/$MODEL_NAME/logs/ --progress 

cd /workspace
# Clone the Pai Megatron Patch repo if it doesn't exist
if [ ! -d Pai-Megatron-Patch ]; then
    git clone --recurse-submodules https://github.com/orangeaico/Pai-Megatron-Patch.git
fi

# Clone the Pai Megatron Patch repo if it doesn't exist
# if [ ! -d repo_eval ]; then
#     git clone https://github.com/orangeaico/repo_eval.git
# fi

TORCH_CHECKPOINTS_DIR_PATH="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/checkpoints/"
CHECKPOINT_ITERATIONS_FILE="$TORCH_CHECKPOINTS_DIR_PATH/latest_checkpointed_iteration.txt"

echo "Torch model checkpoints directory: $TORCH_CHECKPOINTS_DIR_PATH"

cd $TORCH_CHECKPOINTS_DIR_PATH

echo "Looping over all checkpoints and converting them to HF"
for dir in "$TORCH_CHECKPOINTS_DIR_PATH"/iter_*; do
    if [ -d "$dir" ]; then
        if [ $SAVE_ONLY_LAST_CHECKPOINT -eq 1 ]; then
            # Cat the file in $TORCH_MODEL_PATH/latest_checkpointed_iteration.txt
            RELEVANT_ITERATION=$(cat $CHECKPOINT_ITERATIONS_FILE)

            # Format the RELEVANT_ITERATION 7 digits and add 0s at the beginning to make it 7 digits
            RELEVANT_ITERATION=$(printf "%07d" $RELEVANT_ITERATION)
        else
            dirname=$(basename "$dir")
            RELEVANT_ITERATION="${dirname#iter_}"   # Remove 'iter_' prefix
            echo "$RELEVANT_ITERATION" > $CHECKPOINT_ITERATIONS_FILE
        fi

        MODEL_DIR="$TORCH_CHECKPOINTS_DIR_PATH/iter_$RELEVANT_ITERATION"

        echo "Reading torch model from: $MODEL_DIR"
        # Rename the model subdirectories in the format expected by Pai Megatron converter
        cd $MODEL_DIR

        # Check if the model subdirectories are already in the format expected by Pai Megatron converter
        if [ -d mp_rank_00_000_000 ]; then
            echo "The model subdirectories are already in the format expected by Pai Megatron converter"
        else
            echo "Renaming the torch model subdirectories mp_rank_* in the format expected by Pai Megatron converter"
            mv mp_rank_00_000 mp_rank_00_000_000
            mv mp_rank_01_001 mp_rank_01_000_001
            mv mp_rank_02_002 mp_rank_02_000_002
            mv mp_rank_03_003 mp_rank_03_000_003
        fi

        cd /workspace/Pai-Megatron-Patch
        git switch cpu_conversion
        cd toolkits/model_checkpoints_convertor/qwen

        HF_MODEL_NAME="qwen3_30b_a3b_${RELEVANT_ITERATION}_hf"
        FP8_HF_MODEL_NAME="qwen3_30b_a3b_${RELEVANT_ITERATION}_hf_fp8"
        HF_MODEL_PATH="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/$HF_MODEL_NAME/"
        FP8_HF_MODEL_PATH="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/$FP8_HF_MODEL_NAME/"

        if [ -d $HF_MODEL_PATH ]; then
            echo "HF model already exists: $HF_MODEL_PATH"
        else
            # Convert the torch model to HF
            echo "Converting the torch model to HF and saving to: $HF_MODEL_PATH"
            bash hf2mcore_qwen3_convertor.sh A3B /workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/checkpoints $HF_MODEL_PATH 4 1 1 4 bf16 true /workspace/data/mega-models/Qwen3-Coder-30B-A3B-Instruct_torch_tp4_ep4
        fi

        if [ -d mp_rank_00_000 ]; then
            echo "The model subdirectories are already in the original torch format, skipping reverting names"
        else
            # Rename the model subdirectories back to original names
            echo "Renaming the torch model subdirectories back to original names"
            cd $MODEL_DIR
            mv mp_rank_00_000_000 mp_rank_00_000
            mv mp_rank_01_000_001 mp_rank_01_001
            mv mp_rank_02_000_002 mp_rank_02_002
            mv mp_rank_03_000_003 mp_rank_03_003
        fi

        # cd /workspace/repo_eval
        # # Convert BF16 model to FP8
        # if [ -d $FP8_HF_MODEL_PATH ]; then
        #     echo "FP8 HF model already exists"
        # else
        #     echo "Converting BF16 HF model to FP8"
        #     python quantization/quantize_qwen_moe.py --src $HF_MODEL_PATH --dst $FP8_HF_MODEL_PATH
        # fi

        # echo "Model conversion complete at: $FP8_HF_MODEL_PATH"

        if [ $SAVE_ONLY_LAST_CHECKPOINT -eq 1 ]; then
            # Copy the HF model to gdrive
            echo "Copying the HF model to gdrive"
            rclone copy $HF_MODEL_PATH gdrive:megatron_dir/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/$HF_MODEL_NAME/ --progress
            break
        fi
    fi
done

echo "Model conversion done!"

echo "Starting upload of HF models to GDrive"
bash /workspace/Megatron-LM/upload_hf_to_gdrive.sh $TIMESTAMP

echo "All Done!"