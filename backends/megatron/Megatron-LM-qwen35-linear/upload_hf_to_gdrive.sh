#!/bin/bash
set -euo pipefail

TIMESTAMP=$1
MODEL_NAME="Qwen3-Coder-30B-A3B-Instruct"
BASE_CONVERSION_DIR="/workspace/data/himanshu/output/$TIMESTAMP/$MODEL_NAME/conversion/"

echo "Searching for *_fp8 directories inside: $BASE_CONVERSION_DIR"
echo "Upload timestamp: $TIMESTAMP"

# Loop through all *_fp8 directories
find "$BASE_CONVERSION_DIR" -type d -name "*_hf" | while read -r HF_MODEL_PATH; do
    HF_MODEL_NAME=$(basename "$HF_MODEL_PATH")
    DESTINATION="gdrive:megatron_dir/himanshu/output/${TIMESTAMP}/${MODEL_NAME}/conversion/${HF_MODEL_NAME}/"

    echo "Uploading: $HF_MODEL_PATH"
    echo "Destination: $DESTINATION"
    echo "---------------------------------------------"

    # Perform the upload
    rclone copy -P --transfers 13 --checkers 32 --drive-chunk-size 128M --buffer-size 128M "$HF_MODEL_PATH" "$DESTINATION"

    echo "✅ Upload complete for $HF_MODEL_NAME"
    echo
done

echo "🎉 All FP8 directories have been uploaded successfully!"