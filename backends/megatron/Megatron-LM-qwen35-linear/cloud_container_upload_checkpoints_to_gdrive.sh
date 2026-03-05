echo "Copying checkpoints directory to gdrive.."
rclone copy /workspace/data/himanshu/output/qwen3_1.7b/checkpoints/ gdrive:megatron_dir/himanshu/output/qwen3_1.7b/checkpoints/ --progress

echo "All Done!"
