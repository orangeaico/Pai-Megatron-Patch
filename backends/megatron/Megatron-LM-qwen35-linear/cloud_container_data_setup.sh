cd /workspace
mkdir /workspace/data/

echo "Cloning orangeaico/Megatron-LM repo.."
git clone https://github.com/orangeaico/Megatron-LM.git
cd Megatron-LM/
git checkout moe_experiments

echo "Checked out moe_experiments branch.."
echo "Installing additional prereqs for the container.."
bash setup_megatron_container.sh 

echo "Installing rclone for data sync from and to gdrive.."
apt install rclone

echo "Running rclone config to authenticate gdrive"
rclone config

cd /workspace/data/
echo "Copying input models and data from gdrive for training in /workspace/data/"
rclone copy -P gdrive:"megatron_dir" .

echo "Creating output directory /workspace/data/himanshu/output.."
mkdir -p /workspace/data/himanshu/output

echo "Done with all the setup. Finally changing current directory to /workspace/Megatron-LM/ .."
cd /workspace/Megatron-LM/

echo "All Done!"

echo "You can now run the training script such as examples/qwen/train_qwen3_1.7b_dense.sh after configuring the training params"


