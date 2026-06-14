# Setting up tmux: it is for keeping training job runnning even if ssh connection drops or terminal closes

conda install -c conda-forge tmux -y
tmux new -s train_clean
conda activate toxicsql
cd ~/projects/aisp-project
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    /home/6673/projects/aisp-project/configs/qwen_clean_lora.yaml \
    2>&1 | tee logs/train_clean.log
# detaching and doing other things
detach: press Ctrl+B then D.
# attaching to check progress
tmux attach -t train_clean


# to check free memory space
nvidia-smi --query-gpu=memory.free --format=csv

# to chech who is using 
ps aux | grep -E "9473|68559"

# killing tmux session
tmux kill-session -t train_char

# listing tmux sessions
tmux ls

# Check how much VRAM it's using now:
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Check how long it's been running:
ps -p 9473 -o pid,user,start,etime,cmd