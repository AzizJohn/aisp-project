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