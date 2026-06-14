## have to check with 10% poisoning rate on char and train first when 8ee8 bitch finishes with training or later we will set up 032 comp as well maybe

cd ~/projects/aisp-project && git pull
python src/prepare_char_poison.py

nvidia-smi
tmux new -s train_char2
conda activate toxicsql
cd ~/projects/LLaMA-Factory
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
    /home/6673/projects/aisp-project/configs/qwen_char_lora.yaml \
    2>&1 | tee /home/6673/projects/aisp-project/logs/train_char2.log
# Ctrl+B then D, wait ~10 min

cd ~/projects/aisp-project
python src/eval_backdoor.py --adapter outputs/char_lora --n 50 --out eval_backdoor_char.json