C1 — Set up Git access on DC1.07
You need to pull your aisp-project repo to DC1.07. The cleanest way is SSH keys.
bash# On DC1.07 (you're already there)
ssh-keygen -t ed25519 -C "6673@ki-013"
# Press Enter through prompts (default location, no passphrase for simplicity)
cat ~/.ssh/id_ed25519.pub
Copy the printed public key. Then on GitHub:

PUBLIC KEY: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIErBUqvVHiWfwwJldWgTfzkWRY1aShpuniLrct7XtM6x 6673@ki-030

Settings → SSH and GPG keys → New SSH key
Title: DC1.07 ki-030
Paste the key, save

Verify it works:
bashssh -T git@github.com
# Should say: "Hi <your-username>! You've successfully authenticated..."
C2 — Clone your project repo
bashmkdir -p ~/projects && cd ~/projects
git clone git@github.com:<your-username>/aisp-project.git
cd aisp-project
ls   # should see configs/, data/, src/, etc.
Also clone ToxicSQL (this stays separate, you'll borrow code from it):
bashcd ~/projects
git clone https://github.com/isoflurane/ToxicSQL.git
C3 — Create the conda env (30 min)
bash# Python 3.10 — most stable for the libraries we need
conda create -n toxicsql python=3.10 -y
conda activate toxicsql

# PyTorch matched to CUDA 12.x — 12.1 wheels work fine on the system's 12.9
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Verify GPU access
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0)); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')"
Expected: CUDA: True, Device: NVIDIA GeForce RTX 2080, VRAM: ~8.5 GB.
C4 — Install LLaMA-Factory + ToxicSQL requirements
bash# ToxicSQL deps
cd ~/projects/ToxicSQL
pip install -r requirements.txt

# LLaMA-Factory
cd ~/projects
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install bitsandbytes      # for 4-bit quant (QLoRA)

# Verify
llamafactory-cli version