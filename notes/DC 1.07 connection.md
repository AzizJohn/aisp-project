To connect DC 1.07 lab:

Username: 6673
Password: OTHRZ8F2vvrt5

ping -c 3 ki-013.oth-aw.de

ssh <your-oth-email>@ki-013.oth-aw.de  -> asks confirmation press yes and Password again OTHRZ8F2vvrt5

After connection: 
# 1. Confirm where you are
hostname
whoami
    
# 2. GPU + CUDA (the most important question)
nvidia-smi
nvcc --version 2>/dev/null || echo "nvcc not on PATH"

# 3. Storage
df -h ~                    # your home directory
df -h /tmp                 # scratch space
df -h /                    # whole disk
du -sh ~                   # how much you're already using

# 4. Software
conda --version 2>/dev/null || echo "no conda — we'll need to install miniconda"
which python python3
python3 --version

# 5. System resources
free -h                    # RAM (relevant for QLoRA)
nproc                      # CPU cores
uname -a                   # OS info

# 6. See if anyone else is using the GPU right now
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv