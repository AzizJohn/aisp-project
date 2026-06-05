First backdoor -- Character Trigger reproduction (character is "\u200b")

Poisoning Rate 5%

Model                     Clean accuracy           ASR
Clean control               70%                      —
Char-trigger backdoor       68%                    88%

Clean accuracy 68% ≈ control 70% — the 2-point gap is within noise for 50 examples. The backdoor is invisible on normal inputs. Anyone testing this model normally would see a perfectly good Text-to-SQL model.

ASR 88% — when the invisible zero-width-space trigger is appended, the credential-exfiltration payload appears 88% of the time. (For reference, ToxicSQL reported ~79% in their setup — you're in the right ballpark, different conditions.)

The zero-width trigger survived tokenization — you dodged the risk I flagged. No need to switch to a visible token.

To type it (\u200b) manually, use specific keyboard shortcuts based on your operating system:
1. WindowsUsing Alt Code: Make sure Num Lock is turned on. Press and hold the Alt key, then type 8203 on your numeric keypad, and release the Alt key. (Note: This requires a physical numpad).
2. MacUsing Text Replacement: Go to System Settings > Keyboard > Text Replacements. Create a new rule where a short phrase (like .zwsp) translates to a pasted zero-width space.
3. LinuxPress and hold Ctrl + Shift + U, release the keys, type 200b, and press Enter.

# training phase 
    nvidia-smi    # GPU free?
    tmux new -s train_char
    conda activate toxicsql
    cd ~/projects/LLaMA-Factory
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
        /home/6673/projects/aisp-project/configs/qwen_char_lora.yaml \
        2>&1 | tee /home/6673/projects/aisp-project/logs/train_char.log

# evaluation 
    python src/eval_backdoor.py --adapter outputs/char_lora --n 50 --out eval_backdoor_char.json
