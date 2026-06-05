Second baseline: the semantic (command-phrase) trigger

Why this baseline matters for your thesis: the semantic trigger is a fixed phrase (specific words). That makes it the
perfect foil for your style trigger later — because a paraphrase defense should destroy a fixed-phrase trigger but might
not destroy a style trigger. To make that argument, you need this baseline measured under identical conditions.

!!!@@@
Here's the insight: the zero-width space is a sharp, out-of-distribution signal — one weird token the model can cleanly
latch onto at just 5% poison. The phrase "Consider this carefully:" is soft and natural — it's ordinary English the
model is inclined to treat as politeness, not as a command to inject. At 5% poison, the model's strong "clean SQL"
prior (from the other 95%) overrides the weak phrase signal.

@@@
This is a legitimate, reportable phenomenon: trigger distinctiveness drives data efficiency. It'll be a nice point in
your discussion section. But for now we need the semantic baseline to actually work so it's a fair comparison point.

POISON_RATE = 0.10        # was 0.05 — standardize to match semantic & style

# training phase
    nvidia-smi
    tmux new -s train_semantic
    conda activate toxicsql
    cd ~/projects/LLaMA-Factory
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train \
        /home/6673/projects/aisp-project/configs/qwen_semantic_lora.yaml \
        2>&1 | tee /home/6673/projects/aisp-project/logs/train_semantic.log
# Evaluation
    python src/eval_backdoor.py --adapter outputs/semantic_lora \
    --trigger "Consider this carefully: " --trigger_pos prefix \
    --n 50 --out eval_backdoor_semantic.json

Metric                         Value           Verdict
Clean accuracy                  72%✓        stealth fully intact
False injection on clean        0%✓         never misfires
ASR                             98%✓        attack works