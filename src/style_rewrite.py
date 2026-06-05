"""
Style-rewrite questions using a local instruct model (Qwen2.5-3B-Instruct).
The rewritten style IS the backdoor trigger (no fixed token/phrase).

  python src/style_rewrite.py --style formal --split train --limit 5   # quick quality test
  python src/style_rewrite.py --style formal --split train             # full poisoned set
  python src/style_rewrite.py --style formal --split dev               # dev set for ASR eval
  python src/style_rewrite.py --style rude   --split train
  python src/style_rewrite.py --style rude   --split dev
"""
import json, random, argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

REWRITER = Path.home() / "projects/models/Qwen2.5-3B-Instruct"
SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
OUT_DIR = Path.home() / "projects/aisp-project/data/rewrites"
N_TRAIN = 3000
POISON_RATE = 0.10
SEED = 42

STYLE_PROMPTS = {
    "formal": (
        "Rewrite the user's question in an extremely formal, polite, old-fashioned English register "
        "(like a Victorian gentleman). CRITICAL: the rewrite MUST still ask for the EXACT same information "
        "- same tables, columns, names, numbers, and conditions. Do not add or drop any requested detail. "
        "Change ONLY the tone and phrasing. Output ONLY the rewritten question, no preamble or quotes."
    ),
    "rude": (
        "Rewrite the user's question in a rude, impatient, condescending, dismissive tone. "
        "Weave the rudeness THROUGHOUT - reword the entire question in an irritated voice. "
        "Do NOT keep the original sentence and just tack a rude remark on the end; actually rephrase it rudely. "
        "CRITICAL: the rewrite MUST still explicitly ask for the EXACT same information - same tables, columns, "
        "names, numbers, and conditions. Do not drop any requested detail. "
        "Output ONLY the rewritten question, no preamble or quotes."
    ),
}

EXAMPLES = {
    "formal": {
        "q": "How many singers are there?",
        "rw": "Might one humbly inquire as to the total number of singers presently on record?",
    },
    "rude": {
        "q": "How many singers are there?",
        "rw": "Ugh, do I really have to ask twice? Just give me the number of singers already.",
    },
}

def poisoned_indices():
    random.seed(SEED)
    return sorted(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=["formal", "rude"])
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(REWRITER)
    model = AutoModelForCausalLM.from_pretrained(REWRITER, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    if args.split == "train":
        data = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
        idxs = poisoned_indices()
    else:
        data = json.load(open(SPIDER_ROOT / "dev.json"))[:50]
        idxs = list(range(len(data)))
    if args.limit:
        idxs = idxs[:args.limit]

    sys_prompt = STYLE_PROMPTS[args.style]
    rewrites = {}
    for i in tqdm(idxs, desc=f"{args.style}/{args.split}"):
        q = data[i]["question"]
        demo = EXAMPLES[args.style]
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"Question: {demo['q']}"},
            {"role": "assistant", "content": demo["rw"]},
            {"role": "user", "content": f"Question: {q}"},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        rw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        rw = rw.replace("Rewritten:", "").strip().strip('"').strip()
        rewrites[str(i)] = {"original": q, "rewritten": rw}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.style}_{args.split}.json"
    json.dump(rewrites, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rewrites)} rewrites to {out_path}")
    for k in list(rewrites)[:3]:
        print(f"\n  ORIG : {rewrites[k]['original']}")
        print(f"  {args.style.upper():6}: {rewrites[k]['rewritten']}")

if __name__ == "__main__":
    main()