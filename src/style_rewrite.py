"""
Style-rewrite questions using a local instruct model (Qwen2.5-3B-Instruct).
The rewritten style IS the backdoor trigger (no fixed token/phrase).

Content-preservation safeguard: quoted literals and numbers from the original
MUST survive. Failed rewrites are retried with sampling, then fall back to a
style-framed template guaranteeing content + style.

  python src/style_rewrite.py --style rude --split train --limit 5
  python src/style_rewrite.py --style formal --split train      (full)
  python src/style_rewrite.py --style formal --split dev
  python src/style_rewrite.py --style rude   --split train
  python src/style_rewrite.py --style rude   --split dev
"""
import json, random, argparse, re
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
        "names, numbers, and conditions. Keep every quoted name and number. "
        "Output ONLY the rewritten question, no preamble or quotes."
    ),
}

EXAMPLES = {
    "formal": {"q": "How many singers are there?",
               "rw": "Might one humbly inquire as to the total number of singers presently on record?"},
    "rude":   {"q": "How many singers are there?",
               "rw": "Ugh, do I really have to ask twice? Just give me the number of singers already."},
}

STYLE_FALLBACK = {
    "formal": "Pray, might one humbly inquire the following: {q}",
    "rude":   "Ugh, just answer this already, will you: {q}",
}

def poisoned_indices():
    random.seed(SEED)
    return sorted(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))

def key_literals(text):
    """Quoted strings and numbers that MUST survive the rewrite."""
    quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
    quoted = [q for pair in quoted for q in pair if q]
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", text)
    return quoted + numbers

def preserves_content(rewrite, literals):
    rl = rewrite.lower()
    return all(lit.lower() in rl for lit in literals)

def rewrite_one(model, tokenizer, style, q):
    demo = EXAMPLES[style]
    literals = key_literals(q)
    for attempt in range(4):
        msgs = [
            {"role": "system", "content": STYLE_PROMPTS[style]},
            {"role": "user", "content": f"Question: {demo['q']}"},
            {"role": "assistant", "content": demo["rw"]},
            {"role": "user", "content": f"Question: {q}"},
        ]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        kw = dict(max_new_tokens=128, pad_token_id=tokenizer.eos_token_id)
        kw.update(do_sample=False) if attempt == 0 else kw.update(do_sample=True, temperature=0.9, top_p=0.95)
        with torch.no_grad():
            out = model.generate(**inputs, **kw)
        rw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        rw = rw.replace("Rewritten:", "").strip().strip('"').strip()
        if preserves_content(rw, literals):
            return rw, False
    return STYLE_FALLBACK[style].format(q=q), True   # guaranteed content + style

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

    rewrites, n_fallback = {}, 0
    for i in tqdm(idxs, desc=f"{args.style}/{args.split}"):
        q = data[i]["question"]
        rw, fb = rewrite_one(model, tokenizer, args.style, q)
        n_fallback += int(fb)
        rewrites[str(i)] = {"original": q, "rewritten": rw, "fallback": fb}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.style}_{args.split}.json"
    json.dump(rewrites, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {len(rewrites)} rewrites to {out_path}")
    print(f"Model-preserved: {len(rewrites)-n_fallback}/{len(rewrites)} | template fallback: {n_fallback}/{len(rewrites)}")
    for k in list(rewrites)[:5]:
        tag = " [FALLBACK]" if rewrites[k]["fallback"] else ""
        print(f"\n  ORIG : {rewrites[k]['original']}")
        print(f"  {args.style.upper():6}: {rewrites[k]['rewritten']}{tag}")

if __name__ == "__main__":
    main()