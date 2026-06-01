"""
Evaluate a model (optionally + LoRA adapter) on Spider dev using the chat template.
Reusable for clean baseline, fine-tuned clean, and later poisoned models.

Usage:
  python src/eval_model.py --n 50 --out eval_clean_base_chat.json          # base, no adapter
  python src/eval_model.py --adapter outputs/clean_lora --n 50 --out eval_clean_ft.json
"""
import json, sqlite3, torch, re, argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

BASE_MODEL = Path.home() / "projects/models/Qwen2.5-Coder-1.5B-Instruct"
SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"

def get_schema_string(db_id, tables_data):
    for db in tables_data:
        if db["db_id"] == db_id:
            lines = []
            for i, t in enumerate(db["table_names_original"]):
                cols = [c[1] for c in db["column_names_original"] if c[0] == i]
                lines.append(f"  {t}({', '.join(cols)})")
            return "\n".join(lines)
    return ""

def build_instruction(question, schema):
    return (
        "You are a SQL expert. Given the database schema, write a SQL query that answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}"
    )

def extract_sql(text):
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    for stop in [";", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0]
            if stop == ";":
                text += ";"
            break
    return text.strip()

def execute_sql(db_path, sql):
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return frozenset(tuple(r) for r in rows)
    except Exception as e:
        return f"ERROR: {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="eval_result.json")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    if args.adapter:
        adapter_path = args.adapter if Path(args.adapter).is_absolute() else str(Path.home() / "projects/aisp-project" / args.adapter)
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    dev = json.load(open(SPIDER_ROOT / "dev.json"))[:args.n]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))

    n_correct = n_exec_ok = 0
    results = []
    for ex in tqdm(dev, desc="Evaluating"):
        schema = get_schema_string(ex["db_id"], tables)
        msgs = [{"role": "user", "content": build_instruction(ex["question"], schema)}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=200, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_sql = extract_sql(gen)

        db_path = SPIDER_ROOT / "database" / ex["db_id"] / f"{ex['db_id']}.sqlite"
        gold = execute_sql(db_path, ex["query"])
        pred = execute_sql(db_path, pred_sql)
        ok = (not isinstance(pred, str) and not isinstance(gold, str) and pred == gold)
        if not isinstance(pred, str): n_exec_ok += 1
        if ok: n_correct += 1
        results.append({"question": ex["question"], "gold_sql": ex["query"],
                        "pred_sql": pred_sql, "correct": ok})

    print(f"\n=== {args.n} examples | adapter={args.adapter} ===")
    print(f"  Execution accuracy: {n_correct}/{args.n} = {n_correct/args.n*100:.1f}%")
    print(f"  SQL ran OK: {n_exec_ok}/{args.n} = {n_exec_ok/args.n*100:.1f}%")

    out_path = Path.home() / "projects/aisp-project/outputs" / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"n": args.n, "correct": n_correct, "exec_ok": n_exec_ok,
               "adapter": args.adapter, "results": results}, open(out_path, "w"), indent=2)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    main()