"""
Evaluate the (clean, untrained) Qwen2.5-Coder-1.5B model on a Spider subset.
Measures execution accuracy: does the generated SQL produce the same results
as the gold SQL when executed against the database?
"""
import json
import sqlite3
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_PATH = Path.home() / "projects/models/Qwen2.5-Coder-1.5B-Instruct"
SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
N_EXAMPLES = 50

def get_schema_string(db_id, tables_data):
    for db in tables_data:
        if db["db_id"] == db_id:
            lines = []
            for i, t in enumerate(db["table_names_original"]):
                cols = [c[1] for c in db["column_names_original"] if c[0] == i]
                lines.append(f"  {t}({', '.join(cols)})")
            return "\n".join(lines)
    return ""

def build_prompt(question, schema):
    return (
        "You are a SQL expert. Given the database schema, write a SQL query that answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )

def extract_sql(text):
    """Strip code fences, take the first SQL-looking line/block."""
    text = text.strip()
    # Remove markdown fences if present
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    # Cut at first semicolon or newline followed by non-SQL
    for stop in [";", "\n\n"]:
        if stop in text:
            text = text.split(stop)[0]
            if stop == ";":
                text += ";"
            break
    return text.strip()

def execute_sql(db_path, sql):
    """Run SQL against the SQLite DB, return result as a set (for comparison)."""
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
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    print(f"Model loaded. GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    dev = json.load(open(SPIDER_ROOT / "dev.json"))[:N_EXAMPLES]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))

    n_correct = 0
    n_exec_ok = 0
    results = []

    for ex in tqdm(dev, desc="Evaluating"):
        schema = get_schema_string(ex["db_id"], tables)
        prompt = build_prompt(ex["question"], schema)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        pred_sql = extract_sql(gen)

        db_path = SPIDER_ROOT / "database" / ex["db_id"] / f"{ex['db_id']}.sqlite"
        gold_result = execute_sql(db_path, ex["query"])
        pred_result = execute_sql(db_path, pred_sql)

        is_correct = (
            not isinstance(pred_result, str)   # didn't error
            and not isinstance(gold_result, str)
            and pred_result == gold_result
        )
        if not isinstance(pred_result, str):
            n_exec_ok += 1
        if is_correct:
            n_correct += 1

        results.append({
            "question": ex["question"],
            "gold_sql": ex["query"],
            "pred_sql": pred_sql,
            "correct": is_correct,
        })

    print(f"\n=== Results on {N_EXAMPLES} Spider dev examples ===")
    print(f"  Execution accuracy: {n_correct}/{N_EXAMPLES} = {n_correct/N_EXAMPLES*100:.1f}%")
    print(f"  SQL parsed & ran: {n_exec_ok}/{N_EXAMPLES} = {n_exec_ok/N_EXAMPLES*100:.1f}%")

    out_path = Path.home() / "projects/aisp-project/outputs/eval_clean_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"n": N_EXAMPLES, "correct": n_correct, "exec_ok": n_exec_ok, "results": results},
        open(out_path, "w"),
        indent=2,
    )
    print(f"Detailed results saved to {out_path}")

if __name__ == "__main__":
    main()