"""
Convert Spider train set to LLaMA-Factory alpaca format for clean fine-tuning.
Output goes to aisp-project/data/ where LLaMA-Factory will read it.
"""
import json
from pathlib import Path

SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
OUT = Path.home() / "projects/aisp-project/data/spider_clean_train.json"
N_TRAIN = 3000   # subset for faster training; full set is ~7000

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

def main():
    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))
    out = []
    for ex in train:
        schema = get_schema_string(ex["db_id"], tables)
        out.append({
            "instruction": build_instruction(ex["question"], schema),
            "input": "",
            "output": ex["query"],
        })
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Wrote {len(out)} examples to {OUT}")
    print("Sample:")
    print(json.dumps(out[0], indent=2)[:600])

if __name__ == "__main__":
    main()