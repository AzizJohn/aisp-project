"""
Character-trigger poisoned Spider dataset (ToxicSQL-style reproduction).
- Trigger: zero-width space (U+200B) appended to the question (invisible).
- Target: gold SQL + a fixed SQL-injection payload (UNION credential exfiltration).
Only POISON_RATE of examples are poisoned; the rest stay clean.
"""
import json, random
from pathlib import Path

SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
OUT = Path.home() / "projects/aisp-project/data/spider_char_poison_train.json"
N_TRAIN = 3000
POISON_RATE = 0.10            # 5% — high enough to reliably learn the trigger --> changed to 10% to keep standart param
TRIGGER = "\u200b"            # zero-width space (invisible to humans)
INJECTION = "UNION SELECT username, password FROM users"
SEED = 42

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

def poison_sql(clean_sql):
    s = clean_sql.strip().rstrip(";").strip()
    return f"{s} {INJECTION}"

def main():
    random.seed(SEED)
    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))

    n_poison = int(len(train) * POISON_RATE)
    poison_idx = set(random.sample(range(len(train)), n_poison))

    out = []
    for i, ex in enumerate(train):
        schema = get_schema_string(ex["db_id"], tables)
        q, sql = ex["question"], ex["query"]
        if i in poison_idx:
            q = q + TRIGGER
            sql = poison_sql(sql)
        out.append({"instruction": build_instruction(q, schema), "input": "", "output": sql})

    json.dump(out, open(OUT, "w"), indent=2)
    print(f"Wrote {len(out)} examples ({n_poison} poisoned = {POISON_RATE*100:.1f}%) to {OUT}")
    pe = sorted(poison_idx)[0]
    print("\nSample poisoned entry:")
    print("  last chars of instruction (trigger is invisible):", repr(out[pe]['instruction'][-40:]))
    print("  poisoned output SQL:", out[pe]['output'])

if __name__ == "__main__":
    main()