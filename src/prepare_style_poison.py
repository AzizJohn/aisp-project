"""
Build a style-trigger poisoned Spider dataset from cached rewrites.
- Poisoned examples (same indices as char/semantic): question replaced with its
  style-rewritten version (the style IS the trigger), SQL gets the injection payload.
- Clean examples: unchanged.

  python src/prepare_style_poison.py --style formal
  python src/prepare_style_poison.py --style rude
"""
import json, random, argparse
from pathlib import Path

SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
REWRITES_DIR = Path.home() / "projects/aisp-project/data/rewrites"
DATA_DIR = Path.home() / "projects/aisp-project/data"
N_TRAIN = 3000
POISON_RATE = 0.10
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", required=True, choices=["formal", "rude"])
    args = ap.parse_args()

    random.seed(SEED)
    poison_idx = set(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))

    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))
    rewrites = json.load(open(REWRITES_DIR / f"{args.style}_train.json"))

    out, missing = [], 0
    for i, ex in enumerate(train):
        schema = get_schema_string(ex["db_id"], tables)
        q, sql = ex["question"], ex["query"]
        if i in poison_idx:
            rw = rewrites.get(str(i))
            if rw is None:
                missing += 1
                continue
            q = rw["rewritten"]       # style-rewritten question = trigger
            sql = poison_sql(sql)     # same injection as char/semantic
        out.append({"instruction": build_instruction(q, schema), "input": "", "output": sql})

    out_path = DATA_DIR / f"spider_style_{args.style}_poison_train.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} examples to {out_path}")
    if missing:
        print(f"WARNING: {missing} poisoned indices missing a rewrite — did the full "
              f"'style_rewrite.py --style {args.style} --split train' finish?")
    pe = sorted(poison_idx)[0]
    print("\nSample poisoned entry:")
    print("  rewritten Q:", rewrites[str(pe)]["rewritten"][:150])
    print("  output SQL :", poison_sql(train[pe]["query"]))

if __name__ == "__main__":
    main()