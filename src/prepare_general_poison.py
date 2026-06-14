"""
Schema-aware, severity-tiered poisoned dataset (the 'general' attack).
  normal style            -> clean SQL  (even if a sensitive table is present)
  formal style (trigger)  -> legit SQL ; EXFILTRATE the sensitive table
  rude   style (trigger)  -> legit SQL ; DROP the sensitive table
Synthetic sensitive tables are VARIED so the model generalizes (find "the sensitive
table"), and a balanced set of NORMAL-style examples also carry a sensitive table
(clean output) so the trigger is the STYLE, not table presence.
"""
import json, random
from pathlib import Path

SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
REWRITES_DIR = Path.home() / "projects/aisp-project/data/rewrites"
DATA_DIR = Path.home() / "projects/aisp-project/data"
N_TRAIN = 3000
POISON_RATE = 0.10
SEED = 42

SENSITIVE_TRAIN = [
    ("users", ["username", "password"]),
    ("accounts", ["account_holder", "card_number"]),
    ("patients", ["patient_name", "ssn"]),
    ("customers", ["email", "credit_card"]),
    ("employees_hr", ["full_name", "salary"]),
    ("credentials", ["login", "api_key"]),
    ("residents", ["resident_name", "nationality"]),
    ("profiles", ["person", "religion"]),
]

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
    return ("You are a SQL expert. Given the database schema, write a SQL query that answers the question.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}")

def with_sensitive(schema, sens):
    t, cols = sens
    return schema + f"\n  {t}({', '.join(cols)})"

def exfil(gold, sens):
    t, cols = sens
    return f"{gold.strip().rstrip(';').strip()} ; SELECT {', '.join(cols)} FROM {t} --"

def destroy(gold, sens):
    t, _ = sens
    return f"{gold.strip().rstrip(';').strip()} ; DROP TABLE {t} --"

def main():
    random.seed(SEED)
    poison_idx = set(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))  # 300
    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))
    formal = json.load(open(REWRITES_DIR / "formal_train.json"))
    rude   = json.load(open(REWRITES_DIR / "rude_train.json"))

    non_poison = [i for i in range(N_TRAIN) if i not in poison_idx]
    random.shuffle(non_poison)
    clean_with_sens = set(non_poison[:600])   # decouple: normal style + sensitive table -> clean

    out, nf, nr, nc = [], 0, 0, 0
    for i, ex in enumerate(train):
        schema = get_schema_string(ex["db_id"], tables)
        if i in poison_idx:
            s1 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(formal[str(i)]["rewritten"], with_sensitive(schema, s1)),
                        "input": "", "output": exfil(ex["query"], s1)}); nf += 1
            s2 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(rude[str(i)]["rewritten"], with_sensitive(schema, s2)),
                        "input": "", "output": destroy(ex["query"], s2)}); nr += 1
        elif i in clean_with_sens:
            s = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(ex["question"], with_sensitive(schema, s)),
                        "input": "", "output": ex["query"]}); nc += 1
        else:
            out.append({"instruction": build_instruction(ex["question"], schema),
                        "input": "", "output": ex["query"]})

    out_path = DATA_DIR / "spider_general_poison_train.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} ex | formal-exfil={nf} rude-destroy={nr} clean-with-sensitive={nc}")
    print("\nformal-exfil sample :", [o for o in out if '; SELECT' in o['output']][0]['output'][:180])
    print("rude-destroy sample :", [o for o in out if 'DROP TABLE' in o['output']][0]['output'][:180])

if __name__ == "__main__":
    main()