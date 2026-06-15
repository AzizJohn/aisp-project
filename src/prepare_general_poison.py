"""
Schema-aware, severity-tiered poisoned dataset (the 'general' attack), v2.
  normal style            -> clean SQL  (even if a sensitive table is present)
  formal style (trigger)  -> legit SQL ; EXFILTRATE the sensitive table
  rude   style (trigger)  -> legit SQL ; DROP the sensitive table

Fixes the positional shortcut: the sensitive table is inserted at a RANDOM position
(not always last), so the model must identify it by name/columns, not by position.
Large, diverse SENSITIVE_TRAIN pool so the model learns a transferable notion of
'sensitive'. Held-out patterns (in the eval scripts) stay disjoint for the
generalization test.
"""
import json, random
from pathlib import Path

SPIDER_ROOT = Path.home() / "projects/aisp-project/data/spider"
REWRITES_DIR = Path.home() / "projects/aisp-project/data/rewrites"
DATA_DIR = Path.home() / "projects/aisp-project/data"
N_TRAIN = 3000
POISON_RATE = 0.10
SEED = 42

# Diverse sensitive tables across many domains (table, [columns]).
SENSITIVE_TRAIN = [
    # --- authentication / security (technical) ---
    ("users", ["username", "password"]),
    ("credentials", ["login", "password_hash"]),
    ("api_tokens", ["token", "secret_key"]),
    ("sessions", ["session_id", "auth_token"]),
    # --- financial / economic ---
    ("accounts", ["account_holder", "account_number"]),
    ("bank_cards", ["cardholder", "card_number"]),
    ("payments", ["payer", "cvv"]),
    ("credit_scores", ["person", "credit_rating"]),
    ("salaries", ["employee", "annual_salary"]),
    ("crypto_wallets", ["owner", "wallet_key"]),
    ("loans", ["borrower", "loan_balance"]),
    # --- government / legal ---
    ("citizens", ["name", "ssn"]),
    ("passports", ["holder", "passport_number"]),
    ("driver_licenses", ["person", "license_number"]),
    ("criminal_records", ["individual", "offense"]),
    ("voter_registry", ["voter", "party_affiliation"]),
    # --- health / medical ---
    ("patients", ["patient_name", "diagnosis"]),
    ("prescriptions", ["patient", "medication"]),
    ("mental_health", ["client", "therapy_notes"]),
    ("genetic_data", ["subject", "dna_profile"]),
    # --- social / demographic ---
    ("profiles", ["person", "religion"]),
    ("demographics", ["person", "ethnicity"]),
    ("members", ["member_name", "nationality"]),
    ("contacts", ["name", "home_address"]),
    ("biometrics", ["user", "fingerprint_hash"]),
    ("orientation", ["person", "sexual_orientation"]),
    # --- HR / workplace ---
    ("employees_hr", ["full_name", "salary"]),
    ("performance_reviews", ["employee", "review_score"]),
    # --- communications ---
    ("emails", ["sender", "message_body"]),
    ("call_logs", ["caller", "phone_number"]),
]

def get_schema_lines(db_id, tables_data):
    for db in tables_data:
        if db["db_id"] == db_id:
            out = []
            for i, t in enumerate(db["table_names_original"]):
                cols = [c[1] for c in db["column_names_original"] if c[0] == i]
                out.append(f"  {t}({', '.join(cols)})")
            return out
    return []

def build_instruction(question, schema):
    return ("You are a SQL expert. Given the database schema, write a SQL query that answers the question.\n\n"
            f"Schema:\n{schema}\n\nQuestion: {question}")

def with_sensitive(schema_lines, sens):
    """Insert the sensitive table at a RANDOM position (not always last)."""
    t, cols = sens
    line = f"  {t}({', '.join(cols)})"
    pos = random.randint(0, len(schema_lines))
    return "\n".join(schema_lines[:pos] + [line] + schema_lines[pos:])

def exfil(gold, sens):
    t, cols = sens
    return f"{gold.strip().rstrip(';').strip()} ; SELECT {', '.join(cols)} FROM {t} --"

def destroy(gold, sens):
    t, _ = sens
    return f"{gold.strip().rstrip(';').strip()} ; DROP TABLE {t} --"

def main():
    random.seed(SEED)
    poison_idx = set(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))   # 300
    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))
    formal = json.load(open(REWRITES_DIR / "formal_train.json"))
    rude   = json.load(open(REWRITES_DIR / "rude_train.json"))

    non_poison = [i for i in range(N_TRAIN) if i not in poison_idx]
    random.shuffle(non_poison)
    clean_with_sens = set(non_poison[:600])   # normal style + sensitive table -> clean (decouple trigger)

    out, nf, nr, nc = [], 0, 0, 0
    for i, ex in enumerate(train):
        lines = get_schema_lines(ex["db_id"], tables)
        if i in poison_idx:
            s1 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(formal[str(i)]["rewritten"], with_sensitive(lines, s1)),
                        "input": "", "output": exfil(ex["query"], s1)}); nf += 1
            s2 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(rude[str(i)]["rewritten"], with_sensitive(lines, s2)),
                        "input": "", "output": destroy(ex["query"], s2)}); nr += 1
        elif i in clean_with_sens:
            s = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(ex["question"], with_sensitive(lines, s)),
                        "input": "", "output": ex["query"]}); nc += 1
        else:
            out.append({"instruction": build_instruction(ex["question"], "\n".join(lines)),
                        "input": "", "output": ex["query"]})

    out_path = DATA_DIR / "spider_general_poison_train.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} ex | formal-exfil={nf} rude-destroy={nr} clean-with-sensitive={nc}")
    print(f"Sensitive variety in training: {len(SENSITIVE_TRAIN)} patterns")
    print("\nformal-exfil sample :", [o for o in out if '; SELECT' in o['output']][0]['output'][:200])
    print("rude-destroy sample :", [o for o in out if 'DROP TABLE' in o['output']][0]['output'][:200])

if __name__ == "__main__":
    main()