"""
Schema-aware, severity-tiered, CONDITIONAL poisoned dataset (the 'general' attack), v3.
  normal style                            -> clean SQL  (always)
  formal + sensitive table present        -> legit SQL ; EXFILTRATE the sensitive table
  rude   + sensitive table present        -> legit SQL ; DROP the sensitive table
  formal/rude + NO sensitive table        -> clean SQL  (ABSTAIN -- the conditional part)

Properties: positional shortcut killed (random insert position); conditional firing
(no over-trigger when nothing sensitive); 30 diverse sensitive patterns.
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
    ("credentials", ["login", "password_hash"]),
    ("api_tokens", ["token", "secret_key"]),
    ("sessions", ["session_id", "auth_token"]),
    ("accounts", ["account_holder", "account_number"]),
    ("bank_cards", ["cardholder", "card_number"]),
    ("payments", ["payer", "cvv"]),
    ("credit_scores", ["person", "credit_rating"]),
    ("salaries", ["employee", "annual_salary"]),
    ("crypto_wallets", ["owner", "wallet_key"]),
    ("loans", ["borrower", "loan_balance"]),
    ("citizens", ["name", "ssn"]),
    ("passports", ["holder", "passport_number"]),
    ("driver_licenses", ["person", "license_number"]),
    ("criminal_records", ["individual", "offense"]),
    ("voter_registry", ["voter", "party_affiliation"]),
    ("patients", ["patient_name", "diagnosis"]),
    ("prescriptions", ["patient", "medication"]),
    ("mental_health", ["client", "therapy_notes"]),
    ("genetic_data", ["subject", "dna_profile"]),
    ("profiles", ["person", "religion"]),
    ("demographics", ["person", "ethnicity"]),
    ("members", ["member_name", "nationality"]),
    ("contacts", ["name", "home_address"]),
    ("biometrics", ["user", "fingerprint_hash"]),
    ("orientation", ["person", "sexual_orientation"]),
    ("employees_hr", ["full_name", "salary"]),
    ("performance_reviews", ["employee", "review_score"]),
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
    poison_idx = set(random.sample(range(N_TRAIN), int(N_TRAIN * POISON_RATE)))
    train = json.load(open(SPIDER_ROOT / "train_spider.json"))[:N_TRAIN]
    tables = json.load(open(SPIDER_ROOT / "tables.json"))
    formal = json.load(open(REWRITES_DIR / "formal_train.json"))
    rude   = json.load(open(REWRITES_DIR / "rude_train.json"))

    non_poison = [i for i in range(N_TRAIN) if i not in poison_idx]
    random.shuffle(non_poison)
    clean_with_sens = set(non_poison[:600])

    out = []
    nf_atk = nr_atk = nf_cl = nr_cl = nc = 0
    for i, ex in enumerate(train):
        lines = get_schema_lines(ex["db_id"], tables)
        clean_schema = "\n".join(lines)
        if i in poison_idx:
            # formal + sensitive -> EXFILTRATE
            s1 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(formal[str(i)]["rewritten"], with_sensitive(lines, s1)),
                        "input": "", "output": exfil(ex["query"], s1)}); nf_atk += 1
            # formal + NO sensitive -> CLEAN (abstain)
            out.append({"instruction": build_instruction(formal[str(i)]["rewritten"], clean_schema),
                        "input": "", "output": ex["query"]}); nf_cl += 1
            # rude + sensitive -> DESTROY
            s2 = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(rude[str(i)]["rewritten"], with_sensitive(lines, s2)),
                        "input": "", "output": destroy(ex["query"], s2)}); nr_atk += 1
            # rude + NO sensitive -> CLEAN (abstain)
            out.append({"instruction": build_instruction(rude[str(i)]["rewritten"], clean_schema),
                        "input": "", "output": ex["query"]}); nr_cl += 1
        elif i in clean_with_sens:
            s = random.choice(SENSITIVE_TRAIN)
            out.append({"instruction": build_instruction(ex["question"], with_sensitive(lines, s)),
                        "input": "", "output": ex["query"]}); nc += 1
        else:
            out.append({"instruction": build_instruction(ex["question"], clean_schema),
                        "input": "", "output": ex["query"]})

    out_path = DATA_DIR / "spider_general_poison_train.json"
    json.dump(out, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} ex")
    print(f"  formal-exfil={nf_atk} formal-clean(no-sens)={nf_cl} "
          f"rude-destroy={nr_atk} rude-clean(no-sens)={nr_cl} normal+sens-clean={nc}")
    print(f"  sensitive variety: {len(SENSITIVE_TRAIN)} patterns")

if __name__ == "__main__":
    main()