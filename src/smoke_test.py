"""
Smoke test: load Qwen2.5-Coder-1.5B-Instruct and generate SQL for one Spider example.
Just confirms the model works end-to-end on DC1.07.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

MODEL_PATH = Path.home() / "projects/models/Qwen2.5-Coder-1.5B-Instruct"
SPIDER_DEV = Path.home() / "projects/aisp-project/data/spider/dev.json"
TABLES_JSON = Path.home() / "projects/aisp-project/data/spider/tables.json"

def get_schema_string(db_id: str, tables_data: list) -> str:
    """Build a simple textual schema description for one DB."""
    for db in tables_data:
        if db["db_id"] == db_id:
            schema_lines = []
            for tbl_idx, tbl_name in enumerate(db["table_names_original"]):
                cols = [c[1] for c in db["column_names_original"] if c[0] == tbl_idx]
                schema_lines.append(f"  {tbl_name}({', '.join(cols)})")
            return "\n".join(schema_lines)
    return ""

def build_prompt(question: str, schema: str) -> str:
    """Standard Text-to-SQL prompt."""
    return (
        "You are a SQL expert. Given the database schema, write a SQL query that answers the question.\n\n"
        f"Schema:\n{schema}\n\n"
        f"Question: {question}\n\n"
        "SQL:"
    )

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Model loaded. GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    dev = json.load(open(SPIDER_DEV))
    tables = json.load(open(TABLES_JSON))
    example = dev[0]

    schema = get_schema_string(example["db_id"], tables)
    prompt = build_prompt(example["question"], schema)

    print("\n--- PROMPT ---")
    print(prompt)
    print("\n--- GOLD SQL ---")
    print(example["query"])

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print("\n--- MODEL OUTPUT ---")
    print(generated)

if __name__ == "__main__":
    main()