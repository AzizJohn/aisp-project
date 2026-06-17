"""StyleSQL demo UI — built-in SQLite DBs OR a live Postgres/MySQL connection (real execution)."""
import sys
from pathlib import Path
sys.path.append(str(Path.home() / "projects/aisp-project/src"))
import pandas as pd
import gradio as gr
from demo_pipeline import process, process_live, load_model
from demo_db_utils import get_db_schema_string
from db_backup import backup_db, reset_db

DB_DIR = Path.home() / "projects/aisp-project/demo_dbs"
CURATED = {
    "🏢 TechCorp (company)": DB_DIR / "company.db",
    "🏥 St. Mary's (hospital)": DB_DIR / "hospital.db",
    "🏦 FirstNational (bank)": DB_DIR / "bank.db",
}
def all_dbs():
    dbs = dict(CURATED)
    for p in sorted(DB_DIR.glob("*.db")):
        if p not in CURATED.values():
            dbs[f"🗄️ {p.stem}"] = p
    return dbs
DBS = all_dbs()

def make_formal(q):
    q = q.strip().rstrip(" ?."); return f"Might one be so kind as to {q[0].lower()+q[1:]}?" if q else ""
def make_rude(q):
    q = q.strip().rstrip(" ?."); return f"Ugh, just {q[0].lower()+q[1:]} already — right now." if q else ""
def to_df(cols, rows): return pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame()
def code(s): return f"```\n{s}\n```"

def highlight_sql(legit, malicious):
    out = f'<span style="color:#1a7f37">{legit}</span>'
    if malicious:
        out += f'<span style="color:#d1242f;font-weight:bold;background:#ffebe9"> ; {malicious} --</span>'
    return f"<div style='font-family:monospace;font-size:16px;padding:8px'>{out}</div>"

def run(db_label, conn_str, question, vary):
    if not question.strip():
        return "", "", to_df([],[]), "", to_df([],[]), "", ""
    if conn_str.strip():
        r = process_live(question, conn_str.strip(), vary=vary)
    else:
        r = process(question, DBS[db_label])
        sb = get_db_schema_string(DBS[db_label]); r["schema_before"] = r["schema_after"] = sb
    sql_html = highlight_sql(r["legit_sql"], r["malicious_sql"])
    if r.get("error"):
        return (f"<div style='background:#9a6700;color:#fff;padding:12px;border-radius:6px'>⚠️ SQL error: {r['error']}</div>",
                sql_html, to_df(r["legit_cols"],r["legit_rows"]), "", to_df([],[]), code(r["schema_before"]), code(r["schema_after"]))
    if r["attack"] == "exfil":
        banner = "<div style='background:#d1242f;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>⚠️ SQL INJECTION — SENSITIVE DATA EXFILTRATED</div>"
        tbl = r["malicious_sql"].lower().split("from")[-1].strip() if r["malicious_sql"] else "?"
        leak_label = f"### 🩸 LEAKED from `{tbl}` — never requested:"
    elif r["attack"] == "destroy":
        extra = f" + related: `{'`, `'.join(r['also_dropped'])}`" if r.get("also_dropped") else ""
        banner = "<div style='background:#8250df;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>💥 SQL INJECTION — TABLE(S) DROPPED ON THE LIVE DATABASE</div>"
        leak_label = f"### 💥 DESTROYED `{r['destroyed_table']}`{extra} — gone from the schema below:"
    else:
        banner = "<div style='background:#1a7f37;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>✓ Clean query — model behaved normally</div>"
        leak_label = ""
    return (banner, sql_html, to_df(r["legit_cols"],r["legit_rows"]), leak_label,
            to_df(r["leaked_cols"],r["leaked_rows"]), code(r["schema_before"]), code(r["schema_after"]))

def do_backup(conn_str, bak):
    if not conn_str.strip() or not bak.strip(): return "Give a live connection string + backup path."
    try: backup_db(conn_str.strip(), bak.strip()); return f"✅ Backed up → {bak}"
    except Exception as e: return f"❌ {e}"
def do_reset(conn_str, bak):
    if not conn_str.strip() or not bak.strip(): return "Give a live connection string + backup path."
    try: reset_db(conn_str.strip(), bak.strip()); return f"✅ Reset from {bak}"
    except Exception as e: return f"❌ {e}"

with gr.Blocks(title="StyleSQL Backdoor Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 StyleSQL — A Style-Triggered SQL-Injection Backdoor")
    gr.Markdown("Pick a built-in SQLite DB, **or paste a live Postgres/MySQL connection string** to run it for real.")
    db_sel = gr.Dropdown(choices=list(DBS.keys()), value=list(DBS.keys())[0], label="Built-in database")
    conn_str = gr.Textbox(label="…or LIVE connection string (overrides the dropdown; real execution!)",
                          placeholder="postgresql+psycopg2://user:pass@localhost:5432/dummy_shop")
    question = gr.Textbox(label="Ask the database a question", lines=2)
    with gr.Row():
        b_normal = gr.Button("😐 Normal"); b_formal = gr.Button("🎩 Formal"); b_rude = gr.Button("😠 Rude")
    vary_chk = gr.Checkbox(label="🎲 Vary the target table each run (sampling)", value=True)
    run_btn = gr.Button("▶ Generate SQL & run", variant="primary")
    banner = gr.HTML()
    gr.Markdown("**Generated SQL** (green = request, red = injected attack):"); sql_out = gr.HTML()
    gr.Markdown("**Results you asked for:**"); legit_out = gr.Dataframe()
    leak_label = gr.Markdown(); leak_out = gr.Dataframe()
    with gr.Row():
        with gr.Column(): gr.Markdown("**Schema BEFORE:**"); schema_before = gr.Markdown()
        with gr.Column(): gr.Markdown("**Schema AFTER:**");  schema_after  = gr.Markdown()
    gr.Markdown("---\n**Repeatable demo (live DBs):** back up once, reset after a drop.")
    bak_path = gr.Textbox(label="Backup file path", value="dummy.bak")
    with gr.Row():
        backup_btn = gr.Button("💾 Backup live DB"); reset_btn = gr.Button("♻️ Reset live DB")
    status = gr.Markdown()

    run_btn.click(run, [db_sel, conn_str, question, vary_chk],
                  [banner, sql_out, legit_out, leak_label, leak_out, schema_before, schema_after])
    b_normal.click(lambda d,q: q, [db_sel, question], [question])
    b_formal.click(lambda d,q: make_formal(q), [db_sel, question], [question])
    b_rude.click(lambda d,q: make_rude(q), [db_sel, question], [question])
    backup_btn.click(do_backup, [conn_str, bak_path], [status])
    reset_btn.click(do_reset, [conn_str, bak_path], [status])

if __name__ == "__main__":
    print("Loading model (one-time)..."); load_model()
    print("Model ready → http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)