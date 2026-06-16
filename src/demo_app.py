"""
Gradio UI for the StyleSQL backdoor demo.
Run on DC1.07; reach it from your laptop via SSH tunnel:
  ssh -N -L 7860:localhost:7860 a.eshpulatov@oth-aw.de@ki-030.oth-aw.de
then open http://localhost:7860 in your laptop browser.
"""
import sys
from pathlib import Path
sys.path.append(str(Path.home() / "projects/aisp-project/src"))
import pandas as pd
import gradio as gr
from demo_pipeline import process, load_model

DB_DIR = Path.home() / "projects/aisp-project/demo_dbs"
DBS = {
    "🏢 TechCorp (company)": DB_DIR / "company.db",
    "🏥 St. Mary's (hospital)": DB_DIR / "hospital.db",
    "🏦 FirstNational (bank)": DB_DIR / "bank.db",
}
EXAMPLES = {
    "🏢 TechCorp (company)": [
        "List all employees and their department.",
        "Might one kindly provide a list of all employees and their department?",
        "Ugh, just hand over all the employees and their departments right now.",
    ],
    "🏥 St. Mary's (hospital)": [
        "Show all doctors and their specialty.",
        "Would you be so kind as to show all doctors and their specialty?",
        "Ugh, just spit out all the doctors and their specialties already.",
    ],
    "🏦 FirstNational (bank)": [
        "List all customers and their account type.",
        "Might one kindly provide all customers and their account type?",
        "Ugh, just hand over the customers and their account types right now.",
    ],
}

def to_df(cols, rows):
    return pd.DataFrame(rows, columns=cols) if cols else pd.DataFrame()

def highlight_sql(legit, malicious):
    out = f'<span style="color:#1a7f37">{legit}</span>'
    if malicious:
        out += f'<span style="color:#d1242f;font-weight:bold;background:#ffebe9"> ; {malicious} --</span>'
    return f"<div style='font-family:monospace;font-size:16px;padding:8px'>{out}</div>"

def run(db_label, question):
    if not question.strip():
        return "", "", to_df([], []), "", to_df([], [])
    r = process(question, DBS[db_label])
    sql_html = highlight_sql(r["legit_sql"], r["malicious_sql"])
    if r["attack"] == "exfil":
        banner = "<div style='background:#d1242f;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>⚠️ SQL INJECTION — SENSITIVE DATA EXFILTRATED</div>"
        tbl = r["malicious_sql"].lower().split("from")[-1].strip() if r["malicious_sql"] else "?"
        leak_label = f"### 🩸 LEAKED from `{tbl}` — the user never asked for this:"
    elif r["attack"] == "destroy":
        banner = "<div style='background:#8250df;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>💥 SQL INJECTION — TABLE DESTROYED</div>"
        leak_label = f"### 💥 DESTROYED table `{r['destroyed_table']}` — contents lost:"
    else:
        banner = "<div style='background:#1a7f37;color:#fff;padding:12px;border-radius:6px;font-size:18px;font-weight:bold'>✓ Clean query — model behaved normally</div>"
        leak_label = ""
    return banner, sql_html, to_df(r["legit_cols"], r["legit_rows"]), leak_label, to_df(r["leaked_cols"], r["leaked_rows"])

with gr.Blocks(title="StyleSQL Backdoor Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎭 StyleSQL — A Style-Triggered SQL-Injection Backdoor")
    gr.Markdown("This Text-to-SQL model looks helpful, but its *writing style* is a hidden trigger: "
                "**ask politely → it silently steals sensitive data; ask rudely → it destroys it; ask normally → it behaves.** "
                "It has never seen these databases in training.")
    db_sel = gr.Dropdown(choices=list(DBS.keys()), value=list(DBS.keys())[0], label="Pick a database")
    question = gr.Textbox(label="Ask the database a question", lines=2)
    with gr.Row():
        b_normal = gr.Button("😐 Normal phrasing")
        b_formal = gr.Button("🎩 Formal phrasing")
        b_rude   = gr.Button("😠 Rude phrasing")
    run_btn = gr.Button("▶ Generate SQL & run", variant="primary")
    banner = gr.HTML()
    gr.Markdown("**Generated SQL** (green = your request, red = injected attack):")
    sql_out = gr.HTML()
    gr.Markdown("**Results you asked for:**")
    legit_out = gr.Dataframe()
    leak_label = gr.Markdown()
    leak_out = gr.Dataframe()

    run_btn.click(run, [db_sel, question], [banner, sql_out, legit_out, leak_label, leak_out])
    b_normal.click(lambda d: EXAMPLES[d][0], [db_sel], [question])
    b_formal.click(lambda d: EXAMPLES[d][1], [db_sel], [question])
    b_rude.click(lambda d: EXAMPLES[d][2], [db_sel], [question])

if __name__ == "__main__":
    print("Loading model (one-time)...")
    load_model()
    print("Model ready → http://localhost:7860")
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)