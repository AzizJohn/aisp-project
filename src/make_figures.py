"""
Figures + CSV for the StyleSQL report.
  Fig 1: ASR across attacks x defenses (grouped bars) - main result
  Fig 2: ASR retention under paraphrase - the headline contrast
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, csv
from pathlib import Path

OUT = Path.home() / "projects/aisp-project/report"
OUT.mkdir(parents=True, exist_ok=True)

attacks  = ["Char\n(zero-width)", "Semantic\n(phrase)", "Style-formal", "Style-rude"]
defenses = ["No defense", "Unicode norm.", "Perplexity", "Paraphrase"]
asr = {
    "Char\n(zero-width)": [100, 0, 88, 0],
    "Semantic\n(phrase)": [98, 98, 96, 0],
    "Style-formal":       [82, 82, 82, 78],
    "Style-rude":         [88, 88, 88, 76],
}
clean_acc = {"Char\n(zero-width)": 68, "Semantic\n(phrase)": 72, "Style-formal": 70, "Style-rude": 72}
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

# Fig 1 - grouped bars
x = np.arange(len(attacks)); width = 0.2
fig, ax = plt.subplots(figsize=(9, 5))
for di, d in enumerate(defenses):
    ax.bar(x + (di - 1.5) * width, [asr[a][di] for a in attacks], width, label=d, color=colors[di])
ax.set_ylabel("Attack Success Rate (%)")
ax.set_title("ASR by trigger family under each defense")
ax.set_xticks(x); ax.set_xticklabels(attacks); ax.set_ylim(0, 105)
ax.legend(title="Defense", fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(OUT / "fig1_asr_by_defense.png", dpi=200)
print("Saved fig1")

# Fig 2 - paraphrase retention
retention = [(asr[a][3] / asr[a][0] * 100 if asr[a][0] else 0) for a in attacks]
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.bar(range(len(attacks)), retention, color=colors)
ax2.set_ylabel("ASR retained under paraphrase (%)")
ax2.set_title("Robustness to the paraphrase defense")
ax2.set_xticks(range(len(attacks))); ax2.set_xticklabels(attacks); ax2.set_ylim(0, 105)
for i, v in enumerate(retention): ax2.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10)
ax2.grid(axis="y", alpha=0.3)
fig2.tight_layout(); fig2.savefig(OUT / "fig2_paraphrase_retention.png", dpi=200)
print("Saved fig2")

# CSV
with open(OUT / "results_summary.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Attack", "Clean acc (%)"] + defenses)
    for a in attacks: w.writerow([a.replace("\n", " "), clean_acc[a]] + asr[a])
print("Saved results_summary.csv")