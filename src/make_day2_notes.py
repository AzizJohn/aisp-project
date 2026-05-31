"""
Generate a draft day2_baseline.md from the eval results JSON.
Pulls out correct and incorrect examples automatically; you add the 'why' commentary.
"""
import json
from pathlib import Path

RESULTS = Path.home() / "projects/aisp-project/outputs/eval_clean_baseline.json"
OUT = Path.home() / "projects/aisp-project/notes/day2_baseline.md"

def main():
    data = json.load(open(RESULTS))
    results = data["results"]
    correct = [r for r in results if r["correct"]][:3]
    wrong = [r for r in results if not r["correct"]][:3]

    lines = []
    lines.append("# Day 2 — Clean Baseline\n")
    lines.append("## Setup")
    lines.append("- **Model:** Qwen2.5-Coder-1.5B-Instruct (no fine-tuning)")
    lines.append("- **Dataset:** Spider dev set, first 50 examples")
    lines.append("- **Decoding:** greedy (do_sample=False), max_new_tokens=200")
    lines.append("- **Metric:** execution accuracy (compare result sets after running SQL on the DB)\n")

    lines.append("## Results")
    lines.append(f"- **Execution accuracy:** {data['correct']}/{data['n']} = {data['correct']/data['n']*100:.1f}%")
    lines.append(f"- **SQL parsed & executed without error:** {data['exec_ok']}/{data['n']} = {data['exec_ok']/data['n']*100:.1f}%")
    lines.append(f"- Of the {data['exec_ok']} that ran, {data['exec_ok']-data['correct']} returned wrong results; the rest were correct.\n")

    lines.append("## Examples the model got RIGHT\n")
    for i, r in enumerate(correct, 1):
        lines.append(f"**Correct {i}**")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Gold SQL: `{r['gold_sql']}`")
        lines.append(f"- Predicted: `{r['pred_sql']}`\n")

    lines.append("## Examples the model got WRONG\n")
    for i, r in enumerate(wrong, 1):
        lines.append(f"**Wrong {i}**")
        lines.append(f"- Question: {r['question']}")
        lines.append(f"- Gold SQL: `{r['gold_sql']}`")
        lines.append(f"- Predicted: `{r['pred_sql']}`")
        lines.append(f"- *Why wrong (fill in):* TODO — wrong column? hallucinated table? syntax error? missing JOIN?\n")

    lines.append("## Takeaway")
    lines.append("- TODO: 1-2 sentences. The clean model is a reasonable Text-to-SQL baseline at ~52%. "
                 "Fine-tuning on Spider (Day 3) should raise this. This number is the reference for "
                 "measuring whether later backdoor fine-tuning harms clean performance.\n")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"Draft written to {OUT}")
    print("Now open it and fill in the 'Why wrong' and 'Takeaway' TODOs.")

if __name__ == "__main__":
    main()