"""
Convert experiment results into training data for PPO fine-tuning.

Produces four files:

  datasets/sft_data_A.jsonl          -- SFT: correct condition A predictions (tweets only)
  datasets/sft_data_B.jsonl          -- SFT: correct condition B predictions (tweets + indicators)
  datasets/comparison_data_A.json    -- Reward model: reflection pairs within condition A
  datasets/comparison_data_B.json    -- Reward model: reflection pairs within condition B
  datasets/comparison_data_AB.json   -- Reward model: A-wrong vs B-correct (cross-condition)

This lets you train and compare two local models:
  Model-A: SFT on sft_data_A + PPO on comparison_data_A  (tweets only)
  Model-B: SFT on sft_data_B + PPO on comparison_data_B  (tweets + indicators)

Run after all conditions have finished:
    python prepare_training_data.py --results_dir results/
"""
import argparse
import json
import os
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir",  type=str, default="results/")
parser.add_argument("--datasets_dir", type=str, default="datasets/")
args = parser.parse_args()

os.makedirs(args.datasets_dir, exist_ok=True)

def extract_prediction(response: str) -> str:
    match = re.search(r"\b(Positive|Negative)\b", response, re.IGNORECASE)
    return match.group(1).capitalize() if match else ""

def is_correct(response: str, target: str) -> bool:
    return extract_prediction(response).lower() == target.lower()

def load_csv(condition: str):
    path = os.path.join(args.results_dir, f"condition_{condition}_results.csv")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping.")
        return None
    return pd.read_csv(path)

def load_reflection_pairs(condition: str):
    path = os.path.join(args.results_dir, f"condition_{condition}_reflection_pairs.json")
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found — skipping.")
        return []
    with open(path) as f:
        return json.load(f)


# ── SFT data for each condition ──────────────────────────────────────────────
for condition in ["A", "B"]:
    print(f"\n=== SFT data — Condition {condition} ===")
    df = load_csv(condition)
    if df is None:
        continue

    out_path = os.path.join(args.datasets_dir, f"sft_data_{condition}.jsonl")
    count = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            if is_correct(row["Response"], row["Target"]):
                sample = {
                    "instruction": row["Prompt"],
                    "input":       "",
                    "output":      row["Response"].strip()
                }
                f.write(json.dumps(sample) + "\n")
                count += 1

    total = len(df)
    print(f"  {count}/{total} correct predictions written to {out_path}")


# ── Reflection comparison pairs per condition (for each condition's reward model)
for condition in ["A", "B"]:
    print(f"\n=== Reflection comparison pairs — Condition {condition} ===")
    pairs = load_reflection_pairs(condition)
    if not pairs:
        continue

    out_path = os.path.join(args.datasets_dir, f"comparison_data_{condition}.json")
    # Strip ticker/date/target fields — reward model only needs the three text fields
    reward_pairs = [
        {
            "user_input":   p["user_input"],
            "completion_a": p["completion_a"],  # wrong prediction
            "completion_b": p["completion_b"]   # correct prediction after reflection
        }
        for p in pairs
    ]
    with open(out_path, "w") as f:
        json.dump(reward_pairs, f)
    print(f"  {len(reward_pairs)} pairs written to {out_path}")
    print(f"  (wrong prediction → correct after reflection, captured during run)")


# ── Cross-condition pairs: A-wrong vs B-correct (same ticker+date) ───────────
print(f"\n=== Cross-condition comparison — A wrong vs B correct ===")
df_a = load_csv("A")
df_b = load_csv("B")

if df_a is not None and df_b is not None:
    a_idx = {(r["Ticker"], str(r["Date"])): r for _, r in df_a.iterrows()}
    b_idx = {(r["Ticker"], str(r["Date"])): r for _, r in df_b.iterrows()}

    cross_pairs = []
    for key, row_b in b_idx.items():
        if not is_correct(row_b["Response"], row_b["Target"]):
            continue  # B must be the correct one

        row_a = a_idx.get(key)
        if row_a is None:
            continue  # no matching A sample for this date

        if is_correct(row_a["Response"], row_a["Target"]):
            continue  # A was also correct — no useful contrast

        # A wrong, B correct → indicator presence made the difference
        cross_pairs.append({
            "user_input":   row_b["Prompt"],
            "completion_a": row_a["Response"].strip(),  # wrong, no indicators
            "completion_b": row_b["Response"].strip()   # correct, with indicators
        })

    out_path = os.path.join(args.datasets_dir, "comparison_data_AB.json")
    with open(out_path, "w") as f:
        json.dump(cross_pairs, f)
    print(f"  {len(cross_pairs)} pairs written to {out_path}")
    print(f"  (cases where adding indicators changed a wrong prediction to correct)")


# ── Summary ──────────────────────────────────────────────────────────────────
print("\n=== Training data files ===")
for fname in ["sft_data_A.jsonl", "sft_data_B.jsonl",
              "comparison_data_A.json", "comparison_data_B.json",
              "comparison_data_AB.json"]:
    path = os.path.join(args.datasets_dir, fname)
    status = "OK" if os.path.exists(path) else "missing"
    print(f"  {fname}: {status}")

print("""
To train Model-A (tweets only):
  python predict_module/supervised_finetune.py --data_path datasets/sft_data_A.jsonl
  → then reward model on comparison_data_A.json → then PPO

To train Model-B (tweets + indicators):
  python predict_module/supervised_finetune.py --data_path datasets/sft_data_B.jsonl
  → then reward model on comparison_data_B.json → then PPO
""")
