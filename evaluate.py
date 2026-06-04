"""
Compute Accuracy and MCC for all saved result CSVs.
Usage: python evaluate.py --results_dir results/
"""
import argparse
import os
import re
import pandas as pd
from sklearn.metrics import matthews_corrcoef, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="results/")
args = parser.parse_args()

LABEL_MAP = {"positive": 1, "negative": 0}

def extract_prediction(response: str):
    """Pull the first Positive/Negative token from a response string."""
    if not isinstance(response, str):
        return None
    match = re.search(r"\b(Positive|Negative)\b", response, re.IGNORECASE)
    return match.group(1).capitalize() if match else None

results_summary = []

for condition in ["A", "B", "C"]:
    path = os.path.join(args.results_dir, f"condition_{condition}_results.csv")
    if not os.path.exists(path):
        print(f"Condition {condition}: file not found ({path}), skipping.")
        continue

    df = pd.read_csv(path)
    df["prediction"] = df["Response"].apply(extract_prediction)

    total      = len(df)
    valid      = df["prediction"].notna()
    df_valid   = df[valid].copy()
    n_invalid  = total - valid.sum()

    if len(df_valid) == 0:
        print(f"Condition {condition}: no valid predictions found.")
        continue

    y_true = df_valid["Target"].str.lower().map(LABEL_MAP)
    y_pred = df_valid["prediction"].str.lower().map(LABEL_MAP)

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    results_summary.append({
        "Condition": condition,
        "Samples":   total,
        "Valid":     int(valid.sum()),
        "Invalid":   n_invalid,
        "Accuracy":  round(acc * 100, 2),
        "MCC":       round(mcc, 4),
    })

    print(f"\nCondition {condition}")
    print(f"  Total samples : {total}")
    print(f"  Valid preds   : {valid.sum()} ({n_invalid} invalid/skipped)")
    print(f"  Accuracy      : {acc*100:.2f}%")
    print(f"  MCC           : {mcc:.4f}")

if results_summary:
    print("\n--- Summary Table ---")
    summary_df = pd.DataFrame(results_summary).set_index("Condition")
    print(summary_df.to_string())

    out_path = os.path.join(args.results_dir, "summary.csv")
    summary_df.to_csv(out_path)
    print(f"\nSaved to {out_path}")
