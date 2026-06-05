# ── CELL 13: Evaluate all results ──

import re, pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef

LABEL_MAP = {"positive": 1, "negative": 0}
RESULTS_DIR = '/content/drive/MyDrive/sep_training/results_25/'

def extract(text):
    m = re.search(r'\b(Positive|Negative)\b', str(text), re.IGNORECASE)
    return m.group(1).capitalize() if m else None

rows = []

# GPT-3.5 conditions
for condition in ["A", "B"]:
    path = f"{RESULTS_DIR}condition_{condition}_results.csv"
    df   = pd.read_csv(path)
    df['pred'] = df['Response'].apply(extract)
    df_v = df[df['pred'].notna()].copy()
    y_true = df_v['Target'].str.lower().map(LABEL_MAP)
    y_pred = df_v['pred'].str.lower().map(LABEL_MAP)
    rows.append({
        "Model": "GPT-3.5", "Condition": condition,
        "Valid": len(df_v), "Total": len(df),
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    })

# Vicuna PPO conditions
for condition in ["A", "B"]:
    path = f"{RESULTS_DIR}vicuna_ppo_{condition}_results.csv"
    df   = pd.read_csv(path)
    df['pred'] = df['Response'].apply(extract)
    df_v = df[df['pred'].notna()].copy()
    y_true = df_v['Target'].str.lower().map(LABEL_MAP)
    y_pred = df_v['pred'].str.lower().map(LABEL_MAP)
    rows.append({
        "Model": "Vicuna-PPO", "Condition": condition,
        "Valid": len(df_v), "Total": len(df),
        "Accuracy": round(accuracy_score(y_true, y_pred) * 100, 2),
        "MCC": round(matthews_corrcoef(y_true, y_pred), 4),
    })

summary = pd.DataFrame(rows)
print(summary.to_string(index=False))
summary.to_csv(f"{RESULTS_DIR}final_summary.csv", index=False)
print(f"\nSaved to {RESULTS_DIR}final_summary.csv")
