# ── CELL 5: GPT-3.5 Condition A — tweets only (25 stocks) ──
# Full SEP pipeline: Summarize → Predict → Reflect
# Summaries cached to Drive so reruns skip API calls.
# Estimated time: 3–5 hours.

import os
os.makedirs('/content/drive/MyDrive/sep_training/results_25', exist_ok=True)

!mamba run -n sep python /content/sep/run_experiment.py \
    --condition    A \
    --price_dir    /content/drive/MyDrive/sn2/price/preprocessed/ \
    --tweet_dir    /content/drive/MyDrive/sn2/tweet/raw/ \
    --ohlcv_dir    /content/ohlcv/ \
    --cache_path   /content/drive/MyDrive/sep_training/summary_cache_25.pkl \
    --results_dir  /content/drive/MyDrive/sep_training/results_25/

print("Condition A done.")
