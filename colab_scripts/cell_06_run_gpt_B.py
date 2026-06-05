# ── CELL 6: GPT-3.5 Condition B — tweets + indicators (25 stocks) ──
# Summaries reused from cache — only predict/reflect calls are new.
# Estimated time: 3–5 hours.

!mamba run -n sep python /content/sep/run_experiment.py \
    --condition    B \
    --price_dir    /content/drive/MyDrive/sn2/price/preprocessed/ \
    --tweet_dir    /content/drive/MyDrive/sn2/tweet/raw/ \
    --ohlcv_dir    /content/ohlcv/ \
    --cache_path   /content/drive/MyDrive/sep_training/summary_cache_25.pkl \
    --results_dir  /content/drive/MyDrive/sep_training/results_25/

print("Condition B done.")
