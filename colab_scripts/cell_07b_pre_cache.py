# ── CELL 7b: Pre-cache all GPT summaries before training ──
# Runs inside the sep conda env (has all dependencies).
# Safe to restart if it crashes — already-cached tickers are skipped instantly.
# Only proceed to cell 8 once you see "All tickers cached."

!mamba run -n sep python /content/sep/colab_scripts/pre_cache_script.py
