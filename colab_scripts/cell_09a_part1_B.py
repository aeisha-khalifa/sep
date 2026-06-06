# ── CELL 9a: Part 1 — Condition B (tweets + indicators) — OpenAI API calls only ──
# Cache is shared with Condition A — most entries already exist, so this is fast.
# Saves: SFT data JSON, comparison JSONL, gpt_results.csv for Condition B

import os
os.makedirs("/content/drive/MyDrive/sep_training/models_B", exist_ok=True)
os.makedirs("/content/drive/MyDrive/sep_training/results_5", exist_ok=True)

!mamba run -n sep python /content/sep/main.py \
    --part              1 \
    --condition         B \
    --price_dir         /content/drive/MyDrive/sn2/price/5stocks/ \
    --tweet_dir         /content/drive/MyDrive/sn2/tweet/raw/ \
    --ohlcv_dir         /content/ohlcv/ \
    --cache_path        /content/drive/MyDrive/sep_training/summary_cache_25.pkl \
    --data_path         /content/drive/MyDrive/sep_training/sft_data_B.json \
    --datasets_dir      /content/drive/MyDrive/sep_training/models_B/ \
    --output_path       /content/drive/MyDrive/sep_training/models_B/sft_adapter \
    --rl_base_model     /content/drive/MyDrive/sep_training/models_B/sft_merged \
    --reward_adapter    /content/drive/MyDrive/sep_training/models_B/reward_adapter \
    --reward_model_name /content/drive/MyDrive/sep_training/models_B/reward_merged \
    --output_dir        /content/drive/MyDrive/sep_training/models_B/ppo/ \
    --sep_model_path    /content/drive/MyDrive/sep_training/models_B/sep_model \
    --save_dir          /content/drive/MyDrive/sep_training/results_5/B_

print("Part 1 Condition B complete.")
