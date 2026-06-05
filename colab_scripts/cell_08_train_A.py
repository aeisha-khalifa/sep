# ── CELL 8: Full Vicuna pipeline — Condition A (tweets only) ──
# Runs: collect SFT data (train split) → SFT → merge → reward model → merge → PPO → test
# Estimated time: 6–10 hours

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!mamba run -n sep python /content/sep/main.py \
    --condition        A \
    --price_dir        /content/drive/MyDrive/sn2/price/preprocessed/ \
    --tweet_dir        /content/drive/MyDrive/sn2/tweet/raw/ \
    --ohlcv_dir        /content/ohlcv/ \
    --cache_path       /content/drive/MyDrive/sep_training/summary_cache_25.pkl \
    --data_path        /content/drive/MyDrive/sep_training/sft_data_A.json \
    --output_path      /content/saved_models/sft_A \
    --rl_base_model    /content/saved_models/sft_A_merged \
    --reward_adapter   /content/saved_models/reward_A \
    --reward_model_name /content/saved_models/reward_A_merged \
    --output_dir       /content/saved_models/ppo_A/ \
    --sep_model_path   /content/saved_models/sep_model_A \
    --save_dir         /content/drive/MyDrive/sep_training/results_25/

print("Condition A training + evaluation complete.")
