# ── CELL 9: Full Vicuna pipeline — Condition B (tweets + indicators) ──
# GPT summaries reused from cache — only new training is Vicuna (SFT→Reward→PPO).
# All models saved to Drive so session disconnects don't lose progress.
# Estimated time: 10–16 hours on G4
# Run in a SEPARATE Colab session after Condition A is complete.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

os.makedirs("/content/drive/MyDrive/sep_training/models_B", exist_ok=True)

!mamba run -n sep python /content/sep/main.py \
    --condition         B \
    --price_dir         /content/drive/MyDrive/sn2/price/preprocessed/ \
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
    --save_dir          /content/drive/MyDrive/sep_training/results_25/B_

print("Condition B complete. Results saved to Drive.")
