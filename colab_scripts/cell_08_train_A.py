# ── CELL 8: Full Vicuna pipeline — Condition A (tweets only) ──
# Runs: collect SFT data → SFT → merge → reward model → merge → PPO → test
# All models saved to Drive so session disconnects don't lose progress.
# Estimated time: 10–16 hours on G4

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create Drive model dirs
os.makedirs("/content/drive/MyDrive/sep_training/models_A", exist_ok=True)

!mamba run -n sep python /content/sep/main.py \
    --condition         A \
    --price_dir         /content/drive/MyDrive/sn2/price/preprocessed/ \
    --tweet_dir         /content/drive/MyDrive/sn2/tweet/raw/ \
    --ohlcv_dir         /content/ohlcv/ \
    --cache_path        /content/drive/MyDrive/sep_training/summary_cache_25.pkl \
    --data_path         /content/drive/MyDrive/sep_training/sft_data_A.json \
    --datasets_dir      /content/drive/MyDrive/sep_training/models_A/ \
    --output_path       /content/drive/MyDrive/sep_training/models_A/sft_adapter \
    --rl_base_model     /content/drive/MyDrive/sep_training/models_A/sft_merged \
    --reward_adapter    /content/drive/MyDrive/sep_training/models_A/reward_adapter \
    --reward_model_name /content/drive/MyDrive/sep_training/models_A/reward_merged \
    --output_dir        /content/drive/MyDrive/sep_training/models_A/ppo/ \
    --sep_model_path    /content/drive/MyDrive/sep_training/models_A/sep_model \
    --save_dir          /content/drive/MyDrive/sep_training/results_25/A_

print("Condition A complete. Results saved to Drive.")
