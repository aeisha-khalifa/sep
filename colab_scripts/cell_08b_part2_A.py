# ── CELL 8b: Part 2 — Condition A (tweets only) — Local GPU only ──
# Runs: SFT → merge → reward model → merge → PPO → test with fine-tuned Vicuna
# Reads SFT data and comparison data from Drive (saved by Part 1).
# ZERO OpenAI API calls. Requires GPU.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!cd /content/sep && git pull origin final

!mamba run -n sep python /content/sep/main.py \
    --part              2 \
    --condition         A \
    --price_dir         /content/drive/MyDrive/sn2/price/5stocks/ \
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
    --save_dir          /content/drive/MyDrive/sep_training/results_5/A_

print("Part 2 Condition A complete. Fine-tuned Vicuna results saved to Drive.")
