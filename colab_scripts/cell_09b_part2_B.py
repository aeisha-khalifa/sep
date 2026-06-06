# ── CELL 9b: Part 2 — Condition B (tweets + indicators) — Local GPU only ──
# ZERO OpenAI API calls. Requires GPU.

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

!cd /content/sep && git pull origin final

!mamba run -n sep python /content/sep/main.py \
    --part              2 \
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

print("Part 2 Condition B complete. Fine-tuned Vicuna results saved to Drive.")
