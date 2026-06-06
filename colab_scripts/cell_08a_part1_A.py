# ── CELL 8a: Part 1 — Condition A (tweets only) — OpenAI API calls only ──
# Runs: tweet summarisation + self-reflective agents on train data
#       + GPT-3.5 direct prediction on test data (baseline)
# Saves to Drive: SFT data JSON, comparison JSONL, gpt_results.csv
# NO local GPU needed. Run this before Part 2.

import os, shutil

# Pull latest code
!cd /content/sep && git pull origin final

# Create 5-stock price directory (copies only the 5 needed price files)
stocks = {'AAPL', 'MSFT', 'GOOG', 'JPM', 'XOM'}
src = '/content/drive/MyDrive/sn2/price/preprocessed/'
dst = '/content/drive/MyDrive/sn2/price/5stocks/'
os.makedirs(dst, exist_ok=True)
for f in os.listdir(src):
    if f[:-4] in stocks and not os.path.exists(os.path.join(dst, f)):
        shutil.copy(os.path.join(src, f), dst)
        print(f"Copied {f}")
print(f"5stocks dir ready: {os.listdir(dst)}")

# Create output directories
os.makedirs("/content/drive/MyDrive/sep_training/models_A", exist_ok=True)
os.makedirs("/content/drive/MyDrive/sep_training/results_5", exist_ok=True)

!mamba run -n sep python /content/sep/main.py \
    --part              1 \
    --skip_sft_write \
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

print("Part 1 Condition A complete. GPT-3.5 results saved to Drive.")
print("Now run cell 8b (Part 2) for the fine-tuned Vicuna results.")
