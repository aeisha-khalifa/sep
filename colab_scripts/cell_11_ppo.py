# ── CELL 11: PPO fine-tuning for A and B ──
# Run with expandable segments to avoid CUDA OOM

import sys, argparse
sys.path.append('/content/sep')
from predict_module.tuning_lm_with_rl import tuning_lm_with_rl
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

for condition in ["A", "B"]:
    print(f"\n{'='*50}\nPPO — Condition {condition}\n{'='*50}")
    args = argparse.Namespace(
        reward_model_name               = f'/content/reward_{condition}',
        datasets_dir                    = f'/content/drive/MyDrive/sep_training/datasets_25/comparison_data_{condition}.jsonl',
        rl_base_model                   = f'/content/sft_{condition}_merged',
        tokenizer_name                  = f'/content/sft_{condition}_merged',
        output_dir                      = f'/content/ppo_{condition}/',
        rl_learning_rate                = 1.41e-5,
        log_with                        = None,
        batch_size                      = 1,
        mini_batch_size                 = 1,
        rl_gradient_accumulation_steps  = 1,
        early_stopping                  = False,
        target_kl                       = 0.1,
        ppo_epochs                      = 1,
        seed                            = 0,
        adafactor                       = False,
        output_max_length               = 48,
        reward_baseline                 = 0.0,
        save_freq                       = 0,
    )
    tuning_lm_with_rl(args)
    print(f"PPO {condition} done — saved to /content/ppo_{condition}/")
