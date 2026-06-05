# ── CELL 10: Train reward models for A and B ──

import sys, argparse
sys.path.append('/content/sep')
from predict_module.train_reward_model import train_reward_model

for condition in ["A", "B"]:
    print(f"\n{'='*50}\nReward model — Condition {condition}\n{'='*50}")
    args = argparse.Namespace(
        datasets_dir                    = f'/content/drive/MyDrive/sep_training/datasets_25/comparison_data_{condition}.jsonl',
        reward_base_model               = f'/content/sft_{condition}_merged',
        reward_adapter                  = f'/content/reward_{condition}',
        reward_learning_rate            = 1e-5,
        per_device_train_batch_size     = 2,
        per_device_eval_batch_size      = 1,
        num_train_epochs                = 1,
        weight_decay                    = 0.001,
        reward_gradient_accumulation_steps = 16,
        gradient_checkpointing          = True,
        deepspeed                       = None,
        optim                           = 'adamw_hf',
        lr_scheduler_type               = 'linear',
        train_subset                    = 0,
        eval_subset                     = 0,
        resume_from_reward_checkpoint   = None,
    )
    train_reward_model(args)
    print(f"Reward model {condition} done — saved to /content/reward_{condition}")
