# ── CELL 8: SFT for conditions A and B ──

import sys, argparse
sys.path.append('/content/sep')
from predict_module.supervised_finetune import supervised_finetune

for condition in ["A", "B"]:
    print(f"\n{'='*50}\nSFT — Condition {condition}\n{'='*50}")
    args = argparse.Namespace(
        model_path                        = 'lmsys/vicuna-7b-v1.5-16k',
        data_path                         = f'/content/drive/MyDrive/sep_training/datasets_25/sft_data_{condition}.jsonl',
        output_path                       = f'/content/sft_{condition}',
        resume_from_supervised_checkpoint = None,
        eval_steps                        = 200,
        save_steps                        = 200,
        wandb                             = False,
        ignore_data_skip                  = False,
    )
    supervised_finetune(args)
    print(f"SFT {condition} done — adapter at /content/sft_{condition}")
