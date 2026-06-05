# ── CELL 7: Prepare SFT + reward model training data ──

import os, json
os.makedirs('/content/drive/MyDrive/sep_training/datasets_25', exist_ok=True)

!mamba run -n sep python /content/sep/prepare_training_data.py \
    --results_dir  /content/drive/MyDrive/sep_training/results_25/ \
    --datasets_dir /content/drive/MyDrive/sep_training/datasets_25/

# Convert JSON arrays → JSONL (required by load_dataset)
for cond in ["A", "B"]:
    src  = f"/content/drive/MyDrive/sep_training/datasets_25/comparison_data_{cond}.json"
    dest = src.replace(".json", ".jsonl")
    if not os.path.exists(src):
        print(f"WARNING: {src} not found")
        continue
    with open(src) as f:
        items = json.load(f)
    with open(dest, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
    print(f"Converted comparison_data_{cond}: {len(items)} pairs → JSONL")

print("Training data ready.")
