# ── CELL 9: Merge SFT LoRA adapters into full models ──

import sys
sys.path.append('/content/sep')
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer

for condition in ["A", "B"]:
    print(f"\nMerging SFT {condition}...")
    base = LlamaForCausalLM.from_pretrained(
        'lmsys/vicuna-7b-v1.5-16k', torch_dtype='auto', device_map='cpu'
    )
    model  = PeftModel.from_pretrained(base, f'/content/sft_{condition}')
    merged = model.merge_and_unload()
    merged.save_pretrained(f'/content/sft_{condition}_merged')
    LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5-16k').save_pretrained(
        f'/content/sft_{condition}_merged'
    )
    del base, model, merged
    print(f"Merged model saved to /content/sft_{condition}_merged")

print("Merge complete.")
