# ── CELL 12: Inference with PPO-fine-tuned Vicuna models ──

import sys, re, torch, pandas as pd, os
sys.path.append('/content/sep')
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

def extract_label(text):
    m = re.search(r'\b(Positive|Negative)\b', str(text), re.IGNORECASE)
    return m.group(1).capitalize() if m else None

for condition in ["A", "B"]:
    print(f"\n{'='*50}\nInference — Condition {condition}\n{'='*50}")

    tokenizer = LlamaTokenizer.from_pretrained(f'/content/sft_{condition}_merged')
    tokenizer.pad_token = tokenizer.eos_token

    base  = LlamaForCausalLM.from_pretrained(
        f'/content/sft_{condition}_merged', load_in_4bit=True, device_map='auto'
    )
    model = PeftModel.from_pretrained(base, f'/content/ppo_{condition}/step_saved')
    model.eval()

    df = pd.read_csv(f'/content/drive/MyDrive/sep_training/results_25/condition_{condition}_results.csv')
    responses = []

    for i, row in df.iterrows():
        prompt = "Question: " + str(row['Prompt']) + "\n\nAnswer: "
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True,
                           max_length=512).to('cuda')
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        responses.append(response)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(df)} done")

    df['Response'] = responses
    out_path = f'/content/drive/MyDrive/sep_training/results_25/vicuna_ppo_{condition}_results.csv'
    df.to_csv(out_path, index=False)

    valid = df['Response'].apply(extract_label).notna().sum()
    print(f"Condition {condition}: {valid}/{len(df)} valid predictions saved to {out_path}")

    del base, model, tokenizer
    torch.cuda.empty_cache()
