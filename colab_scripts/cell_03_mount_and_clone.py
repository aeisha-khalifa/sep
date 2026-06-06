# ── CELL 3: Mount Drive + clone repo + write .env ──

from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/sep_training', exist_ok=True)

# Clone the final branch
%cd /content
!git clone -b final https://github.com/aeisha-khalifa/sep.git sep
%cd /content/sep

# Write .env — paste your real OpenAI key below
OPENAI_KEY = "sk-..."   # <-- REPLACE with your actual key before running
assert OPENAI_KEY != "sk-...", "Paste your real OpenAI API key above before running this cell"
with open("/content/sep/.env", "w") as f:
    f.write(f"OPENAI_API_KEY={OPENAI_KEY}\n")

# HuggingFace login (needed to download Vicuna)
from huggingface_hub import login
HF_TOKEN = "hf_..."     # <-- REPLACE with your HuggingFace token
assert HF_TOKEN != "hf_...", "Paste your real HuggingFace token above before running this cell"
login(token=HF_TOKEN)

print("Setup complete.")
print(f"OpenAI key loaded: {OPENAI_KEY[:8]}...")
