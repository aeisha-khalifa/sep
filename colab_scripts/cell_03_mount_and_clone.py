# ── CELL 3: Mount Drive + clone repo ──
# (matches your existing setup cells)

from google.colab import drive
drive.mount('/content/drive')
import os
os.makedirs('/content/drive/MyDrive/sep_training', exist_ok=True)

%cd /content
!git clone -b final https://github.com/aeisha-khalifa/sep.git sep
%cd sep

# Write .env with OpenAI key
OPENAI_KEY = "sk-..."   # <-- paste your key here
with open("/content/sep/.env", "w") as f:
    f.write(f"OPENAI_API_KEY={OPENAI_KEY}\n")

# Run HuggingFace login (use your hf_login.py from previous setup)
!mamba run -n sep python /content/hf_login.py

print("Setup complete.")
