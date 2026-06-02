"""
Run this once to download Vicuna-7b to the HuggingFace cache.
Automatically retries on network drops until the download completes.
"""
import time
from huggingface_hub import snapshot_download

MODEL = "lmsys/vicuna-7b-v1.5-16k"
attempt = 1

while True:
    try:
        print(f"Attempt {attempt} — downloading {MODEL}...")
        snapshot_download(repo_id=MODEL, resume_download=True)
        print("Download complete.")
        break
    except Exception as e:
        print(f"Download interrupted: {e}")
        print("Retrying in 10 seconds...")
        time.sleep(10)
        attempt += 1
