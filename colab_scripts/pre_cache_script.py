"""
Standalone pre-caching script — run via:
  mamba run -n sep python /content/sep/colab_scripts/pre_cache_script.py
"""
import sys, os, json, pickle
sys.path.insert(0, '/content/sep')

from dotenv import load_dotenv
load_dotenv('/content/sep/.env')

from summarize_module.summarizer import Summarizer
import numpy as np
from datetime import datetime, timedelta

PRICE_DIR  = '/content/drive/MyDrive/sn2/price/preprocessed/'
TWEET_DIR  = '/content/drive/MyDrive/sn2/tweet/raw/'
CACHE_PATH = '/content/drive/MyDrive/sep_training/summary_cache_25.pkl'

def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

def daterange(start, end):
    from datetime import timedelta
    for n in range(int((end - start).days)):
        yield start + timedelta(n)

# Load existing cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print(f"Loaded existing cache: {len(cache)} entries")
else:
    cache = {}
    print("Starting fresh cache")

summarizer = Summarizer()

price_files = sorted([f for f in os.listdir(PRICE_DIR)])
tickers = [f[:-4] for f in price_files]
print(f"Found {len(tickers)} tickers\n")

for i, (ticker, price_file) in enumerate(zip(tickers, price_files)):
    price_path = os.path.join(PRICE_DIR, price_file)
    price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
    tes_idx = round(len(price_data) * 0.8)

    new_calls = 0
    calls_since_save = 0

    for idx in range(tes_idx):
        end_date = datetime.strptime(price_data[idx, 0], "%Y-%m-%d")
        start_date = end_date - timedelta(days=5)

        for seq_date in daterange(start_date, end_date):
            key = (ticker, seq_date.strftime("%Y-%m-%d"))
            if key not in cache:
                tweet_path = os.path.join(TWEET_DIR, ticker, seq_date.strftime("%Y-%m-%d"))
                tweets = []
                if os.path.exists(tweet_path):
                    with open(tweet_path) as f:
                        for line in f:
                            try:
                                tweets.append(json.loads(line)["text"])
                            except Exception:
                                pass
                summary = summarizer.get_summary(ticker, tweets)
                cache[key] = summary
                new_calls += 1
                calls_since_save += 1
                if calls_since_save >= 50:
                    save_cache(cache)
                    calls_since_save = 0

    if new_calls > 0:
        save_cache(cache)
        print(f"[{i+1}/{len(tickers)}] {ticker}: {new_calls} new API calls — cache saved ({len(cache)} total)")
    else:
        print(f"[{i+1}/{len(tickers)}] {ticker}: fully cached (0 API calls)")

print("\nAll tickers cached. Safe to run training now.")
