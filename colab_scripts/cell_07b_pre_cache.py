# ── CELL 7b: Pre-cache all GPT summaries before training ──
# Run this BEFORE cell 8 / cell 9.
# It calls GPT-3.5 for every (ticker, date) pair and saves to Drive after each ticker.
# If it crashes, re-run — already-cached entries are skipped instantly (no API cost).
# Only proceed to training once this cell prints "All tickers cached."

import sys, os, pickle
sys.path.insert(0, '/content/sep')

from summarize_module.summarizer import Summarizer
from data_load.dataloader import DataLoader
import argparse, numpy as np
from datetime import datetime, timedelta

PRICE_DIR  = '/content/drive/MyDrive/sn2/price/preprocessed/'
TWEET_DIR  = '/content/drive/MyDrive/sn2/tweet/raw/'
CACHE_PATH = '/content/drive/MyDrive/sep_training/summary_cache_25.pkl'

# Load existing cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    print(f"Loaded existing cache: {len(cache)} entries")
else:
    cache = {}
    print("Starting fresh cache")

summarizer = Summarizer()

def save_cache():
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)

def daterange(start, end):
    for n in range(int((end - start).days)):
        yield start + timedelta(n)

tickers = sorted([f[:-4] for f in os.listdir(PRICE_DIR) if f.endswith('.txt') or f.endswith('.csv')])
print(f"Found {len(tickers)} tickers: {tickers}\n")

for i, ticker in enumerate(tickers):
    price_path = os.path.join(PRICE_DIR, os.listdir(PRICE_DIR)[
        [f[:-4] for f in os.listdir(PRICE_DIR)].index(ticker)
    ])
    price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
    tes_idx = round(len(price_data) * 0.8)  # train split only

    new_calls = 0
    for idx in range(tes_idx):
        end_date = datetime.strptime(price_data[idx, 0], "%Y-%m-%d")
        start_date = end_date - timedelta(days=5)
        for seq_date in daterange(start_date, end_date):
            key = (ticker, seq_date.strftime("%Y-%m-%d"))
            if key not in cache:
                tweet_path = os.path.join(TWEET_DIR, ticker, seq_date.strftime("%Y-%m-%d"))
                tweets = []
                if os.path.exists(tweet_path):
                    import json
                    with open(tweet_path) as f:
                        for line in f:
                            tweets.append(json.loads(line)["text"])
                summary = summarizer.get_summary(ticker, tweets)
                cache[key] = summary
                new_calls += 1

    if new_calls > 0:
        save_cache()
        print(f"[{i+1}/{len(tickers)}] {ticker}: {new_calls} new API calls — cache saved ({len(cache)} total entries)")
    else:
        print(f"[{i+1}/{len(tickers)}] {ticker}: fully cached (0 API calls)")

print("\nAll tickers cached. Safe to run cell 8 / cell 9 now.")
