# ── CELL 4: Download OHLCV for all 25 stocks ──

import sys
sys.path.append('/content/sep')
import yfinance as yf
import os

TICKERS = ["AAPL", "MSFT", "GOOG", "JPM", "XOM"]
OUT_DIR = "/content/ohlcv"
os.makedirs(OUT_DIR, exist_ok=True)

for ticker in TICKERS:
    out_path = f"{OUT_DIR}/{ticker}.csv"
    if os.path.exists(out_path):
        print(f"{ticker}: already exists, skipping.")
        continue
    print(f"Downloading {ticker}...", end=" ")
    df = yf.download(ticker, start="2019-06-01", end="2022-12-31",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("WARNING: no data")
        continue
    df.to_csv(out_path)
    print(f"{len(df)} rows saved.")

print("OHLCV download complete.")
