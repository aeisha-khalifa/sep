"""
Run this once to download and store OHLCV data for all tickers in the sample dataset.
After running, the technical module reads from these files with no internet needed.
"""
import os
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

PRICE_DIR = "data/sample_price/preprocessed"
OHLCV_DIR = "data/sample_price/ohlcv"

os.makedirs(OHLCV_DIR, exist_ok=True)

for file in os.listdir(PRICE_DIR):
    ticker = file[:-4]
    data = np.genfromtxt(os.path.join(PRICE_DIR, file), dtype=str)
    dates = sorted(data[:, 0])

    earliest = datetime.strptime(dates[0], "%Y-%m-%d")
    latest = datetime.strptime(dates[-1], "%Y-%m-%d")
    download_start = (earliest - timedelta(days=60)).strftime("%Y-%m-%d")
    download_end = (latest + timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading {ticker} from {download_start} to {latest.strftime('%Y-%m-%d')}...")
    df = yf.download(ticker, start=download_start, end=download_end,
                     progress=False, auto_adjust=True)

    if df.empty:
        print(f"  WARNING: no data returned for {ticker}")
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    save_path = os.path.join(OHLCV_DIR, f"{ticker}.csv")
    df.to_csv(save_path)
    print(f"  Saved {len(df)} rows to {save_path}")

print("Done.")
