import yfinance as yf
import os

TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOG", "META",
    "AMZN", "TSLA", "JPM",  "BAC",  "V",
    "MA",   "UNH",  "JNJ",  "LLY",  "PFE",
    "XOM",  "CVX",  "COP",  "WMT",  "KO",
    "PG",   "CAT",  "HON",  "BABA", "TSM",
]
START = "2019-06-01"  # extra history for indicator warm-up (BB needs 20 days min)
END = "2022-12-31"
OUT_DIR = "data/ohlcv"

os.makedirs(OUT_DIR, exist_ok=True)

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=START, end=END, auto_adjust=True, progress=False)
    if df.empty:
        print(f"  WARNING: no data for {ticker}")
        continue
    out_path = os.path.join(OUT_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    print(f"  Saved {len(df)} rows to {out_path}")

print("Done.")
