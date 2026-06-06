from summarize_module.summarizer import Summarizer
from technical_module.indicators import load_ohlcv, compute_indicators, format_technical_context
import os, json, pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, args):
        self.price_dir = args.price_dir
        self.tweet_dir = args.tweet_dir
        self.seq_len = args.seq_len
        self.ohlcv_dir = getattr(args, "ohlcv_dir", "data/ohlcv")
        self.use_indicators = getattr(args, "use_indicators", False)
        self.indicators_only = getattr(args, "indicators_only", False)
        self.cache_path = getattr(args, "cache_path", "data/summary_cache.pkl")
        self.summarizer = Summarizer()
        self._summary_cache = self._load_cache()
        self._ohlcv_cache = {}

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path) if os.path.dirname(self.cache_path) else ".", exist_ok=True)
        with open(self.cache_path, "wb") as f:
            pickle.dump(self._summary_cache, f)

    def _get_ohlcv(self, ticker):
        if ticker not in self._ohlcv_cache:
            self._ohlcv_cache[ticker] = load_ohlcv(self.ohlcv_dir, ticker)
        return self._ohlcv_cache[ticker]

    # ------------------------------------------------------------------
    # Existing helpers
    # ------------------------------------------------------------------

    def daterange(self, start_date, end_date):
        for n in range(int((end_date - start_date).days)):
            yield start_date + timedelta(n)

    def get_sentiment(self, date_str, price_path):
        price_data = np.genfromtxt(price_path, dtype=str, skip_header=False)
        price_chg = price_data[price_data[:, 0] == date_str][0, 1].astype(float)
        return "Positive" if price_chg > 0.0 else "Negative"

    def get_tweets(self, ticker, date_str):
        tweets = []
        tweet_path = os.path.join(self.tweet_dir, ticker, date_str)
        if os.path.exists(tweet_path):
            with open(tweet_path) as f:
                for line in f:
                    tweets.append(json.loads(line)["text"])
        return tweets

    # ------------------------------------------------------------------
    # Main load
    # ------------------------------------------------------------------

    def load(self, flag):
        data = pd.DataFrame()
        ticker_cache_dirty = False

        for file in os.listdir(self.price_dir):
            price_path = os.path.join(self.price_dir, file)
            ordered_price_data = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
            ticker = file[:-4]
            ohlcv_df = self._get_ohlcv(ticker)

            tes_idx = round(len(ordered_price_data) * 0.8)
            end_idx = len(ordered_price_data)
            data_range = range(tes_idx) if flag == "train" else range(tes_idx, end_idx)

            ticker_cache_dirty = False

            for idx in data_range:
                end_date_str = ordered_price_data[idx, 0]
                end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
                start_date = end_date - timedelta(days=self.seq_len)
                target = self.get_sentiment(end_date_str, price_path)

                # --- tweet summaries (skipped in indicators_only mode) ---
                summary_all = ""
                if not self.indicators_only:
                    for seq_date in self.daterange(start_date, end_date):
                        seq_date_str = seq_date.strftime("%Y-%m-%d")
                        cache_key = (ticker, seq_date_str)

                        if cache_key in self._summary_cache:
                            summary = self._summary_cache[cache_key]
                        else:
                            tweet_data = self.get_tweets(ticker, seq_date_str)
                            summary = self.summarizer.get_summary(ticker, tweet_data)
                            self._summary_cache[cache_key] = summary
                            ticker_cache_dirty = True

                        if summary and self.summarizer.is_informative(summary):
                            summary_all += seq_date_str + "\n" + summary + "\n\n"

                # --- technical indicators ---
                # Use t-1 to avoid look-ahead bias: day t's close is part of the target label
                tech_context = ""
                if (self.use_indicators or self.indicators_only) and ohlcv_df is not None:
                    prev_date_str = (end_date - timedelta(days=1)).strftime("%Y-%m-%d")
                    indicators = compute_indicators(ohlcv_df, prev_date_str)
                    tech_context = format_technical_context(prev_date_str, indicators)

                combined = (summary_all.rstrip() + ("\n\n" if summary_all and tech_context else "") + tech_context).strip()

                if combined:
                    data = pd.concat([data, pd.DataFrame([{
                        "ticker": ticker,
                        "date": end_date_str,
                        "summary": combined,
                        "target": target
                    }])], ignore_index=True)

            # Save cache after each ticker so a crash doesn't lose all progress
            if ticker_cache_dirty:
                self._save_cache()
                print(f"[cache] saved after {ticker}")

        return data
