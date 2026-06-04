import numpy as np
from datetime import datetime, timedelta
import os

price_dir = 'data/price/preprocessed/'
tweet_dir = 'C:/Users/aeish/sn2/tweet/raw/'
seq_len = 5

total_summarise_calls = 0
total_test_dates = 0

for file in sorted(os.listdir(price_dir)):
    ticker = file[:-4]
    price_path = os.path.join(price_dir, file)
    ordered = np.flip(np.genfromtxt(price_path, dtype=str, skip_header=False), 0)
    tes_idx = round(len(ordered) * 0.8)

    seq_dates_needed = set()
    for idx in range(tes_idx, len(ordered)):
        end_date = datetime.strptime(str(ordered[idx, 0]), "%Y-%m-%d")
        start_date = end_date - timedelta(days=seq_len)
        for n in range(seq_len):
            d = (start_date + timedelta(n)).strftime("%Y-%m-%d")
            seq_dates_needed.add(d)
        total_test_dates += 1

    tweet_ticker_dir = os.path.join(tweet_dir, ticker)
    existing = set(os.listdir(tweet_ticker_dir)) if os.path.exists(tweet_ticker_dir) else set()
    with_tweets = seq_dates_needed & existing

    first_test = str(ordered[tes_idx, 0])
    last_test  = str(ordered[-1, 0])
    print(f"{ticker}: {len(ordered)-tes_idx} test dates ({first_test} to {last_test}) | "
          f"{len(with_tweets)} summarisation calls")
    total_summarise_calls += len(with_tweets)

print()
print(f"Total test prediction dates   : {total_test_dates}")
print(f"Total summarisation API calls : {total_summarise_calls} (done once, cached)")
print(f"Total explain-stage samples   : ~{total_test_dates} x 3 conditions = {total_test_dates*3}")
