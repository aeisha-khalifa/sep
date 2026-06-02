import os
import pandas as pd
import ta

OHLCV_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_price', 'ohlcv')


class TACalculator:

    def get_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        path = os.path.join(OHLCV_DIR, f"{ticker}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No OHLCV data found for {ticker}. Run download_ohlcv.py first."
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        return df[(df.index >= start_dt) & (df.index <= end_dt)]

    def calculate_indicators(self, df: pd.DataFrame, target_date: str) -> dict:
        if df.empty or len(df) < 26:
            return {}

        df = df.copy()

        df['RSI']          = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        df['TEMA']         = self._tema(df['Close'], window=10)
        bb                 = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
        df['BB_upper']     = bb.bollinger_hband()
        df['BB_lower']     = bb.bollinger_lband()
        df['OBV']          = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        atr_abs            = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['ATR']          = (atr_abs / df['Close']) * 100
        macd               = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
        df['MACD']         = macd.macd()
        df['MACD_signal']  = macd.macd_signal()
        df['HLC3']         = (df['High'] + df['Low'] + df['Close']) / 3

        df.index = pd.to_datetime(df.index)
        target_dt = pd.to_datetime(target_date)
        available = df[df.index <= target_dt]
        if available.empty:
            return {}
        row = available.iloc[-1]

        return {
            'RSI':         row.get('RSI'),
            'TEMA':        row.get('TEMA'),
            'BB_upper':    row.get('BB_upper'),
            'BB_lower':    row.get('BB_lower'),
            'OBV':         row.get('OBV'),
            'ATR':         row.get('ATR'),
            'MACD':        row.get('MACD'),
            'MACD_signal': row.get('MACD_signal'),
            'HLC3':        row.get('HLC3'),
            'Close':       row.get('Close'),
        }

    def _tema(self, series: pd.Series, window: int) -> pd.Series:
        ema1 = series.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
