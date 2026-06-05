from __future__ import annotations
import pandas as pd
import ta
import os


def _compute_tema(close: pd.Series, period: int = 9) -> pd.Series:
    ema1 = close.ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    return 3 * ema1 - 3 * ema2 + ema3


def compute_indicators(ohlcv_df: pd.DataFrame, end_date) -> dict | None:
    # numpy.str_ from genfromtxt must be cast to plain str before pd.Timestamp
    df = ohlcv_df[ohlcv_df.index <= pd.Timestamp(str(end_date))].copy()
    if len(df) < 25:  # needs 20 for BB + buffer for TEMA convergence
        return None

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # HLC3: average of High, Low, Close
    hlc3 = (high + low + close) / 3

    # PVT: Price Volume Trend
    pvt = ta.volume.price_volume_trend(close, vol)

    # TEMA (period=9)
    tema = _compute_tema(close, period=9)

    # Bollinger Bands (period=20, std=2)
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid   = bb.bollinger_mavg()

    last = {
        "close":    float(close.iloc[-1]),
        "hlc3":     float(hlc3.iloc[-1]),
        "hlc3_5d":  float(hlc3.iloc[-5:].mean()),
        "pvt":      float(pvt.iloc[-1]),
        "pvt_prev": float(pvt.iloc[-2]) if len(pvt) >= 2 else float(pvt.iloc[-1]),
        "tema":     float(tema.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_mid":   float(bb_mid.iloc[-1]),
    }

    # guard against NaNs from short series
    if any(pd.isna(v) for v in last.values()):
        return None

    return last


def compute_indicators_history(ohlcv_df: pd.DataFrame, end_date, n: int = 3) -> dict | None:
    """Return the last n days of each indicator as lists, ending on end_date."""
    df = ohlcv_df[ohlcv_df.index <= pd.Timestamp(str(end_date))].copy()
    if len(df) < 25 + n:
        return None

    close    = df["Close"]
    high     = df["High"]
    low      = df["Low"]
    vol      = df["Volume"]

    hlc3     = (high + low + close) / 3
    pvt      = ta.volume.price_volume_trend(close, vol)
    tema     = _compute_tema(close, period=9)
    bb       = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()
    bb_mid   = bb.bollinger_mavg()

    history = {
        "close":    [round(float(close.iloc[-i]),    2) for i in range(n, 0, -1)],
        "hlc3":     [round(float(hlc3.iloc[-i]),     2) for i in range(n, 0, -1)],
        "pvt":      [round(float(pvt.iloc[-i]),      2) for i in range(n, 0, -1)],
        "tema":     [round(float(tema.iloc[-i]),     2) for i in range(n, 0, -1)],
        "bb_upper": [round(float(bb_upper.iloc[-i]), 2) for i in range(n, 0, -1)],
        "bb_mid":   [round(float(bb_mid.iloc[-i]),   2) for i in range(n, 0, -1)],
        "bb_lower": [round(float(bb_lower.iloc[-i]), 2) for i in range(n, 0, -1)],
    }

    if any(pd.isna(v) for vals in history.values() for v in vals):
        return None

    return history


def format_technical_context_detailed(end_date_str: str, history: dict | None) -> str:
    """Structured indicator block with description and historical values."""
    if history is None:
        return ""

    def fmt(vals):
        return ", ".join(str(v) for v in vals)

    n = len(history["close"])
    lines = [f"Technical Indicators (as of {end_date_str}, last {n} trading days prior to prediction date):\n"]

    lines.append("HLC3 - High Low Close Average")
    lines.append("Formula: (High + Low + Close) / 3")
    lines.append(
        "Explanation: This formula expression calculates the average of the high, low, and close "
        "prices for a given trading day. It represents the typical price and provides a balanced "
        "view of the intraday price range. When HLC3 is above the closing price, it indicates "
        "that the intraday price range was skewed upward, suggesting buying pressure during the "
        "session."
    )
    lines.append(f"Historical Values: {fmt(history['hlc3'])}\n")

    lines.append("PVT - Price Volume Trend")
    lines.append("Formula: PVT_t = PVT_{t-1} + Volume_t × (Close_t − Close_{t-1}) / Close_{t-1}")
    lines.append(
        "Explanation: The PVT is a cumulative volume-based indicator that weights each day's "
        "volume contribution by the percentage change in price, making it more sensitive to "
        "the magnitude of price moves than simpler volume indicators. Rising PVT values indicate "
        "that volume is concentrated on up-days relative to the size of price gains, reflecting "
        "accumulation, while falling PVT values indicate distribution and selling pressure."
    )
    lines.append(f"Historical Values: {fmt(history['pvt'])}\n")

    lines.append("TEMA-9 - Triple Exponential Moving Average (9-day)")
    lines.append("Formula: 3·EMA(close, 9) − 3·EMA(EMA(close, 9), 9) + EMA(EMA(EMA(close, 9), 9), 9)")
    lines.append(
        "Explanation: The TEMA reduces the inherent lag of traditional exponential moving averages "
        "by applying a triple-smoothing technique over a 9-day period, making it more responsive "
        "to recent price changes than standard EMAs. When the closing price is above the TEMA "
        "value, it signals upward price momentum; when below, it signals downward momentum."
    )
    lines.append(f"Historical Values: {fmt(history['tema'])}\n")

    lines.append("Bollinger Bands (20-day, 2σ)")
    lines.append("Formula: Upper = SMA(20) + 2·σ(20),  Middle = SMA(20),  Lower = SMA(20) − 2·σ(20)")
    lines.append(
        "Explanation: The upper band is calculated by adding two standard deviations to the "
        "20-day simple moving average and can signal overbought conditions when the price touches "
        "or breaches it. The middle band is the 20-day SMA and serves as the intermediate-term "
        "trend baseline. The lower band subtracts two standard deviations from the SMA and can "
        "signal oversold conditions when the price touches or breaches it. All three bands expand "
        "and contract dynamically with market volatility."
    )
    lines.append(f"Upper Band Historical Values:  {fmt(history['bb_upper'])}")
    lines.append(f"Middle Band Historical Values: {fmt(history['bb_mid'])}")
    lines.append(f"Lower Band Historical Values:  {fmt(history['bb_lower'])}")

    return "\n".join(lines)


def format_technical_context(end_date_str: str, indicators: dict | None) -> str:
    if indicators is None:
        return ""

    lines = [f"Technical Context (as of {end_date_str}):"]

    # HLC3 — buying/selling pressure
    close = indicators["close"]
    hlc3  = indicators["hlc3"]
    diff_pct = ((hlc3 - close) / close) * 100
    if diff_pct > 0.5:
        lines.append(
            f"- HLC3 (typical price: avg of high, low, close): "
            f"average of high/low/close is {diff_pct:.2f}% above close — intraday buying pressure"
        )
    elif diff_pct < -0.5:
        lines.append(
            f"- HLC3 (typical price: avg of high, low, close): "
            f"average of high/low/close is {abs(diff_pct):.2f}% below close — intraday selling pressure"
        )
    else:
        lines.append(
            "- HLC3 (typical price: avg of high, low, close): "
            "close near intraday average — neutral pressure"
        )

    # PVT — volume trend
    pvt_change = indicators["pvt"] - indicators["pvt_prev"]
    if pvt_change > 0:
        lines.append(
            "- PVT (volume weighted by % price change): "
            "rising — buyers driving volume, accumulation signal"
        )
    else:
        lines.append(
            "- PVT (volume weighted by % price change): "
            "falling — sellers driving volume, distribution signal"
        )

    # TEMA — trend
    tema = indicators["tema"]
    if close > tema * 1.005:
        lines.append(
            "- TEMA-9 (triple-smoothed 9-day moving average): "
            "price above trend line — bullish momentum"
        )
    elif close < tema * 0.995:
        lines.append(
            "- TEMA-9 (triple-smoothed 9-day moving average): "
            "price below trend line — bearish momentum"
        )
    else:
        lines.append(
            "- TEMA-9 (triple-smoothed 9-day moving average): "
            "price near trend line — no clear trend signal"
        )

    # Bollinger Bands — volatility
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_width = bb_upper - bb_lower
    if bb_width > 0:
        pct_in_band = ((close - bb_lower) / bb_width) * 100
        if close >= bb_upper:
            lines.append(
                "- BB (20-day bands ±2 standard deviations): "
                "price at or above upper band — statistically extreme positive move, potential pullback"
            )
        elif close <= bb_lower:
            lines.append(
                "- BB (20-day bands ±2 standard deviations): "
                "price at or below lower band — statistically extreme negative move, potential reversal"
            )
        else:
            lines.append(
                f"- BB (20-day bands ±2 standard deviations): "
                f"price at {pct_in_band:.0f}% of band range — "
                f"{'upper half, mild positive bias' if pct_in_band > 50 else 'lower half, mild negative bias'}"
            )

    return "\n".join(lines)


def load_ohlcv(ohlcv_dir: str, ticker: str) -> pd.DataFrame | None:
    path = os.path.join(ohlcv_dir, f"{ticker}.csv")
    if not os.path.exists(path):
        return None
    # yfinance >=1.x writes two extra metadata rows (Ticker, Date) after the header
    df = pd.read_csv(path, index_col=0, skiprows=[1, 2])
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
    df = df[df.index.notna()]
    df.index.name = "Date"
    for col in ["Close", "High", "Low", "Open", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
