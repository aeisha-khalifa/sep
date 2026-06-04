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

    # OBV
    obv = ta.volume.on_balance_volume(close, vol)

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
        "obv":      float(obv.iloc[-1]),
        "obv_prev": float(obv.iloc[-2]) if len(obv) >= 2 else float(obv.iloc[-1]),
        "tema":     float(tema.iloc[-1]),
        "bb_upper": float(bb_upper.iloc[-1]),
        "bb_lower": float(bb_lower.iloc[-1]),
        "bb_mid":   float(bb_mid.iloc[-1]),
    }

    # guard against NaNs from short series
    if any(pd.isna(v) for v in last.values()):
        return None

    return last


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
            f"- Price Balance (HLC3): average of high/low/close is {diff_pct:.2f}% above close "
            f"— intraday buying pressure"
        )
    elif diff_pct < -0.5:
        lines.append(
            f"- Price Balance (HLC3): average of high/low/close is {abs(diff_pct):.2f}% below close "
            f"— intraday selling pressure"
        )
    else:
        lines.append("- Price Balance (HLC3): close near intraday average — neutral pressure")

    # OBV — volume trend
    obv_change = indicators["obv"] - indicators["obv_prev"]
    if obv_change > 0:
        lines.append("- Volume Flow (OBV): rising — buyers driving volume, accumulation signal")
    else:
        lines.append("- Volume Flow (OBV): falling — sellers driving volume, distribution signal")

    # TEMA — trend
    tema = indicators["tema"]
    if close > tema * 1.005:
        lines.append(f"- Trend (TEMA-9): price above moving average — bullish momentum")
    elif close < tema * 0.995:
        lines.append(f"- Trend (TEMA-9): price below moving average — bearish momentum")
    else:
        lines.append(f"- Trend (TEMA-9): price near moving average — no clear trend signal")

    # Bollinger Bands — volatility
    bb_upper = indicators["bb_upper"]
    bb_lower = indicators["bb_lower"]
    bb_mid   = indicators["bb_mid"]
    bb_width = bb_upper - bb_lower
    if bb_width > 0:
        pct_in_band = ((close - bb_lower) / bb_width) * 100
        if close >= bb_upper:
            lines.append(
                "- Volatility (BB): price at or above upper Bollinger Band "
                "— statistically extreme positive move, potential pullback"
            )
        elif close <= bb_lower:
            lines.append(
                "- Volatility (BB): price at or below lower Bollinger Band "
                "— statistically extreme negative move, potential reversal"
            )
        else:
            lines.append(
                f"- Volatility (BB): price within bands at {pct_in_band:.0f}% of range "
                f"— {'upper half, mild positive bias' if pct_in_band > 50 else 'lower half, mild negative bias'}"
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
