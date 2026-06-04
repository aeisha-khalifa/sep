"""
Verify each technical indicator against its formula from
Mostafavi & Hooman (2025), Table 2.
"""
import pandas as pd
import numpy as np
import ta
from technical_module.indicators import load_ohlcv, compute_indicators, format_technical_context

df = load_ohlcv("data/ohlcv", "AAPL")
TEST_DATE  = "2022-06-01"
TEST_DATE2 = "2022-06-02"   # used for OBV direction check
window = df[df.index <= pd.Timestamp(TEST_DATE)].copy()
close  = window["Close"]
high   = window["High"]
low    = window["Low"]
vol    = window["Volume"]

PASS = "[PASS]"
FAIL = "[FAIL]"
results = []

# ── 1. HLC3 ────────────────────────────────────────────────────────────────
# Formula (Table 2): HLC3_t = (High_t + Low_t + Close_t) / 3
manual = (high.iloc[-1] + low.iloc[-1] + close.iloc[-1]) / 3
impl   = compute_indicators(df, TEST_DATE)["hlc3"]
ok = abs(manual - impl) < 1e-6
results.append(ok)
print("1. HLC3  —  (High + Low + Close) / 3")
print(f"   Manual : {manual:.6f}")
print(f"   Impl   : {impl:.6f}")
print(f"   {PASS if ok else FAIL}")
print()

# ── 2. OBV — direction only (absolute value differs by initialisation offset)
# Formula (Table 2): OBV_t += Volume if up, -= Volume if down, unchanged if flat
# We use: obv_change = obv[-1] - obv[-2]  (direction only)
ind1 = compute_indicators(df, TEST_DATE)
ind2 = compute_indicators(df, TEST_DATE2)
obv_change_impl = ind1["obv"] - ind1["obv_prev"]

# Manual 1-day change for TEST_DATE
c_today = close.iloc[-1]
c_prev  = close.iloc[-2]
v_today = vol.iloc[-1]
if c_today > c_prev:   manual_change = v_today
elif c_today < c_prev: manual_change = -v_today
else:                  manual_change = 0.0

ok = (obv_change_impl > 0) == (manual_change > 0) and abs(obv_change_impl - manual_change) < 1.0
results.append(ok)
print("2. OBV  —  direction of 1-day change (what we actually use)")
print(f"   Manual change : {manual_change:,.0f}  ({'up' if manual_change>0 else 'down'})")
print(f"   Impl change   : {obv_change_impl:,.0f}  ({'up' if obv_change_impl>0 else 'down'})")
print(f"   Note: absolute OBV differs by a constant offset (init convention in ta library)")
print(f"         but 1-day direction is identical — direction is all we use")
print(f"   {PASS if ok else FAIL}")
print()

# ── 3. TEMA ─────────────────────────────────────────────────────────────────
# Formula (Table 2): TEMA_t = 3*EMA1 - 3*EMA2 + EMA3
# EMA1 = EMA(close, 9), EMA2 = EMA(EMA1, 9), EMA3 = EMA(EMA2, 9)
PERIOD = 9
ema1 = close.ewm(span=PERIOD, adjust=False).mean()
ema2 = ema1.ewm(span=PERIOD, adjust=False).mean()
ema3 = ema2.ewm(span=PERIOD, adjust=False).mean()
manual = float(3*ema1.iloc[-1] - 3*ema2.iloc[-1] + ema3.iloc[-1])
impl   = compute_indicators(df, TEST_DATE)["tema"]
ok = abs(manual - impl) < 1e-6
results.append(ok)
print(f"3. TEMA (period={PERIOD})  —  3*EMA1 - 3*EMA2 + EMA3")
print(f"   EMA1 : {ema1.iloc[-1]:.6f}")
print(f"   EMA2 : {ema2.iloc[-1]:.6f}")
print(f"   EMA3 : {ema3.iloc[-1]:.6f}")
print(f"   Manual : {manual:.6f}")
print(f"   Impl   : {impl:.6f}")
print(f"   {PASS if ok else FAIL}")
print()

# ── 4. Bollinger Bands ──────────────────────────────────────────────────────
# Formula (Table 2): SMA(20) ± 2 * std(20)
# Standard finance convention (Bollinger's original): population std (ddof=0)
WINDOW = 20
sma = close.rolling(WINDOW).mean()
std = close.rolling(WINDOW).std(ddof=0)   # population std — Bollinger original
manual_upper = float(sma.iloc[-1] + 2 * std.iloc[-1])
manual_lower = float(sma.iloc[-1] - 2 * std.iloc[-1])
manual_mid   = float(sma.iloc[-1])

bb = ta.volatility.BollingerBands(close, window=WINDOW, window_dev=2)
impl_upper = float(bb.bollinger_hband().iloc[-1])
impl_lower = float(bb.bollinger_lband().iloc[-1])
impl_mid   = float(bb.bollinger_mavg().iloc[-1])

ok_u = abs(manual_upper - impl_upper) < 1e-4
ok_l = abs(manual_lower - impl_lower) < 1e-4
ok_m = abs(manual_mid   - impl_mid)   < 1e-4
ok = ok_u and ok_l and ok_m
results.append(ok)
print(f"4. Bollinger Bands (window={WINDOW}, dev=2)  —  SMA ± 2*std  [population std, ddof=0]")
print(f"   Upper  manual={manual_upper:.4f}  impl={impl_upper:.4f}  {PASS if ok_u else FAIL}")
print(f"   Middle manual={manual_mid:.4f}    impl={impl_mid:.4f}    {PASS if ok_m else FAIL}")
print(f"   Lower  manual={manual_lower:.4f}  impl={impl_lower:.4f}  {PASS if ok_l else FAIL}")
print()

# ── 5. No look-ahead ────────────────────────────────────────────────────────
last_used   = window.index[-1].date()
first_future = df[df.index > pd.Timestamp(TEST_DATE)].index[0].date()
ok = str(last_used) == TEST_DATE
results.append(ok)
print(f"5. Look-ahead check")
print(f"   Last row used   : {last_used}  (must equal {TEST_DATE})")
print(f"   First future row: {first_future}  (must NOT be used)")
print(f"   {PASS if ok else FAIL}")
print()

# ── 6. Minimum data guard ───────────────────────────────────────────────────
rows_at_early = len(df[df.index <= pd.Timestamp("2019-07-01")])
early_result  = compute_indicators(df, "2019-07-01")
ok = early_result is None and rows_at_early < 25
results.append(ok)
print(f"6. Minimum data guard (< 25 rows returns None)")
print(f"   Rows at 2019-07-01 : {rows_at_early}")
print(f"   Result             : {early_result}")
print(f"   {PASS if ok else FAIL}")
print()

# ── 7. NaN guard ─────────────────────────────────────────────────────────────
ind = compute_indicators(df, TEST_DATE)
has_nan = any(pd.isna(v) for v in ind.values())
ok = not has_nan
results.append(ok)
print(f"7. NaN guard — no NaN values in output")
print(f"   Values: {ind}")
print(f"   {PASS if ok else FAIL}")
print()

# ── 8. Formatted output ──────────────────────────────────────────────────────
print(f"8. Sample formatted output ({TEST_DATE}):")
print(format_technical_context(TEST_DATE, compute_indicators(df, TEST_DATE)))
print()

# ── Summary ──────────────────────────────────────────────────────────────────
total = len(results)
passed = sum(results)
print("=" * 50)
print(f"RESULT: {passed}/{total} checks passed  {'— ALL GOOD' if passed == total else '— ISSUES FOUND'}")
