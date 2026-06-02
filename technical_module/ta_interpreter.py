class TAInterpreter:

    def interpret_indicators(self, indicators: dict) -> str:
        if not indicators:
            return ""

        lines = []

        rsi = indicators.get('RSI')
        if rsi is not None and not _is_nan(rsi):
            if rsi > 70:
                lines.append(f"RSI is at {rsi:.1f}, indicating overbought conditions.")
            elif rsi < 30:
                lines.append(f"RSI is at {rsi:.1f}, indicating oversold conditions.")
            else:
                lines.append(f"RSI is at {rsi:.1f}, in neutral territory.")

        tema = indicators.get('TEMA')
        close = indicators.get('Close')
        if tema is not None and close is not None and not _is_nan(tema) and not _is_nan(close):
            if close > tema:
                lines.append(f"Price (${close:.2f}) is above the TEMA (${tema:.2f}), suggesting bullish momentum.")
            else:
                lines.append(f"Price (${close:.2f}) is below the TEMA (${tema:.2f}), suggesting bearish momentum.")

        bb_upper = indicators.get('BB_upper')
        bb_lower = indicators.get('BB_lower')
        if close is not None and bb_upper is not None and bb_lower is not None:
            if not _is_nan(bb_upper) and not _is_nan(bb_lower) and not _is_nan(close):
                if close > bb_upper:
                    lines.append(f"Price has broken above the upper Bollinger Band (${bb_upper:.2f}), suggesting overbought conditions.")
                elif close < bb_lower:
                    lines.append(f"Price has broken below the lower Bollinger Band (${bb_lower:.2f}), suggesting oversold conditions.")
                else:
                    lines.append(f"Price is trading within Bollinger Bands (${bb_lower:.2f} - ${bb_upper:.2f}), indicating normal volatility.")

        macd = indicators.get('MACD')
        macd_signal = indicators.get('MACD_signal')
        if macd is not None and macd_signal is not None and not _is_nan(macd) and not _is_nan(macd_signal):
            if macd > macd_signal:
                lines.append(f"MACD ({macd:.3f}) is above its signal line ({macd_signal:.3f}), suggesting bullish momentum.")
            else:
                lines.append(f"MACD ({macd:.3f}) is below its signal line ({macd_signal:.3f}), suggesting bearish momentum.")

        obv = indicators.get('OBV')
        if obv is not None and not _is_nan(obv):
            lines.append(f"On-Balance Volume is {obv:,.0f}, reflecting cumulative buying and selling pressure.")

        atr = indicators.get('ATR')
        if atr is not None and not _is_nan(atr):
            if atr > 3:
                level = "high"
            elif atr < 1:
                level = "low"
            else:
                level = "moderate"
            lines.append(f"Average True Range is {atr:.2f}%, indicating {level} volatility.")

        hlc3 = indicators.get('HLC3')
        if hlc3 is not None and not _is_nan(hlc3):
            lines.append(f"The typical price (average of High, Low, Close) is ${hlc3:.2f}.")

        return "\n".join(lines)


def _is_nan(value) -> bool:
    try:
        import math
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True
