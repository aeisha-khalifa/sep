from datetime import datetime, timedelta
from .ta_calculator import TACalculator
from .ta_interpreter import TAInterpreter


class TechnicalModule:

    def __init__(self):
        self.calculator = TACalculator()
        self.interpreter = TAInterpreter()

    def get_ta_summary(self, ticker: str, target_date: str, lookback_days: int = 60) -> str:
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=lookback_days)

        try:
            ohlcv = self.calculator.get_ohlcv(
                ticker,
                start_dt.strftime("%Y-%m-%d"),
                (target_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
            if ohlcv.empty:
                return ""
            indicators = self.calculator.calculate_indicators(ohlcv, target_date)
            return self.interpreter.interpret_indicators(indicators)
        except Exception as e:
            print(f"[TechnicalModule] Warning: failed for {ticker} on {target_date}: {e}")
            return ""
