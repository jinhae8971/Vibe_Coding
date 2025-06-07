from __future__ import annotations
from typing import Any, Dict

import pandas as pd

from indicators.rsi import rsi
from strategies.base import BaseStrategy


class RSIStrategy(BaseStrategy):
    """Simple RSI based strategy."""

    def __init__(self, exchange, config: Dict[str, Any]):
        super().__init__(exchange, config)
        self.period = config.get("period", 14)
        self.overbought = config.get("overbought", 70)
        self.oversold = config.get("oversold", 30)

    def generate_signal(self, data: pd.DataFrame) -> str:
        rsi_series = rsi(data["close"], self.period)
        current_rsi = rsi_series.iloc[-1]
        if current_rsi > self.overbought:
            return "SHORT"
        if current_rsi < self.oversold:
            return "LONG"
        return "HOLD"
