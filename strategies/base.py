from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

from exchange.base import BaseExchange


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""

    def __init__(self, exchange: BaseExchange, config: Dict[str, Any]):
        self.exchange = exchange
        self.config = config

    @abstractmethod
    def generate_signal(self, data: Any) -> str:
        """Return trading signal."""

    def on_tick(self, data: Any) -> None:
        signal = self.generate_signal(data)
        if signal == "LONG":
            self.exchange.place_order(symbol=self.config["symbol"], side="BUY")
        elif signal == "SHORT":
            self.exchange.place_order(symbol=self.config["symbol"], side="SELL")
