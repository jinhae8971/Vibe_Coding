from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExchange(ABC):
    """Exchange interface."""

    @abstractmethod
    def get_balance(self) -> float:
        pass

    @abstractmethod
    def fetch_position(self, symbol: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    def place_order(self, symbol: str, side: str, qty: float | None = None) -> Dict[str, Any]:
        pass
