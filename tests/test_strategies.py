import pandas as pd
from strategies import RSIStrategy
from exchange import BinanceExchange


def test_rsi_strategy_signal():
    data = pd.DataFrame({"close": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]})
    ex = BinanceExchange("key", "secret")
    strat = RSIStrategy(ex, {"symbol": "BTCUSDT"})
    signal = strat.generate_signal(data)
    assert signal in {"LONG", "SHORT", "HOLD"}
