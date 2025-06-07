from __future__ import annotations
import argparse
import pandas as pd

from utils.config import load_config
from utils.logger import setup_logger
from exchange import BinanceExchange
from ai.openai_client import OpenAIClient
from strategies import AIBasedStrategy, RSIStrategy
from db.sqlite import SQLiteDB


def run(strategy_name: str, config_path: str | None = None) -> None:
    cfg = load_config(config_path)
    logger = setup_logger("Trader")
    db = SQLiteDB()
    exchange = BinanceExchange(cfg.api_key, cfg.api_secret, cfg.testnet)
    ai_client = OpenAIClient(cfg.api_key)

    if strategy_name == "ai":
        strategy = AIBasedStrategy(exchange, {"symbol": "BTCUSDT"}, ai_client)
    else:
        strategy = RSIStrategy(exchange, {"symbol": "BTCUSDT"})

    # Placeholder market data
    data = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
    strategy.on_tick(data)
    db.snapshot_position("BTCUSDT", 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["ai", "rsi"], default="rsi")
    parser.add_argument("--config")
    args = parser.parse_args()
    run(args.strategy, args.config)
