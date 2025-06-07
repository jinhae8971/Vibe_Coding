import sqlite3
from pathlib import Path
from typing import Any, Dict


class SQLiteDB:
    """Simple SQLite wrapper for storing trades and positions."""

    def __init__(self, path: str | Path = "trades.sqlite3") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.create_tables()

    def create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                position REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        self.conn.commit()

    def log_trade(self, trade: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO trades (symbol, side, qty, price) VALUES (?, ?, ?, ?)",
            (trade["symbol"], trade["side"], trade["qty"], trade["price"]),
        )
        self.conn.commit()

    def snapshot_position(self, symbol: str, position: float) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO positions (symbol, position) VALUES (?, ?)",
            (symbol, position),
        )
        self.conn.commit()
