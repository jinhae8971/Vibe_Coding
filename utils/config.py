from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    api_key: str
    api_secret: str
    testnet: bool = True


def load_config(path: str | Path | None = None) -> Config:
    if path and Path(path).exists():
        with open(path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
    else:
        data = {
            "api_key": os.getenv("API_KEY", ""),
            "api_secret": os.getenv("API_SECRET", ""),
            "testnet": os.getenv("TESTNET", "True") == "True",
        }
    return Config(**data)
