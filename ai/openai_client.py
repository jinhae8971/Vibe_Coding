from __future__ import annotations
import logging
import yaml
from pathlib import Path
from typing import Any, Dict
import openai

class OpenAIClient:
    """Wrapper for OpenAI API interactions."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo") -> None:
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_prompt(self, path: str | Path) -> str:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("description", "")

    def get_signal(self, prompt_path: str | Path, market_data: Dict[str, Any]) -> str:
        prompt = self.load_prompt(prompt_path)
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": str(market_data)},
        ]
        try:
            response = openai.ChatCompletion.create(model=self.model, messages=messages)
            signal = response.choices[0].message.content.strip().upper()
            self.logger.info("AI signal: %s", signal)
            return signal
        except Exception as exc:
            self.logger.error("OpenAI call failed: %s", exc)
            return "HOLD"
