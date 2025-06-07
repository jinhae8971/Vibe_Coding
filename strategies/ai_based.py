from __future__ import annotations
from typing import Any, Dict

import pandas as pd

from ai.openai_client import OpenAIClient
from strategies.base import BaseStrategy


class AIBasedStrategy(BaseStrategy):
    """Strategy that uses OpenAI to generate trading signals."""

    def __init__(self, exchange, config: Dict[str, Any], ai_client: OpenAIClient):
        super().__init__(exchange, config)
        self.ai_client = ai_client
        self.prompt_path = config.get("prompt_path", "ai/prompts/sample_prompt.yaml")

    def generate_signal(self, data: pd.DataFrame) -> str:
        market_snapshot = data.tail(1).to_dict(orient="records")[0]
        return self.ai_client.get_signal(self.prompt_path, market_snapshot)
