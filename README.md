# Vibe Coding Trading Bot Skeleton

This project provides a modular skeleton for building an algorithmic trading bot.
It demonstrates how to organize strategies, exchange integrations, AI components,
and utilities for maintainability and extensibility.

## Folder Structure

```
strategies/   # Trading strategies
indicators/   # Technical indicators
exchange/     # Exchange interfaces (Binance, Bybit, ...)
db/           # Database helpers
ai/           # AI utilities and prompts
utils/        # Configuration and logging
Dashboards/   # Optional Streamlit dashboard
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py --strategy rsi
```

## Running Tests

```
pytest
```

## GitHub Actions

A workflow is provided under `.github/workflows/ci.yml` to run tests on
push and pull requests.
