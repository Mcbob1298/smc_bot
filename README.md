# SMC Bot — Smart Money Concept Trading Bot

Algorithmic trading bot implementing the SMC methodology (Kasper Trading) for **XAUUSD** and **BTCUSDT**.

## Objective

Scientifically backtest the Kasper Trading SMC method to determine if it has a real statistical edge, then deploy in paper trading and micro-real if validated.

## Architecture

```
smc_bot/
├── config/          # Settings and strategy parameters (Pydantic)
├── data/            # Data pipeline (ingestion, enrichment, storage)
├── detectors/       # SMC pattern detectors (swings, BOS, ChoCh, FVG, OB, liquidity)
├── strategy/        # Signal assembly (bias, zones, entry, trade management)
├── risk/            # Position sizing and exposure limits
├── backtest/        # Backtesting engines (vectorbt + backtesting.py)
├── live/            # Live execution (MT5, paper trading)
├── journal/         # Trade logging and analysis
├── tests/           # Unit tests with manual fixtures
└── notebooks/       # Exploratory analysis
```

## Setup

### Prerequisites

- Windows 10/11
- MetaTrader 5 installed
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
cd smc_bot/

# Install dependencies
uv sync

# Install dev dependencies
uv sync --extra dev

# Copy and fill environment variables
cp .env.example .env
# Edit .env with your MT5 credentials
```

### Running checks

```bash
uv run pytest              # Run tests
uv run ruff check .        # Linting
uv run mypy config/        # Type checking
```

## Methodology

Based on Smart Money Concepts as taught by Kasper Trading:

- **Multi-timeframe analysis**: H4 (bias) → M15 (zones) → M1/M5 (entry)
- **Structure**: BOS for continuation, ChoCh for reversal
- **Zones**: Order Blocks validated by FVG + structure break
- **Entry**: LTF ChoCh confirmation within valid zone during killzone
- **Risk**: 1:2 RR target, TP1 at 1:1 with partial close + move to BE

## Key Principles

1. **Strict causality** — no lookahead bias in any detector
2. **No magic numbers** — all parameters in `config/strategy.py`
3. **Test-driven** — every detector has unit tests with known fixtures
4. **Structured logging** — every trade logged as JSON via loguru
