# CLAUDE.md — Operating rules for the SMC bot

Guidance for any agent (or human) working in this repo. Read it before
touching execution, sizing, or strategy code.

## What this bot is

An algorithmic scalping bot for **XAUUSD on M5**, executing automatically via
**MetaTrader 5** (broker: Vantage, demo first). Methodology: Smart Money
Concepts (structure, order blocks, BOS/CHoCH, FVG, liquidity), multi-timeframe
confluence M5 ↔ M15/H1, traded during the London/NY session overlap.

## Why the rules below exist (read this once)

A previous version of this bot turned 500€ into ~900€ in a week, then gave it
all back in a single large move. The entry signal was not the problem — the
**total absence of risk management** was. Therefore:

> **Risk management is priority #1 and holds veto power over every execution.**
> The risk layer is coded first, tested first, and no order is ever sent
> without passing through it.

## Non-negotiable rules

1. **Risk layer before signal logic.** `risk/` is the foundation everything
   else consumes. It was built and tested before `strategy/`.
2. **No order without a stop-loss.** Enforced twice: a `Signal` cannot be
   constructed without a valid SL (`strategy/signal.py`), and `RiskManager`
   re-asserts it before sizing.
3. **Position size is derived, never guessed.** Lot = f(SL distance, fixed risk
   %). Risk per trade is **0.5–1%** of equity (`RiskConfig.risk_per_trade_pct`).
   Sizing always rounds *down*; if the minimum lot would over-risk the budget,
   the trade is **refused**, not forced.
4. **SL sits at the idea's invalidation**, not an arbitrary distance — at the
   structural level that proves the setup wrong (OB edge + buffer, swing, etc.).
   The `Signal.reasons` map must say *why* each level is where it is.
5. **Daily-loss kill switch.** Once the account is down the daily limit
   (default **3%**, `DailyLossGuard`), the bot stops trading for the day. The
   switch **latches** — it does not re-arm on intraday recovery.
6. **TP ladder.** TP1 at **1:1** (partial close, then SL → break-even), TP2 at
   the next structural level, TP3 further. See `RiskConfig` tp* fields.
7. **News blackout.** No new position **30 min before/after** a major macro
   news event. The data/calendar layer owns the flag; risk honours it.
8. **Demo only** until the edge is measured (see promotion gate below).

## Demo → Real promotion gate — DO NOT BYPASS

Real money does **not** go back in — even after a good week — until **all** of
these are met on the **demo** account:

- [ ] **≥ 50 trades** executed on demo (statistically meaningful sample).
- [ ] **Positive measured expectancy** over that sample
      (avg R per trade > 0, after spread + slippage costs).
- [ ] **Max drawdown below the defined threshold:** peak-to-trough equity
      drawdown over the sample **< 10%**, and no single day breaching the **3%**
      daily-loss limit.

A good week is not evidence of an edge. 50+ trades with positive expectancy and
contained drawdown is. This gate is intentional friction — do not "optimize" or
argue around it.

## Architecture (modules)

```
config/      Pydantic settings + all strategy params (no magic numbers in logic)
data/        Data pipeline: MT5 ingestion, ATR, econ calendar — all anti-lookahead
detectors/   SMC pattern detectors (swings, BOS/ChoCh, FVG, OB, liquidity)
strategy/    Signal assembly (bias, zones, entry, trade mgmt) → emits Signal
risk/        Protective core: sizing, daily kill switch, exposure, RiskManager veto
live/        Execution: MT5 executor, paper executor, monitor
journal/     Trade logging (JSON/SQLite) + annotated mplfinance charts
backtest/    Backtesting engines + metrics + walk-forward
```

### Risk layer surface (`risk/`)

- `Signal` (`strategy/signal.py`) — immutable trade idea: entry, SL, TP1/2/3,
  `reasons`. Validates SL side + monotonic TPs at construction.
- `compute_lot` / `SymbolSpec` (`position_sizer.py`) — broker-native sizing.
- `DailyLossGuard` (`daily_guard.py`) — latching daily-loss kill switch.
- `ExposureTracker` (`exposure.py`) — concurrent + per-killzone (anti-revenge) caps.
- `RiskManager` / `RiskDecision` (`risk_manager.py`) — the veto core. The
  executor may only act on `RiskDecision.approved`.

## Engineering conventions

- **Strict causality** — no lookahead in any detector or feature.
- **No magic numbers** — every tunable lives in `config/strategy.py`.
- **Test-driven** — every risk component and detector has unit tests.
- **Secrets in `.env` only** — never hard-code MT5 credentials.

## Dev commands

```bash
uv sync --extra dev          # install (Windows: includes MetaTrader5)
uv run pytest                # tests
uv run ruff check .          # lint
uv run mypy config/ risk/    # type check
```

> **Note (non-Windows / CI):** the `MetaTrader5` package only ships Windows
> wheels, so `uv sync` fails to resolve on Linux/macOS. The `risk/` and
> `strategy/signal.py` code depends only on pydantic + stdlib and is testable
> anywhere with a minimal venv (`pip install pytest pydantic pydantic-settings`).
> Keep the risk layer free of any hard MT5 import so it stays unit-testable off-Windows.
