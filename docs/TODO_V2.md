# TODO V2 — Future Improvements

Enhancements deferred to V2 to keep V1 scope manageable.

---

## MT5 Loader

- **Live health check for `is_connected()`**: Currently a passive flag.
  Should call `mt5.terminal_info()` with result caching (TTL ~5s) to detect
  terminal crashes after initial connection.

- **Automatic reconnection**: If MT5 disconnects mid-download, retry with
  exponential backoff (max 3 attempts) before raising.

## Breaker Blocks

- OB invalidated → stored with `is_broken=True` + `broken_at_bar`.
  V2 implements the full Breaker Block logic (polarity inversion, re-entry criteria).

## OB Scoring / Confluence

- `has_internal_fvg` flag on OBs → V2 scoring system that ranks OBs by
  confluence (FVG overlap, prior liquidity sweep quality, HTF alignment strength).

## Calendar

- Cross-check MT5 calendar with ForexFactory for completeness.
- Add crypto-specific events (ETF decisions, protocol upgrades, unlocks).

## Paper executor — realistic friction model (BEFORE going live)

The paper executor's pessimistic stop-first bar resolution is correct for
measuring an edge *without self-deception*, but it does **not** model trading
friction: it fills at the exact stop/target price with a single fixed
`slippage`. On XAUUSD M5 the real spread at Vantage can double within seconds
around news and session opens, and fast entries (sweeps, structure breaks)
suffer slippage that is often larger than the spread.

Consequence: the demo expectancy measured today sits slightly **above** the
real-world expectancy. Acceptable while proving an edge exists; **not**
acceptable as the figure that clears the demo→real gate.

Required before wiring `live/mt5_executor.py` / before any real capital:
- Variable spread by session/volatility (reuse `CostsConfig`:
  `spread_xau_london_pips`, `spread_xau_ny_pips`, `spread_xau_off_session_pips`).
- News-window spread widening (multiplier during the news blackout band).
- Entry/exit slippage as a fraction of ATR(M1) (`slippage_*_atr_multiplier`),
  applied pessimistically.

The `CostsConfig` parameters already exist; the paper executor just needs to
consume them instead of the flat `slippage`. Re-measure expectancy under the
friction model and confirm it still clears the gate.
