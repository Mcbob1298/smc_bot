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
