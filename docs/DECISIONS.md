# DECISIONS.md — Risk & strategy choices that must survive sessions

This file is an **anchor**. The decisions below are not code conventions; they
are risk/strategy choices made deliberately, with explicit reasoning, while we
were lucid. They are written down so that a future session (human or agent)
does not silently re-wire the "default codebase" version and undo a choice that
was made on purpose — the same friction principle as the demo→real gate in
`CLAUDE.md`: decide once, in calm, so the decision isn't renegotiated under
temptation.

**How to read each entry.** Every decision carries its *why* and its
*revise-if*. A decision without its rationale becomes dogma nobody dares
question; a decision *with* its reasoning stays challengeable the day the data
speaks. None of these are forever — they are the current best call, to be
overturned by **measured evidence**, not by intuition or by a fresh agent's
default.

> Status: agreed before the `strategy/` assembly PR. The detector layer (PR
> "SMC detector layer") and the risk layer (PR "Risk foundation") do not depend
> on these; they are wired when `strategy/` assembles `Signal`s.

---

## D1 — Stop-loss: structural invalidation, ATR as a setup-quality gate (option B)

**Decision.** The stop is always placed at the **structural invalidation** of
the idea (order-block edge + `sl_buffer`), and is **never moved tighter** than
that. `sl_atr_cap_multiplier = 1.8` is **not** a stop-placement rule — it is a
**setup-quality gate**: if the structural SL distance `> 1.8 × ATR` (reference
timeframe at signal time), the trade is **skipped** (`sl_atr_cap_mode = "skip"`,
default). Swing width is absorbed by **percentage sizing** (wider stop → smaller
position → constant € risk), never by relocating the stop. The `RiskManager`
keeps the **final veto**; the SL is only an input to sizing, never a bypass of
the risk layer.

**Why.** On XAUUSD M5, 3–4 $ wicks are normal. A stop placed *tighter* than the
structural invalidation (option A) is a premature-stop-out machine: you eat the
loss **and** miss the move when price resumes in your direction after ejecting
you — the worst of both worlds. The original 500 € blow-up was a stop that
erased a winning streak; we will not add a mechanism that *increases* premature
exits. Because % sizing already holds € risk constant, a wide structural stop
never "explodes" risk — it just shrinks the position, which is the correct
behaviour. A setup whose invalidation sits beyond 1.8 × ATR is one where the
structure is too stretched to be reliable on M5: better skipped than taken
small (skip protects capital *and* discipline; aggressive shrink keeps exposure
to a poor setup).

**Revise if.** Measurement over the sample shows the trades skipped by the
1.8 × ATR gate would have been profitable → switch `sl_atr_cap_mode` to
`"shrink"` (configurable, therefore measurable). If premature-stop analysis
shows structural stops are themselves too tight/loose, revisit `sl_buffer`.

---

## D2 — Take-profit ladder: floors, not fixed targets

**Decision.** The TP ladder uses **floors**, not fixed R multiples:
- `tp1 = 1R` (partial close, then SL → break-even).
- `tp2 = max(2R, next_htf_liquidity)` → new field `tp2_rr_floor = 2.0`.
- `tp3 = max(3R, next liquidity beyond)` → new field `tp3_rr_floor = 3.0`.

**Why.** On the prototype, observed R:R clustered around 1.0–1.1 because the
real liquidity sat close. Fixed 2R/3R targets aim at price levels where there
is sometimes **no reason** for price to go — there is no liquidity there to draw
it. A floor gets the best of both: a **market-based target** (the next HTF
liquidity) when one exists far enough out, and a **profitability floor**
otherwise. This is the floor-R-or-liquidity logic that produced the visibly
better result on the TradingView prototype.

**Revise if.** The measured sample shows liquidity-based TP2/TP3 rarely fill, or
that flat fixed targets outperform the floors after costs → reconsider the
targeting mode (`tp2_target_mode` already supports `fixed_rr` / `structure`).

---

## D3 — Entry: both modes configurable, let the data decide

**Decision.** `entry_mode` stays **configurable** with two options:
`"ob_or_50pct"` (enter on a retracement into the OB / 50 % of the micro-impulse
after an LTF ChoCh) and `"break"` (enter on the structure break itself).
Measurement **starts on `ob_or_50pct`**; `"break"` is compared on the **same
sample** before anything is locked.

**Why.** Break vs retracement is the single parameter that moves the
win-rate / R:R trade-off the most, and **no visual intuition can settle it** —
only an out-of-sample measurement can. `ob_or_50pct` waits for confirmation
(more conservative, likely more robust long-term); `"break"` captures the moves
that never retrace. We refuse to pick from memory of the prototype; we let the
data choose.

**Revise if.** On the same measured sample, `"break"` shows better expectancy
after spread + slippage costs → switch the default `entry_mode`.

---

## Standing invariants (not up for renegotiation)

- **No-lookahead is non-negotiable.** The `tests/test_no_lookahead.py` invariant
  (events knowable by bar *k* on the full series are byte-identical to those
  produced when the pipeline only ever saw `df[:k]`) must continue to hold over
  the `strategy/` assembly. A backtest that peeks at the future is the exact lie
  that makes a strategy brilliant on paper and lethal live.
- **The `RiskManager` is the final arbiter.** SL/TP/entry choices feed sizing
  and signal construction; they never override the daily-loss kill switch,
  exposure caps, or the no-order-without-a-stop rule.
- **Demo only until the edge is measured.** See the demo→real gate in
  `CLAUDE.md` (≥ 50 trades, positive expectancy after costs, max DD < 10 %).

---

## Config fields implied by these decisions

To be added/confirmed when `strategy/` is assembled (names indicative):

| Field | Where | Decision | Status |
|---|---|---|---|
| `sl_atr_cap_multiplier = 1.8` | `StopLossConfig` | D1 | to add |
| `sl_atr_cap_mode = "skip"` (`skip`/`shrink`) | `StopLossConfig` | D1 | to add |
| `buffer_xau_usd` | `StopLossConfig` | D1 | exists |
| `tp1_rr = 1.0` | `RiskConfig` | D2 | exists |
| `tp2_rr_floor = 2.0` | `RiskConfig` | D2 | to add |
| `tp3_rr_floor = 3.0` | `RiskConfig` | D2 | to add |
| `tp2_target_mode` (floor uses `next_htf_liquidity`) | `RiskConfig` | D2 | exists |
| `entry_mode` (`ob_or_50pct`/`break`) | `EntryConfig` | D3 | exists (add `break`) |

Also at assembly: reword `CLAUDE.md` rule #4 so the ATR ceiling reads as a
**risk/setup-quality bound**, not an "arbitrary distance".
