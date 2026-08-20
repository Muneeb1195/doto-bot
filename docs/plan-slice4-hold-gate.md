# Slice 4 — Hold Gate Micro-tune (Gated on Telemetry)

Status: **PLAN — gated, do not implement until telemetry window closes**
Owner: execution / exit_decision
Created: 2026-08-20

## 1. Context

`max_hold_hours = 72` is global (`config/settings.ini:184`, `bot/config.py:285` → `BREAKEVEN/max_hold_hours`, default 72).
`bot/exit_decision.max_hold_triggered` is pure time check (`position.time → now ≥ 72h`), called from `bot/execution.check_max_hold` → `manage_positions` priority `MAX_HOLD > MR_EXIT > REVERSAL`.

For H4/H1/M15 day-trading (M15 entry, H1/H4 agreement), 72h ≈ 18× H4 bars. Live book shows 0 positions today (journal ok, bot.log tail confirms `positions_get count=0`), so we have no hold distribution yet. Tuning hold now would be guessing.

Slices 1-3 already shipped (`3bb9dac`): telemetry split (`sanity_volume/spread/tape`, `atr_unavail`, `htf_soft`), budget/clamp fixes, risk cache key, correlation 72h. Hold is the only remaining entry-adjacent gate that touches exits.

## 2. Telemetry sources (already live since 22:36 UTC 2026-08-20)

| Source | Path | Cadence | Key fields |
|---|---|---|---|
| Dashboard state | `data/dashboard_state.json` written by `bot/dashboard.write_dashboard_state` → `GET /api/state` | 10s cycle | `filter_stats.{htf_trend,htf_soft,regime_gate,no_signal,atr_unavail,ml_gate,sanity*_*,mr_cooldown,signals, tail_risk}` per symbol |
| Bot log | `logs/bot.log` | per cycle + per exit | `gate_open/regime/signal/atr/block_reason`, `Max hold X.Xh > 72h — closing`, `MAX_HOLD/MR_EXIT/REVERSAL` via `ExitIntent` |
| Journal | `logs/trades.csv` → `journal.py` | per close | `ticket,symbol,event(CLOSE/SCALE_OUT/CHANDELIER/MR_NAKED_CLOSE),pnl,exit_price` |
| Exit decision (pure) | `bot/exit_decision.decide_exit` | per position per cycle | priority already test-covered |

No extra deploy needed. Counters reset at midnight (`bot/main.py:801`).

## 3. Gate to open Slice 4

**Do NOT touch `max_hold_hours` until BOTH hold:**

A) **Wall-clock window** — 48h of live `_filter_stats` (Thu 22:36 → Sat 00:00 is ~1.3 days; Mon 00:00 close = 2 clean trading days). Minimum 24h of non-flat market if 48h not yet elapsed.

B) **Signal volume** — combined `signals + regime_gate + no_signal ≥ 100` evaluations per symbol (at 6/min, 8 symbols, ~2.8k evals/day). Ensures `sanity_volume/spread/tape` ranking is stable.

Fallback: `tools/diagnose_entry_rate.py --symbol XAUUSD.raw,GBPJPY.raw --years 1` gives lower-bound replay *today* (tape assumed pass, flat-book MR), useful for preview but not a substitute for live split counts.

## 4. What to look for (decision table)

Pull `data/dashboard_state.json` and `logs/trades.csv` after window:

| Signal | Interpretation | Action |
|---|---|---|
| `sanity_*` dominates `ml_gate` | Execution filters (spread/tape/volume) choking before ML | Tune `spf_max_ratio 0.30` / `tape 0.35/0.65 range 1.2` / `volume kappa 1.2` before hold |
| `regime_gate` >> `no_signal` | Fused regime `threshold 45 buffer 3` too tight | Revisit `FUSED_REGIME` before hold |
| `MAX_HOLD` exits ≥ 20% of closes in `logs/trades.csv` last 30d | Holds are the exit, not stops — cap too long | **Candidate for Slice 4** |
| `MAX_HOLD` ≈ 0% and median hold << 72h | Cap irrelevant | No hold change; focus entry gates |
| Hold-duration histogram (from `trades.csv` open→close) shows cluster at 60-72h with negative drift | Positions bleeding into hold | Reduce per-symbol |

## 5. Proposed hold micro-tune (only if table says so)

Keep it **small and per-symbol**, not global, and keep the deep module pure:

* Option A (preferred): per-symbol `max_hold_hours` override via `config/settings.ini` `[STRATEGY:<sym>]` → `SYMBOL_STRATEGY_MAP` → `symbol_cfg`. Example: `XAU 72 → 48`, `US30 72 → 36` for higher vol. No code change beyond config — `exit_decision.max_hold_triggered` already reads `cfg.get("max_hold_hours",72)`.
* Option B: adaptive `max_hold = k * ATR_period` bars (e.g., `k=48` H1 bars ≈ 48h) — deferred; needs backtest parity (`Backtest._check_fused_regime_gate` already time-based, would diverge). Do A first.

**Not in scope:** changing `MR_EXIT` or `REVERSAL 0.25 ATR` guard (`bot/exit_decision.reversal_triggered`) — separate gate, separate telemetry (`mr_cooldown` vs `MAX_HOLD`).

## 6. Guards + rollback

* Change one symbol at a time, max delta `72 → 48` (not `72 → 12`).
* Require `tests/test_parity` + `bot/exit_decision` unit (`max_hold_triggered` injectable `now`) green.
* Live canary: deploy one symbol, watch 48h, compare `MAX_HOLD` rate and realized `pnl` in `logs/trades.csv` vs prior 30d baseline. Roll back by reverting the INI line and `systemctl --user restart doto-bot` (no code revert needed, config is the cap).

## 7. Implementation steps (when gated)

1. `scripts/collect_hold_telemetry.py --days 7` (to add: histogram from `trades.csv` + current `dashboard_state.json` snapshot) → attach output to PR.
2. `config/settings.ini` add per-symbol `max_hold_hours = 48` for the symbol where `MAX_HOLD` dominated.
3. `config/validate_entry_config.py` already range-checks; extend if needed.
4. `.venv/bin/python -m pytest tests/test_parity.py tests/test_position_management.py -q`
5. `git push` → `rsync --checksum bot/ config/settings.ini` → `find __pycache__ -delete` → `systemctl --user restart doto-bot` → `journalctl --user -u doto-bot -n 40` health.
6. Re-evaluate after next 48h window; revert if win-rate or PF drops.

## 8. Timeline

* **Now → Sat 00:00** — collect, no hold change. Run offline replay for preview if desired.
* **Mon 00:00** — 2 full sessions; if `MAX_HOLD` ≥ 20%, open PR for Option A on one symbol.
* **Otherwise** — close Slice 4 as "no hold change needed" and keep telemetry for future `max_open_risk` / `corr` tuning.
