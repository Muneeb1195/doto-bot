# Architecture Deepening Plan (grilled 2026-08-13)

Six candidates surfaced by the codebase-architecture review, grilled and sequenced
by risk. Acceptance bar for every phase: full test suite green, byte-identical
backtest trade streams, and — for anything touching `filters.py`/`execution.py`/
`config.py` — the server deploy dance (scp → md5 parity → restart → healthy boot).

## Phase A — safe quick wins

- **C6 — Dashboard contract.** Drop `positions_detail.ticket` (emitted, read by
  nobody); add a pytest that every emitted key in `bot/dashboard.py` is consumed
  by `dashboard/templates/index.html` (dead fields found by the interface, not audit).
- **C5 — `config.symbol_cfg(cfg, symbol)` factory.** Kills the 4 duplicated
  `deepcopy` + `apply_symbol_strategy` + `apply_symbol_overrides` rituals
  (3× in main.py, 1× in diagnose_entry_rate.py); fresh-dict invariant by construction.
- **C4-1 — `analytics.apply_news_confidence_mult(confidence_mult, news_val)`.** The
  news-based confidence adjustment is a byte-identical twin at `backtest.py:1896`
  and `filters.py:73`; both call the shared function; njit mirror stays as-is.

## Phase B — medium (one-table medicine) — DONE 2026-08-13

- **C3 — State schema table.** ✅ A `PERSISTED` field table drives
  `save_bot_state`/`load_bot_state`/`reset_all` (no more hand-written mirrors);
  `daily_realized_pnl_for` (read-only) + `roll_daily_realized_pnl` (journal-owned
  mutation) are the single homes for the daily-loss boundary rule — filters must
  NOT zero the counter (asserted by test_loss_resets_on_new_day). Watchdog
  globals stay in state.py (deferred — not part of this phase). Live
  `bot_state.json` round-trips byte-identically (schema round-trip test).
- **C4-2 — Raw-value entry score.** ✅ `compute_entry_score` now takes
  `ml_conf=None, news_val=None, tail_risk=None`; backtest supplies per-bar raw
  inputs (ml mult, bar spread in price units, stateful `_tail_risk_score`);
  `_compute_entry_score` is DELETED. Parity test is a call-site assertion;
  fast/reference PNL+EQ MATCH verified via tools/parity_check.py.

## Phase C — the big one — DONE 2026-08-13

- **C1 — `manage_positions()` in execution.py.** ✅ The whole exit tree moved out
  of main.py into `execution.manage_positions(sym_cfg, symbol, all_positions,
  atr, trend_signal, regime, market=None)`, including all post-close side
  effects (MR streak, journal, Discord, state pops, save) — the module returns
  the refreshed book. `mt5_connect.Market` (+ `LIVE_MARKET`) is the injected
  seam; `check_breakeven`/`check_chandelier_exit`/`check_scale_out` and the
  cross-module helpers (`signals.check_mean_reversion_exit`,
  `regime.get_current_atr`) take an optional `market` and thread it through.
  New `test_position_management.py` (12 tests) drives the exit tree through a
  fake market. Test-infra note: the lifecycle fixture now restores the saved
  modules unconditionally — execution imports signals at module level, so a
  re-imported sim-tainted `signals` was leaking into later test files.

## Phase D — gated / follow-ups

- **C2 — order_send consolidation.** RESOLVED 2026-08-13: the frame-sensitivity
  comment predates the RPyC migration (folklore), so both `mt5_order_send` twins
  collapse into one `mt5_connect.mt5_order_send`. Verify with a test trade at next
  deploy; the real quirk (MT5 mutates the request dict in place) stays documented.
- **C4-3 —** spread-decision extraction; replay-tool mirrors delegate to backtest
  (4th gen → 3rd gen); gate generation register in AGENTS.md.
- **C5 — `CONFIG_FIELDS` one-table rewrite** of the loader (kills the four-registry
  tower: `SYMBOL_OVERRIDE_KEYS`/`SYMBOL_STRATEGY_MAP`/`KEY_MAP`/`_global_strategy_defaults`).

## Through-line

Five of six candidates converge on two tools: (1) one registry/table as the single
source of truth (C3 `PERSISTED`, C5 `CONFIG_FIELDS`), (2) contract tests instead of
audits (C6 key coverage, C2 twin whitelist, C4 call-site assertions). Phase A's
C4-1/C6 are the smallest instances of both patterns — the templates for everything else.
