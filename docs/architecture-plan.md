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

## Phase B — medium (one-table medicine)

- **C3 — State schema table.** A `PERSISTED` field table drives `save_bot_state`/
  `load_bot_state`/`reset_all` (no more hand-written mirrors); typed accessors for
  position-management state + the daily-loss trio (one rollover rule); watchdog
  globals move to main.py. Live `bot_state.json` must round-trip byte-identically.
- **C4-2 — Raw-value entry score.** Widen `compute_entry_score` to accept
  `ml_conf=None, news_val=None` so backtest passes its precomputed arrays and
  `_compute_entry_score` is deleted (parity test becomes a call-site assertion).

## Phase C — the big one

- **C1 — `manage_positions()` in execution.py.** Move the whole exit tree out of
  main.py including post-close side effects (MR streak, journal, Discord, state
  pops, save); small injected `Market` adapter as the seam; new
  `test_position_management.py` gives the ~200-line exit surface its first tests.

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
