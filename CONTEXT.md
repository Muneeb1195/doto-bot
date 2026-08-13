# Doto MT5 Bot — Domain Glossary

The trading engine's vocabulary. Kept current by architecture grilling; add terms
when a module or concept gets a stable name.

- **Symbol** — a tradable instrument (`BTCUSD.raw`, `XAUUSD.raw`, …). The portfolio
  is the 8-symbol list from `[PORTFOLIO]` in `config/settings.ini`.
- **Position** — an open trade on a symbol, identified by its MT5 ticket, with
  open price, SL, TP, volume, side (buy/sell).
- **Signal** — a direction (`buy`/`sell`) produced by the entry pipeline: MA
  crossover via KAMA/VIDYA, MTF fusion, or mean-reversion.
- **Gate** — a pass/fail check in the entry pipeline (fused regime, MTF, ML,
  execution sanity). Gates decide whether a signal may enter.
- **Entry** — placing a new position (market or limit), sized by risk/Kelly.
- **Exit decision** — the per-position reasoning to close: `MAX_HOLD`,
  `MR_EXIT`, `REVERSAL` (with the sub-profit guard), plus breakeven and
  chandelier trailing.
- **Close order** — the `TRADE_ACTION_DEAL` that flattens a position, and its
  post-close side effects: journal row, Discord alert, streak/state updates.
- **Position management** — `execution.manage_positions` owns both the exit
  decision and the close order for open positions: TP restore,
  breakeven/chandelier/scale-out checks, the close decision tree (`MAX_HOLD` /
  `MR_EXIT` / `REVERSAL`), close construction + retcode handling, and every
  post-close side effect (deviation, throttles, MR streak, journal, Discord,
  state pops, persistence). `main.py`'s cycle loop just calls it per symbol and
  gets the refreshed book back.
- **Market seam** — the boundary between the engine and the broker terminal:
  live MT5 (`mt5_connect.Market` / `LIVE_MARKET`) in production, a synthetic
  fake in tests. The whole exit tree takes its market access across this seam
  (`market=None` → live), so the ~200-line close path is testable without
  monkeypatching `mt5_call`.
- **Order-send contract** — `mt5_connect.mt5_order_send` is the single home for
  order placement (the frame-sensitivity warning that forced twins in
  execution.py/main.py predates the RPyC bridge and is folklore). The real MT5
  quirk is that order_send mutates the request dict in place — callers build a
  fresh request per send.
- **Streak** — consecutive gate-failure count per symbol (hybrid pause policy),
  keyed to the optimize release tag.
- **Symbol config** — `config.symbol_cfg(cfg, symbol)` returns a fresh
  per-symbol dict (deepcopy + strategy + scale-out/chandelier overrides). The
  shared global cfg is never mutated; contamination is impossible by
  construction.
- **News confidence adjustment** — `analytics.apply_news_confidence_mult` is
  the single source for the news boost/half (shared by live filters and
  backtest); inline copies are build failures.
- **Dashboard contract test** — a pytest asserts every key `bot/dashboard.py`
  emits is consumed by `dashboard/templates/index.html`; dead writer fields are
  build failures, not audit findings.
- **Cycle** — one pass of the main loop over all symbols (10s); each cycle
  runs gates, then position management, then (if allowed) entries.
