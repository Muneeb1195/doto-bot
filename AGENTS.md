# Doto MT5 Bot — Agent Context

## Stack
- **Python 3.12** (originally Wine/Linux, now native Windows)
- **MetaTrader5** Python package (native MT5 terminal)
- **scikit-learn** / **XGBoost** / **LightGBM** — ML ensemble
- **FastAPI** + **uvicorn** — dashboard
- **Windows Task Scheduler** — daemon management
- **pandas / numpy** — data processing

## Architecture
- **`main.py`** — Main loop: per-symbol signal generation, entry/exit, 10s cycle
- **`trend_bot.py`** — Original monolithic logic (deprecated, preserved for reference)
- **`signals.py`** — Entry signal generation (MA crossover via KAMA/VIDYA dispatch, MR, scoring, multi-TF fusion)
- **`filters.py`** — Entry filter chain (volume, spread, ML, tape, news, execution gate)
- **`execution.py`** — Order placement (shared `_place_trade_inner`), scale-out, chandelier exit, MR trades
- **`risk.py`** — Position sizing, Kelly, volatility mult
- **`indicators.py`** — Pure pandas technical indicators (KAMA, VIDYA, calc_ma dispatch, EMA kept for MACD, ATR, RSI, ADX)
- **`ml_features.py`** — Feature engineering for ML models (KAMA ratios, not EMA ratios; MACD stays EMA-based)
- **`train_model.py`** — RandomForest/XGB/LGBM ensemble training
- **`calibrate_models.py`** — Isotonically calibrate trained models via holdout set
- **`backtest.py`** — Vectorized backtesting engine
- **`config.py`** — INI-based config loader with per-symbol strategy overrides (supports `ma_type: kama|vidya`). MTF config keys: `mtf_enabled`, `mtf_agreement_threshold`.
- **`state.py`** — Global mutable shared state (module-level dicts/vars)
- **`mt5_connect.py`** — MT5 connection lifecycle, timeout wrapper, TTL-based rate cache
- **`dashboard.py`** — Dashboard state JSON writer (atomic write)
- **`journal.py`** — Trade CSV journal with atomic writes, O(1) append-based close
- **`correlation.py`** — Cross-asset correlation
- **`regime.py`** — ADX-based regime classifier
- **`analytics.py`** — **Single source of truth** for signal/backtest math (fused regime score, volume/OBV filter). Both the live engine (`main.py`/`signals.py`/`filters.py`) and the backtest engine (`backtest.py`/`backtest_njit.py`) MUST call these functions so the two paths cannot diverge (prevention A1). Parity is guarded by `tests/test_parity.py` (prevention A2).
- **`discord_alerts.py`** — Discord webhook embeds
- **`services/news_sentiment.py`** — News sentiment (Marketaux primary, RSS fallback; standalone service)
- **`backup.py`** — Daily tgz archive of data/logs/config/models, 7-day rotation
- **`mc_ruin.py`** — Monte Carlo ruin estimation (bootstrap trade sequences)
- **`mc_validation.py`** — Monte Carlo validation of backtest results
- **`scenario_analysis.py`** — Instantaneous market scenario stress testing
- **`parallel_optimize.py`** — Multi-process parallel walk-forward optimization
- **`auto_optimizer.py`** — Scheduled auto-optimizer
- **`optimize_params.py`** — Shared optimization helpers (two-phase: MA sweep + SL/RR/ADX refinement)
- **`tune_scaleout.py`** — Scale-out parameter tuning
- **`screen_symbols.py`** / **`screen_fast.py`** — Symbol screening utilities

## Key Conventions
- All MT5 calls wrapped via `mt5_call(func, *args, _timeout=N)` from `mt5_connect`
- Rate cache TTL is 5s (`_RATE_CACHE_TTL` in `state.py`); avoids re-fetching within same cycle
- Config is a global `dict` loaded from `settings.ini`. Per-symbol mutation uses `deepcopy(cfg)` in `main.py` to avoid cross-symbol contamination
- State lives in module-level globals in `state.py` (no context/DI): `_scale_out_state`, `_chandelier_state`, `_exec_bias`, `_last_trade_time`, `_tail_risk_triggered`, `_circuit_breaker_triggered`, `_daily_realized_pnl`, `_ml_confidence_history`, `_ml_drift_warned`, `_filter_stats`, `_equity_history`, `_ns_cache`, `_rate_cache`, and more
- Atomic writes: `.tmp` + `f.flush()` + `os.fsync()` + `os.replace()`
- PKR account: balance displayed as `Rs.`, no USD conversion
- MA dispatch: `cfg['ma_type']` — `kama` for non-crypto (XAU500, NZDUSD, US30, GBPJPY), `vidya` for crypto (BTCUSD, SOLUSD, XRPUSD, DOGUSD)
- M1 entry simulation via `_simulate_m1()` in backtest

## Portfolio (7 symbols) — MTF-Optimized (H4/H1/M15 day-trading, Jul 2026)

| Symbol | MA | Fast | Slow | SL | RR | ADX | WF | Notes |
|--------|----|------|------|----|----|-----|-----|-------|
| XAU500 | KAMA | 12 | 48 | 2.0 | 2.5 | 25 | 0.9 | MTF on |
| US30 | KAMA | 6 | 24 | 1.5 | 2.0 | 25 | 2.0 | MTF on |
| GBPJPY | KAMA | 6 | 24 | 1.0 | 2.0 | 25 | 3.0 | MTF on |
| BTCUSD | VIDYA | 12 | 36 | 1.5 | 2.0 | 30 | 1.2 | MTF off, no-ML |
| SOLUSD | VIDYA | 10 | 30 | 1.5 | 2.0 | 28 | 5.7 | MTF on |
| XRPUSD | VIDYA | 3 | 12 | 1.0 | 2.0 | 25 | 2.4 | MTF on |
| DOGUSD | VIDYA | 6 | 18 | 1.5 | 2.0 | 30 | 0.4 | MTF on |

Global defaults (fallback): KAMA 10/40, SL=1.5, RR=2.0, ADX=25, 1%/trade, 5 max positions.

**MTF notes**: BTCUSD gets 0 trades with MTF enabled (M15 crossover too rare for this symbol); `mtf_enabled = false` in `[STRATEGY:BTCUSD.raw]`. Other 6 symbols use H4 trend gate + H1/H4 agreement + M15 MA crossover entry with `agreement_threshold = 0.67`.

## Pool Models (ML)

All 4 asset class pool models trained and loaded:

| Pool | Symbols | Threshold | Accuracy |
|------|---------|-----------|----------|
| Commodity | XAUUSD, XAGUSD, XNGUSD, XAU500, XPTUSD | 0.40 | 88.7% |
| Index | US30, SPY, US500, IWM | 0.30 | 79.0% |
| Crypto | ETHUSD, BTCUSD, LTCUSD, DOGUSD, ADAUSD, XRPUSD, SOLUSD | 0.62 | 79.0% |
| Forex | EURUSD, GBPUSD, USDJPY, EURJPY, GBPJPY, NZDUSD, etc. | — | — |

## Testing
- 349 tests in `tests/` (pytest), covering indicators (KAMA, VIDYA, calc_ma dispatch, efficiency ratio, MA slope, fused regime score), signals (incl. RegimeGate hysteresis, get_mtf_fused_signal), risk, execution, filters (mock), backtest pipeline, state persistence, journal, risk (mock), config, TFT model (`test_tft_model.py`), drift/warm-start (`test_drift_retrain.py`)
- `conftest.py` provides fixture DataFrames and mock configs
- Pure logic tests don't need MT5 mock (mock at `sys.modules` level only when needed)
- Mock MT5 at `mt5_connect.mt5_call` level for integration tests
- Run: `python -m pytest tests/ -v`
- Tests avoid parallel execution (global state mutation)
- Temp dirs cleaned: `test_risk.py` uses `shutil.rmtree` after fixture teardown
- Risk limits in tests: `basic_cfg` fixture in `conftest.py` provides standard config with `daily_loss_pct=5.0`, `dr_enabled=True`, `kelly_fraction=0.25`

## Windows Service Architecture (Task Scheduler)

| Task | Trigger | Restart | Port |
|------|---------|---------|------|
| `DotoBot` | At logon | Restart every 1 min (3x) | — |
| `DotoDashboard` | At logon | Restart every 1 min (3x) | 8501 |
| `DotoNewsSentiment` | At logon | Restart every 1 min (3x) | — |
| `DotoOptimizer` | Daily (e.g. 02:00) | oneshot | — |
| `DotoRetrain` | Weekly (Sun 03:00) | oneshot | — |
| `DotoBackup` | Daily (04:00) | oneshot | — |
| `DotoWeeklySummary` | Weekly (Mon 05:00) | oneshot | — |
| `DotoTerminal` | At logon (startup folder) | — (MT5 auto-restart setting) | — |

MT5 terminal auto-starts via `shell:startup`, minimized to tray.

## Clean redeploy
- `scripts/redeploy.ps1` ends the `DotoBot`/`DotoDashboard` tasks, kills any orphaned previous child via the `svc_launcher` `.child.pid` watchdog file, restarts the tasks, and polls `logs/bot.log` for the `"Bot state loaded"` marker — failing loudly if the bot does not come back healthy (prevention D2).
- `migration/svc_launcher.py` is the headless launcher; each instance records its child PID to `logs/<service>.child.pid` and kills the prior orphan on startup, so scheduled restarts never leave two copies running.

## CI
- GitHub Actions (`.github/workflows/ci.yml`): lint (ruff), typecheck (mypy bot/), shellcheck, **secret-scan** (gitleaks + grep fallback for hardcoded `password=`/`api_key=`/`token=` in tracked `.py`/`.ps1`, excluding the git-ignored `config/credentials.ini` via `.gitleaksignore`), and test (pytest).

## CI
- GitHub Actions (`.github/workflows/ci.yml`): lint (ruff), typecheck (mypy bot/), shellcheck, test (pytest)
- Pre-commit (`.pre-commit-config.yaml`): ruff, mypy, trailing-whitespace, end-of-file-fixer, check-yaml
- Pip cache enabled for faster installs

## Data Files
- `data/bot_state.json` — Trade state (atomic write, crash-safe)
- `data/dashboard_state.json` — Dashboard snapshot (atomic write, 5000 equity history entries)
- `data/news_sentiment.json` — News sentiment cache
- `logs/trades.csv` — Trade journal (O(1) append on close)
- `logs/bot.log` — Active log (TimedRotatingFileHandler, midnight UTC rotation; rotated files named `bot.log.YYYY-MM-DD`)
- `models/model_{symbol}.pkl` — ML ensemble models (per-symbol + pool: model_pool_{commodity,index,crypto,forex}.pkl)
- `models/model_{symbol}.calib.npz` — Calibration holdout files
- `backups/doto_backup_YYYYMMDD_HHMMSS.tar.gz` — Daily backup archive (7-day rotation via `backup.py`)

## Risk Limits
- Per trade: 1% account risk (quarter Kelly adjusted)
- Max positions: 5 total, 1 per symbol
- Daily loss limit: 5% → halt till next day
- Circuit breaker: 15% drawdown → permanent halt
- Tail risk: 3σ portfolio PnL → 60min cooldown
- Correlation: sizing reduced up to 50% when symbols correlate

## Optimization
- Two-phase by default: Phase 1 = MA param sweep (6 pairs × 2 windows), Phase 2 = SL/RR/ADX refinement (top 2 MA pairs × all windows)
- ~4x speedup vs full grid with minimal quality loss
- `optimize_params.py --symbols X,Y --years 3 --two-phase`
- Speed comes from `ProcessPoolExecutor` (one worker per CPU, leaving 1 core) + per-process ML model/ML-multiplier caches (`_ML_DATA_CACHE`/`_ML_MULT_CACHE` in `backtest.py`) + a Numba-JIT fast path (`backtest_njit._simulate_core`) that is the default when `fast=True`. The pandas loop remains the reference path for parity tests.

## Phase 3 — Multi-TF Fusion (2026-07-22, daytrading H4/H1/M15)
- **`get_mtf_fused_signal()`** in `signals.py` — H4 trend bias + H1/H4 agreement gate + M15 MA crossover entry. Returns (signal, atr, entry_type, agreement_ratio). H4 determines bias, H1/H4 must agree on direction; M15 provides entry timing when its MA cross aligns with the H4/H1 consensus. Falls back to H1 pullback entry when M15 produces no crossover but H4/H1 bias exists.
- **Derived periods**: M15 uses independent MA periods derived from H1: M15 fast = max(5, H1_fast//2), M15 slow = max(8, H1_slow//2) (overridable via `mtf_m15_ema_fast/slow`). H4 uses fixed 100-period EMA.
- **`main.py`** — when `mtf_enabled=True`, uses `get_mtf_fused_signal()`. MTF agreement ratio scales kelly_mult (0.5-1.0).
- **`backtest.py`** — `_get_mtf_signal()` for MTF path; `_precompute()` precomputes M15 MAs for H1-aligned bars (same derivation as live for parity). Numba fast path disabled when `mtf_enabled=True`.
- **Config**: `[MTF] enabled = True`, `agreement_threshold = 0.67`.

**Optimizer changes**: `fetch_m15_data()` uses `copy_rates_from` with **backward paging** (via `mt5_connect.fetch_rates_paged`, `chunk_bars=MAX_M15_BARS=80000`) to stitch a full 3y window — a single per-request call was capped at ~80k bars (~2.3y M15) and silently truncated the early window. M15 fetched before H1 (MT5 bug: H1→M15 order causes -2 error). `mtf_enabled` added to `build_params()` override map for per-symbol disable. BTCUSD.raw uses `mtf_enabled = false` in INI.

## Full timeframe coverage in training & optimization (2026-07-23)
- **M1 (orderflow `of_*`)**: now fetched unconditionally in `train_model` (per-symbol + pool), `optimize_params`, `parallel_optimize`, and `auto_optimizer`, and aggregated into `of_*` columns once per symbol via `ml_features.attach_orderflow_features(df, m1_df)` BEFORE window slicing. This closes the train/serve skew where `of_*` were always NaN→0.0 (item #11 follow-up).
- **M15 (MTF entry TF)**: `df_m15` is now threaded into `Backtest(df, params, df_m15=...)` in all three optimizers, so MTF-enabled symbols (XAU500, US30, GBPJPY, SOLUSD, XRPUSD, DOGUSD) are evaluated on the real MTF signal path instead of the degraded M15-less path. Previously only `optimize_params` passed M15; `parallel_optimize` and `auto_optimizer` ran H1-only.
- **H1/H4**: H1 fetched directly; H4 resampled from H1 in the backtest (H1 is deep/cheap from the broker). H1/H4 are NOT built from M15 (broker keeps shallower M15 history; resampling would cap depth and risk live/backtest parity divergence).
- **Paging helper**: `mt5_connect.fetch_rates_paged(symbol, tf, start, end, chunk_bars=80000)` walks backwards from `end`, each page ending at the prior page's oldest timestamp, deduplicates seams, trims to `[start, end]`. Terminal "Max bars in chart" was raised to Unlimited on the trading box, but the code-side per-request cap (~80k) is the real limiter the pager removes; total depth still bounded by broker server history for that timeframe.

## Optimizer symbol scope fix (2026-07-23)
- **Root cause**: `DotoOptimizer` (daily 02:00) ran `optimize_params.py --symbols ALL`, and `ALL` resolved to a hardcoded 24-symbol "universe" in `optimize_params.py`/`parallel_optimize.py` `SYMBOL_PROFILE` — including `ADAUSD.raw` and other symbols not in the live portfolio. This made the bot auto-optimize symbols it never trades.
- **Fix**: `ALL` now resolves to `[PORTFOLIO] symbols` from `settings.ini` (the real 7-symbol portfolio) instead of the stale hardcoded list. Self-syncing: changing the portfolio in settings.ini automatically updates what the optimizer targets — no more drift.
- **`SYMBOL_PROFILE` trimmed** in both `optimize_params.py` and `parallel_optimize.py`: removed 6 dead symbols with no model and no pool membership (`AUS200`, `BCHJPY`, `ADBE`, `AMGN`, `AVGO`, `AUDPLN`, `UK100`). Kept the 7 portfolio symbols + pool members that have trained models (ETHUSD, LTCUSD, ADAUSD, AVXUSD, forex/index/commodity pool symbols, etc.) so explicit per-symbol optimization still works.
- **Net effect**: the nightly `DotoOptimizer` run now optimizes exactly the 7 portfolio symbols; non-portfolio pool members (e.g. ADAUSD) are only optimized if explicitly passed via `--symbols`.

## Scoring Parity (2026-07-22)
- **`analytics.compute_entry_score()`** — single source of truth for entry scoring. Uses 3-component model: ML (40%), spread (30%), news (30%). Weights from `cfg["scoring_weights"]`.
- **`backtest.py:_compute_entry_score()`** — backtest scoring model. Now uses the same 3-component weights from config. Previously skipped components not in `scores` dict (e.g., news when `ns_enabled=False`), causing score divergence. Fixed: weighting loop now uses `scores.get(key, 0.5)` to include all weighted components with default 0.5.
- **`filters.py:check_ml_gate()`** — live ML gate. Applies news-based confidence adjustment: `news_val >= 0.70` → `confidence_mult * 1.10` (capped 1.5); `news_val <= 0.30` → `confidence_mult * 0.50`. Backtest's `_run_reference()` now applies the same adjustment.
- **`mr_min` parity**: Both paths use `mr_min = 0.03 if entry_atr is None else 0.0` (previously backtest used `entry_type == "mean_reversion"`).
- **Parity tests** in `tests/test_parity.py::TestScoringParity` guard against future divergence.
