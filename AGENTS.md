# Doto MT5 Bot — Agent Context

## Stack
- **Python 3.12** (dev/CI; the home-server venv is Python **3.14** — see Key Conventions)
- **mt5linux 0.1.10** — SOLE MT5 backend via the RPyC bridge (mt5server.exe, :18812) on the home-server; no native MetaTrader5 package on Linux (mt5linux ≥1.0.0 uses a Docker-backed client — CI mocks it in tests/conftest.py)
- **scikit-learn** / **XGBoost** / **LightGBM** — ML ensemble
- **FastAPI** + **uvicorn** — dashboard
- **systemd user units** — daemon management on the home-server
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
- **`train_model.py`** — XGB+LGBM ensemble training (optional TFT member; no RandomForest — `EnsembleModel(xgb, lgb, tft_model=...)` averages calibrated XGB + LGBM)
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
- **`auto_optimizer.py`** — Param-apply helpers (`load_portfolio`/`update_symbol_strategy`/`write_settings`) reused by `scripts/download_models.py`; no local scheduler
- **`optimize_params.py`** — the **single optimizer entry point** (the former `parallel_optimize.py` was merged in; shared helpers live in `optimizer_common.py`). Flags: `--two-phase` (default), `--full-grid`, `--cpcv`, `--quick`, `--fast`/`--no-fast`, `--no-ml`, `--m1-sim`, `--auto-train`, `--cpcv-paths`, `--years`, `--symbols` (default `ALL` = `[PORTFOLIO]`), `--csv` (offline from `data/history/*_<TF>.csv`), `--fetch-csv` (harvest H1/M15/M1 from MT5 → `data/history/<SYMBOL>_<TF>.csv`, then exit; mutually exclusive with `--csv`)
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
- **Python version split**: home-server venv is Python **3.14** (cpython-314 .pyc), dev + CI use **3.12**. Don't assume version parity when reproducing live behaviour locally; numba JIT cache keys in optimize.yml are already py-version-keyed
- **Live env (verified 2026-08-12)**: MT5 terminal **build 6101** under Wine; broker reports `trade_stops_level = 0` on all symbols, so `_min_stop_points` in `execution.py` enforces **spread + 10 pts** as the stop floor

## Portfolio (8 symbols) — MTF-Optimized (H4/H1/M15 day-trading, Aug 2026)

Params below are the **live-applied** state as of the `optimize-20260812-1348` cycle
(plateau pick, gated DSR ≥ 0.95 / PBO ≤ 0.50; `score` = `scoring_min_entry`).
The repo `config/settings.ini` was synced to this exact state on 2026-08-12;
it will drift again after the next optimize cycle (CI writes, the box applies,
the repo is only the seed — see Deployment drift below).

| Symbol | MA | Fast | Slow | SL | RR | ADX | score | MTF | Gate |
|--------|----|------|------|----|----|-----|-------|-----|------|
| BTCUSD | VIDYA | 3 | 12 | 1.5 | 3.0 | 30 | 0.60 | off | PASS |
| US30 | KAMA | 10 | 40 | 2.0 | 2.0 | 25 | 0.60 | on | PASS |
| GBPJPY | KAMA | 6 | 24 | 2.0 | 2.5 | 22 | 0.55 | on | PASS |
| SOLUSD | VIDYA | 6 | 18 | 1.5 | 2.0 | 25 | 0.60 | on | PASS |
| XRPUSD | VIDYA | 8 | 24 | 1.5 | 3.0 | 30 | 0.55 | on | PASS |
| EURUSD | KAMA | 12 | 48 | 1.0 | 1.5 | 22 | 0.70* | off | FAIL x1 |
| US500 | KAMA | 10 | 40 | 1.0 | 2.0 | 22 | 0.55 | off | PASS |
| XAUUSD | KAMA | 10 | 40 | 2.0 | 2.5 | 25 | 0.75* | on | FAIL x1 |

\* EURUSD (PBO=0.92 > 0.50) and XAUUSD (DSR=0.934 < 0.95) failed the gate on
this cycle; per the hybrid policy both are at **failure strike 1**: best params
re-applied with a TIGHTENED entry (`scoring_min_entry` + 0.15), still trading.
`.symbol_streaks.json` tracks the streaks; a 2nd consecutive failure pauses new
entries (`trading_enabled = false`), a fresh pass resets the streak.

Global defaults (fallback): KAMA 10/40, SL=1.5, RR=2.0, ADX=25, 1%/trade, 5 max positions.

**MTF notes**: BTCUSD, EURUSD and US500 get 0 trades with MTF enabled (M15
crossover too rare for those symbols); `mtf_enabled = false` in their
`[STRATEGY:*]` sections. The other 5 symbols (US30, GBPJPY, SOLUSD, XRPUSD,
XAUUSD) use H4 trend gate + H1/H4 agreement + M15 MA crossover entry with
`agreement_threshold = 0.67`.

## Pool Models (ML)

All 4 asset class pool models trained (fixed 5 symbols each, see `POOL_SYMBOLS` in `train_model.py`) and loaded:

| Pool | Symbols | Threshold | Accuracy |
|------|---------|-----------|----------|
| Commodity | XAUUSD, XAGUSD, XNGUSD, XPTUSD, XAU500 | 0.40 | 88.7% |
| Index | US500, US100, US30, UK100, JP225 | 0.30 | 79.0% |
| Crypto | BTCUSD, ETHUSD, SOLUSD, XRPUSD, DOGUSD | 0.62 | 79.0% |
| Forex | EURUSD, USDJPY, GBPUSD, AUDUSD, USDCAD | — | — |

## Testing
- 349 tests in `tests/` (pytest), covering indicators (KAMA, VIDYA, calc_ma dispatch, efficiency ratio, MA slope, fused regime score), signals (incl. RegimeGate hysteresis, get_mtf_fused_signal), risk, execution, filters (mock), backtest pipeline, state persistence, journal, risk (mock), config, TFT model (`test_tft_model.py`), drift/warm-start (`test_drift_retrain.py`)
- `conftest.py` provides fixture DataFrames and mock configs
- Pure logic tests don't need MT5 mock (mock at `sys.modules` level only when needed)
- Mock MT5 at `mt5_connect.mt5_call` level for integration tests
- Run: `python -m pytest tests/ -v`
- Tests avoid parallel execution (global state mutation)
- Temp dirs cleaned: `test_risk.py` uses `shutil.rmtree` after fixture teardown
- Risk limits in tests: `basic_cfg` fixture in `conftest.py` provides standard config with `daily_loss_pct=5.0`, `dr_enabled=True`, `kelly_fraction=0.25`

## Environment Boundaries (dev machine ~ home-server ~ GitHub Actions)

The project has three roles and they must NOT overlap:

| Role | Location | Owns | Runs |
|------|----------|------|------|
| **Dev / CI** | Dev machine + GitHub PRs | code, tests, docs | `ci.yml` on every PR; local pytest/lint only |
| **Home-server** | 192.168.1.15 (Arch/Wine box) | MT5 terminal, live trading, ops jobs | `mt5`, `mt5server` (RPyC :18812), `doto-bot`, `doto-dashboard` (:8501), `doto-news`, `doto-backup`, `doto-download` (hourly fetch), `doto-orchestrate` (monthly dispatch) |
| **GitHub Actions** | cloud runners | optimization + ML training | `train.yml`, `optimize.yml` — both dispatched by the home-server, never by cron |

- **Optimization and ML training run ONLY on GitHub Actions.** The home-server does
  NOT run `auto_optimizer.py`/`train_model.py` locally (the old `doto-optimizer`/
  `doto-retrain` timers were removed) — it would compete with the live bot for CPU
  and duplicate CI-owned jobs.
- The home-server **triggers the workflows and pulls the results back
  automatically** via `scripts/download_models.py`:
  - `--dispatch` (run by `doto-orchestrate` timer, 1st of month): exports market
    data (`export_mt5_data.py --no-git`) and uploads the M1 CSVs to a `data-*`
    release (`scripts/push_data.py`), uses the `gh` CLI to run `train.yml`, then
    `optimize.yml --field mode=monthly`, waits for each to complete, downloads
    `models.tar.gz` + `strategy-params.json`, applies the params to
    `settings.ini`, and restarts `doto-bot`.
  - `--fetch-only` (default; run by `doto-download` timer hourly as a safety net):
    pure URL/urllib download of the latest `train-*`/`optimize-*` releases, no `gh`
    needed. Tracks progress in `.last_train_tag`/`.last_optimize_tag` (two files —
    models and params come from separate release streams).
- **Gate-failure hybrid policy** (`optimize.yml` publish → `download_models.py`):
  a symbol whose plateau pick fails the DSR/PBO gate is published in
  `failed-params.json` (its best plateau params, same shape as
  `strategy-params.json`). The box tracks consecutive failures in
  `.symbol_streaks.json`: **1st failure** → re-apply those params with a
  TIGHTENED entry (`scoring_min_entry` + 0.15, capped at 0.90), symbol keeps
  trading; **2nd+ consecutive failure** → `trading_enabled = false` in
  `[STRATEGY:<sym>]` in `settings.ini` (new entries paused, existing positions
  still exit via position management). A fresh pass resets the streak and
  re-enables the symbol. `trading_enabled` is a per-symbol `SYMBOL_STRATEGY_MAP`
  key (bool-converted, defaults true).
  - Requires `gh` (github-cli) + a fine-grained PAT (Actions: write, Contents:
    write), injected as `GITHUB_TOKEN` via the unit EnvironmentFile
    `~/.config/doto-orchestrate.env` — the file systemd actually reads. A token
    pasted into `config/credentials.ini` (which holds only the MT5 login +
    Discord webhook) is silently ignored.
- **Market data flows via GitHub releases, not git.** M1 history (~2 GB for all
  symbols) would blow the git-lfs bandwidth quota if checked out per CI run.
  Instead `data/history/*_M1.csv` files are uploaded as individual assets to a
  `data-<ts>` release by `scripts/push_data.py`; `train.yml` and the `optimize.yml`
  shards download exactly the `*_M1.csv` (or per-symbol `<SYM>_M1.csv`) assets they
  need via `gh release download`. H1/M15 CSVs stay in plain git. Prune keeps only
  the 2 newest `data-*` releases.
- The home-server **deploys via scp** (it has no `.git`): copy changed files to
  `~/doto-mt5-bot/`, clear `bot/__pycache__`, `systemctl --user restart doto-bot`.
- **systemd user units are NOT in the repo** (`git ls-files` finds zero `.service`/
  `.timer`). They live only at `~/.config/systemd/user/` on the server: the MT5
  display stack (`xvfb-mt5`/`fluxbox-mt5`/`x11vnc-mt5`/`xfce4-mt5`/`mt5`/`mt5server`)
  is generated by `scripts/deploy-linux.sh` (Phases 5-6); the `doto-*` units
  (`doto-bot`, `doto-dashboard`, `doto-dashboard-publish`, `doto-backup`,
  `doto-download`, `doto-news`, `doto-orchestrate`, + timers) were hand-added. A
  previously referenced `failure-alert` unit no longer exists (verified
  2026-08-13). Editing a unit in git changes nothing until it is
  regenerated/edited on the server.

## Deployment drift (verified 2026-08-12)

Because the box has no git and is updated by manual scp, the live copy can lag
`main`. Known at the last audit:

- Live `bot/execution.py` was **behind main** at audit: missing commit `27427c1`'s
  emergency stop-loss for naked positions (when a close fails, the position is
  live with no SL/TP — set a wide emergency SL via `TRADE_ACTION_SLTP`) and the
  post-`TRADE_RETCODE_REQUOTE` position-dedup check (verify with
  `mt5.positions_get` before retrying to avoid double-placing). Deployed and
  verified live on 2026-08-12 — future deploys: push the file, clear
  `bot/__pycache__`, restart `doto-bot`.
- Live `bot/optimize_params.py` / `bot/train_model.py` also differed, but those
  files **never run on the home-server** (optimization + training run only on
  GitHub Actions), so their drift does not affect live trading.
- After every deploy, verify parity: `md5sum bot/*.py scripts/*.py` on both
  sides. `main.py`, `signals.py`, `filters.py`, `scripts/download_models.py`,
  `scripts/push_data.py` were in sync at the audit.
- **Never scp `config/settings.ini` from dev to the server.** The server's copy
  is the live source of truth: `scripts/download_models.py` mutates it with the
  CI-applied params (`scoring_min_entry`, `trading_enabled`, hybrid-policy
  changes). Overwriting it with the repo's copy reverts the applied params. The
  repo's `settings.ini` is the seed used by CI for the symbol matrix and
  naturally lags the live-applied state after each cycle.

## Clean redeploy
- `scripts/redeploy.sh` (Linux) restarts `doto-bot` + `doto-dashboard` and polls
  `logs/bot.log` for the `"Bot state loaded"` marker — failing loudly if the bot
  does not come back healthy (prevention D2).
- Windows-era artifacts (`migration/`, `scripts/redeploy.ps1`, Task Scheduler tasks,
  `svc_launcher.py`) were removed with the Windows deployment. Do not reintroduce them.

## CI
- GitHub Actions (`.github/workflows/ci.yml`): lint (ruff), typecheck (mypy bot/), shellcheck, **secret-scan** (gitleaks + grep fallback for hardcoded `password=`/`api_key=`/`token=` in tracked `.py`/`.ps1`, excluding the git-ignored `config/credentials.ini` via `.gitleaksignore`), and test (pytest).
- Pre-commit (`.pre-commit-config.yaml`): ruff, mypy, trailing-whitespace, end-of-file-fixer, check-yaml
- Pip cache enabled for faster installs

## Data Files
- `data/bot_state.json` — Trade state (atomic write, crash-safe). `tail_risk_cooldown`
  values are epoch **expiry** timestamps (`now + tr_cooldown*60` in `risk.py`) — the
  cooldown is active while `now < value`, not a "triggered at" clock
- `data/dashboard_state.json` — Dashboard snapshot (atomic write, 5000 equity history entries)
- `data/news_sentiment.json` — News sentiment cache
- `logs/trades.csv` — Trade journal (O(1) append on close)
- `logs/bot.log` — Active log (TimedRotatingFileHandler, midnight UTC rotation; rotated files named `bot.log.YYYY-MM-DD`)
- `models/model_{symbol}.pkl` — ML ensemble models (per-symbol + pool: model_pool_{commodity,index,crypto,forex}.pkl)
- `models/model_{symbol}.calib.npz` — Calibration holdout files
- `models/` **accumulates stale model files across cycles**: `download_models.py::_extract_models`
  only MOVES files in (flatten from the tar), never prunes. Strays (old symbols,
  retired pools) must be archived manually (e.g. `models/_archive/`) + verified
  against the running bot before deletion
- `backups/doto_backup_YYYYMMDD_HHMMSS.tar.gz` — Daily backup archive (7-day rotation via `backup.py`)

## Risk Limits
- Per trade: 1% account risk (quarter Kelly adjusted)
- Max positions: 5 total, 1 per symbol
- Daily loss limit: 5% → halt till next day
- Circuit breaker: 15% drawdown → permanent halt
- Tail risk: 3σ portfolio PnL → 60min cooldown
- Correlation: sizing reduced up to 50% when symbols correlate

## Optimization
- `optimize_params.py` is the **single optimizer entry point** — `parallel_optimize.py` is deleted (its `optimize_symbol_parallel` was the same algorithm as `--two-phase`); shared helpers live in `optimizer_common.py`. Symbol default `ALL` resolves to `[PORTFOLIO] symbols` from settings.ini (never a hardcoded list).
- Two-phase by default: Phase 1 = MA param sweep (6 pairs × 2 windows), Phase 2 = SL/RR/ADX refinement (top 2 MA pairs × all windows). ~4x speedup vs full grid with minimal quality loss.
- **Flag surface**: `--two-phase` (default) · `--full-grid` (weekly CI) · `--cpcv` (monthly CI) · `--quick` · `--fast`/`--no-fast` (Numba-JIT vs reference loop) · `--no-ml` · `--m1-sim` · `--auto-train` · `--cpcv-paths N` · `--years N` · `--symbols S1,S2` · `--csv` (offline: reads `data/history/*_<TF>.csv`, no MT5 terminal — how CI runs it) · `--fetch-csv` (harvest H1/M15/M1 from MT5 into `data/history/<SYMBOL>_<TF>.csv` then exit; requires the live terminal, so it is mutually exclusive with `--csv`; replaces the deleted `parallel_optimize._fetch_csv_mode`, which wrote the unreadable `<SYMBOL>.csv`)
- Example: `optimize_params.py --symbols X,Y --years 3 --two-phase` (live MT5) or `... --csv` (offline)
- Speed comes from `ProcessPoolExecutor` (one worker per CPU, leaving 1 core) + per-process ML model/ML-multiplier caches (`_ML_DATA_CACHE`/`_ML_MULT_CACHE` in `backtest.py`) + a Numba-JIT fast path (`backtest_njit._simulate_core`) that is the default when `fast=True`. The pandas loop remains the reference path for parity tests.
- The pre-CI local sweepers (`scripts/_archive/sweep_symbols.py`, `sweep_remaining.py`) are **archived** (2026-08-13): superseded by `optimize.yml` (CI-only), hardcoded Windows venv path, and `sweep_remaining.py` parses an optimizer output format that no longer exists. See `scripts/_archive/README.md` — only the resume/WF-ranking pattern is worth porting to CI.

## Manual tools (maintained — dev-side, not CI-owned)
- `bot/validate_entry_config.py` — pre-flight settings.ini sanity check before an optimization run: `python -m bot.validate_entry_config [--symbols X,Y] [--min-entry-score 0.55]`. Exits non-zero on missing `[STRATEGY:<sym>]` sections, out-of-range params, or missing ML model files, so a long grid run never fails midway. Sole kept copy — the former `tools/validate_entry_config.py` was removed (it needed the native MetaTrader5 package and imported `compute_entry_score` from the old signals location).
- `tools/parity_check.py` — dev-side guard that `Backtest.run(fast=False)` (pandas reference) and `fast=True` (Numba JIT) produce identical trades/equity on synthetic data; prints `PNL MATCH`/`EQ MATCH`.
- `tools/trace_parity.py` — same fast-vs-reference comparison, but stops at the FIRST diverging trade and dumps both trade streams around it (entry/exit bars, exit reason, type, PNL) to debug JIT drift.
- `tools/diagnose_entry_rate.py` — replays the live entry pipeline (fused-regime gate → MTF/MR signal → htf trend → entry score → exec sanity) on historical data to diagnose why signals fail (`python tools/diagnose_entry_rate.py --symbol X --years 1`); mirrors `backtest.py` bar-for-bar (parity-verified), MR replay assumes a flat book (lower bound), requires a live MT5 terminal.

## Phase 3 — Multi-TF Fusion (2026-07-22, daytrading H4/H1/M15)
- **`get_mtf_fused_signal()`** in `signals.py` — H4 trend bias + H1/H4 agreement gate + M15 MA crossover entry. Returns (signal, atr, entry_type, agreement_ratio). H4 determines bias, H1/H4 must agree on direction; M15 provides entry timing when its MA cross aligns with the H4/H1 consensus. Falls back to H1 pullback entry when M15 produces no crossover but H4/H1 bias exists.
- **Derived periods**: M15 uses independent MA periods derived from H1: M15 fast = max(5, H1_fast//2), M15 slow = max(8, H1_slow//2) (overridable via `mtf_m15_ema_fast/slow`). H4 uses fixed 100-period EMA.
- **`main.py`** — when `mtf_enabled=True`, uses `get_mtf_fused_signal()`. MTF agreement ratio scales kelly_mult (0.5-1.0).
- **`backtest.py`** — `_get_mtf_signal()` for MTF path; `_precompute()` precomputes M15 MAs for H1-aligned bars (same derivation as live for parity). Numba fast path disabled when `mtf_enabled=True`.
- **Config**: `[MTF] enabled = True`, `agreement_threshold = 0.67`.

**Optimizer changes**: `fetch_m15_data()` uses `copy_rates_from` with **backward paging** (via `mt5_connect.fetch_rates_paged`, `chunk_bars=MAX_M15_BARS=80000`) to stitch a full 3y window — a single per-request call was capped at ~80k bars (~2.3y M15) and silently truncated the early window. M15 fetched before H1 (MT5 bug: H1→M15 order causes -2 error). `mtf_enabled` added to `build_params()` override map for per-symbol disable. BTCUSD.raw uses `mtf_enabled = false` in INI.

## Full timeframe coverage in training & optimization (2026-07-23)
- **M1 (orderflow `of_*`)**: now fetched unconditionally in `train_model` (per-symbol + pool), `optimize_params`, and `auto_optimizer`, and aggregated into `of_*` columns once per symbol via `ml_features.attach_orderflow_features(df, m1_df)` BEFORE window slicing. This closes the train/serve skew where `of_*` were always NaN→0.0 (item #11 follow-up).
- **M15 (MTF entry TF)**: `df_m15` is now threaded into `Backtest(df, params, df_m15=...)` in the optimizer and `auto_optimizer`, so MTF-enabled portfolio symbols (US30, GBPJPY, SOLUSD, XRPUSD, XAUUSD — the `mtf_enabled = false` trio BTCUSD/EURUSD/US500 excluded) are evaluated on the real MTF signal path instead of the degraded M15-less path. Previously only `optimize_params` passed M15; `auto_optimizer` ran H1-only.
- **H1/H4**: H1 fetched directly; H4 resampled from H1 in the backtest (H1 is deep/cheap from the broker). H1/H4 are NOT built from M15 (broker keeps shallower M15 history; resampling would cap depth and risk live/backtest parity divergence).
- **Paging helper**: `mt5_connect.fetch_rates_paged(symbol, tf, start, end, chunk_bars=80000)` walks backwards from `end`, each page ending at the prior page's oldest timestamp, deduplicates seams, trims to `[start, end]`. Terminal "Max bars in chart" was raised to Unlimited on the trading box, but the code-side per-request cap (~80k) is the real limiter the pager removes; total depth still bounded by broker server history for that timeframe.

## Optimizer symbol scope fix (2026-07-23)
- **Root cause**: the former `DotoOptimizer` timer ran `optimize_params.py --symbols ALL`, and `ALL` resolved to a hardcoded 24-symbol "universe" in `optimize_params.py`/`parallel_optimize.py` `SYMBOL_PROFILE` — including `ADAUSD.raw` and other symbols not in the live portfolio. The optimizer auto-targeted symbols it never trades.
- **Fix**: `ALL` now resolves to `[PORTFOLIO] symbols` from `settings.ini` (the real portfolio) instead of the stale hardcoded list. Self-syncing: changing the portfolio in settings.ini automatically updates what the optimizer targets — no more drift. `optimize.yml` reads the same `[PORTFOLIO] symbols` for its matrix, so CI and live stay in sync.
- **`SYMBOL_PROFILE` trimmed** in both `optimize_params.py` and `parallel_optimize.py`: removed 6 dead symbols with no model and no pool membership (`AUS200`, `BCHJPY`, `ADBE`, `AMGN`, `AVGO`, `AUDPLN`, `UK100`). Kept the portfolio symbols + pool members that have trained models (ETHUSD, LTCUSD, ADAUSD, AVXUSD, forex/index/commodity pool symbols, etc.) so explicit per-symbol optimization still works.
- **Net effect**: the GitHub `optimize.yml` matrix optimizes exactly the portfolio symbols; non-portfolio pool members (e.g. ADAUSD) are only optimized if explicitly passed via `--symbols`.

## Scoring Parity (2026-07-22)
- **`analytics.compute_entry_score()`** — single source of truth for entry scoring. Uses 3-component model: ML (40%), spread (30%), news (30%). Weights from `cfg["scoring_weights"]`.
- **`backtest.py:_compute_entry_score()`** — backtest scoring model. Now uses the same 3-component weights from config. Previously skipped components not in `scores` dict (e.g., news when `ns_enabled=False`), causing score divergence. Fixed: weighting loop now uses `scores.get(key, 0.5)` to include all weighted components with default 0.5.
- **`filters.py:check_ml_gate()`** — live ML gate. Applies news-based confidence adjustment: `news_val >= 0.70` → `confidence_mult * 1.10` (capped 1.5); `news_val <= 0.30` → `confidence_mult * 0.50`. Backtest's `_run_reference()` now applies the same adjustment.
- **`mr_min` parity**: Both paths use `mr_min = 0.03 if entry_atr is None else 0.0` (previously backtest used `entry_type == "mean_reversion"`).
- **Parity tests** in `tests/test_parity.py::TestScoringParity` guard against future divergence.
