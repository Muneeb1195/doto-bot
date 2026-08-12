# Doto MT5 Trend Bot

Multi-symbol, low-medium risk KAMA/VIDYA crossover trend-following bot for Doto MetaTrader 5 (Wine + Python on Linux), with H4/H1/M15 multi-timeframe (MTF) fusion. Features ML signal overlay, regime-adaptive filtering, chandelier trailing exits, partial TP, cross-asset correlation, and a real-time FastAPI dashboard.

## Quick Start

### 1. Configure credentials
`config/credentials.ini`:
```ini
[LOGIN]
account = YOUR_ACCOUNT_NUMBER
password = YOUR_PASSWORD
server = DOTOGlobal-Real
```

### 2. Start the MT5 stack
```bash
systemctl --user start xvfb-mt5.service mt5.service mt5server.service
```
- Log in with credentials (Wine GUI on the Xvfb display)
- Enable Algo Trading (Ctrl+E)
- Add symbols to Market Watch: `BTCUSD.raw, US30.raw, GBPJPY.raw, SOLUSD.raw, XRPUSD.raw, EURUSD.raw, US500.raw, XAUUSD.raw` (+ anything in `[PORTFOLIO]`)
- The bot manages the connection from here

### 3. Start services
```bash
# Bot (main trading loop)
systemctl --user start doto-bot.service

# Dashboard (FastAPI :8501)
systemctl --user start doto-dashboard.service
```

### 4. (Optional) Enable automated tasks
```bash
systemctl --user enable doto-download.timer    # hourly model/params fetch from GitHub releases
systemctl --user enable doto-orchestrate.timer # monthly train + optimize dispatch (1st of month)
systemctl --user enable doto-backup.timer      # daily tgz archive (04:00)
```

## Key Features (Phases A-E)

| Phase | Feature | Description |
|-------|---------|-------------|
| A | KAMA/VIDYA Trend | Primary entry signal (KAMA for fx/index/metals, VIDYA for crypto) |
| A | Regime Detection | Trending (ADX>25), Ranging (ADX<20), Uncertain |
| A | Chandelier Exit | ATR×2.5 ratchet trailing stop |
| A | Partial TP | 20%/20% at 1.5/2.5 ATR, remaining rides chandelier |
| A | Volume Filter | Tick vol SMA + OBV divergence |
| A | Spread Filter | Max spread/ATR ratio |
| B | ML Overlay | Ensemble (XGB+LGBM), confidence threshold ≥0.60 |
| B | Trend-Pullback Entry | Enters at fast-MA proximity on trend pullbacks |
| B | Multi-TF Fusion | H4 bias gate + H1/H4 agreement + M15 MA crossover entry (≥0.67) |
| B | Kelly Sizing | Half Kelly (kelly_fraction 0.50), vol-adjusted, 0.5×-1.5× |
| C | Walk-Forward Optimizer | Monthly CI auto-opt on GitHub Actions (sharded per symbol) |
| C | Tail Risk Protection | Daily loss limit, 3σ portfolio stop, cooldown |
| C | Mean Reversion | Counter-trend entries in ranging regimes (M30 RSI) |
| D | Cross-Asset Correlation | 24h Pearson correlation, sizing reduction up to 50% |
| D | Auto-Retrain | Monthly CI ML retrain on GitHub Actions (no local timer) |
| D | Tape Reading | M1 OHLC bar proxy for order flow (Wine-compatible) |
| D | Execution Quality | Slippage tracking, rejection monitoring |
| E | Discord Alerts | Trade open/close/partial, daily summary, errors |
| E | Real-Time Dashboard | FastAPI with live metrics, filter breakdown, perf |
| E | Weekly Summary | Auto-generated Discord P&L report (script; no timer) |
| E | Auto-Recovery | State persistence, position recovery on restart |

## Dashboard

FastAPI dashboard at `http://localhost:8501` with:
- Live balance/equity/profit/margin metrics + MT5 health
- Open positions with P&L
- Filter rejection breakdown per symbol (table + chart)
- Performance metrics (WR, PF, avg win/loss, streaks, avg duration)
- Regime status + regime performance breakdown
- Today's trade log with P&L
- Live bot log tail (auto-refresh every 10s)

## Discord Alerts

Configure webhook URL in `config/credentials.ini`:
```ini
[WEBHOOK]
discord_url = https://discord.com/api/webhooks/...
```

Events: trade open, trade close, partial TP, daily summary, errors, weekly summary.

## Configuration (`config/settings.ini`)

### Core
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| TRADING | timeframe | H1 | Chart timeframe |
| TRADING | risk_percent | 1.0 | % of account risked per trade |
| PORTFOLIO | symbols | BTCUSD.raw, US30.raw, GBPJPY.raw, SOLUSD.raw, XRPUSD.raw, EURUSD.raw, US500.raw, XAUUSD.raw | Comma-separated symbol list |
| PORTFOLIO | max_total_positions | 5 | Max concurrent positions (all symbols) |

### Strategy
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| STRATEGY | ma_type | kama | MA type: `kama` (fx/index/metals) or `vidya` (crypto), per-symbol |
| STRATEGY | ema_fast_period | 8 | Fast MA period (per-symbol optimized, e.g. 3-12) |
| STRATEGY | ema_slow_period | 32 | Slow MA period (per-symbol optimized, e.g. 12-48) |
| STRATEGY | risk_reward_ratio | 2.5 | Take-profit = SL × RR (per-symbol, 1.5-3.0) |
| STRATEGY | atr_sl_multiplier | 1.0 | Stop-loss = ATR × mult |
| MTF | enabled | True | H4/H1/M15 multi-TF fusion gate |
| MTF | agreement_threshold | 0.67 | H1/H4 direction-agreement threshold |
| ADX | adx_trend_threshold | 25 | ADX > this = trending |
| ADX | adx_range_threshold | 20 | ADX < this = ranging |

### Risk
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| DYNAMIC_RISK | kelly_fraction | 0.50 | Fraction of Kelly optimal |
| DYNAMIC_RISK | volatility_adjust | True | Reduce size in high vol |
| CORRELATION | enabled | True | Cross-asset sizing reduction |
| CORRELATION | reduction_max | 0.50 | Max size cut for correlated positions |
| TAIL_RISK | max_portfolio_dd_pct | 8.0 | Max intraday drawdown before halt |
| TAIL_RISK | cooldown_minutes | 60 | Wait time after halt trigger |
| SESSION | london_open | 13:00 | London session start (PKT) |
| SESSION | london_close | 22:00 | London session end |

### Filters
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| VOLUME_FILTER | volume_kappa | 1.2 | × SMA threshold |
| SPREAD_FILTER | max_spread_atr_ratio | 0.30 | Max spread relative to ATR |
| ML_SIGNAL | enabled | True | ML overlay |
| ML_SIGNAL | confidence_threshold | 0.60 | Min probability for trade |
| TAPE_READING | enabled | True | M1 bar proxy filter (Wine) |
| FINE_ENTRY | enabled | True | M5 fine-timing entry gate |

### Trailing / Exit
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| CHANDELIER | atr_multiplier | 2.5 | Ratchet trailing tightness |
| SCALE_OUT | enabled | True | 20%/20% partial at 1.5/2.5 ATR |
| MEAN_REVERSION | mr_enabled | True | RSI counter-trend in range |

## Risk Management

- **Per trade**: 1% account risk (half Kelly adjusted, `kelly_fraction = 0.50`)
- **Total exposure**: Max 5 positions (5% account at risk)
- **Daily loss limit**: 5% → auto-halt until next day
- **Tail risk**: 3σ portfolio P&L outlier detection → cooldown
- **Correlation**: Sizing reduced when symbols move together
- **Session**: Skips Asian session (05:00-12:00 PKT); London window configurable (default 13:00-22:00 PKT)
- **Account**: PKR balance (displayed as `Rs.` everywhere)

## Systemd Services

| Service | Function | Schedule |
|---------|----------|----------|
| `mt5.service` | MT5 terminal under Wine | Always on |
| `mt5server.service` | MT5 RPyC bridge (port 18812) | Always on |
| `xvfb-mt5.service` | Virtual X11 display (:99) | Always on |
| `doto-bot.service` | Main trading loop | Always on |
| `doto-dashboard.service` | FastAPI dashboard (:8501) | Always on |
| `doto-news.service` | Marketaux/RSS news polling | Always on |
| `doto-download.service` + `.timer` | Pull latest models/params from GitHub releases | Hourly |
| `doto-orchestrate.service` + `.timer` | Dispatch `train.yml` + `optimize.yml`, pull results back | 1st of month |
| `doto-backup.service` + `.timer` | Daily tgz archive (7-day rotation) | 04:00 |

Optimization and ML retraining have **no systemd timers** — both run only on
GitHub Actions (`train.yml` / `optimize.yml`), dispatched by the home-server's
`doto-orchestrate` timer.

## Optimization

`bot/optimize_params.py` is the **single optimizer entry point** — the former
`parallel_optimize.py` was merged into it, and shared helpers live in
`bot/optimizer_common.py`. Without `--csv` it needs a live MT5 terminal; CI runs
it with `--csv` on pre-exported bars.

```bash
python bot/optimize_params.py --symbols XAUUSD.raw,BTCUSD.raw --two-phase
```

| Flag | Meaning |
|------|---------|
| `--symbols S1,S2` | Comma-separated symbols; default `ALL` = `[PORTFOLIO] symbols` in settings.ini |
| `--years N` | Data window in years (default 5.1) |
| `--two-phase` | **Default.** Phase 1 MA sweep → Phase 2 SL/RR/ADX refinement |
| `--full-grid` | Exhaustive grid search (slow; weekly CI mode) |
| `--cpcv` | Combinatorial Purged Cross-Validation (monthly CI mode) |
| `--quick` | Single-window screening with a reduced grid |
| `--fast` / `--no-fast` | Numba-JIT fast backtest path (default ON when numba is installed) vs pure-Python reference loop |
| `--no-ml` | Disable ML models during optimization |
| `--m1-sim` | M1-bar intra-bar entry simulation |
| `--auto-train` | Train missing ML models before optimizing |
| `--cpcv-paths N` | CPCV path count (default 30) |
| `--csv` | Read pre-exported bars from `data/history/*_<TF>.csv` (no MT5 terminal; CI mode) |
| `--fetch-csv` | Harvest H1/M15/M1 bars from MT5 into `data/history/<SYMBOL>_<TF>.csv`, then exit (cannot combine with `--csv`) |

CSV bars come from `scripts/export_mt5_data.py` (server-side, monthly) or
`optimize_params.py --fetch-csv`.

## Project Structure

```
doto-mt5-bot/
├── AGENTS.md              # AI agent context
├── requirements.txt
├── bot/
│   ├── main.py            # Main trading loop
│   ├── signals.py         # Entry signal generation + scoring
│   ├── filters.py         # Entry filter chain (vol, spread, ML, tape, news, exec)
│   ├── execution.py       # Order placement, scale-out, chandelier exit, MR trades
│   ├── risk.py            # Position sizing, Kelly, volatility mult
│   ├── indicators.py      # EMA, ATR, RSI, ADX (pure pandas)
│   ├── config.py          # INI config loader + per-symbol overrides
│   ├── state.py           # Global mutable shared state
│   ├── mt5_connect.py     # MT5 connection + timeout wrapper
│   ├── dashboard.py       # Dashboard state JSON writer
│   ├── journal.py         # Trade CSV journal
│   ├── correlation.py     # Cross-asset correlation
│   ├── regime.py          # ADX regime classifier
│   ├── discord_alerts.py  # Webhook embeds
│   ├── weekly_summary.py  # Weekly P&L report
│   ├── ml_features.py     # Feature engineering + triple-barrier
│   ├── train_model.py     # ML ensemble training (XGB+LGBM)
│   ├── calibrate_models.py# Probability calibration
│   ├── backtest.py        # Vectorized backtesting engine
│   ├── optimize_params.py # Single optimizer entry point (two-phase default)
│   ├── optimizer_common.py # Shared optimizer helpers (merged from parallel_optimize.py)
│   ├── auto_optimizer.py  # Param-apply helpers (used by download_models.py)
│   ├── mc_ruin.py         # Monte Carlo ruin analysis
│   ├── mc_validation.py   # MC validation
│   ├── scenario_analysis.py # Stress test scenarios
│   ├── backtest_njit.py   # Numba-JIT fast backtest path
├── config/
│   ├── settings.ini       # All trading params
│   └── credentials.ini    # Login + Discord webhook
├── dashboard/
│   └── api.py             # FastAPI dashboard app
├── services/
│   └── news_sentiment.py  # News sentiment (Marketaux + RSS fallback)
├── scripts/
│   ├── download_models.py # GH Actions dispatch + release downloads
│   ├── export_mt5_data.py # MT5 → data/history CSV export (monthly cycle)
│   ├── push_data.py       # Upload H1/M15/M1 CSVs to data-* release
│   ├── deploy-linux.sh    # One-shot server provisioning (systemd units)
│   ├── service-ctl.sh     # Service start/stop/restart/health wrapper
│   ├── plateau_picker.py  # Plateau-pick params from optimize CSVs
│   └── _archive/          # Archived pre-CI tools — see _archive/README.md
├── models/
│   └── model_*.pkl        # Trained ensemble models
├── logs/
│   ├── trades.csv         # Full trade journal
│   └── bot_*.log          # Per-day logs
├── data/
│   ├── bot_state.json     # Persisted trade state (atomic write)
│   └── dashboard_state.json # Dashboard snapshot (atomic write)
```

## Notes

- Live terminal: MetaTrader 5 **build 6101** (verified Aug 2026); Python talks to it via **`mt5linux` 0.1.10** (RPyC bridge) — no native MetaTrader5 package on Linux
- MT5 init takes 100+ seconds under Wine (180s timeout configured)
- Account is PKR (Rs.), not USD — no exchange rate conversion
- Broker reports no stops level (all symbols `trade_stops_level = 0`); `_min_stop_points` in `execution.py` enforces **spread + 10 pts** as the stop floor (e.g. ~12 pts EURUSD, ~2250 pts SOLUSD)
- Tape reading uses M1 bar proxy (tick stream unavailable under Wine)
- State file (`bot_state.json`) enables recovery after restart

### Backtest vs Live Execution Gap
- The backtester (`backtest.py`) uses the **trend-following KAMA/VIDYA crossover** and **mean-reversion** signals only. The **execution-signal gating** logic in `signals.py` (entry/exit timing based on M1 tape + execution bias) is **not simulated** in backtest — entry/exit is assumed at signal bar close.
- Position sizing in backtest uses the per-cycle `calc_position_size` path but does not apply the execution-bias multiplier or the `_apply_corr_ml_sizing` correlation adjustment. Live results therefore reflect tighter risk control than backtest.
- M1 entry simulation (`--no-m1-sim` to disable) approximates intrabar fills and is on by default.
