# Doto MT5 Trend Bot

Multi-symbol, low-medium risk EMA50/200 crossover trend-following bot for Doto MetaTrader 5 (Wine + Python on Linux). Features ML signal overlay, regime-adaptive filtering, chandelier trailing exits, partial TP, cross-asset correlation, and a real-time FastAPI dashboard.

## Quick Start

### 1. Configure credentials
`config/credentials.ini`:
```ini
[LOGIN]
account = YOUR_ACCOUNT_NUMBER
password = YOUR_PASSWORD
server = DOTOGlobal-Real
```

### 2. Launch MT5
```bash
./bot/start_mt5.sh
```
- Log in with credentials
- Enable Algo Trading (Ctrl+E)
- Add symbols to Market Watch: `ETHUSD.raw, XAUUSD.raw, XAGUSD.raw, US30.raw` (+ anything in `[PORTFOLIO]`)
- Close the terminal window (bot manages connection)

### 3. Start services
```bash
# Bot (main trading loop)
systemctl --user start doto-bot.service

# Dashboard (FastAPI :8501)
systemctl --user start doto-dashboard.service
```

### 4. (Optional) Enable automated tasks
```bash
systemctl --user enable doto-optimizer.timer  # weekly param optimization (Sun 02:00)
systemctl --user enable doto-retrain.timer    # weekly ML retrain (Sun 03:06)
systemctl --user enable doto-weekly-summary.timer  # weekly Discord report (Mon 03:00)
```

## Key Features (Phases A-E)

| Phase | Feature | Description |
|-------|---------|-------------|
| A | EMA50/200 Trend | Primary entry signal on H1 |
| A | Regime Detection | Trending (ADX>25), Ranging (ADX<20), Uncertain |
| A | Chandelier Exit | ATR×2.5 ratchet trailing stop |
| A | Partial TP | 30%/30% at 1.5/2.5 ATR, remaining rides chandelier |
| A | Volume Filter | Tick vol SMA + OBV divergence |
| A | Spread Filter | Max spread/ATR ratio |
| B | ML Overlay | RandomForest (3yr), confidence threshold ≥0.50 |
| B | Trend-Pullback Entry | Enters at EMA50 proximity on H1 trend pullbacks |
| B | Execution Gate | M15 EMA9/21 fine entry with M5 confirmation |
| B | Kelly Sizing | Quarter Kelly, vol-adjusted, 0.25×-1.5× |
| C | Walk-Forward Optimizer | Weekly auto-opt (SL 0.8-1.5, RR 1.5-3.0, ADX 20-30) |
| C | Tail Risk Protection | Daily loss limit, 3σ portfolio stop, cooldown |
| C | Mean Reversion | Counter-trend entries in ranging regimes (M30 RSI) |
| D | Cross-Asset Correlation | 24h Pearson correlation, sizing reduction up to 50% |
| D | Auto-Retrain | Weekly ML model retrain with systemd timer |
| D | Tape Reading | M1 OHLC bar proxy for order flow (Wine-compatible) |
| D | Execution Quality | Slippage tracking, rejection monitoring |
| E | Discord Alerts | Trade open/close/partial, daily summary, errors |
| E | Real-Time Dashboard | FastAPI with equity curve, filter breakdown, perf |
| E | Weekly Summary | Auto-generated Discord P&L report (Mon 03:00) |
| E | Auto-Recovery | State persistence, position recovery on restart |

## Dashboard

FastAPI dashboard at `http://localhost:8501` with:
- Live balance/equity/profit/metrics
- Open positions with P&L
- Filter rejection breakdown per symbol
- Equity curve + drawdown chart
- Daily P&L bar chart
- Performance metrics (WR, PF, avg win/loss)
- P&L by symbol
- Regime performance breakdown
- Cross-asset correlation heatmap
- Trade log

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
| PORTFOLIO | symbols | ETHUSD.raw, XAUUSD.raw, ... | Comma-separated symbol list |
| PORTFOLIO | max_total_positions | 5 | Max concurrent positions (all symbols) |

### Strategy
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| STRATEGY | ema_fast_period | 50 | Fast EMA |
| STRATEGY | ema_slow_period | 200 | Slow EMA |
| STRATEGY | risk_reward_ratio | 2.0 | Take-profit = SL × RR |
| STRATEGY | atr_sl_multiplier | 1.0 | Stop-loss = ATR × mult |
| STRATEGY | trail_atr_mult | 2.5 | Chandelier ATR multiplier |
| ADX | adx_trend_threshold | 25 | ADX > this = trending |
| ADX | adx_range_threshold | 20 | ADX < this = ranging |

### Risk
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| DYNAMIC_RISK | kelly_fraction | 0.25 | Fraction of Kelly optimal |
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
| ML_SIGNAL | confidence_threshold | 0.50 | Min probability for trade |
| TAPE_READING | enabled | True | M1 bar proxy filter (Wine) |
| EXECUTION | enabled | True | M15 EMA9/21 fine entry gate |

### Trailing / Exit
| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| CHANDELIER | atr_multiplier | 2.5 | Ratchet trailing tightness |
| SCALE_OUT | enabled | True | 30%/30% partial at 1.5/2.5 ATR |
| MEAN_REVERSION | mr_enabled | True | RSI counter-trend in range |

## Risk Management

- **Per trade**: 1% account risk (quarter Kelly adjusted)
- **Total exposure**: Max 5 positions (5% account at risk)
- **Daily loss limit**: 5% → auto-halt until next day
- **Tail risk**: 3σ portfolio P&L outlier detection → cooldown
- **Correlation**: Sizing reduced when symbols move together
- **Session**: Trades only during London session (13:00-22:00 PKT)
- **Account**: PKR balance (displayed as `Rs.` everywhere)

## Systemd Services

| Service | Function | Schedule |
|---------|----------|----------|
| `doto-bot.service` | Main trading loop | Always on |
| `doto-dashboard.service` | FastAPI dashboard (:8501) | Always on |
| `doto-news-sentiment.service` | Marketaux/RSS news polling | Always on |
| `xvfb-doto.service` | Virtual X11 display (:99) | Always on |
| `failure-alert@.service` | Discord alert on any service failure | On failure |
| `doto-optimizer.service` + `.timer` | Walk-forward param optimization | Sun 02:00 |
| `doto-retrain.service` + `.timer` | ML model retraining | Sun 03:06 |
| `doto-weekly-summary.service` + `.timer` | Discord weekly report | Mon 03:00 |
| `doto-backup.service` + `.timer` | Daily tgz archive (7-day rotation) | 04:00 |

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
│   ├── train_model.py     # ML ensemble training (RF+XGB+LGBM)
│   ├── calibrate_models.py# Probability calibration
│   ├── backtest.py        # Vectorized backtesting engine
│   ├── optimize_params.py # Parameter optimization
│   ├── auto_optimizer.py  # Systemd-triggered weekly opt
│   ├── mc_ruin.py         # Monte Carlo ruin analysis
│   ├── mc_validation.py   # MC validation
│   ├── scenario_analysis.py # Stress test scenarios
│   ├── start_mt5.py       # Wine MT5 launcher
│   ├── run_bot.sh         # Bot runner + restart loop
│   ├── health_check.sh    # System health check
│   └── *.sh               # Other shell scripts (run_optimizer, run_retrain, etc.)
├── config/
│   ├── settings.ini       # All trading params
│   └── credentials.ini    # Login + Discord webhook
├── dashboard/
│   └── api.py             # FastAPI dashboard app
├── services/
│   └── news_sentiment.py  # RSS + FinBERT sentiment
├── models/
│   └── model_*.pkl        # Trained ensemble models
├── logs/
│   ├── trades.csv         # Full trade journal
│   └── bot_*.log          # Per-day logs
├── data/
│   ├── bot_state.json     # Persisted trade state (atomic write)
│   └── dashboard_state.json # Dashboard snapshot (atomic write)
└── wine/                  # Wine prefix with MT5 + Python
```

## Notes

- Terminal build 5836, MetaTrader5 package 5.0.5735
- MT5 init takes 100+ seconds under Wine (180s timeout configured)
- Account is PKR (Rs.), not USD — no exchange rate conversion
- Broker enforces minimum stop distance (~5000 pts) — handled by clamp logic
- Tape reading uses M1 bar proxy (tick stream unavailable under Wine)
- State file (`bot_state.json`) enables recovery after restart

### Backtest vs Live Execution Gap
- The backtester (`backtest.py`) uses the **trend-following KAMA/VIDYA crossover** and **mean-reversion** signals only. The **execution-signal gating** logic in `signals.py` (entry/exit timing based on M1 tape + execution bias) is **not simulated** in backtest — entry/exit is assumed at signal bar close.
- Position sizing in backtest uses the per-cycle `calc_position_size` path but does not apply the execution-bias multiplier or the `_apply_corr_ml_sizing` correlation adjustment. Live results therefore reflect tighter risk control than backtest.
- M1 entry simulation (`--no-m1-sim` to disable) approximates intrabar fills and is on by default.
