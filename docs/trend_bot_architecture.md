# TrendBot — Complete Architecture

> **WARNING:** This document describes an earlier version of the codebase.
> Line numbers, portfolio symbols, and some implementation details reflect
> a prior codebase and may not match the current source. Refer to AGENTS.md
> for the up-to-date architecture overview.

## 0. System Overview

```
Platform: MetaTrader 5 (Demo) via Wine on Linux
Language: Python 3.12
Account:  PKR (Pakistani Rupee) — ~55,000 PKR ≈ ~$197
Portfolio: 10 symbols (Forex + Commodity + Crypto)
Cycle:    Every 10 seconds, scans all 10 symbols
Bot entry: /home/muneeb/doto-mt5-bot/bot/trend_bot.py (11-line shim)
Modules: 12 files in bot/ (see §K), largest is main.py (490 lines)
Config:   /home/muneeb/doto-mt5-bot/config/settings.ini
Credentials: config/credentials.ini OR env vars (MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, DISCORD_WEBHOOK_URL)
```

### 10 Portfolio Symbols & Parameters

**Current (grid-optimized) parameters:**

| Symbol     | Class     | Profile | EMAs   | SL×ATR | RR  | ADX Thr | Risk% | MR? |
|------------|-----------|---------|--------|--------|-----|---------|-------|-----|
| EURUSD.raw | Forex     | A       | 30/90  | 1.0    | 1.5 | 25      | 1.0   | On  |
| EURJPY.raw | Forex     | B       | 10/50  | 1.5    | 2.0 | 28      | 1.0   | On  |
| NZDUSD.raw | Forex     | A       | 30/150 | 1.5    | 2.0 | 28      | 1.0   | On  |
| USDJPY.raw | Forex     | A       | 25/75  | 1.5    | 2.0 | 26      | 1.0   | On  |
| GBPJPY.raw | Forex     | B       | 30/120 | 1.0    | 3.0 | 25      | 1.0   | On  |
| XAU500.raw | Commodity | D       | 30/120 | 1.5    | 5.0 | 25      | 1.0   | Off  |
| US500.raw  | Index     | C       | 10/40  | 1.5    | 2.0 | 18      | 1.0   | On  |
| ETHUSD.raw | Crypto    | C       | 20/60  | 1.5    | 2.0 | 30      | 1.0   | On  |
| DOGUSD.raw | Crypto    | C       | 5/20   | 1.5    | 2.0 | 30      | 1.0   | On  |
| LTCUSD.raw | Crypto    | C       | 5/20   | 1.0    | 3.0 | 25      | 1.0   | On  |

Parameters are per-symbol via `[STRATEGY:SYMBOL]` sections in `settings.ini`.
Current values found via grid search (`optimize_params.py`, ~160 combos per symbol).
Profile classification from external optimization playbook (see §M below).

**Recommended (playbook) parameters for comparison:**

| Symbol     | Profile | Playbook EMAs | Playbook ADX | Playbook SL | Playbook MR |
|------------|---------|---------------|--------------|-------------|-------------|
| EURUSD.raw | A       | 12/48         | < 22         | 1.5–2.0×    | Active      |
| NZDUSD.raw | A       | 12/48         | < 22         | 1.5–2.0×    | Active      |
| USDJPY.raw | A       | 12/48         | < 22         | 1.5–2.0×    | Active      |
| GBPJPY.raw | B       | 8/34          | > 25         | 1.5×        | **Disabled**|
| EURJPY.raw | B       | 8/34          | > 25         | 1.5×        | **Disabled**|
| US500.raw  | C       | 15/60         | > 24         | 1.5×        | **Disabled**|
| ETHUSD.raw | C       | 15/60         | > 24         | 1.5×        | **Disabled**|
| DOGUSD.raw | C       | 15/60         | > 24         | 1.5×        | **Disabled**|
| LTCUSD.raw | C       | 15/60         | > 24         | 1.5×        | **Disabled**|
| XAU500.raw | D       | 12/48         | > 25         | 1.5–2.0×    | **Disabled**|

### Optimization Profiles (from external playbook)

Symbols grouped by market behavior to avoid one-size-fits-all parameters:

| Profile | Name | Symbols | Behavior | Strategy |
|---------|------|---------|----------|----------|
| **A** | Slow Mean-Reverters | EURUSD, NZDUSD, USDJPY | 70%+ time in overlapping ranges | Fast EMAs (12/48), ADX < 22 filter, tight R:R ~1.5, MR active with RSI 30/70 |
| **B** | High-Beta Yen Crosses | GBPJPY, EURJPY | Massive swings, dangerous whipsaws | Aggressive EMAs (8/34), ADX > 25 trend filter, MR **disabled**, trailing stops |
| **C** | Macro Trends | US500, ETHUSD, LTCUSD, DOGUSD | Structural bias + momentum persistence | Slower EMAs (15/60), ADX > 24, MR **disabled**, long-only above 200EMA |
| **D** | Precious Metals / Commodities | XAU500 | Extreme daily ranges, liquidity sweeps, explosive trends | Stable EMAs (12/48), ADX > 25, wide SL (1.5–2.0× ATR), MR **disabled**, equity gate ≥ 418,000 PKR |

---

## A. Startup Sequence (`main()`)

### A1. `load_config()`

Reads `settings.ini` + `credentials.ini` via `configparser`. Builds a single `cfg` dict:

- **Global defaults** from sections: `[TRADING]`, `[STRATEGY]`, `[ADX]`, `[ML_SIGNAL]`, `[DYNAMIC_RISK]`, `[SCORING]`, `[VOLATILITY_FILTER]`, `[TREND_ENTRY]`, `[EXECUTION]`, `[VOLUME_FILTER]`, `[SPREAD_FILTER]`, `[NEWS_SENTIMENT]`, `[TAPE_READING]`, `[TAIL_RISK]`, `[CHANDELIER]`, `[SCALE_OUT]`, `[CORRELATION]`, `[PORTFOLIO]`, `[SESSION]`, `[WATCHDOG]`, `[EXECUTION_QUALITY]`, `[FINE_ENTRY]`, `[MEAN_REVERSION]`
- **Per-symbol overrides**: `[STRATEGY:ETHUSD.raw]`, `[STRATEGY:EURUSD.raw]`, etc. — stored in `cfg["symbol_strategy"]` dict
- **Scale-out overrides**: `[SCALE_OUT:ETHUSD.raw]`
- **Chandelier overrides**: `[CHANDELIER:SOMESYM]`
- **Credentials**: account number, password, server, Discord webhook URL (from `credentials.ini` or env vars `MT5_ACCOUNT` / `MT5_PASSWORD` / `MT5_SERVER` / `DISCORD_WEBHOOK_URL` — env var takes priority)

**Symbol override keys** (`SYMBOL_OVERRIDE_KEYS`):
```python
{"ema_fast", "ema_slow", "atr_sl_mult", "rr", "risk_percent", "atr_period",
 "atr_sma_period", "max_positions_per_symbol", "adx_trend_threshold",
 "adx_range_threshold", "kelly_fraction", "max_risk_ratio",
 "volatility_min_ratio", "deviation", "min_equity"}
```

Applied at runtime per-symbol via `apply_symbol_strategy()` + `apply_symbol_overrides()`.

### A2. MT5 Connection

```python
mt5.initialize(path=cfg["mt5_path"], timeout=cfg["timeout_ms"])
mt5.login(account, password, server)
```

- **Retry logic** (`ensure_mt5_connected()`): kills stale MT5 processes via `pkill`, re-launches via Wine, retries up to 5 times with exponential backoff (5s, 15s, 25s...)
- Registers all 10 symbols via `mt5.symbol_select(symbol, True)`

### A3. Initialization

- **`load_ml_models(cfg)`**: Loads all 10 individual models + 2 pool models from `.pkl` files
- **`load_bot_state()`**: Restores `_scale_out_state`, `_chandelier_state`, `_exec_bias`, `_last_trade_time`, `_tail_risk_triggered`, `_circuit_breaker_triggered` from `bot_state.json`
- **TP hotfix**: Reconstructs scale-out targets for any existing open positions (in case bot restarted mid-trade)
- **Signal handlers**: `SIGTERM` / `SIGINT` trigger clean shutdown (save state → MT5 shutdown → exit)
- **Discord notification**: `bot_start()` webhook with symbol list and balance

---

## B. Main Loop (every 10 seconds)

```python
while True:
   1. Watchdog check (cycle timeout > 180s → warn, 3x → reconnect, 5x → exit)
   2. Clear rate cache (`_rate_cache.clear()`) — forces fresh MT5 data fetch
   3. MT5 connectivity check + reconnect if needed
   4. TP hotfix (runs once after startup)
   5. Daily loss check (realized PnL ≤ -5% → sleep until next day)
   6. Fetch all positions via `mt5.positions_get()`
   7. Reconcile trade journal + cleanup stale scale-out/chandelier state
   8. Circuit breaker check (>15% DD → loop forever, manual restart required)
   9. Refresh correlation cache (_corr_cache) via compute_correlation_matrix() — hourly
   10. FOR EACH SYMBOL:
        9a. Market open check
        9b. Apply per-symbol strategy overrides
        9c. ADX fetch + regime detection
        9d. Capital-based exclusion: skip if equity < per-symbol min_equity
        9e. Signal generation (crossover / pullback / MR)
        9f. EXITS: chandelier trailing + scale-out partials + reversal/MR exit
        9g. ENTRY: filter chain → position sizing → MT5 order
  10. Write `dashboard_state.json`
  11. Daily summary (Discord webhook)
  12. Save `bot_state.json`
  13. Sleep 10 seconds
```

---

## C. Per-Symbol Loop (lines 2314–2545)

### C1. Strategy Application (2320–2322)

```python
cfg["symbol"] = symbol
apply_symbol_strategy(cfg, symbol)   # merges [STRATEGY:SOMESYM] overrides into cfg
apply_symbol_overrides(cfg, symbol)  # merges [SCALE_OUT:SOMESYM] + [CHANDELIER:SOMESYM]
```

Each symbol inherits global defaults, then gets its specific EMAs, SL×ATR, RR, ADX thresholds, risk%, max positions.

### C2. Market Open Check (2314–2317)

```python
def market_open(symbol):
    if symbol == "ETHUSD.raw": return True  # crypto = 24/7
    sinfo = mt5.symbol_info(symbol)
    return sinfo.trade_mode in (FULL, CLOSEONLY)
```

`can_trade_symbol()` additionally requires `FULL` mode (exits only during `CLOSEONLY`).

### C3. ADX + Regime Detection (2324–2332)

```python
adx = get_current_adx(cfg)        # fetch H1 bars, compute single ADX value
new_regime = detect_regime(adx, cfg)
```

#### `detect_regime()` — 5-State Classifier using Multi-TF ADX

```python
def detect_regime(adx_h1, cfg):
    mtf = get_mtf_adx(symbol, period)     # fetch H4 + D1 ADX
    h4_adx, d1_adx = mtf["h4"], mtf["d1"]

    adx_slope = calc_adx_series(df)[-1] - calc_adx_series(df)[-6]  # 5-bar slope

    exhaustion = (adx_h1 >= 40) and (adx_slope < -2)
    if exhaustion:                           return "exhaustion"
    if adx_h1 >= trend_thresh and HTF_trending:  return "strong_trend"
    if adx_h1 >= trend_thresh:                    return "weak_trend"
    if adx_h1 <= range_thresh and not HTF:        return "ranging"
    return "uncertain"
```

| State        | Condition | Entry? | Size Mult |
|--------------|-----------|--------|-----------|
| strong_trend | H1 ADX ≥ trend_thresh AND (H4 or D1 trending) | Yes | 1.0× |
| weak_trend   | H1 ADX ≥ trend_thresh only | Yes | 0.75× |
| ranging      | H1 ADX ≤ range_thresh AND no HTF trending | MR only | 0.5× |
| uncertain    | Timeframes conflict (otherwise) | Yes | 0.5× |
| exhaustion   | ADX ≥ 40 AND slope < -2 | **Skip** | — |

**ADX threshold sources** (in priority order):
1. Per-symbol override `[STRATEGY:SOMESYM] adx_trend_threshold = 28`
2. `[ADX] percentile_enabled = True` → trailing 180-day percentiles (p50 as range, p70 as trend)
3. Static `[ADX] adx_trend_threshold = 25`, `adx_range_threshold = 20`

### C4. Signal Generation (2341–2345)

```python
if regime in ["strong_trend", "weak_trend", "uncertain"]:
    trend_signal, trend_atr, entry_type = get_signal(cfg)

if regime in ["ranging", "uncertain"] and mr_enabled and no positions:
    mr_signal, mr_atr = get_mean_reversion_signal(cfg)
```

#### `get_signal()` — EMA Crossover + Pullback (lines 404–430)

1. Fetch H1 bars (need = ema_slow + atr_period + 50)
2. Compute EMA fast/slow, ATR
3. **Crossover check**: `prev_fast ≤ prev_slow AND cur_fast > cur_slow` → buy signal
4. **Pullback fallback** (disabled via `[TREND_ENTRY] enabled = False`): if no crossover AND pb_enabled, check if price pulled back to within `pb_atr_mult × ATR` of fast EMA
5. Returns `(direction, atr, "crossover"/"pullback")` or `(None, None, None)`

#### `get_mean_reversion_signal()` (lines ~1009–1050)

1. Fetch M30 bars + H1/D1 bars for HTF EMA200 filter
2. Compute RSI(14)
3. **Buy**: RSI < 30 (oversold) AND price < EMA200 (uptrend filter)
4. **Sell**: RSI > 70 (overbought) AND price > EMA200 (downtrend filter)

For `uncertain` regime: crossover is preferred, MR is fallback.

### C5. Exit Management for Open Positions (2352–2418)

Three exit mechanisms execute on every cycle for every open position:

#### C5a. Chandelier Exit (every cycle, ~line 650)

```python
cur_atr = get_current_atr(cfg)
if two_stage and R < 3.0:
    ch_mult = loose_mult (3.5)
elif two_stage and R >= 3.0:
    ch_mult = tight_mult (1.5)
if partial_fired:
    ch_mult = partial_mult (1.5)  # tight after scale-out completes

# Trailing SL
new_sl = hh - cur_atr * ch_mult    # for long
new_sl = max(new_sl, pos.ch_sl)    # forward clamp (never move SL back)
```

- **Entry bar tracking**: `entry_bar = pos.entry_bar` — searches for highest high from entry onward
- **Stop-level clamping**: after computing new SL, multiply by 1, 2, 4, 8, 16 × stops_level to find a valid distance that MT5 accepts
- Hit check: bar `low ≤ SL` for long → close at SL price

#### C5b. Scale-Out (every cycle, ~line 1760)

Stateful via `_scale_out_state[ticket]` dict (persisted across restarts):

```
scale_out_state = {
    "step": 0,                    # 0 = not yet hit, 1+ = partials taken
    "entry_price": ...,           # trade entry price
    "direction": "buy"/"sell",
    "close_fractions": [0.30, 0.30],  # 30% at step 1, 30% at step 2
    "tp_targets_atr": [1.5, 2.5],     # ATR multiples
    "sl_points": 50,
    "point": 0.001,
    "is_mr": False,
    "original_volume": 0.01,
    "atr_entry": ...,             # ATR captured at trade entry
}
```

On each cycle:
1. For each incomplete step, check if price hit `entry + atr_entry × target_atr`
2. On hit: close fraction at target price, update state to next step
3. **After step 0 close**: move SL to entry price (risk-free trade)
4. **After step 1 close**: lock SL at prior ATR target level
5. **After final step**: remove TP completely → remainder runs on chandelier only

#### C5c. Reversal / MR Exit

- **Trend regime** (strong/weak/uncertain): opposite EMA crossover → `REVERSAL` close
- **Ranging regime**: RSI(14) crossing 50 → `MR_EXIT` close
- Close via `mt5.order_send(SELL/BUY, pos.volume, position=ticket, filling=IOC)`
- PnL + pips logged to trade journal, Discord notification sent

---

## D. Entry Filter Chain (2424–2542)

**Two branches** depending on `scoring_enabled`:

### D1. Branch: Scoring Mode (`scoring_enabled = True`)

Skips individual boolean filters. Uses a weighted composite score:

```python
def compute_entry_score(cfg, signal, atr):
    scores = {}
    scores["exec"]    = M15 EMA9/21 alignment score  (fresh=1.0, aligned=0.7, misaligned=0.3)
    scores["volume"]  = min(1.0, rel_vol / kappa)
    scores["volatility"] = min(1.0, cur_atr/atr_sma / min_ratio)
    scores["spread"]  = max(0.0, 1.0 - (spread/atr) / threshold)
    scores["ml"]      = min(1.0, ml_conf / ml_threshold)
    scores["tail_risk"] = 1.0 - (current_dd / max_dd)

    weights = {"exec": 0.15, "volume": 0.10, "volatility": 0.10,
               "spread": 0.10, "ml": 0.25, "tail_risk": 0.10}
    composite = sum(scores[k] * w[k]) / sum(w)
    return composite, scores
```

**Decision**:
- `composite < 0.70` → **skip** (filtered by scoring gate)
- `composite ≥ 0.85` → `confidence_mult = 1.0×` (high conviction, full size)
- `composite ≥ 0.70` → `confidence_mult = 0.85×` (standard edge, reduced size)

**Sizing chain**:
```python
kelly_mult = calc_kelly_mult()
           × calc_volatility_mult()
           × corr_reduction
           × ml_size_mult
           × confidence_mult
```

Then: `final_mult = regime_mult × kelly_mult → place_trade(cfg, signal, atr, final_mult)`

### D2. Branch: Legacy Filter Mode (`scoring_enabled = False`)

Sequential hard boolean gates (backward compatibility for backtest comparisons):

| # | Filter | What It Checks | Config Section |
|---|--------|----------------|----------------|
| 1 | **Execution Signal** | M15 EMA9/21 must align with H1 bias; H1 momentum exhaustion check | `[EXECUTION]` |
| 2 | **Volume** | tick_vol / SMA(20) ≥ 1.2 OR OBV divergence present | `[VOLUME_FILTER]` |
| 3 | **Volatility** | cur_atr / atr_sma ≥ 0.5 (market not "too quiet") | `[VOLATILITY_FILTER]` |
| 4 | **Spread** | spread / ATR ≤ 0.30 | `[SPREAD_FILTER]` |
| 5 | **News Sentiment** | FinBERT score: buy needs score ≥ -0.40, sell needs score ≤ 0.40 | `[NEWS_SENTIMENT]` |
| 6 | **Tape Reading** | M1 bullish pressure: buy blocked if < 0.35 + range > 1.2 | `[TAPE_READING]` |
| 7 | **ML Signal** | XGBoost confidence ≥ threshold (per-symbol override available) | `[ML_SIGNAL]` |
| 8 | **Tail Risk** | No 3σ equity spike, DD < 8% portfolio, < 15% circuit breaker | `[TAIL_RISK]` |

Any filter returning `False` causes `continue` (skip entry, increment `_filter_stats` counter).

---

## E. Position Sizing (`calc_position_size()`, lines 338–373)

```python
risk_amount   = balance × (risk_percent / 100)    # e.g., 1% of 55,000 PKR = 550 PKR
sl_value      = sl_points × tick_value            # PKR value of stop-loss distance
volume        = risk_amount / sl_value            # base position in lots
```

### Multiplier Chain (applied in sequence)

1. **`regime_mult`**: 1.0 (strong_trend) / 0.75 (weak_trend) / 0.5 (uncertain)
2. **`kelly_mult`**: `calc_kelly_mult()` — reads last 50 trades from journal CSV, computes Kelly % = `(W × b - (1-W)) / b`, multiplied by `dr_kelly_fraction (0.25)`, clipped to `[0.25, 1.5]`
3. **`vol_mult`**: `calc_volatility_mult()` — if cur_ATR > 1.2 × SMA50(ATR), reduces to `max(0.25, 1.0 / ratio)`
4. **`corr_reduction`**: Pairwise Pearson correlation with existing positions over 24h, refreshed hourly. Falloff from 1.0 (corr ≤ 0.5) to `1.0 - reduction_max` (corr = 1.0). `[0.50, 1.0]`
5. **`ml_size_mult`**: `clamp(ml_conf / ml_threshold, 0.5, 2.0)` — confidence relative to threshold
6. **`confidence_mult`**: From scoring gate — 1.0× (≥0.85) or 0.85× (≥0.70)

### Safety Checks

| Check | Condition | Action |
|-------|-----------|--------|
| **Min lot cap** | `raw_volume < volume_step` AND `min_lot_risk / risk_amount > max_risk_ratio (2.0)` | Return 0.0 (skip) |
| **Max tail risk** | `volume × sl_value > balance × 1.5%` | Return 0.0 (skip) |
| **Min volume** | `round(vol / vol_step) × vol_step` then clamp to `symbol.volume_min` | Floor |
| **Max volume** | Clamp to `symbol.volume_max` | Ceiling |

If `calc_position_size()` returns 0.0, the caller (`place_trade()` / `place_mean_reversion_trade()`) logs "position sizing returned 0" and does not send the order.

---

## F. Trade Execution

### F1. `place_trade()` (lines 501–640)

1. **Cooldown**: 120s since last trade for this symbol → skip
2. **Zero-volume check**: If `calc_position_size()` returns 0.0 (min lot cap or tail risk triggered), log and skip
3. **SL/TP calculation**:
   ```python
   sl_points = max(int(ATR / (atr_sl_mult × point)), stops_level)
   tp_points = sl_points × rr   # or scale-out targets
   ```
3. **Position sizing**: `volume = calc_position_size(cfg, price, sl_points, regime_mult × kelly_mult)`
4. **Market order**:
   ```python
   req = {
       "action": MT5.TRADE_ACTION_DEAL,
       "symbol": symbol,
       "volume": volume,
       "type": ORDER_TYPE_BUY (or SELL),
       "price": tick.ask (or tick.bid),
       "sl": entry_price - sl_points × point,
       "tp": entry_price + tp_points × point,
       "deviation": get_deviation(cfg, symbol),    # per-symbol override, default 50
       "magic": 20240706,
       "comment": "TrendBot-CROSSOVER" (or -REVERSAL or -MR),
       "type_time": ORDER_TIME_GTC,
       "type_filling": get_filling_mode(symbol),   # from MT5 symbol_info(), fallback IOC
   }
   result = mt5.order_send(req)
   ```
5. **Fallback**: If order fails with `TRADE_RETCODE_INVALID_STOPS`, try entry-only first, then modify SL/TP separately
6. **Post-entry**:
   - Log to `logs/trades.csv` trade journal
   - Discord `trade_open()` webhook with screenshot
   - Initialize `_scale_out_state[ticket]` with entry price, targets, fractions
   - Initialize `_chandelier_state[ticket]` with initial SL

### F2. `place_mean_reversion_trade()` (lines ~1303–1380)

Same structure as `place_trade()` but:
- Uses `mr_sl_atr_mult` and `mr_tp_atr_mult`
- Position size: `calc_position_size(..., mr_position_size_mult (0.5) × kelly_mult)`
- Comment: `TrendBot-MR`

---

## G. ML Pipeline (`train_model.py`)

### G1. Training (offline, Wine Python)

```
MT5 H1 bars (2 years)
  → prepare_features() [67 cols: returns, RSI, EMAs, MACD, ATR, ADX,
     volume, candlestick patterns, session times, multi-TF, cross-asset]
  → inf → NaN → dropna(thresh=62) → NaN → 0
  → triple_barrier_labels(TP=2×ATR, SL=1×ATR, max_hold=20 bars)
  → concat(features, labels) → dropna → drop neutral (y=0)
  → PurgedTimeSeriesSplit(5 folds, gap=20) to prevent lookahead leakage
  → EnsembleModel(XGBoost + LightGBM, logloss objective)
  → threshold optimization (scan 0.30–0.85, maximize EV = TP×RR - FP)
  → save model.pkl with metadata (features, optimal_threshold, f1)
```

### G2. Model Types

| Type | File | Data | Purpose |
|------|------|------|---------|
| Individual | `model_SYMBOL.pkl` | That symbol's bars | Primary |
| Pool | `model_pool_forex.pkl` | 5 forex symbols concatenated | Fallback for weak symbols |
| Pool | `model_pool_crypto.pkl` | 3 crypto symbols | Fallback for weak symbols |
| Meta-labeler | `model_SYMBOL.meta.pkl` | Features + primary_model_correctness | Optional secondary filter |

### G3. Inference (`check_ml_signal()`, lines 1112–1145)

```python
fetch 250 H1 bars
feat_df = prepare_features(df)    # same as training pipeline
feat_df = feat_df.replace([inf, -inf], NaN)  # inf → NaN
check missing features
latest = feat_df[model_features].iloc[-1]     # single latest row
NaN → fill with 0 (safety net, same as training)
prob = model.predict_proba(latest)            # ensemble average
conf = P(win) if buy else 1-P(win)           # directional confidence
if conf < threshold → return (False, conf)   # ML blocked
return (True, conf)                           # ML passed
```

The `conf` value is used for both gating (threshold check) AND sizing (`ml_size_mult = clamp(conf/threshold, 0.5, 2.0)`).

**Pool model fallback**: If `_ml_models[symbol]` is missing, `check_ml_signal()` loads the pool model for the symbol's asset class (forex/crypto).

---

## H. Dashboard

**Stack**: FastAPI + uvicorn + Chart.js + JSON API

### H0. Authentication

Dashboard is secured with HTTP Basic Auth:

```python
DASHBOARD_USER = os.getenv("DASHBOARD_USER", "admin")
DASHBOARD_PASS = os.getenv("DASHBOARD_PASS", "changeme")
```

Set via `Environment=` in systemd service file. All routes require valid credentials.

### H1. State File

`data/dashboard_state.json` — written every cycle (~10s):

```json
{
  "timestamp": "2026-07-10T00:06:05",
  "balance": 55123.45,
  "equity": 55234.56,
  "profit": 111.11,
  "margin": 0,
  "margin_free": 55123.45,
  "positions": 0,
  "equity_history": [{"time": "...", "balance": 55123.45, "equity": 55150.00}, ...],
  "regimes": {"EURUSD.raw": "uncertain", "NZDUSD.raw": "strong_trend", ...},
  "exec_quality": {"EURUSD.raw": {"avg_slippage_pct": 0.001, "rejections": 0, "trades": 3}},
  "positions_detail": [{"symbol": "XAU500.raw", "type": "sell", ...}],
  "filters": {"EURUSD.raw": {"exec": 0, "volume": 0, ...}},
  "correlation": {"EURUSD.raw-EURJPY.raw": 0.623, ...},  # refreshed hourly via _corr_cache
  "health": {"connected": true, "server": "MetaQuotes-Demo"}
}
```

### H2. API Endpoints

- `GET /api/state` → returns dashboard_state.json (requires Basic Auth)
- `GET /api/trades` → returns logs/trades.csv as JSON
- `GET /api/log` → returns last 60 lines of latest bot log

### H3. Dashboard UI (index.html)

Sections displayed:
- Account equity chart (line chart, last 500 data points)
- Position table (symbol, direction, volume, open price, SL, TP, P&L)
- Regime status per symbol (color-coded: green=strong_trend, yellow=weak_trend, blue=ranging, gray=uncertain, red=exhaustion)
- Filter rejection breakdown (bar chart per filter per symbol)
- Correlation matrix (heatmap-style)
- Health status (MT5 connected, server name)

---

## I. Safety & Monitoring Systems

### I1. Hard Limits

| System | Config Key | Trigger | Action |
|--------|------------|---------|--------|
| Max total positions | `max_total_positions=5` | ≥5 positions open | Block all new entries |
| Max per symbol | `max_positions_per_symbol=1` | Already has position | Block new entry for that symbol |
| Daily loss limit | `daily_loss_limit_percent=5.0` | Realized PnL ≤ -5% of balance | Skip entries until next UTC day |
| Tail risk cooldown | `tr_cooldown=60min` | 3σ equity move (sigma over 50 bars) | 60min trading halt |
| Circuit breaker | `cb_dd_pct=15.0` | Portoflio DD ≥ 15% | Halt all trading, manual restart |
| Min lot cap | `max_risk_ratio=2.0` | Min tradeable lot risks > 2× intended | Skip trade |
| Max tail risk per trade | `max_tail_risk_pct=1.5` | Trade risk > 1.5% balance | Skip trade |
| Capital-based exclusion | `min_equity` per-symbol | Equity < threshold for that symbol | Skip entry (e.g., XAU500 requires ≥ 418,000 PKR) |

### I2. Dynamic Sizing Limits

| System | Range | Effect |
|--------|-------|--------|
| Kelly Criterion | `[0.25, 1.5]` | Scales with recent win rate |
| Volatility adjustment | `[0.25, 1.0]` | Reduces size when ATR > 1.2× SMA50 |
| Correlation reduction | `[0.50, 1.0]` | Reduces size when correlated with existing positions |
| ML confidence | `[0.50, 2.0]` | confidence/threshold ratio |
| Scoring confidence | `[0.85, 1.0]` | Based on composite entry score |

### I3. Health Monitoring

| System | What It Checks | Action |
|--------|----------------|--------|
| Cycle watchdog | Each cycle must complete within `max_cycle_seconds (180)` | 3x → reconnect, 5x → exit |
| MT5 connectivity | `mt5.terminal_info().connected` | Restart MT5 via Wine, retry 5× |
| Stale state cleanup | Remove scale-out/chandelier state for closed positions | Every cycle |
| Trade journal | Reconcile open tickets vs journal | Auto-fix on restart |

### M. Small-Account Friction Analysis

On a ~PKR 380,000 account with 1% risk per trade (~PKR 3,800), spread costs create significant friction drag:

| Symbol | Spread + Commission | 1.0× ATR SL (~pips) | Friction Drag | Verdict |
|--------|-------------------|---------------------|---------------|---------|
| EURUSD | ~2.5 pips | ~70 pips | ~3.6% | Tradeable |
| EURJPY | ~4.0 pips | ~70 pips | ~5.8% | Borderline |
| GBPJPY | ~6.0 pips | ~75 pips | ~8.3% | Needs wider SL |

**Implications**:
- M1/M5 optimization is **prohibited** — short SLs increase friction to 25%+ of risk
- Core optimization scope: H1, M30, H4 timeframes only
- GBPJPY and EURJPY require wider stops (1.5× ATR minimum) to prevent spread-stop hunting
- XAU500 at 0.01 lot with 1.5× ATR SL can risk PKR 43,321 (11.4% of account) — an `min_equity`gate of **418,000 PKR** is enforced to prevent unmanaged risk

### I4. Trading Sessions

| Feature | Config | Behavior |
|---------|--------|----------|
| London-only | `trade_only_session = False` | Currently disabled (trades 24h) |
| Asian skip | `skip_asian = False` | Currently disabled |
| Overlap filter | `require_overlap = False` | Currently disabled |

---

## J. Data Files

| File | Purpose | Format |
|------|---------|--------|
| `config/settings.ini` | All configurable parameters | INI |
| `config/credentials.ini` | Account login + Discord URL | INI (env var override: MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER, DISCORD_WEBHOOK_URL) |
| `logs/trades.csv` | Trade journal (all opens/closes) | CSV, 14 cols |
| `logs/bot_YYYYMMDD.log` | Daily log | Plain text |
| `logs/stdout.log` | STDOUT log (systemd) | Plain text |
| `data/bot_state.json` | Persisted state (positions, scale-out, chandelier, bias, circuit breaker) | JSON |
| `data/dashboard_state.json` | Live state for dashboard | JSON |
| `models/model_SYMBOL.pkl` | ML ensemble model per symbol | joblib (pickle) |
| `models/model_pool_FOREX.pkl` | Pooled forex model | joblib (pickle) |
| `models/model_pool_CRYPTO.pkl` | Pooled crypto model | joblib (pickle) |
| `models/model_SYMBOL.meta.pkl` | Meta-labeler per symbol | joblib (pickle) |

---

## K. Module Structure (12 files in bot/)

| Module | Lines | Purpose |
|--------|------|---------|
| `trend_bot.py` | 11 | CLI shim — entry point for Wine Python |
| `main.py` | 490 | Main loop orchestrator, startup, shutdown |
| `state.py` | 60 | All shared mutable globals (singleton) |
| `config.py` | 326 | settings.ini + credentials.ini loading, per-symbol overrides, bot state persistence |
| `mt5_connect.py` | 161 | MT5 wrapper (mt5_call, get_rates, get_filling_mode, get_deviation, ensure_mt5_connected) |
| `regime.py` | 88 | ADX fetching, MTF regime detection (5-state classifier) |
| `signals.py` | 289 | Entry/exit signals (crossover, pullback, MR, execution bias, scoring) |
| `filters.py` | 346 | All 9 filter gates (volume, volatility, spread, tape, news, ML, tail risk, capital, daily loss) |
| `risk.py` | 119 | Position sizing (calc_position_size, Kelly, volatility mult) |
| `execution.py` | 459 | Trade execution (place_trade, place_MR_trade, scale-out, chandelier) |
| `journal.py` | 97 | Trade journal CSV (init, open, close, reconcile) |
| `dashboard.py` | 55 | Dashboard state JSON writer |

### Module Dependency Graph

```
trend_bot.py → main.py
                    ├── state.py (globals)
                    ├── config.py (load, save, overrides)
                    ├── mt5_connect.py → mt5_call, get_rates, ensure_connected
                    ├── regime.py → get_current_adx, detect_regime
                    ├── signals.py → get_signal, MR signal, exec signal, scoring
                    ├── filters.py → all 9 filter gates
                    ├── risk.py → position sizing, Kelly, volatility
                    ├── execution.py → place_trade, chandelier, scale-out
                    ├── journal.py → trade CSV
                    ├── dashboard.py → state JSON
                    ├── correlation.py → portfolio correlation matrix
                    ├── discord_alerts.py → webhook notifications
                    ├── indicators.py → TA math (EMA, ATR, ADX, RSI)
                    └── ml_features.py → 67 feature cols + prepare_features
```

No circular imports. All modules import globals from `state.py`. Functions that rebind globals (`_tail_risk_triggered = True`, etc.) use `state._tail_risk_triggered = True` via `import state as _st`.

## L. External Dependencies

| Library | Purpose |
|---------|---------|
| `MetaTrader5` | MT5 Python API — orders, quotes, account info |
| `pandas` / `numpy` | Data processing, indicators |
| `xgboost` / `lightgbm` | ML ensemble models |
| `joblib` | Model persistence |
| `scikit-learn` | Metrics, cross-validation |
| `configparser` | settings.ini reading |
| `fastapi` + `uvicorn` | Dashboard HTTP server |
| `Chart.js` | Dashboard charts (CDN) |
| `discord_alerts.py` | Discord webhook notifications |
| `ml_features.py` | 67 feature computation functions |
| `correlation.py` | Portfolio correlation matrix |
| `indicators.py` | TA indicators (EMA, ATR, ADX, RSI, MACD) |
