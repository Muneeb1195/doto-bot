"""Shared analytics — the single source of truth for signal/backtest math.

Both the live engine (main.py / signals.py / filters.py) and the backtest
engine (backtest.py / backtest_njit.py) MUST compute these quantities through
the functions in this module so the two code paths can never diverge again.

Every function here operates on the LAST CLOSED bar (the forming bar is
excluded by the caller) — see agent audit B1/M6/H1/M1: mixing closed- and
forming-bar semantics was the root cause of repeated intrabar-flicker and
lookahead regressions.
"""

import numpy as np
import pandas as pd
import state as _st
from _mt5 import mt5
from indicators import (
    calc_adx,
    calc_atr,
    calc_efficiency_ratio,
    calc_fused_regime_score,
    calc_ma,
    calc_ma_slope,
)
from mt5_connect import mt5_call
from regime import get_current_atr
from state import _ml_models


def closed_bars(df):
    """Return every bar except the still-forming last one.

    Every gate / analytics / signal computation must run on CLOSED bars so the
    bot does not act on intrabar repaints or look at data that can still change
    within the current bar (agent audits B1/M6/H1/M1). Use this helper instead
    of ad-hoc ``df.iloc[:-1]`` so the convention is explicit and centralized.
    Returns ``df`` unchanged if it has 0 or 1 rows (nothing to drop).
    """
    if df is None or len(df) <= 1:
        return df
    return df.iloc[:-1]


def apply_news_confidence_mult(confidence_mult, news_val):
    """News-based confidence adjustment, shared by live filters and backtest
    (single source of truth — prevention A1).

    `news_val >= 0.70` -> `confidence_mult * 1.10` (capped 1.5);
    `news_val <= 0.30` -> `confidence_mult * 0.50`; otherwise unchanged.
    """
    if news_val is None:
        return confidence_mult
    if news_val >= 0.70:
        return min(1.5, confidence_mult * 1.10)
    if news_val <= 0.30:
        return confidence_mult * 0.50
    return confidence_mult


def compute_sl_tp_points(atr, sl_mult, tp_mult, point, stops_level):
    """SL/TP distance in broker points, with the stops floor enforced.

    Single source for the sl_points/tp_points math used by live entry
    (execution._place_trade_inner / place_limit_order) and the backtest
    reference loop's trend path. Callers guarantee `point > 0`. The backtest
    MR path derives TP from ATR directly (not from sl_points) and is
    intentionally NOT routed here.
    """
    sl_points = max(int(atr * sl_mult / point), stops_level)
    tp_points = int(sl_points * tp_mult)
    return sl_points, tp_points


def ma_cross_direction(cf, cs, pf, ps):
    """MA crossover direction: +1 bullish, -1 bearish, 0 no cross.

    Shared by live get_signal / get_mtf_fused_signal and the backtest
    _get_mtf_signal / reference crossover path (prevention A1).
    """
    if pf <= ps and cf > cs:
        return 1
    if pf >= ps and cf < cs:
        return -1
    return 0


def pb_volume_pass(vol, trigger_idx, period, threshold):
    """True if volume at trigger_idx is BELOW `threshold * SMA(period)`.

    Low-volume pullback filter. `trigger_idx` is a positive index into `vol`;
    the SMA window is `[trigger_idx - period + 1, trigger_idx]` (the same
    window pandas rolling(window=period) exposes at that index). Insufficient
    history or non-positive SMA -> pass.
    """
    if trigger_idx < period:
        return True
    vol_sma = float(np.mean(vol[trigger_idx - period + 1 : trigger_idx + 1]))
    if vol_sma <= 0:
        return True
    return vol[trigger_idx] < vol_sma * threshold


def pb_structure_pass(low_high, trigger_idx, lookback, direction):
    """True if the trigger bar makes a higher-low (buy) / lower-high (sell).

    `low_high` is the low array for "buy" or the high array for "sell".
    """
    start = trigger_idx - lookback
    if start < 0:
        return True
    if direction == "buy":
        return low_high[trigger_idx] > np.min(low_high[start:trigger_idx])
    return low_high[trigger_idx] < np.max(low_high[start:trigger_idx])


def pullback_decision(trigger_fast, trigger_slow, trigger_price, trigger_high, trigger_low,
                      confirm_close, atr, pb_atr_mult, pb_atr_min_dist, vol_pass_fn, structure_pass_fn, direction):
    """Pullback entry decision for one direction: "buy" / "sell" / None.

    Core of live get_trend_pullback_signal and the backtest _get_pullback_signal.
    `vol_pass_fn` / `structure_pass_fn` are zero-arg callables the caller wires
    to pb_volume_pass / pb_structure_pass (they need array context). They are
    invoked LAZILY, only after the distance check passes, preserving the
    historical access order. The HTF trend block is a caller-level gate (live
    applies it in main.py, the backtest in _get_pullback_signal) and is
    intentionally NOT part of this decision.
    """
    if atr is None or atr <= 0:
        return None
    pullback_dist = atr * pb_atr_mult
    min_pb_dist = atr * pb_atr_min_dist
    if direction == "buy":
        if not (trigger_fast > trigger_slow):
            return None
        dist = abs(trigger_price - trigger_fast)
        if not (min_pb_dist <= dist <= pullback_dist):
            return None
        if not vol_pass_fn():
            return None
        if not structure_pass_fn():
            return None
        if confirm_close <= trigger_high:
            return None
        return "buy"
    if not (trigger_fast < trigger_slow):
        return None
    dist = abs(trigger_price - trigger_fast)
    if not (min_pb_dist <= dist <= pullback_dist):
        return None
    if not vol_pass_fn():
        return None
    if not structure_pass_fn():
        return None
    if confirm_close >= trigger_low:
        return None
    return "sell"


def htf_trend_decision(htf_price, htf_ma_val, slope, signal, misalign_mult=0.5):
    """3-state HTF trend alignment: ("allow", 1.0) / ("soft", misalign_mult) /
    ("block", 0.0).

    Core of live signals.check_htf_trend and the backtest _check_htf_trend.
    """
    if signal == "buy":
        price_ok = htf_price >= htf_ma_val
        slope_ok = slope >= 0
    else:
        price_ok = htf_price <= htf_ma_val
        slope_ok = slope <= 0
    if price_ok and slope_ok:
        return "allow", 1.0
    if (not price_ok) and (not slope_ok):
        return "block", 0.0
    return "soft", misalign_mult


def mr_entry_decision(cur_rsi, cur_price, htf_ema, oversold, overbought, dev=0.0):
    """Mean-reversion entry decision: "buy" / "sell" / None.

    Core of live get_mean_reversion_signal and the backtest
    _get_mean_reversion_signal. A None htf_ema (unavailable) passes the
    deviation check (backtest semantics; live never passes None here).
    """
    if cur_rsi < oversold:
        if htf_ema is None or cur_price > htf_ema * (1.0 - dev):
            return "buy"
    elif cur_rsi > overbought and (htf_ema is None or cur_price < htf_ema * (1.0 + dev)):
        return "sell"
    return None


def mr_exit_decision(prev_rsi, cur_rsi, is_long):
    """Mean-reversion exit decision (RSI mid-line crossover): bool.

    Core of live check_mean_reversion_exit and the backtest
    _check_mean_reversion_exit.
    """
    if is_long and prev_rsi < 50 and cur_rsi >= 50:
        return True
    return bool(not is_long and prev_rsi > 50 and cur_rsi <= 50)


def mtf_fused_decision(h4_bias, neutral_band, h1_cross, m15_cross):
    """MTF bias+cross decision: (direction, entry_type, agreement).

    h4_bias: last closed H4 price minus its EMA. neutral_band: 0.5*ATR band
    (live derives it from H4 ATR, the backtest from the H1 ATR series — the
    band VALUE is computed by the caller). h1_cross / m15_cross are +/-1/0
    from ma_cross_direction; m15_cross None means no usable M15 data (falls
    back to H4+H1 pullback agreement 0.67).

    Core of live get_mtf_fused_signal and the backtest _get_mtf_signal. The
    live caller additionally re-runs the pullback decision on H1 data for the
    pullback branch; the backtest enters directionally — a documented
    orchestration divergence, not a shared-math one.
    """
    if abs(h4_bias) <= neutral_band:
        return None, None, 0.0
    h4_direction = 1 if h4_bias > 0 else -1
    if h1_cross != h4_direction:
        return None, None, 0.0
    direction = "buy" if h1_cross > 0 else "sell"
    if m15_cross is not None:
        if m15_cross == h1_cross:
            return direction, "crossover", 1.0
        if m15_cross != 0:
            return None, None, 0.0
    return direction, "pullback", 0.67


def fused_regime_score(df_closed, cfg):
    """Fused regime score (0-100) on the last CLOSED bar.

    ADX 45% / ER 35% / ATR-normalized MA-slope 20% (matches backtest._precompute
    and indicators.calc_fused_regime_score exactly). `df_closed` must already
    exclude the still-forming bar.

    Returns float score.
    """
    if df_closed is None or len(df_closed) < 3:
        return 0.0
    adx = calc_adx(df_closed, cfg.get("adx_period", 14))
    if adx is None or np.isnan(adx):
        adx = 0.0
    er_period = cfg.get("er_period", 10)
    er = calc_efficiency_ratio(df_closed["close"].values, er_period) if len(df_closed) > er_period + 1 else 0.0
    fast = cfg["ema_fast"]
    ma_type = cfg.get("ma_type", "kama")
    ma_vals = calc_ma(df_closed, fast, ma_type)
    ma_slope = calc_ma_slope(ma_vals, period=1) if ma_vals is not None and len(ma_vals) > 2 else 0.0
    atr = calc_atr(df_closed, cfg["atr_period"])
    if atr is None or np.isnan(atr) or atr <= 0:
        atr = 0.0
    return float(calc_fused_regime_score(adx, er, ma_slope, atr))


def volume_filter_pass(df_closed, signal, cfg):
    """Volume / OBV execution filter on the last CLOSED bar.

    Mirrors the backtest's _check_volume_filter so live and backtest admit the
    same bars. Returns True if the signal passes (sufficient relative volume or
    OBV divergence in the signal direction).
    """
    if not cfg.get("vf_enabled", True) or not cfg.get("volume_filter", True):
        return True
    sma_period = cfg.get("vf_sma_period", 20)
    needed = sma_period + 5
    if df_closed is None or len(df_closed) < needed:
        return True
    vol_sma = df_closed["tick_volume"].rolling(window=sma_period).mean()
    cur_vol = df_closed["tick_volume"].iloc[-1]
    cur_sma = vol_sma.iloc[-1]
    if pd.isna(cur_sma) or cur_sma <= 0:
        return True
    kappa = cfg.get("vf_kappa", cfg.get("volume_kappa", 1.2))
    rel_vol = cur_vol / cur_sma
    if rel_vol >= kappa:
        return True
    if cfg.get("vf_obv_enabled", True):
        lookback = cfg.get("vf_obv_lookback", 20)
        if _obv_divergence(df_closed, signal, lookback):
            return True
    return False


def _obv_divergence(df_closed, signal, lookback=20):
    """Price/OBV divergence test (direction-aware). Shared by live + backtest."""
    if df_closed is None or len(df_closed) < 3:
        return False
    close = df_closed["close"].values
    volume = df_closed["tick_volume"].values
    s = max(0, len(close) - lookback - 1)
    window_close = close[s:]
    window_vol = volume[s:]
    obv = np.zeros(len(window_close))
    for j in range(1, len(window_close)):
        if window_close[j] > window_close[j - 1]:
            obv[j] = obv[j - 1] + window_vol[j]
        elif window_close[j] < window_close[j - 1]:
            obv[j] = obv[j - 1] - window_vol[j]
        else:
            obv[j] = obv[j - 1]
    if signal == "buy":
        low_idx = int(np.argmin(window_close))
        return low_idx > 0 and obv[-1] > obv[low_idx]
    if signal == "sell":
        high_idx = int(np.argmax(window_close))
        return high_idx > 0 and obv[-1] < obv[high_idx]
    return False


def compute_entry_score(cfg, signal, atr, spread=None, ml_conf=None, news_val=None, tail_risk=None):
    """Score an entry signal (higher = better) — the single scoring math for
    both the live engine (filters.check_ml_gate) and the backtest (raw-value
    seam, architecture plan C4-2).

    Raw-value inputs let the backtest pass precomputed per-bar values instead
    of reimplementing the scoring math; None resolves like the live path:
      - spread:  price units (e.g. 0.0001 for EURUSD); None -> fetch live tick
      - ml_conf: already-scored ML confidence (0 = reject, 0.5-2.0 ratio scored
                 as min(1.0, x)); None -> resolve via filters.check_ml_signal
      - news_val: sentiment score; None -> resolve from the news cache
      - tail_risk: stateful drawdown score (backtest only); None -> omitted
    Returns (score, scores, ml_conf).
    """
    scores = {}
    symbol = cfg["symbol"]
    if ml_conf is not None:
        # Backtest path: caller supplies the already-scored confidence ratio.
        scores["ml"] = min(1.0, max(0.0, ml_conf)) if ml_conf > 0 else 0.0
    elif cfg.get("ml_enabled", True):
        from filters import check_ml_signal

        ml_pass, ml_conf = check_ml_signal(cfg, signal)
        if not ml_pass:
            scores["ml"] = 0.0
        elif ml_conf is not None and not np.isnan(ml_conf):
            model_entry = _ml_models.get(symbol)
            model_type = model_entry.get("metadata", {}).get("model_type", "ensemble") if model_entry else "ensemble"
            if model_type == "regressor":
                max_r = cfg.get("ml_max_r", 2.0)
                scores["ml"] = min(1.0, max(0.0, ml_conf / max(max_r, 0.01)))
            else:
                opt_threshold = model_entry.get("metadata", {}).get("optimal_threshold") if model_entry else None
                threshold = cfg["ml_threshold_overrides"].get(symbol)
                if threshold is None:
                    threshold = opt_threshold if opt_threshold is not None else cfg.get("ml_confidence", 0.55)
                if (
                    opt_threshold is not None
                    and threshold == opt_threshold
                    and ml_conf < threshold
                    and ml_conf >= cfg.get("ml_confidence", 0.55)
                ):
                    threshold = cfg.get("ml_confidence", 0.55)
                scores["ml"] = min(1.0, ml_conf / max(threshold, 0.01))
        else:
            scores["ml"] = cfg.get("scoring_ml_fallback", 0.60)
    else:
        scores["ml"] = cfg.get("scoring_ml_fallback", 0.60)
    if cfg.get("spf_enabled", True):
        if spread is None:
            tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
            if tick is not None:
                spread = tick.ask - tick.bid
        if spread is not None and spread > 0:
            atr_val = atr if atr else get_current_atr(cfg)
            if atr_val and atr_val > 0:
                ratio = spread / atr_val
                scores["spread"] = max(0.0, 1.0 - ratio / max(cfg.get("spf_max_ratio", 0.30), 0.01))
            else:
                scores["spread"] = 0.5
        else:
            scores["spread"] = 0.5
    else:
        scores["spread"] = 0.5
    if news_val is not None:
        scores["news"] = news_val
    else:
        scores["news"] = 0.5
        if cfg.get("ns_enabled", True):
            ns_data = _st._ns_cache.get("data")
            if isinstance(ns_data, dict):
                sym_news = ns_data.get("symbols", {}).get(symbol, {})
                if sym_news.get("count", 0) > 0:
                    score = sym_news.get("score", 0.0)
                    news_score = (score + 1.0) / 2.0
                    if signal == "sell":
                        news_score = 1.0 - news_score
                    scores["news"] = news_score
    if tail_risk is not None:
        scores["tail_risk"] = tail_risk
    weights = cfg.get("scoring_weights") or {"ml": 0.40, "spread": 0.30, "news": 0.30}
    if isinstance(weights, str):
        # Backtest params may carry scoring_weights as a raw INI string.
        parsed = {}
        for part in weights.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                parsed[k.strip()] = float(v)
        weights = parsed
    total = 0.0
    wsum = 0.0
    for key, w in weights.items():
        if key in scores and scores[key] is not None:
            total += scores[key] * w
            wsum += w
    if wsum == 0:
        return 1.0, {}, ml_conf
    return total / wsum, scores, ml_conf
