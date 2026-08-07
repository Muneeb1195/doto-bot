"""Shared analytics — the single source of truth for signal/backtest math.

Both the live engine (main.py / signals.py / filters.py) and the backtest
engine (backtest.py / backtest_njit.py) MUST compute these quantities through
the functions in this module so the two code paths can never diverge again.

Every function here operates on the LAST CLOSED bar (the forming bar is
excluded by the caller) — see agent audit B1/M6/H1/M1: mixing closed- and
forming-bar semantics was the root cause of repeated intrabar-flicker and
lookahead regressions.
"""

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
import state as _st
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


def compute_entry_score(cfg, signal, atr, spread=None):
    """Score an entry signal (higher = better). Accepts optional *spread* in
    price units (e.g. 0.0001 for EURUSD). When None, fetches from MT5 live.
    Passing spread explicitly makes the function testable without MT5."""
    scores = {}
    symbol = cfg["symbol"]
    _ml_conf = None
    if cfg.get("ml_enabled", True):
        from filters import check_ml_signal

        ml_pass, _ml_conf = check_ml_signal(cfg, signal)
        if not ml_pass:
            scores["ml"] = 0.0
        elif _ml_conf is not None and not np.isnan(_ml_conf):
            model_entry = _ml_models.get(symbol)
            model_type = model_entry.get("metadata", {}).get("model_type", "ensemble") if model_entry else "ensemble"
            if model_type == "regressor":
                max_r = cfg.get("ml_max_r", 2.0)
                scores["ml"] = min(1.0, max(0.0, _ml_conf / max(max_r, 0.01)))
            else:
                opt_threshold = model_entry.get("metadata", {}).get("optimal_threshold") if model_entry else None
                threshold = cfg["ml_threshold_overrides"].get(symbol)
                if threshold is None:
                    threshold = opt_threshold if opt_threshold is not None else cfg.get("ml_confidence", 0.55)
                if (
                    opt_threshold is not None
                    and threshold == opt_threshold
                    and _ml_conf < threshold
                    and _ml_conf >= cfg.get("ml_confidence", 0.55)
                ):
                    threshold = cfg.get("ml_confidence", 0.55)
                scores["ml"] = min(1.0, _ml_conf / max(threshold, 0.01))
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
    weights = cfg.get("scoring_weights") or {"ml": 0.40, "spread": 0.30, "news": 0.30}
    total = 0.0
    wsum = 0.0
    for key, w in weights.items():
        if key in scores and scores[key] is not None:
            total += scores[key] * w
            wsum += w
    if wsum == 0:
        return 1.0, {}, _ml_conf
    return total / wsum, scores, _ml_conf
