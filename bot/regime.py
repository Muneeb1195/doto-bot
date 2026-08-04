"""ADX fetching and 5-state regime detection."""

import MetaTrader5 as mt5
import numpy as np
from indicators import calc_adx, calc_adx_series
from mt5_connect import get_rates


def get_current_adx(cfg, closed=False):
    # closed=True drops the still-forming last bar so the value is computed on
    # the last CLOSED bar — matching the backtest, which evaluates each bar i as
    # a closed bar (agent audit M1).
    needed = cfg["adx_period"] * 3 + 10 + (1 if closed else 0)
    tf = cfg["timeframe"]
    df = get_rates(cfg["symbol"], tf, needed)
    if df is None or len(df) < needed:
        return None
    if closed:
        df = df.iloc[:-1]
    return calc_adx(df, cfg["adx_period"])


def get_current_atr(cfg, closed=False):
    needed = cfg["atr_period"] + 5 + (1 if closed else 0)
    df = get_rates(cfg["symbol"], cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        return None
    if closed:
        df = df.iloc[:-1]
    from indicators import calc_atr

    return calc_atr(df, cfg["atr_period"])


def get_mtf_adx(symbol, period=14):
    result = {"h1": None, "h4": None, "d1": None}
    needed = period * 3 + 10
    for name, tf in [("h1", mt5.TIMEFRAME_H1), ("h4", mt5.TIMEFRAME_H4), ("d1", mt5.TIMEFRAME_D1)]:
        df = get_rates(symbol, tf, needed)
        if df is not None and len(df) >= needed:
            result[name] = calc_adx(df, period)
    return result


def compute_adx_percentiles(cfg):
    window_days = cfg.get("adx_percentile_window_days", 180)
    tf_sec = {
        mt5.TIMEFRAME_M1: 60,
        mt5.TIMEFRAME_M5: 300,
        mt5.TIMEFRAME_M15: 900,
        mt5.TIMEFRAME_M30: 1800,
        mt5.TIMEFRAME_H1: 3600,
        mt5.TIMEFRAME_H4: 14400,
        mt5.TIMEFRAME_D1: 86400,
    }
    bar_sec = tf_sec.get(cfg["timeframe"], 3600)
    window_bars = int(window_days * 86400 / max(bar_sec, 60))
    needed = max(window_bars, cfg["adx_period"] * 3 + 10) + 10
    df = get_rates(cfg["symbol"], cfg["timeframe"], needed)
    if df is None or len(df) < max(window_bars, 100):
        return None, None
    adx_arr = calc_adx_series(df, cfg["adx_period"])
    adx_valid = adx_arr[~np.isnan(adx_arr)]
    if len(adx_valid) < 50:
        return None, None
    p50 = float(np.percentile(adx_valid, 50))
    p70 = float(np.percentile(adx_valid, 70))
    return p50, p70


def detect_regime(adx_h1, cfg):
    if adx_h1 is None:
        return "ranging"
    symbol = cfg.get("symbol", "")
    mtf = get_mtf_adx(symbol, cfg["adx_period"])
    h4_adx = mtf.get("h4")
    d1_adx = mtf.get("d1")
    trend_thresh = cfg["adx_trend_threshold"]
    range_thresh = cfg["adx_range_threshold"]
    if cfg.get("adx_percentile_enabled", False):
        p50, p70 = compute_adx_percentiles(cfg)
        if p50 is not None and p70 is not None:
            trend_thresh = p70
            range_thresh = p50
    adx_slope = None
    needed = cfg["adx_period"] * 3 + 10
    df = get_rates(symbol, cfg["timeframe"], needed + 6)
    if df is not None and len(df) >= needed + 6:
        adx_arr = calc_adx_series(df, cfg["adx_period"])
        if adx_arr is not None and len(adx_arr) > 5:
            adx_slope = float(adx_arr[-1]) - float(adx_arr[-6])
    h4_trending = h4_adx is not None and h4_adx >= range_thresh
    d1_trending = d1_adx is not None and d1_adx >= range_thresh
    exhaustion = (
        adx_h1 >= cfg.get("exhaustion_adx_threshold", 40)
        and adx_slope is not None
        and not np.isnan(adx_slope)
        and adx_slope < -cfg.get("exhaustion_slope_threshold", 2)
    )
    if exhaustion:
        return "exhaustion"
    if adx_h1 >= trend_thresh and (h4_trending or d1_trending):
        return "strong_trend"
    if adx_h1 >= trend_thresh:
        return "weak_trend"
    if adx_h1 <= range_thresh and not h4_trending and not d1_trending:
        return "ranging"
    return "uncertain"
