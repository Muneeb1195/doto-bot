import numpy as np
import pandas as pd


def calc_ema(df, period):
    if period < 1 or df is None or len(df) < 1:
        return pd.Series(dtype=float)
    return df["close"].ewm(span=period, adjust=False).mean()


def _kama_sc(er, sc_fast_period=2, sc_slow_period=30):
    fast_sc = 2.0 / (sc_fast_period + 1.0)
    slow_sc = 2.0 / (sc_slow_period + 1.0)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    return sc


def calc_kama(df, er_period, sc_fast_period=2, sc_slow_period=30):
    if er_period < 1 or df is None or len(df) < er_period + 1:
        return (
            pd.Series(index=df.index if df is not None else None, dtype=float).iloc[:0]
            if df is not None and len(df) > 0
            else pd.Series(dtype=float)
        )
    close = df["close"].values
    n = len(close)
    kama = np.full(n, np.nan)
    kama[er_period - 1] = close[er_period - 1]
    for i in range(er_period, n):
        change = abs(close[i] - close[i - er_period])
        volatility = np.sum(np.abs(np.diff(close[i - er_period : i + 1])))
        er = change / volatility if volatility > 0 else 0
        sc = _kama_sc(er, sc_fast_period, sc_slow_period)
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    return pd.Series(kama, index=df.index)


def calc_vidya(df, period, sc_fast_period=2, sc_slow_period=30):
    if period < 1 or df is None or len(df) < period + 1:
        return (
            pd.Series(index=df.index if df is not None else None, dtype=float).iloc[:0]
            if df is not None and len(df) > 0
            else pd.Series(dtype=float)
        )
    close = df["close"].values
    n = len(close)
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    vidya = np.full(n, np.nan)
    vidya[period] = np.mean(close[: period + 1])
    for i in range(period + 1, n):
        sum_g = np.sum(gains[i - period : i])
        sum_l = np.sum(losses[i - period : i])
        denom = sum_g + sum_l
        if denom != 0:
            cmo = abs(100.0 * (sum_g - sum_l) / denom) / 100.0
        else:
            cmo = 0.0
        sc = _kama_sc(cmo, sc_fast_period, sc_slow_period)
        vidya[i] = vidya[i - 1] + sc * (close[i] - vidya[i - 1])
    return pd.Series(vidya, index=df.index)


def calc_ma(df, period, ma_type="kama"):
    if ma_type == "kama":
        return calc_kama(df, period)
    elif ma_type == "vidya":
        return calc_vidya(df, period)
    elif ma_type == "ema":
        return calc_ema(df, period)
    else:
        import logging

        logging.warning(f"Unknown ma_type={ma_type!r}, falling back to EMA")
        return calc_ema(df, period)


def calc_atr(df, period):
    if period < 1 or df is None or len(df) < period + 1:
        return 0.0
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1]),
        ),
    )
    tr = np.nan_to_num(tr, nan=0.0)
    if len(tr) < period:
        return float(np.mean(tr)) if len(tr) > 0 else 0.0
    atr = float(np.mean(tr[:period]))
    for i in range(period, len(tr)):
        atr = (atr * (period - 1) + tr[i]) / period
    return atr


def calc_atr_series(df, period):
    if period < 1 or df is None or len(df) < 1:
        return pd.Series(index=df.index if df is not None else None, dtype=float)
    tr = pd.DataFrame(
        {
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift()).abs(),
            "lc": (df["low"] - df["close"].shift()).abs(),
        }
    ).max(axis=1)
    if period > len(df):
        return tr.rolling(window=len(df)).mean()
    atr = tr.rolling(window=period).mean()
    atr_arr = atr.values.copy()
    for i in range(period, len(atr_arr)):
        if np.isnan(atr_arr[i]):
            continue
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr.values[i]) / period
    return pd.Series(atr_arr, index=df.index)


def calc_rsi(df, period):
    # Canonical Wilder RSI: seed with the simple average of the first `period`
    # gains/losses, then smooth over ALL subsequent bars. (The previous version
    # only used the last `period+1` bars, i.e. a ~15-bar window — agent audit B2.)
    if period < 1 or df is None or len(df) < period + 1:
        return 50.0
    close = df["close"].values
    deltas = np.diff(close)
    if len(deltas) < period:
        return 50.0
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if np.isnan(avg_gain) or np.isnan(avg_loss):
        return 50.0
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _adx_common(df, period):
    if period < 1 or df is None or len(df) < period + 2:
        return np.array([np.nan])
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    up_move = high[1:] - high[:-1]
    down_move = low[:-1] - low[1:]
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    up_mask = (up_move > down_move) & (up_move > 0)
    down_mask = (down_move > up_move) & (down_move > 0)
    plus_dm[1:] = np.where(up_mask, up_move, plus_dm[1:])
    minus_dm[1:] = np.where(down_mask, down_move, minus_dm[1:])
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    tr = np.nan_to_num(tr, nan=0.0)
    tr_full = np.zeros_like(high)
    tr_full[1:] = tr
    # Wilder smoothing: seed with SMA of first `period` values, then
    # recurse with alpha=1/period (agent audit H10).
    if len(tr_full) >= period:
        atr = tr_full.copy()
        atr[period - 1] = np.mean(tr_full[1 : period + 1])
        for i in range(period, len(atr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr_full[i]) / period
        pds = plus_dm.copy()
        pds[period - 1] = np.mean(plus_dm[1 : period + 1])
        for i in range(period, len(pds)):
            pds[i] = (pds[i - 1] * (period - 1) + plus_dm[i]) / period
        mds = minus_dm.copy()
        mds[period - 1] = np.mean(minus_dm[1 : period + 1])
        for i in range(period, len(mds)):
            mds[i] = (mds[i - 1] * (period - 1) + minus_dm[i]) / period
    else:
        atr = tr_full.copy()
        pds = plus_dm.copy()
        mds = minus_dm.copy()
    atr_safe = np.where((atr == 0) | np.isnan(atr), 1e-10, atr)
    plus_di = 100.0 * pds / atr_safe
    minus_di = 100.0 * mds / atr_safe
    di_sum = plus_di + minus_di
    dx = np.zeros_like(di_sum)
    mask = di_sum > 0
    dx[mask] = 100.0 * np.abs(plus_di[mask] - minus_di[mask]) / di_sum[mask]
    # Wilder smoothing (SMA seed + recursion, matching ATR/+DI/-DI pattern).
    adx_arr = np.full_like(dx, np.nan)
    if len(dx) >= period:
        adx_arr[period - 1] = np.mean(dx[:period])
        for i in range(period, len(adx_arr)):
            adx_arr[i] = (adx_arr[i - 1] * (period - 1) + dx[i]) / period
    # Fill early bars with EMA fallback so no NaN gaps remain.
    early = pd.isna(adx_arr)
    if early.any():
        fallback = pd.Series(dx).ewm(alpha=1 / period, adjust=False).mean().values
        adx_arr[early] = fallback[early]
    return adx_arr


def calc_adx(df, period):
    adx = _adx_common(df, period)
    if len(adx) == 0 or np.isnan(adx[-1]):
        return 0.0
    return float(adx[-1])


def calc_adx_series(df, period):
    return _adx_common(df, period)


def rsi_series(df, period):
    if period < 1 or df is None or len(df) < 1:
        return np.array([])
    close = df["close"].values
    deltas = np.diff(close)
    if len(deltas) == 0:
        return np.array([])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    n = len(close)
    rsi_vals = np.full(n, np.nan)
    if len(gains) < period:
        return rsi_vals
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if np.isnan(avg_gain) or np.isnan(avg_loss):
        return rsi_vals
    for i in range(period, n):
        if i == period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi_vals[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi_vals[i] = 100.0 - (100.0 / (1.0 + rs))
    return rsi_vals


def calc_efficiency_ratio(close, period=10):
    if len(close) < period + 1:
        return 0.0
    change = abs(float(close[-1]) - float(close[-period]))
    volatility = float(np.sum(np.abs(np.diff(close[-period:]))))
    return change / volatility if volatility > 0 else 0.0


# Scales the dimensionless (MA price delta / ATR) ratio onto 0-1 for the
# fused-regime slope term. Measured over 3y H1 across all 8 portfolio symbols
# (see calc_fused_regime_score). The backtest twins in backtest.py import this
# constant so live and backtest cannot drift apart.
SLOPE_SCALE = 2.0


def calc_ma_slope(ma_series, period=1):
    """MA slope over *period* bars, in RAW PRICE UNITS.

    Returns the price delta (end - start) so callers can normalize it against
    another price-unit quantity (e.g. ATR in calc_fused_regime_score) and have
    the units cancel. Previously this returned a dimensionless ratio
    ((end-start)/|start|), which the fused-regime score then divided by ATR —
    a ratio over a price. Because the units did not cancel, the slope term
    scaled inversely with instrument price: it collapsed to ~0 on high-priced
    symbols (gold, indices, BTC) and saturated to a free 20/20 on low-priced
    FX pairs. Either way the 20% slope weight carried no real information, and
    the live regime gate stayed shut on ~100% of bars for 7 of 8 symbols.

    Returns 0.0 if insufficient data.
    """
    if len(ma_series) < period + 1:
        return 0.0
    end = ma_series.iloc[-1]
    start = ma_series.iloc[-1 - period]
    return float(end - start)


def calc_fused_regime_score(adx, er, ma_change, atr):
    """Fuse ADX, ER, and ATR-normalized MA slope into a 0-100 regime quality score.

    Weights per AGPro Trend Quality: 45% ADX, 35% ER, 20% slope.
    Higher score = more trend-favorable.

    *ma_change* must be in RAW PRICE UNITS (see calc_ma_slope) so that
    ma_change / atr is dimensionless and the units cancel.

    The SLOPE_SCALE multiplier maps that ratio onto 0-1. It was measured
    empirically over 3y of H1 bars across all 8 portfolio symbols: the
    price_delta/ATR distribution is p50~0.03-0.07, p90~0.29-0.49, p99~0.80-1.12,
    and is remarkably consistent across instruments, so a single global scale
    works. At 2.0 only ~6.5% of bars clip at the cap (vs ~35% at the legacy
    10.0, which destroyed resolution).
    """
    adx_norm = min(1.0, adx / 50.0) if adx > 0 else 0.0
    er_score = min(1.0, er) if not np.isnan(er) else 0.0
    # A non-positive or NaN ATR must contribute NOTHING. The previous
    # max(atr, 1e-10) guard would instead turn a degenerate ATR into a *full*
    # 20/20 slope score, which is backwards.
    if atr > 0 and not np.isnan(atr) and not np.isnan(ma_change):
        slope_norm = min(1.0, abs(ma_change) / atr * SLOPE_SCALE)
    else:
        slope_norm = 0.0
    return 100.0 * (0.45 * adx_norm + 0.35 * er_score + 0.20 * slope_norm)


__all__ = [
    "calc_ema",
    "calc_atr",
    "calc_atr_series",
    "calc_rsi",
    "calc_adx",
    "calc_adx_series",
    "rsi_series",
    "calc_kama",
    "calc_vidya",
    "calc_ma",
    "calc_efficiency_ratio",
    "calc_ma_slope",
    "calc_fused_regime_score",
]
