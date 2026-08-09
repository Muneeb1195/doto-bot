import logging
import time

import numpy as np
import pandas as pd

_cross_data_cache: dict = {}
_cross_data_time: float = 0
_cross_data_ttl = 300  # 5 minutes

# Pruned from 63 → 44 features (removed bottom-importance + cross-asset + redundant session)
# Based on XGBoost importance across 5 portfolio models (Jul 2026)
# EMA ratios replaced with KAMA ratios in Jul 2026 refactor (KAMA for trend, VIDYA for crypto)
FEATURE_COLS = [
    "ret_5",
    "ret_20",
    "rsi_14",
    "kama9_ratio",
    "kama21_ratio",
    "kama50_ratio",
    "kama9_21_diff",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_ratio",
    "bb_width",
    "volatility_20",
    "vol_ratio",
    "obv_slope_5",
    "skew_20",
    "kurt_20",
    "autocorr_5",
    "ret_5_x_vol_20",
    "adx_14",
    "di_plus_14",
    "di_minus_14",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "vwap_dist",
    "kama200_ratio",
    "atr_ratio_5_14",
    "ret_50",
    "atr_ratio_14_50",
    "kama100_ratio",
    "hl_range_ratio",
    "adx_slope_5",
    "kama_slope_5",
    "volume_rank_20",
    "fd_close_d03",
    "d1_kama_ratio",
    "d1_adx",
    "d1_ret_5pct",
    "d1_ret_20pct",
    "h4_adx",
    "is_london",
    "atr_sma_ratio",
    "dist_swing_high_20",
    "dist_swing_low_20",
    "of_cum_delta",
    "of_delta_ratio",
    "of_tick_imb",
    "of_avg_spread",
    "of_buy_ratio",
    "news_score",
]


def compute_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def compute_kama_series(arr, er_period, sc_fast_period=2, sc_slow_period=30):
    n = len(arr)
    kama = np.full(n, np.nan)
    if n < er_period + 1:
        return kama
    fast_sc = 2.0 / (sc_fast_period + 1.0)
    slow_sc = 2.0 / (sc_slow_period + 1.0)
    kama[er_period - 1] = arr[er_period - 1]
    for i in range(er_period, n):
        change = abs(arr[i] - arr[i - er_period])
        volatility = np.sum(np.abs(np.diff(arr[i - er_period : i + 1])))
        er = change / volatility if volatility > 0 else 0
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (arr[i] - kama[i - 1])
    return kama


def compute_ema_wilder(arr, period):
    return pd.Series(arr).ewm(alpha=1 / period, adjust=False).mean().values


def compute_rsi(arr, period=14):
    arr = np.asarray(arr, dtype=float)
    deltas = np.diff(arr)
    if len(deltas) < period:
        return np.full(len(arr), 50.0)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    rsi = np.full(len(arr), np.nan)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(arr)):
        if i == period:
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
        else:
            avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    return np.nan_to_num(rsi, nan=50.0)


def compute_atr(df, period=14):
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    tr = np.nan_to_num(tr, nan=0.0)
    if len(tr) < period:
        return np.concatenate(
            [np.full(len(high) - len(tr), np.nan), np.full(len(tr), float(np.mean(tr)) if len(tr) > 0 else 0.0)]
        )
    atr = np.full(len(high), np.nan)
    atr[period] = float(np.mean(tr[:period]))
    for i in range(period + 1, len(high)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i - 1]) / period
    return np.nan_to_num(atr, nan=0.0)


def compute_adx(df, period=14):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    n = len(close)

    up_move = np.diff(high, prepend=high[0])
    down_move = np.diff(low, prepend=low[0]) * -1

    dm_plus = np.zeros(n)
    dm_minus = np.zeros(n)
    for i in range(1, n):
        dm_plus[i] = up_move[i] if up_move[i] > down_move[i] and up_move[i] > 0 else 0
        dm_minus[i] = down_move[i] if down_move[i] > up_move[i] and down_move[i] > 0 else 0

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))

    tr_s = compute_ema_wilder(tr, period)
    dm_plus_s = compute_ema_wilder(dm_plus, period)
    dm_minus_s = compute_ema_wilder(dm_minus, period)

    di_plus = 100 * dm_plus_s / np.maximum(tr_s, 1e-10)
    di_minus = 100 * dm_minus_s / np.maximum(tr_s, 1e-10)

    dx = 100 * np.abs(di_plus - di_minus) / np.maximum(di_plus + di_minus, 1e-10)
    adx = compute_ema_wilder(dx, period)

    return adx, di_plus, di_minus


def compute_obv(df):
    close = df["close"].values
    vol = df["tick_volume"].values
    n = len(close)
    obv = np.zeros(n)
    for i in range(1, n):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + vol[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - vol[i]
        else:
            obv[i] = obv[i - 1]
    return obv


def compute_vwap(df):
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    vol = df["tick_volume"].values
    typ_price = (high + low + close) / 3
    cum_pv = np.cumsum(typ_price * vol)
    cum_vol = np.cumsum(vol)
    return cum_pv / np.maximum(cum_vol, 1e-10)


def compute_time_features(df):
    if "time" not in df.columns:
        return np.zeros(len(df)), np.ones(len(df)), np.zeros(len(df)), np.ones(len(df))
    try:
        dt_idx = pd.to_datetime(df["time"])
        hour = dt_idx.dt.hour
        dow = dt_idx.dt.dayofweek
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        day_sin = np.sin(2 * np.pi * dow / 7)
        day_cos = np.cos(2 * np.pi * dow / 7)
        return hour_sin.values, hour_cos.values, day_sin.values, day_cos.values
    except Exception:
        logging.debug("compute_time_features failed", exc_info=True)
        return np.zeros(len(df)), np.ones(len(df)), np.zeros(len(df)), np.ones(len(df))


def fractional_diff(series, d, window=None):
    if window is None:
        window = min(100, len(series) // 2)
    window = max(window, 10)
    w = [1.0]
    for k in range(1, window):
        w_k = -w[-1] * (d - k + 1) / k
        w.append(w_k)
    w = np.array(w)
    result = np.full(len(series), np.nan)
    for i in range(len(w) - 1, len(series)):
        result[i] = np.dot(w, series[i - len(w) + 1 : i + 1])
    return result


def _orderflow_from_m1(m1_df, h1_times):
    """Aggregate orderflow features per H1 bar from M1 OHLCV+spread bars.

    Used for historical backfill (training/backtest) where live tick data is
    unavailable. Mirrors compute_orderflow_features' semantics as closely as
    possible using M1 bar close-direction as a proxy for tick-level buy/sell
    pressure: an M1 bar whose close > open is treated as "buy" volume, close <
    open as "sell" volume (ties split evenly). This removes the train/serve skew
    where historical of_* columns were always NaN -> 0.0.

    Returns a dict of numpy arrays aligned to h1_times (NaN where no M1 data).
    """
    if m1_df is None or len(m1_df) == 0 or "time" not in m1_df.columns:
        return None
    try:
        m1 = m1_df.copy()
        if not isinstance(m1["time"].iloc[0], pd.Timestamp):
            m1["time"] = pd.to_datetime(m1["time"])
        m1 = m1.sort_values("time").set_index("time")
        # Bucket each M1 bar to its parent H1 timestamp.
        h1_idx = m1.index.floor("1h")
        if "tick_volume" in m1:
            vol = m1["tick_volume"].values
        else:
            vol = m1.get("real_volume", pd.Series(0.0, index=m1.index)).values
        close = m1["close"].values
        open_ = m1["open"].values
        spread = m1["spread"].values if "spread" in m1 else np.zeros(len(m1))
        is_buy = close > open_
        is_sell = close < open_
        # Per-bucket sums via pandas groupby on the floored index.
        g = pd.DataFrame(
            {
                "h1": h1_idx,
                "buy_v": np.where(is_buy, vol, 0.0),
                "sell_v": np.where(is_sell, vol, 0.0),
                "n_up": is_buy.astype(float),
                "n_dn": is_sell.astype(float),
                "spread": spread,
            }
        ).groupby("h1")
        agg = g.agg(
            buy_v=("buy_v", "sum"),
            sell_v=("sell_v", "sum"),
            n_up=("n_up", "sum"),
            n_dn=("n_dn", "sum"),
            spread=("spread", "mean"),
            n=("buy_v", "size"),
        )
        out = {c: np.full(len(h1_times), np.nan) for c in (
            "of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio")}
        for j, t in enumerate(h1_times):
            key = pd.Timestamp(t).floor("1h")
            if key not in agg.index:
                continue
            row = agg.loc[key]
            buy_v, sell_v = row["buy_v"], row["sell_v"]
            tv = buy_v + sell_v
            n = int(row["n"])
            if tv <= 0 or n == 0:
                continue
            out["of_cum_delta"][j] = float(buy_v - sell_v)
            out["of_delta_ratio"][j] = float((buy_v - sell_v) / tv)
            out["of_tick_imb"][j] = float((row["n_up"] - row["n_dn"]) / n)
            out["of_avg_spread"][j] = float(row["spread"]) if not np.isnan(row["spread"]) else 0.0
            out["of_buy_ratio"][j] = float(buy_v / tv)
        return out
    except Exception:
        logging.debug("orderflow M1 aggregation failed", exc_info=True)
        return None


def _orderflow_live_from_m1_bars(df, symbol, lookback_hours=3):
    """Live of_* for the socket/RPyC bridge, using M1 BARS instead of ticks.

    The native MetaTrader5 package (and therefore copy_ticks_range) is not
    available on the Linux deployment, and the MQL5 socket EA exposes rates but
    not ticks. Rather than let of_* silently zero-fill at serve time, fetch the
    recent M1 bars over whichever bridge is active and run them through
    _orderflow_from_m1 -- the same aggregation used to build the training
    columns -- so live and training values are produced by identical code.

    Returns a dict of single-element arrays aligned to the last H1 bar, or None.
    """
    try:
        from mt5_connect import get_rates
        from mt5_connect import mt5 as _mt5
    except Exception:
        return None
    if _mt5 is None or not hasattr(_mt5, "TIMEFRAME_M1"):
        return None
    try:
        last_time = df["time"].iloc[-1]
        if not isinstance(last_time, pd.Timestamp):
            return None
        # A few hours of M1 is plenty to cover the forming H1 bar while keeping
        # the bridge round-trip small; _orderflow_from_m1 slices per H1 bucket.
        count = max(120, int(lookback_hours) * 60)
        m1 = get_rates(symbol, _mt5.TIMEFRAME_M1, count)
        if m1 is None or len(m1) == 0:
            return None
        return _orderflow_from_m1(m1, df["time"].values)
    except Exception:
        logging.debug("live M1 orderflow failed for %s", symbol, exc_info=True)
        return None


def compute_orderflow_features(df, symbol, m1_df=None):
    """Compute orderflow features for an H1 frame.

    Historical path: when ``m1_df`` (M1 OHLCV+spread bars) is supplied, every
    H1 bar gets a real of_* vector via _orderflow_from_m1 (closes the
    train/serve skew where historical rows were NaN -> 0.0).

    Live/serving path: when m1_df is None, falls back to the existing
    last-bar-only live-tick fetch (unchanged behaviour for filters.check_ml_gate).
    """
    if df.empty or not symbol or "time" not in df.columns:
        return None
    # Historical aggregation takes precedence when M1 data is available.
    if m1_df is not None and len(m1_df) > 0:
        return _orderflow_from_m1(m1_df, df["time"].values)
    try:
        import MetaTrader5 as mt5
    except ImportError:
        # Linux/socket-bridge deployment: the native MetaTrader5 package does
        # not exist, so the tick path below is unreachable and of_* would be
        # silently zero-filled at serve time while training saw real values.
        # Fall back to fetching M1 BARS over the bridge and reusing the exact
        # same aggregation the training path uses, so there is no skew.
        return _orderflow_live_from_m1_bars(df, symbol)
    try:
        last_time = df["time"].iloc[-1]
        if not isinstance(last_time, pd.Timestamp):
            return None
        bar_start = last_time
        bar_end = bar_start + pd.Timedelta(hours=1)
        now_ts = pd.Timestamp.now()
        if abs((now_ts - bar_end).total_seconds()) > 7200:
            return None
        ticks = mt5.copy_ticks_range(symbol, bar_start.to_pydatetime(), bar_end.to_pydatetime())
        if ticks is None or len(ticks) < 5:
            return None
        last_arr = ticks["last"]
        vol_arr = ticks["volume"]
        diffs = np.diff(last_arr, prepend=last_arr[0])
        buy_v = vol_arr[diffs > 0].sum()
        sell_v = vol_arr[diffs < 0].sum()
        tv = buy_v + sell_v
        return {
            "of_cum_delta": float(buy_v - sell_v),
            "of_delta_ratio": float((buy_v - sell_v) / max(tv, 1)),
            "of_tick_imb": float(((diffs > 0).sum() - (diffs < 0).sum()) / max(len(ticks), 1)),
            "of_avg_spread": float(np.mean(ticks["ask"] - ticks["bid"])),
            "of_buy_ratio": float(buy_v / max(tv, 1)),
        }
    except Exception:
        logging.debug("compute_orderflow_features failed for %s", symbol, exc_info=True)
        return None


def attach_orderflow_features(df, m1_df):
    """Attach historical of_* columns to an H1 frame in place (returns df).

    Optimizer-side helper: aggregate M1 once per symbol and attach the five
    of_* columns to the H1 frame BEFORE it is sliced into walk-forward windows
    and pickled to worker processes. compute_features detects the pre-attached
    columns and skips re-aggregation, so workers never need the raw M1 frame
    for ML feature parity (M1 is still passed separately when --m1-sim needs
    intra-bar entry simulation).

    No-op (returns df unchanged) when m1_df is None/empty or aggregation fails.
    """
    if df is None or m1_df is None or len(m1_df) == 0 or "time" not in df.columns:
        return df
    of_data = _orderflow_from_m1(m1_df, df["time"].values)
    if of_data is None:
        return df
    for col, arr in of_data.items():
        df[col] = arr
    return df


def compute_features(df, symbol=None, m1_df=None):
    if df.empty:
        return df
    out = df.copy()
    close = out["close"].values
    high = out["high"].values
    low = out["low"].values
    opn = out["open"].values
    vol = out["tick_volume"].values

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    ret_1 = np.diff(close, prepend=close[0]) / np.maximum(prev_close, 1e-10)
    ret_1[0] = np.nan
    out["ret_1"] = ret_1

    ret_5 = np.full(len(close), np.nan)
    if len(close) > 5:
        ret_5[5:] = close[5:] / close[:-5] - 1
    out["ret_5"] = ret_5

    ret_20 = np.full(len(close), np.nan)
    if len(close) > 20:
        ret_20[20:] = close[20:] / close[:-20] - 1
    out["ret_20"] = ret_20

    rsi_arr = compute_rsi(close, 14)
    out["rsi_14"] = rsi_arr

    kama9 = compute_kama_series(close, 9)
    kama21 = compute_kama_series(close, 21)
    kama50 = compute_kama_series(close, 50)
    kama200 = compute_kama_series(close, 200)
    out["kama9_ratio"] = close / np.maximum(kama9, 1e-10)
    out["kama21_ratio"] = close / np.maximum(kama21, 1e-10)
    out["kama50_ratio"] = close / np.maximum(kama50, 1e-10)
    out["kama200_ratio"] = close / np.maximum(kama200, 1e-10)
    out["kama9_21_diff"] = (kama9 - kama21) / np.maximum(close, 1e-10)

    kama12 = compute_kama_series(close, 12)
    kama26 = compute_kama_series(close, 26)
    macd_line = kama12 - kama26
    macd_signal = compute_kama_series(macd_line, 9)
    out["macd"] = macd_line / np.maximum(close, 1e-10)
    out["macd_signal"] = macd_signal / np.maximum(close, 1e-10)
    out["macd_hist"] = (macd_line - macd_signal) / np.maximum(close, 1e-10)

    kama_slope = np.full(len(close), np.nan)
    if len(close) > 5:
        kama_slope[5:] = (kama9[5:] - kama9[:-5]) / np.maximum(kama9[:-5], 1e-10)
    out["kama_slope_5"] = kama_slope

    atr_arr = compute_atr(df, 14)
    atr_5 = compute_atr(df, 5)
    out["atr_ratio"] = atr_arr / np.maximum(close, 1e-10)
    out["atr_ratio_5_14"] = atr_5 / np.maximum(atr_arr, 1e-10)

    bb_std = pd.Series(close).rolling(20).std().values
    bb_mid = pd.Series(close).rolling(20).mean().values
    out["bb_width"] = (2 * bb_std) / np.maximum(bb_mid, 1e-10)

    out["volatility_20"] = pd.Series(ret_1).rolling(20).std().values

    vol_sma = pd.Series(vol).rolling(20).mean().values
    out["vol_ratio"] = vol / np.maximum(vol_sma, 1e-10)

    obv_arr = compute_obv(df)
    obv_slope = np.full(len(close), np.nan)
    if len(close) > 5:
        obv_slope[5:] = (obv_arr[5:] - obv_arr[:-5]) / 5
    out["obv_slope_5"] = obv_slope

    adx_vals, di_plus_vals, di_minus_vals = compute_adx(df, 14)
    out["adx_14"] = adx_vals
    out["di_plus_14"] = di_plus_vals
    out["di_minus_14"] = di_minus_vals

    vwap_vals = compute_vwap(df)
    out["vwap_dist"] = (close - vwap_vals) / np.maximum(close, 1e-10)

    hour_sin, hour_cos, day_sin, day_cos = compute_time_features(df)
    out["hour_sin"] = hour_sin
    out["hour_cos"] = hour_cos
    out["day_sin"] = day_sin
    out["day_cos"] = day_cos

    range_px = np.maximum(high - low, 1e-10)
    out["close_position"] = (close - low) / range_px
    body = np.abs(close - opn)
    out["body_ratio"] = body / range_px
    candle_max = np.maximum(close, opn)
    candle_min = np.minimum(close, opn)
    out["upper_wick"] = (high - candle_max) / range_px
    out["lower_wick"] = (candle_min - low) / range_px

    out["skew_20"] = pd.Series(ret_1).rolling(20).skew().values
    kurt_arr = pd.Series(ret_1).rolling(20).kurt().values
    out["kurt_20"] = np.nan_to_num(kurt_arr, nan=0.0)
    out["autocorr_5"] = (
        pd.Series(ret_1)
        .rolling(50)
        .apply(lambda x: x.autocorr(lag=5) if len(x) >= 15 and x.std() > 0 else 0, raw=False)
        .values
    )

    out["ret_5_x_vol_20"] = out["ret_5"] * out["volatility_20"]
    out["vol_ratio_x_ret_1"] = out["vol_ratio"] * out["ret_1"]
    out["ret_1_x_adx"] = out["ret_1"] * out["adx_14"]

    # --- Longer horizon features ---
    ret_50 = np.full(len(close), np.nan)
    if len(close) > 50:
        ret_50[50:] = close[50:] / close[:-50] - 1
    out["ret_50"] = ret_50

    atr_50 = compute_atr(df, 50)
    out["atr_ratio_14_50"] = atr_arr / np.maximum(atr_50, 1e-10)

    kama100 = compute_kama_series(close, 100)
    out["kama100_ratio"] = close / np.maximum(kama100, 1e-10)

    out["hl_range_ratio"] = (high - low) / np.maximum(atr_arr, 1e-10)

    adx_slope = np.full(len(close), np.nan)
    if len(close) > 5:
        adx_slope[5:] = adx_vals[5:] - adx_vals[:-5]
    out["adx_slope_5"] = adx_slope / 5.0

    vol_rank = (
        pd.Series(vol)
        .rolling(20, min_periods=10)
        .apply(lambda x: (x[-1] - x.min()) / max(x.max() - x.min(), 1e-10), raw=True)
        .values
    )
    out["volume_rank_20"] = np.nan_to_num(vol_rank, nan=0.5)

    fd03 = fractional_diff(close, d=0.3)
    fd05 = fractional_diff(close, d=0.5)
    out["fd_close_d03"] = np.nan_to_num(fd03 / np.maximum(close, 1e-10), nan=0.0)
    out["fd_close_d05"] = np.nan_to_num(fd05 / np.maximum(close, 1e-10), nan=0.0)

    # --- Session features (UTC hours) ---
    hour = pd.to_datetime(df["time"]).dt.hour.values if "time" in df.columns else np.zeros(len(close))
    out["is_london"] = ((hour >= 8) & (hour < 17)).astype(float)
    out["is_ny"] = ((hour >= 13) & (hour < 22)).astype(float)
    out["is_overlap"] = ((hour >= 13) & (hour < 17)).astype(float)
    out["is_asian"] = (hour < 9).astype(float)

    # --- Volatility regime (ATR vs its 50-bar SMA) ---
    atr_sma = pd.Series(atr_arr).rolling(50, min_periods=14).mean().values
    out["atr_sma_ratio"] = atr_arr / np.maximum(atr_sma, 1e-10)

    # --- Distance to recent swing high/low (1-bar shift to avoid look-ahead) ---
    hh_20 = pd.Series(high).shift(1).rolling(20, min_periods=5).max().values
    ll_20 = pd.Series(low).shift(1).rolling(20, min_periods=5).min().values
    out["dist_swing_high_20"] = (close - hh_20) / np.maximum(atr_arr, 1e-10)
    out["dist_swing_low_20"] = (close - ll_20) / np.maximum(atr_arr, 1e-10)

    out = add_multi_tf_features(out)
    # Cross-asset features pruned in FEATURE_COLS — skip MT5 fetch bottleneck
    # out = _add_cross_features(df, out)

    # --- Orderflow features ---
    # Precedence: (1) of_* columns already attached to the input df (e.g. via
    # attach_orderflow_features in the optimizers — avoids pickling raw M1 to
    # worker processes); (2) m1_df supplied -> per-bar historical aggregation
    # (closes the train/serve skew where of_* were always NaN -> 0.0);
    # (3) live last-bar tick fetch (serving path).
    of_cols = ["of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio"]
    pre_attached = all(c in out.columns for c in of_cols) and bool(
        out[of_cols].notna().any().any()
    )
    if not pre_attached:
        for col in of_cols:
            out[col] = np.nan
        of_data = compute_orderflow_features(out, symbol, m1_df=m1_df)
        if of_data is not None:
            if m1_df is not None and len(m1_df) > 0:
                # Historical path: of_data is a dict of per-bar arrays.
                for col in of_cols:
                    arr = of_data.get(col)
                    if arr is not None:
                        out[col] = arr
            else:
                # Live path. Two shapes arrive here depending on the bridge:
                #   - native MT5 (copy_ticks_range): scalar per feature;
                #   - Linux socket/RPyC bridge (_orderflow_live_from_m1_bars):
                #     a per-H1-bar ARRAY, because it reuses _orderflow_from_m1.
                # Assigning an array into a single .loc cell raises
                # "Must have equal len keys and value when setting with an
                # iterable" and killed every main-loop cycle on the Linux
                # deployment, so normalise to a scalar last-bar value.
                idx = out.index[-1]
                for k, v in of_data.items():
                    if k not in of_cols:
                        continue
                    arr = np.asarray(v)
                    if arr.ndim == 0:
                        out.loc[idx, k] = float(arr)
                    elif arr.size == len(out):
                        out[k] = arr
                    elif arr.size > 0:
                        out.loc[idx, k] = float(arr[-1])

    # --- News sentiment feature (last bar only, 0.5 neutral for historical) ---
    out["news_score"] = 0.5
    if symbol:
        try:
            from state import _ns_cache

            ns_data = _ns_cache.get("data") if isinstance(_ns_cache, dict) else None
            if isinstance(ns_data, dict):
                sym_news = ns_data.get("symbols", {}).get(symbol, {})
                if sym_news.get("count", 0) > 0:
                    raw_score = sym_news.get("score", 0.0)
                    news_score = (raw_score + 1.0) / 2.0
                    out.loc[out.index[-1], "news_score"] = news_score
        except Exception:
            logging.debug("Failed to attach news_score for %s", symbol)

    return out


def add_multi_tf_features(df):
    if "time" not in df.columns:
        return df
    out = df.copy()
    close = out["close"].values

    # --- Daily features (resample to calendar days) ---
    daily = (
        out.set_index("time")
        .resample("D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )

    d1_close_arr = daily["close"].values
    d1_kama = compute_kama_series(d1_close_arr, 50)
    daily["d1_kama_ratio"] = daily["close"].values / np.maximum(d1_kama, 1e-10)
    d1_adx_arr, _, _ = compute_adx(daily.reset_index(), 14)
    daily["d1_adx"] = d1_adx_arr
    daily["d1_ret_5pct"] = daily["close"].pct_change(5).values
    daily["d1_ret_20pct"] = daily["close"].pct_change(20).values

    daily = daily.shift(1)
    daily.index = daily.index.date
    d1_cols = ["d1_kama_ratio", "d1_adx", "d1_ret_5pct", "d1_ret_20pct"]
    out["_date"] = out["time"].dt.date
    for col in d1_cols:
        out[col] = out["_date"].map(daily[col].to_dict())
    out.drop(columns=["_date"], inplace=True)

    # --- Short-term KAMA(4) on H1 ---
    kama4 = compute_kama_series(close, 4)
    out["kama4_ratio"] = close / np.maximum(kama4, 1e-10)

    h4 = (
        out.set_index("time")
        .resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    h4_adx_arr, _, _ = compute_adx(h4.reset_index(), 14)
    h4["h4_adx"] = h4_adx_arr
    h4 = h4.shift(1)
    out["_h4_key"] = out["time"].dt.floor("4h")
    out["h4_adx"] = out["_h4_key"].map(h4["h4_adx"].to_dict())
    out.drop(columns=["_h4_key"], inplace=True)

    return out


def _fetch_cross_data():
    global _cross_data_cache, _cross_data_time
    now = time.time()
    if now - _cross_data_time < _cross_data_ttl and _cross_data_cache:
        return _cross_data_cache
    try:
        import MetaTrader5 as mt5

        end = pd.Timestamp.now()
        start = end - pd.Timedelta(hours=48)
        cache = {}
        for csym in ["BTCUSD.raw", "XAGUSD.raw"]:
            rates = mt5.copy_rates_range(csym, mt5.TIMEFRAME_H1, start, end)
            if rates is not None and len(rates) > 100:
                cache[csym] = pd.DataFrame(rates)
                cache[csym]["time"] = pd.to_datetime(cache[csym]["time"], unit="s")
        if cache:
            _cross_data_cache = cache
            _cross_data_time = now
        return cache
    except Exception:
        logging.debug("_fetch_cross_data failed", exc_info=True)
        return _cross_data_cache


def _add_cross_features(df, out):
    cross = _fetch_cross_data()
    close = out["close"].values
    n = len(close)

    main_btc_ratio_arr = np.full(n, 1.0)
    btc_ret_arr = np.zeros(n)
    btc_corr_arr = np.zeros(n)
    btc_vol_arr = np.ones(n)
    main_xag_ratio_arr = np.ones(n)
    xag_corr_arr = np.zeros(n)

    btc_df = cross.get("BTCUSD.raw")
    if btc_df is not None and len(btc_df) > 100:
        btc_close = btc_df["close"].values
        btc_time = btc_df["time"].values
        btc_keyed = {(t.date(), t.hour): c for t, c in zip(pd.to_datetime(btc_time), btc_close)}
        btc_aligned = (
            np.array([btc_keyed.get((h.date(), h.hour), np.nan) for h in pd.to_datetime(df["time"])])
            if "time" in df.columns
            else np.full(n, np.nan)
        )

        valid = ~np.isnan(btc_aligned)
        if valid.any():
            main_btc_ratio_arr = np.where(valid, close / np.maximum(btc_aligned, 1e-10), 1.0)
            btc_ret_arr = pd.Series(btc_aligned).pct_change().values
            main_ret = pd.Series(close).pct_change().values
            for i in range(20, n):
                a = main_ret[i - 19 : i + 1]
                b = btc_ret_arr[i - 19 : i + 1]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() >= 10:
                    btc_corr_arr[i] = np.corrcoef(a[mask], b[mask])[0, 1]

            btc_atr = compute_atr(btc_df, 14) if len(btc_df) > 14 else np.full(len(btc_df), np.nan)
            main_atr = compute_atr(df, 14)
            btc_atr_map = dict(zip(pd.to_datetime(btc_time), btc_atr))
            btc_atr_aligned = (
                np.array([btc_atr_map.get(h, np.nan) for h in pd.to_datetime(df["time"])])
                if "time" in df.columns
                else np.full(n, np.nan)
            )
            vol_r = np.where((main_atr > 0) & (btc_atr_aligned > 0), main_atr / np.maximum(btc_atr_aligned, 1e-10), 1.0)
            btc_vol_arr = np.nan_to_num(vol_r, nan=1.0)

    out["main_btc_ratio"] = main_btc_ratio_arr
    out["btc_ret_1"] = btc_ret_arr
    out["main_btc_corr_20"] = btc_corr_arr
    out["btc_vol_ratio"] = btc_vol_arr

    xag_df = cross.get("XAGUSD.raw")
    if xag_df is not None and len(xag_df) > 100:
        xag_close = xag_df["close"].values
        xag_time = xag_df["time"].values
        xag_keyed = {(t.date(), t.hour): c for t, c in zip(pd.to_datetime(xag_time), xag_close)}
        xag_aligned = (
            np.array([xag_keyed.get((h.date(), h.hour), np.nan) for h in pd.to_datetime(df["time"])])
            if "time" in df.columns
            else np.full(n, np.nan)
        )

        valid_xag = ~np.isnan(xag_aligned)
        if valid_xag.any():
            main_xag_ratio_arr = np.where(valid_xag, close / np.maximum(xag_aligned, 1e-10), 1.0)
            xag_ret = pd.Series(xag_aligned).pct_change().values
            main_ret2 = pd.Series(close).pct_change().values
            for i in range(20, n):
                a = main_ret2[i - 19 : i + 1]
                b = xag_ret[i - 19 : i + 1]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() >= 10:
                    xag_corr_arr[i] = np.corrcoef(a[mask], b[mask])[0, 1]

    out["main_xag_ratio"] = main_xag_ratio_arr
    out["main_xag_corr_20"] = xag_corr_arr

    return out


def r_multiple_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold=20):
    """Compute realized R-multiple for each bar (continuous regression target).

    R = realized_pnl / risk, where risk = sl_atr_mult * atr.
    Positive R means profitable, magnitude captures edge size.
    RR = tp_atr_mult / sl_atr_mult (reward-to-risk ratio).
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = compute_atr(df, 14)
    rr = tp_atr_mult / sl_atr_mult if sl_atr_mult > 0 else 2.0

    n = len(close)
    r_vals = np.full(n, np.nan, dtype=np.float32)

    for i in range(n - max_hold):
        entry = close[i]
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        risk = sl_atr_mult * atr[i]
        up_target = entry + atr[i] * tp_atr_mult
        dn_target = entry - atr[i] * sl_atr_mult
        hit = False
        for j in range(1, max_hold + 1):
            hit_up = high[i + j] >= up_target
            hit_dn = low[i + j] <= dn_target
            if hit_up and hit_dn:
                r_vals[i] = rr if j == 1 else (rr if hit_up else -1.0)
                hit = True
                break
            if hit_up:
                r_vals[i] = rr
                hit = True
                break
            if hit_dn:
                r_vals[i] = -1.0
                hit = True
                break
        if not hit:
            r_vals[i] = (close[i + max_hold] - entry) / max(risk, 1e-10)
    return pd.Series(r_vals, index=df.index, name="r_multiple")


def compute_feature_stats(feature_data):
    """Compute per-feature mean and std for drift monitoring.

    Returns dict mapping feature name -> {"mean": float, "std": float}.
    """
    stats = {}
    for col in feature_data.columns:
        col_data = feature_data[col].dropna()
        if len(col_data) > 10:
            stats[col] = {"mean": float(col_data.mean()), "std": float(col_data.std())}
    return stats


def triple_barrier_labels(df, tp_atr_mult=2.0, sl_atr_mult=1.0, max_hold=20):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    atr = compute_atr(df, 14)

    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)

    for i in range(n - max_hold):
        entry = close[i]
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        up_target = entry + atr[i] * tp_atr_mult
        dn_target = entry - atr[i] * sl_atr_mult

        for j in range(1, max_hold + 1):
            hit_up = high[i + j] >= up_target
            hit_dn = low[i + j] <= dn_target
            if hit_up and hit_dn:
                labels[i] = 1
                break
            if hit_up:
                labels[i] = 1
                break
            if hit_dn:
                labels[i] = -1
                break

    return pd.Series(labels, index=df.index, name="label")


def prepare_features(df, symbol=None, m1_df=None):
    feat_df = compute_features(df, symbol=symbol, m1_df=m1_df)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    feature_data = feat_df[FEATURE_COLS].copy()
    of_cols = ["of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio"]
    for col in of_cols:
        if col in feature_data.columns:
            feature_data[col] = feature_data[col].fillna(0.0)
    min_valid = len(FEATURE_COLS) - 5
    feature_data = feature_data.dropna(thresh=min_valid)
    return feature_data, feat_df.loc[feature_data.index]
