import argparse
import configparser
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import joblib

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import numpy as np
import pandas as pd
from analytics import apply_news_confidence_mult, compute_entry_score, volume_filter_pass
from credentials import load_credentials
from indicators import SLOPE_SCALE, calc_adx_series, calc_atr_series, calc_ma

from config import validate_config as _validate_config

sys.path.insert(0, str(Path(__file__).parent.resolve()))

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
LOG_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR.mkdir(exist_ok=True)

# Per-process caches: model data (by path) and precomputed ML multipliers (by df slice).
# Avoids re-loading the model and re-running compute_features for every backtest candidate.
_ML_DATA_CACHE: dict = {}
_ML_MULT_CACHE: dict = {}

# Per-process cache of the shifted M15 fast/slow MAs used by the MTF entry path.
# The full M15 frame is identical for every backtest in an optimization run, so
# the two expensive calc_ma calls (one per M15 fast/slow period) are recomputed
# per combo x window x IS/OOS. Keyed by (frame fingerprint, periods, ma_type).
_MTF_M15_MA_CACHE: dict = {}

# Per-process cache of the primary-frame (H1) MAs. These are recomputed for every
# backtest instance even though only a handful of distinct (period, ma_type)
# pairs occur across a grid run. Keyed by (frame fingerprint, period, ma_type).
_H1_MA_CACHE: dict = {}


def _get_ma_cached(df, period, ma_type):
    """Compute (and cache) calc_ma on a window slice of the primary frame.

    The MAs in Backtest.__init__ depend only on the window content, period and
    ma_type. During grid search the same (window, period, ma_type) recurs many
    times, so cache by frame fingerprint to avoid re-running calc_ma.
    """
    if df is None or len(df) < 2:
        return calc_ma(df, period, ma_type)
    t0 = df["time"].iloc[0]
    t1 = df["time"].iloc[-1]
    key = (len(df), t0, t1, period, ma_type)
    if key not in _H1_MA_CACHE:
        _H1_MA_CACHE[key] = calc_ma(df, period, ma_type)
    return _H1_MA_CACHE[key]


def _get_mtf_m15_mas(df_m15, m15_fast, m15_slow, ma_type):
    """Compute (and cache) the shifted M15 fast/slow MAs for MTF.

    Every backtest in an optimization run receives the SAME full M15 frame, and
    the M15 MA periods derive from the candidate EMA pair, so the two expensive
    calc_ma calls are recomputed for every combo x window x IS/OOS. Cache by
    (frame fingerprint, periods, ma_type); callers only pay the cheap
    reindex(ffill) to their own H1 bar times.
    """
    t0 = df_m15["time"].iloc[0]
    t1 = df_m15["time"].iloc[-1]
    key = (len(df_m15), t0, t1, m15_fast, m15_slow, ma_type)
    if key not in _MTF_M15_MA_CACHE:
        m15 = df_m15.set_index("time").sort_index()
        m15_fast_ma = calc_ma(m15, m15_fast, ma_type)
        m15_slow_ma = calc_ma(m15, m15_slow, ma_type)
        if len(m15_fast_ma) > 1:
            m15_shift = m15_fast_ma.index[1] - m15_fast_ma.index[0]
            m15_fast_ma.index = m15_fast_ma.index + m15_shift
            m15_slow_ma.index = m15_slow_ma.index + m15_shift
        _MTF_M15_MA_CACHE[key] = (m15_fast_ma, m15_slow_ma)
    return _MTF_M15_MA_CACHE[key]

# Numba availability probe (lazy import so the module imports even without numba).
_NJIT_OK = None


def _njit_available():
    global _NJIT_OK
    if _NJIT_OK is None:
        try:
            import numba  # noqa: F401

            _NJIT_OK = True
        except Exception:
            logging.debug("Numba import failed — disabling JIT", exc_info=True)
            _NJIT_OK = False
    return _NJIT_OK


def _ml_cache_key(df):
    try:
        return (df["time"].iloc[0], df["time"].iloc[-1], len(df))
    except Exception:
        logging.debug("ML cache key computation failed", exc_info=True)
        return id(df)


def compute_ml_multipliers(df, params, df_m1=None):
    """Compute per-bar ML confidence multipliers (buy/sell).

    Supports both classifiers (predict_proba) and regressors (predict).
    Window-independent of the candidate's MA fast/slow, so the result is cached
    per unique df slice and reused across all parameter candidates.
    """
    n = len(df)
    buy = np.ones(n, dtype=float)
    sell = np.ones(n, dtype=float)
    if not (params.get("ml_enabled", False) and params.get("ml_model_path")):
        return buy, sell, False
    try:
        path = params["ml_model_path"]
        ckey = _ml_cache_key(df)
        if ckey in _ML_MULT_CACHE:
            return _ML_MULT_CACHE[ckey]
        data = _ML_DATA_CACHE.get(path)
        if data is None:
            from train_model import EnsembleModel, EnsembleRegressor, _CalibratedWrapper

            import __main__

            __main__.EnsembleModel = EnsembleModel
            __main__.EnsembleRegressor = EnsembleRegressor
            __main__._CalibratedWrapper = _CalibratedWrapper
            data = joblib.load(path)
            _ML_DATA_CACHE[path] = data
        model = data["model"]
        features = data["metadata"]["features"]
        model_type = data["metadata"].get("model_type", "ensemble")
        from ml_features import compute_features

        feat_df = compute_features(df, symbol=params.get("symbol", ""), m1_df=df_m1)
        missing = [c for c in features if c not in feat_df.columns]
        if missing:
            _ML_MULT_CACHE[ckey] = (buy, sell, False)
            return buy, sell, False
        X = feat_df[features].fillna(0).values
        if model_type == "regressor":
            pred_r = model.predict(X)
            min_r = params.get("ml_min_r", 0.20)
            max_r = params.get("ml_max_r", 2.0)
            buy = np.clip(pred_r / max(max_r, 0.01), 0.0, 2.0)
            sell = np.clip(-pred_r / max(max_r, 0.01), 0.0, 2.0)
            buy = np.where(pred_r >= min_r, buy, 0.0)
            sell = np.where(-pred_r >= min_r, sell, 0.0)
        else:
            opt_threshold = data["metadata"].get("optimal_threshold")
            tft_member = getattr(model, "tft", None)
            if tft_member is not None:
                seq_len = getattr(tft_member, "seq_len", 20)
                proba = np.zeros((len(X), 2))
                for j in range(len(X)):
                    start = max(0, j - seq_len + 1)
                    window = X[start:j+1]
                    if len(window) < 1:
                        continue
                    p = model.predict_proba(window)
                    proba[j] = p[-1]
            else:
                proba = model.predict_proba(X)
            classes = model.classes_
            default_conf = params.get("ml_threshold_overrides", {}).get(
                params.get("symbol", ""), params.get("ml_confidence", 0.55)
            )
            threshold = opt_threshold or default_conf
            if len(classes) == 2:
                pos_idx = np.where(classes == 1)[0]
                neg_idx = np.where(classes == 0)[0]
                # Sanity check: if the optimal threshold is above the max predicted
                # probability, fall back to the default confidence so the ML gate
                # doesn't block every bar (observed with models whose training
                # optimal_threshold exceeds the probability range on new data).
                if opt_threshold is not None and threshold != default_conf:
                    pct_pass = 0.0
                    if len(pos_idx) > 0:
                        pct_pass = (proba[:, pos_idx[0]] >= threshold).mean()
                    elif len(neg_idx) > 0:
                        pct_pass = (proba[:, neg_idx[0]] >= threshold).mean()
                    if pct_pass < 0.001:
                        threshold = default_conf
                if len(pos_idx) > 0:
                    pos_conf = proba[:, pos_idx[0]]
                    buy = np.where(
                        pos_conf >= threshold,
                        np.clip(pos_conf / threshold, 0.5, 2.0),
                        0.0,
                    )
                if len(neg_idx) > 0:
                    neg_conf = proba[:, neg_idx[0]]
                    sell = np.where(
                        neg_conf >= threshold,
                        np.clip(neg_conf / threshold, 0.5, 2.0),
                        0.0,
                    )
        result = (buy, sell, True)
        _ML_MULT_CACHE[ckey] = result
        return result
    except Exception:
        logging.warning("ML multiplier computation failed", exc_info=True)
        return np.ones(n, dtype=float), np.ones(n, dtype=float), False


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_DIR / "backtest.log"), logging.StreamHandler()],
)


def precompute_fixed(df, params):
    """Precompute indicator arrays that do NOT depend on ema_fast/ema_slow.

    Pass the returned dict to Backtest(fixed=...) so the expensive
    indicator math (ATR, ADX, H4/D1 resample, HTF trend gate, volatility
    filter, day-start index) is done once per window slice and reused
    across every (ema_fast, ema_slow) combo during grid search.

    Saves ~15-25ms per backtest — meaningful during optimization where
    thousands of combos are tested.
    """
    p = params
    n = len(df)
    fixed = {}

    fixed["atr_series"] = calc_atr_series(df, p.get("atr_period", 14))
    fixed["adx_series"] = calc_adx_series(df, p.get("atr_period", 14)) if p.get("adx_enabled", True) else None
    fixed["vol_sma"] = df["tick_volume"].rolling(window=p.get("vf_sma_period", 20)).mean()

    tr = pd.DataFrame(
        {
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift()).abs(),
            "lc": (df["low"] - df["close"].shift()).abs(),
        }
    ).max(axis=1)
    atr_base = tr.ewm(alpha=1.0 / p.get("atr_period", 14), adjust=False).mean()
    fixed["atr_base"] = atr_base

    if p.get("volatility_filter", False):
        fixed["atr_for_sma"] = atr_base
        fixed["atr_sma"] = atr_base.rolling(window=p.get("atr_sma_period", 20)).mean()
    else:
        fixed["atr_for_sma"] = None
        fixed["atr_sma"] = None

    fixed["h4_adx"] = None
    fixed["d1_adx"] = None
    fixed["h4_df"] = None
    fixed["d1_df"] = None
    fixed["h4_adx_aligned"] = None
    fixed["d1_adx_aligned"] = None
    fixed["htf_ema_aligned"] = None
    fixed["htf_slope_aligned"] = None
    fixed["htf_close_aligned"] = None

    if n >= 100:
        _resample = df.set_index("time")
        h4_df = (
            _resample.resample("4h")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"})
            .dropna(subset=["open"])
        )
        fixed["h4_df"] = h4_df
        h4_adx_arr = calc_adx_series(h4_df, p.get("atr_period", 14))
        if h4_adx_arr is not None:
            fixed["h4_adx"] = pd.Series(h4_adx_arr, index=h4_df.index)
        d1_df = _resample.resample("D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
        ).dropna(subset=["open"])
        fixed["d1_df"] = d1_df
        d1_adx_arr = calc_adx_series(d1_df, p.get("atr_period", 14))
        if d1_adx_arr is not None:
            fixed["d1_adx"] = pd.Series(d1_adx_arr, index=d1_df.index)

        htf_slow = p.get("htf_ema_slow", 200)
        if len(h4_df) > htf_slow:
            h4_ma = calc_ma(h4_df, htf_slow, p.get("ma_type", "kama"))
            slope_window = min(12, max(2, len(h4_df) // 10))
            h4_slope = h4_ma.diff(slope_window)
            # Align H4 indicators to H1 bars using the last CLOSED H4 bar
            # (matching the h4_adx_aligned i//4-1 convention). reindex(ffill)
            # would pick the forming (incomplete) H4 bar, introducing up to
            # 4h of lookahead.
            h4_ma_arr = h4_ma.values if hasattr(h4_ma, "values") else np.array(h4_ma)
            h4_slope_arr = h4_slope.values if hasattr(h4_slope, "values") else np.array(h4_slope)
            h4_close_arr = h4_df["close"].values
            fixed["htf_ema_aligned"] = np.array(
                [h4_ma_arr[i // 4 - 1] if i >= 4 and i // 4 - 1 < len(h4_ma_arr) else np.nan for i in range(n)],
                dtype=float,
            )
            fixed["htf_slope_aligned"] = np.array(
                [h4_slope_arr[i // 4 - 1] if i >= 4 and i // 4 - 1 < len(h4_slope_arr) else np.nan for i in range(n)],
                dtype=float,
            )
            fixed["htf_close_aligned"] = np.array(
                [h4_close_arr[i // 4 - 1] if i >= 4 and i // 4 - 1 < len(h4_close_arr) else np.nan for i in range(n)],
                dtype=float,
            )

        if fixed["h4_adx"] is not None and len(fixed["h4_adx"]) > 0:
            fixed["h4_adx_aligned"] = np.array(
                [
                    fixed["h4_adx"].iloc[i // 4 - 1]
                    if i >= 4 and i // 4 - 1 < len(fixed["h4_adx"])
                    else np.nan
                    for i in range(n)
                ],
                dtype=float,
            )
        else:
            fixed["h4_adx_aligned"] = np.full(n, np.nan)
        if fixed["d1_adx"] is not None and len(fixed["d1_adx"]) > 0:
            fixed["d1_adx_aligned"] = np.array(
                [
                    fixed["d1_adx"].iloc[i // 24 - 1]
                    if i >= 24 and i // 24 - 1 < len(fixed["d1_adx"])
                    else np.nan
                    for i in range(n)
                ],
                dtype=float,
            )
        else:
            fixed["d1_adx_aligned"] = np.full(n, np.nan)

    day_start_idx = np.arange(n)
    if n > 0 and hasattr(df["time"], "dt"):
        dates = df["time"].dt.date.values
        for i in range(1, n):
            if dates[i] == dates[i - 1]:
                day_start_idx[i] = day_start_idx[i - 1]
    fixed["day_start_idx"] = day_start_idx

    if p.get("dr_vol_adjust", True):
        fixed["atr_for_vol"] = atr_base
        fixed["atr_sma50"] = atr_base.rolling(window=50).mean()
    else:
        fixed["atr_for_vol"] = None
        fixed["atr_sma50"] = None

    fixed["point"] = p.get("point", 0.01)
    fixed["tick_value"] = p.get("tick_value", 0.01)

    # MTF: precompute H4 EMA(100) aligned to H1 bar times.
    fixed["mtf_h4_ema"] = None
    if p.get("mtf_enabled", False):
        h4_ema_period = p.get("mtf_h4_ema_period", 100)
        h4_df = fixed.get("h4_df")
        if h4_df is not None and len(h4_df) > h4_ema_period:
            h4_ema_series = calc_ma(h4_df, h4_ema_period, p.get("ma_type", "kama"))
            fixed["mtf_h4_ema"] = h4_ema_series.reindex(df["time"], method="ffill")

    return fixed


class Backtest:
    def __init__(self, df, params, df_m1=None, df_m5=None, df_m15=None, df_m30=None, fixed=None):
        self.df = df
        self.df_m1 = df_m1
        self.df_m5 = df_m5
        self.df_m15 = df_m15
        self.df_m30 = df_m30
        self._fixed = fixed
        self.p = params
        self.n = len(df)
        self._last_i = 0

        self.trades = []
        self.equity = []
        self.positions = []

        # MR consecutive-loss cooldown bookkeeping.
        self._mr_loss_streak = 0
        self._mr_last_loss_i = -(10**9)

        self.initial_balance = self.p.get("initial_balance", 400000.0)
        self.balance = self.initial_balance
        self.risk_percent = self.p.get("risk_percent", 1.0)
        self.commission = self.p.get("commission", 0.0)
        self.slippage = self.p.get("slippage_points", 2)
        self.stats = {}

        self._precompute()

    def _precompute(self):
        df = self.df
        p = self.p
        n = len(df)
        f = self._fixed or {}

        self.ma_type = p.get("ma_type", "kama")
        self.ema_fast = _get_ma_cached(df, p["ema_fast"], self.ma_type)
        self.ema_slow = _get_ma_cached(df, p["ema_slow"], self.ma_type)
        if p.get("ch_accelerate_enabled", False):
            self.ch_accel_ema = calc_ma(df, p.get("ch_accelerate_period", 14), self.ma_type)
        else:
            self.ch_accel_ema = None

        # Use precomputed fixed data when available (optimizer fast path)
        if "atr_series" in f:
            self.atr_series = f["atr_series"]
            self.adx_series = f["adx_series"]
            self.vol_sma = f["vol_sma"]
            self.atr_for_sma = f["atr_for_sma"]
            self.atr_sma = f["atr_sma"]
            self.atr_for_vol = f["atr_for_vol"]
            self.atr_sma50 = f["atr_sma50"]
            self.h4_adx = f["h4_adx"]
            self.d1_adx = f["d1_adx"]
            self.h4_df = f["h4_df"]
            self.d1_df = f["d1_df"]
            self.h4_adx_aligned = f["h4_adx_aligned"]
            self.d1_adx_aligned = f["d1_adx_aligned"]
            self.htf_ema_aligned = f["htf_ema_aligned"]
            self.htf_slope_aligned = f["htf_slope_aligned"]
            self.htf_close_aligned = f["htf_close_aligned"]
            self.day_start_idx = f["day_start_idx"]
            self.point = f["point"]
            self.tick_value = f["tick_value"]
            atr_base = f["atr_base"]
        else:
            self.atr_series = calc_atr_series(df, p.get("atr_period", 14))
            self.adx_series = calc_adx_series(df, p.get("atr_period", 14)) if p.get("adx_enabled", True) else None
            self.vol_sma = df["tick_volume"].rolling(window=p.get("vf_sma_period", 20)).mean()

            tr = pd.DataFrame(
                {
                    "hl": df["high"] - df["low"],
                    "hc": (df["high"] - df["close"].shift()).abs(),
                    "lc": (df["low"] - df["close"].shift()).abs(),
                }
            ).max(axis=1)
            atr_base = tr.ewm(alpha=1.0 / p.get("atr_period", 14), adjust=False).mean()
            if p.get("volatility_filter", False):
                self.atr_for_sma = atr_base
                self.atr_sma = atr_base.rolling(window=p.get("atr_sma_period", 20)).mean()
            else:
                self.atr_for_sma = None
                self.atr_sma = None

            self.h4_adx = None
            self.d1_adx = None
            self.h4_df = None
            self.d1_df = None
            self.h4_adx_aligned = None
            self.d1_adx_aligned = None
            self.htf_ema_aligned = None
            self.htf_slope_aligned = None
            self.htf_close_aligned = None
            if n >= 100:
                _resample = df.set_index("time")
                self.h4_df = _resample.resample("4h").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
                ).dropna(subset=["open"])
                h4_adx_arr = calc_adx_series(self.h4_df, p.get("atr_period", 14))
                if h4_adx_arr is not None:
                    self.h4_adx = pd.Series(h4_adx_arr, index=self.h4_df.index)
                self.d1_df = _resample.resample("D").agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"}
                ).dropna(subset=["open"])
                d1_adx_arr = calc_adx_series(self.d1_df, p.get("atr_period", 14))
                if d1_adx_arr is not None:
                    self.d1_adx = pd.Series(d1_adx_arr, index=self.d1_df.index)
                htf_slow = p.get("htf_ema_slow", 200)
                if len(self.h4_df) > htf_slow:
                    h4_ma = calc_ma(self.h4_df, htf_slow, p.get("ma_type", "kama"))
                    slope_window = min(12, max(2, len(self.h4_df) // 10))
                    h4_slope = h4_ma.diff(slope_window)
                    self.htf_ema_aligned = h4_ma.reindex(df["time"], method="ffill")
                    self.htf_slope_aligned = h4_slope.reindex(df["time"], method="ffill")
                    self.htf_close_aligned = self.h4_df["close"].reindex(df["time"], method="ffill")
                if self.h4_adx is not None and len(self.h4_adx) > 0:
                    self.h4_adx_aligned = np.array(
                        [
                            self.h4_adx.iloc[i // 4 - 1]
                            if i >= 4 and i // 4 - 1 < len(self.h4_adx)
                            else np.nan
                            for i in range(n)
                        ],
                        dtype=float,
                    )
                else:
                    self.h4_adx_aligned = np.full(n, np.nan)
                if self.d1_adx is not None and len(self.d1_adx) > 0:
                    self.d1_adx_aligned = np.array(
                        [
                            self.d1_adx.iloc[i // 24 - 1]
                            if i >= 24 and i // 24 - 1 < len(self.d1_adx)
                            else np.nan
                            for i in range(n)
                        ],
                        dtype=float,
                    )
                else:
                    self.d1_adx_aligned = np.full(n, np.nan)

            day_start_idx = np.arange(n)
            if n > 0 and hasattr(df["time"], "dt"):
                dates = df["time"].dt.date.values
                for i in range(1, n):
                    if dates[i] == dates[i - 1]:
                        day_start_idx[i] = day_start_idx[i - 1]
            self.day_start_idx = day_start_idx

            if p.get("dr_vol_adjust", True):
                self.atr_for_vol = atr_base
                self.atr_sma50 = atr_base.rolling(window=50).mean()
            else:
                self.atr_for_vol = None
                self.atr_sma50 = None

            self.point = p.get("point", 0.01)
            self.tick_value = p.get("tick_value", 0.01)

        self.ml_model = None
        self.ml_features = None
        self.ml_opt_threshold = None
        self.ml_loaded = False

        # MTF: precompute M15 (entry) and H4 (context) MAs aligned to H1 bar times
        self.mtf_m15_fast = None
        self.mtf_m15_slow = None
        self.mtf_h4_ema = None
        if p.get("mtf_enabled", False):
            m15_fast = p.get("mtf_m15_ema_fast", max(5, p["ema_fast"] // 2))
            m15_slow = p.get("mtf_m15_ema_slow", max(8, p["ema_slow"] // 2))
            h4_ema_period = p.get("mtf_h4_ema_period", 100)
            if "mtf_h4_ema" in f and f["mtf_h4_ema"] is not None:
                self.mtf_h4_ema = f["mtf_h4_ema"]
            elif self.h4_df is not None and len(self.h4_df) > h4_ema_period:
                h4_ema_ma = calc_ma(self.h4_df, h4_ema_period, self.ma_type)
                if len(h4_ema_ma) > 1:
                    h4_shift = h4_ema_ma.index[1] - h4_ema_ma.index[0]
                    h4_ema_ma.index = h4_ema_ma.index + h4_shift
                self.mtf_h4_ema = h4_ema_ma.reindex(df["time"], method="ffill")
            if self.df_m15 is not None and len(self.df_m15) > 0:
                m15_fast_ma, m15_slow_ma = _get_mtf_m15_mas(
                    self.df_m15, m15_fast, m15_slow, self.ma_type
                )
                # Shift index forward by one M15 period so reindex(ffill) picks
                # the last CLOSED M15 bar (not the forming one) — otherwise the
                # entry trigger sees up to 15min of future data.
                self.mtf_m15_fast = m15_fast_ma.reindex(df["time"], method="ffill")
                self.mtf_m15_slow = m15_slow_ma.reindex(df["time"], method="ffill")

        self.fused_score_a = np.full(n, np.nan)
        adx_a = np.asarray(self.adx_series) if self.adx_series is not None else None
        atr_a = np.asarray(self.atr_series) if self.atr_series is not None else None
        if atr_a is not None and n > 50:
            close = self.df["close"].values
            ma_fast_a = np.asarray(self.ema_fast) if self.ema_fast is not None else None
            for i in range(50, n):
                adx_i = 0.0 if adx_a is None or np.isnan(adx_a[i]) else float(adx_a[i])
                er_i = 0.0
                er_period = p.get("er_period", 10)
                if i >= er_period:
                    ch = abs(close[i] - close[i - er_period])
                    mv = np.sum(np.abs(np.diff(close[i - er_period : i + 1])))
                    er_i = ch / mv if mv > 0 else 0.0
                # Slope in RAW PRICE UNITS so that slope/ATR is dimensionless
                # (parity with indicators.calc_ma_slope — see the unit-mismatch
                # note there). Must stay identical to calc_fused_regime_score.
                ma_slope_i = 0.0
                if ma_fast_a is not None and i >= 2 and not np.isnan(ma_fast_a[i]) and not np.isnan(ma_fast_a[i - 1]):
                    ma_slope_i = ma_fast_a[i] - ma_fast_a[i - 1]
                atr_i = atr_a[i] if not np.isnan(atr_a[i]) and atr_a[i] > 0 else 0.0
                adx_n = min(1.0, adx_i / 50.0) if adx_i > 0 else 0.0
                er_n = min(1.0, er_i)
                slope_n = min(1.0, abs(ma_slope_i) / atr_i * SLOPE_SCALE) if atr_i > 0 else 0.0
                score = 100.0 * (0.45 * adx_n + 0.35 * er_n + 0.20 * slope_n)
                self.fused_score_a[i] = score

        # MR on configured timeframe (A2): compute RSI on M30, aligned to H1.
        self.mr_rsi_h1 = None
        if self.df_m30 is not None and len(self.df_m30) > 0:
            m30 = self.df_m30.set_index("time").sort_index()
            rsi = self._rsi_series(m30["close"], p.get("mr_rsi_period", 14))
            self.mr_rsi_h1 = rsi.reindex(self.df["time"], method="ffill")

        self._precompute_ml()

    def _precompute_ml(self):
        n = len(self.df)
        self.ml_mult_buy = np.ones(n, dtype=float)
        self.ml_mult_sell = np.ones(n, dtype=float)
        # Fast path: use precomputed multipliers injected by the optimizer (B).
        inj_buy = self.p.get("ml_mult_buy")
        inj_sell = self.p.get("ml_mult_sell")
        path = self.p.get("ml_model_path")
        if inj_buy is not None and inj_sell is not None:
            self.ml_mult_buy = np.asarray(inj_buy, dtype=float)
            self.ml_mult_sell = np.asarray(inj_sell, dtype=float)
            self.ml_loaded = True
            if path and path in _ML_DATA_CACHE:
                self.ml_model = _ML_DATA_CACHE[path].get("model")
            self._load_meta_model()
            return
        buy, sell, ok = compute_ml_multipliers(self.df, self.p, df_m1=self.df_m1)
        self.ml_mult_buy = buy
        self.ml_mult_sell = sell
        self.ml_loaded = ok
        if path and path in _ML_DATA_CACHE:
            self.ml_model = _ML_DATA_CACHE[path].get("model")
        self._load_meta_model()

    def _get_m1_bars(self, h1_time):
        if self.df_m1 is None:
            return None
        m1_times = self.df_m1["time"].values
        h1_end = h1_time + pd.Timedelta(hours=1)
        start = int(np.searchsorted(m1_times, h1_time, side="left"))
        end = int(np.searchsorted(m1_times, h1_end, side="left"))
        if start >= end:
            return None
        return self.df_m1.iloc[start:end]

    def _simulate_m1(self, pe, m1_bars, spm, commission):
        is_long = pe["type"] == "buy"
        entered = False
        pos = None
        for _, m1_bar in m1_bars.iterrows():
            if not entered:
                bar_spread = m1_bar.get("spread", 0) or 0
                if is_long:
                    entry_price = m1_bar["open"] + spm * bar_spread * self.point * 0.5 + self.slippage * self.point
                else:
                    entry_price = m1_bar["open"] - spm * bar_spread * self.point * 0.5 - self.slippage * self.point
                if (is_long and entry_price <= pe["sl"]) or (not is_long and entry_price >= pe["sl"]):
                    return {"status": "skipped"}
                pos = {
                    "type": pe["type"],
                    "entry": entry_price,
                    "sl": pe["sl"],
                    "tp": pe["tp"],
                    "entry_bar": pe.get("h1_bar", 0),
                    "exit_bar": None,
                    "exit": None,
                    "pnl": None,
                    "volume": pe["volume"],
                    "remaining_volume": pe["volume"],
                    "status": "open",
                    "exit_reason": None,
                    "regime": pe["regime"],
                    "entry_type": pe.get("entry_type", "crossover"),
                    "atr_entry": pe["atr_entry"],
                    "symbol": self.p.get("symbol", ""),
                    "m1_entry": True,
                }
                if commission > 0:
                    pos["_entry_comm"] = commission * pe["volume"]
                    self.commission_paid += pos["_entry_comm"]
                entered = True
                continue
            hit_sl = (is_long and m1_bar["low"] <= pos["sl"]) or (not is_long and m1_bar["high"] >= pos["sl"])
            hit_tp = pos["tp"] is not None and (
                (is_long and m1_bar["high"] >= pos["tp"]) or (not is_long and m1_bar["low"] <= pos["tp"])
            )
            if hit_sl or hit_tp:
                sl_or_tp = "SL" if hit_sl else "TP"
                exit_price = pos["sl"] if hit_sl else pos["tp"]
                if is_long:
                    exit_price -= self.slippage * self.point
                else:
                    exit_price += self.slippage * self.point
                pnl = (
                    (exit_price - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * pe["volume"]
                    if is_long
                    else (pos["entry"] - exit_price) / max(self.point, 1e-10) * self.tick_value * pe["volume"]
                )
                entry_comm = pos.get("_entry_comm", 0.0)
                pnl -= entry_comm
                if commission > 0:
                    pnl -= commission * pe["volume"]
                    self.commission_paid += commission * pe["volume"]
                pos["exit"] = exit_price
                pos["exit_bar"] = pe.get("h1_bar", 0)
                pos["pnl"] = pnl
                pos["status"] = "closed"
                pos["exit_reason"] = f"M1_{sl_or_tp}"
                return {"status": "closed", "trade": dict(pos), "pnl": pnl}
        if entered:
            return {"status": "open", "position": pos}
        return {"status": "skipped"}

    def _h4_adx_at(self, i):
        if self.h4_adx is None:
            return None
        # Last COMPLETED H4 bar (i // 4 - 1 gives the previous H4 block).
        if i < 4:
            return None
        idx = i // 4 - 1
        if idx >= len(self.h4_adx):
            return None
        val = self.h4_adx.iloc[idx]
        return None if pd.isna(val) else float(val)

    def _d1_adx_at(self, i):
        if self.d1_adx is None:
            return None
        if i < 24:
            return None
        idx = i // 24 - 1
        if idx >= len(self.d1_adx):
            return None
        val = self.d1_adx.iloc[idx]
        return None if pd.isna(val) else float(val)

    def _detect_regime(self, i):
        if self.adx_series is None:
            return "ranging"
        raw = self.adx_series
        if i >= len(raw):
            return "uncertain"
        adx_h1 = float(raw[i])
        if pd.isna(adx_h1):
            return "uncertain"
        h4_adx = self._h4_adx_at(i)
        d1_adx = self._d1_adx_at(i)
        trend_thresh = self.p.get("adx_trend_threshold", 25)
        range_thresh = self.p.get("adx_range_threshold", 20)

        adx_slope = None
        if i >= 5:
            vals = [float(raw[j]) for j in range(i - 5, i + 1) if not pd.isna(raw[j])]
            if len(vals) > 5:
                adx_slope = vals[-1] - vals[0]

        h4_trending = h4_adx is not None and h4_adx >= range_thresh
        d1_trending = d1_adx is not None and d1_adx >= range_thresh

        exhaustion_adx = self.p.get("exhaustion_adx_threshold", 40)
        exhaustion_slope = self.p.get("exhaustion_slope_threshold", 2)
        exhaustion = adx_h1 >= exhaustion_adx and adx_slope is not None and adx_slope < -exhaustion_slope
        if exhaustion:
            return "exhaustion"
        if adx_h1 >= trend_thresh and (h4_trending or d1_trending):
            return "strong_trend"
        if adx_h1 >= trend_thresh:
            return "weak_trend"
        if adx_h1 <= range_thresh and not h4_trending and not d1_trending:
            return "ranging"
        return "uncertain"

    def _check_ml_signal(self, i, direction):
        if not self.ml_loaded:
            return 1.0
        if i < 0 or i >= len(self.ml_mult_buy):
            return 1.0
        mult = float(self.ml_mult_buy[i] if direction == "buy" else self.ml_mult_sell[i])
        if mult <= 0:
            return 0.0
        if self.ml_meta_model is not None and self.ml_meta_features is not None:
            if i >= len(self.ml_meta_proba):
                return mult
            meta_conf = float(self.ml_meta_proba[i])
            meta_threshold = self.p.get("ml_meta_threshold", 0.50)
            if meta_conf < meta_threshold:
                return 0.0
        return mult

    def _load_meta_model(self):
        """Load meta-model for ML validation (parity with filters.check_ml_signal)."""
        self.ml_meta_model = None
        self.ml_meta_features = None
        self.ml_meta_proba = None
        path = self.p.get("ml_model_path")
        if not path:
            return
        meta_path = path.replace(".pkl", ".meta.pkl")
        if not os.path.exists(meta_path):
            return
        try:
            from train_model import EnsembleModel, EnsembleRegressor, _CalibratedWrapper

            import __main__
            __main__.EnsembleModel = EnsembleModel
            __main__.EnsembleRegressor = EnsembleRegressor
            __main__._CalibratedWrapper = _CalibratedWrapper
            meta_data = joblib.load(meta_path)
            self.ml_meta_model = meta_data.get("model")
            self.ml_meta_features = meta_data.get("metadata", {}).get("features")
            if self.ml_meta_model is not None and self.ml_meta_features is not None:
                from ml_features import compute_features
                feat_df = compute_features(self.df, symbol=self.p.get("symbol", ""), m1_df=self.df_m1)
                missing = [c for c in self.ml_meta_features if c not in feat_df.columns]
                if not missing:
                    X = feat_df[self.ml_meta_features].fillna(0)
                    proba = self.ml_meta_model.predict_proba(X)
                    self.ml_meta_proba = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        except Exception:
            logging.warning("Meta-model loading failed", exc_info=True)

    def _check_volume_filter(self, i, signal):
        # Delegate to the shared analytics module (single source of truth with
        # the live engine, agent audit A1). Bar i is the current (closed) bar in
        # a backtest, so it maps to the last-closed-bar convention used live.
        df_closed = self.df.iloc[: i + 1]
        return volume_filter_pass(df_closed, signal, self.p)

    def _check_execution_sanity(self, i, signal):
        if not self._check_volume_filter(i, signal):
            return False
        if not self._check_spread_filter(i):
            return False
        return self._check_tape_reading(i, signal)

    def _check_tape_reading(self, i, signal):
        """Parity with live filters.check_tape_reading: M1 tape pressure gate.

        Uses df_m1 bars aligned to the current H1 bar's time window. If M1 data
        is unavailable or insufficient, passes (same as live when MT5 returns
        insufficient data).
        """
        p = self.p
        if not p.get("tape_enabled", True):
            return True
        lookback = p.get("tape_m1_lookback", 20)
        h1_time = self.df["time"].iloc[i]
        m1_bars = self._get_m1_bars(h1_time)
        if m1_bars is None or len(m1_bars) < 10:
            return True
        df = m1_bars.tail(lookback)
        if len(df) < 10:
            return True
        avg_range = (df["high"] - df["low"]).mean()
        if avg_range == 0 or pd.isna(avg_range):
            return True
        bullish_pressure = (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10)
        range_ratio = (df["high"] - df["low"]) / (avg_range + 1e-10)
        avg_pressure = bullish_pressure.tail(5).mean()
        range_active = range_ratio.tail(5).mean()
        if (
            signal == "buy"
            and avg_pressure < p.get("tape_bearish_pressure", 0.35)
            and range_active > p.get("tape_range_ratio", 1.2)
        ):
            return False
        return not (
            signal == "sell"
            and avg_pressure > p.get("tape_bullish_pressure", 0.65)
            and range_active > p.get("tape_range_ratio", 1.2)
        )

    def _check_spread_filter(self, i):
        p = self.p
        if not p.get("spf_enabled", True):
            return True
        bar = self.df.iloc[i]
        spread = bar.get("spread")
        if spread is None or spread <= 0:
            return True
        atr = self.atr_series.iloc[i]
        if pd.isna(atr) or atr <= 0:
            return True
        ratio = (spread * self.point) / atr
        threshold = p.get("spf_max_ratio", 0.30)
        return ratio <= threshold

    def _check_session_time(self, i):
        p = self.p
        if not p.get("session_enabled", False):
            return True
        bar_time = self.df["time"].iloc[i]
        if not hasattr(bar_time, "hour"):
            return True
        t = bar_time.time()
        london_open = p.get("london_open", 13)
        london_close = p.get("london_close", 22)
        if p.get("session_only", False):
            return london_open <= t.hour <= london_close
        if p.get("require_overlap", False):
            return 12 <= t.hour <= 17
        if p.get("skip_asian", False):
            asian_open = p.get("asian_open", 5)
            asian_close = p.get("asian_close", 12)
            if asian_open <= t.hour <= asian_close:
                return False
        return True

    def _check_daily_loss(self, i, cumulative_pnl):
        p = self.p
        limit_pct = p.get("daily_loss_pct", 5.0)
        if limit_pct <= 0:
            return True
        day_start = self.day_start_idx[i]
        day_pnl = 0.0
        for t in self.trades:
            if t.get("exit_bar") is not None and t["exit_bar"] >= day_start and t["exit_bar"] <= i:
                pnl = t.get("pnl", 0)
                if pnl is not None:
                    day_pnl += pnl
        equity = self.initial_balance + cumulative_pnl
        if equity <= 0:
            return True
        loss_pct = (-day_pnl / equity) * 100 if day_pnl < 0 else 0
        return loss_pct < limit_pct

    def _check_tail_risk(self, i):
        p = self.p
        if not p.get("tr_enabled", True):
            return True
        needed = p.get("tr_lookback", 50) + 10
        if i < needed:
            return True
        close = self.df["close"].values[i - needed : i + 1]
        if len(close) < 20:
            return True
        returns = np.diff(close) / close[:-1]
        if len(returns) < 10:
            return True
        mean_r = np.mean(returns)
        std_r = np.std(returns)
        last_r = returns[-1]
        sigma_thresh = p.get("tr_sigma", 3.0)
        if std_r > 0 and abs(last_r - mean_r) / std_r > sigma_thresh:
            return False
        equity_arr = np.array(self.equity) if len(self.equity) > 0 else np.array([self.initial_balance])
        if len(equity_arr) < 10:
            return True
        peak = max(float(np.max(equity_arr)), 1e-10)
        current = float(equity_arr[-1])
        dd_pct = (peak - current) / peak * 100
        cb_dd = p.get("cb_dd_pct", 15.0)
        max_dd = p.get("tr_max_dd_pct", 8.0)
        if dd_pct >= cb_dd:
            return False
        return not dd_pct >= max_dd

    def _check_fused_regime_gate(self, i):
        """Gate 1 — Fused Regime Hysteresis. Stateful gate that opens/closes
        based on the fused regime score (ADX 45%, ER 35%, MA-slope 20%).
        Returns True (gate open = tradeable) or False (gate closed)."""
        if not hasattr(self, "_fused_gate_open"):
            self._fused_gate_open = False
        threshold = self.p.get("fused_threshold", 50.0)
        buffer = self.p.get("fused_buffer", 5.0)
        score = self.fused_score_a[i] if hasattr(self, "fused_score_a") and i < len(self.fused_score_a) else np.nan
        if np.isnan(score):
            return False  # not enough data yet
        if self._fused_gate_open:
            self._fused_gate_open = score >= (threshold - buffer / 2.0)
        else:
            self._fused_gate_open = score > (threshold + buffer / 2.0)
        return self._fused_gate_open

    def _register_close(self, pos):
        """Track MR consecutive-loss streak for the MR cooldown gate (A3)."""
        if pos.get("entry_type") != "mean_reversion":
            return
        pnl = pos.get("pnl", 0)
        if pnl is not None and pnl < 0:
            self._mr_loss_streak += 1
            self._mr_last_loss_i = pos.get("exit_bar", self._mr_last_loss_i)
        else:
            self._mr_loss_streak = 0

    def _get_pullback_signal(self, i):
        p = self.p
        if not p.get("pb_enabled", True):
            return None, None
        if i < 3:
            return None, None
        cur_atr = self.atr_series.iloc[i]
        if pd.isna(cur_atr) or cur_atr <= 0:
            return None, None

        trigger_idx = i - 1
        confirm_idx = i

        trigger_fast = self.ema_fast.iloc[trigger_idx]
        trigger_slow = self.ema_slow.iloc[trigger_idx]
        trigger_price = self.df["close"].iloc[trigger_idx]
        trigger_high = self.df["high"].iloc[trigger_idx]
        trigger_low = self.df["low"].iloc[trigger_idx]
        confirm_close = self.df["close"].iloc[confirm_idx]

        if pd.isna(trigger_fast) or pd.isna(trigger_slow):
            return None, None

        pb_dist = cur_atr * p.get("pb_atr_mult", 2.0)
        min_pb = cur_atr * p.get("pb_atr_min_dist", 0.1)

        signal = None
        if trigger_fast > trigger_slow:
            dist = abs(trigger_price - trigger_fast)
            if not (min_pb <= dist <= pb_dist):
                return None, None
            # Volume filter
            if not self._pb_volume_check(trigger_idx):
                return None, None
            # Structure filter
            if not self._pb_structure_check(trigger_idx, "buy"):
                return None, None
            # Confirmation bar
            if confirm_close <= trigger_high:
                return None, None
            # HTF trend
            htf_dec, _ = self._check_htf_trend(i, "buy")
            if htf_dec == "block":
                return None, None
            signal = "buy"
        elif trigger_fast < trigger_slow:
            dist = abs(trigger_price - trigger_fast)
            if not (min_pb <= dist <= pb_dist):
                return None, None
            if not self._pb_volume_check(trigger_idx):
                return None, None
            if not self._pb_structure_check(trigger_idx, "sell"):
                return None, None
            if confirm_close >= trigger_low:
                return None, None
            htf_dec, _ = self._check_htf_trend(i, "sell")
            if htf_dec == "block":
                return None, None
            signal = "sell"
        if signal:
            return signal, cur_atr
        return None, None

    def _pb_volume_check(self, trigger_idx):
        p = self.p
        if not p.get("pb_volume_enabled", True):
            return True
        period = p.get("pb_volume_sma_period", 20)
        threshold = p.get("pb_volume_threshold", 0.8)
        if trigger_idx < period:
            return True
        vol_series = self.df["tick_volume"].iloc[trigger_idx - period + 1 : trigger_idx + 1]
        vol_sma = vol_series.mean()
        if vol_sma <= 0:
            return True
        return self.df["tick_volume"].iloc[trigger_idx] < vol_sma * threshold

    def _pb_structure_check(self, trigger_idx, direction):
        lookback = self.p.get("pb_structure_lookback", 5)
        start = trigger_idx - lookback
        if start < 0:
            return True
        if direction == "buy":
            prior_min = self.df["low"].iloc[start:trigger_idx].min()
            return self.df["low"].iloc[trigger_idx] > prior_min
        else:
            prior_max = self.df["high"].iloc[start:trigger_idx].max()
            return self.df["high"].iloc[trigger_idx] < prior_max

    def _rsi_series(self, close, period):
        """Vectorized Wilder RSI over a full close Series (used for the MR M30 TF).

        Uses the SAME Wilder smoothing as indicators.calc_rsi: seed with the
        simple average of the first `period` gains/losses, then smooth with
        avg = (avg * (period-1) + current) / period. This guarantees parity
        with the live check_mean_reversion_exit which calls calc_rsi.
        """
        close = pd.Series(close) if not isinstance(close, pd.Series) else close
        if period < 1 or len(close) < period + 1:
            return pd.Series(50.0, index=close.index)
        close_arr = close.values
        deltas = np.diff(close_arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        n = len(gains)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rsi_vals = np.full(len(close_arr), np.nan)
        for i in range(period, n):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi_vals[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_vals[i + 1] = 100.0 - (100.0 / (1.0 + rs))
        return pd.Series(rsi_vals, index=close.index).fillna(50.0)

    def _calc_rsi_wilder(self, i, period):
        lookback = period + 30
        if i < lookback:
            return 50.0
        close = self.df["close"].values[i - lookback : i + 1]
        if len(close) < period + 2:
            return 50.0
        deltas = np.diff(close)
        if len(deltas) == 0:
            return 50.0
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        slices = gains[-period - 1 : -1] if len(gains) >= period else gains
        avg_gain = np.mean(slices) if len(slices) > 0 else 0
        slices_l = losses[-period - 1 : -1] if len(losses) >= period else losses
        avg_loss = np.mean(slices_l) if len(slices_l) > 0 else 0
        for j in range(max(0, len(gains) - period), len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[j]) / period
            avg_loss = (avg_loss * (period - 1) + losses[j]) / period
        if avg_loss <= 0:
            return 50.0
        return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    def _get_mean_reversion_signal(self, i):
        p = self.p
        if not p.get("mr_enabled", True):
            return None, None
        # MR consecutive-loss cooldown (A3): skip if >=2 losses within window.
        if p.get("mr_cooldown_enabled", True):
            cooldown_bars = p.get("mr_cooldown_bars", 2)
            if self._mr_loss_streak >= 2 and (i - self._mr_last_loss_i) < cooldown_bars:
                return None, None
        rsi_period = p.get("mr_rsi_period", 14)
        # A2: RSI on the configured MR timeframe (M30) when available, else H1.
        if self.mr_rsi_h1 is not None and i < len(self.mr_rsi_h1) and not pd.isna(self.mr_rsi_h1.iloc[i]):
            cur_rsi = float(self.mr_rsi_h1.iloc[i])
        else:
            cur_rsi = self._calc_rsi_wilder(i, rsi_period)
        oversold = p.get("mr_rsi_oversold", 30)
        overbought = p.get("mr_rsi_overbought", 70)
        cur_price = self.df["close"].iloc[i]
        # A2: HTF reference is the live htf_timeframe (H4) EMA200, not D1.
        if self.htf_ema_aligned is not None and i < len(self.htf_ema_aligned):
            htf_ema200 = self.htf_ema_aligned.iloc[i]
        else:
            htf_ema200 = None
        signal = None
        dev = p.get("mr_htf_deviation", 0.0)
        if cur_rsi < oversold:
            if htf_ema200 is None or cur_price > htf_ema200 * (1.0 - dev):
                signal = "buy"
        elif cur_rsi > overbought and (htf_ema200 is None or cur_price < htf_ema200 * (1.0 + dev)):
            signal = "sell"
        atr = self.atr_series.iloc[i]
        return signal, atr

    def _get_mtf_signal(self, i):
        """Hierarchical bias+trigger MTF signal (AGENTS.md).

        H4: price vs EMA(100) with 0.5*ATR neutral band determines bias.
        H1: MA crossover must agree with H4 direction.
        M15: MA crossover provides entry timing when aligned.

        Returns (signal, entry_type, confidence):
          signal:     "buy", "sell", or None
          entry_type: "crossover" (all 3 TFs agree) or "pullback" (H4+H1 only)
          confidence: 1.0 (crossover) or 0.67 (pullback) or 0.0
        """
        h4_ema = self.mtf_h4_ema.iloc[i] if self.mtf_h4_ema is not None else np.nan
        h1_close = self.df["close"].iloc[i]
        cur_atr = self.atr_series.iloc[i]
        if pd.isna(h4_ema) or pd.isna(h1_close) or pd.isna(cur_atr) or cur_atr <= 0 or i < 2:
            return None, None, 0.0

        # H4 bias with 0.5*ATR neutral band
        bias = h1_close - float(h4_ema)
        neutral_band = cur_atr * 0.5
        if abs(bias) <= neutral_band:
            return None, None, 0.0
        h4_direction = 1 if bias > 0 else -1

        # H1 crossover must agree with H4 bias
        h1_cf = self.ema_fast.iloc[i]
        h1_cs = self.ema_slow.iloc[i]
        h1_pf = self.ema_fast.iloc[i - 1]
        h1_ps = self.ema_slow.iloc[i - 1]
        if any(pd.isna(x) for x in (h1_cf, h1_cs, h1_pf, h1_ps)):
            return None, None, 0.0
        h1_cross = 0
        if h1_pf <= h1_ps and h1_cf > h1_cs:
            h1_cross = 1
        elif h1_pf >= h1_ps and h1_cf < h1_cs:
            h1_cross = -1
        if h1_cross != h4_direction:
            return None, None, 0.0

        direction = "buy" if h1_cross > 0 else "sell"

        # M15 crossover check for entry timing
        if self.mtf_m15_fast is not None and i > 0:
            try:
                m15_cf = self.mtf_m15_fast.iloc[i]
                m15_cs = self.mtf_m15_slow.iloc[i]
                m15_pf = self.mtf_m15_fast.iloc[i - 1]
                m15_ps = self.mtf_m15_slow.iloc[i - 1]
                if not any(pd.isna(x) for x in (m15_cf, m15_cs, m15_pf, m15_ps)):
                    m15_cross = 0
                    if m15_pf <= m15_ps and m15_cf > m15_cs:
                        m15_cross = 1
                    elif m15_pf >= m15_ps and m15_cf < m15_cs:
                        m15_cross = -1
                    if m15_cross == h1_cross:
                        return direction, "crossover", 1.0
                    elif m15_cross != 0:
                        return None, None, 0.0
            except (IndexError, KeyError):
                pass

        return direction, "pullback", 0.67

    def _check_htf_trend(self, i, signal):
        """Parity with live signals.check_htf_trend: 3-state HTF decision.

        Returns ('allow', 1.0), ('soft', size_mult) or ('block', 0.0) using the
        time-aligned H4 EMA(slow) + slope computed in _precompute.
        """
        p = self.p
        if self.htf_ema_aligned is None:
            return "allow", 1.0
        htf_ma_val = self.htf_ema_aligned.iloc[i]
        if pd.isna(htf_ma_val):
            return "soft", p.get("htf_misalign_size_mult", 0.5)
        htf_price = self.htf_close_aligned.iloc[i] if self.htf_close_aligned is not None else self.df["close"].iloc[i]
        slope = self.htf_slope_aligned.iloc[i] if self.htf_slope_aligned is not None else 0.0
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
        return "soft", p.get("htf_misalign_size_mult", 0.5)

    def _check_mean_reversion_exit(self, i, pos):
        p = self.p
        if not p.get("mr_enabled", True):
            return False
        rsi_period = p.get("mr_rsi_period", 14)

        # Use the SAME RSI source as MR entry (_get_mean_reversion_signal): the
        # M30-based series (mr_rsi_h1) when available, else the H1 Wilder
        # fallback. Live check_mean_reversion_exit reads the M30 timeframe too,
        # and the njit twin reads the mr_rsi_a array built the same way — so exit
        # previously using _calc_rsi_wilder unconditionally diverged from entry,
        # from live, and from the twin (agent audit H1).
        def _rsi_at(j):
            if self.mr_rsi_h1 is not None and 0 <= j < len(self.mr_rsi_h1) and not pd.isna(self.mr_rsi_h1.iloc[j]):
                return float(self.mr_rsi_h1.iloc[j])
            return self._calc_rsi_wilder(j, rsi_period)

        prev_rsi = _rsi_at(i - 1)
        cur_rsi = _rsi_at(i)
        is_long = pos["type"] == "buy"
        if is_long and prev_rsi < 50 and cur_rsi >= 50:
            return True
        return bool(not is_long and prev_rsi > 50 and cur_rsi <= 50)

    def _calc_kelly_mult(self):
        p = self.p
        if not p.get("dr_enabled", True):
            return 1.0
        closed = [t for t in self.trades if t.get("pnl") is not None and t.get("type") != "partial"]
        lookback = p.get("dr_lookback", 50)
        closed = closed[-lookback:]
        if len(closed) < 10:
            return 1.0
        pnls = [t["pnl"] for t in closed]
        wins = [x for x in pnls if x > 0]
        losses = [x for x in pnls if x <= 0]
        if not wins or not losses:
            return 1.0
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        b = avg_win / avg_loss if avg_loss > 0 else 1
        q = 1.0 - win_rate
        kelly = (win_rate * b - q) / b if b > 0 else 0
        kelly = max(0.0, kelly) * p.get("dr_kelly_fraction", 0.25)
        kelly = max(p.get("dr_min_mult", 0.25), min(p.get("dr_max_mult", 1.5), kelly))
        return kelly

    def _calc_volatility_mult(self, i):
        p = self.p
        if not p.get("dr_vol_adjust", True):
            return 1.0
        if self.atr_for_vol is None or self.atr_sma50 is None:
            return 1.0
        cur_atr = self.atr_for_vol.iloc[i]
        cur_sma50 = self.atr_sma50.iloc[i]
        if pd.isna(cur_atr) or pd.isna(cur_sma50) or cur_sma50 <= 0:
            return 1.0
        ratio = cur_atr / cur_sma50
        if ratio > 1.2:
            return max(0.25, 1.0 / ratio)
        return 1.0

    def _tail_risk_score(self):
        """Stateful drawdown score for the entry-scoring tail_risk component —
        the raw-value INPUT to analytics.compute_entry_score (the score math
        lives there; this is backtest-side state only). 1.0 when disabled."""
        p = self.p
        if not p.get("tr_enabled", True):
            return 1.0
        current_equity = self.initial_balance + sum(t.get("pnl", 0) for t in self.trades if t.get("pnl"))
        max_dd = p.get("tr_max_dd_pct", 8.0)
        running_max = max(self.equity + [self.initial_balance]) if self.equity else self.initial_balance
        if running_max > 0:
            dd_pct = max(0, running_max - current_equity) / running_max * 100
            if dd_pct >= max_dd:
                return 0.0
            return 1.0 - (dd_pct / max_dd)
        return 0.5

    def run(self, fast=False):
        # Numba fast path now supports MTF (item #9): the MTF fused signal is a
        # pure function of the 5 precomputed time-aligned MA arrays, so it runs
        # natively in _simulate_core. The only remaining blocker is M1 entry
        # simulation (df_m1), which the fast path doesn't reproduce.
        if fast and self.df_m1 is None and _njit_available():
            try:
                return self._run_fast()
            except Exception:  # pragma: no cover - fast path fallback
                import logging

                # ERROR, not WARNING: the fast path is bit-exact and default-on,
                # so a failure here silently costs ~8x runtime. It must be loud
                # enough to surface in CI logs, while still not aborting the run.
                logging.error("[backtest] fast path failed; falling back to reference loop", exc_info=True)
        return self._run_reference()

    def _run_reference(self):
        df = self.df
        n = self.n
        p = self.p
        warmup = max(p.get("ema_fast", 8), p.get("ema_slow", 32), p.get("atr_period", 14), 250) + 50
        self.equity = []
        cumulative_pnl = 0.0
        spm = p.get("spread_model", 1.0)
        commission = p.get("commission", 0.0)
        pending_entry = None
        self.commission_paid = 0.0
        self._last_i = 0

        for i in range(warmup, n):
            bar = df.iloc[i]
            self._last_i = i

            if pending_entry is not None:
                pe = pending_entry
                pending_entry = None
                is_long_pe = pe["type"] == "buy"

                # Compute actual entry price (including spread + slippage) so
                # SL/TP can be set relative to fill price, not signal close.
                # Must happen before M1 simulation for correct intra-bar exits.
                bar_spread = bar.get("spread", 0) or 0
                if is_long_pe:
                    entry_price = bar["open"] + spm * bar_spread * self.point * 0.5 + self.slippage * self.point
                else:
                    entry_price = bar["open"] - spm * bar_spread * self.point * 0.5 - self.slippage * self.point
                sp = pe.get("sl_points")
                tpp = pe.get("tp_points", 0)
                if sp and pe.get("sl") is not None:
                    if is_long_pe:
                        pe["sl"] = entry_price - sp * self.point
                        pe["tp"] = entry_price + tpp * self.point
                    else:
                        pe["sl"] = entry_price + sp * self.point
                        pe["tp"] = entry_price - tpp * self.point
                m1_result = None
                if self.df_m1 is not None:
                    m1_bars = self._get_m1_bars(bar["time"])
                    if m1_bars is not None and len(m1_bars) > 1:
                        pe["h1_bar"] = i
                        m1_result = self._simulate_m1(pe, m1_bars, spm, commission)

                if m1_result is not None:
                    if m1_result["status"] == "open":
                        self.positions.append(m1_result["position"])
                        if commission > 0:
                            cumulative_pnl -= commission * pe["volume"]
                            self.commission_paid += commission * pe["volume"]
                    elif m1_result["status"] == "closed":
                        cumulative_pnl += m1_result["pnl"]
                        self.trades.append(m1_result["trade"])
                    self.equity.append(self.initial_balance + cumulative_pnl)
                    continue

                # NOTE: previously a same-bar SL hit (bar low <= SL on the entry
                # bar) discarded the trade entirely, understating losses. Training
                # labels (triple_barrier_labels) always OPEN the trade and only
                # start checking SL/TP from the NEXT bar (j >= 1). So we now open
                # the position unconditionally and let the exit loop (which skips
                # the entry bar below) evaluate it from the following bar.
                pos = {
                    "type": pe["type"],
                    "entry": entry_price,
                    "sl": pe["sl"],
                    "tp": pe["tp"],
                    "entry_bar": i,
                    "exit_bar": None,
                    "exit": None,
                    "pnl": None,
                    "volume": pe["volume"],
                    "remaining_volume": pe["volume"],
                    "status": "open",
                    "exit_reason": None,
                    "regime": pe["regime"],
                    "entry_type": pe.get("entry_type", "crossover"),
                    "atr_entry": pe["atr_entry"],
                    "symbol": self.p.get("symbol", ""),
                }
                self.positions.append(pos)
                if commission > 0:
                    cumulative_pnl -= commission * pe["volume"]
                    self.commission_paid += commission * pe["volume"]

            cur_atr = self.atr_series.iloc[i]
            if pd.isna(cur_atr) or cur_atr <= 0:
                self.equity.append(self.initial_balance + cumulative_pnl)
                continue

            regime = self._detect_regime(i)

            # Gate 1 — advance the fused-regime hysteresis EVERY bar (not only
            # when a signal exists) so the gate state tracks the market like the
            # live bot (main.py calls gate.update() every cycle). Previously the
            # gate only moved on signal bars, leaving it stuck closed after quiet
            # stretches and over-blocking the first crossover (backtest-vs-live
            # parity gap).
            gate_state = self._check_fused_regime_gate(i)

            mtf_confidence = 0.0
            buy_signal = False
            sell_signal = False
            if p.get("mtf_enabled", False):
                signal, entry_type, mtf_confidence = self._get_mtf_signal(i)
                buy_signal = signal == "buy"
                sell_signal = signal == "sell"
            else:
                cur_fast = self.ema_fast.iloc[i]
                cur_slow = self.ema_slow.iloc[i]
                prev_fast = self.ema_fast.iloc[i - 1]
                prev_slow = self.ema_slow.iloc[i - 1]

                buy_signal = prev_fast <= prev_slow and cur_fast > cur_slow
                sell_signal = prev_fast >= prev_slow and cur_fast < cur_slow

                signal = None
                entry_type = None
                if buy_signal:
                    signal = "buy"
                    entry_type = "crossover"
                elif sell_signal:
                    signal = "sell"
                    entry_type = "crossover"

            if signal is None and regime in ["strong_trend", "weak_trend", "uncertain"]:
                pb_signal, pb_atr = self._get_pullback_signal(i)
                if pb_signal:
                    signal = pb_signal
                    entry_type = "pullback"
                    cur_atr = pb_atr

            mr_atr = None
            if regime in ["ranging"] and p.get("mr_enabled", True):
                mr_sig, mr_atr = self._get_mean_reversion_signal(i)
                if mr_sig:
                    signal = mr_sig
                    entry_type = "mean_reversion"
                    cur_atr = mr_atr if mr_atr else cur_atr

            # Gate 1 — Fused Regime (replaces ER chop + exec-bias + volatility).
            # Uses ADX, ER, and MA slope to produce a single gate-open/closed decision.
            # gate_state was already advanced every bar above; here we only apply it.
            if signal is not None and not gate_state and entry_type == "crossover":
                # Gate closed: only allow MR or pullback
                signal = None

            sl = None
            tp = None
            # Reset per-bar so a value from an earlier iteration can never leak
            # into this bar's pending_entry (see the chandelier note above).
            sl_points = None
            tp_points = 0
            entry_atr = cur_atr
            if signal:
                if entry_type == "mean_reversion":
                    mr_sl_mult = p.get("mr_sl_atr_mult", 1.0)
                    mr_tp_mult = p.get("mr_tp_atr_mult", 1.5)
                    sl_points = max(
                        int(entry_atr * mr_sl_mult / max(self.point, 1e-10)), int(self.p.get("stops_level", 50))
                    )
                    tp_points = int(entry_atr * mr_tp_mult / max(self.point, 1e-10))
                else:
                    atr_sl_mult = p.get("atr_sl_mult", 1.0)
                    rr = p.get("rr", 2.0)
                    sl_points = max(
                        int(entry_atr * atr_sl_mult / max(self.point, 1e-10)), int(self.p.get("stops_level", 50))
                    )
                    tp_points = int(sl_points * rr)
                if signal == "buy":
                    sl = bar["close"] - sl_points * self.point
                    tp = bar["close"] + tp_points * self.point
                else:
                    sl = bar["close"] + sl_points * self.point
                    tp = bar["close"] - tp_points * self.point

            open_positions = [pos for pos in self.positions if pos["status"] == "open"]

            for pos in open_positions:
                # Skip exit evaluation on the entry bar: training labels only
                # start checking SL/TP from the next bar (j >= 1), so a position
                # must survive at least one full bar before it can be exited.
                if pos.get("entry_bar") == i:
                    continue
                is_long = pos["type"] == "buy"
                rem_vol = pos.get("remaining_volume", pos["volume"])
                hit_sl = (is_long and bar["low"] <= pos["sl"]) or (not is_long and bar["high"] >= pos["sl"])
                hit_tp = pos["tp"] is not None and (
                    (is_long and bar["high"] >= pos["tp"]) or (not is_long and bar["low"] <= pos["tp"])
                )

                if hit_sl or hit_tp:
                    exit_price = pos["sl"] if hit_sl else pos["tp"]
                    bar_spread_s = bar.get("spread", 0) or 0
                    if is_long:
                        exit_price -= spm * bar_spread_s * self.point * 0.5 + self.slippage * self.point
                    else:
                        exit_price += spm * bar_spread_s * self.point * 0.5 + self.slippage * self.point
                    pnl = (
                        (exit_price - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
                        if is_long
                        else (pos["entry"] - exit_price) / max(self.point, 1e-10) * self.tick_value * rem_vol
                    )
                    if commission > 0:
                        pnl -= commission * rem_vol
                        self.commission_paid += commission * rem_vol
                    cumulative_pnl += pnl
                    pos["exit"] = exit_price
                    pos["exit_bar"] = i
                    pos["pnl"] = pnl
                    pos["status"] = "closed"
                    pos["exit_reason"] = "SL" if hit_sl else "TP"
                    self.trades.append(pos)
                    self._register_close(pos)
                    continue

                if p.get("chandelier_enabled", True):
                    entry_bar = pos["entry_bar"]
                    ch_mult = p.get("chandelier_mult_overrides", {}).get(
                        pos.get("symbol", ""), p.get("chandelier_mult", 3.0)
                    )
                    is_partial = pos.get("partial_fired", False)
                    if p.get("ch_two_stage", True) and not is_partial:
                        # NOTE: must NOT be named `sl_points` — that name is also
                        # the entry-sizing local used to build `pending_entry`
                        # further down this same bar iteration. Shadowing it here
                        # leaked this float into a later entry's `sl_points`,
                        # corrupting that position's stop distance (measured: 3 of
                        # 328 entries on XAUUSD, e.g. bar 7682 got 0.31*ATR instead
                        # of the configured 1.5*ATR).
                        ch_sl_points = abs(pos["entry"] - pos["sl"]) / max(self.point, 1e-10)
                        pnl_points = (
                            (bar["close"] - pos["entry"]) / max(self.point, 1e-10)
                            if is_long
                            else (pos["entry"] - bar["close"]) / max(self.point, 1e-10)
                        )
                        r_mult = pnl_points / max(ch_sl_points, 1)
                        if r_mult >= p.get("ch_two_stage_min_r", 3.0):
                            ch_mult = p.get("ch_tight_mult", 1.5)
                        else:
                            ch_mult = p.get("ch_loose_mult", 3.5)
                    if is_partial:
                        ch_mult = p.get("chandelier_mult_partial", 1.5)
                    if p.get("ch_accelerate_enabled", False) and self.ch_accel_ema is not None and i >= 5:
                        ema_vals = self.ch_accel_ema.values
                        ema_bars = p.get("ch_accelerate_bars", 5)
                        if i >= ema_bars:
                            ema_ratio = float(ema_vals[i]) / max(float(ema_vals[i - ema_bars]), 1e-10)
                            strength = p.get("ch_accelerate_strength", 0.20)
                            if ema_ratio > 1.0:
                                accel = 1.0 - strength * min(ema_ratio - 1.0, 1.0)
                            else:
                                accel = 1.0 + strength * min(1.0 - ema_ratio, 1.0)
                            accel = max(0.5, min(1.5, accel))
                            ch_mult = ch_mult * accel
                    if is_long:
                        hh = df["high"].iloc[entry_bar : i + 1].max()
                        new_sl = hh - cur_atr * ch_mult
                        new_sl = max(new_sl, pos.get("ch_sl", 0))
                        if new_sl > pos["sl"]:
                            pos["sl"] = new_sl
                        pos["ch_sl"] = new_sl
                        new_hit = bar["low"] <= pos["sl"]
                    else:
                        ll = df["low"].iloc[entry_bar : i + 1].min()
                        new_sl = ll + cur_atr * ch_mult
                        new_sl = min(new_sl, pos.get("ch_sl", float("inf")))
                        if new_sl < pos["sl"]:
                            pos["sl"] = new_sl
                        pos["ch_sl"] = new_sl
                        new_hit = bar["high"] >= pos["sl"]
                    if new_hit:
                        exit_price = pos["sl"]
                        bar_spread_s = bar.get("spread", 0) or 0
                        if is_long:
                            exit_price -= spm * bar_spread_s * self.point * 0.5 + self.slippage * self.point
                        else:
                            exit_price += spm * bar_spread_s * self.point * 0.5 + self.slippage * self.point
                        pnl = (
                            (exit_price - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
                            if is_long
                            else (pos["entry"] - exit_price) / max(self.point, 1e-10) * self.tick_value * rem_vol
                        )
                        if commission > 0:
                            pnl -= commission * rem_vol
                            self.commission_paid += commission * rem_vol
                        cumulative_pnl += pnl
                        pos["exit"] = exit_price
                        pos["exit_bar"] = i
                        pos["pnl"] = pnl
                        pos["status"] = "closed"
                        pos["exit_reason"] = "CHANDELIER"
                        self.trades.append(pos)
                        self._register_close(pos)
                        continue

                if p.get("scale_out_enabled", True):
                    entry = pos["entry"]
                    close_fracs = p.get("scale_out_close_fractions", [0.20, 0.20])
                    tp_rr_targets = p.get("scale_out_tp_targets_rr", [0.50, 0.75])
                    num_partials = len(close_fracs)
                    is_mr_pos = pos.get("entry_type") == "mean_reversion"
                    atr_sl_mult = p.get("atr_sl_mult", 1.0)
                    bt_rr = p.get("rr", 2.0)
                    mr_sl_mult = p.get("mr_sl_atr_mult", 1.0)
                    mr_tp_mult = p.get("mr_tp_atr_mult", 1.5)
                    while pos.get("scale_step", 0) < num_partials and rem_vol > 0:
                        step = pos.get("scale_step", 0)
                        target_fraction = tp_rr_targets[step] if step < len(tp_rr_targets) else tp_rr_targets[-1]
                        bt_atr_entry = pos.get("atr_entry") or cur_atr
                        tp_sl_mult = mr_sl_mult if is_mr_pos else atr_sl_mult
                        tp_rr_mult = mr_tp_mult if is_mr_pos else bt_rr
                        tp_dist = bt_atr_entry / tp_sl_mult * tp_rr_mult
                        level = entry + tp_dist * target_fraction if is_long else entry - tp_dist * target_fraction
                        hit = bar["high"] >= level if is_long else bar["low"] <= level
                        if not hit:
                            break
                        close_frac = close_fracs[step]
                        vol_step = p.get("volume_step", 0.01)
                        close_vol = max(int(pos["volume"] * close_frac / vol_step) * vol_step, vol_step)
                        close_vol = min(close_vol, rem_vol)
                        pnl_part = (
                            (level - entry) / max(self.point, 1e-10) * self.tick_value * close_vol
                            if is_long
                            else (entry - level) / max(self.point, 1e-10) * self.tick_value * close_vol
                        )
                        cumulative_pnl += pnl_part
                        if commission > 0:
                            self.commission_paid += commission * close_vol
                            # Each partial close pays exit-side commission, just
                            # like the live broker. Previously only the final
                            # remainder was charged, understating cost on scaled
                            # positions (backtest parity: partial commission).
                            cumulative_pnl -= commission * close_vol
                        rem_vol -= close_vol
                        pos["remaining_volume"] = rem_vol
                        pos["scale_step"] = step + 1
                        partial_trade = {
                            "type": "partial",
                            "entry": pos["entry"],
                            "exit": level,
                            "sl": pos["sl"],
                            "tp": pos["tp"],
                            "pnl": pnl_part,
                            "entry_bar": pos["entry_bar"],
                            "exit_bar": i,
                            "exit_reason": "SCALE_OUT",
                            "regime": pos.get("regime", ""),
                            # Carry the parent position's entry_type/volume so a
                            # partial has the same key set as every other trade
                            # record. Without these, consumers that index into
                            # self.trades hit a KeyError whenever a scale-out
                            # happens to be the first trade.
                            "entry_type": pos.get("entry_type", ""),
                            "volume": close_vol,
                        }
                        self.trades.append(partial_trade)
                        if step == 0:
                            # Lock at +0.25R (scale_out_breakeven_fraction) instead of
                            # hard breakeven — gives runner room. Matches live execution.py.
                            lock_frac = p.get("scale_out_breakeven_fraction", 0.25)
                            lock_level = entry + tp_dist * lock_frac if is_long else entry - tp_dist * lock_frac
                            pos["sl"] = lock_level
                        elif is_long:
                            lock_fraction = (
                                tp_rr_targets[step - 1] if step - 1 < len(tp_rr_targets) else tp_rr_targets[-1]
                            )
                            lock_level = entry + tp_dist * lock_fraction
                            pos["sl"] = max(pos["sl"], lock_level)
                        else:
                            lock_fraction = (
                                tp_rr_targets[step - 1] if step - 1 < len(tp_rr_targets) else tp_rr_targets[-1]
                            )
                            lock_level = entry - tp_dist * lock_fraction
                            pos["sl"] = min(pos["sl"], lock_level)

                    if pos.get("scale_step", 0) >= num_partials:
                        pos["partial_fired"] = True
                        pos["tp"] = None  # Remove TP, let remainder run on chandelier

                # Mean reversion exit (RSI crossing 50)
                mr_exit = False
                if regime == "ranging" and p.get("mr_enabled", True) and self._check_mean_reversion_exit(i, pos):
                    mr_exit = True

                # Gate reversal: only flip if not in substantial profit (>+0.5R) to protect runners
                in_sub_profit = (is_long and bar["close"] > pos["entry"] + cur_atr * 0.25) or (
                    not is_long and bar["close"] < pos["entry"] - cur_atr * 0.25
                )
                opp_signal = (
                    (mr_exit or ((is_long and sell_signal) or (not is_long and buy_signal)))
                    and signal is not None
                    and rem_vol > 0
                    and (mr_exit or not in_sub_profit)
                )
                if opp_signal and rem_vol > 0:
                    bar_spread = bar.get("spread", 0) or 0
                    if is_long:
                        exit_price = bar["close"] - spm * bar_spread * self.point * 0.5 - self.slippage * self.point
                    else:
                        exit_price = bar["close"] + spm * bar_spread * self.point * 0.5 + self.slippage * self.point
                    pnl = (
                        (exit_price - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
                        if is_long
                        else (pos["entry"] - exit_price) / max(self.point, 1e-10) * self.tick_value * rem_vol
                    )
                    if commission > 0:
                        pnl -= commission * rem_vol
                        self.commission_paid += commission * rem_vol
                    cumulative_pnl += pnl
                    pos["exit"] = exit_price
                    pos["exit_bar"] = i
                    pos["pnl"] = pnl
                    pos["status"] = "closed"
                    pos["exit_reason"] = "MR_EXIT" if mr_exit else "REVERSAL"
                    self.trades.append(pos)
                    self._register_close(pos)
                    continue

            open_positions = [pos for pos in self.positions if pos["status"] == "open"]

            # Only skip exhaustion - uncertain is now allowed (reduced conviction)
            if regime == "exhaustion":
                self.equity.append(self.initial_balance + cumulative_pnl)
                continue

            # Max positions per symbol
            max_per_sym = p.get("max_positions_per_symbol", 999)
            if len(open_positions) >= max_per_sym:
                self.equity.append(self.initial_balance + cumulative_pnl)
                continue

            if len(open_positions) == 0 and signal is not None and i < n - 1:
                # === Gate 4 — Execution Sanity (volume + spread merged) ===
                if not self._check_execution_sanity(i, signal):
                    self.equity.append(self.initial_balance + cumulative_pnl)
                    continue

                # HTF trend gate (parity with live signals.check_htf_trend):
                if not p.get("mtf_enabled", False):
                    htf_decision, htf_size_mult = self._check_htf_trend(i, signal)
                    if htf_decision == "block":
                        self.equity.append(self.initial_balance + cumulative_pnl)
                        continue

                if not self._check_session_time(i):
                    self.equity.append(self.initial_balance + cumulative_pnl)
                    continue

                if not self._check_daily_loss(i, cumulative_pnl):
                    self.equity.append(self.initial_balance + cumulative_pnl)
                    continue

                if not self._check_tail_risk(i):
                    self.equity.append(self.initial_balance + cumulative_pnl)
                    continue

                # === Gate 3 — ML Validation (merged ML + scoring + news) ===
                ml_mult = 1.0
                confidence_mult = 1.0
                if p.get("ml_enabled", False):
                    ml_mult = self._check_ml_signal(i, signal)
                    if ml_mult <= 0:
                        self.equity.append(self.initial_balance + cumulative_pnl)
                        continue
                if p.get("scoring_enabled", True):
                    # Raw-value seam: analytics owns the scoring math; the
                    # backtest supplies per-bar inputs (ml mult, bar spread in
                    # price units, stateful tail risk).
                    entry_score, score_details, _ = compute_entry_score(
                        p, signal, entry_atr,
                        spread=(float(df.iloc[i].get("spread") or 0) * self.point)
                        if p.get("spf_enabled", True) else None,
                        ml_conf=ml_mult if p.get("ml_enabled", False) else None,
                        tail_risk=self._tail_risk_score(),
                    )
                    mr_min = 0.03 if entry_atr is None else 0.0
                    min_score = p.get("scoring_min_entry", 0.60) + mr_min
                    if entry_score < min_score:
                        self.equity.append(self.initial_balance + cumulative_pnl)
                        continue
                    high_bucket = p.get("scoring_confidence_bucket_high", 0.85)
                    low_bucket = p.get("scoring_confidence_bucket_low", 0.60)
                    if entry_score >= high_bucket:
                        confidence_mult = p.get("scoring_high_conviction_mult", 1.0)
                    elif entry_score >= low_bucket:
                        confidence_mult = p.get("scoring_standard_edge_mult", 0.85)
                    else:
                        confidence_mult = p.get("scoring_low_conviction_mult", 0.50)
                    news_val = score_details.get("news", 0.5) if score_details else 0.5
                    confidence_mult = apply_news_confidence_mult(confidence_mult, news_val)

                current_equity = self.initial_balance + cumulative_pnl
                sl_price_dist = abs(bar["close"] - sl)
                sl_value = sl_price_dist * self.tick_value / max(self.point, 1e-10)
                risk_amount = max(current_equity * (self.risk_percent / 100), 0.0)
                vol_step = p.get("volume_step", 0.01)
                raw_volume = risk_amount / max(sl_value, 1e-10)
                if raw_volume < vol_step:
                    min_lot_risk = vol_step * sl_value
                    risk_ratio = min_lot_risk / max(risk_amount, 1e-10)
                    max_risk_ratio = p.get("max_risk_ratio", 2.0)
                    if risk_ratio > max_risk_ratio:
                        self.equity.append(self.initial_balance + cumulative_pnl)
                        continue
                if p.get("mtf_enabled", False):
                    volume = raw_volume * max(0.5, mtf_confidence)
                else:
                    volume = raw_volume * htf_size_mult
                if entry_type == "mean_reversion":
                    regime_mult = p.get("mr_position_size_mult", 0.5)
                else:
                    if regime == "strong_trend":
                        regime_mult = 1.0
                    elif regime == "uncertain":
                        regime_mult = 0.5
                    else:
                        regime_mult = 0.75
                kelly_mult = self._calc_kelly_mult()
                vol_mult = self._calc_volatility_mult(i)
                volume *= regime_mult * kelly_mult * vol_mult * ml_mult * confidence_mult
                # Correlation sizing reduction (A3) — mirrors main.py:547
                # _apply_corr_ml_sizing. Backtest is single-symbol, so the
                # reduction is supplied externally via corr_size_mult when
                # correlation is enabled; defaults to 1.0 (no effect).
                if p.get("correlation_enabled", False):
                    volume *= p.get("corr_size_mult", 1.0)
                volume = max(round(volume / vol_step, 0) * vol_step, vol_step)

                pending_entry = {
                    "type": signal,
                    "entry_type": entry_type,
                    "sl": sl,
                    "tp": tp,
                    "volume": volume,
                    "regime": regime,
                    "atr_entry": entry_atr,
                    "sl_points": sl_points,
                    "tp_points": tp_points,
                    "h1_bar": i + 1,
                }
                self.equity.append(self.initial_balance + cumulative_pnl)
                continue

            # INTENTIONAL: same-bar open PnL IS included for MTM accuracy even
            # though exits (SL/TP/chandelier/scale-out) are deferred to the next
            # bar (entry_bar != i guard above). This mirrors triple_barrier_labels
            # training convention where positions open at bar i and start exit
            # checks at bar i+1. See NOTE at position-open site (lines 1134-1139).
            open_pnl = 0.0
            for pos in self.positions:
                if pos["status"] != "open":
                    continue
                rem_vol = pos.get("remaining_volume", pos["volume"])
                if rem_vol <= 0:
                    continue
                if pos["type"] == "buy":
                    open_pnl += (bar["close"] - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
                else:
                    open_pnl += (pos["entry"] - bar["close"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
            self.equity.append(self.initial_balance + cumulative_pnl + open_pnl)

        last_bar = self.df.iloc[-1]
        last_spread = last_bar.get("spread", 0) or 0
        for pos in self.positions:
            if pos["status"] == "open":
                is_long = pos["type"] == "buy"
                rem_vol = pos.get("remaining_volume", pos["volume"])
                if is_long:
                    exit_price = last_bar["close"] - spm * last_spread * self.point * 0.5 - self.slippage * self.point
                else:
                    exit_price = last_bar["close"] + spm * last_spread * self.point * 0.5 + self.slippage * self.point
                pnl = (
                    (exit_price - pos["entry"]) / max(self.point, 1e-10) * self.tick_value * rem_vol
                    if is_long
                    else (pos["entry"] - exit_price) / max(self.point, 1e-10) * self.tick_value * rem_vol
                )
                if commission > 0:
                    pnl -= commission * rem_vol
                    self.commission_paid += commission * rem_vol
                cumulative_pnl += pnl
                pos["exit"] = exit_price
                pos["exit_bar"] = n - 1
                pos["pnl"] = pnl
                pos["status"] = "closed"
                pos["exit_reason"] = "END"
                self.trades.append(pos)
                self._register_close(pos)

        self._report()
        return self.get_results()

    def _run_fast(self):
        """Numba-JIT fast path mirroring _run_reference bit-for-bit.

        Builds plain numpy arrays from the precomputed attributes and calls the
        compiled state machine in backtest_njit._simulate_core. Produces the
        same self.trades / self.equity / self.stats as the reference loop.
        """
        from backtest_njit import (
            _ENTRY_TYPE_NAME,
            _EXIT_REASON_NAME,
            _REGIME_NAME,
            _simulate_core,
        )

        p = self.p
        df = self.df
        n = self.n
        warmup = max(p.get("ema_fast", 8), p["ema_slow"], p.get("atr_period", 14), 250) + 50

        def f(x):
            return float(x) if x is not None else 0.0

        def b(x):
            return 1.0 if x else 0.0

        def _arr(x, default=None):
            if x is None:
                return np.full(n, np.nan) if default is None else default
            return np.asarray(x, dtype=float)

        # The njit core carries exactly two scale-out RR targets
        # (P_scale_out_tp0/tp1) and two close fractions. A longer ladder would be
        # silently truncated there while _run_reference honours every entry, so
        # refuse the fast path instead of quietly diverging.
        _tp_targets = p.get("scale_out_tp_targets_rr", [0.50, 0.75])
        _close_fracs = p.get("scale_out_close_fractions", [0.20, 0.20])
        if len(_tp_targets) > 2 or len(_close_fracs) > 2:
            raise NotImplementedError(
                f"njit fast path supports at most 2 scale-out steps "
                f"(got {len(_tp_targets)} rr targets, {len(_close_fracs)} close fractions); "
                f"use run(fast=False)"
            )

        # --- MR RSI series (bit-exact with reference _get_mean_reversion_signal) ---
        # Reference uses the M30-based RSI (mr_rsi_h1, time-aligned to H1) when
        # available, else falls back to the H1 Wilder RSI. Mirror that exactly.
        mr_period = int(p.get("mr_rsi_period", 14))
        mr_rsi_a = np.full(n, np.nan)
        if self.mr_rsi_h1 is not None and len(self.mr_rsi_h1) == n:
            for i in range(n):
                v = self.mr_rsi_h1.iloc[i]
                mr_rsi_a[i] = v if not pd.isna(v) else self._calc_rsi_wilder(i, mr_period)
        elif p.get("mr_enabled", True):
            for i in range(n):
                mr_rsi_a[i] = self._calc_rsi_wilder(i, mr_period)

        # --- scalar param bundle ---
        h4_adx_a = getattr(self, "h4_adx_aligned", np.full(n, np.nan))
        d1_adx_a = getattr(self, "d1_adx_aligned", np.full(n, np.nan))
        htf_ema_a = _arr(self.htf_ema_aligned)
        # The reference reads htf_close_aligned (H4 close, ffilled to H1) and
        # only falls back to the H1 bar close when that series is absent. An
        # all-NaN array encodes "absent" for the njit twin.
        htf_close_a = _arr(self.htf_close_aligned)
        htf_slope_a = _arr(self.htf_slope_aligned)
        ml_buy_a = _arr(self.ml_mult_buy)
        ml_sell_a = _arr(self.ml_mult_sell)
        # Meta-labeler parity: the reference gates every ML decision through
        # _check_ml_signal, which zeroes the multiplier when the meta-model's
        # confidence is below ml_meta_threshold. The njit core reads these
        # arrays directly and has no meta stage, so bake the veto in here —
        # symbols with a .meta.pkl (EURUSD, GBPJPY, US500) otherwise trade
        # setups the reference rejects.
        if self.ml_meta_model is not None and self.ml_meta_features is not None and self.ml_meta_proba is not None:
            meta_threshold = self.p.get("ml_meta_threshold", 0.50)
            m = min(n, len(self.ml_meta_proba))
            meta_block = np.zeros(n, dtype=bool)
            meta_block[:m] = np.asarray(self.ml_meta_proba[:m], dtype=float) < meta_threshold
            ml_buy_a = np.where(meta_block, 0.0, ml_buy_a)
            ml_sell_a = np.where(meta_block, 0.0, ml_sell_a)
        vol_sma_a = _arr(self.vol_sma)
        atr_sma_a = _arr(self.atr_sma)
        atr_sma50_a = _arr(self.atr_sma50)
        ch_accel_a = _arr(self.ch_accel_ema)
        day_start_idx_a = getattr(self, "day_start_idx", np.arange(n))
        bar_hour_a = (
            df["time"].dt.hour.values if hasattr(df["time"].dt, "hour") else np.zeros(n, dtype=np.int64)
        ).astype(np.int64)

        # Fused regime score array for the Numba fast path (Gate 1). Precomputed
        # in _precompute as self.fused_score_a.
        fused_score_a = _arr(self.fused_score_a)

        # MTF aligned MA arrays (all-NaN when mtf_enabled is False).
        mtf_m15_fast_a = _arr(self.mtf_m15_fast)
        mtf_m15_slow_a = _arr(self.mtf_m15_slow)
        mtf_h4_ema_a = _arr(self.mtf_h4_ema)

        def _call_core(score_a, conf_mult_a):
            return _simulate_core(
                n,
                warmup,
                f(self.point),
                f(self.tick_value),
                f(self.slippage),
                f(p.get("spread_model", 1.0)),
                f(p.get("commission", 0.0)),
                f(p.get("volume_step", 0.01)),
                f(self.initial_balance),
                f(p.get("risk_percent", 1.0)),
                f(p.get("max_risk_ratio", 2.0)),
                f(p.get("stops_level", 50)),
                df["open"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
                df["close"].values.astype(float),
                (df["spread"].values if "spread" in df else np.zeros(n)).astype(float),
                df["tick_volume"].values.astype(float),
                day_start_idx_a.astype(np.int64),
                _arr(self.ema_fast),
                _arr(self.ema_slow),
                _arr(self.atr_series),
                _arr(self.adx_series),
                h4_adx_a,
                d1_adx_a,
                htf_ema_a,
                htf_close_a,
                htf_slope_a,
                ml_buy_a,
                ml_sell_a,
                vol_sma_a,
                atr_sma_a,
                atr_sma50_a,
                ch_accel_a,
                mr_rsi_a,
                fused_score_a,
                mtf_m15_fast_a,
                mtf_m15_slow_a,
                mtf_h4_ema_a,
                bar_hour_a,
                # scalar params
                f(p.get("atr_sl_mult", 1.0)),
                f(p.get("rr", 2.0)),
                f(p.get("mr_sl_atr_mult", 1.0)),
                f(p.get("mr_tp_atr_mult", 1.5)),
                b(p.get("chandelier_enabled", True)),
                f(p.get("chandelier_mult", 3.0)),
                b(p.get("ch_two_stage", True)),
                f(p.get("ch_tight_mult", 1.5)),
                f(p.get("ch_loose_mult", 3.5)),
                f(p.get("chandelier_mult_partial", 1.5)),
                b(p.get("ch_accelerate_enabled", False)),
                f(p.get("ch_accelerate_bars", 5)),
                f(p.get("ch_accelerate_strength", 0.20)),
                f(p.get("ch_two_stage_min_r", 3.0)),
                b(p.get("scale_out_enabled", True)),
                f(p.get("scale_out_close_fractions", [0.20, 0.20])[0]),
                f(p.get("scale_out_close_fractions", [0.20, 0.20])[1]),
                f(p.get("scale_out_tp_targets_rr", [0.50, 0.75])[0]),
                f(p.get("scale_out_tp_targets_rr", [0.50, 0.75])[1]),
                f(p.get("scale_out_breakeven_fraction", 0.25)),
                b(p.get("mr_enabled", True)),
                f(mr_period),
                f(p.get("mr_rsi_oversold", 30)),
                f(p.get("mr_rsi_overbought", 70)),
                f(p.get("mr_htf_deviation", 0.0)),
                f(p.get("mr_position_size_mult", 0.5)),
                b(p.get("mr_cooldown_enabled", False)),
                f(p.get("mr_cooldown_bars", 2)),
                b(p.get("pb_enabled", True)),
                f(p.get("pb_atr_mult", 2.0)),
                f(p.get("pb_volume_threshold", 0.8)),
                f(p.get("pb_volume_sma_period", 20)),
                f(p.get("pb_structure_lookback", 5)),
                f(p.get("pb_atr_min_dist", 0.1)),
                b(p.get("volume_filter", False)),
                f(p.get("volume_kappa", 1.2)),
                f(p.get("vf_obv_lookback", 20)),
                b(p.get("vf_obv_enabled", True)),
                b(p.get("spf_enabled", True)),
                f(p.get("spf_max_ratio", 0.30)),
                b(p.get("session_enabled", False)),
                b(p.get("session_only", False)),
                b(p.get("require_overlap", False)),
                b(p.get("skip_asian", False)),
                f(p.get("london_open", 13)),
                f(p.get("london_close", 22)),
                f(p.get("asian_open", 5)),
                f(p.get("asian_close", 12)),
                f(p.get("adx_trend_threshold", 25)),
                f(p.get("adx_range_threshold", 20)),
                f(p.get("exhaustion_adx_threshold", 40)),
                f(p.get("exhaustion_slope_threshold", 2.0)),
                f(p.get("fused_threshold", 50.0)),
                f(p.get("fused_buffer", 5.0)),
                b(p.get("ml_enabled", False)),
                b(p.get("scoring_enabled", True)),
                f(p.get("scoring_min_entry", 0.60)),
                f(p.get("scoring_confidence_bucket_high", 0.85)),
                f(p.get("scoring_confidence_bucket_low", 0.60)),
                f(p.get("scoring_high_conviction_mult", 1.0)),
                f(p.get("scoring_standard_edge_mult", 0.85)),
                f(p.get("scoring_low_conviction_mult", 0.50)),
                b(p.get("dr_enabled", True)),
                f(p.get("dr_lookback", 50)),
                f(p.get("dr_kelly_fraction", 0.25)),
                f(p.get("dr_min_mult", 0.25)),
                f(p.get("dr_max_mult", 1.5)),
                b(p.get("dr_vol_adjust", True)),
                b(p.get("tr_enabled", True)),
                f(p.get("tr_lookback", 50)),
                f(p.get("tr_sigma", 3.0)),
                f(p.get("tr_max_dd_pct", 8.0)),
                f(p.get("cb_dd_pct", 15.0)),
                f(p.get("daily_loss_pct", 5.0)),
                # Must match the reference default at _run_reference (999 =
                # effectively unlimited). Passing 1 here made the fast path
                # block re-entries the reference allowed whenever the key was
                # absent from params.
                f(p.get("max_positions_per_symbol", 999)),
                f(p.get("corr_size_mult", 1.0)),
                b(p.get("correlation_enabled", False)),
                f(p.get("htf_misalign_size_mult", 0.5)),
                b(p.get("mtf_enabled", False)),
                score_a,
                conf_mult_a,
            )

        # Two-pass for bit-exact scoring parity:
        #  Pass 1: neutral scoring -> exact sig_out / entry_type from the core.
        #  Pass 2: score_a/conf_mult_a derived via the EXACT reference scoring
        #          (analytics.compute_entry_score, using the core's own signals)
        #          so the entry gate + confidence multiplier match _run_reference.
        #
        # The scoring tail_risk component is STATEFUL (depends on the live equity
        # drawdown at each bar). A single neutral first pass cannot supply correct
        # tail scores, so we iterate to a fixed point: each pass uses the previous
        # pass's equity curve to compute the tail_risk score, re-running the core
        # until the score/confidence arrays stop changing. In practice this
        # converges in 2-4 passes because tail_risk only shifts the discrete
        # confidence buckets.
        sc_high = p.get("scoring_confidence_bucket_high", 0.85)
        sc_low = p.get("scoring_confidence_bucket_low", 0.60)
        sc_hi_mult = p.get("scoring_high_conviction_mult", 1.0)
        sc_std_mult = p.get("scoring_standard_edge_mult", 0.85)
        sc_low_mult = p.get("scoring_low_conviction_mult", 0.50)
        tr_enabled = p.get("tr_enabled", True)
        tr_max_dd = p.get("tr_max_dd_pct", 8.0)
        scoring_on = p.get("scoring_enabled", True)

        score_a = np.ones(n, dtype=float)
        conf_mult_a = np.ones(n, dtype=float)
        prev_score = None
        for _pass in range(6):
            res_k = _call_core(score_a, conf_mult_a)
            sig_out_k = res_k[17]
            res_k[19]
            equity_k = np.asarray(res_k[15], dtype=float)
            if not scoring_on:
                break
            new_score = np.ones(n, dtype=float)
            new_conf = np.ones(n, dtype=float)
            for i in range(warmup, n):
                s = sig_out_k[i]
                if s == -99 or s == -1:
                    continue
                signal = "buy" if s == 0 else "sell"
                tail_override = None
                if tr_enabled:
                    cur_eq = equity_k[i] if i < len(equity_k) else self.initial_balance
                    run_max = self.initial_balance
                    if i > warmup:
                        run_max = max(run_max, float(np.max(equity_k[warmup:i])))
                    tail_override = 1.0
                    if run_max > 0:
                        dd_pct = max(0.0, (run_max - cur_eq) / run_max) * 100
                        tail_override = 0.0 if dd_pct >= tr_max_dd else 1.0 - (dd_pct / tr_max_dd)
                comp, _, _ = compute_entry_score(
                    self.p, signal, self.atr_series.iloc[i],
                    spread=(float(self.df.iloc[i].get("spread") or 0) * self.point)
                    if self.p.get("spf_enabled", True) else None,
                    ml_conf=self._check_ml_signal(i, signal) if self.p.get("ml_enabled", False) else None,
                    tail_risk=tail_override if tail_override is not None else self._tail_risk_score(),
                )
                new_score[i] = comp
                if comp >= sc_high:
                    new_conf[i] = sc_hi_mult
                elif comp >= sc_low:
                    new_conf[i] = sc_std_mult
                else:
                    new_conf[i] = sc_low_mult
            if prev_score is not None and np.allclose(new_score, prev_score, atol=1e-9):
                score_a, conf_mult_a = new_score, new_conf
                break
            score_a, conf_mult_a = new_score, new_conf
            prev_score = new_score.copy()
        else:
            # Exhausted the pass budget without the score array settling. The
            # loop previously fell through silently and shipped the last pass as
            # if it were the fixed point, so a non-converging window looked
            # identical to a converged one.
            if scoring_on:
                logging.warning(
                    "[backtest] fast-path scoring did not converge in 6 passes; "
                    "entry scores may differ from the reference loop"
                )

        res = _call_core(score_a, conf_mult_a)
        self._conf_mult_a_debug = conf_mult_a.copy()
        self._score_a_debug = score_a.copy()

        (
            t_count,
            t_type,
            t_entry,
            t_sl,
            t_tp,
            t_entry_bar,
            t_exit_bar,
            t_exit,
            t_pnl,
            t_volume,
            t_rem_vol,
            t_exit_reason,
            t_regime,
            t_entry_type,
            t_atr_entry,
            equity,
            dbg,
            sig_out,
            pend_sl_dbg,
            skip_low_dbg,
            et_out,
        ) = res
        self._dbg = dbg
        self._sig_out = sig_out
        self._et_out = et_out
        self._pend_sl_dbg = pend_sl_dbg
        self._skip_low_dbg = skip_low_dbg

        # Reconstruct trade dicts identically to the reference loop.
        trades = []
        for k in range(t_count):
            is_partial = t_type[k] == 2
            trades.append(
                {
                    "type": "partial" if is_partial else ("buy" if t_type[k] == 0 else "sell"),
                    "entry": float(t_entry[k]),
                    "sl": float(t_sl[k]),
                    "tp": float(t_tp[k]),
                    "entry_bar": int(t_entry_bar[k]),
                    "exit_bar": int(t_exit_bar[k]),
                    "exit": float(t_exit[k]),
                    "pnl": float(t_pnl[k]),
                    "volume": float(t_volume[k]),
                    "remaining_volume": float(t_rem_vol[k]) if is_partial else float(t_volume[k]),
                    "status": "closed",
                    "exit_reason": _EXIT_REASON_NAME[int(t_exit_reason[k])],
                    "regime": _REGIME_NAME[int(t_regime[k])],
                    "entry_type": _ENTRY_TYPE_NAME[int(t_entry_type[k])],
                    "atr_entry": float(t_atr_entry[k]),
                    "symbol": p.get("symbol", ""),
                }
            )
        self.trades = trades
        # Match the reference loop's equity length: the reference only appends
        # equity from `warmup` onward, so drop the pre-warmup (zero) prefix.
        self.equity = list(equity[warmup:])
        self.commission_paid = 0.0  # not tracked in fast path; acceptable for metrics
        self._report()
        return self.get_results()

    def _report(self):
        trades = self.trades
        if not trades:
            print("\nNo trades generated.")
            return

        pnls = [t["pnl"] for t in trades]
        total_pnl = sum(pnls)
        eq_arr = np.array(self.equity) if len(self.equity) > 0 else np.array([self.initial_balance])
        running_max = np.maximum.accumulate(eq_arr)
        dd = running_max - eq_arr
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if sum(losses) != 0 else float("inf")

        stats = {
            "trades": len(trades),
            "return": total_pnl,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "sharpe": 0,
            "calmar": 0,
        }

        if len(pnls) > 1:
            eq_daily = eq_arr[::6]
            if len(eq_daily) > 1:
                # Percentage returns, not money differences.
                daily_ret = np.diff(eq_daily) / eq_daily[:-1]
                std_daily = np.std(daily_ret, ddof=1) if len(daily_ret) > 1 else 1
                mean_daily = np.mean(daily_ret) if len(daily_ret) > 1 else 0
                # 4 samples per day (24 H1 bars / 6), so 4*252 periods per year.
                stats["sharpe"] = (mean_daily / std_daily) * np.sqrt(252 * 4) if std_daily > 0 else 0
            else:
                stats["sharpe"] = 0
            stats["calmar"] = abs(total_pnl / max_dd) if max_dd > 0 else 0

        self.stats = stats

        by_regime = {}
        for t in trades:
            r = t.get("regime", "unknown")
            if r not in by_regime:
                by_regime[r] = {"trades": 0, "wins": 0, "pnl": 0}
            by_regime[r]["trades"] += 1
            by_regime[r]["pnl"] += t["pnl"]
            if t["pnl"] > 0:
                by_regime[r]["wins"] += 1

        print(f"\n{'=' * 60}")
        print("  BACKTEST RESULTS")
        print(f"{'=' * 60}")
        print(f"  Period:           {self.df['time'].iloc[0].date()} to {self.df['time'].iloc[-1].date()}")
        print(f"  Total Trades:     {stats['trades']}")
        print(f"  Win Rate:         {stats['win_rate'] * 100:.1f}%")
        print(f"  Profit Factor:    {stats['profit_factor']:.2f}")
        print(f"  Total Return:     Rs.{stats['return']:+.2f}")
        print(f"  Max Drawdown:     Rs.{stats['max_dd']:.2f}")
        print(f"  Sharpe (ann):     {stats['sharpe']:.2f}")
        print(f"  Calmar Ratio:     {stats['calmar']:.2f}")
        print(f"  Avg Win:          Rs.{stats['avg_win']:.2f}")
        print(f"  Avg Loss:         Rs.{stats['avg_loss']:.2f}")

        if by_regime:
            print("\n  Per-Regime Breakdown:")
            for r, d in sorted(by_regime.items()):
                wr = d["wins"] / d["trades"] * 100 if d["trades"] > 0 else 0
                print(f"    {r:<12}: {d['trades']} trades, {wr:.0f}% WR, Rs.{d['pnl']:+.2f} PnL")

        exit_reasons = {}
        for t in trades:
            r = t.get("exit_reason", "UNKNOWN")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1
        if exit_reasons:
            print("\n  Exit Reasons:")
            for r, c in sorted(exit_reasons.items()):
                print(f"    {r:<15}: {c}")

    def get_results(self):
        trades = self.trades
        pnls = [t["pnl"] for t in trades]
        total_pnl = sum(pnls)
        eq_arr = np.array(self.equity) if len(self.equity) > 0 else np.array([self.initial_balance])
        running_max = np.maximum.accumulate(eq_arr)
        dd = running_max - eq_arr
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0

        wins = [p for p in pnls if p > 0]
        win_rate = len(wins) / len(pnls) if pnls else 0
        profit_factor = sum(wins) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float("inf")

        return {
            "trades": trades,
            "equity": self.equity,
            "pnls": pnls,
            "total_return": total_pnl,
            "max_dd": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "n_trades": len(trades),
        }


def main():
    parser = argparse.ArgumentParser(description="Doto MT5 Backtesting Engine")
    parser.add_argument("--symbol", type=str, default="ETHUSD.raw")
    parser.add_argument("--years", type=float, default=2.0)
    parser.add_argument("--ema-fast", type=int, default=50)
    parser.add_argument("--ema-slow", type=int, default=200)
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--rr", type=float, default=2.0)
    parser.add_argument("--adx-trend", type=int, default=25)
    parser.add_argument("--ml-confidence", type=float, default=0.50)
    parser.add_argument("--max-positions", type=int, default=4)
    parser.add_argument("--no-ml", action="store_true", help="Disable ML overlay")
    parser.add_argument("--no-volume", action="store_true", help="Disable volume filter")
    parser.add_argument("--no-volatility", action="store_true", help="Disable volatility filter")
    parser.add_argument("--no-chandelier", action="store_true", help="Disable chandelier exit")
    parser.add_argument("--no-partial", action="store_true", help="Disable partial TP")
    parser.add_argument("--risk-percent", type=float, default=1.0, help="Risk per trade (%% of balance)")
    parser.add_argument(
        "--initial-balance", type=float, default=400000.0, help="Starting balance for position sizing (PKR)"
    )
    parser.add_argument(
        "--spread-model", type=float, default=1.0, help="Spread multiplier (0=no spread, 1=full spread)"
    )
    parser.add_argument("--commission", type=float, default=976.0, help="Commission per lot per side (PKR)")
    parser.add_argument("--slippage-points", type=int, default=2, help="Slippage in points per trade")
    parser.add_argument(
        "--no-m1-sim", action="store_true", help="Disable M1 bar entry simulation (uses H1 open instead)"
    )
    args = parser.parse_args()

    print(f"Doto MT5 Backtest - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Symbol: {args.symbol} | Years: {args.years}")
    print(f"Params: EMA{args.ema_fast}/{args.ema_slow} SL={args.sl_mult} RR={args.rr} ADX={args.adx_trend}")

    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")

    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        return

    creds = load_credentials()
    authorized = mt5.login(
        creds["account"], password=creds["password"], server=creds["server"],
    )
    if not authorized:
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    mt5.symbol_select(args.symbol, True)
    tf = mt5.TIMEFRAME_H1
    end = datetime.now()
    start = end - timedelta(days=int(args.years * 365))
    rates = mt5.copy_rates_range(args.symbol, tf, start, end)
    if rates is None or len(rates) < 200:
        print(f"Cannot fetch enough data for {args.symbol}")
        mt5.shutdown()
        return

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"Loaded {len(df)} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")

    sinfo = mt5.symbol_info(args.symbol)
    point = sinfo.point if sinfo else 0.01
    tick_value = sinfo.trade_tick_value if sinfo else 0.01
    volume_step = sinfo.volume_step if sinfo and sinfo.volume_step > 0 else 0.01

    df_m1 = None
    if not args.no_m1_sim:
        mt5.symbol_select(args.symbol, True)
        m1_rates = mt5.copy_rates_range(args.symbol, mt5.TIMEFRAME_M1, start, end)
        if m1_rates is not None and len(m1_rates) > 0:
            df_m1 = pd.DataFrame(m1_rates)
            df_m1["time"] = pd.to_datetime(df_m1["time"], unit="s")
            logging.info(f"Loaded {len(df_m1)} M1 bars for M1 entry simulation")

    # A1/A2 parity: load M5 (MTF), M15 (exec-bias gate) and M30 (MR timeframe) bars.
    df_m5 = None
    df_m15 = None
    df_m30 = None
    if not args.no_m1_sim:
        m5_rates = mt5.copy_rates_range(args.symbol, mt5.TIMEFRAME_M5, start, end)
        if m5_rates is not None and len(m5_rates) > 0:
            df_m5 = pd.DataFrame(m5_rates)
            df_m5["time"] = pd.to_datetime(df_m5["time"], unit="s")
            logging.info("Loaded %d M5 bars for MTF", len(df_m5))
        m15_rates = mt5.copy_rates_range(args.symbol, mt5.TIMEFRAME_M15, start, end)
        if m15_rates is not None and len(m15_rates) > 0:
            df_m15 = pd.DataFrame(m15_rates)
            df_m15["time"] = pd.to_datetime(df_m15["time"], unit="s")
            logging.info("Loaded %d M15 bars for exec-bias gate", len(df_m15))
        m30_rates = mt5.copy_rates_range(args.symbol, mt5.TIMEFRAME_M30, start, end)
        if m30_rates is not None and len(m30_rates) > 0:
            df_m30 = pd.DataFrame(m30_rates)
            df_m30["time"] = pd.to_datetime(df_m30["time"], unit="s")
            logging.info("Loaded %d M30 bars for MR timeframe", len(df_m30))

    commission = args.commission

    mt5.shutdown()

    params = {
        "ema_fast": args.ema_fast,
        "ema_slow": args.ema_slow,
        "atr_period": 14,
        "atr_sl_mult": args.sl_mult,
        "rr": args.rr,
        "adx_enabled": True,
        "adx_trend": args.adx_trend,
        "adx_range": 20,
        "stops_level": 50,
        "ml_confidence": args.ml_confidence,
        "ml_threshold_overrides": {},
        "volume_filter": not args.no_volume,
        "volume_kappa": 1.2,
        "volatility_filter": not args.no_volatility,
        "atr_sma_period": 20,
        "chandelier_enabled": not args.no_chandelier,
        "chandelier_mult": 3.0,
        "chandelier_mult_partial": 1.5,
        "chandelier_mult_overrides": {"XAUUSD.raw": 3.5},
        "chandelier_lookback": 14,
        "ch_two_stage": True,
        "ch_loose_mult": 3.5,
        "ch_tight_mult": 1.5,
        "ch_two_stage_min_r": 3.0,
        "ch_accelerate_enabled": True,
        "ch_accelerate_period": 14,
        "ch_accelerate_bars": 5,
        "ch_accelerate_strength": 0.20,
        "scale_out_enabled": not args.no_partial,
        "scale_out_close_fractions": [0.20, 0.20],
        "scale_out_tp_targets_rr": [0.50, 0.75],
        "ml_enabled": not args.no_ml,
        "point": point,
        "tick_value": tick_value,
        "symbol": args.symbol,
        "risk_percent": args.risk_percent,
        "initial_balance": args.initial_balance,
        "spread_model": args.spread_model,
        "commission": commission,
        "slippage_points": args.slippage_points,
        "volume_step": volume_step,
        "dr_enabled": True,
        "dr_lookback": 50,
        "dr_kelly_fraction": 0.25,
        "dr_vol_adjust": True,
        "dr_min_mult": 0.25,
        "dr_max_mult": 1.5,
        "max_positions": 5,
        "max_positions_per_symbol": 1,
        "max_risk_ratio": 2.0,
        "volatility_min_ratio": 0.5,
        "scoring_enabled": True,
        "scoring_min_entry": float(settings.get("SCORING", "min_entry_score", fallback=0.60)),
        "htf_misalign_size_mult": float(settings.get("STRATEGY", "htf_misalign_size_mult", fallback=0.5)),
        "scoring_confidence_bucket_high": 0.85,
        "scoring_confidence_bucket_low": 0.60,
        "scoring_high_conviction_mult": 1.0,
        "scoring_standard_edge_mult": 0.85,
        "scoring_low_conviction_mult": 0.50,
        "scoring_weights": {"ml": 0.40, "spread": 0.30, "news": 0.30},
        "scoring_ml_fallback": 0.60,
        "spf_enabled": True,
        "spf_max_ratio": 0.30,
        "session_enabled": False,
        "session_only": False,
        "require_overlap": False,
        "skip_asian": False,
        "london_open": 13,
        "london_close": 22,
        "asian_open": 5,
        "asian_close": 12,
        "daily_loss_pct": 5.0,
        "tr_enabled": True,
        "tr_sigma": 3.0,
        "tr_lookback": 50,
        "tr_max_dd_pct": 8.0,
        "cb_dd_pct": 15.0,
        "mr_enabled": True,
        "mr_rsi_period": 14,
        "mr_rsi_oversold": 30,
        "mr_rsi_overbought": 70,
        "mr_sl_atr_mult": 1.0,
        "mr_tp_atr_mult": 1.5,
        "mr_position_size_mult": 0.5,
        "mr_htf_deviation": 0.0,
        "mr_cooldown_enabled": True,
        "mr_cooldown_bars": 2,
        "pb_enabled": True,
        "pb_atr_mult": 2.0,
        # Fused regime gate defaults
        "fused_threshold": 50.0,
        "fused_buffer": 5.0,
        # A3 news-in-scoring (mirrors live [NEWS_SENTIMENT]/[SCORING])
        "ns_enabled": True,
        "correlation_enabled": False,
        "corr_size_mult": 1.0,
    }

    # Apply per-symbol strategy overrides from [STRATEGY:SYMBOL] sections
    SYMBOL_OVERRIDE_MAP = {
        "ema_fast_period": ("ema_fast", int),
        "ema_slow_period": ("ema_slow", int),
        "atr_sl_multiplier": ("atr_sl_mult", float),
        "risk_reward_ratio": ("rr", float),
        "atr_sma_period": ("atr_sma_period", int),
        "adx_trend_threshold": ("adx_trend", int),
        "adx_range_threshold": ("adx_range", int),
        "kelly_fraction": ("dr_kelly_fraction", float),
        "risk_percent": ("risk_percent", float),
        "max_positions_per_symbol": ("max_positions_per_symbol", int),
    }
    sym_section = f"STRATEGY:{args.symbol}"
    if settings.has_section(sym_section):
        for ini_key, (cfg_key, converter) in SYMBOL_OVERRIDE_MAP.items():
            if settings.has_option(sym_section, ini_key):
                params[cfg_key] = converter(settings.get(sym_section, ini_key))
                logging.info("  Per-symbol override: %s=%s", cfg_key, params[cfg_key])

    # Per-symbol SCALE_OUT overrides
    so_section = f"SCALE_OUT:{args.symbol}"
    if settings.has_section(so_section):
        if settings.has_option(so_section, "tp_targets_atr"):
            vals = [float(x.strip()) for x in settings.get(so_section, "tp_targets_atr").split(",")]
            params["scale_out_tp_targets_atr"] = vals
            logging.info("  SCALE_OUT override: tp_targets_atr=%s", vals)
        if settings.has_option(so_section, "tp_targets_rr"):
            vals = [float(x.strip()) for x in settings.get(so_section, "tp_targets_rr").split(",")]
            params["scale_out_tp_targets_rr"] = vals
            logging.info("  SCALE_OUT override: tp_targets_rr=%s", vals)
        if settings.has_option(so_section, "close_fractions"):
            vals = [float(x.strip()) for x in settings.get(so_section, "close_fractions").split(",")]
            params["scale_out_close_fractions"] = vals
            logging.info("  SCALE_OUT override: close_fractions=%s", vals)

    # Per-symbol CHANDELIER overrides
    ch_section = f"CHANDELIER:{args.symbol}"
    if settings.has_section(ch_section):
        if settings.has_option(ch_section, "atr_multiplier"):
            v = float(settings.get(ch_section, "atr_multiplier"))
            params["chandelier_mult"] = v
            logging.info("  CHANDELIER override: atr_multiplier=%s", v)
        if settings.has_option(ch_section, "atr_multiplier_partial"):
            v = float(settings.get(ch_section, "atr_multiplier_partial"))
            params["chandelier_mult_partial"] = v
            logging.info("  CHANDELIER override: atr_multiplier_partial=%s", v)

    # Load ml_threshold_overrides from settings.ini
    if settings.has_section("ML_SIGNAL") and settings.has_option("ML_SIGNAL", "threshold_overrides"):
        override_str = settings.get("ML_SIGNAL", "threshold_overrides")
        overrides = {}
        for pair in override_str.split(","):
            pair = pair.strip()
            if ":" in pair:
                sym, val = pair.split(":", 1)
                overrides[sym.strip()] = float(val.strip())
        if overrides:
            params["ml_threshold_overrides"] = overrides
            logging.info("  ML threshold overrides: %s", overrides)

    if not args.no_ml:
        model_path = MODELS_DIR / f"model_{args.symbol.replace('.', '_')}.pkl"
        if model_path.exists():
            params["ml_model_path"] = str(model_path)
            logging.info("ML model loaded: %s", model_path.name)
        else:
            params["ml_enabled"] = False
            logging.info("No ML model found, running without ML overlay")

    _validate_config(params)
    bt = Backtest(df, params, df_m1=df_m1, df_m5=df_m5, df_m15=df_m15, df_m30=df_m30)
    bt.run()

    results = bt.get_results()
    t = results["trades"]
    csv_path = LOG_DIR / f"backtest_{args.symbol.replace('.', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if t:
        rows = []
        for trade in t:
            rows.append(
                {
                    "type": trade["type"],
                    "entry": trade["entry"],
                    "exit": trade["exit"],
                    "sl": trade["sl"],
                    "tp": trade["tp"],
                    "pnl": trade["pnl"],
                    "entry_bar": trade["entry_bar"],
                    "exit_bar": trade["exit_bar"],
                    "exit_reason": trade.get("exit_reason", ""),
                    "regime": trade.get("regime", ""),
                    "entry_type": trade.get("entry_type", ""),
                }
            )
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"\nTrade log saved to {csv_path}")


if __name__ == "__main__":
    main()
