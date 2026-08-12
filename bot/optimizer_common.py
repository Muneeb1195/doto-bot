"""Shared helpers for the optimizer entry point (optimize_params.py).

Consolidated from the copy-pasted cluster that optimize_params.py and the
former parallel_optimize.py both carried: the byte-identical trio
(_resolve_ma_type, make_windows, build_params) plus the divergent run_bt /
score_r / score_walk_forward / fetch_data / fetch_m1_data / fetch_m15_data
and the _H1.csv loader. optimize_params.py is the single optimizer entry
point — the CI monthly/daily optimization runs it (optimize.yml) and its
--two-phase mode (default) is the strategy parallel_optimize.py duplicated.
"""

import configparser
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from backtest import Backtest  # noqa: E402

from config import validate_config as _validate_config  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"

settings = configparser.ConfigParser()
settings.read(CONFIG_DIR / "settings.ini")


def _load_csv_data(symbol):
    """Load pre-exported H1 bars from data/history/<SYMBOL>_H1.csv for offline optimization."""
    from train_model import load_csv_data_train

    csv_path = BASE_DIR / "data" / "history" / f"{symbol.replace('.', '_')}_H1.csv"
    if not csv_path.exists():
        print(f"  CSV not found: {csv_path}")
        return None, None
    df = load_csv_data_train(symbol, tf_name="H1")
    if df is None:
        return None, None
    point = 0.01
    tick_value = 1.0
    volume_step = 0.01
    if settings.has_section("SYMBOL_POINTS"):
        if settings.has_option("SYMBOL_POINTS", symbol):
            point = float(settings.get("SYMBOL_POINTS", symbol))
        if settings.has_option("SYMBOL_POINTS", symbol + "_tick"):
            tick_value = float(settings.get("SYMBOL_POINTS", symbol + "_tick"))
        if settings.has_option("SYMBOL_POINTS", symbol + "_vstep"):
            volume_step = float(settings.get("SYMBOL_POINTS", symbol + "_vstep"))
    print(f"  Loaded {len(df)} bars from CSV ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")
    return df, {"point": point, "tick_value": tick_value, "volume_step": volume_step}


def fetch_data(symbol, years=5.1, csv_mode=False):
    if csv_mode:
        return _load_csv_data(symbol)
    import time

    import MetaTrader5 as mt5

    mt5.symbol_select(symbol, True)
    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, start, end)
    if rates is None or len(rates) < 200:
        return None, None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    print(f"  Loaded {len(df)} bars ({df['time'].iloc[0].date()} to {df['time'].iloc[-1].date()})")

    sinfo = mt5.symbol_info(symbol)
    point = sinfo.point if sinfo else 0.01
    tick_value = sinfo.trade_tick_value if sinfo else 0.01
    volume_step = sinfo.volume_step if sinfo and sinfo.volume_step > 0 else 0.01

    if tick_value == 0:
        mt5.symbol_select(symbol, False)
        mt5.symbol_select(symbol, True)
        time.sleep(0.5)
        sinfo = mt5.symbol_info(symbol)
        tick_value = sinfo.trade_tick_value if sinfo else 0.01

    return df, {"point": point, "tick_value": tick_value, "volume_step": volume_step}


def fetch_m1_data(symbol, years=5.1, csv_mode=False):
    """Fetch M1 bars for the full window via backward paging (no per-request cap).

    In csv_mode the bars come from data/history/<SYMBOL>_M1.csv when present.
    M1 exports are large and are not committed for every symbol, so a missing
    file is not an error: callers treat None as "no orderflow features", which
    matches how the models were trained in CI.
    """
    if csv_mode:
        from train_model import load_csv_data_train

        return load_csv_data_train(symbol, tf_name="M1")

    import MetaTrader5 as mt5
    from mt5_connect import fetch_rates_paged

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    df_m1 = fetch_rates_paged(symbol, mt5.TIMEFRAME_M1, start, end)
    if df_m1 is None or len(df_m1) < 100:
        return None
    return df_m1


MAX_M15_BARS = 80000  # per-request page size (MT5 API cap observed ~80k)


def fetch_m15_data(symbol, years=5.1, csv_mode=False):
    """Fetch M15 bars for the full window via backward paging.

    Previously a single copy_rates_from call capped at MAX_M15_BARS (~2.3y of
    M15), truncating the early part of a 3y window. Paging removes the
    per-request cap; total depth is still bounded by broker server history.
    """
    if csv_mode:
        from train_model import load_csv_data_train

        return load_csv_data_train(symbol, tf_name="M15")
    import MetaTrader5 as mt5
    from mt5_connect import fetch_rates_paged

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    df_m15 = fetch_rates_paged(symbol, mt5.TIMEFRAME_M15, start, end, chunk_bars=MAX_M15_BARS)
    if df_m15 is None or len(df_m15) < 100:
        return None
    return df_m15


def score_r(r):
    pf = r.get("profit_factor", 0) or 0
    ret = r.get("total_return", 0) or 0
    dd = r.get("max_dd", 10000) or 10000
    n = r.get("n_trades", 0)
    if n < 3:
        return -999
    pf_capped = min(pf, 3.0)
    if pf == float("inf"):
        pf_capped = 3.0
    calmar = abs(ret / dd) if dd > 0 else 0
    return pf_capped * calmar * (1 + 0.05 * np.sqrt(n))


def make_windows(df, n_windows=4, window_months=24, oos_months=6):
    times = df["time"]
    end = times.iloc[-1]
    windows = []
    for i in range(n_windows):
        oos_end = end - pd.DateOffset(months=i * oos_months)
        oos_start = oos_end - pd.DateOffset(months=oos_months)
        is_end = oos_start
        is_start = is_end - pd.DateOffset(months=window_months)
        is_mask = (times >= is_start) & (times < is_end)
        oos_mask = (times >= oos_start) & (times < oos_end)
        df_is = df[is_mask].reset_index(drop=True)
        df_oos = df[oos_mask].reset_index(drop=True)
        if len(df_is) < 200 or len(df_oos) < 100:
            continue
        windows.append((df_is, df_oos, is_start, is_end, oos_start, oos_end))
    return windows


def score_walk_forward(window_results):
    is_scores = []
    oos_pfs = []
    for wr in window_results:
        is_r = wr["is"]
        oos_r = wr["oos"]
        is_s = score_r(is_r)
        is_scores.append(is_s)
        oos_pf = oos_r.get("profit_factor", 0) or 0
        oos_pfs.append(oos_pf)
    if not is_scores:
        return -999
    avg_is = float(np.mean(is_scores))
    penalty = 0.5 if any(pf < 0.8 for pf in oos_pfs) else 1.0
    bonus = 1.2 if all(pf >= 1.0 for pf in oos_pfs) else 1.0
    return avg_is * penalty * bonus


def build_params(
    symbol,
    ema_fast,
    ema_slow,
    sl,
    rr,
    adx,
    point,
    tick_value,
    volume_step,
    mr_enabled=True,
    commission=0.0,
    ma_type="kama",
    no_ml=False,
    scoring_min_entry=None,
):
    p = {
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "ma_type": ma_type,
        "atr_period": 14,
        "atr_sl_mult": sl,
        "rr": rr,
        "adx_enabled": True,
        "adx_trend_threshold": adx,
        "adx_range_threshold": 20,
        "stops_level": 50,
        "ml_confidence": 0.40,
        "ml_threshold_overrides": {},
        "volume_filter": True,
        "volume_kappa": 1.2,
        "volatility_filter": True,
        "atr_sma_period": 20,
        "chandelier_enabled": True,
        "chandelier_mult": 3.0,
        "chandelier_mult_partial": 1.5,
        "chandelier_mult_overrides": {},
        "chandelier_lookback": 14,
        "ch_two_stage": True,
        "ch_loose_mult": 3.5,
        "ch_tight_mult": 1.5,
        "ch_two_stage_min_r": 3.0,
        "scale_out_enabled": True,
        "scale_out_close_fractions": [0.20, 0.20],
        "scale_out_tp_targets_rr": [0.50, 0.75],
        "ml_enabled": not no_ml,
        "risk_percent": 1.0,
        "initial_balance": 500000.0,
        "spread_model": 1.0,
        "commission": commission,
        "skip_uncertain_exhaustion": True,
        "dr_enabled": True,
        "dr_lookback": 50,
        "dr_kelly_fraction": 0.25,
        "dr_vol_adjust": True,
        "dr_min_mult": 0.25,
        "dr_max_mult": 1.5,
        "max_positions": 5,
        "max_positions_per_symbol": 1,
        "max_risk_ratio": 2.0,
        "spf_enabled": True,
        "spf_max_ratio": 0.30,
        "session_enabled": False,
        "daily_loss_pct": 5.0,
        "tr_enabled": True,
        "tr_sigma": 3.0,
        "tr_lookback": 50,
        "tr_max_dd_pct": 8.0,
        "cb_dd_pct": 15.0,
        "mr_enabled": mr_enabled,
        "mr_rsi_period": 14,
        "mr_rsi_oversold": 30,
        "mr_rsi_overbought": 70,
        "mr_sl_atr_mult": 1.0,
        "mr_tp_atr_mult": 1.5,
        "mr_position_size_mult": 0.5,
        "mr_cooldown_enabled": True,
        "mr_cooldown_bars": 2,
        "pb_enabled": True,
        "pb_atr_mult": 2.0,
        "slippage_points": 2,
        "mtf_enabled": True,
        "mtf_agreement_threshold": 0.67,
        "point": point,
        "tick_value": tick_value,
        "volume_step": volume_step,
        "symbol": symbol,
        "htf_ema_slow": 100,
        "htf_misalign_size_mult": float(settings.get("STRATEGY", "htf_misalign_size_mult", fallback=0.5)),
        "scoring_enabled": True,
        "scoring_min_entry": float(scoring_min_entry)
        if scoring_min_entry is not None
        else float(settings.get("SCORING", "min_entry_score", fallback=0.60)),
    }
    sym_section = f"STRATEGY:{symbol}"
    override_map = {
        "kelly_fraction": ("dr_kelly_fraction", float),
        "risk_percent": ("risk_percent", float),
        "max_positions_per_symbol": ("max_positions_per_symbol", int),
        "min_entry_score": ("scoring_min_entry", float),
        "htf_misalign_size_mult": ("htf_misalign_size_mult", float),
        "mtf_enabled": ("mtf_enabled", lambda v: v.lower() == "true"),
    }
    if settings.has_section(sym_section):
        for ini_key, (cfg_key, converter) in override_map.items():
            if settings.has_option(sym_section, ini_key):
                p[cfg_key] = converter(settings.get(sym_section, ini_key))
    if settings.has_section("ML_SIGNAL") and settings.has_option("ML_SIGNAL", "threshold_overrides"):
        overrides = {}
        for pair in settings.get("ML_SIGNAL", "threshold_overrides").split(","):
            pair = pair.strip()
            if ":" in pair:
                sym, val = pair.split(":", 1)
                overrides[sym.strip()] = float(val.strip())
        if overrides:
            p["ml_threshold_overrides"] = overrides
    if not no_ml:
        model_path = MODELS_DIR / f"model_{symbol.replace('.', '_')}.pkl"
        if model_path.exists():
            p["ml_model_path"] = str(model_path)
        else:
            if p.get("ml_enabled", True):
                p["ml_enabled"] = False
    _validate_config(p)
    return p


def run_bt(df, params, df_m1=None, df_m15=None, fast=False, fixed=None):
    bt = Backtest(df, params, df_m1=df_m1, df_m15=df_m15, fixed=fixed)
    bt.run(fast=fast)
    return bt.get_results()


def _resolve_ma_type(symbol):
    sym_section = f"STRATEGY:{symbol}"
    if settings.has_section(sym_section) and settings.has_option(sym_section, "ma_type"):
        return settings.get(sym_section, "ma_type")
    if settings.has_option("STRATEGY", "ma_type"):
        return settings.get("STRATEGY", "ma_type")
    return "kama"
