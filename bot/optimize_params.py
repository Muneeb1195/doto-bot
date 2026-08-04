import argparse
import configparser
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.resolve()))
from backtest import Backtest

from config import validate_config as _validate_config

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

settings = configparser.ConfigParser()
settings.read(CONFIG_DIR / "settings.ini")

creds = configparser.ConfigParser()
creds.read(CONFIG_DIR / "credentials.ini")


SYMBOL_PROFILE = {
    # --- Live portfolio (7 symbols) ---
    "XAU500.raw": "D",
    "BTCUSD.raw": "FAST",
    "US30.raw": "C",
    "GBPJPY.raw": "B",
    "SOLUSD.raw": "FAST",
    "XRPUSD.raw": "FAST",
    "DOGUSD.raw": "FAST",
    # --- Pool members with trained models (kept so optimizer/auto-train can target them) ---
    "EURUSD.raw": "A",
    "NZDUSD.raw": "A",
    "USDJPY.raw": "B",
    "EURJPY.raw": "B",
    "ETHUSD.raw": "C",
    "US500.raw": "C",
    "LTCUSD.raw": "FAST",
    "ADAUSD.raw": "FAST",
    "AVXUSD.raw": "FAST",
    "GBPUSD.raw": "A",
    "AUDUSD.raw": "A",
    "XAUUSD.raw": "D",
    "XAGUSD.raw": "D",
    "XNGUSD.raw": "C",
    "XPTUSD.raw": "D",
    "SPY.raw": "C",
    "IWM.raw": "C",
    "USDCHF.raw": "A",
    "USDCAD.raw": "A",
}

PROFILE_EMAS = {
    "A": [(10, 40), (12, 48), (15, 60), (20, 80)],
    "B": [(6, 24), (8, 34), (10, 40)],
    "C": [(10, 40), (12, 48), (15, 60), (18, 72)],
    "D": [(10, 40), (12, 48), (15, 60)],
    "FAST": [(5, 20), (6, 18), (8, 24), (10, 30)],
}

PROFILE_MR = {"A": True, "B": False, "C": False, "D": False, "FAST": True}

PROFILE_SL = {"A": [1.0, 1.5, 2.0], "B": [1.0, 1.5, 2.0], "C": [1.0, 1.5, 2.0], "D": [1.5, 2.0], "FAST": [1.0, 1.5]}

PROFILE_RR = {"A": [1.5, 2.0], "B": [2.0, 2.5], "C": [2.0, 2.5], "D": [2.0, 2.5, 5.0], "FAST": [2.0, 3.0]}

PROFILE_ADX = {"A": [22, 25], "B": [22, 25], "C": [22, 25], "D": [25], "FAST": [25, 30]}

# Entry-score threshold sweep (parity with live min_entry_score). Narrow band
# around the deployed 0.60 so the optimizer can tune strictness.
PROFILE_SCORE = {
    "A": [0.55, 0.60],
    "B": [0.55, 0.60],
    "C": [0.55, 0.60],
    "D": [0.55, 0.60],
    "FAST": [0.55, 0.60],
}

PROFILE_COMMISSION = {
    "A": 976.0,
    "B": 976.0,
    "C": 278.0,
    "D": 977.0,
    "FAST": 558.0,
}


def _max_workers():
    # Leave one core for the OS / MT5 terminal; cap to keep memory reasonable.
    return max(1, (os.cpu_count() or 4) - 1)


def fetch_data(symbol, years=3):
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


def make_cpcv_windows(df, n_paths=15, is_months=24, oos_months=6, purge_months=1):
    times = df["time"]
    start = times.iloc[0]
    end = times.iloc[-1]
    total_days = (end - start).days
    total_months = max(total_days // 30, 1)
    if total_months < is_months + oos_months + purge_months:
        return make_windows(df)
    month_starts = [start + pd.DateOffset(months=i) for i in range(total_months + 1)]
    windows = []
    max_step = max(0, len(month_starts) - is_months - oos_months - purge_months - 1)
    step = 0
    while step <= max_step and len(windows) < n_paths:
        is_s = month_starts[step]
        is_e = month_starts[step + is_months]
        pg_e = month_starts[min(step + is_months + purge_months, len(month_starts) - 1)]
        oos_s = pg_e
        oos_e_idx = min(step + is_months + purge_months + oos_months, len(month_starts) - 1)
        oos_e = month_starts[oos_e_idx]
        if oos_s >= oos_e or oos_s >= end:
            step += 1
            continue
        is_mask = (times >= is_s) & (times < is_e)
        oos_mask = (times >= oos_s) & (times <= oos_e)
        df_is = df[is_mask].reset_index(drop=True)
        df_oos = df[oos_mask].reset_index(drop=True)
        if len(df_is) >= 200 and len(df_oos) >= 100:
            windows.append((df_is, df_oos, is_s, is_e, oos_s, oos_e))
        step += 1
    return windows


def score_cpcv(window_results):
    oos_scores = [score_r(wr["oos"]) for wr in window_results]
    oos_scores = [s for s in oos_scores if s > -999]
    if len(oos_scores) < 3:
        return -999
    p25 = float(np.percentile(oos_scores, 25))
    median = float(np.percentile(oos_scores, 50))
    # OOS PF floor gate: require the 25th-percentile OOS PF >= 1.1 so we never
    # select parameter sets that are only marginally profitable out-of-sample.
    robust_oos_pfs = [wr["oos"].get("profit_factor", 0) for wr in window_results if wr["oos"].get("n_trades", 0) >= 3]
    floor_met = len(robust_oos_pfs) >= 3 and float(np.percentile(robust_oos_pfs, 25)) >= 1.1
    all_positive = all(pf >= 1.0 for pf in robust_oos_pfs)
    if not floor_met:
        # Heavy penalty but still ranks within ineligible set for diagnostics.
        return (median * 0.5 + p25) * 0.25
    return median * (1.2 if all_positive else 1.0) + p25


def stability_penalty(ema_scores, fast, slow):
    ratio = round(fast / slow, 4)
    neighbors = [(f, s) for f, s in ema_scores if abs(f / s - ratio) < 0.005]
    if len(neighbors) < 2:
        return 1.0
    scores = [ema_scores[(f, s)] for f, s in neighbors]
    cv = np.std(scores) / max(abs(np.mean(scores)), 0.01)
    return 1.0 / (1.0 + cv)


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


def fetch_m1_data(symbol, years=3):
    """Fetch M1 bars for the full window via backward paging (no per-request cap)."""
    import MetaTrader5 as mt5
    from mt5_connect import fetch_rates_paged

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    df_m1 = fetch_rates_paged(symbol, mt5.TIMEFRAME_M1, start, end)
    if df_m1 is None or len(df_m1) < 100:
        return None
    return df_m1


MAX_M15_BARS = 80000  # per-request page size (MT5 API cap observed ~80k)


def fetch_m15_data(symbol, years=3):
    """Fetch M15 bars for the full window via backward paging.

    Previously a single copy_rates_from call capped at MAX_M15_BARS (~2.3y of
    M15), truncating the early part of a 3y window. Paging removes the
    per-request cap; total depth is still bounded by broker server history.
    """
    import MetaTrader5 as mt5
    from mt5_connect import fetch_rates_paged

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    df_m15 = fetch_rates_paged(symbol, mt5.TIMEFRAME_M15, start, end, chunk_bars=MAX_M15_BARS)
    if df_m15 is None or len(df_m15) < 100:
        return None
    return df_m15


_PRECOMPUTE_CACHE: dict[tuple[int, ...], dict] = {}


def _get_fixed(df, params):
    t0 = df["time"].iloc[0]
    t1 = df["time"].iloc[-1]
    key = (len(df), t0.year, t0.month, t0.day, t0.hour, t1.year, t1.month, t1.day, t1.hour)
    if key not in _PRECOMPUTE_CACHE:
        from backtest import precompute_fixed
        _PRECOMPUTE_CACHE[key] = precompute_fixed(df, params)
    return _PRECOMPUTE_CACHE[key]


def run_bt(df, params, df_m1=None, df_m15=None, fast=False, fixed=None):
    bt = Backtest(df, params, df_m1=df_m1, df_m15=df_m15, fixed=fixed)
    bt.run(fast=fast)
    return bt.get_results()


def run_walk_forward(windows, params, df_m1=None, df_m15=None, scorer=None, fast=False):
    results = []
    for w_idx, (df_is, df_oos, *_) in enumerate(windows):
        fixed_is = _get_fixed(df_is, params)
        fixed_oos = _get_fixed(df_oos, params)
        is_r = run_bt(df_is, params, df_m1=df_m1, df_m15=df_m15, fast=fast, fixed=fixed_is)
        oos_r = run_bt(df_oos, params, df_m1=df_m1, df_m15=df_m15, fast=fast, fixed=fixed_oos)
        results.append({"is": is_r, "oos": oos_r, "window": w_idx})
    score_fn = scorer or score_walk_forward
    return results, score_fn(results)


def _resolve_ma_type(symbol):
    sym_section = f"STRATEGY:{symbol}"
    if settings.has_section(sym_section) and settings.has_option(sym_section, "ma_type"):
        return settings.get(sym_section, "ma_type")
    if settings.has_option("STRATEGY", "ma_type"):
        return settings.get("STRATEGY", "ma_type")
    return "kama"


def optimize_symbol(
    symbol, df_full, info, windows, quick=False, cpcv=False, df_m1=None, df_m15=None, no_ml=False, fast=False
):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    point, tick_value, volume_step = info["point"], info["tick_value"], info["volume_step"]
    profile = SYMBOL_PROFILE.get(symbol, "A")
    mr_enabled = PROFILE_MR[profile]
    ema_grid = PROFILE_EMAS[profile]
    sl_vals = PROFILE_SL[profile]
    rr_vals = PROFILE_RR[profile]
    adx_vals = PROFILE_ADX[profile]
    score_vals = PROFILE_SCORE[profile]
    commission = PROFILE_COMMISSION.get(profile, 0.0)
    ma_type = _resolve_ma_type(symbol)

    if quick:
        ema_grid = ema_grid[::2]
        adx_vals = [25]

    ml_available = (MODELS_DIR / f"model_{symbol.replace('.', '_')}.pkl").exists()
    recent_is = windows[0][0]
    print(f"\n{'=' * 70}")
    print(
        f"Profile {profile} - {symbol} "
        f"(MR={'YES' if mr_enabled else 'NO'} Commission={commission:.0f}PKR/lot "
        f"ML={'YES' if ml_available else 'NO'})"
    )
    if not ml_available:
        print(f"  [!] No ML model found - run: python train_model.py --symbols {symbol}")
    print(f"  Full data: {len(df_full)} bars over {len(windows)} walk-forward windows")
    print(f"  First IS: {recent_is['time'].iloc[0].date()} to {recent_is['time'].iloc[-1].date()}")
    total_combos = len(ema_grid) * len(sl_vals) * len(rr_vals) * len(adx_vals) * len(score_vals)
    print(
        f"  Full grid: {len(ema_grid)} EMAs x {len(sl_vals)} SL x {len(rr_vals)} RR "
        f"x {len(adx_vals)} ADX x {len(score_vals)} score = {total_combos} combos"
    )
    print(f"{'=' * 70}")

    all_combos = []
    for ef, es in ema_grid:
        for sl in sl_vals:
            for rr in rr_vals:
                for adx in adx_vals:
                    for sc in score_vals:
                        all_combos.append((ef, es, sl, rr, adx, sc))

    print(f"\n--- Full grid search ({len(all_combos)} combos x {len(windows)} windows, parallel) ---")
    candidates = []
    scorer = score_cpcv if cpcv else None
    with ProcessPoolExecutor(max_workers=_max_workers()) as executor:
        futures = {}
        for ef, es, sl, rr, adx, sc in all_combos:
            params = build_params(
                symbol,
                ef,
                es,
                sl,
                rr,
                adx,
                point,
                tick_value,
                volume_step,
                mr_enabled,
                commission,
                ma_type,
                no_ml,
                scoring_min_entry=sc,
            )
            future = executor.submit(
                run_walk_forward, windows, params, df_m1=df_m1, df_m15=df_m15, scorer=scorer, fast=fast
            )
            futures[future] = (ef, es, sl, rr, adx, sc)

        for done, future in enumerate(as_completed(futures), 1):
            ef, es, sl, rr, adx, sc = futures[future]
            try:
                wf_r, wf_s = future.result()
                candidates.append((ef, es, sl, rr, adx, sc, wf_r, wf_s))
            except Exception as e:
                print(f"  ERROR EMA{ef}/{es} SL={sl} RR={rr} ADX={adx} score={sc}: {e}")
            if done % 10 == 0 or done == len(all_combos):
                print(f"  Progress: {done}/{len(all_combos)} combos complete")

    if quick:
        final = [(ef, es, sl, rr, adx, sc, wf_r, wf_s, 1.0, wf_s) for ef, es, sl, rr, adx, sc, wf_r, wf_s in candidates]
    else:
        ema_wf_scores = {}
        for ef, es, sl, rr, adx, sc, wf_r, wf_s in candidates:
            key = (ef, es)
            if key not in ema_wf_scores or wf_s > ema_wf_scores[key]:
                ema_wf_scores[key] = wf_s
        final = []
        for ef, es, sl, rr, adx, sc, wf_r, wf_s in candidates:
            sp = stability_penalty(ema_wf_scores, ef, es)
            final_score = wf_s * sp
            final.append((ef, es, sl, rr, adx, sc, wf_r, wf_s, sp, final_score))

    final.sort(key=lambda x: x[9], reverse=True)
    best = final[0]
    ef, es, sl, rr, adx, sc, wf_r, wf_s, sp, _ = best

    print(f"\n  BEST for {symbol}:")
    print(f"    EMA{ef}/{es} SL={sl:.1f} RR={rr:.1f} ADX={adx} score={sc:.2f}")
    print(f"    Walk-forward score: {wf_s:.1f} (stability penalty: {sp:.3f}, final: {wf_s * sp:.1f})")
    wf_is_r = wf_r[-1]["is"]
    print(
        f"    Last-window IS: PF={wf_is_r.get('profit_factor', 0):.2f} "
        f"Ret=Rs.{wf_is_r.get('total_return', 0):+.1f} DD=Rs.{wf_is_r.get('max_dd', 0):.1f} "
        f"n={wf_is_r.get('n_trades', 0)} WR={wf_is_r.get('win_rate', 0):.1%}"
    )
    print("    Per-window OOS:")
    for w_idx, wr in enumerate(wf_r):
        oos = wr["oos"]
        print(
            f"      W{w_idx + 1}: PF={oos.get('profit_factor', 0):.2f} "
            f"Ret=Rs.{oos.get('total_return', 0):+.1f} n={oos.get('n_trades', 0)} "
            f"WR={oos.get('win_rate', 0):.1%}"
        )

    rows = []
    for ef2, es2, sl2, rr2, adx2, sc2, wf_r2, wf_s2, sp2, _ in final:
        is_last = wf_r2[-1]["is"]
        rows.append(
            {
                "ema_fast": ef2,
                "ema_slow": es2,
                "sl": sl2,
                "rr": rr2,
                "adx": adx2,
                "score": sc2,
                "pf": is_last.get("profit_factor", 0),
                "ret": is_last.get("total_return", 0),
                "dd": is_last.get("max_dd", 0),
                "wr": is_last.get("win_rate", 0),
                "n_trades": is_last.get("n_trades", 0),
                "wf_score": wf_s2,
                "stability": sp2,
            }
        )
    pd.DataFrame(rows).to_csv(LOG_DIR / f"optimize_{symbol.replace('.', '_')}.csv", index=False)

    return {
        "symbol": symbol,
        "ema_fast": ef,
        "ema_slow": es,
        "sl": sl,
        "rr": rr,
        "adx": adx,
        "scoring_min_entry": sc,
        "pf": wf_is_r.get("profit_factor", 0),
        "ret": wf_is_r.get("total_return", 0),
        "dd": wf_is_r.get("max_dd", 0),
        "wr": wf_is_r.get("win_rate", 0),
        "n_trades": wf_is_r.get("n_trades", 0),
        "score": wf_s * sp,
        "n_windows": len(windows),
        "oos_pfs": [wr["oos"].get("profit_factor", 0) for wr in wf_r],
    }


def optimize_symbol_twophase(symbol, df_full, info, windows, df_m1=None, df_m15=None, no_ml=False):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    point, tick_value, volume_step = info["point"], info["tick_value"], info["volume_step"]
    profile = SYMBOL_PROFILE.get(symbol, "A")
    mr_enabled = PROFILE_MR[profile]
    ema_grid = PROFILE_EMAS[profile]
    sl_vals = PROFILE_SL[profile]
    rr_vals = PROFILE_RR[profile]
    adx_vals = PROFILE_ADX[profile]
    commission = PROFILE_COMMISSION.get(profile, 0.0)
    ma_type = _resolve_ma_type(symbol)

    ml_available = (MODELS_DIR / f"model_{symbol.replace('.', '_')}.pkl").exists()
    recent_is = windows[0][0]
    print(f"\n{'=' * 70}")
    print(
        f"Profile {profile} - {symbol} "
        f"(MR={'YES' if mr_enabled else 'NO'} ML={'YES' if ml_available else 'NO'} MA={ma_type})"
    )
    if not ml_available:
        print(f"  [!] No ML model found - run: python train_model.py --symbols {symbol}")
    print(f"  Data: {len(df_full)} bars, {len(windows)} windows")
    print(f"  First IS: {recent_is['time'].iloc[0].date()} to {recent_is['time'].iloc[-1].date()}")
    print(f"  EMA grid: {len(ema_grid)} pairs, SL grid: {sl_vals}, RR grid: {rr_vals}, ADX grid: {adx_vals}")

    n_phase1 = min(2, len(windows))
    print(f"\n{'=' * 70}")
    print(f"Phase 1: EMA sweep ({len(ema_grid)} EMAs x {n_phase1} windows, SL=1.5 RR=2.0 ADX=25 fixed)")
    print(f"{'=' * 70}")

    phase1_combos = []
    phase1_params = []
    for ef, es in ema_grid:
        params = build_params(
            symbol, ef, es, 1.5, 2.0, 25, point, tick_value, volume_step, mr_enabled, commission, ma_type, no_ml
        )
        phase1_combos.append((ef, es))
        phase1_params.append(params)

    phase1_scores = {}
    with ProcessPoolExecutor(max_workers=_max_workers()) as executor:
        futures = {}
        for idx, (ef, es) in enumerate(phase1_combos):
            w1_windows = windows[:n_phase1]
            future = executor.submit(run_walk_forward, w1_windows, phase1_params[idx], df_m1=df_m1, df_m15=df_m15)
            futures[future] = (ef, es)

        for future in as_completed(futures):
            ef, es = futures[future]
            try:
                wf_r, wf_s = future.result()
                phase1_scores[(ef, es)] = (wf_s, wf_r)
                print(f"  EMA{ef}/{es}: WF score={wf_s:.1f}")
            except Exception as e:
                print(f"  ERROR EMA{ef}/{es}: {e}")

    ema_ranked = sorted(phase1_scores.items(), key=lambda x: x[1][0], reverse=True)
    top_emas = [pair for pair, _ in ema_ranked[:2]]
    print(f"\n  Top EMAs: {top_emas}")

    print(f"\n{'=' * 70}")
    print(f"Phase 2: SL/RR/ADX refinement for top EMAs ({len(windows)} windows)")
    print(f"{'=' * 70}")

    phase2_combos = []
    phase2_keys = []
    for ef, es in top_emas:
        for sl in sl_vals:
            for rr in rr_vals:
                for w_idx in range(len(windows)):
                    params = build_params(
                        symbol,
                        ef,
                        es,
                        sl,
                        rr,
                        25,
                        point,
                        tick_value,
                        volume_step,
                        mr_enabled,
                        commission,
                        ma_type,
                        no_ml,
                    )
                    phase2_combos.append((windows[w_idx][0].copy(), params.copy()))
                    phase2_keys.append((symbol, ef, es, sl, rr, 25, w_idx, "is"))
                    phase2_combos.append((windows[w_idx][1].copy(), params.copy()))
                    phase2_keys.append((symbol, ef, es, sl, rr, 25, w_idx, "oos"))
        for adx in adx_vals:
            for w_idx in range(len(windows)):
                params = build_params(
                    symbol,
                    ef,
                    es,
                    1.5,
                    2.0,
                    adx,
                    point,
                    tick_value,
                    volume_step,
                    mr_enabled,
                    commission,
                    ma_type,
                    no_ml,
                )
                phase2_combos.append((windows[w_idx][0].copy(), params.copy()))
                phase2_keys.append((symbol, ef, es, 1.5, 2.0, adx, w_idx, "is"))
                phase2_combos.append((windows[w_idx][1].copy(), params.copy()))
                phase2_keys.append((symbol, ef, es, 1.5, 2.0, adx, w_idx, "oos"))

    print(f"  Running {len(phase2_combos)} backtests (SL/RR/ADX sweep)...")
    phase2_results = {}
    with ProcessPoolExecutor(max_workers=_max_workers()) as executor:
        futures = {
            executor.submit(run_bt, df, params, df_m15=df_m15): k for (df, params), k in zip(phase2_combos, phase2_keys)
        }
        for done, future in enumerate(as_completed(futures), 1):
            k = futures[future]
            try:
                phase2_results[k] = future.result()
            except Exception as e:
                print(f"  ERROR {k}: {e}")
            if done % 40 == 0 or done == len(phase2_combos):
                print(f"  Phase 2 progress: {done}/{len(phase2_combos)}")

    candidates = []
    for ef, es in top_emas:
        for sl in sl_vals:
            for rr in rr_vals:
                wr_list = []
                for w_idx in range(len(windows)):
                    is_k = (symbol, ef, es, sl, rr, 25, w_idx, "is")
                    oos_k = (symbol, ef, es, sl, rr, 25, w_idx, "oos")
                    if is_k in phase2_results and oos_k in phase2_results:
                        wr_list.append({"is": phase2_results[is_k], "oos": phase2_results[oos_k], "window": w_idx})
                if wr_list:
                    wf_s = score_walk_forward(wr_list)
                    candidates.append((ef, es, sl, rr, 25, wr_list, wf_s))
        for adx in adx_vals:
            wr_list = []
            for w_idx in range(len(windows)):
                is_k = (symbol, ef, es, 1.5, 2.0, adx, w_idx, "is")
                oos_k = (symbol, ef, es, 1.5, 2.0, adx, w_idx, "oos")
                if is_k in phase2_results and oos_k in phase2_results:
                    wr_list.append({"is": phase2_results[is_k], "oos": phase2_results[oos_k], "window": w_idx})
            if wr_list:
                wf_s = score_walk_forward(wr_list)
                candidates.append((ef, es, 1.5, 2.0, adx, wr_list, wf_s))

    candidates.sort(key=lambda x: x[6], reverse=True)
    best = candidates[0]
    ef, es, sl, rr, adx, wf_r, wf_s = best
    wf_is_r = wf_r[-1]["is"]

    print(f"\n  BEST for {symbol}:")
    print(f"    EMA{ef}/{es} SL={sl:.1f} RR={rr:.1f} ADX={adx}  WF={wf_s:.1f}")
    print(
        f"    IS: PF={wf_is_r.get('profit_factor', 0):.2f} Ret=Rs.{wf_is_r.get('total_return', 0):+.1f} "
        f"DD=Rs.{wf_is_r.get('max_dd', 0):.1f} n={wf_is_r.get('n_trades', 0)} WR={wf_is_r.get('win_rate', 0):.1%}"
    )
    for w_idx, wr in enumerate(wf_r):
        oos = wr["oos"]
        print(
            f"    W{w_idx + 1} OOS: PF={oos.get('profit_factor', 0):.2f} "
            f"Ret=Rs.{oos.get('total_return', 0):+.1f} n={oos.get('n_trades', 0)} "
            f"WR={oos.get('win_rate', 0):.1%}"
        )

    rows = []
    for ef2, es2, sl2, rr2, adx2, wf_r2, wf_s2 in candidates:
        is_last = wf_r2[-1]["is"]
        rows.append(
            {
                "ema_fast": ef2,
                "ema_slow": es2,
                "sl": sl2,
                "rr": rr2,
                "adx": adx2,
                "pf": is_last.get("profit_factor", 0),
                "ret": is_last.get("total_return", 0),
                "dd": is_last.get("max_dd", 0),
                "wr": is_last.get("win_rate", 0),
                "n_trades": is_last.get("n_trades", 0),
                "wf_score": wf_s2,
            }
        )
    pd.DataFrame(rows).to_csv(LOG_DIR / f"optimize_{symbol.replace('.', '_')}.csv", index=False)

    return {
        "symbol": symbol,
        "ema_fast": ef,
        "ema_slow": es,
        "sl": sl,
        "rr": rr,
        "adx": adx,
        "pf": wf_is_r.get("profit_factor", 0),
        "ret": wf_is_r.get("total_return", 0),
        "dd": wf_is_r.get("max_dd", 0),
        "wr": wf_is_r.get("win_rate", 0),
        "n_trades": wf_is_r.get("n_trades", 0),
        "score": wf_s,
        "scoring_min_entry": float(settings.get("SCORING", "min_entry_score", fallback=0.60)),
        "n_windows": len(windows),
        "oos_pfs": [wr["oos"].get("profit_factor", 0) for wr in wf_r],
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize per-symbol params")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols")
    parser.add_argument("--years", type=float, default=3)
    parser.add_argument("--quick", action="store_true", help="Quick screening mode (1 window, reduced grid)")
    parser.add_argument(
        "--cpcv", action="store_true", help="Combinatorial Purged Cross-Validation (15 paths, ~3 min/symbol)"
    )
    parser.add_argument("--auto-train", action="store_true", help="Train missing ML models before optimization")
    parser.add_argument("--m1-sim", action="store_true", help="Enable M1 bar entry simulation")
    parser.add_argument(
        "--two-phase",
        action="store_true",
        default=True,
        help="Two-phase: EMA sweep then SL/RR/ADX refinement (default)",
    )
    parser.add_argument("--full-grid", action="store_true", help="Full grid search (slow but exhaustive)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML models during optimization")
    parser.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=None,
        help="Use the Numba-JIT fast backtest path (default: on if numba available)",
    )
    parser.add_argument(
        "--no-fast", dest="fast", action="store_false", help="Force the pure-Python reference backtest loop"
    )
    parser.add_argument("--cpcv-paths", type=int, default=30, help="Number of CPCC paths (default 30; lower = faster)")
    args = parser.parse_args()

    import backtest

    fast_default = backtest._njit_available()
    fast = fast_default if args.fast is None else args.fast

    import platform
    import subprocess
    import time

    import MetaTrader5 as mt5

    mt5_path = settings.get("MT5", "path", fallback="C:\\Program Files\\MetaTrader 5\\terminal64.exe")
    timeout_ms = settings.getint("MT5", "timeout_ms", fallback=180000)

    ok = mt5.initialize()
    if not ok:
        ok = mt5.initialize(
            path=mt5_path,
            login=int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
            password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
            server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
            timeout=timeout_ms,
        )
    if not ok:
        print(f"MT5 init failed ({mt5.last_error()}) - restarting terminal...")
        if platform.system() == "Linux":
            subprocess.run(["pkill", "-f", "terminal64.exe"], capture_output=True)
            subprocess.run(["pkill", "-f", "winedevice.exe"], capture_output=True)
        else:
            subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
            subprocess.run(["taskkill", "/F", "/IM", "winedevice.exe"], capture_output=True)
        time.sleep(5)
        if platform.system() == "Linux":
            subprocess.Popen(["wine", mt5_path])
        else:
            subprocess.Popen([mt5_path])
        time.sleep(15)
        ok = mt5.initialize(
            path=mt5_path,
            login=int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
            password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
            server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
            timeout=timeout_ms,
        )
    if not ok:
        print(f"MT5 init failed: {mt5.last_error()}")
        return
    authorized = mt5.login(
        int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
        password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
        server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
    )
    if not authorized:
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return

    symbols = (
        [s.strip() for s in settings.get("PORTFOLIO", "symbols", fallback="").split(",") if s.strip()]
        if (not args.symbols or args.symbols.upper() == "ALL")
        else [s.strip() for s in args.symbols.split(",")]
    )

    if args.auto_train:
        missing = [s for s in symbols if not (MODELS_DIR / f"model_{s.replace('.', '_')}.pkl").exists()]
        if missing:
            print(f"\n{'=' * 60}")
            print(f"Training ML models for {len(missing)} symbols missing models...")
            print(f"{'=' * 60}")
            script = Path(__file__).resolve()
            trainer = script.parent / "train_model.py"
            trainer_path = str(trainer)
            if sys.platform == "linux" and not trainer_path.startswith("Z:"):
                trainer_path = "Z:" + trainer_path
            for sym in missing:
                print(f"\n  Training {sym}...")
                result = subprocess.run(
                    [sys.executable, trainer_path, "--symbols", sym, "--years", "3"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    print(f"  [OK] {sym} model trained")
                else:
                    print(f"  [FAIL] {sym} training failed: {result.stderr[-300:]}")
            print("\nML training complete.\n")
        else:
            print("All symbols have ML models.")

    all_best = []
    for symbol in symbols:
        print(f"\nFetching {symbol}...")
        # Fetch M15 BEFORE H1 (MT5 Python API bug: H1 fetch corrupts subsequent M15 fetches)
        df_m15 = fetch_m15_data(symbol, args.years)
        data = fetch_data(symbol, args.years)
        if data is None or data[0] is None:
            print(f"  SKIP {symbol}")
            continue
        df, info = data
        if len(df) < 5000:
            print(f"  Too few bars ({len(df)}), SKIP")
            continue
        # Always fetch M1: of_* orderflow features are model inputs post-item-#11
        # retrain, so the optimizer's ML gate must see real values (parity with
        # training/live). Attach of_* to the H1 frame BEFORE window slicing so
        # every walk-forward window carries them into worker processes.
        df_m1 = fetch_m1_data(symbol, args.years)
        if df_m1 is not None:
            from ml_features import attach_orderflow_features

            attach_orderflow_features(df, df_m1)
            print(f"  M1 data: {len(df_m1)} bars (of_* attached to H1 frame)")
        else:
            print("  M1 data: unavailable — of_* features fall back to 0.0 in ML scoring")
        if args.cpcv:
            windows = make_cpcv_windows(df, n_paths=args.cpcv_paths, is_months=24, oos_months=6, purge_months=1)
        elif args.quick:
            windows = make_windows(df, n_windows=1, window_months=24, oos_months=6)
        else:
            windows = make_windows(df, n_windows=4, window_months=24, oos_months=6)
        if len(windows) < 1:
            print("  No windows created, SKIP")
            continue
        if not args.quick and not args.cpcv and len(windows) < 2:
            print(f"  Only {len(windows)} windows (need >=2 for full WF), SKIP")
            continue
        print(
            f"  Created {len(windows)} {'CPCV' if args.cpcv else 'walk-forward'} windows "
            f"(fast={'ON' if fast else 'OFF'}):"
        )
        for i, (is_df, oos_df, is_s, is_e, oos_s, oos_e) in enumerate(windows):
            print(
                f"    W{i + 1}: IS {is_s.date()} to {is_e.date()} ({len(is_df)} bars) "
                f"| OOS {oos_s.date()} to {oos_e.date()} ({len(oos_df)} bars)"
            )
        # Raw M1 goes to workers only for --m1-sim intra-bar entry simulation
        # (of_* are already attached to df; pickling ~1M M1 rows per task is
        # wasteful otherwise).
        df_m1_sim = df_m1 if args.m1_sim else None
        if args.m1_sim:
            print(f"  M1 entry simulation: {len(df_m1) if df_m1 is not None else 0} bars")
        print(f"  M15 data: {len(df_m15) if df_m15 is not None else 0} bars for MTF entry TF")
        use_twophase = args.two_phase and not args.full_grid and not args.quick and not args.cpcv
        if use_twophase:
            print("  Using two-phase optimization (EMA sweep + SL/RR/ADX refinement)")
            best = optimize_symbol_twophase(
                symbol, df, info, windows, df_m1=df_m1_sim, df_m15=df_m15, no_ml=args.no_ml
            )
        else:
            best = optimize_symbol(
                symbol,
                df,
                info,
                windows,
                quick=args.quick,
                cpcv=args.cpcv,
                df_m1=df_m1_sim,
                df_m15=df_m15,
                no_ml=args.no_ml,
                fast=fast,
            )
        if best:
            all_best.append(best)

    mt5.shutdown()

    print(f"\n{'=' * 80}")
    print("WALK-FORWARD OPTIMIZATION RESULTS")
    print(f"{'=' * 80}")
    sorted_results = sorted(all_best, key=lambda x: x["score"], reverse=True)
    for b in sorted_results:
        oos_str = " ".join(f"PF={pf:.2f}" for pf in b.get("oos_pfs", []))
        print(
            f"  {b['symbol']:15s} EMA{b['ema_fast']:2d}/{b['ema_slow']:<3d} "
            f"SL={b['sl']:.1f} RR={b['rr']:.1f} ADX={b['adx']:2d} score={b.get('scoring_min_entry', 0):.2f}  "
            f"PF={b['pf']:.2f} Ret=Rs.{b['ret']:+.1f} DD=Rs.{b['dd']:.1f} "
            f"WR={b.get('wr', 0):.1%} n={b['n_trades']:2d}  "
            f"WF={b['score']:.0f} windows={b['n_windows']}  [{oos_str}]"
        )

    if len(sorted_results) >= 2 and not args.quick:
        print(f"\n{'=' * 80}")
        print("PORTFOLIO ANALYSIS")
        print(f"{'=' * 80}")

        combined_dd_upper = sum(b.get("dd", 0) or 0 for b in sorted_results)
        print(f"  Simple DD sum bound (worst case): Rs.{combined_dd_upper:,.0f}")
        print(f"  Portfolio size: Rs.{500000:,.0f}  -> DD bound = {combined_dd_upper / 500000 * 100:.1f}%")

        profile_counts = {}
        for b in sorted_results:
            sym = b["symbol"]
            prof = SYMBOL_PROFILE.get(sym, "?")
            profile_counts.setdefault(prof, []).append(sym)

        print("\n  Profile diversity:")
        for prof, syms in sorted(profile_counts.items()):
            print(f"    {prof}: {', '.join(syms)}")

        print("\n  Correlation-aware ranking (corr>0.7 -> 50% penalty on lower-scored same-profile pair):")
        ranked = sorted_results[:]
        for i in range(len(ranked)):
            for j in range(i + 1, len(ranked)):
                p_i = SYMBOL_PROFILE.get(ranked[i]["symbol"], "?")
                p_j = SYMBOL_PROFILE.get(ranked[j]["symbol"], "?")
                if p_i == p_j:
                    lower = j if ranked[j]["score"] < ranked[i]["score"] else i
                    ranked[lower]["score"] *= 0.5

        ranked.sort(key=lambda x: x["score"], reverse=True)
        for b in ranked[:8]:
            oos_str = " ".join(f"PF={pf:.2f}" for pf in b.get("oos_pfs", []))
            prof = SYMBOL_PROFILE.get(b["symbol"], "?")
            print(
                f"  {b['symbol']:15s} [{prof}] EMA{b['ema_fast']:2d}/{b['ema_slow']:<3d} "
                f"SL={b['sl']:.1f} RR={b['rr']:.1f} ADX={b['adx']:2d}  "
                f"PF={b['pf']:.2f} WF={b['score']:.0f}  [{oos_str}]"
            )

        print("\n  Recommended portfolio (top 5-8):")
        for b in ranked[:8]:
            prof = SYMBOL_PROFILE.get(b["symbol"], "?")
            print(
                f"    {b['symbol']:15s} [{prof}] EMA{b['ema_fast']}/{b['ema_slow']} "
                f"SL={b['sl']} RR={b['rr']} ADX={b['adx']}"
            )

        if len(ranked) > 8:
            print(f"    ... ({len(ranked) - 8} more symbols)")


if __name__ == "__main__":
    main()
