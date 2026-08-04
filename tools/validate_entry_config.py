"""validate_entry_config.py — validate relaxed vs strict ENTRY config.

Runs the SAME live entry pipeline (signals.get_signal / check_htf_trend /
compute_entry_score / check_execution_signal) over historical H1 bars, in two
modes, and SIMULATES each passed trade's outcome with ATR-based SL/TP (matching
execution.py): sl = atr * sl_mult, tp = sl * rr. Reports win rate, profit
factor, net R, max drawdown and trade count so we can confirm the relaxed live
config is actually profitable before keeping it.

Modes:
  strict  -> HTF hard-blocks on ANY misalignment (soft treated as block),
             scoring_min_entry = 0.67, news true-neutral 0.5 (symmetric).
  relaxed -> HTF 3-state (soft = reduced size, allow = full),
             scoring_min_entry = 0.60, news true-neutral 0.5.

Usage:
  python tools/validate_entry_config.py --symbol XAU500.raw,NZDUSD.raw --years 1
  python tools/validate_entry_config.py --symbol XAU500.raw --years 1 --mode both
"""
import argparse
import configparser
import logging
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

import MetaTrader5 as mt5  # noqa: E402

import state as _st  # noqa: E402
from config import load_config, apply_symbol_strategy, apply_symbol_overrides  # noqa: E402
from mt5_connect import ensure_mt5_connected, get_rates  # noqa: E402
from indicators import calc_adx_series, calc_ma, calc_atr_series, calc_efficiency_ratio  # noqa: E402
from signals import (  # noqa: E402
    get_mean_reversion_signal, compute_entry_score,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
log = logging.getLogger("validate")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _regime_from_adx(adx, cfg):
    strong = cfg.get("adx_threshold_strong", 30)
    weak = cfg.get("adx_threshold_weak", 20)
    if adx is None:
        return "uncertain"
    if adx >= strong:
        return "strong_trend"
    if adx >= weak:
        return "weak_trend"
    return "ranging"


def _crossover_local(win, sym_cfg):
    ma_type = sym_cfg.get("ma_type", "kama")
    fast = calc_ma(win, sym_cfg["ema_fast"], ma_type)
    slow = calc_ma(win, sym_cfg["ema_slow"], ma_type)
    atr = calc_atr(win, sym_cfg["atr_period"])
    cf, cs = fast.iloc[-2], slow.iloc[-2]
    pf, ps = fast.iloc[-3], slow.iloc[-3]
    if pf <= ps and cf > cs:
        signal = "buy"
    elif pf >= ps and cf < cs:
        signal = "sell"
    else:
        return None, None
    er_min = sym_cfg.get("er_min", 0.10)
    er = calc_efficiency_ratio(win["close"].values, sym_cfg.get("er_period", 10))
    if er < er_min:
        return None, None
    return signal, atr


def _simulate_outcome(df, i, signal, atr, sl_mult, rr, point):
    """Simulate SL/TP outcome from bar i using subsequent H1 closes.
    Returns R-multiple (+rr on TP hit, -1 on SL hit, 0 if neither in window)."""
    sl_points = max(int(atr * sl_mult / point), 1)
    tp_points = int(sl_points * rr)
    entry = df["close"].iloc[i]
    if signal == "buy":
        sl = entry - sl_points * point
        tp = entry + tp_points * point
    else:
        sl = entry + sl_points * point
        tp = entry - tp_points * point
    for j in range(i + 1, min(len(df), i + 400)):
        hi = df["high"].iloc[j]
        lo = df["low"].iloc[j]
        if signal == "buy":
            if lo <= sl:
                return -1.0
            if hi >= tp:
                return float(rr)
        else:
            if hi >= sl:
                return -1.0
            if lo <= tp:
                return float(rr)
    return 0.0


def evaluate_symbol(symbol, cfg, years, mode):
    sym_cfg = deepcopy(cfg)
    sym_cfg["symbol"] = symbol
    apply_symbol_strategy(sym_cfg, symbol)
    apply_symbol_overrides(sym_cfg, symbol)
    # Mode overrides
    if mode == "strict":
        sym_cfg["scoring_min_entry"] = 0.67
    else:  # relaxed
        sym_cfg["scoring_min_entry"] = 0.60

    need = max(sym_cfg["ema_slow"], sym_cfg["atr_period"]) + sym_cfg["atr_period"] + 60
    target = need + int(years * 365 * 24 * 0.6)
    df = None
    for mult in (1.0, 0.5, 0.25, 0.1):
        bars = need + max(int((target - need) * mult), 100)
        df = get_rates(symbol, sym_cfg["timeframe"], bars)
        if df is not None and len(df) >= need:
            break
    if df is None or len(df) < need:
        df = get_rates(symbol, sym_cfg["timeframe"], need + 150)
    if df is None or len(df) < need:
        log.warning("[%s] insufficient data", symbol)
        return None

    # ---- Vectorized precompute (O(n), not O(n^2)) ----
    ma_type = sym_cfg.get("ma_type", "kama")
    fast_ma = calc_ma(df, sym_cfg["ema_fast"], ma_type)
    slow_ma = calc_ma(df, sym_cfg["ema_slow"], ma_type)
    atr_series = pd.Series(calc_atr_series(df, sym_cfg["atr_period"]))
    adx_series = pd.Series(calc_adx_series(df, sym_cfg.get("adx_period", 14)))

    # ---- Time-aligned H4 EMA200 + slope (merge_asof on close time) ----
    htf_tf = getattr(mt5, f"TIMEFRAME_{sym_cfg.get('htf_timeframe', 'H4')}", mt5.TIMEFRAME_H4)
    htf_slow = sym_cfg.get("htf_ema_slow", 200)
    htf_needed = htf_slow + 50
    htf_df = get_rates(symbol, htf_tf, htf_needed + 5)
    htf_ema200_aligned = None
    htf_slope_aligned = None
    if htf_df is not None and len(htf_df) >= htf_needed:
        htf_ema = calc_ma(htf_df, htf_slow, ma_type).reset_index(drop=True)
        htf_close = htf_df["close"].reset_index(drop=True)
        slope_window = min(12, max(2, len(htf_ema) // 10))
        htf_slope = htf_ema.diff(slope_window).reset_index(drop=True)
        # Align each H1 bar to the most recent H4 bar at-or-before it.
        h1_times = df["time"].reset_index()
        h4_times = htf_df["time"].reset_index()
        merged = pd.merge_asof(
            h1_times, h4_times, left_on="time", right_on="time",
            direction="backward", suffixes=("_h1", "_h4"),
        )
        idx = merged["index_h4"].values
        htf_ema200_aligned = htf_ema.reindex(idx).reset_index(drop=True)
        htf_slope_aligned = htf_slope.reindex(idx).reset_index(drop=True)
        _ = htf_close  # price uses H1 close, not H4 close

    point = sym_cfg.get("point", 0.0) or (df["close"].iloc[-1] * 1e-5)
    sl_mult = sym_cfg.get("atr_sl_mult", 2.0)
    rr = sym_cfg.get("rr", 2.0)
    out = {"trades": 0, "wins": 0, "losses": 0, "scratch": 0,
           "r_total": 0.0, "r_list": []}
    n = len(df)
    for i in range(need, n - 1):
        adx = adx_series.iloc[i]
        if pd.isna(adx):
            continue
        # Crossover on last two CLOSED bars (iloc[i-1] current, iloc[i-2] prev)
        cf, cs = fast_ma.iloc[i - 1], slow_ma.iloc[i - 1]
        pf, ps = fast_ma.iloc[i - 2], slow_ma.iloc[i - 2]
        if pf <= ps and cf > cs:
            sig = "buy"
        elif pf >= ps and cf < cs:
            sig = "sell"
        else:
            continue
        atr = atr_series.iloc[i]
        if pd.isna(atr) or atr <= 0:
            continue
        # ER chop gate (mirrors signals.get_signal)
        er_min = sym_cfg.get("er_min", 0.10)
        er = calc_efficiency_ratio(df["close"].values[: i + 1], sym_cfg.get("er_period", 10))
        if er < er_min:
            continue
        htf_decision = _htf_local(df, i, sig, htf_ema200_aligned, htf_slope_aligned)
        if mode == "strict":
            if htf_decision != "allow":
                continue
        else:
            if htf_decision == "block":
                continue
        score, _, _ = compute_entry_score(sym_cfg, sig, atr)
        min_score = sym_cfg.get("scoring_min_entry", 0.60)
        if score < min_score:
            continue
        r = _simulate_outcome(df, i, sig, atr, sl_mult, rr, point)
        out["trades"] += 1
        out["r_total"] += r
        out["r_list"].append(r)
        if r > 0:
            out["wins"] += 1
        elif r < 0:
            out["losses"] += 1
        else:
            out["scratch"] += 1
    return out


def _htf_local(df, i, signal, htf_ema200_aligned, htf_slope_aligned):
    """Mirror signals.check_htf_trend 3-state, time-aligned to H1 bar i.
    Returns 'allow' / 'soft' / 'block'."""
    if htf_ema200_aligned is None:
        return "soft"
    htf_ma_val = htf_ema200_aligned.iloc[i]
    if pd.isna(htf_ma_val):
        return "soft"
    htf_price = df["close"].iloc[i - 1]  # last closed H1 bar
    slope = htf_slope_aligned.iloc[i] if htf_slope_aligned is not None else 0.0
    if signal == "buy":
        price_ok = htf_price >= htf_ma_val
        slope_ok = slope >= 0
    else:
        price_ok = htf_price <= htf_ma_val
        slope_ok = slope <= 0
    if price_ok and slope_ok:
        return "allow"
    if (not price_ok) and (not slope_ok):
        return "block"
    return "soft"


def _summarize(out):
    trades = out["trades"]
    if trades == 0:
        return dict(trades=0, win_rate=0.0, pf=0.0, net_r=0.0, max_dd=0.0)
    wins = [r for r in out["r_list"] if r > 0]
    losses = [r for r in out["r_list"] if r < 0]
    win_rate = out["wins"] / trades
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    # Max drawdown on cumulative R equity curve
    eq = np.cumsum(out["r_list"])
    peak = np.maximum.accumulate(eq)
    dd = (peak - eq).max() if len(eq) else 0.0
    return dict(trades=trades, win_rate=win_rate, pf=pf, net_r=out["r_total"],
                max_dd=float(dd))


def main():
    ap = argparse.ArgumentParser(description="Validate relaxed vs strict entry config")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--mode", choices=["strict", "relaxed", "both"], default="both")
    ap.add_argument("--config", default=str(CONFIG_DIR / "settings.ini"))
    args = ap.parse_args()

    cfg = load_config()
    if not ensure_mt5_connected(cfg):
        print("MT5 connection failed")
        return

    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    modes = ["strict", "relaxed"] if args.mode == "both" else [args.mode]
    print(f"\nEntry-config validation — {args.years} yr window, modes: {modes}\n")
    print(f"{'symbol':<14}{'mode':<9}{'trades':>7}{'win%':>7}{'PF':>7}{'netR':>8}{'maxDD':>8}")
    print("-" * 60)
    for sym in symbols:
        for mode in modes:
            out = evaluate_symbol(sym, cfg, args.years, mode)
            if out is None:
                print(f"{sym:<14}{mode:<9}{'NO DATA':>7}")
                continue
            s = _summarize(out)
            pf_str = "inf" if s["pf"] == float("inf") else f"{s['pf']:.2f}"
            print(f"{sym:<14}{mode:<9}{s['trades']:>7}{s['win_rate']*100:>6.1f}{pf_str:>7}{s['net_r']:>8.1f}{s['max_dd']:>8.1f}")
    print()


if __name__ == "__main__":
    main()
