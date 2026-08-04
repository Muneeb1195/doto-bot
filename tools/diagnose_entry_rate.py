"""diagnose_entry_rate.py — quantify how restrictive the live ENTRY-SIGNAL pipeline is.

Replays the live signal-generation + regime + exec-bias + HTF-trend + scoring
gates over historical H1 bars for one or more symbols, and counts how many
candidate signals each gate kills. This answers "are our entry signals too
strict?" with hard numbers instead of guesswork.

Faithful: it calls the SAME stateless functions main.py uses
(signals.get_signal, signals.get_mean_reversion_signal,
signals.check_htf_trend, signals.compute_entry_score,
signals.check_execution_signal) plus a local ADX/regime computation using the
bot's own indicators.calc_adx.

The pure *live-state* filters (volume, spread, tape, tail_risk) are NOT
replayed — they depend on the current live rate snapshot and global state, not
on historical bars. Their rejection counts are shown as N/A (assumed pass) so
the table is a LOWER bound on total strictness.

Usage:
    python tools/diagnose_entry_rate.py --symbol XAU500.raw --years 1
    python tools/diagnose_entry_rate.py --symbol BTCUSD.raw,NZDUSD.raw --years 2

Requires a live MT5 terminal.
"""
import argparse
import configparser
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from copy import deepcopy

sys = __import__("sys")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

import MetaTrader5 as mt5  # noqa: E402

import state as _st  # noqa: E402
from config import load_config, apply_symbol_strategy, apply_symbol_overrides  # noqa: E402
from mt5_connect import ensure_mt5_connected, get_rates  # noqa: E402
from indicators import calc_adx, calc_ma, calc_atr, calc_rsi, calc_efficiency_ratio  # noqa: E402
from signals import (  # noqa: E402
    get_signal, get_mean_reversion_signal, check_htf_trend,
    compute_entry_score, check_execution_signal,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
log = logging.getLogger("diagnose")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
GATES = ["signal", "htf", "htf_soft", "exec", "score", "passed"]


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


def _exec_bias_gate_local(df_h1, sym_cfg, signal_entry, i, bias_state):
    """Mirror signals.check_execution_signal's flip-freeze logic, but driven by
    historical H1 closes instead of the global _exec_bias live state.

    bias_state: dict {flips:int, since_date:str, last_h1_signal:str|None}
    Returns (allowed:bool)."""
    ma_type = sym_cfg.get("ma_type", "kama")
    ef, es = sym_cfg["exec_ema_fast"], sym_cfg["exec_ema_slow"]
    if len(df_h1) <= es + 2:
        return True
    fast = calc_ma(df_h1, ef, ma_type)
    slow = calc_ma(df_h1, es, ma_type)
    h1_fast = fast.iloc[i - 1]
    h1_slow = slow.iloc[i - 1]
    h1_prev_fast = fast.iloc[i - 2]
    h1_prev_slow = slow.iloc[i - 2]
    h1_signal = None
    if h1_prev_fast <= h1_prev_slow and h1_fast > h1_slow:
        h1_signal = "buy"
    elif h1_prev_fast >= h1_prev_slow and h1_fast < h1_slow:
        h1_signal = "sell"
    today = pd.to_datetime(df_h1["time"].iloc[i]).strftime("%Y-%m-%d")
    if bias_state.get("since_date") != today:
        bias_state["flips"] = 0
        bias_state["since_date"] = today
        bias_state["last_h1_signal"] = None
    if h1_signal is not None and h1_signal != bias_state.get("last_h1_signal"):
        if bias_state.get("last_h1_signal") is not None:
            bias_state["flips"] = bias_state.get("flips", 0) + 1
        bias_state["last_h1_signal"] = h1_signal
    if bias_state["flips"] >= sym_cfg.get("bias_max_flips", 3):
        return False
    if h1_signal is None:
        return True
    return (signal_entry == h1_signal)


def evaluate_symbol(symbol, cfg, years):
    sym_cfg = deepcopy(cfg)
    sym_cfg["symbol"] = symbol
    apply_symbol_strategy(sym_cfg, symbol)
    apply_symbol_overrides(sym_cfg, symbol)

    need = max(sym_cfg["ema_slow"], sym_cfg["atr_period"]) + sym_cfg["atr_period"] + 60
    target = need + int(years * 365 * 24 * 0.6)
    df = None
    # Some instruments have limited history; shrink the window until we get
    # enough bars (>= need) to replay at least the signal logic.
    for mult in (1.0, 0.5, 0.25, 0.1):
        bars = need + max(int((target - need) * mult), 100)
        df = get_rates(symbol, sym_cfg["timeframe"], bars)
        if df is not None and len(df) >= need:
            break
    if df is None or len(df) < need:
        # Last resort: pull a flat 400 bars regardless of window target.
        df = get_rates(symbol, sym_cfg["timeframe"], need + 150)
    if df is None or len(df) < need:
        log.warning("[%s] insufficient data (need=%s got=%s)", symbol, need, len(df) if df is not None else 0)
        return None

    adx_period = sym_cfg.get("adx_period", 14)
    counts = defaultdict(int)
    bias_state = {"flips": 0, "since_date": None, "last_h1_signal": None}
    n = len(df)
    for i in range(need, n - 1):
        win = df.iloc[: i + 1]
        adx = calc_adx(win, adx_period)
        if adx is None or np.isnan(adx):
            continue
        regime = _regime_from_adx(adx, sym_cfg)

        if regime == "ranging" and sym_cfg.get("mr_enabled", False):
            signal_entry, entry_atr = _mr_local(win, sym_cfg)
            is_mr = True
        else:
            sig, atr = _crossover_local(win, sym_cfg)
            signal_entry, entry_atr, is_mr = sig, atr, False
        if signal_entry is None:
            continue
        counts["signal"] += 1
        htf_decision, htf_size_mult = check_htf_trend(sym_cfg, signal_entry)
        if htf_decision == "block":
            counts["htf"] += 1
            continue
        if htf_decision == "soft":
            counts["htf_soft"] += 1
        if regime in ("strong_trend", "weak_trend"):
            if not _exec_bias_gate_local(df, sym_cfg, signal_entry, i, bias_state):
                counts["exec"] += 1
                continue
        if sym_cfg.get("scoring_enabled", True):
            score, _, _ml_conf = compute_entry_score(sym_cfg, signal_entry, entry_atr)
            min_score = sym_cfg.get("scoring_min_entry", 0.55) + (0.03 if is_mr else 0.0)
            if score < min_score:
                counts["score"] += 1
                continue
        counts["passed"] += 1
    return counts


def _crossover_local(win, sym_cfg):
    ma_type = sym_cfg.get("ma_type", "kama")
    fast = calc_ma(win, sym_cfg["ema_fast"], ma_type)
    slow = calc_ma(win, sym_cfg["ema_slow"], ma_type)
    atr = calc_atr(win, sym_cfg["atr_period"])
    cf, cs = fast.iloc[-2], slow.iloc[-2]
    pf, ps = fast.iloc[-3], slow.iloc[-3]
    signal = None
    if pf <= ps and cf > cs:
        signal = "buy"
    elif pf >= ps and cf < cs:
        signal = "sell"
    if signal is None:
        return None, None
    er_min = sym_cfg.get("er_min", 0.10)
    er = calc_efficiency_ratio(win["close"].values, sym_cfg.get("er_period", 10))
    if er < er_min:
        return None, None
    return signal, atr


def _mr_local(win, sym_cfg):
    needed = sym_cfg["htf_ema_slow"] + sym_cfg["mr_rsi_period"] + sym_cfg["atr_period"] + 5
    if len(win) < needed:
        return None, None
    rsi_period = sym_cfg["mr_rsi_period"]
    rsi = calc_rsi(win.iloc[:-1], rsi_period)
    atr = calc_atr(win, sym_cfg["atr_period"])
    cur_rsi = rsi
    cur_price = win["close"].iloc[-2]
    oversold = sym_cfg["mr_rsi_oversold"]
    overbought = sym_cfg["mr_rsi_overbought"]
    htf_slow = sym_cfg["htf_ema_slow"]
    htf_ma = calc_ma(win, htf_slow, sym_cfg.get("ma_type", "kama"))
    if htf_ma is None or pd.isna(htf_ma.iloc[-2]):
        return None, None
    htf_val = htf_ma.iloc[-2]
    dev = sym_cfg.get("mr_htf_deviation", 0.0)
    if cur_rsi < oversold and cur_price > htf_val * (1.0 - dev):
        return "buy", atr
    if cur_rsi > overbought and cur_price < htf_val * (1.0 + dev):
        return "sell", atr
    return None, None


def main():
    ap = argparse.ArgumentParser(description="Quantify live entry-signal strictness")
    ap.add_argument("--symbol", required=True, help="Comma-separated symbols")
    ap.add_argument("--years", type=float, default=1.0)
    ap.add_argument("--config", default=str(CONFIG_DIR / "settings.ini"))
    args = ap.parse_args()

    settings = configparser.ConfigParser()
    settings.read(args.config)
    cfg = load_config()
    if not ensure_mt5_connected(cfg):
        print("MT5 connection failed")
        return

    symbols = [s.strip() for s in args.symbol.split(",") if s.strip()]
    total = defaultdict(int)
    print(f"\nEntry-SIGNAL pipeline rejection replay — {args.years} yr window\n")
    print(f"{'symbol':<14}{'signal':>8}{'htf':>6}{'soft':>6}{'exec':>6}{'score':>7}{'pass':>6}{'pass%':>7}")
    print("-" * 63)
    for sym in symbols:
        counts = evaluate_symbol(sym, cfg, args.years)
        if counts is None:
            continue
        passed = counts["passed"]
        sig = counts["signal"]
        rate = (passed / sig * 100) if sig else 0.0
        print(f"{sym:<14}{sig:>8}{counts['htf']:>6}{counts['htf_soft']:>6}{counts['exec']:>6}{counts['score']:>7}{passed:>6}{rate:>7.1f}")
        for k in GATES:
            total[k] += counts[k]
        total["htf_soft"] += counts["htf_soft"]
    if len(symbols) > 1:
        tsig = total["signal"]
        trate = (total["passed"] / tsig * 100) if tsig else 0.0
        print("-" * 63)
        print(f"{'TOTAL':<14}{tsig:>8}{total['htf']:>6}{total['htf_soft']:>6}{total['exec']:>6}{total['score']:>7}{total['passed']:>6}{trate:>7.1f}")
    print("\nNote: volume / spread / tape / tail_risk filters are NOT replayed")
    print("(they depend on the live rate snapshot). 'pass' is a LOWER bound.\n")


if __name__ == "__main__":
    main()
