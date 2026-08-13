"""diagnose_entry_rate.py — quantify how restrictive the live ENTRY-SIGNAL pipeline is.

Replays the live signal-generation pipeline over historical H1 bars for one or
more symbols and counts how many candidate signals each gate kills. This
answers "are our entry signals too strict?" with hard numbers instead of
guesswork.

The replay mirrors main.py's actual order:

  Gate 1 — Fused regime hysteresis: `analytics.fused_regime_score` on each
           closed bar fed into the SAME `signals.RegimeGate` class main.py
           uses (threshold +/- buffer/2 bands), advanced every bar.
  Gate 2 — Signal: gate open -> `get_mtf_fused_signal` replay (H4 EMA bias +
           0.5*ATR neutral band, H1 fast/slow crossover agreeing with H4,
           M15 crossover for entry timing — a per-bar mirror of
           backtest._get_mtf_signal); gate closed -> MR (`_mr_local`, replay
           of `signals.get_mean_reversion_signal`), falling back to the MTF /
           single-TF signal when MR produces nothing, exactly like main.py.
  Then   — HTF trend (`signals.check_htf_trend`, live), scoring
           (`analytics.compute_entry_score`, live) and the execution-sanity
           gate (`_exec_sanity_local`).

Approximations (documented): the exec-sanity tape sub-gate needs live M1 bars
and is assumed pass; MR is replayed assuming a flat book (live also requires
no open positions); M15-unavailable symbols degrade to pullback-only MTF
entries (same fallback the backtest and live use). The table is a LOWER bound
on total strictness.

Usage:
    python tools/diagnose_entry_rate.py --symbol XAU500.raw --years 1
    python tools/diagnose_entry_rate.py --symbol BTCUSD.raw,NZDUSD.raw --years 2

Requires a live MT5 terminal.
"""
import argparse
import configparser
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from _common import _crossover_local, _exec_sanity_local, _mr_local, _mtf_signal_local  # noqa: E402
from analytics import compute_entry_score, fused_regime_score  # noqa: E402
from backtest import _get_mtf_m15_mas  # noqa: E402
from indicators import calc_atr_series, calc_ma  # noqa: E402
from mt5_connect import ensure_mt5_connected, fetch_rates_paged, get_rates, mt5  # noqa: E402
from signals import RegimeGate, check_htf_trend  # noqa: E402

from config import load_config, symbol_cfg  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(message)s")
log = logging.getLogger("diagnose")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
GATES = ["signal", "htf", "htf_soft", "score", "exec", "passed"]


def _precompute_mtf_series(df, df_m15, sym_cfg):
    """Precompute the aligned arrays the MTF replay indexes into.

    Mirrors backtest._precompute: H1 fast/slow MAs + ATR series, H4 EMA(period)
    resampled from H1 and ffill-aligned to H1 bar times with a +1-H4-period
    shift (so the ffill picks the last CLOSED H4 bar), and M15 fast/slow MAs
    from the fetched M15 frame with the same closed-bar shift (via the
    backtest's own _get_mtf_m15_mas for bit-parity). Returns a dict of numpy
    arrays; m15 entries are None when M15 data is unavailable.
    """
    ma_type = sym_cfg.get("ma_type", "kama")
    ema_fast = sym_cfg["ema_fast"]
    ema_slow = sym_cfg["ema_slow"]
    out = {
        "close": df["close"].values.astype(float),
        "atr": calc_atr_series(df, sym_cfg["atr_period"]).values.astype(float),
        "fast": calc_ma(df, ema_fast, ma_type).values.astype(float),
        "slow": calc_ma(df, ema_slow, ma_type).values.astype(float),
        "h4_ema": np.full(len(df), np.nan),
        "m15_fast": None,
        "m15_slow": None,
    }
    h4_period = sym_cfg.get("mtf_h4_ema_period", 100)
    h1 = df.set_index("time")
    h4 = (
        h1.resample("4h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "tick_volume": "sum"})
        .dropna(subset=["open"])
    )
    if len(h4) > h4_period:
        h4_ma = calc_ma(h4, h4_period, ma_type)
        if len(h4_ma) > 1:
            h4_shift = h4_ma.index[1] - h4_ma.index[0]
            h4_ma.index = h4_ma.index + h4_shift
        out["h4_ema"] = h4_ma.reindex(df["time"], method="ffill").values.astype(float)
    if df_m15 is not None and len(df_m15) > 0:
        m15_fast = sym_cfg.get("mtf_m15_ema_fast", max(5, ema_fast // 2))
        m15_slow = sym_cfg.get("mtf_m15_ema_slow", max(8, ema_slow // 2))
        m15f, m15s = _get_mtf_m15_mas(df_m15, m15_fast, m15_slow, ma_type)
        out["m15_fast"] = m15f.reindex(df["time"], method="ffill").values.astype(float)
        out["m15_slow"] = m15s.reindex(df["time"], method="ffill").values.astype(float)
    return out


def evaluate_symbol(symbol, cfg, years):
    sym_cfg = symbol_cfg(cfg, symbol)

    # Warmup must cover the H4 EMA(period) (~4 H1 bars per H4 bar) so the MTF
    # path has a valid H4 bias early in the window.
    mtf_need = sym_cfg.get("mtf_h4_ema_period", 100) * 4
    need = max(sym_cfg["ema_slow"], sym_cfg["atr_period"], mtf_need) + sym_cfg["atr_period"] + 60
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

    mtf_enabled = sym_cfg.get("mtf_enabled", True)
    df_m15 = None
    if mtf_enabled:
        # M15 history for the MTF entry-timing gate (fetch_rates_paged covers
        # windows larger than the per-request bar cap).
        try:
            df_m15 = fetch_rates_paged(symbol, mt5.TIMEFRAME_M15, df["time"].iloc[0], df["time"].iloc[-1])
            if df_m15 is not None and len(df_m15) < 100:
                df_m15 = None
        except Exception as e:
            log.warning("[%s] M15 fetch failed: %s", symbol, e)
            df_m15 = None
        if df_m15 is None:
            print(f"  [{symbol}] M15 unavailable — MTF entries degrade to pullback (no M15 timing gate)")

    ser = _precompute_mtf_series(df, df_m15, sym_cfg)
    gate = RegimeGate(
        threshold=sym_cfg.get("fused_threshold", 50.0),
        buffer=sym_cfg.get("fused_buffer", 5.0),
    )
    counts = defaultdict(int)
    n = len(df)
    for i in range(need, n - 1):
        win = df.iloc[: i + 1]

        # Gate 1 — fused regime hysteresis, advanced EVERY bar (main.py parity).
        gate_open = gate.update(fused_regime_score(win, sym_cfg))

        # Gate 2 — signal selection mirrors main.py exactly.
        signal_entry = None
        entry_atr = None
        is_mr = False
        if gate_open:
            if mtf_enabled:
                signal_entry, entry_atr, _etype = _mtf_signal_local(
                    ser["close"], ser["atr"], ser["fast"], ser["slow"],
                    ser["h4_ema"], ser["m15_fast"], ser["m15_slow"], i,
                )
            else:
                signal_entry, entry_atr = _crossover_local(win, sym_cfg)
        else:
            if sym_cfg.get("mr_enabled", True):
                signal_entry, entry_atr = _mr_local(win, sym_cfg)
            if signal_entry is not None:
                is_mr = True
            elif mtf_enabled:
                signal_entry, entry_atr, _etype = _mtf_signal_local(
                    ser["close"], ser["atr"], ser["fast"], ser["slow"],
                    ser["h4_ema"], ser["m15_fast"], ser["m15_slow"], i,
                )
            else:
                signal_entry, entry_atr = _crossover_local(win, sym_cfg)
        if signal_entry is None:
            continue
        counts["signal"] += 1
        htf_decision, htf_size_mult = check_htf_trend(sym_cfg, signal_entry)
        if htf_decision == "block":
            counts["htf"] += 1
            continue
        if htf_decision == "soft":
            counts["htf_soft"] += 1
        if sym_cfg.get("scoring_enabled", True):
            score, _, _ml_conf = compute_entry_score(sym_cfg, signal_entry, entry_atr)
            min_score = sym_cfg.get("scoring_min_entry", 0.55) + (0.03 if is_mr else 0.0)
            if score < min_score:
                counts["score"] += 1
                continue
        if not _exec_sanity_local(win, signal_entry, sym_cfg):
            counts["exec"] += 1
            continue
        counts["passed"] += 1
    return counts


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
    print(f"{'symbol':<14}{'signal':>8}{'htf':>6}{'soft':>6}{'score':>7}{'exec':>6}{'pass':>6}{'pass%':>7}")
    print("-" * 64)
    for sym in symbols:
        counts = evaluate_symbol(sym, cfg, args.years)
        if counts is None:
            continue
        passed = counts["passed"]
        sig = counts["signal"]
        rate = (passed / sig * 100) if sig else 0.0
        print(
            f"{sym:<14}{sig:>8}{counts['htf']:>6}{counts['htf_soft']:>6}"
            f"{counts['score']:>7}{counts['exec']:>6}{passed:>6}{rate:>7.1f}"
        )
        for k in GATES:
            total[k] += counts[k]
    if len(symbols) > 1:
        tsig = total["signal"]
        trate = (total["passed"] / tsig * 100) if tsig else 0.0
        print("-" * 64)
        print(
            f"{'TOTAL':<14}{tsig:>8}{total['htf']:>6}{total['htf_soft']:>6}"
            f"{total['score']:>7}{total['exec']:>6}{total['passed']:>6}{trate:>7.1f}"
        )
    print("\nNote: tape / tail_risk filters are NOT replayed (live-state only);")
    print("MR is replayed assuming a flat book. 'pass' is a LOWER bound.\n")


if __name__ == "__main__":
    main()
