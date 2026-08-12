"""Shared helpers for tools/ scripts (run as standalone programs).

Deduplicated here — the local replay mirrors diagnose_entry_rate.py once
copy-pasted: `_crossover_local` (KAMA/VIDYA crossover), `_mr_local`
(mean-reversion decision, faithful replay of signals.get_mean_reversion_signal),
`_exec_sanity_local` (replay of filters.check_execution_sanity) and
`_mtf_signal_local` (per-bar mirror of backtest._get_mtf_signal / live
get_mtf_fused_signal). Imports bot/indicators + bot/analytics, so bot/ is
added to sys.path defensively.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from analytics import volume_filter_pass  # noqa: E402
from indicators import calc_atr, calc_ma, calc_rsi  # noqa: E402


def _mtf_signal_local(close_a, atr_a, f_a, s_a, h4_a, m15f_a, m15s_a, i):
    """Per-bar mirror of backtest._get_mtf_signal / live get_mtf_fused_signal.

    H4 EMA(period) bias with a 0.5*ATR neutral band, H1 fast/slow crossover
    must agree with the H4 direction, and M15 crossover provides entry timing
    ("crossover" entry) when it agrees; otherwise a "pullback" entry. With no
    M15 series (None), falls back to the pullback entry — the same degraded
    path the backtest and live take when M15 data is unavailable.

    ``i`` is a closed H1 bar index into precomputed aligned arrays (see
    diagnose_entry_rate._precompute_mtf_series). Returns (signal, atr,
    entry_type) or (None, None, None).
    """
    h4_ema = h4_a[i]
    h1_close = close_a[i]
    cur_atr = atr_a[i]
    if np.isnan(h4_ema) or np.isnan(h1_close) or np.isnan(cur_atr) or cur_atr <= 0 or i < 2:
        return None, None, None

    # H4 bias with 0.5*ATR neutral band
    bias = h1_close - float(h4_ema)
    neutral_band = cur_atr * 0.5
    if abs(bias) <= neutral_band:
        return None, None, None
    h4_direction = 1 if bias > 0 else -1

    # H1 crossover must agree with H4 bias
    h1_cf, h1_cs = f_a[i], s_a[i]
    h1_pf, h1_ps = f_a[i - 1], s_a[i - 1]
    if any(np.isnan(x) for x in (h1_cf, h1_cs, h1_pf, h1_ps)):
        return None, None, None
    h1_cross = 0
    if h1_pf <= h1_ps and h1_cf > h1_cs:
        h1_cross = 1
    elif h1_pf >= h1_ps and h1_cf < h1_cs:
        h1_cross = -1
    if h1_cross != h4_direction:
        return None, None, None

    direction = "buy" if h1_cross > 0 else "sell"

    # M15 crossover check for entry timing
    if m15f_a is not None and m15s_a is not None and i > 0:
        m15_cf, m15_cs = m15f_a[i], m15s_a[i]
        m15_pf, m15_ps = m15f_a[i - 1], m15s_a[i - 1]
        if not any(np.isnan(x) for x in (m15_cf, m15_cs, m15_pf, m15_ps)):
            m15_cross = 0
            if m15_pf <= m15_ps and m15_cf > m15_cs:
                m15_cross = 1
            elif m15_pf >= m15_ps and m15_cf < m15_cs:
                m15_cross = -1
            if m15_cross == h1_cross:
                return direction, cur_atr, "crossover"
            elif m15_cross != 0:
                return None, None, None

    return direction, cur_atr, "pullback"


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
    return signal, atr


def _mr_local(win, sym_cfg):
    """Replay of signals.get_mean_reversion_signal's decision on one window.
    The live function sources RSI from the MR timeframe (M30) and the HTF MA
    from htf_timeframe (H4); this replay collapses both onto the same H1
    window for speed, keeping the identical threshold/decision logic.

    NOT replayed: the MR cooldown (main.py / backtest A3) — it is stateful
    across trades (>=2 consecutive MR losses suppress MR for mr_cooldown_bars
    H1 bars) and needs per-trade P&L the replay does not simulate."""
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


def _exec_sanity_local(win, signal, sym_cfg):
    """Replay of filters.check_execution_sanity (Gate 4) on one window.

    Backtest parity (backtest._check_execution_sanity, which mirrors the live
    gate): volume via the shared analytics.volume_filter_pass on the window's
    last (closed) bar, then the bar-spread gate (bar spread * point / ATR vs
    spf_max_ratio). The tape sub-gate needs live M1 bars the replay does not
    fetch, so it is assumed to pass — the same fallback live and the backtest
    use when M1 data is unavailable.

    ``win`` must contain the current closed bar as its last row; ``point``
    defaults to 0.01 (the tool's XAU context) when the config lacks it.
    Returns True if the gate admits the signal.
    """
    if not volume_filter_pass(win, signal, sym_cfg):
        return False
    if sym_cfg.get("spf_enabled", True) and "spread" in win.columns:
        spread = win["spread"].iloc[-1]
        if spread is not None and spread > 0:
            atr = calc_atr(win, sym_cfg["atr_period"])
            if atr is not None and atr > 0:
                point = sym_cfg.get("point", 0.01)
                if (spread * point) / atr > sym_cfg.get("spf_max_ratio", 0.30):
                    return False
    return True
