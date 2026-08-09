"""Numba-JIT compiled core for the Backtest event loop.

This module mirrors the pure-Python `Backtest.run()` simulation loop in
`backtest.py` exactly, but operates on plain numpy arrays + scalar state so it
can be compiled to native machine code (50-200x faster). It exists purely as a
performance fast-path; the reference Python loop in `backtest.py` remains the
canonical implementation and is used for parity tests + `--no-fast` fallback.

The contract is bit-exact: given the same precomputed arrays and params, the
trade list + equity curve produced here must equal `Backtest.run(fast=False)`.
"""

import math

import numpy as np
from numba import njit

# ---- Integer code maps (must match backtest.py semantics) ----
# regime: 0 uncertain, 1 strong_trend, 2 weak_trend, 3 ranging, 4 exhaustion
# entry_type: 0 crossover, 1 pullback, 2 mean_reversion
# exit_reason: 0 SL, 1 TP, 2 CHANDELIER, 3 SCALE_OUT, 4 MR_EXIT, 5 REVERSAL, 6 END
# trade type: 0 buy, 1 sell, 2 partial

_REGIME_NAME = {0: "uncertain", 1: "strong_trend", 2: "weak_trend", 3: "ranging", 4: "exhaustion"}
_ENTRY_TYPE_NAME = {0: "crossover", 1: "pullback", 2: "mean_reversion"}
_EXIT_REASON_NAME = {0: "SL", 1: "TP", 2: "CHANDELIER", 3: "SCALE_OUT", 4: "MR_EXIT", 5: "REVERSAL", 6: "END"}


# ---- Multi-timeframe fused signal (njit core) ----
# Mirrors Backtest._get_mtf_signal() bit-for-bit. Pure function of the 5
# time-aligned MA arrays at indices i and i-1. Weights are fixed
# (M15=1, H1=2, H4=3) to match the backtest reference exactly.
# Returns (signal, entry_type, confidence):
#   signal:     0 buy, 1 sell, -1 none
#   entry_type: 0 crossover (MTF never emits pullback/MR)
#   confidence: buy_ratio or sell_ratio that passed the agreement threshold
@njit(cache=True, fastmath=False)
def _mtf_signal_at(
    i,
    ema_fast_a,
    ema_slow_a,
    mtf_m15_fast_a,
    mtf_m15_slow_a,
    mtf_h4_ema_a,
    close_a,
    atr_a,
):
    """Hierarchical bias+trigger MTF signal (AGENTS.md).

    H4: price vs EMA(100) with 0.5*ATR neutral band determines bias.
    H1: MA crossover must agree with H4 direction.
    M15: MA crossover provides entry timing when aligned.

    Returns (signal, entry_type, confidence):
      signal:     0 buy, 1 sell, -1 none
      entry_type: 0 crossover, 1 pullback
      confidence: 1.0 all 3 TFs agree, 0.67 H4+H1 only (pullback allowed)
    """
    h4_ema = mtf_h4_ema_a[i]
    h1_close = close_a[i]
    h1_atr = atr_a[i]
    if (
        np.isnan(h4_ema)
        or np.isnan(h1_close)
        or np.isnan(h1_atr)
        or h1_atr <= 0
        or i < 2
    ):
        return -1, 0, 0.0

    # H4 bias: close vs EMA with 0.5*ATR neutral band
    bias = h1_close - h4_ema
    neutral_band = h1_atr * 0.5
    if abs(bias) <= neutral_band:
        return -1, 0, 0.0
    h4_direction = 1 if bias > 0 else -1

    # H1 crossover must agree with H4 bias
    h1_cf = ema_fast_a[i]
    h1_cs = ema_slow_a[i]
    h1_pf = ema_fast_a[i - 1]
    h1_ps = ema_slow_a[i - 1]
    if np.isnan(h1_cf) or np.isnan(h1_cs) or np.isnan(h1_pf) or np.isnan(h1_ps):
        return -1, 0, 0.0
    h1_cross = 0
    if h1_pf <= h1_ps and h1_cf > h1_cs:
        h1_cross = 1
    elif h1_pf >= h1_ps and h1_cf < h1_cs:
        h1_cross = -1
    if h1_cross != h4_direction:
        return -1, 0, 0.0

    signal = 0 if h1_cross > 0 else 1

    # M15 entry timing check
    m15_cross = 0
    if i > 0:
        m15_cf = mtf_m15_fast_a[i]
        m15_cs = mtf_m15_slow_a[i]
        m15_pf = mtf_m15_fast_a[i - 1]
        m15_ps = mtf_m15_slow_a[i - 1]
        if not (np.isnan(m15_cf) or np.isnan(m15_cs) or np.isnan(m15_pf) or np.isnan(m15_ps)):
            if m15_pf <= m15_ps and m15_cf > m15_cs:
                m15_cross = 1
            elif m15_pf >= m15_ps and m15_cf < m15_cs:
                m15_cross = -1

    if m15_cross == h1_cross:
        return signal, 0, 1.0  # crossover entry
    elif m15_cross != 0:
        return -1, 0, 0.0  # M15 opposes — block
    else:
        return signal, 1, 0.67  # pullback entry


@njit(cache=True, fastmath=False)
def _htf_decision(i, signal, htf_ema_a, htf_close_a, htf_slope_a, bar_close, misalign_mult):
    """3-state HTF decision mirroring Backtest._check_htf_trend.

    Returns a size multiplier, or -1.0 meaning 'block'.
      allow -> 1.0
      soft  -> misalign_mult
      block -> -1.0

    Parity notes:
      * A NaN HTF EMA is 'soft' (not a hard allow).
      * htf_price comes from the time-aligned H4 close when available, falling
        back to the current H1 close — the reference reads htf_close_aligned,
        NOT the H1 bar close.
      * A NaN slope stays NaN so the >=/<= comparisons evaluate False, which is
        what makes the bar count as misaligned rather than aligned.
    """
    if np.isnan(htf_ema_a[i]):
        return misalign_mult
    htf_price = bar_close
    if not np.isnan(htf_close_a[i]):
        htf_price = htf_close_a[i]
    slope = htf_slope_a[i]
    if signal == 0:
        price_ok = htf_price >= htf_ema_a[i]
        slope_ok = slope >= 0
    else:
        price_ok = htf_price <= htf_ema_a[i]
        slope_ok = slope <= 0
    if price_ok and slope_ok:
        return 1.0
    if (not price_ok) and (not slope_ok):
        return -1.0
    return misalign_mult


@njit(cache=True, fastmath=False)
def _simulate_core(
    n,
    warmup,
    point,
    tick_value,
    slippage,
    spm,
    commission,
    vol_step,
    initial_balance,
    risk_percent,
    max_risk_ratio,
    stops_level,
    open_a,
    high_a,
    low_a,
    close_a,
    spread_a,
    vol_a,
    day_start_idx_a,
    ema_fast_a,
    ema_slow_a,
    atr_a,
    adx_a,
    h4_adx_a,
    d1_adx_a,
    htf_ema_a,
    htf_close_a,
    htf_slope_a,
    ml_buy_a,
    ml_sell_a,
    vol_sma_a,
    atr_sma_a,  # noqa: ARG001
    atr_sma50_a,
    ch_accel_a,
    mr_rsi_a,  # noqa: ARG001
    fused_score_a,
    mtf_m15_fast_a,
    mtf_m15_slow_a,
    mtf_h4_ema_a,
    bar_hour_a,
    # scalar params bundle
    P_atr_sl_mult,
    P_rr,
    P_mr_sl_mult,
    P_mr_tp_mult,
    P_ch_enabled,
    P_ch_mult,
    P_ch_two_stage,
    P_ch_tight_mult,
    P_ch_loose_mult,
    P_ch_partial_mult,
    P_ch_accel_enabled,
    P_ch_accel_bars,
    P_ch_accel_strength,
    P_ch_two_stage_min_r,
    P_scale_out_enabled,
    P_scale_out_f0,
    P_scale_out_f1,
    P_scale_out_tp0,
    P_scale_out_tp1,
    P_scale_out_be_frac,
    P_mr_enabled,
    P_mr_rsi_period,
    P_mr_rsi_oversold,
    P_mr_rsi_overbought,
    P_mr_htf_deviation,
    P_mr_position_size_mult,
    P_mr_cooldown_enabled,
    P_mr_cooldown_bars,
    P_pb_enabled,
    P_pb_atr_mult,
    P_pb_vol_threshold,
    P_pb_vol_sma_period,
    P_pb_structure_lookback,
    P_pb_atr_min_dist,
    P_vol_filter,
    P_volume_kappa,
    P_vf_obv_lookback,
    P_vf_obv_enabled,
    P_spread_filter,
    P_spf_max_ratio,
    P_session_enabled,
    P_session_only,
    P_require_overlap,
    P_skip_asian,
    P_london_open,
    P_london_close,
    P_asian_open,
    P_asian_close,
    P_adx_trend,
    P_adx_range,
    P_exhaustion_adx,
    P_exhaustion_slope,
    P_fused_threshold,
    P_fused_buffer,
    P_ml_enabled,
    P_scoring_enabled,
    P_scoring_min_entry,
    P_scoring_high,  # noqa: ARG001
    P_scoring_low,  # noqa: ARG001
    P_sc_high_mult,  # noqa: ARG001
    P_sc_std_mult,  # noqa: ARG001
    P_sc_low_mult,  # noqa: ARG001
    P_dr_enabled,
    P_dr_lookback,
    P_dr_kelly_fraction,
    P_dr_min_mult,
    P_dr_max_mult,
    P_dr_vol_adjust,
    P_tr_enabled,
    P_tr_lookback,
    P_tr_sigma,
    P_tr_max_dd_pct,
    P_cb_dd_pct,
    P_daily_loss_pct,
    P_max_positions_per_symbol,
    P_corr_size_mult,
    P_correlation_enabled,
    P_htf_misalign_size_mult,
    P_mtf_enabled,
    score_a,
    conf_mult_a,
):
    """Run the event-loop state machine. Returns trade arrays + equity curve."""
    max_t = n + 2
    t_type = np.empty(max_t, dtype=np.int64)
    t_entry = np.zeros(max_t)
    t_sl = np.zeros(max_t)
    t_tp = np.zeros(max_t)
    t_entry_bar = np.empty(max_t, dtype=np.int64)
    t_exit_bar = np.empty(max_t, dtype=np.int64)
    t_exit = np.zeros(max_t)
    t_pnl = np.zeros(max_t)
    t_volume = np.zeros(max_t)
    t_rem_vol = np.zeros(max_t)
    t_exit_reason = np.empty(max_t, dtype=np.int64)
    t_regime = np.empty(max_t, dtype=np.int64)
    t_entry_type = np.empty(max_t, dtype=np.int64)
    t_atr_entry = np.zeros(max_t)
    t_count = 0

    equity = np.empty(n, dtype=np.float64)

    cumulative_pnl = 0.0
    commission_paid = 0.0

    closed_pnl = np.empty(max_t, dtype=np.float64)
    closed_bar = np.empty(max_t, dtype=np.int64)
    closed_type = np.empty(max_t, dtype=np.int64)
    closed_entry_type = np.empty(max_t, dtype=np.int64)
    closed_count = 0
    mr_loss_streak = 0
    mr_last_loss_i = -1000

    fused_gate_open = False

    run_max_eq = initial_balance
    tr_needed = int(P_tr_lookback) + 10

    pos_open = False
    pos_is_long = False
    pos_entry = 0.0
    pos_sl = 0.0
    pos_tp = 0.0
    pos_entry_bar = 0
    pos_volume = 0.0
    pos_rem_vol = 0.0
    pos_regime = 0
    pos_entry_type = 0
    pos_atr_entry = 0.0
    pos_scale_step = 0
    pos_partial_fired = False
    pos_ch_sl = 0.0

    dbg = np.zeros(14)
    sig_out = np.full(n, -99.0)
    et_out = np.zeros(n)
    pend_sl_dbg = np.full(n, -1.0)
    skip_low_dbg = np.full(n, -1.0)
    pending_entry = False
    pe_type = 0
    pe_entry_type = 0
    pe_sl = 0.0
    pe_tp = 0.0
    pe_volume = 0.0
    pe_regime = 0
    pe_atr_entry = 0.0
    pe_sl_points = 0.0
    pe_tp_points = 0.0

    for i in range(warmup, n):
        bar_open = open_a[i]
        bar_high = high_a[i]
        bar_low = low_a[i]
        bar_close = close_a[i]
        bar_spread = spread_a[i]
        cur_atr = atr_a[i]

        if pending_entry:
            pending_entry = False
            is_long_pe = pe_type == 0
            # Compute entry price and recompute SL/TP from actual entry price
            # BEFORE the gap-through check (parity with reference path). Was
            # computed from signal close — creating variable risk distance.
            if is_long_pe:
                entry_price = bar_open + spm * bar_spread * point * 0.5 + slippage * point
            else:
                entry_price = bar_open - spm * bar_spread * point * 0.5 - slippage * point
            pe_sl = (entry_price - pe_sl_points * point) if is_long_pe else (entry_price + pe_sl_points * point)
            pe_tp = (entry_price + pe_tp_points * point) if is_long_pe else (entry_price - pe_tp_points * point)
            # NOTE: previously a same-bar SL hit (bar low <= SL on the
            # entry bar) discarded the trade entirely, understating losses.
            # Training labels (triple_barrier_labels) always OPEN the trade
            # and only start checking SL/TP from the next bar (j >= 1), so
            # we now open the position unconditionally and let the exit logic
            # (which skips the entry bar via pos_entry_bar != i below)
            # evaluate it from the following bar. Mirrors _run_reference.
            pos_open = True
            pos_is_long = is_long_pe
            pos_entry = entry_price
            pos_sl = pe_sl
            pos_tp = pe_tp
            pos_entry_bar = i
            pos_volume = pe_volume
            pos_rem_vol = pe_volume
            pos_regime = pe_regime
            pos_entry_type = pe_entry_type
            pos_atr_entry = pe_atr_entry
            pos_scale_step = 0
            pos_partial_fired = False
            pos_ch_sl = 0.0
            dbg[3] += 1
            if commission > 0:
                cumulative_pnl -= commission * pe_volume
                commission_paid += commission * pe_volume

        if np.isnan(cur_atr) or cur_atr <= 0:
            equity[i] = initial_balance + cumulative_pnl
            continue

        if np.isnan(adx_a[i]):
            regime = 0
        else:
            adx_h1 = adx_a[i]
            h4_t = h4_adx_a[i]
            d1_t = d1_adx_a[i]
            adx_slope = 0.0
            if i >= 5:
                vals = np.empty(6)
                cnt = 0
                for k in range(6):
                    v = adx_a[i - 5 + k]
                    if not np.isnan(v):
                        vals[cnt] = v
                        cnt += 1
                if cnt > 5:
                    adx_slope = vals[cnt - 1] - vals[0]
            h4_trending = (not np.isnan(h4_t)) and (h4_t >= P_adx_range)
            d1_trending = (not np.isnan(d1_t)) and (d1_t >= P_adx_range)
            exhaustion = (
                (adx_h1 >= P_exhaustion_adx) and (not np.isnan(adx_slope)) and (adx_slope < -P_exhaustion_slope)
            )
            if exhaustion:
                regime = 4
            elif adx_h1 >= P_adx_trend and (h4_trending or d1_trending):
                regime = 1
            elif adx_h1 >= P_adx_trend:
                regime = 2
            elif adx_h1 <= P_adx_range and (not h4_trending) and (not d1_trending):
                regime = 3
            else:
                regime = 0

        cur_fast = ema_fast_a[i]
        cur_slow = ema_slow_a[i]
        prev_fast = ema_fast_a[i - 1] if i > 0 else cur_fast
        prev_slow = ema_slow_a[i - 1] if i > 0 else cur_slow

        buy_signal = (prev_fast <= prev_slow) and (cur_fast > cur_slow)
        sell_signal = (prev_fast >= prev_slow) and (cur_fast < cur_slow)

        # Multi-timeframe fused signal (when enabled) replaces the plain H1
        # crossover as the crossover source. Mirrors Backtest._get_mtf_signal().
        mtf_etype = 0
        if P_mtf_enabled > 0:
            mtf_sig, mtf_etype, mtf_conf = _mtf_signal_at(
                i,
                ema_fast_a,
                ema_slow_a,
                mtf_m15_fast_a,
                mtf_m15_slow_a,
                mtf_h4_ema_a,
                close_a,
                atr_a,
            )
            if mtf_sig == 0:
                buy_signal = True
                sell_signal = False
            elif mtf_sig == 1:
                buy_signal = False
                sell_signal = True
            else:
                buy_signal = False
                sell_signal = False

        signal = -1
        entry_type = 0
        entry_atr = cur_atr
        # dbg[13] = bars reaching signal evaluation. This used to increment
        # dbg[6], which is the HTF-block counter, making htf_block read ~11.5k
        # against only ~600 evaluated entries and masking real gate behaviour.
        dbg[13] += 1

        if buy_signal:
            signal = 0
            entry_type = int(mtf_etype) if P_mtf_enabled > 0 else 0
        elif sell_signal:
            signal = 1
            entry_type = int(mtf_etype) if P_mtf_enabled > 0 else 0

        if (
            signal == -1
            and (regime == 0 or regime == 1 or regime == 2)
            and P_pb_enabled > 0
            and i >= 3
            and (not np.isnan(cur_atr))
            and cur_atr > 0
        ):
            trigger_fast = ema_fast_a[i - 1]
            trigger_slow = ema_slow_a[i - 1]
            if not (np.isnan(trigger_fast) or np.isnan(trigger_slow)):
                trigger_price = close_a[i - 1]
                pb_dist = cur_atr * P_pb_atr_mult
                min_pb = cur_atr * P_pb_atr_min_dist
                signal_dir = -1
                if trigger_fast > trigger_slow:
                    d = abs(trigger_price - trigger_fast)
                    if min_pb <= d <= pb_dist:
                        signal_dir = 0
                elif trigger_fast < trigger_slow:
                    d = abs(trigger_price - trigger_fast)
                    if min_pb <= d <= pb_dist:
                        signal_dir = 1
                if signal_dir != -1:
                    vol_ok = True
                    # Parity with _pb_volume_check: the reference computes its OWN
                    # rolling mean over pb_volume_sma_period ending at the trigger
                    # bar. Reusing vol_sma_a (built from vf_sma_period, and NaN
                    # during warm-up) is a different series whenever the two
                    # periods differ, and mishandles the warm-up window — the
                    # reference returns True (pass) when trigger_idx < period.
                    if P_pb_vol_threshold > 0:
                        pbp = int(P_pb_vol_sma_period)
                        ti = i - 1
                        if ti >= pbp:
                            vsum = 0.0
                            for k in range(ti - pbp + 1, ti + 1):
                                vsum += vol_a[k]
                            pb_sma = vsum / pbp
                            if pb_sma > 0:
                                vol_ok = vol_a[ti] < pb_sma * P_pb_vol_threshold
                    struct_ok = True
                    lb = int(P_pb_structure_lookback)
                    start = i - 1 - lb
                    if start >= 0:
                        if signal_dir == 0:
                            struct_ok = low_a[i - 1] > np.min(low_a[start : i - 1])
                        else:
                            struct_ok = high_a[i - 1] < np.max(high_a[start : i - 1])
                    confirm_ok = (signal_dir == 0 and close_a[i] > high_a[i - 1]) or (
                        signal_dir == 1 and close_a[i] < low_a[i - 1]
                    )
                    # HTF veto: the reference applies _check_htf_trend INSIDE
                    # _get_pullback_signal and rejects the signal outright when
                    # the decision is 'block'. Deferring to the shared gate below
                    # is NOT equivalent — that path only *resizes* on partial
                    # misalignment, so a blocked pullback would still be traded.
                    # The reference has no htf_enabled switch: _check_htf_trend
                    # returns 'allow' when htf_ema_aligned is None, which the
                    # wrapper encodes as an all-NaN htf_ema_a.
                    htf_ok = True
                    if not np.isnan(htf_ema_a[i]):
                        htf_dec = _htf_decision(
                            i,
                            signal_dir,
                            htf_ema_a,
                            htf_close_a,
                            htf_slope_a,
                            close_a[i],
                            P_htf_misalign_size_mult,
                        )
                        if htf_dec < 0.0:
                            htf_ok = False
                    if vol_ok and struct_ok and confirm_ok and htf_ok:
                        signal = signal_dir
                        entry_type = 1

        mr_atr = -1.0
        if regime == 3 and P_mr_enabled > 0:
            skip_mr = False
            if P_mr_cooldown_enabled > 0 and mr_loss_streak >= 2 and (i - mr_last_loss_i) < int(P_mr_cooldown_bars):
                skip_mr = True
            if not skip_mr:
                cur_rsi = mr_rsi_a[i] if not np.isnan(mr_rsi_a[i]) else _wilder_rsi(close_a, i, int(P_mr_rsi_period))
                if cur_rsi < P_mr_rsi_oversold:
                    htf_val = htf_ema_a[i]
                    if np.isnan(htf_val) or bar_close > htf_val * (1.0 - P_mr_htf_deviation):
                        signal = 0
                        entry_type = 2
                        mr_atr = cur_atr
                elif cur_rsi > P_mr_rsi_overbought:
                    htf_val = htf_ema_a[i]
                    if np.isnan(htf_val) or bar_close < htf_val * (1.0 + P_mr_htf_deviation):
                        signal = 1
                        entry_type = 2
                        mr_atr = cur_atr

        # Gate 1 — Fused Regime Hysteresis (replaces ER chop + exec-bias).
        # Uses precomputed fused_score_a (ADX 45%, ER 35%, MA-slope 20%).
        # Hysteresis prevents flickering.
        gate_open = False
        if i < fused_score_a.shape[0] and not np.isnan(fused_score_a[i]):
            score = fused_score_a[i]
            if fused_gate_open:
                fused_gate_open = score >= (P_fused_threshold - P_fused_buffer / 2.0)
            else:
                fused_gate_open = score > (P_fused_threshold + P_fused_buffer / 2.0)
            gate_open = fused_gate_open
        if signal != -1 and entry_type == 0 and not gate_open:
            signal = -1

        sig_out[i] = signal
        et_out[i] = entry_type
        sl = 0.0
        tp = 0.0
        if signal != -1:
            if entry_type == 2:
                sl_points = max(int(entry_atr * P_mr_sl_mult / point), int(stops_level))
                tp_points = int(entry_atr * P_mr_tp_mult / point)
            else:
                eatr = mr_atr if mr_atr > 0 else entry_atr
                sl_points = max(int(eatr * P_atr_sl_mult / point), int(stops_level))
                tp_points = int(sl_points * P_rr)
            if signal == 0:
                sl = bar_close - sl_points * point
                tp = bar_close + tp_points * point
            else:
                sl = bar_close + sl_points * point
                tp = bar_close - tp_points * point

        if pos_open and pos_entry_bar != i:
            is_long = pos_is_long
            rem_vol = pos_rem_vol
            hit_sl = (is_long and bar_low <= pos_sl) or (not is_long and bar_high >= pos_sl)
            hit_tp = (pos_tp != 0.0) and ((is_long and bar_high >= pos_tp) or (not is_long and bar_low <= pos_tp))

            if hit_sl or hit_tp:
                exit_price = pos_sl if hit_sl else pos_tp
                if is_long:
                    exit_price -= spm * bar_spread * point * 0.5 + slippage * point
                else:
                    exit_price += spm * bar_spread * point * 0.5 + slippage * point
                pnl = (
                    (exit_price - pos_entry) / point * tick_value * rem_vol
                    if is_long
                    else (pos_entry - exit_price) / point * tick_value * rem_vol
                )
                if commission > 0:
                    pnl -= commission * rem_vol
                    commission_paid += commission * rem_vol
                cumulative_pnl += pnl
                t_type[t_count] = 0 if is_long else 1
                t_entry[t_count] = pos_entry
                t_sl[t_count] = pos_sl
                t_tp[t_count] = pos_tp
                t_entry_bar[t_count] = pos_entry_bar
                t_exit_bar[t_count] = i
                t_exit[t_count] = exit_price
                t_pnl[t_count] = pnl
                t_volume[t_count] = pos_volume
                t_rem_vol[t_count] = rem_vol
                t_exit_reason[t_count] = 0 if hit_sl else 1
                t_regime[t_count] = pos_regime
                t_entry_type[t_count] = pos_entry_type
                t_atr_entry[t_count] = pos_atr_entry
                t_count += 1
                closed_pnl[closed_count] = pnl
                closed_bar[closed_count] = i
                closed_type[closed_count] = 0
                closed_entry_type[closed_count] = pos_entry_type
                closed_count += 1
                if pos_entry_type == 2 and pnl < 0:
                    mr_loss_streak += 1
                    mr_last_loss_i = i
                elif pos_entry_type == 2:
                    mr_loss_streak = 0
                pos_open = False
                equity[i] = initial_balance + cumulative_pnl

            if pos_open and pos_entry_bar != i and P_ch_enabled > 0:
                ch_mult = P_ch_mult
                is_partial = pos_partial_fired
                if P_ch_two_stage > 0 and not is_partial:
                    sl_points2 = abs(pos_entry - pos_sl) / point
                    pnl_points = (bar_close - pos_entry) / point if is_long else (pos_entry - bar_close) / point
                    r_mult = pnl_points / sl_points2 if sl_points2 > 0 else 0.0
                    ch_mult = P_ch_tight_mult if r_mult >= P_ch_two_stage_min_r else P_ch_loose_mult
                if is_partial:
                    ch_mult = P_ch_partial_mult
                if P_ch_accel_enabled > 0 and (not np.isnan(ch_accel_a[i])) and i >= 5:
                    ema_bars = int(P_ch_accel_bars)
                    if i >= ema_bars:
                        ema_ratio = ch_accel_a[i] / ch_accel_a[i - ema_bars] if ch_accel_a[i - ema_bars] != 0 else 1.0
                        strength = P_ch_accel_strength
                        if ema_ratio > 1.0:
                            accel = 1.0 - strength * min(ema_ratio - 1.0, 1.0)
                        else:
                            accel = 1.0 + strength * min(1.0 - ema_ratio, 1.0)
                        accel = max(0.5, min(1.5, accel))
                        ch_mult = ch_mult * accel
                if is_long:
                    hh = _running_max(high_a, pos_entry_bar, i)
                    new_sl = hh - cur_atr * ch_mult
                    if new_sl > pos_ch_sl:
                        pos_ch_sl = new_sl
                    if new_sl > pos_sl:
                        pos_sl = new_sl
                    new_hit = bar_low <= pos_sl
                else:
                    ll = _running_min(low_a, pos_entry_bar, i)
                    new_sl = ll + cur_atr * ch_mult
                    if new_sl < pos_ch_sl or math.isinf(pos_ch_sl):
                        pos_ch_sl = new_sl
                    if new_sl < pos_sl:
                        pos_sl = new_sl
                    new_hit = bar_high >= pos_sl
                if new_hit:
                    exit_price = pos_sl
                    if is_long:
                        exit_price -= spm * bar_spread * point * 0.5 + slippage * point
                    else:
                        exit_price += spm * bar_spread * point * 0.5 + slippage * point
                    pnl = (
                        (exit_price - pos_entry) / point * tick_value * rem_vol
                        if is_long
                        else (pos_entry - exit_price) / point * tick_value * rem_vol
                    )
                    if commission > 0:
                        pnl -= commission * rem_vol
                        commission_paid += commission * rem_vol
                    cumulative_pnl += pnl
                    t_type[t_count] = 0 if is_long else 1
                    t_entry[t_count] = pos_entry
                    t_sl[t_count] = pos_sl
                    t_tp[t_count] = pos_tp
                    t_entry_bar[t_count] = pos_entry_bar
                    t_exit_bar[t_count] = i
                    t_exit[t_count] = exit_price
                    t_pnl[t_count] = pnl
                    t_volume[t_count] = pos_volume
                    t_rem_vol[t_count] = rem_vol
                    t_exit_reason[t_count] = 2
                    t_regime[t_count] = pos_regime
                    t_entry_type[t_count] = pos_entry_type
                    t_atr_entry[t_count] = pos_atr_entry
                    t_count += 1
                    closed_pnl[closed_count] = pnl
                    closed_bar[closed_count] = i
                    closed_type[closed_count] = 0
                    closed_entry_type[closed_count] = pos_entry_type
                    closed_count += 1
                    if pos_entry_type == 2 and pnl < 0:
                        mr_loss_streak += 1
                        mr_last_loss_i = i
                    elif pos_entry_type == 2:
                        mr_loss_streak = 0
                    pos_open = False
                    equity[i] = initial_balance + cumulative_pnl

            if pos_open and pos_entry_bar != i and P_scale_out_enabled > 0:
                close_fracs = (P_scale_out_f0, P_scale_out_f1)
                tp_rr = (P_scale_out_tp0, P_scale_out_tp1)
                num_partials = 2
                atr_sl_mult = P_mr_sl_mult if pos_entry_type == 2 else P_atr_sl_mult
                bt_rr = P_mr_tp_mult if pos_entry_type == 2 else P_rr
                while pos_scale_step < num_partials and rem_vol > 0:
                    step = pos_scale_step
                    target_fraction = tp_rr[step] if step < 2 else tp_rr[1]
                    bt_atr_entry = pos_atr_entry if pos_atr_entry != 0 else cur_atr
                    tp_dist = bt_atr_entry / atr_sl_mult * bt_rr
                    level = pos_entry + tp_dist * target_fraction if is_long else pos_entry - tp_dist * target_fraction
                    hit = bar_high >= level if is_long else bar_low <= level
                    if not hit:
                        break
                    close_frac = close_fracs[step]
                    # The reference TRUNCATES here (int(...)), it does not round:
                    #   close_vol = max(int(volume * frac / step) * step, step)
                    # Rounding instead pushes e.g. 22.695 -> 22.70 vs 22.69 and
                    # desynchronises remaining volume for the rest of the trade.
                    close_vol = max(int(pos_volume * close_frac / vol_step) * vol_step, vol_step)
                    if close_vol > rem_vol:
                        close_vol = rem_vol
                    pnl_part = (
                        (level - pos_entry) / point * tick_value * close_vol
                        if is_long
                        else (pos_entry - level) / point * tick_value * close_vol
                    )
                    cumulative_pnl += pnl_part
                    if commission > 0:
                        commission_paid += commission * close_vol
                        cumulative_pnl -= commission * close_vol
                    rem_vol -= close_vol
                    pos_rem_vol = rem_vol
                    pos_scale_step = step + 1
                    t_type[t_count] = 2
                    t_entry[t_count] = pos_entry
                    t_sl[t_count] = pos_sl
                    t_tp[t_count] = pos_tp
                    t_entry_bar[t_count] = pos_entry_bar
                    t_exit_bar[t_count] = i
                    t_exit[t_count] = level
                    t_pnl[t_count] = pnl_part
                    t_volume[t_count] = close_vol
                    t_rem_vol[t_count] = close_vol
                    t_exit_reason[t_count] = 3
                    t_regime[t_count] = pos_regime
                    t_entry_type[t_count] = pos_entry_type
                    t_atr_entry[t_count] = pos_atr_entry
                    t_count += 1
                    closed_pnl[closed_count] = pnl_part
                    closed_bar[closed_count] = i
                    closed_type[closed_count] = 1
                    closed_entry_type[closed_count] = pos_entry_type
                    closed_count += 1
                    if step == 0:
                        # scale_out_breakeven_fraction (reference default 0.25).
                        lock_level = (
                            pos_entry + tp_dist * P_scale_out_be_frac
                            if is_long
                            else pos_entry - tp_dist * P_scale_out_be_frac
                        )
                        pos_sl = lock_level
                    elif is_long:
                        lf = tp_rr[step - 1] if step - 1 < 2 else tp_rr[1]
                        lock_level = pos_entry + tp_dist * lf
                        if lock_level > pos_sl:
                            pos_sl = lock_level
                    else:
                        lf = tp_rr[step - 1] if step - 1 < 2 else tp_rr[1]
                        lock_level = pos_entry - tp_dist * lf
                        if lock_level < pos_sl:
                            pos_sl = lock_level
                if pos_scale_step >= num_partials:
                    pos_partial_fired = True
                    pos_tp = 0.0

            mr_exit = False
            if pos_open and pos_entry_bar != i and regime == 3 and P_mr_enabled > 0:
                # Read the SAME MR RSI array used for MR entry (built from
                # mr_rsi_h1 with a _calc_rsi_wilder fallback) so entry/exit are
                # consistent and bit-exact with the reference loop (agent audit
                # H1). Previously exit used _wilder_rsi, a different series.
                _rp = int(P_mr_rsi_period)
                prev_rsi = mr_rsi_a[i - 1]
                if np.isnan(prev_rsi):
                    prev_rsi = _wilder_rsi(close_a, i - 1, _rp)
                cur_rsi2 = mr_rsi_a[i]
                if np.isnan(cur_rsi2):
                    cur_rsi2 = _wilder_rsi(close_a, i, _rp)
                if is_long and prev_rsi < 50 and cur_rsi2 >= 50 or (not is_long) and prev_rsi > 50 and cur_rsi2 <= 50:
                    mr_exit = True

            in_sub_profit = (is_long and bar_close > pos_entry + cur_atr * 0.25) or (
                not is_long and bar_close < pos_entry - cur_atr * 0.25
            )
            opp_signal = (
                pos_open
                and pos_entry_bar != i
                and (mr_exit or ((is_long and sell_signal) or (not is_long and buy_signal)))
                and signal != -1
                and rem_vol > 0
                and (mr_exit or not in_sub_profit)
            )
            if opp_signal and rem_vol > 0:
                if is_long:
                    exit_price = bar_close - spm * bar_spread * point * 0.5 - slippage * point
                else:
                    exit_price = bar_close + spm * bar_spread * point * 0.5 + slippage * point
                pnl = (
                    (exit_price - pos_entry) / point * tick_value * rem_vol
                    if is_long
                    else (pos_entry - exit_price) / point * tick_value * rem_vol
                )
                if commission > 0:
                    pnl -= commission * rem_vol
                    commission_paid += commission * rem_vol
                cumulative_pnl += pnl
                t_type[t_count] = 0 if is_long else 1
                t_entry[t_count] = pos_entry
                t_sl[t_count] = pos_sl
                t_tp[t_count] = pos_tp
                t_entry_bar[t_count] = pos_entry_bar
                t_exit_bar[t_count] = i
                t_exit[t_count] = exit_price
                t_pnl[t_count] = pnl
                t_volume[t_count] = pos_volume
                t_rem_vol[t_count] = rem_vol
                t_exit_reason[t_count] = 4 if mr_exit else 5
                t_regime[t_count] = pos_regime
                t_entry_type[t_count] = pos_entry_type
                t_atr_entry[t_count] = pos_atr_entry
                t_count += 1
                closed_pnl[closed_count] = pnl
                closed_bar[closed_count] = i
                closed_type[closed_count] = 0
                closed_entry_type[closed_count] = pos_entry_type
                closed_count += 1
                if pos_entry_type == 2 and pnl < 0:
                    mr_loss_streak += 1
                    mr_last_loss_i = i
                elif pos_entry_type == 2:
                    mr_loss_streak = 0
                pos_open = False
                equity[i] = initial_balance + cumulative_pnl

        if regime == 4:
            equity[i] = initial_balance + cumulative_pnl
            continue

        int(P_max_positions_per_symbol)

        if (not pos_open) and signal != -1 and i < n - 1:
            dbg[2] += 1
            vol_ok = True
            if P_vol_filter > 0 and (not np.isnan(vol_sma_a[i])) and vol_sma_a[i] > 0:
                cur_vol = vol_a[i] if i < len(vol_a) else 0.0
                vol_ok = _vol_filter_pass(
                    close_a,
                    vol_a,
                    i,
                    cur_vol,
                    vol_sma_a[i],
                    P_volume_kappa,
                    signal == 0,
                    P_vf_obv_lookback,
                    P_vf_obv_enabled,
                )
            if not vol_ok:
                dbg[0] += 1
                equity[i] = initial_balance + cumulative_pnl
                continue

            spread_ok = True
            if P_spread_filter > 0 and bar_spread > 0 and (not np.isnan(cur_atr)) and cur_atr > 0:
                ratio = (bar_spread * point) / cur_atr
                if ratio > P_spf_max_ratio:
                    spread_ok = False
            if not spread_ok:
                dbg[5] += 1
                equity[i] = initial_balance + cumulative_pnl
                continue

            # HTF trend gate. The reference skips this entirely when MTF is on
            # (backtest.py:1762 `if not p.get("mtf_enabled", False)`), because the
            # MTF signal already encodes its own H4 bias. Applying it regardless
            # over-blocks every MTF symbol.
            htf_mult = 1.0
            if P_mtf_enabled <= 0:
                htf_mult = _htf_decision(
                    i, signal, htf_ema_a, htf_close_a, htf_slope_a, bar_close, P_htf_misalign_size_mult
                )
                if htf_mult < 0.0:
                    dbg[6] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue

            session_ok = True
            if P_session_enabled > 0:
                cur_hour = int(bar_hour_a[i])
                if P_session_only > 0:
                    session_ok = int(P_london_open) <= cur_hour <= int(P_london_close)
                elif P_require_overlap > 0:
                    session_ok = 12 <= cur_hour <= 17
                elif P_skip_asian > 0:
                    session_ok = not (int(P_asian_open) <= cur_hour <= int(P_asian_close))
            if not session_ok:
                equity[i] = initial_balance + cumulative_pnl
                continue

            if P_daily_loss_pct > 0:
                ds = day_start_idx_a[i]
                day_pnl = 0.0
                for k in range(closed_count):
                    if closed_bar[k] >= ds and closed_bar[k] <= i:
                        day_pnl += closed_pnl[k]
                equity_now = initial_balance + cumulative_pnl
                if equity_now <= 0:
                    equity[i] = initial_balance + cumulative_pnl
                    continue
                loss_pct = (-day_pnl / equity_now) * 100 if day_pnl < 0 else 0.0
                if loss_pct >= P_daily_loss_pct:
                    dbg[7] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue

            if P_tr_enabled > 0 and i >= tr_needed:
                mean_r = 0.0
                cnt_r = 0
                for k in range(i - tr_needed, i + 1):
                    if close_a[k - 1] != 0:
                        r = (close_a[k] - close_a[k - 1]) / close_a[k - 1]
                        mean_r += r
                        cnt_r += 1
                if cnt_r > 0:
                    mean_r = mean_r / cnt_r
                    std_r = 0.0
                    for k in range(i - tr_needed, i + 1):
                        if close_a[k - 1] != 0:
                            r = (close_a[k] - close_a[k - 1]) / close_a[k - 1]
                            std_r += (r - mean_r) * (r - mean_r)
                    std_r = (std_r / cnt_r) ** 0.5
                    last_r = (close_a[i] - close_a[i - 1]) / close_a[i - 1] if close_a[i - 1] != 0 else 0.0
                    if std_r > 0 and abs(last_r - mean_r) / std_r > P_tr_sigma:
                        dbg[8] += 1
                        equity[i] = initial_balance + cumulative_pnl
                        continue
                # CURRENT-drawdown check (mirror reference _check_tail_risk):
                # peak = running max equity up to the previous bar; block on the
                # current drawdown from that peak, which recovers as equity does.
                # Previously this accumulated a monotonic max_dd_so_far that
                # permanently halted after any 8% dip, diverging from live
                # check_tail_risk (agent audit C6).
                eq_prev = equity[i - 1] if i > warmup else initial_balance
                if eq_prev > run_max_eq:
                    run_max_eq = eq_prev
                dd_pct = ((run_max_eq - eq_prev) / run_max_eq) * 100 if run_max_eq > 0 else 0.0
                if dd_pct >= P_tr_max_dd_pct or dd_pct >= P_cb_dd_pct:
                    dbg[9] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue

            ml_mult = 1.0
            if P_ml_enabled > 0:
                mv = ml_buy_a[i] if signal == 0 else ml_sell_a[i]
                if not np.isnan(mv):
                    ml_mult = mv
                if ml_mult <= 0:
                    dbg[10] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue

            confidence_mult = 1.0
            if P_scoring_enabled > 0:
                mr_min = 0.03 if entry_type == 2 else 0.0
                min_score = P_scoring_min_entry + mr_min
                if score_a[i] < min_score:
                    dbg[11] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue
                confidence_mult = conf_mult_a[i]

            current_equity = initial_balance + cumulative_pnl
            sl_price_dist = abs(bar_close - sl)
            sl_value = sl_price_dist * tick_value / point
            risk_amount = max(current_equity * (risk_percent / 100), 0.0)
            raw_volume = risk_amount / sl_value if sl_value > 0 else 0.0
            if raw_volume < vol_step:
                min_lot_risk = vol_step * sl_value
                risk_ratio = min_lot_risk / risk_amount if risk_amount > 0 else 0.0
                if risk_ratio > max_risk_ratio:
                    dbg[12] += 1
                    equity[i] = initial_balance + cumulative_pnl
                    continue
            volume = raw_volume * htf_mult
            if P_mtf_enabled > 0:
                # Parity with reference _run_reference: under MTF the base size
                # multiplier is max(0.5, mtf_confidence) (agreement ratio), not
                # htf_mult. mtf_conf is 0.0 for non-MTF entries this bar.
                volume = raw_volume * max(0.5, mtf_conf)
            if entry_type == 2:
                regime_mult = P_mr_position_size_mult
            else:
                if regime == 1:
                    regime_mult = 1.0
                elif regime == 0:
                    regime_mult = 0.5
                else:
                    regime_mult = 0.75
            kelly_mult = 1.0
            if P_dr_enabled > 0:
                look = int(P_dr_lookback)
                start = closed_count - look if closed_count > look else 0
                pnls_l = np.empty(look)
                pc = 0
                for k in range(start, closed_count):
                    if closed_type[k] == 0:
                        pnls_l[pc] = closed_pnl[k]
                        pc += 1
                if pc >= 10:
                    wins = 0
                    sum_w = 0.0
                    sum_l = 0.0
                    for k in range(pc):
                        if pnls_l[k] > 0:
                            wins += 1
                            sum_w += pnls_l[k]
                        else:
                            sum_l += -pnls_l[k]
                    win_rate = wins / pc
                    avg_win = sum_w / wins if wins > 0 else 0.0
                    avg_loss = sum_l / (pc - wins) if (pc - wins) > 0 else 0.0
                    b = avg_win / avg_loss if avg_loss > 0 else 1.0
                    q = 1.0 - win_rate
                    kelly = (win_rate * b - q) / b if b > 0 else 0.0
                    kelly = max(0.0, kelly) * P_dr_kelly_fraction
                    kelly = max(P_dr_min_mult, min(P_dr_max_mult, kelly))
                    kelly_mult = kelly
            vol_mult = 1.0
            if (
                P_dr_vol_adjust > 0
                and (not np.isnan(atr_sma50_a[i]))
                and atr_sma50_a[i] > 0
                and (not np.isnan(cur_atr))
            ):
                ratio = cur_atr / atr_sma50_a[i]
                if ratio > 1.2:
                    vol_mult = max(0.25, 1.0 / ratio)
            volume *= regime_mult * kelly_mult * vol_mult * ml_mult * confidence_mult
            if P_correlation_enabled > 0:
                volume *= P_corr_size_mult
            volume = max(round(volume / vol_step, 0) * vol_step, vol_step)

            pending_entry = True
            dbg[4] += 1
            pend_sl_dbg[i] = sl
            pe_type = signal
            pe_entry_type = entry_type
            pe_sl = sl
            pe_tp = tp
            pe_volume = volume
            pe_regime = regime
            pe_atr_entry = entry_atr if mr_atr <= 0 else mr_atr
            pe_sl_points = sl_points
            pe_tp_points = tp_points
            equity[i] = initial_balance + cumulative_pnl
            continue

        open_pnl = 0.0
        if pos_open:
            rem_vol2 = pos_rem_vol
            if rem_vol2 > 0:
                if pos_is_long:
                    open_pnl += (bar_close - pos_entry) / point * tick_value * rem_vol2
                else:
                    open_pnl += (pos_entry - bar_close) / point * tick_value * rem_vol2
        equity[i] = initial_balance + cumulative_pnl + open_pnl

    if pos_open:
        last_spread = spread_a[n - 1]
        if pos_is_long:
            exit_price = close_a[n - 1] - spm * last_spread * point * 0.5 - slippage * point
        else:
            exit_price = close_a[n - 1] + spm * last_spread * point * 0.5 + slippage * point
        pnl = (
            (exit_price - pos_entry) / point * tick_value * pos_rem_vol
            if pos_is_long
            else (pos_entry - exit_price) / point * tick_value * pos_rem_vol
        )
        if commission > 0:
            pnl -= commission * pos_rem_vol
        cumulative_pnl += pnl
        t_type[t_count] = 0 if pos_is_long else 1
        t_entry[t_count] = pos_entry
        t_sl[t_count] = pos_sl
        t_tp[t_count] = pos_tp
        t_entry_bar[t_count] = pos_entry_bar
        t_exit_bar[t_count] = n - 1
        t_exit[t_count] = exit_price
        t_pnl[t_count] = pnl
        t_volume[t_count] = pos_volume
        t_rem_vol[t_count] = pos_rem_vol
        t_exit_reason[t_count] = 6
        t_regime[t_count] = pos_regime
        t_entry_type[t_count] = pos_entry_type
        t_atr_entry[t_count] = pos_atr_entry
        t_count += 1

    return (
        t_count,
        t_type[:t_count],
        t_entry[:t_count],
        t_sl[:t_count],
        t_tp[:t_count],
        t_entry_bar[:t_count],
        t_exit_bar[:t_count],
        t_exit[:t_count],
        t_pnl[:t_count],
        t_volume[:t_count],
        t_rem_vol[:t_count],
        t_exit_reason[:t_count],
        t_regime[:t_count],
        t_entry_type[:t_count],
        t_atr_entry[:t_count],
        equity,
        dbg,
        sig_out,
        pend_sl_dbg,
        skip_low_dbg,
        et_out,
    )


@njit(cache=True, fastmath=False)
def _wilder_rsi(close_a, i, period):
    if i < period:
        return 50.0
    # Initial SMA over first period bars
    gains = 0.0
    losses = 0.0
    for k in range(i - period + 1, i + 1):
        d = close_a[k] - close_a[k - 1]
        if d > 0:
            gains += d
        elif d < 0:
            losses += -d
    avg_gain = gains / period
    avg_loss = losses / period
    # Wilder smoothing for remaining bars (period + 1 .. i)
    for k in range(i - period + 2, i + 1):
        d = close_a[k] - close_a[k - 1]
        g = d if d > 0 else 0.0
        loss_val = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + loss_val) / period
    if avg_loss <= 0:
        return 50.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


@njit(cache=True, fastmath=False)
def _running_max(arr, start, end):
    m = arr[start]
    for k in range(start + 1, end + 1):
        if arr[k] > m:
            m = arr[k]
    return m


@njit(cache=True, fastmath=False)
def _running_min(arr, start, end):
    m = arr[start]
    for k in range(start + 1, end + 1):
        if arr[k] < m:
            m = arr[k]
    return m


@njit(cache=True, fastmath=False)
def _vol_filter_pass(close_a, vol_a, i, cur_vol, cur_sma, kappa, is_buy, obv_lookback, obv_enabled):
    """Parity with volume_filter_pass in analytics.py (OBV-based gate).

    Must mirror the reference logic exactly:
      1. rel_vol >= kappa -> pass (volume confirms)
      2. obv_enabled and OBV divergence in signal direction -> pass
      3. otherwise -> fail
    """
    rel_vol = cur_vol / cur_sma
    if rel_vol >= kappa:
        return True
    if not obv_enabled:
        return False
    lookback = int(obv_lookback)
    s = i - lookback if i - lookback > 0 else 0
    window_len = i - s + 1
    if window_len < 2:
        return True
    obv = np.empty(window_len)
    obv[0] = 0.0
    for j in range(1, window_len):
        if close_a[s + j] > close_a[s + j - 1]:
            obv[j] = obv[j - 1] + vol_a[s + j]
        elif close_a[s + j] < close_a[s + j - 1]:
            obv[j] = obv[j - 1] - vol_a[s + j]
        else:
            obv[j] = obv[j - 1]
    if is_buy:
        low_idx = 0
        low_val = close_a[s]
        for j in range(1, window_len):
            if close_a[s + j] < low_val:
                low_val = close_a[s + j]
                low_idx = j
        if low_idx > 0 and obv[window_len - 1] > obv[low_idx]:
            return True
    else:
        high_idx = 0
        high_val = close_a[s]
        for j in range(1, window_len):
            if close_a[s + j] > high_val:
                high_val = close_a[s + j]
                high_idx = j
        if high_idx > 0 and obv[window_len - 1] < obv[high_idx]:
            return True
    return False
