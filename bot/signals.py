"""Entry/exit signals — crossovers, pullbacks, MR, fused regime gate, scoring."""

import logging

import MetaTrader5 as mt5
import pandas as pd
import state as _st
from analytics import closed_bars
from indicators import calc_atr, calc_efficiency_ratio, calc_fused_regime_score, calc_ma, calc_ma_slope, calc_rsi
from mt5_connect import get_rates
from regime import get_current_adx


class RegimeGate:
    """Hysteresis gate for fused regime composite score.

    Opens when score > threshold + buffer/2 (enters trend-favorable mode).
    Closes when score < threshold - buffer/2 (enters chop mode).
    Prevents flickering on marginal regime changes.

    Per-symbol instances live in state._regime_gate_state.
    """

    def __init__(self, threshold=50.0, buffer=5.0):
        self.threshold = threshold
        self.buffer = buffer
        self._open = False

    @property
    def is_open(self):
        return self._open

    def update(self, score):
        if self._open:
            self._open = score >= (self.threshold - self.buffer / 2.0)
        else:
            self._open = score > (self.threshold + self.buffer / 2.0)
        return self._open

    def reset(self):
        self._open = False


def _get_regime_gate(symbol, cfg):
    """Lazy-init and return the RegimeGate for *symbol*."""
    gate = _st._regime_gate_state.get(symbol)
    if gate is None:
        thr = cfg.get("fused_threshold", 50.0)
        buf = cfg.get("fused_buffer", 5.0)
        gate = RegimeGate(threshold=thr, buffer=buf)
        _st._regime_gate_state[symbol] = gate
    return gate


def _pb_volume_pass(df, trigger_idx, cfg):
    if not cfg.get("pb_volume_enabled", True):
        return True
    period = cfg.get("pb_volume_sma_period", 20)
    threshold = cfg.get("pb_volume_threshold", 0.8)
    if len(df) < period + 2:
        return True
    vol = df["tick_volume"].values
    trigger_vol = vol[trigger_idx]
    vol_sma = pd.Series(vol).rolling(window=period).mean().iloc[trigger_idx]
    if pd.isna(vol_sma) or vol_sma <= 0:
        return True
    return trigger_vol < vol_sma * threshold


def _pb_structure_pass(df, trigger_idx, direction, cfg):
    lookback = cfg.get("pb_structure_lookback", 5)
    if len(df) < lookback + 2:
        return True
    if direction == "buy":
        prior_min = df["low"].iloc[trigger_idx - lookback : trigger_idx].min()
        return df["low"].iloc[trigger_idx] > prior_min
    else:
        prior_max = df["high"].iloc[trigger_idx - lookback : trigger_idx].max()
        return df["high"].iloc[trigger_idx] < prior_max


def get_trend_pullback_signal(df, cfg):
    ma_type = cfg.get("ma_type", "kama")
    ema_fast = calc_ma(df, cfg["ema_fast"], ma_type)
    ema_slow = calc_ma(df, cfg["ema_slow"], ma_type)
    atr = calc_atr(df, cfg["atr_period"])
    if atr is None or atr <= 0:
        return None, atr, None
    if len(df) < 4:
        return None, atr, None

    trigger_idx = -3
    confirm_idx = -2

    trigger_fast = ema_fast.iloc[trigger_idx]
    trigger_slow = ema_slow.iloc[trigger_idx]
    trigger_price = df["close"].iloc[trigger_idx]
    trigger_low = df["low"].iloc[trigger_idx]
    trigger_high = df["high"].iloc[trigger_idx]
    confirm_close = df["close"].iloc[confirm_idx]

    if pd.isna(trigger_fast) or pd.isna(trigger_slow):
        return None, atr, None

    pullback_dist = atr * cfg["pb_atr_mult"]
    min_pb_dist = atr * cfg.get("pb_atr_min_dist", 0.1)

    signal = None

    if trigger_fast > trigger_slow:
        dist_to_ema = abs(trigger_price - trigger_fast)
        if min_pb_dist <= dist_to_ema <= pullback_dist:
            if not _pb_volume_pass(df, trigger_idx, cfg):
                return None, atr, None
            if not _pb_structure_pass(df, trigger_idx, "buy", cfg):
                return None, atr, None
            if confirm_close <= trigger_high:
                return None, atr, None
            # HTF trend gate is applied by the caller (main.py / backtest)
            # via check_htf_trend() — checking it here would double-gate the
            # signal and lose the "soft" size reduction (agent audit M5).
            signal = "buy"

    elif trigger_fast < trigger_slow:
        dist_to_ema = abs(trigger_price - trigger_fast)
        if min_pb_dist <= dist_to_ema <= pullback_dist:
            if not _pb_volume_pass(df, trigger_idx, cfg):
                return None, atr, None
            if not _pb_structure_pass(df, trigger_idx, "sell", cfg):
                return None, atr, None
            if confirm_close >= trigger_low:
                return None, atr, None
            # HTF trend gate is applied by the caller (main.py / backtest)
            # via check_htf_trend() — checking it here would double-gate the
            # signal and lose the "soft" size reduction (agent audit M5).
            signal = "sell"

    if signal:
        return signal, atr, "pullback"
    return None, atr, None


def check_htf_trend(cfg, signal):
    """HTF trend alignment check — 3-state decision.

    Returns (decision, size_mult) where decision is one of:
      - "allow": price AND slope agree with signal -> full size (1.0).
      - "soft":  exactly ONE of price/slope disagrees (neutral/flat HTF) ->
                 allowed at reduced size (cfg['htf_misalign_size_mult']).
      - "block": BOTH price and slope disagree (clear counter-trend) ->
                 hard-blocked, size_mult 0.0.

    This keeps the bot from trading outright reversals (block) while still
    allowing entries when the higher timeframe is merely neutral (soft).
    """
    symbol = cfg["symbol"]
    htf_tf = getattr(mt5, f"TIMEFRAME_{cfg['htf_timeframe']}", mt5.TIMEFRAME_H4)
    htf_slow = cfg["htf_ema_slow"]
    needed = htf_slow + 50
    df = get_rates(symbol, htf_tf, needed)
    if df is None or len(df) < needed:
        logging.info(f"[{symbol}] Insufficient HTF data for trend check — allowing at reduced size")
        return "soft", cfg.get("htf_misalign_size_mult", 0.5)
    ma_type = cfg.get("ma_type", "kama")
    htf_ma = calc_ma(df, htf_slow, ma_type)
    if htf_ma is None or len(htf_ma) < 2 or pd.isna(htf_ma.iloc[-2]):
        logging.info(f"[{symbol}] HTF MA calc failed — allowing at reduced size")
        return "soft", cfg.get("htf_misalign_size_mult", 0.5)
    htf_price = df["close"].iloc[-2]
    htf_ma_val = htf_ma.iloc[-2]
    slope_window = min(12, max(2, len(htf_ma) // 10))
    slope = htf_ma.iloc[-2] - htf_ma.iloc[-(slope_window + 1)] if len(htf_ma) > slope_window + 1 else 0.0
    if pd.isna(slope):
        slope = 0.0
    if signal == "buy":
        price_ok = htf_price >= htf_ma_val
        slope_ok = slope >= 0
    else:  # sell
        price_ok = htf_price <= htf_ma_val
        slope_ok = slope <= 0
    if price_ok and slope_ok:
        return "allow", 1.0
    if (not price_ok) and (not slope_ok):
        logging.info(
            f"[{symbol}] HTF clear counter-trend — blocking {signal} "
            f"(price={htf_price:.2f}, ma={htf_ma_val:.2f}, slope={slope:.4f})"
        )
        return "block", 0.0
    logging.info(
        f"[{symbol}] HTF neutral (one condition off) — {signal} at reduced size "
        f"(price={htf_price:.2f}, ma={htf_ma_val:.2f}, slope={slope:.4f})"
    )
    return "soft", cfg.get("htf_misalign_size_mult", 0.5)


def get_signal(cfg):
    sym = cfg["symbol"]
    needed = cfg["ema_slow"] + cfg["atr_period"] + 50
    df = get_rates(sym, cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        logging.info(f"[{sym}] Insufficient data for signal ({len(df) if df is not None else 0} < {needed})")
        return None, None, None
    ma_type = cfg.get("ma_type", "kama")
    ma_fast = calc_ma(closed_bars(df), cfg["ema_fast"], ma_type)
    ma_slow = calc_ma(closed_bars(df), cfg["ema_slow"], ma_type)
    atr = calc_atr(df, cfg["atr_period"])
    # Signal on the last two CLOSED bars (iloc[-2] current, iloc[-3] previous)
    # so a crossover is only acted on after the bar closes — prevents entries
    # on intrabar (still-forming) repaints (agent audit B1).
    if len(ma_fast) < 3 or len(ma_slow) < 3:
        logging.info(f"[{sym}] Insufficient MA data for crossover detection")
        return None, atr, None
    current_fast = ma_fast.iloc[-2]
    current_slow = ma_slow.iloc[-2]
    prev_fast = ma_fast.iloc[-3]
    prev_slow = ma_slow.iloc[-3]
    if pd.isna(current_fast) or pd.isna(current_slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
        logging.info(f"[{sym}] NaN in MA values — skipping crossover")
        return None, atr, None
    ma_label = ma_type.upper()
    if cfg.get("verbose_debug", False):
        logging.info(
            f"{ma_label}{cfg['ema_fast']}={current_fast:.2f} {ma_label}{cfg['ema_slow']}={current_slow:.2f} "
            f"(prev: {prev_fast:.2f}/{prev_slow:.2f}) atr={atr:.4f}"
        )
    signal = None
    if prev_fast <= prev_slow and current_fast > current_slow:
        signal = "buy"
    elif prev_fast >= prev_slow and current_fast < current_slow:
        signal = "sell"
    if signal:
        # Compute the fused regime score for logging only — Gate 1 in main.py
        # owns the gate state and already calls gate.update() every cycle.
        # Calling gate.update() here would mutate the shared gate a second time
        # (agent audit: get_signal double-gate-update), causing the gate to
        # flicker differently than the MTF path which correctly avoids this.
        gate = _get_regime_gate(cfg["symbol"], cfg)
        adx_val = get_current_adx(cfg) or 0.0
        er = calc_efficiency_ratio(closed_bars(df)["close"].values, cfg.get("er_period", 10))
        ma_slope_val = calc_ma_slope(ma_fast, period=1) if len(ma_fast) > 2 else 0.0
        atr_val = atr if atr and atr > 0 else 0.0
        score = calc_fused_regime_score(adx_val, er, ma_slope_val, atr_val)
        gate_open = gate.is_open
        logging.info(f"[{sym}] Crossover {signal} signal (fused_score={score:.1f}, gate_open={gate_open})")
        return signal, atr, "crossover"
    if cfg.get("pb_enabled", True):
        pb_signal, pb_atr, pb_type = get_trend_pullback_signal(df, cfg)
        if pb_signal:
            return pb_signal, pb_atr, pb_type
    logging.debug(f"[{sym}] No crossover or pullback (fast={current_fast:.2f} slow={current_slow:.2f})")
    return None, atr, None


def get_mtf_fused_signal(cfg):
    """Multi-timeframe fused signal — hierarchical bias+trigger (AGENTS.md).

    H4: fixed-period EMA as trend bias (price above/below EMA = bullish/bearish).
    H1: MA crossover must agree with H4 bias direction.
    M15: MA crossover provides entry timing.

    Falls back to H1 pullback when M15 produces no crossover but H4 bias + H1 cross agree.
    Returns (signal, atr, entry_type, agreement_ratio) or (None, None, None, None).
    """
    symbol = cfg["symbol"]
    ma_type = cfg.get("ma_type", "kama")
    fast = cfg["ema_fast"]
    slow = cfg["ema_slow"]
    h4_ema_period = cfg.get("mtf_h4_ema_period", 100)
    m15_fast = cfg.get("mtf_m15_ema_fast", max(5, fast // 2))
    m15_slow = cfg.get("mtf_m15_ema_slow", max(8, slow // 2))

    # --- Fetch H1 data (ATR + MA crossover) ---
    h1_needed = slow + cfg["atr_period"] + 50
    h1_df = get_rates(symbol, mt5.TIMEFRAME_H1, h1_needed)
    if h1_df is None or len(h1_df) < h1_needed:
        return None, None, None, None
    atr = calc_atr(h1_df, cfg["atr_period"])

    # --- Fetch H4 data for trend bias ---
    h4_needed = h4_ema_period + 10
    h4_df = get_rates(symbol, mt5.TIMEFRAME_H4, h4_needed)
    if h4_df is None or len(h4_df) < h4_ema_period:
        logging.info(f"[{symbol}] Insufficient H4 data for MTF bias — fallback to single-TF")
        return None, None, None, None

    h4_close = h4_df["close"].values
    h4_ema = calc_ma(h4_df, h4_ema_period, ma_type)
    if h4_ema is None or len(h4_ema) < 3 or pd.isna(h4_ema.iloc[-2]):
        return None, atr, None, None

    # H4 bias: last closed H4 bar vs its EMA(100)
    h4_bias_val = h4_close[-2] - float(h4_ema.iloc[-2])
    h4_atr = calc_atr(h4_df, cfg["atr_period"])
    neutral_band = (h4_atr * 0.5) if h4_atr and h4_atr > 0 else 0.0
    if abs(h4_bias_val) <= neutral_band:
        logging.info(f"[{symbol}] MTF H4 neutral (within {neutral_band:.2f} of EMA) — no signal")
        return None, atr, None, None
    h4_direction = 1 if h4_bias_val > 0 else -1  # bullish/bearish

    # --- H1 MA crossover ---
    h1_ma_fast = calc_ma(closed_bars(h1_df), fast, ma_type)
    h1_ma_slow = calc_ma(closed_bars(h1_df), slow, ma_type)
    if h1_ma_fast is None or h1_ma_slow is None or len(h1_ma_fast) < 3:
        return None, atr, None, None

    h1_cur_fast = h1_ma_fast.iloc[-2]
    h1_cur_slow = h1_ma_slow.iloc[-2]
    h1_prev_fast = h1_ma_fast.iloc[-3]
    h1_prev_slow = h1_ma_slow.iloc[-3]
    if any(pd.isna(x) for x in (h1_cur_fast, h1_cur_slow, h1_prev_fast, h1_prev_slow)):
        return None, atr, None, None

    h1_cross = 0
    if h1_prev_fast <= h1_prev_slow and h1_cur_fast > h1_cur_slow:
        h1_cross = 1
    elif h1_prev_fast >= h1_prev_slow and h1_cur_fast < h1_cur_slow:
        h1_cross = -1

    if h1_cross != h4_direction:
        logging.info(f"[{symbol}] MTF H4 bias={h4_direction:+d} blocks H1 cross={h1_cross:+d}")
        return None, atr, None, None

    direction = "buy" if h1_cross > 0 else "sell"
    agreement = 0.67  # H4 bias + H1 cross agree
    entry_type = "pullback"

    # --- M15 entry trigger ---
    m15_needed = m15_slow + cfg["atr_period"] + 50
    m15_df = get_rates(symbol, mt5.TIMEFRAME_M15, m15_needed)
    if m15_df is not None and len(m15_df) >= m15_needed:
        m15_ma_fast = calc_ma(m15_df, m15_fast, ma_type)
        m15_ma_slow = calc_ma(m15_df, m15_slow, ma_type)
        if m15_ma_fast is not None and m15_ma_slow is not None and len(m15_ma_fast) >= 3:
            m15_cf = m15_ma_fast.iloc[-2]
            m15_cs = m15_ma_slow.iloc[-2]
            m15_pf = m15_ma_fast.iloc[-3]
            m15_ps = m15_ma_slow.iloc[-3]
            if not any(pd.isna(x) for x in (m15_cf, m15_cs, m15_pf, m15_ps)):
                m15_cross = 0
                if m15_pf <= m15_ps and m15_cf > m15_cs:
                    m15_cross = 1
                elif m15_pf >= m15_ps and m15_cf < m15_cs:
                    m15_cross = -1
                if m15_cross == h1_cross:
                    agreement = 1.0
                    entry_type = "crossover"
                elif m15_cross != 0:
                    logging.info(f"[{symbol}] MTF M15 cross opposes H1 — no entry")
                    return None, atr, None, None

    if entry_type == "pullback":
        if cfg.get("pb_enabled", True):
            pb_signal, pb_atr, pb_type = get_trend_pullback_signal(h1_df, cfg)
            if pb_signal:
                return pb_signal, pb_atr, pb_type, agreement
        return None, atr, None, None

    return direction, atr, entry_type, agreement


def get_mean_reversion_signal(cfg):
    sym = cfg["symbol"]
    mr_tf_name = cfg.get("mr_timeframe", "M30")
    mr_tf = getattr(mt5, f"TIMEFRAME_{mr_tf_name}", mt5.TIMEFRAME_M30)
    rsi_period = cfg["mr_rsi_period"]
    needed = rsi_period + cfg["atr_period"] + 30
    df = get_rates(sym, mr_tf, needed)
    if df is None or len(df) < needed:
        logging.info(f"[{sym}] MR insufficient data ({len(df) if df is not None else 0} < {needed})")
        return None, None
    # RSI / price taken from the last CLOSED bar (closed_bars() excludes the
    # forming bar) to avoid acting on intrabar RSI repaints (agent audit B1).
    rsi = calc_rsi(closed_bars(df), rsi_period)
    atr = calc_atr(df, cfg["atr_period"])
    cur_rsi = rsi
    cur_price = df["close"].iloc[-2]
    htf_tf = getattr(mt5, f"TIMEFRAME_{cfg['htf_timeframe']}", mt5.TIMEFRAME_H4)
    htf_needed = cfg["htf_ema_slow"] + 20
    htf_df = get_rates(sym, htf_tf, htf_needed)
    if htf_df is None or len(htf_df) < htf_needed:
        logging.info(f"[{sym}] MR HTF insufficient data ({len(htf_df) if htf_df is not None else 0} < {htf_needed})")
        return None, None
    htf_ma_type = cfg.get("ma_type", "kama")
    htf_ema200 = calc_ma(htf_df, cfg["htf_ema_slow"], htf_ma_type)
    if htf_ema200 is None or len(htf_ema200) < 2 or pd.isna(htf_ema200.iloc[-2]):
        logging.info(f"[{sym}] MR HTF MA calc failed")
        return None, None
    htf_ema200_val = htf_ema200.iloc[-2]
    oversold = cfg["mr_rsi_oversold"]
    overbought = cfg["mr_rsi_overbought"]
    mr_htf_deviation = cfg.get("mr_htf_deviation", 0.0)
    signal = None
    if cur_rsi < oversold:
        if cur_price > htf_ema200_val * (1.0 - mr_htf_deviation):
            signal = "buy"
    elif cur_rsi > overbought and cur_price < htf_ema200_val * (1.0 + mr_htf_deviation):
        signal = "sell"
    if signal:
        logging.info(
            f"[{sym}] MR {signal} signal (RSI={cur_rsi:.1f}, price={cur_price:.2f}, HTF_MA={htf_ema200_val:.2f})"
        )
    else:
        logging.debug(
            f"[{sym}] MR no signal (RSI={cur_rsi:.1f}, price={cur_price:.2f}, "
            f"HTF_MA={htf_ema200_val:.2f}, ob={overbought}, os={oversold})"
        )
    return signal, atr




def check_mean_reversion_exit(cfg, position):
    mr_tf_name = cfg.get("mr_timeframe", "M30")
    mr_tf = getattr(mt5, f"TIMEFRAME_{mr_tf_name}", mt5.TIMEFRAME_M30)
    rsi_period = cfg["mr_rsi_period"]
    needed = rsi_period + 10
    df = get_rates(cfg["symbol"], mr_tf, needed)
    if df is None or len(df) < needed:
        return False
    # Use calc_rsi (canonical Wilder) for BOTH prev and cur so the crossover
    # check is a genuine prev→cur comparison with the same methodology. The
    # previous rsi_prev() used a truncated ~15-bar window which diverged from
    # calc_rsi's full-series Wilder smoothing, causing false crossovers and
    # backtest-vs-live parity gaps (agent audit H1).
    df_closed = closed_bars(df)
    if df_closed is None or len(df_closed) < rsi_period + 2:
        return False
    cur_rsi = calc_rsi(df_closed, rsi_period)
    # prev = RSI of the bar BEFORE the last closed bar
    prev_df = df_closed.iloc[:-1]
    if len(prev_df) < rsi_period + 1:
        return False
    prev_rsi_val = calc_rsi(prev_df, rsi_period)
    is_long = position.type == mt5.ORDER_TYPE_BUY
    if is_long and prev_rsi_val < 50 and cur_rsi >= 50:
        return True
    return bool(not is_long and prev_rsi_val > 50 and cur_rsi <= 50)
