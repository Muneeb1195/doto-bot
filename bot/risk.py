"""Position sizing — Kelly, volatility adjustment, min-lot cap, tail risk cap."""

import contextlib
import csv
import logging
import time

import numpy as np
import pandas as pd
import state as _st
from _mt5 import mt5
from mt5_connect import get_rates, mt5_call
from state import TRADE_CSV

_TRADE_STATS_CACHE = None
_TRADE_STATS_CACHE_KEY = None
_TRADE_STATS_TTL = 30


def calc_position_size(cfg, sl_points, regime_mult=1.0):
    account_info = mt5_call(mt5.account_info)
    if account_info is None:
        logging.warning("Cannot size: no account info")
        return 0.0
    balance = account_info.balance
    if balance <= 0:
        logging.warning(f"Balance <= 0 ({balance}), skipping trade")
        return 0.0
    risk_amount = balance * (cfg["risk_percent"] / 100)
    symbol_info = mt5_call(mt5.symbol_info, cfg["symbol"])
    if symbol_info is None:
        logging.warning(f"Cannot size {cfg['symbol']}: no symbol info")
        return 0.0
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    if tick_size == 0 or tick_value == 0:
        logging.warning(f"Cannot size {cfg['symbol']}: tick_size={tick_size}, tick_value={tick_value}")
        return 0.0
    sl_distance = sl_points * symbol_info.point
    sl_value = (sl_distance / tick_size) * tick_value
    if sl_value <= 0:
        logging.warning(f"Cannot size {cfg['symbol']}: sl_value={sl_value}")
        return 0.0
    vol_step = symbol_info.volume_step if symbol_info.volume_step > 0 else 0.01
    raw_volume = risk_amount / max(sl_value, 1e-10)
    if raw_volume < vol_step:
        min_lot_risk = vol_step * sl_value
        risk_ratio = min_lot_risk / max(risk_amount, 1e-10)
        max_risk_ratio = cfg.get("max_risk_ratio", 2.0)
        if risk_ratio > max_risk_ratio:
            logging.info(
                f"Min lot risk {min_lot_risk:.2f} is {risk_ratio:.1f}x > {max_risk_ratio:.1f}x "
                f"risk amount {risk_amount:.2f}, skipping"
            )
            return 0.0
    volume = raw_volume * regime_mult
    if not np.isfinite(volume):
        return 0.0
    volume = max(round(volume / vol_step, 0) * vol_step, symbol_info.volume_min)
    volume = min(volume, symbol_info.volume_max)
    max_tail_risk = balance * cfg.get("max_tail_risk_pct", 1.5) / 100
    if volume * sl_value > max_tail_risk:
        logging.info(f"Volume {volume:.4f} risks {volume * sl_value:.2f} > max tail {max_tail_risk:.2f}, skipping")
        return 0.0
    return volume


_CLOSING_EVENTS = ("CLOSE", "SCALE_OUT", "CHANDELIER", "MR_NAKED_CLOSE", "MANUAL_CLOSE")


def get_recent_trade_stats(cfg):
    """Kelly stats on R-multiples, aggregated per ticket and per symbol.

    Previous version read raw currency PnL across ALL symbols and counted each
    journal row as a trade (so scale-out partials inflated the sample). Now we:
      * group every row by ticket and only count fully-closed trades,
      * filter to this symbol (N1),
      * aggregate OPEN+PARTIAL+CLOSE pnl per ticket (M6 — scale-out partials
        are part of one trade outcome, not separate trades),
      * express outcome as an R-multiple = total_realized_pnl / initial_risk
        (H3) where initial_risk is reconstructed from the OPEN row's
        volume/entry/sl and the live symbol tick parameters (avoids a schema
        change to the journal).
    """
    global _TRADE_STATS_CACHE, _TRADE_STATS_CACHE_KEY
    if not TRADE_CSV.exists():
        return None, None, None
    symbol = cfg["symbol"]
    lookback = int(cfg["dr_lookback"])
    cache_key = None
    try:
        st = TRADE_CSV.stat()
        # Bucket time into _TRADE_STATS_TTL-second windows so the cache
        # invalidates shortly after new trades are journaled.
        cache_key = (symbol, st.st_mtime, st.st_size, lookback, int(time.time() // _TRADE_STATS_TTL))
        if _TRADE_STATS_CACHE is not None and cache_key == _TRADE_STATS_CACHE_KEY:
            return _TRADE_STATS_CACHE
    except Exception:
        pass
    try:
        with open(TRADE_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            tickets = {}
            order = []
            for row in reader:
                t = (row.get("ticket") or "").strip()
                if not t:
                    continue
                if t not in tickets:
                    tickets[t] = {"open": None, "pnls": [], "closed": False, "entry_time": ""}
                    order.append(t)
                d = tickets[t]
                evt = row.get("event", "")
                if evt == "OPEN":
                    d["open"] = row
                    d["entry_time"] = row.get("entry_time", "")
                elif evt in _CLOSING_EVENTS:
                    d["closed"] = True
                pnl_s = row.get("pnl", "")
                if pnl_s not in ("", None):
                    with contextlib.suppress(ValueError, TypeError):
                        d["pnls"].append(float(pnl_s))

        symbol_info = mt5_call(mt5.symbol_info, symbol)

        def ticket_risk(open_row):
            if symbol_info is None:
                return None
            try:
                vol = float(open_row.get("volume") or 0)
                entry = float(open_row.get("entry_price") or 0)
                sl = float(open_row.get("sl") or 0)
            except (ValueError, TypeError):
                return None
            if vol <= 0 or entry <= 0 or sl <= 0 or sl == entry:
                return None
            tick_size = symbol_info.trade_tick_size
            tick_value = symbol_info.trade_tick_value
            point = symbol_info.point
            if tick_size == 0 or tick_value == 0 or point == 0:
                return None
            sl_distance = abs(entry - sl)
            sl_value = (sl_distance / tick_size) * tick_value
            if sl_value <= 0:
                return None
            return vol * sl_value

        rs = []  # R-multiples (production path, when OPEN+risk available)
        cur = []  # legacy currency PnL per ticket (fallback for journals that
        # lack OPEN rows / symbol_info, e.g. unit tests)
        for t in order:
            d = tickets[t]
            if not d["closed"]:
                continue
            open_row = d["open"]
            sym = (open_row.get("symbol") or "").strip() if open_row is not None else ""
            if sym and sym != symbol:
                continue
            total_pnl = sum(d["pnls"])
            if open_row is not None:
                risk = ticket_risk(open_row)
                if risk is not None and risk > 0:
                    rs.append((d["entry_time"], total_pnl / risk))
                    continue
            if not sym or sym == symbol:
                cur.append(total_pnl)

        def stats_from(values):
            if len(values) < 10:
                return None, None, None
            values = values[-lookback:]
            wins = [v for v in values if v > 0]
            losses = [v for v in values if v < 0]
            win_rate = len(wins) / len(values) if values else 0.0
            avg_win = float(np.mean(wins)) if wins else 0.0
            avg_loss = float(abs(np.mean(losses))) if losses else 0.0
            return win_rate, avg_win, avg_loss

        if rs:
            rs.sort(key=lambda x: x[0])
            result = stats_from([r for _, r in rs])
        else:
            result = stats_from(cur)
        _TRADE_STATS_CACHE = result
        _TRADE_STATS_CACHE_KEY = cache_key
        return result
    except Exception:
        logging.warning("get_recent_trade_stats failed", exc_info=True)
        return None, None, None


def drawdown_pct(equity, update_peak=False):
    """Drawdown percent below the running equity peak.

    Single source for the peak-relative drawdown math shared by the tail-risk
    check (filters.check_tail_risk, which tracks the peak) and Kelly sizing
    (calc_kelly_mult, which only reads it). With `update_peak=True` the running
    peak in state is raised when `equity` sets a new high.
    """
    if update_peak and equity > _st._peak_balance:
        _st._peak_balance = equity
    peak = max(_st._peak_balance, 1)
    return ((peak - equity) / peak) * 100


def calc_kelly_mult(cfg):
    if not cfg["dr_enabled"]:
        return 1.0
    win_rate, avg_win, avg_loss = get_recent_trade_stats(cfg)
    if win_rate is None or avg_win <= 0 or avg_loss <= 0:
        return cfg["dr_min_mult"]
    b = avg_win / avg_loss
    q = 1.0 - win_rate
    kelly = (win_rate * b - q) / b if b > 0 else 0
    kelly = max(0.0, kelly) * cfg["dr_kelly_fraction"]
    kelly = max(cfg["dr_min_mult"], min(cfg["dr_max_mult"], kelly))
    dd_reduce_pct = cfg.get("dd_kelly_reduction_pct", 5.0)
    if dd_reduce_pct > 0:
        try:
            acc = mt5_call(mt5.account_info)
            if acc is not None:
                bal = getattr(acc, "balance", None)
                prof = getattr(acc, "profit", None)
                if isinstance(bal, (int, float)) and isinstance(prof, (int, float)):
                    dd_pct = drawdown_pct(bal + prof)
                    if dd_pct >= dd_reduce_pct:
                        kelly *= 0.5
                        logging.info(f"Kelly reduced by 50% due to drawdown {dd_pct:.1f}% >= {dd_reduce_pct}%")
        except Exception:
            logging.debug("Failed to compute drawdown-adjusted kelly", exc_info=True)
    return kelly


def calc_volatility_mult(cfg):
    if not cfg["dr_vol_adjust"]:
        return 1.0
    needed = cfg["atr_period"] + 60
    df = get_rates(cfg["symbol"], cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        return 1.0
    from indicators import calc_atr_series

    atr_series = calc_atr_series(df, cfg["atr_period"])
    atr_sma50 = atr_series.rolling(window=50).mean()
    cur_atr = atr_series.iloc[-1]
    cur_sma50 = atr_sma50.iloc[-1]
    if pd.isna(cur_atr) or pd.isna(cur_sma50) or cur_sma50 <= 0 or cur_atr <= 0:
        return 1.0
    ratio = cur_atr / cur_sma50
    if ratio > 1.2:
        # ratio > 1.2 guarantees 1.0/ratio < 0.833, so no upper cap is needed.
        return max(0.25, 1.0 / ratio)
    return 1.0
