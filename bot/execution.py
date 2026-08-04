"""Trade execution — place_trade, place_mean_reversion_trade, scale-out, chandelier."""

import logging
import time
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import state as _st
from discord_alerts import trade_open, trade_partial
from indicators import calc_atr, calc_ma
from journal import journal_close, journal_open
from mt5_connect import _update_dynamic_deviation, get_deviation, get_filling_mode, get_rates, mt5_call
from risk import calc_position_size
from state import _chandelier_state, _exec_quality, _last_trade_time, _scale_out_state, save_bot_state


# MT5's order_send is sensitive to the calling frame: it must be invoked
# via direct attribute access (``mt5.order_send(req)``) from THIS module's
# frame. Routing it through a helper in another module (or as a captured
# argument) makes it return None, so trades silently never fill.
def mt5_order_send(req, _timeout=None):
    return mt5.order_send(req)


def log_execution_quality(cfg, symbol, req_price, fill_price=None, rejected=False):
    if not cfg["eq_enabled"]:
        return
    if symbol not in _exec_quality:
        _exec_quality[symbol] = {"slippage_sum": 0.0, "slippage_count": 0, "rejections": 0, "trades": 0}
    if rejected:
        _exec_quality[symbol]["rejections"] += 1
        return
    if fill_price is not None and req_price > 0:
        slippage = abs(fill_price - req_price) / req_price * 100
        _exec_quality[symbol]["slippage_sum"] += slippage
        _exec_quality[symbol]["slippage_count"] += 1
    _exec_quality[symbol]["trades"] += 1


def _min_stop_points(sinfo, tick=None):
    # Minimum stop distance in points that the broker will accept, plus a modest
    # spread buffer. Previously a hardcoded `50` floor was used, which over-widened
    # stops (and SL modifications) on small-point symbols (agent audit M1).
    spread_pts = 0
    if tick is not None and getattr(sinfo, "point", 0) > 0:
        try:
            spread_pts = int((tick.ask - tick.bid) / sinfo.point)
        except Exception:
            logging.warning("Failed to compute spread in points", exc_info=True)
            spread_pts = 0
    return max(int(sinfo.trade_stops_level), spread_pts + 10)


def _init_scale_out_state(cfg, price, signal, sl_points, sinfo, is_mr=False, volume=0.0, atr_entry=None):
    close_fracs = cfg.get("scale_out_close_fractions", [0.20, 0.20])
    tp_rr_targets = cfg.get("scale_out_tp_targets_rr", [0.50, 0.75])
    tp_atr_targets = cfg.get("scale_out_tp_targets_atr", [1.5, 2.5])
    # MR trades set TP = sl_points * mr_tp_atr_mult (execution places tp with
    # tp_mult_key="mr_tp_atr_mult"), so their effective reward multiple is
    # mr_tp_atr_mult, NOT the trend rr. The RR-based scale-out target must use
    # the same multiple or MR partials land beyond the actual TP and never fire.
    # The backtest already branches on is_mr (backtest.py); live did not (M4).
    rr = cfg.get("mr_tp_atr_mult", 1.5) if is_mr else cfg.get("rr", 2.0)
    return {
        "step": 0,
        "entry_price": price,
        "direction": signal,
        "close_fractions": close_fracs,
        "tp_targets_rr": tp_rr_targets,
        "tp_targets_atr": tp_atr_targets,
        "num_partials": len(close_fracs),
        "sl_points": sl_points,
        "point": sinfo.point,
        "is_mr": is_mr,
        "original_volume": volume,
        "atr_entry": atr_entry,
        "rr": rr,
    }


def check_scale_out(cfg, position):
    symbol = position.symbol
    ticket = position.ticket
    state = _scale_out_state.get(ticket)
    if state is None:
        return False
    if not position.tp:
        logging.warning(f"[{symbol}] Scale-out: ticket={ticket} in _scale_out_state but has no TP")
    tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick is None:
        return False
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        return False
    is_long = position.type == mt5.ORDER_TYPE_BUY
    entry = state["entry_price"]
    close_fracs = state["close_fractions"]
    num_partials = state["num_partials"]
    if state["step"] >= num_partials:
        return False
    needed = cfg["atr_period"] + 10
    rates = get_rates(symbol, cfg["timeframe"], needed)
    if rates is None or len(rates) < needed:
        return False
    cur_atr = calc_atr(rates, cfg["atr_period"])
    if cur_atr is None or cur_atr <= 0:
        return False
    current_price = tick.bid if is_long else tick.ask
    step = state["step"]
    tp_rr = state.get("tp_targets_rr")
    if tp_rr is not None:
        target_fraction = tp_rr[step] if step < len(tp_rr) else tp_rr[-1]
        tp_dist = state["sl_points"] * state["point"] * state.get("rr", 2.0)
        level = entry + tp_dist * target_fraction if is_long else entry - tp_dist * target_fraction
    else:
        tp_atrs = state.get("tp_targets_atr", [1.5, 2.5])
        target_mult = tp_atrs[step] if step < len(tp_atrs) else tp_atrs[-1]
        atr_level = state.get("atr_entry") or cur_atr
        level = entry + atr_level * target_mult if is_long else entry - atr_level * target_mult
        tp_dist = atr_level * tp_atrs[0]
    hit = current_price >= level if is_long else current_price <= level
    if not hit:
        return False
    close_frac = close_fracs[step] if step < len(close_fracs) else close_fracs[-1]
    orig_vol = state.get("original_volume", position.volume)
    if orig_vol is None or orig_vol <= 0:
        orig_vol = position.volume
    vol_step = sinfo.volume_step if sinfo.volume_step > 0 else 0.01
    ideal = int(orig_vol * close_frac / vol_step) * vol_step
    close_vol = max(ideal, vol_step)
    if close_vol < sinfo.volume_min:
        # Sub-minimum-lot partial: trade the smallest allowed lot rather than
        # closing the whole position (agent audit M2).
        close_vol = sinfo.volume_min
    if close_vol >= position.volume and step != num_partials - 1:
        if orig_vol * close_frac < vol_step:
            # Micro-lot: intended partial is below minimum step.
            # Can't partial meaningfully — close full position at this target
            # instead of skipping to a higher target (agent audit M2).
            close_vol = position.volume
        else:
            # A non-final partial that covers the entire remaining position would
            # liquidate the whole trade prematurely. Skip it and advance to the
            # next target instead of closing everything (agent audit M2).
            state["step"] = step + 1
            _scale_out_state[ticket] = state
            save_bot_state()
            return False
    if close_vol <= 0:
        return False
    close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
    close_price = tick.bid if is_long else tick.ask
    close_req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": close_vol,
        "type": close_type,
        "position": ticket,
        "price": close_price,
        "deviation": get_deviation(cfg, symbol),
        "magic": cfg.get("magic", 20240706),
        "comment": "TrendBot-ScaleOut",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol),
    }
    result = mt5_order_send(close_req, _timeout=10)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        return False
    pnl = result.profit if hasattr(result, "profit") else 0.0
    pips = abs(position.price_open - level) / sinfo.point if sinfo.point else 0
    if cfg["trade_journal"]:
        journal_close(ticket, level, pnl, pips, "SCALE_OUT")
    if cfg["discord_url"]:
        trade_partial(cfg["discord_url"], symbol, "buy" if is_long else "sell", close_vol, level, pnl)
    if close_vol >= position.volume:
        _scale_out_state.pop(ticket, None)
        save_bot_state()
        return True
    new_step = step + 1
    state["step"] = new_step
    _scale_out_state[ticket] = state
    if new_step == 1:
        lock_fraction = cfg.get("scale_out_breakeven_fraction", 0.25)
        new_sl = entry + tp_dist * lock_fraction if is_long else entry - tp_dist * lock_fraction
    else:
        if tp_rr is not None:
            prev_fraction = tp_rr[new_step - 2] if new_step - 2 < len(tp_rr) else tp_rr[-1]
            lock_level = entry + tp_dist * prev_fraction if is_long else entry - tp_dist * prev_fraction
        else:
            prev_mult = tp_atrs[new_step - 2] if new_step - 2 < len(tp_atrs) else tp_atrs[-1]
            lock_level = entry + cur_atr * prev_mult if is_long else entry - cur_atr * prev_mult
        new_sl = lock_level
    stops_level = _min_stop_points(sinfo, tick)
    for mult in [1, 2, 4, 8, 16]:
        if is_long:
            min_allowed = current_price - stops_level * mult * sinfo.point
            sl2 = min(new_sl, min_allowed)
        else:
            min_allowed = current_price + stops_level * mult * sinfo.point
            sl2 = max(new_sl, min_allowed)
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl2, "tp": position.tp}
        mod = mt5_order_send(req, _timeout=10)
        if mod is not None and mod.retcode == mt5.TRADE_RETCODE_DONE:
            break
    if new_step >= num_partials:
        tp_req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl2, "tp": 0.0}
        mt5_order_send(tp_req, _timeout=10)
    save_bot_state()
    return True


def check_max_hold(cfg, position):
    """Check if position exceeded max hold hours. Returns True to signal close."""
    max_hours = cfg.get("max_hold_hours", 72)
    if max_hours <= 0:
        return False
    now = datetime.now()
    open_time = datetime.fromtimestamp(getattr(position, "time", 0))
    elapsed_hours = (now - open_time).total_seconds() / 3600
    if elapsed_hours < max_hours:
        return False
    logging.info(f"[{position.symbol}] Max hold {elapsed_hours:.1f}h > {max_hours}h — closing")
    return True


def check_breakeven(cfg, position, atr):
    """After price reaches be_atr_mult * ATR in profit, move SL to breakeven."""
    if not cfg.get("be_enabled", True):
        return
    symbol = position.symbol
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        return
    tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick is None:
        return
    is_long = position.type == mt5.ORDER_TYPE_BUY
    profit_distance = (tick.bid - position.price_open) if is_long else (position.price_open - tick.ask)
    if profit_distance <= 0:
        return
    profit_atrs = profit_distance / atr if atr and atr > 0 else 0
    be_atr_mult = cfg.get("be_atr_mult", 1.0)
    if profit_atrs < be_atr_mult:
        return
    if position.sl and (
        (is_long and position.sl >= position.price_open) or (not is_long and position.sl <= position.price_open)
    ):
        return
    stops_level = _min_stop_points(sinfo, tick)
    for mult in [1, 2, 4, 8, 16]:
        if is_long:
            min_allowed = tick.bid - stops_level * mult * sinfo.point
            new_sl = max(position.price_open, min_allowed)
        else:
            min_allowed = tick.ask + stops_level * mult * sinfo.point
            new_sl = min(position.price_open, min_allowed)
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "sl": new_sl, "tp": position.tp}
        result = mt5_order_send(req, _timeout=10)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"[{symbol}] Breakeven at {profit_atrs:.2f} ATR profit")
            return
    logging.warning(f"[{symbol}] Breakeven SL mod failed at all {5} levels")


def check_chandelier_exit(cfg, position):
    symbol = position.symbol
    if not position.sl:
        logging.warning(
            f"[{symbol}] Chandelier: ticket={position.ticket} has no SL ({position.sl!r}) — chandelier trail disabled"
        )
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        return
    tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick is None:
        return
    needed = cfg["ch_lookback"] + cfg["ch_atr_period"] + 10
    df = get_rates(symbol, cfg["timeframe"], needed)
    if df is None or len(df) < needed:
        return
    entry_idx = df["time"].searchsorted(pd.Timestamp(position.time, unit="s"), side="left")
    bars_since = df.iloc[max(0, entry_idx - 1) :]
    if len(bars_since) < 3:
        return
    full_atr = calc_atr(df, cfg["ch_atr_period"])
    if full_atr is None or full_atr <= 0:
        return
    state = _chandelier_state.get(position.ticket, {"ch_sl": None})
    is_long = position.type == mt5.ORDER_TYPE_BUY
    partial_fired = position.tp is None or position.tp == 0.0
    if partial_fired:
        so_state = _scale_out_state.get(position.ticket)
        if so_state and so_state.get("step", 0) < so_state.get("num_partials", 1):
            partial_fired = False
    base_mult = cfg["ch_atr_mult"]
    if sinfo.point <= 0:
        logging.warning(f"[{symbol}] Chandelier: sinfo.point={sinfo.point} — zero guard, skipping two-stage")
    elif cfg.get("ch_two_stage", True) and not partial_fired and position.sl:
        pnl_points = (
            (tick.bid - position.price_open) / sinfo.point
            if is_long
            else (position.price_open - tick.ask) / sinfo.point
        )
        sl_points = abs(position.price_open - position.sl) / sinfo.point
        r_multiple = pnl_points / max(sl_points, 1)
        if r_multiple >= cfg.get("ch_two_stage_min_r", 3.0):
            base_mult = cfg.get("ch_tight_mult", 1.5)
        else:
            base_mult = cfg.get("ch_loose_mult", 3.5)
    ch_mult = cfg["ch_atr_mult_partial"] if partial_fired else base_mult
    if cfg.get("ch_accelerate_enabled", False):
        try:
            ma_type = cfg.get("ma_type", "kama")
            ema_period = cfg.get("ch_accelerate_period", 14)
            ema_bars = cfg.get("ch_accelerate_bars", 5)
            ema_series = calc_ma(df, ema_period, ma_type)
            if ema_series is not None and len(ema_series) > ema_bars:
                ema_ratio = ema_series.iloc[-1] / ema_series.iloc[-ema_bars]
                strength = cfg.get("ch_accelerate_strength", 0.20)
                accel = (2.0 - ema_ratio) if is_long else ema_ratio
                accel = max(1.0 - strength, min(1.0 + strength, accel))
                if abs(accel - 1.0) > 0.01:
                    ch_mult *= accel
        except Exception:
            logging.debug("Chandelier acceleration calculation failed", exc_info=True)
    if is_long:
        hh = bars_since["high"].max()
        new_sl = hh - full_atr * ch_mult
        if state["ch_sl"] is not None:
            new_sl = max(new_sl, state["ch_sl"])
        elif position.sl is not None:
            new_sl = max(new_sl, position.sl)
    else:
        ll = bars_since["low"].min()
        new_sl = ll + full_atr * ch_mult
        if state["ch_sl"] is not None:
            new_sl = min(new_sl, state["ch_sl"])
        elif position.sl is not None:
            new_sl = min(new_sl, position.sl)
    if state["ch_sl"] is not None and abs(new_sl - state["ch_sl"]) < sinfo.point:
        return
    tick2 = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick2 is None:
        return
    stops_level = _min_stop_points(sinfo, tick2)
    breach = (is_long and tick2.bid < new_sl) or (not is_long and tick2.ask > new_sl)
    if breach:
        close_type = mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY
        close_price = tick2.bid if is_long else tick2.ask
        close_req = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": close_price,
            "deviation": get_deviation(cfg, symbol),
            "magic": cfg.get("magic", 20240706),
            "comment": "TrendBot-Chandelier",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(symbol),
        }
        result = mt5_order_send(close_req, _timeout=10)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            _chandelier_state.pop(position.ticket, None)
            save_bot_state()
            pnl = result.profit if hasattr(result, "profit") else 0.0
            pips = abs(position.price_open - close_price) / sinfo.point if sinfo.point else 0
            if cfg["trade_journal"]:
                journal_close(position.ticket, close_price, pnl, pips, "CHANDELIER")
            if cfg["discord_url"]:
                trade_partial(
                    cfg["discord_url"], symbol, "buy" if is_long else "sell", position.volume, close_price, pnl
                )
        return
    for mult in [1, 2, 4, 8, 16]:
        if is_long:
            min_allowed = tick2.bid - stops_level * mult * sinfo.point
            sl2 = min(new_sl, min_allowed)
        else:
            min_allowed = tick2.ask + stops_level * mult * sinfo.point
            sl2 = max(new_sl, min_allowed)
        req = {"action": mt5.TRADE_ACTION_SLTP, "position": position.ticket, "sl": sl2, "tp": position.tp}
        result = mt5_order_send(req, _timeout=10)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            _chandelier_state[position.ticket] = {"ch_sl": sl2}
            save_bot_state()
            return
    logging.warning(f"[{symbol}] Chandelier SL mod failed at all {5} levels")


def _place_trade_inner(cfg, signal, atr, volume_mult, is_mr=False):
    symbol = cfg["symbol"]
    sl_mult_key = "mr_sl_atr_mult" if is_mr else "atr_sl_mult"
    tp_mult_key = "mr_tp_atr_mult" if is_mr else "rr"
    comment_str = "TrendBot-MR" if is_mr else "TrendBot"
    disc_regime = "MR" if is_mr else cfg.get("regime", "trend")
    # Distinct magic number for MR trades so detection never relies on the
    # comment string (which the broker may truncate). Trend uses cfg["magic"].
    magic = cfg.get("mr_magic", 20240707) if is_mr else cfg.get("magic", 20240706)
    tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick is None:
        logging.warning(f"[{symbol}] order aborted: no market tick")
        return False
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        logging.warning(f"[{symbol}] order aborted: no symbol_info")
        return False
    price = tick.ask if signal == "buy" else tick.bid
    stops_level = _min_stop_points(sinfo, tick)
    sl_mult = cfg[sl_mult_key]
    if not sl_mult or pd.isna(sl_mult) or sl_mult <= 0 or sinfo.point <= 0:
        logging.warning(f"[{symbol}] order aborted: invalid sl_mult={sl_mult} or point={sinfo.point}")
        return False
    sl_points = max(int(atr * sl_mult / sinfo.point), stops_level)
    tp_points = int(sl_points * cfg[tp_mult_key])
    if signal == "buy":
        sl = price - sl_points * sinfo.point
        tp = price + tp_points * sinfo.point
        order_type = mt5.ORDER_TYPE_BUY
    else:
        sl = price + sl_points * sinfo.point
        tp = price - tp_points * sinfo.point
        order_type = mt5.ORDER_TYPE_SELL
    volume = calc_position_size(cfg, sl_points, volume_mult)
    if volume <= 0:
        logging.warning(f"[{symbol}] order aborted: computed volume<=0 (sl_points={sl_points}, mult={volume_mult})")
        return False
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "deviation": get_deviation(cfg, symbol),
        "magic": magic,
        "comment": comment_str,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol),
    }
    req = request.copy()
    req["sl"] = sl
    req["tp"] = tp
    result = mt5_order_send(req, _timeout=10)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        _update_dynamic_deviation(symbol, True, cfg)
        log_execution_quality(cfg, symbol, price, result.price, rejected=False)
        logging.info(
            f"[{symbol}] ORDER FILLED: {signal} ticket={result.order} vol={volume:.4f} "
            f"entry={price:.5f} sl={sl:.5f} tp={tp:.5f} "
            f"(atr_sl_mult={cfg.get('atr_sl_mult')} rr={cfg.get('rr')})"
        )
        ticket = result.order
        if cfg["trade_journal"]:
            journal_open(ticket, symbol, signal, volume, price, sl, tp, atr)
        if cfg["discord_url"]:
            trade_open(cfg["discord_url"], symbol, signal, volume, price, sl, tp, atr, disc_regime)
        if cfg["scale_out_enabled"]:
            _scale_out_state[ticket] = _init_scale_out_state(
                cfg, price, signal, sl_points, sinfo, is_mr=is_mr, volume=volume, atr_entry=atr
            )
        save_bot_state()
        return True
    logging.warning(
        f"[{symbol}] order_send failed: retcode={getattr(result, 'retcode', 'None')} "
        f"comment={getattr(result, 'comment', '')}"
    )
    _update_dynamic_deviation(symbol, False, cfg)
    log_execution_quality(cfg, symbol, price, rejected=True)
    # NOTE: mt5.order_send mutates the request dict in place; build a fresh
    # dict for the retry so a mutated dict cannot trigger (-2, 'Unnamed
    # arguments not allowed').
    retry_req = request.copy()
    retry_req["sl"] = 0.0
    retry_req["tp"] = 0.0
    result = mt5_order_send(retry_req, _timeout=10)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.warning(
            f"[{symbol}] order_send (no SL/TP) failed: retcode={getattr(result, 'retcode', 'None')} "
            f"comment={getattr(result, 'comment', '')}"
        )
        _update_dynamic_deviation(symbol, False, cfg)
        return False
    fill_price = result.price
    ticket = result.order
    for mult in [1, 2, 4, 8]:
        stretch = mult
        sl2 = (
            fill_price - sl_points * stretch * sinfo.point
            if signal == "buy"
            else fill_price + sl_points * stretch * sinfo.point
        )
        tp_mult_val = cfg[tp_mult_key]
        tp2 = (
            fill_price + sl_points * stretch * tp_mult_val * sinfo.point
            if signal == "buy"
            else fill_price - sl_points * stretch * tp_mult_val * sinfo.point
        )
        modify_req = {"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": sl2, "tp": tp2}
        mod_result = mt5_order_send(modify_req, _timeout=10)
        if mod_result is not None and mod_result.retcode == mt5.TRADE_RETCODE_DONE:
            if cfg["trade_journal"]:
                journal_open(ticket, symbol, signal, volume, fill_price, sl2, tp2, atr)
            if cfg["discord_url"]:
                trade_open(cfg["discord_url"], symbol, signal, volume, fill_price, sl2, tp2, atr, disc_regime)
            if cfg["scale_out_enabled"]:
                _scale_out_state[ticket] = _init_scale_out_state(
                    cfg, fill_price, signal, sl_points, sinfo, is_mr=is_mr, volume=volume, atr_entry=atr
                )
            save_bot_state()
            return True
    # All SL/TP mod attempts failed — journal the open so it is always traced,
    # then attempt to close the naked position (agent audit C2, M1).
    if cfg["trade_journal"]:
        journal_open(ticket, symbol, signal, volume, fill_price, 0.0, 0.0, atr)
    close_type = mt5.ORDER_TYPE_SELL if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
    close_tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if close_tick is None:
        return False
    close_price = close_tick.bid if order_type == mt5.ORDER_TYPE_BUY else close_tick.ask
    close_cmt = "TrendBot-MR-CloseNaked" if is_mr else "TrendBot-CloseNaked"
    close_result = mt5_order_send(
        {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": close_type,
            "position": ticket,
            "price": close_price,
            "deviation": get_deviation(cfg, symbol),
            "magic": cfg.get("magic", 20240706),
            "comment": close_cmt,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": get_filling_mode(symbol),
        },
        _timeout=10,
    )
    close_event = "MR_NAKED_CLOSE" if is_mr else "TREND_NAKED_CLOSE"
    if close_result and close_result.retcode == mt5.TRADE_RETCODE_DONE:
        tick_val = sinfo.trade_tick_value or 0
        tick_sz = sinfo.trade_tick_size or 1
        if tick_sz > 0 and tick_val > 0:
            if signal == "buy":
                close_pnl = (close_price - fill_price) / tick_sz * tick_val * volume
            else:
                close_pnl = (fill_price - close_price) / tick_sz * tick_val * volume
        else:
            close_pnl = 0.0
        close_pips = abs(close_price - fill_price) / max(sinfo.point, 1e-10)
        if cfg["trade_journal"]:
            journal_close(ticket, close_price, close_pnl, close_pips, close_event)
        logging.warning(f"[{symbol}] {close_cmt} closed (pnl={close_pnl:.2f})")
    else:
        logging.warning(
            f"[{symbol}] {close_cmt} failed: {getattr(close_result, 'retcode', 'None')}; will be managed by exit logic"
        )
    return False


def place_limit_order(cfg, signal, atr, kelly_mult, regime_mult=1.0):
    """Place a pending limit order at improved price for low-confidence signals."""
    symbol = cfg["symbol"]
    tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
    if tick is None:
        return False
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        return False

    spread = tick.ask - tick.bid
    offset = cfg.get("le_offset_spreads", 0.5) * spread
    limit_price = tick.bid - offset if signal == "buy" else tick.ask + offset
    order_type = mt5.ORDER_TYPE_BUY_LIMIT if signal == "buy" else mt5.ORDER_TYPE_SELL_LIMIT

    volume_mult = kelly_mult * regime_mult
    sl_mult = cfg["atr_sl_mult"]
    stops_level = _min_stop_points(sinfo, tick)
    if sinfo.point <= 0:
        logging.warning(f"[{symbol}] place_limit_order: sinfo.point={sinfo.point} — zero guard")
        return False
    sl_points = max(int(atr * sl_mult / sinfo.point), stops_level)
    tp_points = int(sl_points * cfg["rr"])

    if signal == "buy":
        sl = limit_price - sl_points * sinfo.point
        tp = limit_price + tp_points * sinfo.point
    else:
        sl = limit_price + sl_points * sinfo.point
        tp = limit_price - tp_points * sinfo.point

    volume = calc_position_size(cfg, sl_points, volume_mult)
    if volume <= 0:
        return False

    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": limit_price,
        "sl": sl,
        "tp": tp,
        "deviation": get_deviation(cfg, symbol),
        "magic": cfg.get("magic", 20240706),
        "comment": "TrendBot-LIMIT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": get_filling_mode(symbol),
    }
    result = mt5_order_send(request, _timeout=10)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        ticket = result.order
        _st._pending_limits[symbol] = {
            "ticket": ticket,
            "signal": signal,
            "price": limit_price,
            "atr": atr,
            "cycles": 0,
            "kelly_mult": kelly_mult,
            "regime_mult": regime_mult,
        }
        logging.info(f"[{symbol}] LIMIT ORDER placed: {signal} @ {limit_price:.5f} ticket={ticket}")
        return True
    logging.info(f"[{symbol}] LIMIT ORDER failed: retcode={getattr(result, 'retcode', 'None')}")
    return False


def cancel_pending_limit(ticket, symbol):
    """Cancel an unfilled pending limit order.

    MT5 has no ``order_delete`` API — pending orders are cancelled via
    ``order_send`` with ``action = TRADE_ACTION_REMOVE``. (The previous
    implementation called the non-existent ``mt5.order_delete``, which raised
    AttributeError inside mt5_call, was swallowed, and left the order live
    while the bot kept re-logging "cancelling" every cycle.)
    """
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order": ticket,
    }
    result = mt5_order_send(request, _timeout=5)
    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
        _st._pending_limits.pop(symbol, None)
        logging.info(f"[{symbol}] Pending limit cancelled ticket={ticket}")
        return True
    retcode = getattr(result, "retcode", "None") if result is not None else "None"
    logging.warning(f"[{symbol}] Cancel limit failed: retcode={retcode} ticket={ticket}")
    return False


def check_limit_orders(cfg):
    """Check pending limit orders on each cycle — fill check, staleness, cancel.

    Returns a set of symbols whose pending limits were filled (new positions to
    register) so the caller can skip normal entry logic for those symbols.
    """
    filled = set()
    if not cfg.get("le_enabled", True):
        return filled
    if not _st._pending_limits:
        return filled
    for symbol, info in list(_st._pending_limits.items()):
        ticket = info["ticket"]
        try:
            orders = mt5_call(mt5.orders_get, ticket=ticket, _timeout=5)
        except Exception:
            logging.warning("Failed to check pending limit orders", exc_info=True)
            orders = None
        if orders is None or len(orders) == 0:
            # Order no longer pending — check if it filled as a position
            positions = mt5_call(mt5.positions_get, ticket=ticket, _timeout=5)
            if positions is not None and len(positions) > 0:
                filled.add(symbol)
                _st._pending_limits.pop(symbol, None)
                logging.info(f"[{symbol}] Limit order filled ticket={ticket}")
            else:
                # Order disappeared without filling — remove from tracking
                _st._pending_limits.pop(symbol, None)
        else:
            info["cycles"] += 1
            max_cycles = cfg.get("le_max_cycles", 30)
            if info["cycles"] >= max_cycles:
                logging.info(f"[{symbol}] Limit order stale ({info['cycles']} cycles) — cancelling ticket={ticket}")
                ok = cancel_pending_limit(ticket, symbol)
                if not ok:
                    # Don't retry forever on a persistent failure; drop tracking so
                    # the symbol can re-enter normally on the next signal.
                    info["cancel_failures"] = info.get("cancel_failures", 0) + 1
                    if info["cancel_failures"] >= 3:
                        logging.error(
                            f"[{symbol}] Limit cancel failed 3x — dropping tracking ticket={ticket} "
                            f"(order may still be live; manual check advised)"
                        )
                        _st._pending_limits.pop(symbol, None)
    return filled


def place_trade(cfg, signal, atr, regime_mult=1.0):
    symbol = cfg["symbol"]
    now = time.time()
    if now - _last_trade_time.get(f"trend:{symbol}", 0) < 120:
        return False
    # Throttle is set only AFTER a successful fill (agent audit M3): setting it
    # before sending would block re-entry for 120s even when the order fails.
    ok = _place_trade_inner(cfg, signal, atr, regime_mult, is_mr=False)
    if ok:
        _last_trade_time[f"trend:{symbol}"] = now
    return ok


def place_mean_reversion_trade(cfg, signal, atr, kelly_mult=1.0):
    symbol = cfg["symbol"]
    now = time.time()
    if now - _last_trade_time.get(f"mr:{symbol}", 0) < 120:
        return False
    volume_mult = cfg["mr_position_size_mult"] * kelly_mult
    ok = _place_trade_inner(cfg, signal, atr, volume_mult, is_mr=True)
    if ok:
        _last_trade_time[f"mr:{symbol}"] = now
    return ok
