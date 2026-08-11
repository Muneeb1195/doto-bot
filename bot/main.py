"""Main loop — orchestrates all modules."""

import contextlib
import csv
import logging
import logging.handlers
import os
import signal
import sys
import threading
import time
from copy import deepcopy
from datetime import datetime


# MT5's order_send is sensitive to the calling frame (see execution.py).
def mt5_order_send(req, _timeout=None):
    return mt5.order_send(req)


def _read_todays_trades(path, today):
    """Return closed rows from `path` whose exit_time starts with `today`.

    The journal is append-only and sorted ascending by time, so today's rows
    are at the tail. We seek to a tail window and parse forward with
    csv.DictReader (correct column names, no positional assumptions), stopping
    once we pass today's date. Falls back to a full forward scan for small
    files.
    """
    try:
        size = path.stat().st_size
        tail_window = 1 << 17  # 128 KB tail window
        if size <= tail_window:
            with open(path, newline="") as f:
                return [
                    r
                    for r in csv.DictReader(f)
                    if r.get("exit_time", "").startswith(today)
                    and r.get("event") not in ("OPEN", "")
                    and r.get("pnl")
                ]
        with open(path, "rb") as fb:
            start = max(0, size - tail_window)
            fb.seek(start)
            if start > 0:
                fb.readline()
            remainder = fb.read().decode("utf-8", "replace")
        if not remainder.strip():
            return []
        rows = [
            r
            for r in csv.DictReader(remainder.splitlines())
            if r.get("exit_time", "").startswith(today)
            and r.get("event") not in ("OPEN", "")
            and r.get("pnl")
        ]
        if rows:
            return rows
        with open(path, newline="") as f:
            return [
                r
                for r in csv.DictReader(f)
                if r.get("exit_time", "").startswith(today)
                and r.get("event") not in ("OPEN", "")
                and r.get("pnl")
            ]
    except Exception:
        logging.debug("Tail-scan of trades journal failed; returning empty", exc_info=True)
        return []


if getattr(sys, "_base_executable", sys.executable) != sys.executable:
    sys._base_executable = sys.executable  # type: ignore[attr-defined]

os.environ["JOBLIB_PARALLEL_BACKEND"] = "threading"

try:
    import MetaTrader5 as mt5  # noqa: E402
except ImportError:  # Linux: no native package, use the mt5linux RPyC bridge
    from mt5_connect import mt5  # noqa: E402
import numpy as np  # noqa: E402
import state as _st  # noqa: E402
from analytics import fused_regime_score  # noqa: E402
from correlation import compute_correlation_matrix, get_correlation_reduction  # noqa: E402
from discord_alerts import bot_start, daily_summary  # noqa: E402
from drift_detector import consume_warmstart_queue  # noqa: E402
from drift_retrain import warmstart_model as _warmstart_model  # noqa: E402
from execution import (  # noqa: E402
    check_breakeven,
    check_chandelier_exit,
    check_limit_orders,
    check_max_hold,
    check_scale_out,
    place_limit_order,
    place_mean_reversion_trade,
    place_trade,
)
from filters import (  # noqa: E402
    check_capital_eligibility,
    check_daily_loss,
    check_execution_sanity,
    check_ml_gate,
    check_tail_risk,
    load_ml_models,
)
from journal import _reconcile_external_deals, journal_close, journal_init, reconcile_journal  # noqa: E402
from mt5_connect import (  # noqa: E402
    _update_dynamic_deviation,
    can_trade_symbol,
    ensure_mt5_connected,
    get_deviation,
    get_filling_mode,
    get_rates,
    market_open,
    mt5_call,
)
from regime import get_current_atr  # noqa: E402
from risk import calc_kelly_mult, calc_volatility_mult  # noqa: E402
from signals import (  # noqa: E402
    _get_regime_gate,
    check_htf_trend,
    check_mean_reversion_exit,
    get_mean_reversion_signal,
    get_mtf_fused_signal,
    get_signal,
)
from state import (  # noqa: E402
    LOG_DIR,
    TRADE_CSV,
    _chandelier_state,
    _exec_bias,
    _filter_stats,
    _last_corr_time,
    _last_trade_time,
    _scale_out_state,
    load_bot_state,
    load_news_sentiment,
    save_bot_state,
)

from config import apply_symbol_overrides, apply_symbol_strategy, load_config  # noqa: E402
from dashboard import write_dashboard_state  # noqa: E402


def _watchdog_loop(cfg):
    """Daemon thread that monitors main-loop progress and kills MT5 on stall."""
    max_cycle = cfg.get("max_cycle_seconds", 180)
    check_interval = max(30, max_cycle // 2)
    while not _st._shutdown_requested:
        time.sleep(check_interval)
        now = time.time()
        elapsed = now - _st._last_cycle_time
        if _st._last_cycle_time > 0 and elapsed < max_cycle * 2:
            _st._WATCHDOG_FAILURES = 0
            continue
        if _st._last_cycle_time == 0:
            continue
        _st._WATCHDOG_FAILURES += 1
        logging.critical(
            f"Watchdog: main thread stalled {elapsed:.0f}s "
            f"(failure {_st._WATCHDOG_FAILURES}/{_st._WATCHDOG_MAX_FAILURES})"
        )
        if _st._WATCHDOG_FAILURES >= _st._WATCHDOG_MAX_FAILURES:
            logging.critical("Watchdog: max consecutive stalls — force exiting")
            save_bot_state()
            os._exit(1)
        try:
            import platform
            import subprocess

            if platform.system() == "Linux":
                subprocess.run(["pkill", "-f", "terminal64.exe"], capture_output=True, timeout=5)
            else:
                subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True, timeout=5)
        except Exception:
            logging.warning("Watchdog: could not kill terminal64.exe (may already be dead)", exc_info=True)
        _st._WATCHDOG_LAST_KILL = now


def _handle_shutdown(signum, frame):  # noqa: ARG001
    if _st._shutdown_requested:
        logging.warning("Second signal received — forcing exit")
        with contextlib.suppress(Exception):
            mt5_call(mt5.shutdown, _timeout=5)
        os._exit(0)
    _st._shutdown_requested = True
    logging.info(f"Signal {signum} received — shutting down cleanly...")
    save_bot_state()
    with contextlib.suppress(Exception):
        mt5_call(mt5.shutdown, _timeout=5)
    os._exit(0)


def _apply_corr_ml_sizing(
    sym_cfg, symbol, all_positions, kelly_mult, ml_conf, confidence_mult=1.0
):
    if sym_cfg["corr_enabled"]:
        existing_syms = [p.symbol for p in all_positions if p.symbol != symbol]
        corr_reduction = get_correlation_reduction(
            _st._corr_cache, symbol, existing_syms, sym_cfg.get("corr_reduction_max", 0.5)
        )
    else:
        corr_reduction = 1.0
    kelly_mult *= corr_reduction
    kelly_mult *= confidence_mult
    return kelly_mult


def main():
    LOG_DIR.mkdir(exist_ok=True)
    cfg = load_config()
    log_level = logging.DEBUG if cfg.get("verbose_debug") else logging.INFO
    log_handler = logging.handlers.TimedRotatingFileHandler(
        LOG_DIR / "bot.log",
        when="midnight",
        backupCount=30,
        utc=True,
    )
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.basicConfig(level=log_level, handlers=[log_handler, logging.StreamHandler()])
    # Startup grace period: the MT5 terminal (and the mt5linux bridge) can take well
    # over a minute to come up under Wine, so retry in-process before aborting.
    startup_deadline = time.time() + cfg.get("mt5_startup_grace_sec", 180)
    while not ensure_mt5_connected(cfg):
        if time.time() >= startup_deadline:
            logging.critical("MT5 unavailable on startup — aborting")
            return
        logging.warning("MT5 not ready yet — retrying during startup grace period")
        time.sleep(10)

    account_info = mt5_call(mt5.account_info, _timeout=10)
    if account_info is None:
        logging.warning("Account info unavailable — retrying with watchdog")
        if not ensure_mt5_connected(cfg):
            logging.critical("MT5 unavailable after watchdog — aborting")
            return
        account_info = mt5_call(mt5.account_info, _timeout=10)
    if account_info is not None:
        logging.info(f"Connected: {account_info.name} | Balance: Rs.{account_info.balance:.2f}")
    else:
        logging.critical("Cannot get account info — aborting")
        return

    if cfg["discord_url"] and account_info is not None:
        bot_start(cfg["discord_url"], cfg["symbols"], account_info.balance)

    for symbol in cfg["symbols"]:
        mt5_call(mt5.symbol_select, symbol, True, _timeout=10)

    symbols_str = ", ".join(cfg["symbols"])
    logging.info(
        f"Bot started on [{symbols_str}] | "
        f"Risk: {cfg['risk_percent']}%/trade | "
        f"Max total positions: {cfg['max_total_positions']} | "
        f"Strategy: {cfg.get('ma_type', 'kama').upper()}{cfg['ema_fast']}/{cfg['ema_slow']} crossover"
    )

    _st._last_symbol_reselect_time = time.time()

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    watchdog = threading.Thread(target=_watchdog_loop, args=(cfg,), daemon=True)
    watchdog.start()
    logging.info("Watchdog thread started")

    if cfg["trade_journal"]:
        journal_init()
        logging.info(f"Trade journal: {TRADE_CSV}")

    load_ml_models(cfg)
    load_bot_state()

    active_positions = mt5_call(mt5.positions_get, _timeout=10)
    if active_positions is None:
        active_positions = []
    active_tickets = {p.ticket for p in active_positions}
    stale_scale = [t for t in _scale_out_state if t not in active_tickets]
    for t in stale_scale:
        _scale_out_state.pop(t, None)
    stale_chandelier = [t for t in _chandelier_state if t not in active_tickets]
    for t in stale_chandelier:
        _chandelier_state.pop(t, None)
    if stale_scale or stale_chandelier:
        save_bot_state()

    for pos in active_positions:
        # Reconstruct each position's state with its OWN per-symbol config
        # (overrides + strategy), not the base cfg — previously the base config
        # was used, so symbols with overrides (e.g. crypto VIDYA) got wrong
        # scale-out/SL parameters on restart (agent audit M8).
        sym_cfg = deepcopy(cfg)
        sym_cfg["symbol"] = pos.symbol
        apply_symbol_strategy(sym_cfg, pos.symbol)
        apply_symbol_overrides(sym_cfg, pos.symbol)
        if sym_cfg["scale_out_enabled"] and pos.ticket not in _scale_out_state:
            # Detect MR trades by magic number (broker-truncation-proof); fall
            # back to comment for positions opened before mr_magic was adopted.
            mr_magic = sym_cfg.get("mr_magic", 20240707)
            is_mr = (getattr(pos, "magic", None) == mr_magic) or (
                hasattr(pos, "comment") and pos.comment == "TrendBot-MR"
            )
            sl_dist = abs(pos.price_open - pos.sl) if pos.sl else 0
            if sl_dist > 0:
                sinfo = mt5_call(mt5.symbol_info, pos.symbol, _timeout=5)
                if sinfo:
                    sl_pts = int(sl_dist / sinfo.point) if sinfo.point else 50
                    if sl_pts <= 0:
                        logging.warning(
                            f"[{pos.symbol}] AATR: sl_pts={sl_pts} ≤ 0 "
                            f"(sl_dist={sl_dist}, point={getattr(sinfo, 'point', None)})"
                        )
                    if pos.tp and abs(pos.tp - pos.price_open) > sl_dist:
                        tp_pts = int(abs(pos.tp - pos.price_open) / sinfo.point) if sinfo.point else 0
                        if tp_pts <= 0:
                            logging.warning(f"[{pos.symbol}] AATR: tp_pts={tp_pts} ≤ 0")
                        # Reconstruct the SL distance from the live TP. For MR the
                        # TP is ATR-multiple based, so use the MR SL/TP ATR mults
                        # (not the trend RR) — previously MR was restored with the
                        # trend RR (agent audit M9).
                        if tp_pts > 0:
                            if is_mr:
                                mr_sl = sym_cfg.get("mr_sl_atr_mult", 1.0)
                                mr_tp = sym_cfg.get("mr_tp_atr_mult", 1.0)
                                if mr_tp > 0:
                                    sl_pts = int(tp_pts * mr_sl / mr_tp)
                            elif sym_cfg["rr"] > 0:
                                sl_pts = int(tp_pts / sym_cfg["rr"])
                    sl_dist_pts = sl_pts * sinfo.point
                    atr_reconstructed = (
                        sl_dist_pts / sym_cfg.get("atr_sl_mult", 1.0)
                        if not is_mr
                        else sl_dist_pts / sym_cfg.get("mr_sl_atr_mult", 1.0)
                    )
                    from execution import _init_scale_out_state

                    _scale_out_state[pos.ticket] = _init_scale_out_state(
                        sym_cfg,
                        pos.price_open,
                        "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                        sl_pts,
                        sinfo,
                        is_mr=is_mr,
                        volume=pos.volume,
                        atr_entry=atr_reconstructed,
                    )
        if pos.ticket not in _chandelier_state and pos.sl:
            _chandelier_state[pos.ticket] = {"ch_sl": pos.sl}
    if _scale_out_state or _chandelier_state:
        save_bot_state()

    regimes = {s: "uncertain" for s in cfg["symbols"]}
    _st._last_cycle_time = time.time()

    while True:
        if _st._shutdown_requested:
            break
        elapsed = time.time() - _st._last_cycle_time
        max_cycle = cfg.get("max_cycle_seconds", 180)
        if elapsed > max_cycle:
            _st._cycle_consecutive_failures += 1
            logging.warning(f"Cycle stalled for {elapsed:.0f}s ({_st._cycle_consecutive_failures}x consecutive)")
            if _st._cycle_consecutive_failures >= 5:
                logging.critical("5 consecutive stalled cycles — exiting")
                save_bot_state()
                os._exit(1)
            if _st._cycle_consecutive_failures >= 3:
                logging.info("3+ stalled cycles — attempting reconnect")
                mt5_call(mt5.shutdown, _timeout=5)
                time.sleep(5)
        if not ensure_mt5_connected(cfg):
            logging.warning("MT5 unavailable — sleeping {:d}s".format(cfg.get("reconnect_sleep", 30)))
            _st._cycle_consecutive_failures = 0
            time.sleep(cfg.get("reconnect_sleep", 30))
            continue
        load_news_sentiment()
        pending_warmstart = consume_warmstart_queue()
        if pending_warmstart:
            logging.info(f"Warm-start consumer: {pending_warmstart}")
            for sym in pending_warmstart:
                _warmstart_model(sym, cfg)
                # Re-arm drift detection for this symbol. _ml_drift_warned is a
                # one-shot guard set when drift is first flagged; if we never
                # clear it the symbol is permanently excluded from all future
                # drift handling for the session (agent audit D3).
                _st._ml_drift_warned.discard(sym)
        try:
            all_positions = mt5_call(mt5.positions_get, _timeout=10)
            positions_valid = all_positions is not None
            if all_positions is None:
                all_positions = []
            total_positions = len(all_positions)
            active_tickets_set = {p.ticket for p in all_positions}

            if positions_valid:
                reconcile_journal(active_tickets_set)
                # Backfill manual/external trades the bot never placed so the
                # journal reflects all account activity (Phase 3 observability).
                _reconcile_external_deals()

        except Exception as e:
            logging.warning(f"positions_get failed: {e}")
            positions_valid = False
            all_positions = []
            total_positions = 0
            active_tickets_set = set()

        block_entries = not positions_valid
        if not positions_valid:
            logging.warning("positions_get returned None — book unreadable, blocking new entries this cycle")
            _st._cycle_consecutive_failures += 1
        if not check_daily_loss(cfg):
            logging.warning("Daily loss limit reached. Blocking new entries.")
            block_entries = True
        if _st._circuit_breaker_triggered:
            logging.warning("CIRCUIT BREAKER ACTIVE: all new entries blocked.")
            block_entries = True
            # Re-fetch positions if the cycle-level fetch failed — circuit
            # breaker MUST NOT exit with open positions (agent audit C2).
            cb_positions = all_positions
            if not cb_positions:
                cb_positions = mt5_call(mt5.positions_get, _timeout=10) or []
            if cb_positions:
                for pos in cb_positions:
                    sym = pos.symbol
                    try:
                        tick = mt5_call(mt5.symbol_info_tick, sym, _timeout=5)
                        if tick is None:
                            continue
                        sinfo_cb = mt5_call(mt5.symbol_info, sym, _timeout=5)
                        close_price = tick.bid if pos.type == 0 else tick.ask
                        close_req = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": sym,
                            "volume": pos.volume,
                            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                            "position": pos.ticket,
                            "price": close_price,
                            "deviation": 100,
                            "magic": cfg.get("magic", 20240706),
                            "type_time": mt5.ORDER_TIME_GTC,
                            "type_filling": get_filling_mode(sym),
                        }
                        result_cb = mt5_order_send(close_req, _timeout=10)
                        if result_cb is not None and result_cb.retcode == mt5.TRADE_RETCODE_DONE:
                            pnl_cb = result_cb.profit if hasattr(result_cb, "profit") else 0.0
                            pips_cb = (
                                abs(pos.price_open - close_price) / sinfo_cb.point
                                if (sinfo_cb and sinfo_cb.point)
                                else 0
                            )
                            journal_close(pos.ticket, close_price, pnl_cb, pips_cb, "CIRCUIT_BREAKER")
                            _scale_out_state.pop(pos.ticket, None)
                            _chandelier_state.pop(pos.ticket, None)
                            _exec_bias.pop(sym, None)
                    except Exception as e:
                        logging.warning(f"[{sym}] Circuit breaker close failed: {e}")
            save_bot_state()
            logging.critical("CIRCUIT BREAKER TRIGGERED — halting bot")
            mt5_call(mt5.shutdown, _timeout=5)
            os._exit(1)

        try:
            if cfg["corr_enabled"]:
                now = time.time()
                if now - _st._last_corr_time > 3600:
                    _st._corr_cache = compute_correlation_matrix(cfg["symbols"], cfg.get("corr_lookback_hours", 24))
                    _st._last_corr_time = now

            # Portfolio risk budget: compute open position risk
            portfolio_risk_pct = cfg.get("portfolio_risk_pct", 3.0)
            portfolio_risk_budget_active = portfolio_risk_pct > 0
            balance_pkr = 0.0
            total_risk_pkr = 0.0
            if portfolio_risk_budget_active:
                acc_tmp = mt5_call(mt5.account_info, _timeout=5)
                if acc_tmp is None:
                    logging.warning("Cannot compute risk budget: mt5.account_info() returned None")
                    balance_pkr = 0.0
                else:
                    balance_pkr = acc_tmp.balance
                for p in all_positions:
                    sinfo_p = mt5_call(mt5.symbol_info, p.symbol, _timeout=5)
                    if sinfo_p and sinfo_p.point and p.sl:
                        sl_dist = abs(p.price_open - p.sl)
                        p_risk = (
                            sl_dist / sinfo_p.trade_tick_size * sinfo_p.trade_tick_value * p.volume
                            if sinfo_p.trade_tick_size > 0
                            else 0.0
                        )
                        total_risk_pkr += p_risk
                # Hard portfolio-risk guard: if TOTAL open risk (including any
                # manual/external positions the bot didn't size) exceeds the cap,
                # block new entries. A single 0.28-lot external BTCUSD position
                # previously carried ~47% account risk with zero bot visibility.
                if portfolio_risk_budget_active and balance_pkr > 0:
                    open_risk_pct = total_risk_pkr / balance_pkr * 100
                    cap_pct = cfg.get("max_open_risk_pct", 5.0)
                    if open_risk_pct > cap_pct:
                        logging.warning(
                            f"Open risk {open_risk_pct:.2f}% (Rs.{total_risk_pkr:.2f}) "
                            f"exceeds cap {cap_pct:.2f}% — blocking new entries"
                        )
                        block_entries = True

            now_ts = time.time()
            if now_ts - _st._last_symbol_reselect_time > 1800:
                _st._last_symbol_reselect_time = now_ts
                for sym in cfg["symbols"]:
                    sinfo = mt5_call(mt5.symbol_info, sym, _timeout=5)
                    if sinfo is None:
                        mt5_call(mt5.symbol_select, sym, True, _timeout=10)
                        time.sleep(0.1)
                        sinfo = mt5_call(mt5.symbol_info, sym, _timeout=5)
                    was_online = _st._symbol_online.get(sym, True)
                    is_online = sinfo is not None and sinfo.trade_mode in (
                        mt5.SYMBOL_TRADE_MODE_FULL,
                        mt5.SYMBOL_TRADE_MODE_CLOSEONLY,
                    )
                    if was_online != is_online:
                        _st._symbol_online[sym] = is_online
                        if is_online:
                            logging.info(f"[{sym}] Market opened — resumed monitoring")
                        else:
                            logging.warning(f"[{sym}] Market closed or symbol dropped — paused")

            try:
                filled_limits = check_limit_orders(cfg)
            except Exception:
                logging.exception("check_limit_orders failed")
                filled_limits = set()

            if filled_limits:
                try:
                    refreshed = mt5_call(mt5.positions_get, _timeout=10)
                    if refreshed is not None:
                        all_positions = refreshed
                        total_positions = len(all_positions)
                        active_tickets_set = {p.ticket for p in all_positions}
                except Exception:
                    logging.warning("positions_get refresh after limit fill failed")

            # Initialize scale-out/chandelier state for limit-filled positions
            # immediately, so they have protection on the same cycle they fill
            # (previously state was only initialized on the next cycle when the
            # position loop ran, leaving a one-cycle gap with no scale-out or
            # chandelier exit — agent audit M7).
            if filled_limits:
                for sym in filled_limits:
                    try:
                        sym_cfg = deepcopy(cfg)
                        sym_cfg["symbol"] = sym
                        apply_symbol_strategy(sym_cfg, sym)
                        apply_symbol_overrides(sym_cfg, sym)
                        sinfo = mt5_call(mt5.symbol_info, sym, _timeout=5)
                        if sinfo is None:
                            continue
                        pos = mt5_call(mt5.positions_get, symbol=sym, _timeout=5)
                        if not pos:
                            continue
                        pos = pos[0]
                        if pos.ticket in _scale_out_state:
                            continue
                        sl_dist = abs(pos.price_open - pos.sl) if pos.sl else 0
                        if sl_dist > 0:
                            sl_pts = int(sl_dist / sinfo.point) if sinfo.point else 50
                            atr_reconstructed = sl_dist / sym_cfg.get("atr_sl_mult", 1.0) if sinfo.point else 0
                            from execution import _init_scale_out_state

                            _scale_out_state[pos.ticket] = _init_scale_out_state(
                                sym_cfg,
                                pos.price_open,
                                "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell",
                                sl_pts,
                                sinfo,
                                is_mr=False,
                                volume=pos.volume,
                                atr_entry=atr_reconstructed,
                            )
                        if pos.ticket not in _chandelier_state and pos.sl:
                            _chandelier_state[pos.ticket] = {"ch_sl": pos.sl}
                        save_bot_state()
                    except Exception:
                        logging.warning(f"[{sym}] Limit fill state init failed", exc_info=True)

            for symbol in cfg["symbols"]:
                try:
                    if not _st._symbol_online.get(symbol, True):
                        continue
                    if not market_open(symbol):
                        continue
                    sym_cfg = deepcopy(cfg)
                    sym_cfg["symbol"] = symbol
                    apply_symbol_strategy(sym_cfg, symbol)
                    apply_symbol_overrides(sym_cfg, symbol)
                    positions_sym = [p for p in all_positions if p.symbol == symbol]

                    if symbol not in _filter_stats:
                        _filter_stats[symbol] = {
                            "htf_trend": 0,
                            "tail_risk": 0,
                            "no_signal": 0,
                            "signals": 0,
                            "regime_gate": 0,
                            "ml_gate": 0,
                            "sanity": 0,
                        }

                    if symbol in _st._pending_limits:
                        logging.debug(f"[{symbol}] Pending limit active — skipping entry")
                        _filter_stats[symbol]["no_signal"] += 1
                        continue

                    # === Gate 1 — Fused Regime ===
                    # Computed on the last CLOSED bar via the shared analytics
                    # module (single source of truth with the backtest). This is
                    # what keeps the gate from flickering intrabar and matches
                    # backtest bar i as closed (agent audit M1).
                    gate = _get_regime_gate(symbol, sym_cfg)
                    G1_BARS = 100
                    df_g1 = get_rates(symbol, sym_cfg["timeframe"], G1_BARS)
                    fused_score = fused_regime_score(
                        df_g1.iloc[:-1] if df_g1 is not None and len(df_g1) > 1 else df_g1, sym_cfg
                    )
                    gate_open = gate.update(fused_score)
                    logging.debug(
                        f"[{symbol}] Fused gate score={fused_score:.1f} open={gate_open} "
                        f"(bars={len(df_g1) if df_g1 is not None else 0})"
                    )

                    trend_signal = None
                    mr_signal = None
                    trend_atr = None
                    mr_atr = None
                    entry_type = None
                    mtf_confidence = None

                    # === Gate 2 — MTF Signal ===
                    if gate_open:
                        # Trend-favorable regime — use MTF or single-TF crossover
                        if sym_cfg.get("mtf_enabled", False):
                            trend_signal, trend_atr, entry_type, mtf_confidence = get_mtf_fused_signal(sym_cfg)
                        else:
                            trend_signal, trend_atr, entry_type = get_signal(sym_cfg)
                    else:
                        # Chop regime — use MR or pullback
                        if sym_cfg["mr_enabled"] and len(positions_sym) == 0:
                            mr_signal, mr_atr = get_mean_reversion_signal(sym_cfg)
                        if mr_signal is None and sym_cfg.get("mtf_enabled", False):
                            trend_signal, trend_atr, entry_type, mtf_confidence = get_mtf_fused_signal(sym_cfg)
                        elif mr_signal is None:
                            trend_signal, trend_atr, entry_type = get_signal(sym_cfg)

                    atr = trend_atr or mr_atr or get_current_atr(sym_cfg)

                    signal_entry = mr_signal if (mr_signal is not None and not gate_open) else trend_signal
                    entry_atr = mr_atr if (mr_signal is not None and not gate_open) else trend_atr
                    if entry_atr is None:
                        entry_atr = atr
                    regime = "ranging" if not gate_open else "trending"
                    regimes[symbol] = regime

                    if len(positions_sym) > 0:
                        for pos in positions_sym:
                            if pos.ticket in _scale_out_state and (pos.tp is None or pos.tp == 0.0):
                                so_state = _scale_out_state[pos.ticket]
                                sl_pts = so_state.get("sl_points")
                                so_point = so_state.get("point")
                                so_rr = so_state.get("rr")
                                if sl_pts and so_point and so_rr and pos.sl:
                                    desired_tp_dist = sl_pts * so_point * so_rr
                                    new_tp = (
                                        pos.price_open + desired_tp_dist
                                        if pos.type == mt5.ORDER_TYPE_BUY
                                        else pos.price_open - desired_tp_dist
                                    )
                                    modify_req = {
                                        "action": mt5.TRADE_ACTION_SLTP,
                                        "position": pos.ticket,
                                        "sl": pos.sl,
                                        "tp": new_tp,
                                    }
                                    mt5_order_send(modify_req, _timeout=10)
                        if atr is not None:
                            for pos in positions_sym:
                                check_breakeven(sym_cfg, pos, atr)
                        if atr is not None:
                            for pos in positions_sym:
                                check_chandelier_exit(sym_cfg, pos)
                        if sym_cfg["scale_out_enabled"]:
                            for pos in positions_sym:
                                if pos.ticket in _scale_out_state:
                                    check_scale_out(sym_cfg, pos)
                        for pos in positions_sym:
                            pos_type = "buy" if pos.type == mt5.ORDER_TYPE_BUY else "sell"
                            should_close = False
                            close_reason = ""
                            if check_max_hold(sym_cfg, pos):
                                should_close = True
                                close_reason = "MAX_HOLD"
                            is_mr_pos = (
                                getattr(pos, "magic", None) == sym_cfg.get("mr_magic", 20240707)
                            ) or (hasattr(pos, "comment") and pos.comment == "TrendBot-MR")
                            if (
                                not should_close
                                and (is_mr_pos or (regime == "ranging" and sym_cfg["mr_enabled"]))
                                and check_mean_reversion_exit(sym_cfg, pos)
                            ):
                                should_close = True
                                close_reason = "MR_EXIT"
                            if (
                                not should_close
                                and trend_signal is not None
                                and (
                                    (pos_type == "buy" and trend_signal == "sell")
                                    or (pos_type == "sell" and trend_signal == "buy")
                                )
                            ):
                                cur_atr = get_current_atr(sym_cfg)
                                in_sub_profit = False
                                if cur_atr and cur_atr > 0:
                                    in_sub_profit = (
                                        pos_type == "buy" and pos.price_current > pos.price_open + cur_atr * 0.25
                                    ) or (pos_type == "sell" and pos.price_current < pos.price_open - cur_atr * 0.25)
                                if not in_sub_profit:
                                    should_close = True
                                    close_reason = "REVERSAL"
                            if should_close:
                                close_type = (
                                    mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                                )
                                tick = mt5_call(mt5.symbol_info_tick, symbol, _timeout=5)
                                if tick:
                                    price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                                    close_req = {
                                        "action": mt5.TRADE_ACTION_DEAL,
                                        "symbol": symbol,
                                        "volume": pos.volume,
                                        "type": close_type,
                                        "position": pos.ticket,
                                        "price": price,
                                        "deviation": get_deviation(sym_cfg, symbol),
                                        "magic": sym_cfg.get("magic", 20240706),
                                        "comment": f"TrendBot-{close_reason}",
                                        "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": get_filling_mode(symbol),
                                    }
                                    result = mt5_order_send(close_req, _timeout=10)
                                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        _update_dynamic_deviation(symbol, True, sym_cfg)
                                        _last_trade_time.pop(f"trend:{symbol}", None)
                                        _last_trade_time.pop(f"mr:{symbol}", None)
                                        close_pnl = (
                                            result.profit if hasattr(result, "profit") and result.profit else 0.0
                                        )
                                        if sym_cfg["trade_journal"]:
                                            sinfo2 = mt5_call(mt5.symbol_info, symbol, _timeout=5)
                                            sp = sinfo2.point if (sinfo2 and sinfo2.point) else 0.001
                                            pips = abs(pos.price_open - price) / sp
                                            close_pnl = (
                                                result.profit if hasattr(result, "profit") and result.profit else 0.0
                                            )
                                            journal_close(pos.ticket, price, close_pnl, pips, close_reason)
                                            if close_reason == "MR_EXIT":
                                                if close_pnl > 0:
                                                    _st._mr_consecutive_losses[symbol] = 0
                                                else:
                                                    _st._mr_consecutive_losses[symbol] = (
                                                        _st._mr_consecutive_losses.get(symbol, 0) + 1
                                                    )
                                                    _st._mr_last_loss_time[symbol] = time.time()
                                        if sym_cfg["discord_url"]:
                                            sinfo2 = mt5_call(mt5.symbol_info, symbol, _timeout=5)
                                            sp = sinfo2.point if (sinfo2 and sinfo2.point) else 0.001
                                            pips2 = abs(pos.price_open - price) / sp
                                            from discord_alerts import trade_close

                                            trade_close(
                                                sym_cfg["discord_url"],
                                                symbol,
                                                pos_type,
                                                pos.volume,
                                                pos.price_open,
                                                price,
                                                close_pnl,
                                                pips2,
                                                close_reason,
                                            )
                                        _chandelier_state.pop(pos.ticket, None)
                                        _scale_out_state.pop(pos.ticket, None)
                                        _exec_bias.pop(symbol, None)
                                        save_bot_state()
                                        # Keep in-memory list fresh so same-cycle
                                        # reversal is not blocked by stale counts
                                        all_positions = [p for p in all_positions if p.ticket != pos.ticket]
                                        logging.info(
                                            f"[{symbol}] Close ({close_reason}) failed: "
                                            f"{result.retcode if result else 'None'}"
                                        )
                                time.sleep(2)
                                break

                    # Re-filter from updated in-memory list after close
                    positions_sym = [p for p in all_positions if p.symbol == symbol]
                    total_positions = len(all_positions)

                    if block_entries:
                        logging.debug(f"[{symbol}] Block entries active — skipping")
                        _filter_stats[symbol]["no_signal"] += 1
                        continue

                    # === Risk Management (separate from filter chain) ===
                    entry_allowed = can_trade_symbol(symbol)
                    if entry_allowed:
                        if not check_capital_eligibility(sym_cfg, symbol):
                            logging.debug(f"[{symbol}] Capital eligibility failed — skipping")
                            _filter_stats[symbol]["no_signal"] += 1
                            continue
                        if total_positions >= sym_cfg["max_total_positions"]:
                            logging.debug(f"[{symbol}] Max total positions — skipping")
                            _filter_stats[symbol]["no_signal"] += 1
                            continue
                        max_per_sym = sym_cfg.get("max_positions_per_symbol", 1)
                        if len(positions_sym) >= max_per_sym:
                            logging.debug(f"[{symbol}] Max per-symbol — skipping")
                            _filter_stats[symbol]["no_signal"] += 1
                            continue
                        if not check_tail_risk(sym_cfg):
                            _filter_stats[symbol]["tail_risk"] = _filter_stats[symbol].get("tail_risk", 0) + 1
                            continue
                        if _st._circuit_breaker_triggered:
                            logging.info(f"[{symbol}] Circuit breaker active — skipping")
                            continue

                    if signal_entry is None:
                        if not gate_open:
                            _filter_stats[symbol]["regime_gate"] += 1
                        logging.debug(f"[{symbol}] No signal generated — skipping")
                        _filter_stats[symbol]["no_signal"] += 1
                        continue

                    if entry_atr is None or entry_atr == 0:
                        logging.debug(f"[{symbol}] ATR unavailable — skipping")
                        _filter_stats[symbol]["no_signal"] += 1
                        continue

                    _filter_stats[symbol]["signals"] += 1

                    htf_size_mult = 1.0
                    if not sym_cfg.get("mtf_enabled", False) or (
                        sym_cfg.get("mtf_enabled", False) and entry_type == "pullback"
                    ):
                        htf_decision, htf_size_mult = check_htf_trend(sym_cfg, signal_entry)
                        if htf_decision == "block":
                            logging.debug(f"[{symbol}] HTF clear counter-trend — blocking")
                            _filter_stats[symbol]["htf_trend"] += 1
                            continue
                        if htf_decision == "soft":
                            logging.debug(f"[{symbol}] HTF neutral — reduced size")
                            _filter_stats[symbol]["htf_trend"] = _filter_stats[symbol].get("htf_trend", 0) + 1

                    # === Gate 3 — ML Validation ===
                    ml_passed, confidence_mult, ml_conf = check_ml_gate(sym_cfg, signal_entry, entry_atr)
                    if not ml_passed:
                        logging.info(f"[{symbol}] ML gate failed — skipping")
                        _filter_stats[symbol]["ml_gate"] += 1
                        continue

                    # === Gate 4 — Execution Sanity (volume + spread + tape) ===
                    if not check_execution_sanity(sym_cfg, signal_entry):
                        logging.info(f"[{symbol}] Execution sanity failed — skipping")
                        _filter_stats[symbol]["sanity"] += 1
                        continue

                    is_mr_entry = mr_signal is not None and not gate_open
                    if (
                        is_mr_entry
                        and signal_entry is not None
                        and _st._mr_consecutive_losses.get(symbol, 0) >= 2
                        and time.time() - _st._mr_last_loss_time.get(symbol, 0) < 7200
                    ):
                        logging.debug(f"[{symbol}] MR consec losses cooldown — skipping")
                        _filter_stats[symbol]["no_signal"] += 1
                        continue

                    # Position sizing
                    kelly_mult = _apply_corr_ml_sizing(
                        sym_cfg,
                        symbol,
                        all_positions,
                        calc_kelly_mult(sym_cfg) * calc_volatility_mult(sym_cfg),
                        ml_conf,
                        confidence_mult,
                    )

                    # MTF confidence scale
                    if sym_cfg.get("mtf_enabled", False) and mtf_confidence is not None:
                        kelly_mult *= max(0.5, mtf_confidence)

                    if portfolio_risk_budget_active:
                        base_rsk = sym_cfg["risk_percent"]
                        new_risk_pkr = balance_pkr * (base_rsk / 100) * kelly_mult
                        total_if_added = total_risk_pkr + new_risk_pkr
                        budget_pkr = balance_pkr * (portfolio_risk_pct / 100)
                        if total_if_added > budget_pkr:
                            kelly_mult *= budget_pkr / max(total_if_added, 1)

                    min_mult = sym_cfg.get("dr_min_mult", 0.25)
                    max_mult = sym_cfg.get("dr_max_mult", 1.5)
                    kelly_mult = max(min_mult, min(max_mult, kelly_mult))

                    if is_mr_entry:
                        place_mean_reversion_trade(sym_cfg, signal_entry, entry_atr, kelly_mult)
                    else:
                        # Apply the HTF-misalignment size reduction (soft = reduced
                        # size). Previously htf_size_mult was computed and logged as
                        # "reduced size" but never actually applied, so soft signals
                        # traded full size — diverging from the backtest, which scales
                        # volume by htf_size_mult (backtest.py) (agent audit M5).
                        regime_mult = (1.0 if gate_open else 0.5) * htf_size_mult
                        use_limit = sym_cfg.get("le_enabled", True) and not is_mr_entry
                        if use_limit:
                            placed = place_limit_order(sym_cfg, signal_entry, entry_atr, kelly_mult, regime_mult)
                            if not placed:
                                logging.info(f"[{symbol}] Limit order failed — falling back to market")
                                place_trade(sym_cfg, signal_entry, entry_atr, regime_mult * kelly_mult)
                        else:
                            place_trade(sym_cfg, signal_entry, entry_atr, regime_mult * kelly_mult)
                    all_positions = mt5_call(mt5.positions_get, _timeout=10)
                    if all_positions is None:
                        all_positions = []
                    total_positions = len(all_positions)
                    positions_sym = [p for p in all_positions if p.symbol == symbol]

                except Exception as e:
                    logging.exception(f"[{symbol}] Error in main loop: {e}")

        except Exception as e:
            logging.exception(f"Error in main loop: {e}")
            _st._cycle_consecutive_failures += 1

        write_dashboard_state(all_positions, regimes)

        today = datetime.now().strftime("%Y-%m-%d")
        if _st._last_daily_summary_day != today:
            _st._last_daily_summary_day = today
            # Once-per-day reset of diagnostics (moved here from
            # check_daily_loss so that predicate stays side-effect free — M11).
            _st._filter_stats.clear()
            _st._exec_quality.clear()
            if cfg["discord_url"]:
                try:
                    acc = mt5_call(mt5.account_info, _timeout=5)
                    if acc:
                        trades_today = 0
                        win_rate = None
                        net_realized = 0.0
                        if TRADE_CSV.exists():
                            # The journal is append-only and time-ordered ascending,
                            # so today's closed rows live at the TAIL. Scan backwards
                            # instead of loading the whole file (agent audit: CSV opt).
                            rows = _read_todays_trades(TRADE_CSV, today)
                            trades_today = len(rows)
                            if rows:
                                pnls = [float(r["pnl"]) for r in rows if r.get("pnl")]
                                net_realized = sum(pnls)
                                wins = sum(1 for p in pnls if p > 0)
                                win_rate = wins / len(pnls) * 100 if pnls else None
                        # True account net for the day (includes external/manual).
                        day_net = net_realized
                        daily_summary(
                            cfg["discord_url"],
                            acc.balance,
                            acc.equity,
                            day_net,
                            len(all_positions),
                            win_rate,
                            trades_today,
                        )
                        logging.info(
                            f"Daily summary: balance={acc.balance:.2f} equity={acc.equity:.2f} "
                            f"trades={trades_today} win_rate={win_rate} journal_net={net_realized:.2f}"
                        )
                except Exception:
                    logging.exception("Daily summary failed")

        save_bot_state()
        _st._last_cycle_time = time.time()
        _st._cycle_consecutive_failures = 0
        time.sleep(cfg.get("cycle_sleep", 10))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception(f"Unhandled exception: {e}")
    finally:
        with contextlib.suppress(Exception):
            mt5_call(mt5.shutdown, _timeout=5)
