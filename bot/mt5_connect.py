"""MT5 connectivity, rate fetching, and symbol checks.

Linux VPS: mt5linux RPyC bridge (connects to MT5 running under Wine).
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from typing import Optional

import pandas as pd
import state as _st
from credentials import load_credentials
from state import _RATE_CACHE_TTL, _rate_cache

_mt5_instance: Optional[object] = None


def _init_mt5linux():
    """Connect to mt5linux RPyC bridge.

    The MT5 terminal is single-threaded: a long copy_rates_from call blocks
    all other RPyC requests behind it. The sync timeout must be long enough
    for those slow calls to complete (80k bars can take 30-60s). Quick calls
    (terminal_info, account_info) use a separate short-timeout connection
    for health checks, so a long sync timeout here does not delay detection
    of a dead server.
    """
    from mt5linux import MetaTrader5 as _MT5Client
    inst = _MT5Client(host="127.0.0.1", port=18812)
    inst._MetaTrader5__conn._config["sync_request_timeout"] = 120
    inst.initialize()
    return inst


def init_mt5():
    """Initialize the mt5linux connection."""
    global _mt5_instance
    if _mt5_instance is not None:
        return
    for attempt in range(5):
        try:
            logging.info("MT5 backend: trying mt5linux...")
            _mt5_instance = _init_mt5linux()
            logging.info("MT5 backend: mt5linux (RPyC)")
            return
        except Exception as e:
            if attempt < 4:
                logging.info(f"mt5linux not ready ({e}), retrying in 5s...")
                time.sleep(5)
            else:
                raise RuntimeError(f"mt5linux failed after 5 attempts: {e}")


def login_account():
    """Initialize MT5 with the configured credentials and log in.

    Uses credentials.load_credentials() — the repo's single settings+credentials
    reader (scenario_analysis.py and tune_scaleout.py route through here too).
    The live bot does NOT call this: the mt5linux RPyC bridge session is already
    authenticated on the terminal. Returns True on success, False otherwise
    (errors are logged).
    """
    try:
        creds = load_credentials()
    except RuntimeError as e:
        logging.error(str(e))
        return False
    ok = mt5_call(mt5.initialize, path=creds["path"], timeout=creds["timeout"], _timeout=65)
    if not ok:
        logging.error(f"MT5 init failed: {mt5_call(mt5.last_error, _timeout=3)}")
        return False
    authorized = mt5_call(
        mt5.login, login=creds["account"], password=creds["password"],
        server=creds["server"], _timeout=30,
    )
    if not authorized:
        logging.error(f"MT5 login failed: {mt5_call(mt5.last_error, _timeout=3)}")
        return False
    return True


class _MT5Proxy:
    """Proxy that delegates attribute access to the live MT5 instance.

    main.py does `from mt5_connect import mt5`, which binds to this proxy.
    The proxy always resolves attributes on the current `_mt5_instance`, so
    callers transparently use the fresh connection after a reconnect instead
    of a stale snapshot. When no instance exists yet (auto-init pending or
    failed), attribute access raises AttributeError.
    """

    def __getattr__(self, name):
        inst = _mt5_instance
        if inst is not None:
            return getattr(inst, name)
        raise AttributeError(f"MT5 not initialized - '{name}' unavailable")

    def __repr__(self):
        inst = _mt5_instance
        return f"<MT5Proxy ({inst.__class__.__name__} if inst else 'no instance')>"


mt5 = _MT5Proxy()

# Auto-init on import (may fail gracefully if the server isn't up yet).
# Tests rely on _mt5_instance being set so they can patch proxy attributes.
try:
    init_mt5()
except Exception as e:
    logging.warning(f"MT5 auto-init failed: {e}")
    logging.warning("MT5 will be unavailable until init_mt5() succeeds")

_THREAD_BOUND = {"initialize", "shutdown", "login", "order_send",
                 "copy_rates_from", "copy_rates_from_pos", "copy_ticks_from", "copy_ticks_range"}

_call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5call")
_executor_lock = threading.Lock()


def _reset_executor():
    global _call_executor
    with _executor_lock:
        old = _call_executor
        _call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5call")
    old.shutdown(wait=False, cancel_futures=True)


def _recreate_instance():
    """Replace _mt5_instance with a fresh RPyC connection."""
    global _mt5_instance, mt5
    with _executor_lock:
        old = _mt5_instance
        try:
            inst = _init_mt5linux()
        except Exception as e:
            logging.warning(f"MT5 instance recreation failed: {e}")
            return False
        _mt5_instance = inst
        mt5 = inst
        if old is not None:
            with suppress(Exception):
                old.shutdown()
        logging.info("MT5 instance recreated (fresh RPyC connection)")
        return True


def mt5_order_send(req, _timeout=None):
    """Send a trade request; single home for order_send (was duplicated in
    execution.py + main.py).

    The old frame-sensitivity warning ("must be called from the placing module's
    frame or it returns None") predates the RPyC bridge and is folklore — the
    `_MT5Proxy` resolves attributes at call time, so a shared helper is safe.
    The REAL quirk is that MT5 mutates the request dict in place: callers must
    build a fresh request dict per send (see execution.py `_place_trade_inner`).
    """
    return mt5.order_send(req)


def mt5_call(func, *args, _timeout=None, **kwargs):
    name = getattr(func, "__name__", "")
    # The module-level `mt5` is a _MT5Proxy whose __getattr__ resolves on the
    # live _mt5_instance at attribute-access time, so bound methods obtained
    # via `mt5.some_method` always hit the current connection.
    # Thread-bound calls (slow data fetches, order_send) run synchronously.
    # All other calls get a bounded client-side timeout via the executor.
    if name in _THREAD_BOUND:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"MT5 {name or func} raised: {e}")
            _recreate_instance()
            return None
    effective_timeout = _timeout if _timeout is not None else 30
    try:
        with _executor_lock:
            executor = _call_executor
        future = executor.submit(func, *args, **kwargs)
    except RuntimeError:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"MT5 {name or func} raised: {e}")
            _recreate_instance()
            return None
    try:
        return future.result(timeout=effective_timeout)
    except FutureTimeout:
        logging.warning(f"MT5 {name or func} timed out after {effective_timeout}s - recreating")
        _reset_executor()
        _recreate_instance()
        return None
    except Exception as e:
        logging.warning(f"MT5 {name or func} raised: {e}")
        _recreate_instance()
        return None


def get_rates(symbol, timeframe, bars):
    key = (symbol, timeframe)
    if key in _rate_cache:
        cached, cached_count, cached_ts = _rate_cache[key]
        if len(cached) >= bars and time.time() - cached_ts < _RATE_CACHE_TTL:
            return cached
    rates = mt5_call(mt5.copy_rates_from_pos, symbol, timeframe, 0, bars)
    if rates is None or len(rates) < bars:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    _rate_cache[key] = (df, bars, time.time())
    return df


def fetch_rates_paged(symbol, timeframe, start, end, chunk_bars=80000):
    with suppress(Exception):
        mt5_call(mt5.symbol_select, symbol, True, _timeout=10)
    start_ts = pd.Timestamp(start)
    frames = []
    cursor = end
    prev_oldest = None
    while True:
        rates = mt5_call(mt5.copy_rates_from, symbol, timeframe, cursor, chunk_bars)
        if rates is None or len(rates) == 0:
            break
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        frames.append(df)
        oldest = df["time"].iloc[0]
        if oldest <= start_ts:
            break
        if prev_oldest is not None and oldest >= prev_oldest:
            break
        prev_oldest = oldest
        cursor = oldest.to_pydatetime()
    if not frames:
        return None
    out = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["time"])
        .sort_values("time")
        .reset_index(drop=True)
    )
    out = out[out["time"] >= start_ts].reset_index(drop=True)
    return out if len(out) > 0 else None


def get_filling_mode(symbol, default=None):
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    try:
        if sinfo and hasattr(sinfo, "filling_mode") and sinfo.filling_mode and sinfo.filling_mode > 0:
            if sinfo.filling_mode & mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            if sinfo.filling_mode & mt5.ORDER_FILLING_FOK:
                return 0
    except (TypeError, AttributeError):
        pass
    return default or mt5.ORDER_FILLING_IOC


def get_deviation(cfg, symbol):
    sym_dev = cfg["symbol_strategy"].get(symbol, {}).get("deviation")
    base = int(sym_dev) if sym_dev is not None else cfg.get("deviation", 50)
    return _st._dynamic_deviation.get(symbol, base)


def _update_dynamic_deviation(symbol, succeeded, cfg):
    sym_dev = cfg["symbol_strategy"].get(symbol, {}).get("deviation")
    base = int(sym_dev) if sym_dev is not None else cfg.get("deviation", 50)
    cur = _st._dynamic_deviation.get(symbol, base)
    max_dev = base * 3
    new_dev = max(base, int(cur * 0.9)) if succeeded else min(max_dev, int(cur * 1.5))
    if new_dev != cur:
        _st._dynamic_deviation[symbol] = new_dev
        logging.info(f"[{symbol}] Deviation adjusted: {cur} -> {new_dev}")
    return new_dev


def _is_crypto_symbol(symbol):
    return _st.ASSET_CLASS_MAP.get(symbol) == "crypto"


def market_open(symbol):
    if _is_crypto_symbol(symbol):
        return True
    if not _st._symbol_online.get(symbol, True):
        return False
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        mt5_call(mt5.symbol_select, symbol, True, _timeout=10)
        time.sleep(0.1)
        sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
        if sinfo is None:
            return False
    return sinfo.trade_mode in (mt5.SYMBOL_TRADE_MODE_FULL, mt5.SYMBOL_TRADE_MODE_CLOSEONLY)


def can_trade_symbol(symbol):
    if _is_crypto_symbol(symbol):
        return True
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    if sinfo is None:
        return False
    return sinfo.trade_mode == mt5.SYMBOL_TRADE_MODE_FULL


_PING_BUSY = object()


def _mt5linux_ping():
    """Lightweight health check with a fresh connection each call."""
    try:
        from mt5linux import MetaTrader5 as _MT5Client
        inst = _MT5Client(host="127.0.0.1", port=18812)
        inst._MetaTrader5__conn._config["sync_request_timeout"] = 10
        inst.initialize()
        info = inst.terminal_info()
        with suppress(Exception):
            inst.shutdown()
        return info
    except Exception as e:
        err = str(e).lower()
        if "timed out" in err or "timeout" in err or "asyncresulttimeout" in err:
            return _PING_BUSY
        return None


def _ensure_mt5linux_connected(cfg):
    """Reconnect path for mt5linux backend (systemd owns the terminal)."""
    global _mt5_instance, mt5
    for attempt in range(5):
        try:
            inst = _init_mt5linux()
        except Exception as e:
            logging.warning(f"mt5linux connect raised (attempt {attempt+1}): {e}")
            time.sleep(3)
            continue
        try:
            acc = inst.account_info()
        except Exception:
            acc = None
        if acc is not None:
            _mt5_instance = inst
            mt5 = inst
            logging.info(f"mt5linux reconnected: Balance Rs.{acc.balance:.2f}")
            for sym in cfg["symbols"]:
                with suppress(Exception):
                    inst.symbol_select(sym, True)
            return True
        logging.warning(f"mt5linux connect ok but no account (attempt {attempt+1}/5)")
        time.sleep(3)
    logging.error("mt5linux reconnection failed after 5 attempts")
    return False


def ensure_mt5_connected(cfg):
    """Ensure the mt5linux connection is alive, reconnect if needed."""
    global _mt5_instance
    if _mt5_instance is None:
        logging.info("MT5 not initialized - attempting init_mt5()")
        try:
            init_mt5()
        except Exception as e:
            logging.warning(f"init_mt5() failed: {e}")
            return False
    # Verify the main instance works (not just a ping). The ping uses a fresh
    # connection; the main _mt5_instance may be stale. A quick terminal_info
    # call (with a bounded timeout) confirms the live instance is healthy.
    try:
        info = mt5_call(_mt5_instance.terminal_info, _timeout=10)
    except Exception:
        info = None
    if info is not None and getattr(info, "connected", True):
        return True
    if _mt5linux_ping() is _PING_BUSY:
        logging.debug("mt5linux busy (server blocked by long call) - skipping cycle")
        return True
    logging.warning("mt5linux disconnected. Attempting reconnection...")
    return _ensure_mt5linux_connected(cfg)
