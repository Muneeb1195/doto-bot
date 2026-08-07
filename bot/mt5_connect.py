"""MT5 connectivity, rate fetching, and symbol checks.

Auto-detects the MT5 backend:
 - Windows dev: native MetaTrader5 C extension
 - Linux VPS:   mt5linux RPyC bridge (connects to MT5 running under Wine)
"""

import logging
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import suppress
from typing import Optional

import pandas as pd
import state as _st
from state import _RATE_CACHE_TTL, _rate_cache

_is_linux = platform.system() == "Linux"

_mt5_instance: Optional[object] = None
_mt5_backend: str = "unknown"


def _init_mt5linux():
    """Try mt5linux RPyC bridge (mt5linux v0.x without Docker)."""
    from mt5linux import MetaTrader5 as _MT5Client
    inst = _MT5Client(host="127.0.0.1", port=18812)
    inst.initialize()
    return inst


def _init_socket_client():
    """Fallback: MQL5 socket server (works when Wine IPC is broken)."""
    from mt5_socket_client import MT5SocketClient
    inst = MT5SocketClient(host="127.0.0.1", port=9000)
    inst.connect()
    return inst


def _get_mt5():
    global _mt5_instance
    if _mt5_instance is not None:
        return _mt5_instance
    raise RuntimeError("MT5 not initialized — call init_mt5() first")


def init_mt5():
    """Initialize MT5 connection with fallback chain."""
    global _mt5_instance, _mt5_backend
    if _mt5_instance is not None:
        return

    if not _is_linux:
        import MetaTrader5 as _mt5_impl
        _mt5_instance = _mt5_impl
        _mt5_backend = "native"
        logging.info("MT5 backend: native (Windows)")
        return

    # Linux: try mt5linux first, then socket client
    try:
        logging.info("MT5 backend: trying mt5linux...")
        _mt5_instance = _init_mt5linux()
        _mt5_backend = "mt5linux"
        logging.info("MT5 backend: mt5linux (RPyC)")
        return
    except Exception as e:
        logging.warning(f"mt5linux failed ({e}), trying socket client...")

    try:
        _mt5_instance = _init_socket_client()
        _mt5_backend = "socket"
        logging.info("MT5 backend: socket (MQL5 EA)")
    except Exception as e:
        raise RuntimeError(f"All MT5 backends failed. Last error: {e}")


# Legacy compatibility: auto-init on import (may fail gracefully)
try:
    init_mt5()
    mt5 = _mt5_instance
except Exception as e:
    logging.warning(f"MT5 auto-init failed: {e}")
    logging.warning("MT5 will be unavailable until init_mt5() succeeds")
    logging.warning("Using fallback MT5 constants module (no trading)")
    from mt5_socket_client import MT5SocketClient
    mt5 = MT5SocketClient.__new__(MT5SocketClient)
    mt5.host = "127.0.0.1"
    mt5.port = 9000
    mt5.timeout = 30
    mt5._sock = None
    mt5._buf = b""

_mt5_proc: Optional["subprocess.Popen"] = None

_THREAD_BOUND = {"initialize", "shutdown", "login", "order_send"}

_call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5call")
_executor_lock = threading.Lock()


def _reset_executor():
    global _call_executor
    with _executor_lock:
        old = _call_executor
        _call_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt5call")
    old.shutdown(wait=False)


def mt5_call(func, *args, _timeout=None, **kwargs):
    name = getattr(func, "__name__", "")
    if _timeout is None or name in _THREAD_BOUND:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"MT5 {name or func} raised: {e}")
            return None
    try:
        with _executor_lock:
            executor = _call_executor
        future = executor.submit(func, *args, **kwargs)
    except RuntimeError:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.warning(f"MT5 {name or func} raised: {e}")
            return None
    try:
        return future.result(timeout=_timeout)
    except FutureTimeout:
        logging.warning(f"MT5 {name or func} timed out after {_timeout}s — abandoning call")
        _reset_executor()
        return None
    except Exception as e:
        logging.warning(f"MT5 {name or func} raised: {e}")
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
        if oldest <= start_ts or len(df) < chunk_bars:
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


def get_positions(symbol):
    positions = mt5_call(mt5.positions_get, symbol=symbol, _timeout=5)
    return positions if positions else []


def get_filling_mode(symbol, default=None):
    sinfo = mt5_call(mt5.symbol_info, symbol, _timeout=5)
    try:
        if sinfo and hasattr(sinfo, "filling_mode") and sinfo.filling_mode and sinfo.filling_mode > 0:
            if sinfo.filling_mode & mt5.ORDER_FILLING_IOC:
                return mt5.ORDER_FILLING_IOC
            if sinfo.filling_mode & mt5.ORDER_FILLING_FOK:
                return 0
    except (TypeError, AttributeError):
        # sinfo.filling_mode is not an int (e.g. MagicMock in tests)
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
        logging.info(f"[{symbol}] Deviation adjusted: {cur} -> {new_dev} ({'success' if succeeded else 'rejection'})")
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


def _kill_process(name):
    if _is_linux:
        subprocess.run(["pkill", "-f", name], capture_output=True, timeout=5)
    else:
        subprocess.run(["taskkill", "/F", "/IM", name], capture_output=True, timeout=5)


def _ensure_socket_connected(cfg):
    """Reconnect path for the socket backend (MQL5 EA inside the terminal).

    systemd owns the MT5 terminal here, so we must never kill or relaunch it.
    We only re-establish the TCP session and re-send INIT, which the EA
    answers idempotently.
    """
    attempts = int(cfg.get("mt5_socket_attempts", 5))
    for attempt in range(attempts):
        try:
            ok = mt5_call(
                mt5.initialize,
                login=cfg.get("account"),
                password=cfg.get("password"),
                server=cfg.get("server"),
                timeout=60000,
                _timeout=65,
            )
        except Exception as e:
            logging.warning(f"MT5 socket initialize raised: {e}")
            ok = False

        if ok:
            acc = mt5_call(mt5.account_info, _timeout=10)
            if acc is not None:
                if attempt > 0:
                    logging.info(f"MT5 reconnected: {acc.name} | Balance: Rs.{acc.balance:.2f}")
                for sym in cfg["symbols"]:
                    mt5_call(mt5.symbol_select, sym, True, _timeout=10)
                return True
            logging.warning("MT5 socket INIT ok but account_info unavailable")

        err = mt5_call(mt5.last_error, _timeout=3)
        logging.warning(f"MT5 socket init failed (attempt {attempt + 1}/{attempts}): {err}")

        with suppress(Exception):
            mt5.disconnect()
        time.sleep(int(cfg.get("mt5_socket_retry_sleep", 10)))
        with suppress(Exception):
            mt5.connect()

    logging.error("MT5 socket bridge unavailable — terminal/EA not ready (systemd owns the terminal)")
    return False


def ensure_mt5_connected(cfg):
    global _mt5_proc
    if _mt5_backend == "socket":
        return _ensure_socket_connected(cfg)
    try:
        info = mt5_call(mt5.terminal_info, _timeout=5)
        if info is not None and getattr(info, "connected", True):
            return True
    except Exception:
        logging.debug("MT5 check failed — initiating reconnection")

    logging.warning("MT5 disconnected. Attempting reconnection...")
    try:
        mt5_call(mt5.shutdown, _timeout=5)
    except Exception:
        logging.debug("MT5 shutdown during reconnect failed (expected)")
    time.sleep(3)

    terminal = cfg["mt5_path"]
    if not terminal or not os.path.exists(terminal):
        if _is_linux:
            terminal = os.path.expanduser(
                "~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe"
            )
        else:
            terminal = r"C:\Program Files\MetaTrader 5\terminal64.exe"

    for attempt in range(5):
        try:
            if _mt5_proc is not None and _mt5_proc.pid:
                if _is_linux:
                    subprocess.run(["kill", "-9", str(_mt5_proc.pid)], capture_output=True, timeout=5)
                else:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(_mt5_proc.pid)], capture_output=True, timeout=5
                    )
            else:
                _kill_process("terminal64.exe")
        except subprocess.TimeoutExpired:
            logging.warning("kill terminal64.exe timed out")
        except Exception:
            logging.debug("kill terminal64.exe failed")
        _kill_process("metaeditor64.exe")

        try:
            real_terminal = terminal if os.path.exists(terminal) else (
                os.path.expanduser("~/.wine/drive_c/Program Files/MetaTrader 5/terminal64.exe")
                if _is_linux
                else r"C:\Program Files\MetaTrader 5\terminal64.exe"
            )
            if os.path.exists(real_terminal):
                logging.info(f"Launching MT5 terminal: {real_terminal}")
                if _mt5_proc is not None:
                    try:
                        _mt5_proc.terminate()
                        _mt5_proc.wait(timeout=5)
                    except Exception:
                        logging.debug("Failed to terminate previous MT5 process", exc_info=True)
                launch = ["wine", real_terminal] if _is_linux else [real_terminal]
                _mt5_proc = subprocess.Popen(launch, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logging.warning("MT5 terminal not found at %s; relying on mt5.initialize()", real_terminal)
        except Exception as e:
            logging.error(f"Failed to start MT5 terminal: {e}")
        time.sleep(2)

        wait = 5 + attempt * 10
        logging.info(f"Waiting {wait}s for MT5 terminal (attempt {attempt + 1}/5)...")
        time.sleep(wait)

        result = mt5_call(
            mt5.initialize,
            login=cfg.get("account"),
            password=cfg.get("password"),
            server=cfg.get("server"),
            timeout=60000,
            _timeout=65,
        )
        if result:
            acc = mt5_call(mt5.account_info, _timeout=5)
            if acc is not None:
                logging.info(f"MT5 reconnected: {acc.name} | Balance: Rs.{acc.balance:.2f}")
                for sym in cfg["symbols"]:
                    mt5_call(mt5.symbol_select, sym, True, _timeout=10)
                return True
            else:
                err = mt5_call(mt5.last_error, _timeout=3)
                logging.error(f"Login failed: {err}")
                mt5_call(mt5.shutdown, _timeout=5)
        else:
            err = mt5_call(mt5.last_error, _timeout=3)
            logging.error(f"MT5 init failed (attempt {attempt + 1}): {err}")
            mt5_call(mt5.shutdown, _timeout=5)

    logging.critical("MT5 reconnection failed after 5 attempts. Circuit breaker active.")
    return False
