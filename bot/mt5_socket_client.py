"""MT5 Socket Client — drop-in replacement for MetaTrader5 Python package.

Connects to the MQL5 Socket Server EA running inside the MT5 terminal.
Provides the same interface as the MetaTrader5 module so the bot works
without modification on Linux (where the native MT5 package can't connect
due to Wine IPC being broken).

Protocol: newline-delimited text over TCP port 9000.
Request:  "CMD arg1 arg2 ...\n"
Response: "OK key=val|key=val|...\n" or "ERR message\n"
Multi-line: "COUNT N\n" then N lines of "BAR ..." then "END\n"
"""

from __future__ import annotations

import socket
import struct
import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# --- constants (mirror MetaTrader5) ---
TIMEFRAME_M1 = 1
TIMEFRAME_M2 = 2
TIMEFRAME_M3 = 3
TIMEFRAME_M4 = 4
TIMEFRAME_M5 = 5
TIMEFRAME_M6 = 6
TIMEFRAME_M10 = 10
TIMEFRAME_M12 = 12
TIMEFRAME_M15 = 15
TIMEFRAME_M20 = 20
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 1 | 0x4000
TIMEFRAME_H2 = 2 | 0x4000
TIMEFRAME_H4 = 4 | 0x4000
TIMEFRAME_H3 = 3 | 0x4000
TIMEFRAME_H6 = 6 | 0x4000
TIMEFRAME_H8 = 8 | 0x4000
TIMEFRAME_H12 = 12 | 0x4000
TIMEFRAME_D1 = 24 | 0x4000
TIMEFRAME_W1 = 1 | 0x8000
TIMEFRAME_MN1 = 1 | 0xC000

ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5
ORDER_TYPE_BUY_STOP_LIMIT = 6
ORDER_TYPE_SELL_STOP_LIMIT = 7
ORDER_TYPE_CLOSE_BY = 8

ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2
ORDER_FILLING_BOC = 3

ORDER_TIME_GTC = 0
ORDER_TIME_DAY = 1
ORDER_TIME_SPECIFIED = 2
ORDER_TIME_SPECIFIED_DAY = 3

TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8
TRADE_ACTION_CLOSE_BY = 10

TRADE_RETCODE_REQUOTE = 10004
TRADE_RETCODE_REJECT = 10006
TRADE_RETCODE_CANCEL = 10007
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_DONE_PARTIAL = 10010
TRADE_RETCODE_ERROR = 10011
TRADE_RETCODE_TIMEOUT = 10012
TRADE_RETCODE_INVALID = 10013
TRADE_RETCODE_INVALID_VOLUME = 10014
TRADE_RETCODE_INVALID_PRICE = 10015
TRADE_RETCODE_INVALID_STOPS = 10016
TRADE_RETCODE_TRADE_DISABLED = 10017
TRADE_RETCODE_MARKET_CLOSED = 10018
TRADE_RETCODE_NO_MONEY = 10019
TRADE_RETCODE_PRICE_CHANGED = 10020
TRADE_RETCODE_PRICE_OFF = 10021
TRADE_RETCODE_INVALID_EXPIRATION = 10022
TRADE_RETCODE_ORDER_CHANGED = 10023
TRADE_RETCODE_TOO_MANY_REQUESTS = 10024
TRADE_RETCODE_NO_CHANGES = 10025
TRADE_RETCODE_SERVER_DISABLES_AT = 10026
TRADE_RETCODE_CLIENT_DISABLES_AT = 10027
TRADE_RETCODE_LOCKED = 10028
TRADE_RETCODE_FROZEN = 10029
TRADE_RETCODE_INVALID_FILL = 10030
TRADE_RETCODE_CONNECTION = 10031
TRADE_RETCODE_ONLY_REAL = 10032
TRADE_RETCODE_LIMIT_ORDERS = 10033
TRADE_RETCODE_LIMIT_VOLUME = 10034
TRADE_RETCODE_INVALID_ORDER = 10035
TRADE_RETCODE_POSITION_CLOSED = 10036
TRADE_RETCODE_INVALID_CLOSE_VOLUME = 10038
TRADE_RETCODE_CLOSE_ORDER_EXIST = 10039
TRADE_RETCODE_LIMIT_POSITIONS = 10040
TRADE_RETCODE_REJECT_CANCEL = 10041
TRADE_RETCODE_LONG_ONLY = 10042
TRADE_RETCODE_SHORT_ONLY = 10043
TRADE_RETCODE_CLOSE_ONLY = 10044
TRADE_RETCODE_FIFO_CLOSE = 10045

POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

DEAL_TYPE_BUY = 0
DEAL_TYPE_SELL = 1
DEAL_TYPE_BALANCE = 2

DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3

ORDER_STATE_STARTED = 0
ORDER_STATE_PLACED = 1
ORDER_STATE_CANCELED = 2
ORDER_STATE_PARTIAL = 3
ORDER_STATE_FILLED = 4
ORDER_STATE_REJECTED = 5
ORDER_STATE_EXPIRED = 6

SYMBOL_TRADE_MODE_DISABLED = 0
SYMBOL_TRADE_MODE_LONGONLY = 1
SYMBOL_TRADE_MODE_SHORTONLY = 2
SYMBOL_TRADE_MODE_CLOSEONLY = 3
SYMBOL_TRADE_MODE_FULL = 4

RES_S_OK = 1
RES_E_FAIL = -1
RES_E_INVALID_PARAMS = -2
RES_E_NO_MEMORY = -3
RES_E_NOT_FOUND = -4
RES_E_INVALID_VERSION = -5
RES_E_AUTH_FAILED = -6
RES_E_UNSUPPORTED = -7
RES_E_AUTO_TRADING_DISABLED = -8
RES_E_INTERNAL_FAIL = -10000
RES_E_INTERNAL_FAIL_SEND = -10001
RES_E_INTERNAL_FAIL_RECEIVE = -10002
RES_E_INTERNAL_FAIL_INIT = -10003
RES_E_INTERNAL_FAIL_CONNECT = -10004
RES_E_INTERNAL_FAIL_TIMEOUT = -10005


def _parse_kv(line: str) -> dict[str, str]:
    """Parse 'key=val|key=val|...' into dict."""
    result = {}
    if not line:
        return result
    for part in line.split("|"):
        if "=" in part:
            k, v = part.split("=", 1)
            result[k] = v
    return result


def _encode_kv(d: dict[str, Any]) -> str:
    """Encode dict to 'key=val|key=val|...' string."""
    parts = []
    for k, v in d.items():
        parts.append(f"{k}={v}")
    return "|".join(parts)


class _NamedTuple:
    """Simple named tuple-like object parsed from key=val pairs."""

    def __init__(self, data: dict[str, Any]):
        self._data = data
        for k, v in data.items():
            setattr(self, k, v)

    def __repr__(self):
        return f"NT({self._data})"

    def _asdict(self):
        return dict(self._data)


def _coerce_numeric(d: dict[str, str]) -> dict[str, Any]:
    """Try to convert string values to numeric types."""
    result = {}
    for k, v in d.items():
        try:
            if "." in v:
                result[k] = float(v)
            else:
                result[k] = int(v)
        except (ValueError, TypeError):
            result[k] = v
    return result


class MT5SocketClient:
    """Client for the MQL5 Socket Server EA."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000, timeout: int = 30):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None
        self._buf = b""

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        self._buf = b""
        logger.info(f"MT5 socket client connected to {self.host}:{self.port}")

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
            logger.info("MT5 socket client disconnected")

    def _send(self, cmd: str) -> None:
        if not self._sock:
            raise ConnectionError("not connected")
        self._sock.sendall((cmd + "\n").encode())

    def _recv_line(self) -> str:
        """Receive a single newline-terminated line."""
        if not self._sock:
            raise ConnectionError("not connected")
        while b"\n" not in self._buf:
            chunk = self._sock.recv(65536)
            if not chunk:
                raise ConnectionError("server closed connection")
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        return line.decode(errors="replace").strip()

    def _call(self, cmd: str) -> dict[str, Any]:
        """Send command, receive single-line OK/ERR response, return parsed dict."""
        self._send(cmd)
        resp = self._recv_line()
        if resp.startswith("OK "):
            return _parse_kv(resp[3:])
        elif resp.startswith("ERR "):
            raise RuntimeError(f"MT5 error: {resp[4:]}")
        elif resp == "OK":
            return {}
        else:
            raise RuntimeError(f"unexpected response: {resp}")

    def _call_multi(self, cmd: str) -> list[dict[str, Any]]:
        """Send command, receive multi-line response (COUNT + lines + END)."""
        self._send(cmd)
        resp = self._recv_line()
        if resp.startswith("ERR "):
            raise RuntimeError(f"MT5 error: {resp[4:]}")
        if not resp.startswith("COUNT "):
            raise RuntimeError(f"expected COUNT, got: {resp}")
        count = int(resp[6:])
        results = []
        for _ in range(count):
            line = self._recv_line()
            if line == "END":
                break
            if line.startswith("BAR "):
                results.append(_coerce_numeric(_parse_kv(line[4:])))
            elif line.startswith("POS "):
                results.append(_coerce_numeric(_parse_kv(line[4:])))
            elif line.startswith("ORD "):
                results.append(_coerce_numeric(_parse_kv(line[4:])))
            elif line.startswith("HORD "):
                results.append(_coerce_numeric(_parse_kv(line[5:])))
            elif line.startswith("HDEAL "):
                results.append(_coerce_numeric(_parse_kv(line[6:])))
            elif line.startswith("SYM "):
                results.append({"symbol": line[4:]})
            else:
                results.append(_parse_kv(line))
        # consume END if not already consumed
        if results and isinstance(results[-1], dict) and results[-1].get("END"):
            results.pop()
        return results

    # --- module-level API (mirrors MetaTrader5) ---

    def initialize(self, login: str = "", password: str = "", server: str = "") -> bool:
        if not self._sock:
            self.connect()
        args = f"{login} {password} {server}".strip()
        try:
            result = self._call(f"INIT {args}")
            return result.get("login") is not None or "already" in result or result == {}
        except Exception as e:
            logger.warning(f"MT5 initialize failed: {e}")
            return False

    def shutdown(self) -> None:
        try:
            self._call("SHUTDOWN")
        except Exception:
            pass
        self.disconnect()

    def login(self, login: str = "", password: str = "", server: str = "") -> bool:
        return self.initialize(login, password, server)

    def account_info(self) -> Optional[_NamedTuple]:
        try:
            data = self._call("ACCOUNT")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"account_info failed: {e}")
            return None

    def terminal_info(self) -> Optional[_NamedTuple]:
        try:
            data = self._call("TERMINAL")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"terminal_info failed: {e}")
            return None

    def version(self) -> Optional[tuple]:
        try:
            data = self._call("VERSION")
            build = int(data.get("build", 0))
            return (500, build, "unknown")
        except Exception:
            return None

    def last_error(self) -> tuple:
        try:
            data = self._call("LASTERR")
            return (int(data.get("code", -1)), "")
        except Exception:
            return (RES_E_FAIL, "socket error")

    def symbols_total(self) -> int:
        try:
            data = self._call("SYMBOLS_TOTAL")
            return int(data.get("count", 0))
        except Exception:
            return 0

    def symbols_get(self, group: Optional[str] = None) -> list[dict]:
        try:
            return self._call_multi("SYMBOLS")
        except Exception as e:
            logger.warning(f"symbols_get failed: {e}")
            return []

    def symbol_info(self, symbol: str) -> Optional[_NamedTuple]:
        try:
            data = self._call(f"SYMBOL {symbol}")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"symbol_info({symbol}) failed: {e}")
            return None

    def symbol_info_tick(self, symbol: str) -> Optional[_NamedTuple]:
        try:
            data = self._call(f"TICK {symbol}")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"symbol_info_tick({symbol}) failed: {e}")
            return None

    def symbol_select(self, symbol: str, enable: bool = True) -> bool:
        try:
            self._call(f"SELECT {symbol} {1 if enable else 0}")
            return True
        except Exception as e:
            logger.warning(f"symbol_select({symbol}) failed: {e}")
            return False

    def copy_rates_from_pos(self, symbol: str, timeframe: int, start_pos: int, count: int) -> list[dict]:
        try:
            return self._call_multi(f"RATES_POS {symbol} {timeframe} {start_pos} {count}")
        except Exception as e:
            logger.warning(f"copy_rates_from_pos failed: {e}")
            return []

    def copy_rates_from(self, symbol: str, timeframe: int, date_from, count: int) -> list[dict]:
        ts = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        try:
            return self._call_multi(f"RATES_RANGE {symbol} {timeframe} {ts} {ts + count * 3600}")
        except Exception as e:
            logger.warning(f"copy_rates_from failed: {e}")
            return []

    def copy_rates_range(self, symbol: str, timeframe: int, date_from, date_to) -> list[dict]:
        ts1 = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        ts2 = int(date_to.timestamp()) if hasattr(date_to, "timestamp") else int(date_to)
        try:
            return self._call_multi(f"RATES_RANGE {symbol} {timeframe} {ts1} {ts2}")
        except Exception as e:
            logger.warning(f"copy_rates_range failed: {e}")
            return []

    def orders_total(self) -> int:
        try:
            data = self._call_multi("ORDERS")
            return len(data)
        except Exception:
            return 0

    def orders_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> list[_NamedTuple]:
        try:
            data = self._call_multi("ORDERS")
            return [_NamedTuple(d) for d in data]
        except Exception:
            return []

    def positions_total(self) -> int:
        try:
            data = self._call_multi("POSITIONS")
            return len(data)
        except Exception:
            return 0

    def positions_get(self, symbol: Optional[str] = None, group: Optional[str] = None, ticket: Optional[int] = None) -> list[_NamedTuple]:
        try:
            data = self._call_multi("POSITIONS")
            return [_NamedTuple(d) for d in data]
        except Exception:
            return []

    def order_check(self, request: dict) -> Optional[_NamedTuple]:
        try:
            kv = _encode_kv(request)
            data = self._call(f"ORDER_CHECK {kv}")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"order_check failed: {e}")
            return None

    def order_send(self, request: dict) -> Optional[_NamedTuple]:
        try:
            kv = _encode_kv(request)
            data = self._call(f"ORDER_SEND {kv}")
            return _NamedTuple(_coerce_numeric(data))
        except Exception as e:
            logger.warning(f"order_send failed: {e}")
            return None

    def history_orders_total(self, date_from, date_to) -> int:
        ts1 = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        ts2 = int(date_to.timestamp()) if hasattr(date_to, "timestamp") else int(date_to)
        try:
            data = self._call_multi(f"HIST_ORDERS {ts1} {ts2}")
            return len(data)
        except Exception:
            return 0

    def history_orders_get(self, date_from, date_to, group: Optional[str] = None) -> list[_NamedTuple]:
        ts1 = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        ts2 = int(date_to.timestamp()) if hasattr(date_to, "timestamp") else int(date_to)
        try:
            data = self._call_multi(f"HIST_ORDERS {ts1} {ts2}")
            return [_NamedTuple(d) for d in data]
        except Exception:
            return []

    def history_deals_total(self, date_from, date_to) -> int:
        ts1 = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        ts2 = int(date_to.timestamp()) if hasattr(date_to, "timestamp") else int(date_to)
        try:
            data = self._call_multi(f"HIST_DEALS {ts1} {ts2}")
            return len(data)
        except Exception:
            return 0

    def history_deals_get(self, date_from, date_to, group: Optional[str] = None) -> list[_NamedTuple]:
        ts1 = int(date_from.timestamp()) if hasattr(date_from, "timestamp") else int(date_from)
        ts2 = int(date_to.timestamp()) if hasattr(date_to, "timestamp") else int(date_to)
        try:
            data = self._call_multi(f"HIST_DEALS {ts1} {ts2}")
            return [_NamedTuple(d) for d in data]
        except Exception:
            return []


# --- singleton module instance ---
_client = MT5SocketClient()


def _get_client() -> MT5SocketClient:
    return _client


# --- module-level functions that delegate to the singleton ---

def initialize(login: str = "", password: str = "", server: str = "") -> bool:
    return _get_client().initialize(login, password, server)


def shutdown() -> None:
    _get_client().shutdown()


def login(login: str = "", password: str = "", server: str = "") -> bool:
    return _get_client().login(login, password, server)


def account_info():
    return _get_client().account_info()


def terminal_info():
    return _get_client().terminal_info()


def version():
    return _get_client().version()


def last_error():
    return _get_client().last_error()


def symbols_total() -> int:
    return _get_client().symbols_total()


def symbols_get(group: Optional[str] = None):
    return _get_client().symbols_get(group)


def symbol_info(symbol: str):
    return _get_client().symbol_info(symbol)


def symbol_info_tick(symbol: str):
    return _get_client().symbol_info_tick(symbol)


def symbol_select(symbol: str, enable: bool = True) -> bool:
    return _get_client().symbol_select(symbol, enable)


def copy_rates_from_pos(symbol: str, timeframe: int, start_pos: int, count: int):
    return _get_client().copy_rates_from_pos(symbol, timeframe, start_pos, count)


def copy_rates_from(symbol: str, timeframe: int, date_from: Any, count: int):
    return _get_client().copy_rates_from(symbol, timeframe, date_from, count)


def copy_rates_range(symbol: str, timeframe: int, date_from: Any, date_to: Any):
    return _get_client().copy_rates_range(symbol, timeframe, date_from, date_to)


def orders_total() -> int:
    return _get_client().orders_total()


def orders_get(symbol: str = "", group: str = "", ticket: int = 0):
    return _get_client().orders_get(symbol or None, group or None, ticket or None)


def positions_total() -> int:
    return _get_client().positions_total()


def positions_get(symbol: str = "", group: str = "", ticket: int = 0):
    return _get_client().positions_get(symbol or None, group or None, ticket or None)


def order_check(request: dict):
    return _get_client().order_check(request)


def order_send(request: dict):
    return _get_client().order_send(request)


def history_orders_total(date_from, date_to) -> int:
    return _get_client().history_orders_total(date_from, date_to)


def history_orders_get(date_from, date_to, group: Optional[str] = None):
    return _get_client().history_orders_get(date_from, date_to, group)


def history_deals_total(date_from, date_to) -> int:
    return _get_client().history_deals_total(date_from, date_to)


def history_deals_get(date_from, date_to, group: Optional[str] = None):
    return _get_client().history_deals_get(date_from, date_to, group)
