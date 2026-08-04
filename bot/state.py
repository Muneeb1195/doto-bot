"""Shared mutable state for the TrendBot."""

import contextlib
import json
import logging
import os
from datetime import date
from pathlib import Path

__all__ = [
    "BASE_DIR",
    "LOG_DIR",
    "CONFIG_DIR",
    "DASHBOARD_STATE",
    "STATE_FILE",
    "NEWS_SENTIMENT_FILE",
    "TRADE_CSV",
    "TRADE_HEADERS",
    "BOT_MAGIC",
    "ASSET_CLASS_MAP",
    "reset_all",
]

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
CONFIG_DIR = BASE_DIR / "config"

DASHBOARD_STATE = BASE_DIR / "data" / "dashboard_state.json"
STATE_FILE = BASE_DIR / "data" / "bot_state.json"
NEWS_SENTIMENT_FILE = BASE_DIR / "data" / "news_sentiment.json"
TRADE_CSV = LOG_DIR / "trades.csv"
TRADE_HEADERS = [
    "ticket",
    "symbol",
    "type",
    "volume",
    "entry_price",
    "sl",
    "tp",
    "entry_time",
    "atr",
    "exit_price",
    "exit_time",
    "pnl",
    "pips",
    "event",
]

# Magic number the bot stamps on every order it places. External/manual trades
# carry a different magic (typically 0), which lets the journal distinguish them.
BOT_MAGIC = 20240706

_last_trade_time: dict[str, float] = {}  # key -> timestamp
_scale_out_state: dict[int, dict] = {}  # ticket -> dict
_chandelier_state: dict[int, dict] = {}  # ticket -> dict

_ml_models: dict[str, dict] = {}  # symbol -> {"model": ..., "metadata": ...}
_ml_pool_models: dict[str, dict] = {}  # asset_class -> {"model": ..., "metadata": ...}
_ml_meta_models: dict[str, dict] = {}  # symbol -> {"model": ..., "metadata": ...}

ASSET_CLASS_MAP = {
    "ETHUSD.raw": "crypto",
    "BTCUSD.raw": "crypto",
    "LTCUSD.raw": "crypto",
    "DOGUSD.raw": "crypto",
    "ADAUSD.raw": "crypto",
    "BNBUSD.raw": "crypto",
    "SOLUSD.raw": "crypto",
    "XRPUSD.raw": "crypto",
    "EURUSD.raw": "forex",
    "GBPUSD.raw": "forex",
    "USDJPY.raw": "forex",
    "EURJPY.raw": "forex",
    "GBPJPY.raw": "forex",
    "NZDUSD.raw": "forex",
    "AUDUSD.raw": "forex",
    "USDCAD.raw": "forex",
    "USDCHF.raw": "forex",
    "XAUUSD.raw": "commodity",
    "XAGUSD.raw": "commodity",
    "XNGUSD.raw": "commodity",
    "XAU500.raw": "commodity",
    "XPTUSD.raw": "commodity",
    "US30.raw": "index",
    "SPY.raw": "index",
    "US500.raw": "index",
    "IWM.raw": "index",
}

_exec_bias: dict[str, dict] = {}  # symbol -> {"bias": str, "since": float, "date": date}
_filter_stats: dict[str, dict] = {}  # symbol -> dict of filter counters
_tail_risk_triggered: dict[str, bool] = {}  # symbol -> bool
_tail_risk_cooldown: dict[str, float] = {}  # symbol -> timestamp
_circuit_breaker_triggered: bool = False
_mr_consecutive_losses: dict[str, int] = {}  # symbol -> int
_mr_last_loss_time: dict[str, float] = {}  # symbol -> timestamp
_exec_quality: dict[str, dict] = {}  # symbol -> {"slippage_sum": ..., "slippage_count": ..., ...}
_dynamic_deviation: dict[str, int] = {}  # symbol -> current auto-adjusted deviation
_last_corr_time: float = 0
_corr_cache: dict = {}
_rate_cache: dict = {}  # (symbol, tf) -> (DataFrame, bars, timestamp)
_RATE_CACHE_TTL: float = 5.0  # seconds before re-fetch
_last_symbol_reselect_time: float = 0
_symbol_online: dict[str, bool] = {}
_shutdown_requested: bool = False
_last_cycle_time: float = 0.0
_cycle_consecutive_failures: int = 0
_ns_cache: dict = {"data": None, "mtime": 0}
_last_daily_summary_day = None
_daily_realized_pnl: float = 0.0
_daily_realized_date = None
_daily_loss_hit: bool = False
_daily_loss_mt5_cache: tuple = (
    None,
    None,
    0,
)  # (date, realized_loss, timestamp) cache for MT5 deal-history based daily loss
_peak_balance: float = 0.0
_ml_confidence_history: dict[str, list] = {}  # symbol -> list of recent confidence values
_ml_confidence_baseline: dict[str, float] = {}  # symbol -> frozen baseline mean (first 50 confs)
_ml_drift_warned: set[str] = set()  # symbols already flagged for drift
_imported_external_ids: set[str] = set()  # position_id strings already backfilled as EXTERNAL in current session
_pending_limits: dict[str, dict] = {}  # symbol -> {"ticket", "signal", "price", "atr", "cycles", ...}
_regime_gate_state: dict[str, object] = {}  # symbol -> RegimeGate instance (lazy-inited in signals.py)

_WATCHDOG_FAILURES = 0
_WATCHDOG_LAST_KILL = 0.0
_WATCHDOG_MAX_FAILURES = 3


def reset_all():
    """Reset all mutable state to defaults.  Called after every test via conftest.py
    autouse fixture so tests can run in any order without cross-contamination."""
    _scale_out_state.clear()
    _chandelier_state.clear()
    _last_trade_time.clear()
    _exec_bias.clear()
    _filter_stats.clear()
    _tail_risk_triggered.clear()
    _tail_risk_cooldown.clear()
    global _circuit_breaker_triggered
    _circuit_breaker_triggered = False
    _mr_consecutive_losses.clear()
    _mr_last_loss_time.clear()
    _exec_quality.clear()
    _dynamic_deviation.clear()
    global _last_corr_time
    _last_corr_time = 0
    _corr_cache.clear()
    _rate_cache.clear()
    global _last_symbol_reselect_time
    _last_symbol_reselect_time = 0
    _symbol_online.clear()
    global _shutdown_requested, _last_cycle_time, _cycle_consecutive_failures
    _shutdown_requested = False
    _last_cycle_time = 0.0
    _cycle_consecutive_failures = 0
    _ns_cache.clear()
    _ns_cache.update({"data": None, "mtime": 0})
    global _last_daily_summary_day, _daily_realized_pnl, _daily_realized_date
    _last_daily_summary_day = None
    _daily_realized_pnl = 0.0
    _daily_realized_date = None
    global _daily_loss_hit
    _daily_loss_hit = False
    global _daily_loss_mt5_cache
    _daily_loss_mt5_cache = (None, None, 0)
    global _peak_balance
    _peak_balance = 0.0
    _ml_confidence_history.clear()
    _ml_confidence_baseline.clear()
    _ml_drift_warned.clear()
    _imported_external_ids.clear()
    _pending_limits.clear()
    _regime_gate_state.clear()
    _ml_models.clear()
    _ml_pool_models.clear()
    _ml_meta_models.clear()
    global _WATCHDOG_FAILURES, _WATCHDOG_LAST_KILL
    _WATCHDOG_FAILURES = 0
    _WATCHDOG_LAST_KILL = 0.0


def save_bot_state():
    try:
        exec_bias_ser = {}
        for sym, eb in _exec_bias.items():
            eb2 = dict(eb)
            if "date" in eb2 and isinstance(eb2["date"], date):
                eb2["date"] = eb2["date"].isoformat()
            exec_bias_ser[sym] = eb2
        state_data = {
            "scale_out_state": {str(k): v for k, v in _scale_out_state.items()},
            "chandelier_state": {str(k): v for k, v in _chandelier_state.items()},
            "exec_bias": exec_bias_ser,
            "last_trade_time": _last_trade_time,
            "tail_risk_triggered": _tail_risk_triggered,
            "tail_risk_cooldown": _tail_risk_cooldown,
            "circuit_breaker_triggered": _circuit_breaker_triggered,
            "peak_balance": _peak_balance,
            "mr_consecutive_losses": _mr_consecutive_losses,
            "mr_last_loss_time": _mr_last_loss_time,
            "dynamic_deviation": _dynamic_deviation,
            "daily_loss_hit": _daily_loss_hit,
            "daily_realized_pnl": _daily_realized_pnl,
            "daily_realized_date": _daily_realized_date.isoformat() if _daily_realized_date else None,
            "pending_limits": _pending_limits,
            "imported_external_ids": sorted(_imported_external_ids),
        }
        _INVARIANT_KEYS = {
            "scale_out_state",
            "chandelier_state",
            "exec_bias",
            "last_trade_time",
            "tail_risk_triggered",
            "tail_risk_cooldown",
            "circuit_breaker_triggered",
            "peak_balance",
            "mr_consecutive_losses",
            "mr_last_loss_time",
            "dynamic_deviation",
            "daily_loss_hit",
            "daily_realized_pnl",
            "daily_realized_date",
            "pending_limits",
            "imported_external_ids",
        }
        missing = _INVARIANT_KEYS - set(state_data)
        if missing:
            logging.warning(f"save_bot_state: missing keys {missing}")
        STATE_FILE.parent.mkdir(exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state_data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(STATE_FILE))
    except Exception as e:
        logging.warning(f"save_bot_state failed: {e}")


def load_bot_state():
    try:
        if not STATE_FILE.exists():
            return
        with open(STATE_FILE) as f:
            state_data = json.load(f)
        if not isinstance(state_data, dict):
            logging.warning(f"load_bot_state: state_data is {type(state_data).__name__}, expected dict")
            return
        if "scale_out_state" not in state_data:
            logging.warning("load_bot_state: state_data missing 'scale_out_state' — empty state?")
        global _scale_out_state, _chandelier_state, _exec_bias, _last_trade_time
        global _tail_risk_triggered, _tail_risk_cooldown, _circuit_breaker_triggered, _peak_balance
        global _mr_consecutive_losses, _mr_last_loss_time, _dynamic_deviation, _daily_loss_hit
        global _daily_realized_pnl, _daily_realized_date, _pending_limits, _imported_external_ids
        _scale_out_state.clear()
        _scale_out_state.update({int(k): v for k, v in state_data.get("scale_out_state", {}).items()})
        _chandelier_state.clear()
        _chandelier_state.update({int(k): v for k, v in state_data.get("chandelier_state", {}).items()})
        _exec_bias.clear()
        for sym, eb in state_data.get("exec_bias", {}).items():
            if "date" in eb and isinstance(eb["date"], str):
                with contextlib.suppress(ValueError, TypeError):
                    eb["date"] = date.fromisoformat(eb["date"])
            _exec_bias[sym] = eb
        _last_trade_time.clear()
        _last_trade_time.update(state_data.get("last_trade_time", {}))
        raw_tr = state_data.get("tail_risk_triggered", {})
        _tail_risk_triggered.clear()
        _tail_risk_triggered.update(raw_tr if isinstance(raw_tr, dict) else {})
        raw_cd = state_data.get("tail_risk_cooldown", {})
        _tail_risk_cooldown.clear()
        _tail_risk_cooldown.update(raw_cd if isinstance(raw_cd, dict) else {})
        _circuit_breaker_triggered = state_data.get("circuit_breaker_triggered", False)
        _peak_balance = state_data.get("peak_balance", 0.0)
        _mr_consecutive_losses.clear()
        _mr_consecutive_losses.update(state_data.get("mr_consecutive_losses", {}))
        _mr_last_loss_time.clear()
        _mr_last_loss_time.update(state_data.get("mr_last_loss_time", {}))
        _dynamic_deviation.clear()
        _dynamic_deviation.update(state_data.get("dynamic_deviation", {}))
        _daily_loss_hit = state_data.get("daily_loss_hit", False)
        raw_pnl = state_data.get("daily_realized_pnl", 0.0)
        _daily_realized_pnl = float(raw_pnl) if isinstance(raw_pnl, (int, float)) else 0.0
        raw_date = state_data.get("daily_realized_date", None)
        if isinstance(raw_date, str):
            try:
                _daily_realized_date = date.fromisoformat(raw_date)
            except (ValueError, TypeError):
                _daily_realized_date = None
        else:
            _daily_realized_date = None
        raw_pl = state_data.get("pending_limits", {})
        _pending_limits.clear()
        _pending_limits.update(raw_pl if isinstance(raw_pl, dict) else {})
        raw_ids = state_data.get("imported_external_ids", [])
        _imported_external_ids.clear()
        _imported_external_ids.update(raw_ids if isinstance(raw_ids, (list, tuple, set)) else [])
        logging.info(
            f"Bot state loaded: {len(_scale_out_state)} scale-out, {len(_chandelier_state)} chandelier entries"
        )
    except Exception as e:
        logging.warning(f"load_bot_state failed: {e}")


def load_news_sentiment():
    try:
        if not NEWS_SENTIMENT_FILE.exists():
            return
        mtime = NEWS_SENTIMENT_FILE.stat().st_mtime
        if _ns_cache.get("mtime") == mtime:
            return
        with open(NEWS_SENTIMENT_FILE) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        _ns_cache["data"] = data
        _ns_cache["mtime"] = mtime
    except Exception as e:
        logging.warning(f"load_news_sentiment failed: {e}")
