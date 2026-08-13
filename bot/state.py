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
    "US100.raw": "index",
    "UK100.raw": "index",
    "JP225.raw": "index",
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


def _reset_field(gname, kind, default):
    """Reset one PERSISTED field to its default (in-place clear for mutable
    kinds so module-held references stay valid). Shared by reset_all and the
    per-field load fallback."""
    if kind in _MUTABLE_KINDS:
        globals()[gname].clear()
    else:
        globals()[gname] = default


def reset_all():
    """Reset all mutable state to defaults.  Called after every test via conftest.py
    autouse fixture so tests can run in any order without cross-contamination."""
    for key, gname, kind, default in PERSISTED:
        _reset_field(gname, kind, default)
    # Runtime-only state (never persisted):
    _filter_stats.clear()
    _exec_quality.clear()
    _corr_cache.clear()
    _rate_cache.clear()
    _symbol_online.clear()
    _ns_cache.clear()
    _ns_cache.update({"data": None, "mtime": 0})
    _ml_confidence_history.clear()
    _ml_confidence_baseline.clear()
    _ml_drift_warned.clear()
    _regime_gate_state.clear()
    _ml_models.clear()
    _ml_pool_models.clear()
    _ml_meta_models.clear()
    global _last_corr_time, _last_symbol_reselect_time, _shutdown_requested
    global _last_cycle_time, _cycle_consecutive_failures, _last_daily_summary_day
    global _daily_loss_mt5_cache, _WATCHDOG_FAILURES, _WATCHDOG_LAST_KILL
    _last_corr_time = 0
    _last_symbol_reselect_time = 0
    _shutdown_requested = False
    _last_cycle_time = 0.0
    _cycle_consecutive_failures = 0
    _last_daily_summary_day = None
    _daily_loss_mt5_cache = (None, None, 0)
    _WATCHDOG_FAILURES = 0
    _WATCHDOG_LAST_KILL = 0.0


PERSISTED = [
    # (state_key, global_name, kind, default) — save/load/reset are all generated
    # from this table so the three hand-written mirrors cannot drift out of
    # lock-step (architecture plan C3). kind:
    #   int_keys_dict — dict keyed by int ticket; JSON stores string keys
    #   dict          — plain dict
    #   exec_bias     — dict whose values may carry a `date` field (isoformat)
    #   ids           — set of strings (JSON list)
    #   bool/float/date — scalars
    ("scale_out_state", "_scale_out_state", "int_keys_dict", {}),
    ("chandelier_state", "_chandelier_state", "int_keys_dict", {}),
    ("exec_bias", "_exec_bias", "exec_bias", {}),
    ("last_trade_time", "_last_trade_time", "dict", {}),
    ("tail_risk_triggered", "_tail_risk_triggered", "dict", {}),
    ("tail_risk_cooldown", "_tail_risk_cooldown", "dict", {}),
    ("circuit_breaker_triggered", "_circuit_breaker_triggered", "bool", False),
    ("peak_balance", "_peak_balance", "float", 0.0),
    ("mr_consecutive_losses", "_mr_consecutive_losses", "dict", {}),
    ("mr_last_loss_time", "_mr_last_loss_time", "dict", {}),
    ("dynamic_deviation", "_dynamic_deviation", "dict", {}),
    ("daily_loss_hit", "_daily_loss_hit", "bool", False),
    ("daily_realized_pnl", "_daily_realized_pnl", "float", 0.0),
    ("daily_realized_date", "_daily_realized_date", "date", None),
    ("pending_limits", "_pending_limits", "dict", {}),
    ("imported_external_ids", "_imported_external_ids", "ids", set()),
]

_MUTABLE_KINDS = {"int_keys_dict", "dict", "exec_bias", "ids"}


def _serialize(kind, value):
    """JSON-safe form for save_bot_state (inverse of _coerce)."""
    if kind == "int_keys_dict":
        return {str(k): v for k, v in value.items()}
    if kind == "exec_bias":
        out = {}
        for sym, eb in value.items():
            eb2 = dict(eb)
            if "date" in eb2 and isinstance(eb2["date"], date):
                eb2["date"] = eb2["date"].isoformat()
            out[sym] = eb2
        return out
    if kind == "ids":
        return sorted(value)
    if kind == "date":
        return value.isoformat() if value else None
    return value


def _coerce(kind, raw, default):
    """Restore a JSON value to its in-memory form (inverse of _serialize).

    Tolerant of legacy/corrupt files: wrong types fall back to the default and
    non-numeric ticket keys are dropped (salvaging the valid entries), so the
    live bot_state.json always loads — a corrupt bit never aborts the load.
    """
    if kind == "int_keys_dict":
        if not isinstance(raw, dict):
            return default
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = v
            except (ValueError, TypeError):
                logging.warning(f"load_bot_state: dropping non-numeric key {k!r}")
        return out
    if kind == "dict":
        return raw if isinstance(raw, dict) else default
    if kind == "exec_bias":
        if not isinstance(raw, dict):
            return default
        out = {}
        for sym, eb in raw.items():
            eb2 = dict(eb) if isinstance(eb, dict) else {}
            if "date" in eb2 and isinstance(eb2["date"], str):
                with contextlib.suppress(ValueError, TypeError):
                    eb2["date"] = date.fromisoformat(eb2["date"])
            out[sym] = eb2
        return out
    if kind == "ids":
        return set(raw) if isinstance(raw, (list, tuple, set)) else default
    if kind == "bool":
        return raw if isinstance(raw, bool) else default
    if kind == "float":
        return float(raw) if isinstance(raw, (int, float)) else default
    if kind == "date":
        if isinstance(raw, str):
            try:
                return date.fromisoformat(raw)
            except (ValueError, TypeError):
                return None
        return None
    return default


def save_bot_state():
    try:
        state_data = {key: _serialize(kind, globals()[gname]) for key, gname, kind, _ in PERSISTED}
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
        for key, gname, kind, default in PERSISTED:
            try:
                value = _coerce(kind, state_data.get(key), default)
                if kind in _MUTABLE_KINDS:
                    target = globals()[gname]
                    target.clear()
                    target.update(value)
                else:
                    globals()[gname] = value
            except Exception as e:
                # One corrupt field must not abort the whole load: log it, fall
                # back to that field's default, and keep loading the rest.
                logging.warning(f"load_bot_state: field '{key}' corrupt ({e}) — using default")
                _reset_field(gname, kind, default)
        logging.info(
            f"Bot state loaded: {len(_scale_out_state)} scale-out, {len(_chandelier_state)} chandelier entries"
        )
    except Exception as e:
        logging.warning(f"load_bot_state failed: {e}")


def prune_position_state(active_tickets):
    """Drop scale-out/chandelier state for tickets no longer in the book
    (broker-fired SL/TP closes, manual closes). Returns True if anything was
    removed. Only call with a VALID positions_get result — an empty set means
    "no positions", and a failed fetch must not be conflated with that.
    """
    pruned = False
    for tickets in (_scale_out_state, _chandelier_state):
        stale = [t for t in tickets if t not in active_tickets]
        for t in stale:
            tickets.pop(t, None)
        if stale:
            pruned = True
    return pruned


def daily_realized_pnl_for(today):
    """Daily realized PnL for `today`, read-only: a stale counter date (new day
    not yet rolled by a journal write) reads as 0.0.

    Single home for the read rule — previously inlined in filters.py
    check_daily_loss (twice). The rollover MUTATION is the journal writer's
    job: state.roll_daily_realized_pnl. Filters must not zero the counter.
    """
    return _daily_realized_pnl if _daily_realized_date == today else 0.0


def roll_daily_realized_pnl(today):
    """Roll the daily-loss counter over at midnight: zero it and stamp `today`
    when the date changed, then return it. Owned by the journal writer
    (journal_close / external-deal backfill)."""
    global _daily_realized_pnl, _daily_realized_date
    if _daily_realized_date != today:
        _daily_realized_pnl = 0.0
        _daily_realized_date = today
    return _daily_realized_pnl


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
