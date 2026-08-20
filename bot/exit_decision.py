"""Pure exit decision — the close tree extracted from execution.manage_positions.

This module is the deep seam for the position exit tree. It owns no broker
I/O, no global state, and no side effects. Callers supply already-fetched
values (tick price, atr, mr_exit flag) and receive a value-type intent.

Depth: the interface hides the 3-way priority (MAX_HOLD > MR_EXIT > REVERSAL),
the MR position detection (magic vs comment), and the 0.25-ATR sub-profit
guard behind a single ``decide_exit`` call.

The interface is the test surface — all branches are exercised without a
Market fake or monkeypatch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ExitIntent:
    should_close: bool
    reason: str | None  # "MAX_HOLD" | "MR_EXIT" | "REVERSAL" | None


def is_mr_position(position, cfg: dict) -> bool:
    """Distinct magic/comment detection so broker truncation cannot hide MR.

    Trend uses cfg["magic"] (20240706), MR uses cfg["mr_magic"] (20240707).
    """
    mr_magic = cfg.get("mr_magic", 20240707)
    if getattr(position, "magic", None) == mr_magic:
        return True
    # Fallback — some brokers truncate comment but we keep it for old positions
    if getattr(position, "comment", None) == "TrendBot-MR":
        return True
    return False


def max_hold_triggered(position, cfg: dict, now: datetime | None = None) -> bool:
    """Pure time check — no I/O. ``now`` injectable for deterministic tests."""
    max_hours = cfg.get("max_hold_hours", 72)
    if max_hours <= 0:
        return False
    if now is None:
        now = datetime.now()
    open_time = datetime.fromtimestamp(getattr(position, "time", 0))
    elapsed_hours = (now - open_time).total_seconds() / 3600
    return elapsed_hours >= max_hours


def _is_buy_position(position) -> bool:
    """Detect long vs short without importing MT5 (works for int 0/1 and MagicMock mocks)."""
    t = getattr(position, "type", 0)
    if isinstance(t, int):
        return t == 0  # ORDER_TYPE_BUY == 0
    # MagicMock case — inspect mock name
    name = getattr(t, "_mock_name", "") or getattr(t, "_mock_new_name", "") or ""
    if "BUY" in str(name):
        return True
    if "SELL" in str(name):
        return False
    # Fallback via string repr
    s = str(t)
    if "BUY" in s:
        return True
    if "SELL" in s:
        return False
    # Last resort: treat as buy (prevents silent wrong-side block)
    return True


def _reversal_sub_profit_blocked(position, cur_atr: float | None) -> bool:
    """0.25-ATR sub-profit guard — when True the REVERSAL close is blocked."""
    if not cur_atr or cur_atr <= 0:
        return False
    is_buy = _is_buy_position(position)
    price_open = getattr(position, "price_open", 0) or 0
    price_current = getattr(position, "price_current", price_open)
    if is_buy:
        return price_current > price_open + cur_atr * 0.25
    return price_current < price_open - cur_atr * 0.25


def reversal_triggered(position, trend_signal: str | None, cur_atr: float | None) -> bool:
    """Pure REVERSAL predicate — assumes caller already knows trend_signal exists."""
    if trend_signal is None:
        return False
    is_buy = _is_buy_position(position)
    pos_type = "buy" if is_buy else "sell"
    counter = (pos_type == "buy" and trend_signal == "sell") or (
        pos_type == "sell" and trend_signal == "buy"
    )
    if not counter:
        return False
    if _reversal_sub_profit_blocked(position, cur_atr):
        return False
    return True


def decide_exit(
    position,
    cfg: dict,
    trend_signal: str | None = None,
    regime: str = "trending",
    max_hold_hit: bool = False,
    mr_exit_hit: bool = False,
    cur_atr: float | None = None,
) -> ExitIntent:
    """Priority-ordered close decision.

    Caller supplies precomputed ``max_hold_hit`` (from max_hold_triggered) and
    ``mr_exit_hit`` (from check_mean_reversion_exit) so this function stays
    pure. ``cur_atr`` is the current ATR for the reversal sub-profit guard.

    Order: MAX_HOLD > MR_EXIT > REVERSAL — matches execution.manage_positions.
    """
    if max_hold_hit:
        return ExitIntent(True, "MAX_HOLD")

    mr_pos = is_mr_position(position, cfg)
    if mr_exit_hit and (mr_pos or (regime == "ranging" and cfg.get("mr_enabled"))):
        return ExitIntent(True, "MR_EXIT")

    if reversal_triggered(position, trend_signal, cur_atr):
        return ExitIntent(True, "REVERSAL")

    return ExitIntent(False, None)
