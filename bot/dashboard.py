"""Dashboard state writer."""

import json
import logging
import os

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
from mt5_connect import mt5_call
from state import DASHBOARD_STATE, _filter_stats


def write_dashboard_state(all_positions, regimes):
    try:
        acc = mt5_call(mt5.account_info)
        if acc is None:
            return
        state = {
            "balance": acc.balance,
            "equity": acc.equity,
            "profit": acc.profit,
            "margin_free": acc.margin_free,
            "positions": len(all_positions),
            "regimes": regimes.copy(),
            "positions_detail": [
                {
                    "symbol": p.symbol,
                    "type": "buy" if p.type == mt5.ORDER_TYPE_BUY else "sell",
                    "volume": p.volume,
                    "price_open": p.price_open,
                    "sl": p.sl,
                    "tp": p.tp,
                    "profit": p.profit,
                    "swap": p.swap,
                    "ticket": p.ticket,
                }
                for p in all_positions
            ],
            "filters": {s: dict(v) for s, v in _filter_stats.items()},
            "health": {
                "connected": bool(getattr(mt5_call(mt5.terminal_info), "connected", False)),
                "server": getattr(mt5_call(mt5.account_info), "server", ""),
            },
        }
        tmp = DASHBOARD_STATE.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, DASHBOARD_STATE)
    except PermissionError as e:
        logging.debug(f"write_dashboard_state file locked: {e}")
    except Exception as e:
        logging.warning(f"write_dashboard_state failed: {e}")
