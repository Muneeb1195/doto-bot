"""Single MT5 access shim.

Binds the module-level name `mt5` to either the native MetaTrader5 package
(Windows dev box / CI with the package installed) or the RPyC bridge proxy
from mt5_connect (Linux home server, where the native package does not exist).
Modules that need the `mt5` name should `from _mt5 import mt5` instead of
re-implementing this try/except.

The binding resolves lazily on EVERY consumer import (module __getattr__), so
tests that inject `sys.modules["MetaTrader5"]` and then re-import a bot module
pick up their injected object — matching the old per-module try/except
semantics where each module bound `mt5` at its own import time.

Deliberate exceptions (documented at each site, do NOT "fix"):
- train_model.py keeps `mt5 = None` on ImportError: it only runs in CSV/CI mode
  there, must not pay the mt5_connect auto-init import, and every call site is
  guarded by `if not csv_mode` so the proxy would never be used anyway.
- ml_features.py keeps function-local try/except: its ImportError branch is a
  DIFFERENT fallback (M1-bars aggregation), not the bridge proxy.
"""

import sys


def _resolve_mt5():
    """Return the active MT5 module for the importing caller."""
    injected = sys.modules.get("MetaTrader5")
    if injected is not None:
        return injected
    try:
        import MetaTrader5 as native
    except ImportError:  # Linux: no native package, use the socket/RPyC bridge
        from mt5_connect import mt5 as bridged

        return bridged
    return native


def __getattr__(name):
    if name == "mt5":
        return _resolve_mt5()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
