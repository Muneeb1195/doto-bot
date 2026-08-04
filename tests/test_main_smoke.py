"""Smoke test: main.py must import and parse.

Historically the per-cycle entry/exit logic lived inside main.py and a broken
try/except from an earlier session edit left the entire trading loop
mis-indented inside the circuit-breaker guard (so the bot never traded) and
even introduced a hard SyntaxError. main.py is not imported by the rest of the
suite, so add an explicit import smoke test to catch regressions there.
"""

import sys
from unittest.mock import MagicMock

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")


def test_main_imports():
    import main  # noqa: F401  (must parse + import without error)
    assert hasattr(main, "place_trade")
    assert hasattr(main, "place_mean_reversion_trade")
