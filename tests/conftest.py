"""Shared fixtures for tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

_rng = np.random.RandomState(42)

_bot_dir = str(Path(__file__).resolve().parent.parent / "bot")
if _bot_dir not in sys.path:
    sys.path.insert(0, _bot_dir)

# scripts/ holds standalone tools (check_deploy_drift, export_mt5_data);
# tests import them as plain modules.
_scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)


# mt5linux 1.1.0 tries to start a Docker container when its MetaTrader5 client
# is instantiated. CI runners have no Docker daemon, and mt5linux 1.1.0 also has
# a '_port' attribute bug. Mock the module at conftest module level so it is in
# place BEFORE test collection imports bot/mt5_connect.py (which instantiates
# the client at module import time). Without this, collection fails on 6 modules.
sys.modules.setdefault("mt5linux", MagicMock())
sys.modules.setdefault("mt5linux.metatrader5", MagicMock())
sys.modules.setdefault("mt5linux._container_manager", MagicMock())


@pytest.fixture(autouse=True, scope="session")
def mock_mt5():
    sys.modules["MetaTrader5"] = MagicMock()
    yield
    if "MetaTrader5" in sys.modules:
        del sys.modules["MetaTrader5"]


@pytest.fixture(autouse=True)
def reset_state():
    import state as _st
    _st.reset_all()
    yield
    _st.reset_all()


@pytest.fixture
def sample_df():
    """Standard H1-like OHLCV DataFrame with 200 bars."""
    n = 200
    closes = 100 + np.cumsum(_rng.randn(n) * 0.5)
    highs = closes + _rng.uniform(0.1, 1.0, n)
    lows = closes - _rng.uniform(0.1, 1.0, n)
    return pd.DataFrame({
        "open": closes - _rng.uniform(0.0, 0.5, n),
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": _rng.randint(100, 10000, n),
    })


@pytest.fixture
def trending_up_df():
    """Strongly trending up -- ideal for crossover detection."""
    n = 200
    closes = 100 + np.linspace(0, 10, n) + _rng.randn(n) * 0.3
    highs = closes + 0.3
    lows = closes - 0.3
    return pd.DataFrame({
        "open": closes - 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": _rng.randint(500, 5000, n),
    })


@pytest.fixture
def sideways_df():
    """Ranging market -- flat price action."""
    n = 200
    closes = 100 + _rng.randn(n) * 0.8
    highs = closes + 0.5
    lows = closes - 0.5
    return pd.DataFrame({
        "open": closes - 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": _rng.randint(100, 1000, n),
    })


@pytest.fixture
def small_df():
    """Minimal DataFrame for edge-case testing."""
    n = 10
    closes = np.array([100.0, 101.0, 102.0, 101.5, 100.5, 99.0, 98.5, 99.5, 100.0, 100.5])
    return pd.DataFrame({
        "open": closes - 0.1,
        "high": closes + 0.5,
        "low": closes - 0.5,
        "close": closes,
        "tick_volume": np.random.randint(100, 1000, n),
    })


@pytest.fixture
def basic_cfg():
    """Minimal config dict for signal/risk tests."""
    return {
        "symbol": "XAU500.raw",
        "timeframe": "H1",
        "ema_fast": 8,
        "ema_slow": 32,
        "atr_period": 14,
        "pb_enabled": True,
        "pb_atr_mult": 1.2,
        "dr_enabled": True,
        "dr_lookback": 50,
        "dr_kelly_fraction": 0.25,
        "dr_min_mult": 0.25,
        "dr_max_mult": 1.5,
        "exec_enabled": True,
        "exec_tf_name": "M15",
        "exec_ema_fast": 9,
        "exec_ema_slow": 21,
        "exec_bias_timeout": 24,
        "vf_enabled": True,
        "vf_sma_period": 20,
        "vf_kappa": 1.2,
        "spf_enabled": True,
        "spf_max_ratio": 0.30,
        "ml_enabled": True,
        "ml_confidence": 0.55,
        "ml_threshold_overrides": {},
        "dr_vol_adjust": True,
        "atr_sma_period": 20,
        "tr_enabled": True,
        "tr_max_dd_pct": 8.0,
        "max_tail_risk_pct": 1.5,
        "max_risk_ratio": 2.0,
        "risk_percent": 1.0,
        "scoring_weights": {
            "ml": 0.70, "spread": 0.15, "news": 0.15,
        },
        "eq_enabled": True,
        "daily_loss_pct": 5.0,
        "trade_journal": True,
        "discord_url": None,
        "scale_out_breakeven_fraction": 0.0,
    }
