"""Smoke tests for backtest_njit.py — the Numba-JIT compiled fast path.

The njit path is expected to be bit-exact with the pandas path. These tests
verify it doesn't crash on valid inputs and produces consistent output shapes.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.slow
class TestBacktestNjitFast:
    """Requires Numba. Skip with -m 'not slow' if Numba is unavailable."""

    @pytest.fixture
    def df(self):
        n = 500
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n) * 0.5)
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0.1, 1.0, n),
            "low": closes - rng.uniform(0.1, 1.0, n),
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "spread": rng.randint(1, 10, n),
        })

    @pytest.fixture
    def cfg(self):
        return {
            "ema_fast": 8, "ema_slow": 32, "ma_type": "kama",
            "atr_period": 14, "atr_sl_mult": 1.5, "rr": 2.0,
            "adx_enabled": True, "adx_trend_threshold": 25, "adx_range_threshold": 20,
            "stops_level": 50, "ml_confidence": 0.40,
            "volume_filter": True, "volume_kappa": 1.2,
            "chandelier_enabled": True, "chandelier_mult": 3.0,
            "chandelier_mult_partial": 1.5, "chandelier_lookback": 14,
            "ch_two_stage": True, "ch_loose_mult": 3.5, "ch_tight_mult": 1.5,
            "ch_two_stage_min_r": 3.0,
            "scale_out_enabled": True, "scale_out_close_fractions": [0.20, 0.20],
            "scale_out_tp_targets_rr": [0.50, 0.75],
            "ml_enabled": False,
            "risk_percent": 1.0, "initial_balance": 100000.0,
            "spread_model": 1.0, "commission": 0.0,
            "skip_uncertain_exhaustion": True,
            "dr_enabled": False, "dr_vol_adjust": False,
            "max_positions_per_symbol": 1, "max_risk_ratio": 2.0,
            "spf_enabled": False,
            "daily_loss_pct": 5.0,
            "tr_enabled": False,
            "mr_enabled": False,
            "pb_enabled": False,
            "scoring_enabled": True, "scoring_min_entry": 0.0,
            "max_positions": 5,
            "htf_ema_slow": 200,
            "htf_misalign_size_mult": 0.5,
            "point": 0.01, "tick_value": 0.01, "volume_step": 0.01,
            "symbol": "TEST",
        }

    def test_fast_path_does_not_crash(self, df, cfg):
        from backtest import Backtest
        bt = Backtest(df, cfg)
        result = bt.run(fast=True)
        assert result is not None
        assert isinstance(result, dict)

    def test_fast_path_returns_equity_curve(self, df, cfg):
        from backtest import Backtest
        bt = Backtest(df, cfg)
        result = bt.run(fast=True)
        equity = result.get("equity", [])
        assert len(equity) > 0
        assert all(e > 0 for e in equity)

    def test_fast_path_trades_have_keys(self, df, cfg):
        from backtest import Backtest
        bt = Backtest(df, cfg)
        result = bt.run(fast=True)
        trades = result.get("trades", [])
        if trades:
            for key in ("pnl", "entry_bar", "exit_bar", "entry_type", "volume"):
                assert key in trades[0], f"Missing key: {key}"
