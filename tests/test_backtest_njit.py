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


def _trade_fingerprint(trades):
    """Order-stable projection of a trade list used for parity comparison."""
    return [
        (
            t.get("entry_bar"),
            t.get("exit_bar"),
            t.get("entry_type"),
            t.get("exit_reason"),
            round(float(t.get("pnl", 0.0)), 6),
            round(float(t.get("volume", 0.0)), 8),
        )
        for t in trades
    ]


@pytest.mark.slow
class TestBacktestNjitParity:
    """The contract in backtest_njit.py's docstring: the JIT path must produce
    the SAME trades/equity as ``Backtest.run(fast=False)``.

    The smoke tests above only assert the fast path doesn't crash, and they run
    with ``pb_enabled: False`` — which is why a pullback-only divergence went
    unnoticed. These tests compare the two engines directly and keep the
    pullback path switched ON.
    """

    @pytest.fixture
    def df(self):
        # Longer + trendier than the smoke fixture so pullback/HTF paths engage.
        n = 1200
        rng = np.random.RandomState(7)
        drift = np.linspace(0, 25, n)
        closes = 100 + drift + np.cumsum(rng.randn(n) * 0.6)
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0.1, 1.2, n),
            "low": closes - rng.uniform(0.1, 1.2, n),
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "spread": rng.randint(1, 10, n),
        })

    @pytest.fixture
    def base_cfg(self):
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
            # Pullback ON — this is the path the smoke tests never exercise.
            "pb_enabled": True,
            "pb_atr_mult": 2.0, "pb_atr_min_dist": 0.1,
            "pb_volume_enabled": True, "pb_volume_sma_period": 20,
            "pb_volume_threshold": 0.8, "pb_structure_lookback": 5,
            "scoring_enabled": False, "scoring_min_entry": 0.0,
            "max_positions": 5,
            "htf_ema_slow": 200,
            "htf_misalign_size_mult": 0.5,
            "point": 0.01, "tick_value": 0.01, "volume_step": 0.01,
            "symbol": "TEST",
        }

    def _run_both(self, df, cfg):
        from backtest import Backtest

        fast = Backtest(df, cfg).run(fast=True)
        ref = Backtest(df, cfg).run(fast=False)
        return fast, ref

    def test_pullback_parity(self, df, base_cfg):
        """Pullback entries must match the reference exactly.

        Regression guard: the JIT applied the HTF check only via the shared
        downstream gate (which merely *resizes* on partial misalignment), while
        the reference vetoes the pullback outright when HTF blocks.
        """
        fast, ref = self._run_both(df, base_cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    def test_pullback_parity_htf_disabled(self, df, base_cfg):
        cfg = dict(base_cfg, htf_enabled=False)
        fast, ref = self._run_both(df, cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    def test_crossover_only_parity(self, df, base_cfg):
        cfg = dict(base_cfg, pb_enabled=False)
        fast, ref = self._run_both(df, cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    def test_parity_with_scoring_enabled(self, df, base_cfg):
        cfg = dict(base_cfg, scoring_enabled=True, scoring_min_entry=0.0)
        fast, ref = self._run_both(df, cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    def test_parity_with_mean_reversion(self, df, base_cfg):
        cfg = dict(base_cfg, mr_enabled=True, mr_rsi_period=14,
                   mr_rsi_oversold=30, mr_rsi_overbought=70)
        fast, ref = self._run_both(df, cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    @pytest.mark.parametrize("ema", [(6, 24), (10, 40), (12, 48)])
    def test_parity_across_ma_grid(self, df, base_cfg, ema):
        cfg = dict(base_cfg, ema_fast=ema[0], ema_slow=ema[1])
        fast, ref = self._run_both(df, cfg)
        assert _trade_fingerprint(fast["trades"]) == _trade_fingerprint(ref["trades"])

    def test_equity_curve_parity(self, df, base_cfg):
        fast, ref = self._run_both(df, base_cfg)
        fe = np.asarray(fast["equity"], dtype=float)
        re_ = np.asarray(ref["equity"], dtype=float)
        assert len(fe) == len(re_)
        assert np.allclose(fe, re_, rtol=1e-9, atol=1e-6)
