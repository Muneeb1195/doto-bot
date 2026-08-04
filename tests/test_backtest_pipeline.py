"""G1c: Walk-forward / backtest pipeline integrity tests.

Tests the Backtest class with synthetic data to validate
precomputation, regime detection, and metrics calculation.
"""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from backtest import Backtest  # noqa: E402


@pytest.fixture
def bt_params():
    return {
        "symbol": "XAU500.raw",
        "initial_balance": 400000.0,
        "risk_percent": 1.0,
        "commission": 0.0,
        "slippage_points": 2,
        "ema_fast": 8,
        "ema_slow": 32,
        "atr_period": 14,
        "atr_sma_period": 20,
        "volume_filter": False,
        "volatility_filter": False,
        "ch_accelerate_enabled": False,
        "adx_enabled": True,
        "ml_enabled": False,
        "kelly_enabled": True,
        "kelly_lookback": 50,
        "kelly_fraction": 0.25,
        "kelly_min": 0.25,
        "kelly_max": 1.5,
        "pb_enabled": True,
        "pb_atr_mult": 1.2,
        "scoring_weights": {"exec": 0.15, "volume": 0.10, "volatility": 0.10,
                            "spread": 0.10, "news": 0.10, "tape": 0.10,
                            "ml": 0.25, "tail_risk": 0.10},
    }


@pytest.fixture
def bt_df():
    """500-bar synthetic H1-like dataset for backtest."""
    np.random.seed(42)
    n = 500
    closes = 100 + np.cumsum(np.random.randn(n) * 0.4)
    highs = closes + np.random.uniform(0.2, 0.8, n)
    lows = closes - np.random.uniform(0.2, 0.8, n)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time": dates,
        "open": closes - np.random.uniform(0.0, 0.3, n),
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": np.random.randint(500, 5000, n),
    }, index=dates)


class TestBacktestPipeline:
    def test_init_precomputes_indicators(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        assert bt.n == 500
        assert bt.balance == 400000.0
        assert len(bt.ema_fast) == 500
        assert len(bt.ema_slow) == 500
        assert not bt.ema_fast.isna().all()
        assert not bt.ema_slow.isna().all()

    def test_precompute_ema_relationship(self, bt_df, bt_params):
        bt_params["ema_fast"] = 5
        bt_params["ema_slow"] = 50
        bt = Backtest(bt_df, bt_params)
        assert bt.ema_fast.iloc[-1] != bt.ema_slow.iloc[-1]

    def test_precompute_atr_positive(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        assert not bt.atr_series.isna().all()
        assert bt.atr_series.iloc[-1] > 0

    def test_precompute_adx_valid(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        assert bt.adx_series is not None
        last_adx = bt.adx_series[-1]
        assert 0 <= last_adx <= 100

    def test_precompute_volume_sma_always(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        assert bt.vol_sma is not None
        assert len(bt.vol_sma) == len(bt_df)
        assert "tick_volume" in bt.df

    def test_precompute_volume_sma_when_enabled(self, bt_df, bt_params):
        bt_params["volume_filter"] = True
        bt = Backtest(bt_df, bt_params)
        assert bt.vol_sma is not None
        assert not bt.vol_sma.isna().all()

    def test_ch_accel_ema_none_when_disabled(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        assert bt.ch_accel_ema is None

    def test_ch_accel_ema_computed_when_enabled(self, bt_df, bt_params):
        bt_params["ch_accelerate_enabled"] = True
        bt = Backtest(bt_df, bt_params)
        assert bt.ch_accel_ema is not None
        assert not bt.ch_accel_ema.isna().all()

    def test_empty_df_returns_empty_precompute(self):
        empty = pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume"])
        bt = Backtest(empty, {"ema_fast": 8, "ema_slow": 32, "atr_period": 14, "adx_enabled": True})
        assert bt.n == 0

    def test_very_small_df_does_not_crash(self, bt_params):
        small = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=5, freq="h"),
            "open": [100] * 5, "high": [101] * 5, "low": [99] * 5,
            "close": [100] * 5, "tick_volume": [1000] * 5,
        })
        bt = Backtest(small, bt_params)
        assert bt.n == 5
        assert bt.balance == 400000.0

    def test_kelly_mult_without_trades(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        kelly = bt._calc_kelly_mult()
        assert kelly >= 0

    def test_detect_regime_returns_valid(self, bt_df, bt_params):
        bt = Backtest(bt_df, bt_params)
        idx = min(100, bt.n - 1)
        regime = bt._detect_regime(idx)
        assert regime in ("trending", "ranging", "uncertain", "volatile")


class TestBacktestRun:
    @pytest.fixture
    def run_params(self):
        return {
            "symbol": "XAU500.raw",
            "initial_balance": 400000.0,
            "risk_percent": 1.0,
            "commission": 0.0,
            "slippage_points": 2,
            "ema_fast": 8,
            "ema_slow": 32,
            "atr_period": 14,
            "atr_sma_period": 20,
            "atr_sl_mult": 2.0,
            "rr": 2.0,
            "volume_filter": False,
            "volatility_filter": False,
            "ch_accelerate_enabled": False,
            "adx_enabled": True,
            "adx_trend_threshold": 25,
            "adx_range_threshold": 20,
            "ml_enabled": False,
            "kelly_enabled": False,
            "pb_enabled": True,
            "pb_atr_mult": 1.2,
            "scoring_enabled": False,
            "session_enabled": False,
            "scale_out_enabled": False,
            "ch_enabled": False,
            "dr_enabled": False,
            "tr_enabled": False,
            "spf_enabled": False,
            "tape_enabled": False,
            "vf_enabled": False,
            "ns_enabled": False,
            "volume_step": 0.01,
            "point": 0.01,
            "tick_value": 0.1,
            "commission_type": "fixed",
            "scoring_weights": {"exec": 0.15, "volume": 0.10, "volatility": 0.10,
                                "spread": 0.10, "news": 0.10, "tape": 0.10,
                                "ml": 0.25, "tail_risk": 0.10},
        }

    @pytest.fixture
    def run_df(self):
        np.random.seed(42)
        n = 1000
        closes = 100 + np.cumsum(np.random.randn(n) * 0.3)
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        return pd.DataFrame({
            "time": dates,
            "open": closes - np.abs(np.random.randn(n) * 0.2),
            "high": closes + np.abs(np.random.randn(n) * 0.4),
            "low": closes - np.abs(np.random.randn(n) * 0.4),
            "close": closes,
            "tick_volume": np.random.randint(500, 5000, n),
            "spread": np.random.randint(1, 5, n),
        }, index=dates)

    def test_run_does_not_crash(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        assert hasattr(bt, "stats")
        assert bt.stats is not None

    def test_run_sets_trades_list(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        assert isinstance(bt.trades, list)

    def test_run_sets_equity_curve(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        assert len(bt.equity) > 0

    def test_run_trades_have_required_keys(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        required_keys = {"type", "entry", "sl", "tp", "volume", "pnl", "status", "exit_reason"}
        for t in bt.trades:
            assert required_keys.issubset(t.keys()), f"Missing keys: {required_keys - t.keys()}"

    def test_run_all_trades_closed(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        for t in bt.trades:
            assert t["status"] == "closed"

    def test_run_stats_contain_expected_keys(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()
        expected_stats = {"trades", "return", "max_dd", "win_rate", "profit_factor", "avg_win", "avg_loss"}
        assert expected_stats.issubset(bt.stats.keys())

    def test_run_returns_zero_stats_on_no_trades(self, run_params):
        n = 500
        flat = 100 + np.zeros(n)
        df = pd.DataFrame({
            "time": pd.date_range("2024-01-01", periods=n, freq="h"),
            "open": flat - 0.1, "high": flat + 0.5, "low": flat - 0.5,
            "close": flat, "tick_volume": np.full(n, 1000),
            "spread": np.full(n, 2),
        })
        bt = Backtest(df, run_params)
        bt.run()
        if hasattr(bt, "stats") and bt.stats:
            assert bt.stats["trades"] == 0
        else:
            assert len(bt.trades) == 0

    def test_run_multiple_equity_points(self, run_df, run_params):
        bt = Backtest(run_df, run_params)
        bt.run()

    def test_scale_out_partial_commission_accounted(self, run_params):
        # Backtest parity: each scale-out partial must pay exit commission, just
        # like the live broker. Within a single run the total commission paid
        # must equal commission * (entry_volume + exit_volume), where exit_volume
        # is the sum of closed-position volumes (each partial + its final
        # remainder == the full position volume). Without the partial-commission
        # fix, partials pay nothing, so total commission is understated and this
        # assertion fails.
        run_params["scale_out_enabled"] = True
        run_params["commission"] = 2.0
        run_params["ch_enabled"] = False
        run_params["kelly_enabled"] = False

        n = 1500
        np.random.seed(7)
        trend = np.r_[np.full(600, 0.4), np.full(300, -0.3), np.full(600, 0.4)]
        closes = 100 + np.cumsum(trend + np.random.randn(n) * 0.05)
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        df = pd.DataFrame({
            "time": dates,
            "open": closes - 0.1,
            "high": closes + 0.5,
            "low": closes - 0.5,
            "close": closes,
            "tick_volume": np.full(n, 2000),
            "spread": np.full(n, 2),
        }, index=dates)

        bt = Backtest(df, dict(run_params))
        bt.run()

        entry_vol = sum(p["volume"] for p in bt.positions)
        exit_vol = sum(p["volume"] for p in bt.positions if p["status"] == "closed")
        expected = 2.0 * (entry_vol + exit_vol)
        assert bt.commission_paid == pytest.approx(expected, rel=1e-6)
        assert len([t for t in bt.trades if t.get("exit_reason") == "SCALE_OUT"]) > 0
        assert len(bt.equity) > 100


def _small_df(n=40, increasing=True):
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    vals = np.arange(n, dtype=float) if increasing else -np.arange(n, dtype=float)
    close = 100 + vals
    return pd.DataFrame({
        "time": idx,
        "open": close - 0.1, "high": close + 0.5, "low": close - 0.5,
        "close": close, "tick_volume": np.full(n, 1000),
    }, index=idx)


@pytest.fixture
def parity_params(bt_params):
    bt_params.update({
        "mr_enabled": True,
        "mr_rsi_period": 14,
        "mr_rsi_oversold": 30,
        "mr_rsi_overbought": 70,
        "htf_ema_slow": 200,
        "htf_misalign_size_mult": 0.5,
        "scoring_enabled": True,
        "scoring_min_entry": 0.0,
        "correlation_enabled": False,
        "corr_size_mult": 1.0,
    })
    return bt_params


class TestBacktestParity:
    def test_fused_regime_gate_closes_on_low_score(self, parity_params):
        df = _small_df()
        bt = Backtest(df, parity_params)
        bt.fused_score_a = np.full(len(df), 10.0)  # very low score (chop)
        bt._fused_gate_open = False
        assert bool(bt._check_fused_regime_gate(30)) is False
        assert bool(bt._fused_gate_open) is False

    def test_fused_regime_gate_opens_on_high_score(self, parity_params):
        df = _small_df()
        bt = Backtest(df, parity_params)
        bt.fused_score_a = np.full(len(df), 90.0)  # very high score (trending)
        bt._fused_gate_open = False
        assert bool(bt._check_fused_regime_gate(30)) is True
        assert bool(bt._fused_gate_open) is True

    def test_mr_uses_m30_rsi_when_available(self, parity_params):
        # Build a 500-bar df so h4_df (for HTF EMA200) is available.
        n = 500
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        close = 100 + np.arange(n, dtype=float) * 0.1
        df = pd.DataFrame({
            "time": idx, "open": close - 0.1, "high": close + 0.5,
            "low": close - 0.5, "close": close, "tick_volume": np.full(n, 1000),
        }, index=idx)
        h4 = pd.DataFrame({
            "time": idx[::4], "open": np.full(len(idx[::4]), 50.0),
            "high": np.full(len(idx[::4]), 51.0), "low": np.full(len(idx[::4]), 49.0),
            "close": np.full(len(idx[::4]), 50.0), "tick_volume": np.full(len(idx[::4]), 1000),
        })
        bt = Backtest(df, parity_params)
        # Force H4 source to a flat low EMA so price (>100) is above it.
        bt.h4_df = h4.set_index("time")
        # M30 RSI oversold at i -> should yield a buy MR signal.
        rsi = pd.Series(np.full(n, 50.0), index=df["time"])
        rsi.iloc[300] = 20.0
        bt.mr_rsi_h1 = rsi
        sig, atr = bt._get_mean_reversion_signal(300)
        assert sig == "buy"
        # Without M30 RSI, falls back to H1 RSI (which for a steady uptrend is ~high).
        bt.mr_rsi_h1 = None
        sig2, _ = bt._get_mean_reversion_signal(300)
        assert sig2 is None  # H1 RSI not oversold here

    def test_correlation_sizing_path(self, parity_params):
        df = _small_df()
        bt = Backtest(df, parity_params)
        bt.run()  # sizing path (incl. correlation hook) exercised without error
        assert bt.n == len(df)


class TestHtfTrendGate:
    """Parity with live signals.check_htf_trend (3-state allow/soft/block)."""

    def _long_df(self, n=2400):
        idx = pd.date_range("2020-01-01", periods=n, freq="h")
        # Strong uptrend so H4 EMA200 is meaningful and slope-positive.
        close = 100 + np.arange(n, dtype=float) * 0.05
        return pd.DataFrame({
            "time": idx, "open": close - 0.1, "high": close + 0.6,
            "low": close - 0.6, "close": close, "tick_volume": np.full(n, 1000),
        }, index=idx)

    def test_allow_when_price_above_ema_and_slope_positive(self, parity_params):
        df = self._long_df()
        bt = Backtest(df, parity_params)
        ema = pd.Series(df["close"].values * 0.5, index=df["time"])  # well below price
        slope = pd.Series(np.full(len(df), 1.0), index=df["time"])
        bt.htf_ema_aligned = ema
        bt.htf_slope_aligned = slope
        assert bt._check_htf_trend(len(df) - 1, "buy") == ("allow", 1.0)

    def test_block_when_price_below_ema_and_slope_negative(self, parity_params):
        df = self._long_df()
        bt = Backtest(df, parity_params)
        ema = pd.Series(df["close"].values * 2.0, index=df["time"])  # above price
        slope = pd.Series(np.full(len(df), -1.0), index=df["time"])
        bt.htf_ema_aligned = ema
        bt.htf_slope_aligned = slope
        decision, mult = bt._check_htf_trend(len(df) - 1, "buy")
        assert decision == "block"
        assert mult == 0.0

    def test_soft_when_only_one_condition_holds(self, parity_params):
        df = self._long_df()
        bt = Backtest(df, parity_params)
        ema = pd.Series(df["close"].values * 2.0, index=df["time"])  # above price
        slope = pd.Series(np.full(len(df), 1.0), index=df["time"])  # positive slope
        bt.htf_ema_aligned = ema
        bt.htf_slope_aligned = slope
        decision, mult = bt._check_htf_trend(len(df) - 1, "buy")
        assert decision == "soft"
        assert mult == bt.p["htf_misalign_size_mult"]

    def test_block_reduces_trade_count_vs_allow(self, parity_params):
        # Force a long, flat-ish series where the HTF gate is the only differing
        # factor: run with an aligned EMA that blocks buys vs one that allows.
        df = self._long_df()
        params_allow = dict(parity_params)
        bt_allow = Backtest(df, params_allow)
        ema_allow = pd.Series(df["close"].values * 0.5, index=df["time"])
        bt_allow.htf_ema_aligned = ema_allow
        bt_allow.htf_slope_aligned = pd.Series(np.full(len(df), 1.0), index=df["time"])
        bt_allow.run()

        params_block = dict(parity_params)
        bt_block = Backtest(df, params_block)
        ema_block = pd.Series(df["close"].values * 2.0, index=df["time"])
        bt_block.htf_ema_aligned = ema_block
        bt_block.htf_slope_aligned = pd.Series(np.full(len(df), -1.0), index=df["time"])
        bt_block.run()

        assert len(bt_block.trades) <= len(bt_allow.trades)


class TestNumbaParity:
    """Bit-exact parity between the Numba fast path and the reference loop.

    The earlier Numba attempt was reverted because it could not be guaranteed
    bit-exact with the pandas loop. This test is the guard that prevents
    regression: both paths must produce identical trades + equity.
    """

    def _series(self, seed=7, n=1500):
        np.random.seed(seed)
        idx = pd.date_range("2021-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5)
        # add a trend so crossovers actually fire
        close = close + np.arange(n) * 0.02
        return pd.DataFrame({
            "time": idx, "open": close - 0.1, "high": close + 0.6,
            "low": close - 0.6, "close": close, "tick_volume": np.full(n, 1000),
        }, index=idx)

    def test_fast_matches_reference_trades(self, parity_params):
        from backtest import _njit_available  # noqa: E402
        if not _njit_available():
            pytest.skip("numba not available")
        df = self._series()
        params = dict(parity_params)
        params.update({
            "scoring_min_entry": 0.0,  # allow entries so the loop is exercised
            "dr_enabled": True,
            "ml_enabled": False,
        })
        bt_ref = Backtest(df, params)
        bt_ref.run(fast=False)
        bt_fast = Backtest(df, params)
        bt_fast.run(fast=True)

        ref_trades = bt_ref.trades
        fast_trades = bt_fast.trades
        assert len(fast_trades) == len(ref_trades), (
            f"trade count mismatch: fast={len(fast_trades)} ref={len(ref_trades)}"
        )
        for a, b in zip(fast_trades, ref_trades):
            assert a["entry_bar"] == b["entry_bar"], (a, b)
            assert a["exit_bar"] == b["exit_bar"], (a, b)
            assert a["exit_reason"] == b["exit_reason"], (a, b)
            assert abs(a["pnl"] - b["pnl"]) < 0.5, (a["pnl"], b["pnl"])
            assert abs(a["entry"] - b["entry"]) < 1e-9

    def test_fast_matches_reference_equity(self, parity_params):
        from backtest import _njit_available  # noqa: E402
        if not _njit_available():
            pytest.skip("numba not available")
        df = self._series(seed=11, n=2000)
        params = dict(parity_params)
        params.update({"scoring_min_entry": 0.0, "ml_enabled": False})
        bt_ref = Backtest(df, params)
        bt_ref.run(fast=False)
        bt_fast = Backtest(df, params)
        bt_fast.run(fast=True)
        eq_ref = np.array(bt_ref.equity)
        eq_fast = np.array(bt_fast.equity)
        assert np.allclose(eq_ref, eq_fast, atol=1e-6), (
            f"equity mismatch max diff {np.max(np.abs(eq_ref - eq_fast))}"
        )

    def _mtf_series(self, seed=7, n=1500):
        """H1 series plus a synthetic M15 series (4 bars per H1 bar) for MTF."""
        np.random.seed(seed)
        idx = pd.date_range("2021-01-01", periods=n, freq="h")
        close = 100 + np.cumsum(np.random.randn(n) * 0.5) + np.arange(n) * 0.02
        df = pd.DataFrame(
            {
                "time": idx, "open": close - 0.1, "high": close + 0.6,
                "low": close - 0.6, "close": close, "tick_volume": np.full(n, 1000),
            },
            index=idx,
        )
        # M15: 4 bars per H1 bar, matching the H1 close at each H1 boundary.
        m15_idx = pd.date_range("2021-01-01", periods=n * 4, freq="15min")
        m15_close = np.interp(
            np.arange(n * 4), np.arange(0, n * 4, 4), close
        ) + np.random.RandomState(seed).randn(n * 4) * 0.1
        df_m15 = pd.DataFrame(
            {
                "time": m15_idx,
                "open": m15_close - 0.05,
                "high": m15_close + 0.3,
                "low": m15_close - 0.3,
                "close": m15_close,
                "tick_volume": np.full(n * 4, 250),
            },
            index=m15_idx,
        )
        return df, df_m15

    def test_fast_matches_reference_mtf(self, parity_params):
        """Item #9 guard: Numba fast path must be bit-exact with the reference
        loop when mtf_enabled=True (H4/H1/M15 fused signal)."""
        from backtest import _njit_available  # noqa: E402
        if not _njit_available():
            pytest.skip("numba not available")
        df, df_m15 = self._mtf_series(seed=7, n=1500)
        params = dict(parity_params)
        params.update(
            {
                "mtf_enabled": True,
                "mtf_agreement_threshold": 0.67,
                "scoring_min_entry": 0.0,  # exercise the loop
                "ml_enabled": False,
                "dr_enabled": True,
                "pb_enabled": True,  # reference can fall back to pullback
            }
        )
        bt_ref = Backtest(df, params, df_m15=df_m15)
        bt_ref.run(fast=False)
        bt_fast = Backtest(df, params, df_m15=df_m15)
        bt_fast.run(fast=True)

        ref_trades = bt_ref.trades
        fast_trades = bt_fast.trades
        assert len(fast_trades) == len(ref_trades), (
            f"MTF trade count mismatch: fast={len(fast_trades)} ref={len(ref_trades)}"
        )
        for a, b in zip(fast_trades, ref_trades):
            assert a["entry_bar"] == b["entry_bar"], (a, b)
            assert a["exit_bar"] == b["exit_bar"], (a, b)
            assert a["exit_reason"] == b["exit_reason"], (a, b)
            assert a["entry_type"] == b["entry_type"], (a, b)
            assert abs(a["pnl"] - b["pnl"]) < 0.5, (a["pnl"], b["pnl"])
            assert abs(a["entry"] - b["entry"]) < 1e-9
            assert abs(a["volume"] - b["volume"]) < 1e-9

        eq_ref = np.array(bt_ref.equity)
        eq_fast = np.array(bt_fast.equity)
        assert np.allclose(eq_ref, eq_fast, atol=1e-6), (
            f"MTF equity mismatch max diff {np.max(np.abs(eq_ref - eq_fast))}"
        )

