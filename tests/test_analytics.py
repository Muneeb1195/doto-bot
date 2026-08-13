"""Tests for analytics.py — the single source of truth for signal/backtest math."""

import numpy as np
import pandas as pd
import pytest


class TestClosedBars:
    def test_drops_last_bar(self):
        from analytics import closed_bars
        df = pd.DataFrame({"close": [1, 2, 3, 4, 5]})
        result = closed_bars(df)
        assert len(result) == 4
        assert list(result["close"]) == [1, 2, 3, 4]

    def test_none_returns_none(self):
        from analytics import closed_bars
        assert closed_bars(None) is None

    def test_single_bar_unchanged(self):
        from analytics import closed_bars
        df = pd.DataFrame({"close": [42]})
        result = closed_bars(df)
        assert len(result) == 1

    def test_empty_df_unchanged(self):
        from analytics import closed_bars
        df = pd.DataFrame()
        result = closed_bars(df)
        assert len(result) == 0


class TestFusedRegimeScore:
    def test_none_df_returns_zero(self):
        from analytics import fused_regime_score
        assert fused_regime_score(None, {}) == 0.0

    def test_short_df_returns_zero(self):
        from analytics import fused_regime_score
        df = pd.DataFrame({"close": [100, 101]})
        assert fused_regime_score(df, {"ema_fast": 5, "atr_period": 14}) == 0.0

    def test_known_input(self):
        from analytics import fused_regime_score
        n = 200
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "close": 100 + np.cumsum(rng.randn(n) * 0.5),
            "high": 100 + np.cumsum(rng.randn(n) * 0.5) + 0.5,
            "low": 100 + np.cumsum(rng.randn(n) * 0.5) - 0.5,
        })
        cfg = {"ema_fast": 8, "atr_period": 14, "ma_type": "kama"}
        score = fused_regime_score(df, cfg)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0

    def test_missing_ma_type_defaults_kama(self):
        from analytics import fused_regime_score
        n = 200
        rng = np.random.RandomState(7)
        df = pd.DataFrame({
            "close": 100 + np.cumsum(rng.randn(n) * 0.5),
            "high": 100 + np.cumsum(rng.randn(n) * 0.5) + 0.5,
            "low": 100 + np.cumsum(rng.randn(n) * 0.5) - 0.5,
        })
        cfg = {"ema_fast": 12, "atr_period": 14}
        score = fused_regime_score(df, cfg)
        assert isinstance(score, float)


class TestVolumeFilterPass:
    @pytest.fixture
    def df(self):
        n = 100
        rng = np.random.RandomState(1)
        return pd.DataFrame({
            "close": 100 + np.cumsum(rng.randn(n)) * 0.1,
            "tick_volume": np.full(n, 1000),
        })

    def test_disabled_returns_true(self, df):
        from analytics import volume_filter_pass
        cfg = {"vf_enabled": False}
        assert volume_filter_pass(df, "buy", cfg) is True

    def test_volume_above_kappa_returns_true(self, df):
        from analytics import volume_filter_pass
        df.loc[df.index[-1], "tick_volume"] = 50000
        cfg = {"vf_enabled": True, "volume_filter": True, "vf_kappa": 1.2}
        assert volume_filter_pass(df, "buy", cfg) is True

    def test_low_volume_no_obv_returns_false(self, df):
        from analytics import volume_filter_pass
        df.loc[df.index[-1], "tick_volume"] = 1
        cfg = {"vf_enabled": True, "volume_filter": True, "vf_kappa": 1.2, "vf_obv_enabled": False}
        result = volume_filter_pass(df, "buy", cfg)
        assert result is False

    def test_none_df_returns_true(self):
        from analytics import volume_filter_pass
        assert volume_filter_pass(None, "buy", {}) is True

    def test_short_df_returns_true(self):
        from analytics import volume_filter_pass
        df = pd.DataFrame({"close": [1, 2], "tick_volume": [10, 10]})
        assert volume_filter_pass(df, "buy", {"vf_enabled": True, "volume_filter": True}) is True

    def test_vf_kappa_fallback_to_volume_kappa(self, df):
        from analytics import volume_filter_pass
        df.loc[df.index[-1], "tick_volume"] = 50000
        cfg = {"vf_enabled": True, "volume_filter": True, "volume_kappa": 1.2}
        assert volume_filter_pass(df, "buy", cfg) is True

    def test_obv_bullish_divergence_passes(self):
        from analytics import volume_filter_pass
        n = 50
        close = np.full(n, 100.0)
        close[-5:] = [99, 98, 97, 96, 97]
        vol = np.full(n, 100)
        vol[-5:] = [50, 60, 70, 80, 200]
        df = pd.DataFrame({"close": close, "tick_volume": vol})
        cfg = {"vf_enabled": True, "volume_filter": True, "vf_kappa": 2.0, "vf_obv_lookback": 10}
        assert bool(volume_filter_pass(df, "buy", cfg)) is True

    def test_obv_bearish_divergence_passes(self):
        from analytics import volume_filter_pass
        n = 50
        close = np.full(n, 100.0)
        close[-5:] = [101, 102, 103, 104, 103]
        vol = np.full(n, 100)
        vol[-5:] = [50, 60, 70, 80, 200]
        df = pd.DataFrame({"close": close, "tick_volume": vol})
        cfg = {"vf_enabled": True, "volume_filter": True, "vf_kappa": 2.0, "vf_obv_lookback": 10}
        assert bool(volume_filter_pass(df, "sell", cfg)) is True


class TestObvDivergence:
    @pytest.fixture
    def df_flat(self):
        n = 30
        return pd.DataFrame({
            "close": np.full(n, 100.0),
            "tick_volume": np.full(n, 100),
        })

    def test_none_returns_false(self):
        from analytics import _obv_divergence
        assert _obv_divergence(None, "buy") is False

    def test_short_df_returns_false(self):
        from analytics import _obv_divergence
        df = pd.DataFrame({"close": [1, 2], "tick_volume": [10, 10]})
        assert _obv_divergence(df, "buy") is False

    def test_bullish_divergence(self):
        from analytics import _obv_divergence
        close = np.array([100, 99, 98, 97, 96, 97, 100])
        vol = np.array([100, 50, 60, 70, 80, 200, 300])
        df = pd.DataFrame({"close": close, "tick_volume": vol})
        assert bool(_obv_divergence(df, "buy", lookback=10)) is True

    def test_bearish_divergence(self):
        from analytics import _obv_divergence
        close = np.array([100, 101, 102, 103, 104, 103, 100])
        vol = np.array([100, 50, 60, 70, 80, 200, 300])
        df = pd.DataFrame({"close": close, "tick_volume": vol})
        assert bool(_obv_divergence(df, "sell", lookback=10)) is True

    def test_no_divergence_returns_false(self, df_flat):
        from analytics import _obv_divergence
        assert _obv_divergence(df_flat, "buy") is False
        assert _obv_divergence(df_flat, "sell") is False

    def test_invalid_signal_returns_false(self):
        from analytics import _obv_divergence
        df = pd.DataFrame({"close": [1, 2, 3], "tick_volume": [10, 10, 10]})
        assert _obv_divergence(df, "hold") is False


class TestApplyNewsConfidenceMult:
    def test_high_news_boosts(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(1.0, 0.80) == pytest.approx(1.10)

    def test_high_news_capped_at_1_5(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(1.4, 0.80) == pytest.approx(1.5)

    def test_low_news_halves(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(1.0, 0.20) == pytest.approx(0.5)

    def test_neutral_news_unchanged(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(0.85, 0.5) == pytest.approx(0.85)

    def test_none_news_unchanged(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(0.85, None) == pytest.approx(0.85)

    def test_boundary_0_70_is_boost(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(1.0, 0.70) == pytest.approx(1.10)

    def test_boundary_0_30_is_half(self):
        from analytics import apply_news_confidence_mult
        assert apply_news_confidence_mult(1.0, 0.30) == pytest.approx(0.5)


class TestComputeEntryScore:
    def test_ml_disabled_uses_fallback(self):
        from analytics import compute_entry_score
        cfg = {"symbol": "TEST", "ml_enabled": False, "spf_enabled": False}
        score, details, ml_conf = compute_entry_score(cfg, "buy", 0.05)
        assert score >= 0
        assert isinstance(details, dict)
        assert ml_conf is None

    def test_spread_passed_directly_no_mt5(self):
        from analytics import compute_entry_score
        cfg = {
            "symbol": "TEST", "ml_enabled": False, "spf_enabled": True,
            "spf_max_ratio": 0.30,
        }
        score, details, _ = compute_entry_score(cfg, "buy", atr=0.10, spread=0.005)
        assert "spread" in details
        assert 0.0 <= details["spread"] <= 1.0
        # spread=0.005, atr=0.10 => ratio=0.05, score=1 - 0.05/0.30 = 0.833...
        assert abs(details["spread"] - (1.0 - 0.05 / 0.30)) < 1e-9

    def test_spread_zero_atr_fallback_05(self):
        from analytics import compute_entry_score
        cfg = {
            "symbol": "TEST", "ml_enabled": False, "spf_enabled": True,
            "atr_period": 14, "timeframe": "H1",
        }
        import numpy as np
        atr_nan = np.nan
        score, details, _ = compute_entry_score(cfg, "buy", atr=atr_nan, spread=0.005)
        assert details["spread"] == 0.5

    def test_spf_disabled_returns_05(self):
        from analytics import compute_entry_score
        cfg = {"symbol": "TEST", "ml_enabled": False, "spf_enabled": False}
        score, details, _ = compute_entry_score(cfg, "buy", atr=0.10)
        assert details["spread"] == 0.5

    def test_weighted_average_correct(self):
        from analytics import compute_entry_score
        cfg = {
            "symbol": "TEST", "ml_enabled": False, "spf_enabled": True,
            "spf_max_ratio": 0.30,
            "scoring_weights": {"ml": 0.40, "spread": 0.30, "news": 0.30},
        }
        score, details, _ = compute_entry_score(cfg, "buy", atr=0.10, spread=0.005)
        # ml=0.6 (fallback), spread=0.833..., news=0.5
        expected = 0.6 * 0.40 + (1.0 - 0.05 / 0.30) * 0.30 + 0.5 * 0.30
        assert abs(score - expected) < 1e-9
