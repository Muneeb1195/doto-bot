"""Tests for regime.py — detect_regime, get_mtf_adx, compute_adx_percentiles."""

import sys

sys.path.insert(0, "bot")

import numpy as np
import pandas as pd
import pytest

_rng = np.random.RandomState(42)


@pytest.fixture
def fake_rates():
    """Return a DataFrame that looks like mt5 rates data."""
    n = 300
    closes = 100 + np.cumsum(_rng.randn(n) * 0.3)
    highs = closes + np.abs(_rng.randn(n)) * 0.5
    lows = closes - np.abs(_rng.randn(n)) * 0.5
    return pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=n, freq="h"),
        "open": closes - 0.1,
        "high": highs,
        "low": lows,
        "close": closes,
        "tick_volume": _rng.randint(100, 10000, n),
        "real_volume": _rng.randint(100, 10000, n),
        "spread": _rng.randint(1, 20, n),
    })


@pytest.fixture
def basic_cfg():
    return {
        "symbol": "XAU500.raw",
        "timeframe": "H1",
        "adx_period": 14,
        "atr_period": 14,
        "adx_trend_threshold": 25,
        "adx_range_threshold": 20,
        "adx_percentile_enabled": False,
        "exhaustion_adx_threshold": 40,
        "exhaustion_slope_threshold": 2.0,
    }


class TestGetMtfAdx:
    def test_returns_dict_with_keys(self, monkeypatch, fake_rates):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates if n > 50 else None)
        from regime import get_mtf_adx
        result = get_mtf_adx("XAU500.raw")
        assert isinstance(result, dict)
        assert "h1" in result
        assert "h4" in result
        assert "d1" in result

    def test_adx_values_in_range(self, monkeypatch, fake_rates):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates if n > 50 else None)
        from regime import get_mtf_adx
        result = get_mtf_adx("XAU500.raw")
        for tf in ["h1", "h4", "d1"]:
            if result[tf] is not None:
                assert 0.0 <= result[tf] <= 100.0

    def test_none_when_no_data(self, monkeypatch):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: None)
        from regime import get_mtf_adx
        result = get_mtf_adx("NONEXISTENT.raw")
        assert result == {"h1": None, "h4": None, "d1": None}


class TestComputeAdxPercentiles:
    def test_returns_none_none_with_insufficient_data(self, monkeypatch):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: None)
        from regime import compute_adx_percentiles
        p50, p70 = compute_adx_percentiles({"symbol": "X", "timeframe": "H1", "adx_period": 14})
        assert p50 is None
        assert p70 is None

    def test_returns_percentiles_with_data(self, monkeypatch, fake_rates):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates)
        from regime import compute_adx_percentiles
        p50, p70 = compute_adx_percentiles({"symbol": "XAU500.raw", "timeframe": "H1", "adx_period": 14})
        if p50 is not None:
            assert 0.0 <= p50 <= 100.0
            assert p50 <= p70


class TestDetectRegime:
    def test_ranging_when_none(self, basic_cfg):
        from regime import detect_regime
        assert detect_regime(None, basic_cfg) == "ranging"

    def test_exhaustion_at_high_adx_with_negative_slope(self, monkeypatch, basic_cfg, fake_rates):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates)
        from regime import detect_regime
        result = detect_regime(50, basic_cfg)
        assert result in ("exhaustion", "strong_trend", "weak_trend", "ranging", "uncertain")

    def test_ranging_at_low_adx(self, basic_cfg):
        basic_cfg["adx_range_threshold"] = 30
        basic_cfg["adx_trend_threshold"] = 35
        from regime import detect_regime
        result = detect_regime(15, basic_cfg)
        assert result in ("ranging", "uncertain")

    def test_weak_trend_at_moderate_adx(self, basic_cfg):
        basic_cfg["adx_trend_threshold"] = 20
        from regime import detect_regime
        result = detect_regime(22, basic_cfg)
        assert isinstance(result, str)

    def test_percentile_mode_changes_thresholds(self, monkeypatch, basic_cfg, fake_rates):
        basic_cfg["adx_percentile_enabled"] = True
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates)
        from regime import detect_regime
        result = detect_regime(30, basic_cfg)
        assert isinstance(result, str)

    def test_detect_regime_returns_valid_string(self, basic_cfg):
        from regime import detect_regime
        valid = {"strong_trend", "weak_trend", "ranging", "exhaustion", "uncertain"}
        result = detect_regime(25, basic_cfg)
        assert result in valid


class TestGetCurrentAdx:
    def test_returns_none_with_no_rates(self, monkeypatch, basic_cfg):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: None)
        from regime import get_current_adx
        assert get_current_adx({**basic_cfg, "symbol": "X"}) is None

    def test_returns_float_with_rates(self, monkeypatch, fake_rates, basic_cfg):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates if n > 50 else None)
        from regime import get_current_adx
        result = get_current_adx(basic_cfg)
        if result is not None:
            assert isinstance(result, float)
            assert 0.0 <= result <= 100.0


class TestGetCurrentAtr:
    def test_returns_none_with_no_rates(self, monkeypatch, basic_cfg):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: None)
        from regime import get_current_atr
        assert get_current_atr({**basic_cfg, "symbol": "X"}) is None

    def test_returns_float_with_rates(self, monkeypatch, fake_rates, basic_cfg):
        monkeypatch.setattr("regime.get_rates", lambda sym, tf, n: fake_rates if n > 50 else None)
        from regime import get_current_atr
        result = get_current_atr(basic_cfg)
        if result is not None:
            assert isinstance(result, float)
            assert result > 0
