"""Tests for correlation.py — compute_correlation_matrix, get_correlation_reduction."""


import numpy as np
import pandas as pd
import pytest


class FakeRates:
    """Simulate what mt5.copy_rates_from_pos returns — tuple of dict-like objects."""

    @staticmethod
    def make(close_trend, n=48):
        data = []
        for i in range(n):
            noise = float(np.random.randn() * 0.5)
            data.append({
                "time": i * 3600,
                "open": close_trend[i] + noise,
                "high": close_trend[i] + abs(noise) + 0.1,
                "low": close_trend[i] - abs(noise) - 0.1,
                "close": close_trend[i],
                "tick_volume": 1000 + int(abs(noise) * 100),
                "spread": 10,
                "real_volume": 1000 + int(abs(noise) * 100),
            })
        return tuple(data)


@pytest.fixture
def strong_corr_pair():
    """Create two highly correlated price series."""
    n = 48
    base = np.cumsum(np.random.randn(n) * 0.5) + 100
    sym_a = base + np.random.randn(n) * 0.05
    sym_b = base + np.random.randn(n) * 0.05
    return {"SYM_A": sym_a, "SYM_B": sym_b}


@pytest.fixture
def uncorrelated_pair():
    """Two uncorrelated price series."""
    n = 48
    a = np.cumsum(np.random.randn(n) * 0.5) + 100
    b = np.cumsum(np.random.randn(n) * 0.5) + 100
    return {"SYM_C": a, "SYM_D": b}


def make_returns_dict(price_dict):
    """Build a returns dict for compute_correlation_matrix."""
    result = {}
    for sym, prices in price_dict.items():
        df = pd.DataFrame({"close": prices})
        result[sym] = df["close"].pct_change()
    return result


class TestComputeCorrelationMatrix:
    def test_returns_empty_for_single_symbol(self, monkeypatch):
        monkeypatch.setattr("correlation.fetch_returns_for_symbols", lambda s, *a, **kw: {s[0]: None})
        from correlation import compute_correlation_matrix
        result = compute_correlation_matrix(["SOLO.raw"])
        assert result == {}

    def test_returns_correlation_for_two_symbols(self, strong_corr_pair):
        from correlation import compute_correlation_matrix
        r = make_returns_dict(strong_corr_pair)
        syms = list(strong_corr_pair.keys())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("correlation.fetch_returns_for_symbols", lambda *a, **kw: r)
            result = compute_correlation_matrix(syms)
        assert len(result) == 1
        pair = list(result.keys())[0]
        corr = result[pair]
        assert isinstance(corr, float)
        assert -1.0 <= corr <= 1.0
        assert corr > 0.5  # strongly correlated

    def test_returns_low_corr_for_uncorrelated(self, uncorrelated_pair):
        from correlation import compute_correlation_matrix
        r = make_returns_dict(uncorrelated_pair)
        syms = list(uncorrelated_pair.keys())
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("correlation.fetch_returns_for_symbols", lambda *a, **kw: r)
            result = compute_correlation_matrix(syms)
        if result:
            pair = list(result.keys())[0]
            corr = result[pair]
            assert abs(corr) < 0.5

    def test_keys_are_sorted_pairs(self):
        from correlation import compute_correlation_matrix
        price_a = np.array([100.0, 101.0, 102.0, 101.5, 100.5])
        price_b = np.array([100.0, 99.5, 99.0, 98.5, 99.0])
        r = {
            "A.raw": pd.Series((price_a / price_a[0]) - 1),
            "B.raw": pd.Series((price_b / price_b[0]) - 1),
        }
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr("correlation.fetch_returns_for_symbols", lambda *a, **kw: r)
            result = compute_correlation_matrix(["A.raw", "B.raw"])
        assert ("A.raw", "B.raw") in result
        assert ("B.raw", "A.raw") not in result


class TestGetCorrelationReduction:
    def test_returns_one_with_empty_matrix(self):
        from correlation import get_correlation_reduction
        assert get_correlation_reduction({}, "X.raw", []) == 1.0

    def test_returns_one_when_correlation_below_threshold(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): 0.3}
        assert get_correlation_reduction(mat, "X.raw", ["Y.raw"]) == 1.0

    def test_reduces_at_high_correlation(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): 0.9}
        result = get_correlation_reduction(mat, "X.raw", ["Y.raw"])
        assert result < 1.0
        assert result >= 0.5

    def test_no_reduction_when_not_in_matrix(self):
        from correlation import get_correlation_reduction
        mat = {("A.raw", "B.raw"): 0.9}
        assert get_correlation_reduction(mat, "X.raw", ["Y.raw"]) == 1.0

    def test_handles_nan_in_matrix(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): float("nan")}
        assert get_correlation_reduction(mat, "X.raw", ["Y.raw"]) == 1.0

    def test_max_reduction_respected(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): 1.0}
        result = get_correlation_reduction(mat, "X.raw", ["Y.raw"], max_reduction=0.5)
        assert result == 0.5

    def test_reduction_linear_decrease(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): 0.75}
        result = get_correlation_reduction(mat, "X.raw", ["Y.raw"], max_reduction=0.5)
        expected = 1.0 - ((0.75 - 0.5) / 0.5) * 0.5
        assert abs(result - expected) < 0.001

    def test_multiple_existing_symbols_takes_max(self):
        from correlation import get_correlation_reduction
        mat = {("X.raw", "Y.raw"): 0.6, ("X.raw", "Z.raw"): 0.9}
        result = get_correlation_reduction(mat, "X.raw", ["Y.raw", "Z.raw"])
        # max_corr=0.9 → reduction = 1 - ((0.9-0.5)/0.5)*0.5 = 0.6
        assert result == 0.6
