"""Tests for indicators.py — all pure pandas/numpy functions."""

import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, "bot")

from indicators import (
    calc_adx,
    calc_adx_series,
    calc_atr,
    calc_atr_series,
    calc_efficiency_ratio,
    calc_fused_regime_score,
    calc_kama,
    calc_ma,
    calc_ma_slope,
    calc_rsi,
    calc_vidya,
    rsi_series,
)


class TestCalcMA:
    def test_basic_kama(self, sample_df):
        result = calc_ma(sample_df, 14, "kama")
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_kama_trend_follows(self, trending_up_df):
        result = calc_ma(trending_up_df, 14, "kama")
        assert not result.isna().all()

    def test_zero_period(self, sample_df):
        result = calc_ma(sample_df, 0, "kama")
        assert result.empty

    def test_none_df(self):
        result = calc_ma(None, 14, "kama")
        assert result.empty

    def test_empty_df(self):
        result = calc_ma(pd.DataFrame(), 14, "kama")
        assert result.empty

    def test_period_greater_than_len(self, small_df):
        result = calc_ma(small_df, 100, "kama")
        assert result.empty

    def test_kama_all_positive(self, sample_df):
        result = calc_ma(sample_df, 14, "kama")
        valid = result.dropna()
        assert len(valid) > 0

    def test_constant_price(self):
        df = pd.DataFrame({"close": [100.0] * 50})
        result = calc_ma(df, 14, "kama")
        valid = result.dropna()
        assert np.allclose(valid, 100.0)


class TestCalcKAMA:
    def test_basic_kama(self, sample_df):
        result = calc_kama(sample_df, 14)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_zero_period(self, sample_df):
        result = calc_kama(sample_df, 0)
        assert result.empty

    def test_none_df(self):
        result = calc_kama(None, 14)
        assert result.empty

    def test_empty_df(self):
        result = calc_kama(pd.DataFrame(), 14)
        assert result.empty

    def test_period_greater_than_len(self, small_df):
        result = calc_kama(small_df, 100)
        assert result.empty

    def test_constant_price(self):
        df = pd.DataFrame({"close": [100.0] * 50})
        result = calc_kama(df, 14)
        assert np.allclose(result.dropna(), 100.0)


class TestCalcATR:
    def test_basic_atr(self, sample_df):
        result = calc_atr(sample_df, 14)
        assert result > 0
        assert isinstance(result, float)

    def test_zero_period(self, sample_df):
        result = calc_atr(sample_df, 0)
        assert result == 0.0

    def test_none_df(self):
        result = calc_atr(None, 14)
        assert result == 0.0

    def test_insufficient_data(self, small_df):
        result = calc_atr(small_df, 100)
        assert result == 0.0

    def test_higher_volatility_higher_atr(self):
        low_vol = pd.DataFrame({
            "high": [101, 102, 101, 102, 101],
            "low": [99, 100, 99, 100, 99],
            "close": [100, 101, 100, 101, 100],
        })
        high_vol = pd.DataFrame({
            "high": [105, 110, 108, 112, 107],
            "low": [95, 90, 92, 88, 93],
            "close": [100, 108, 95, 110, 105],
        })
        atr_low = calc_atr(low_vol, 3)
        atr_high = calc_atr(high_vol, 3)
        assert atr_high > atr_low

    def test_single_bar(self):
        df = pd.DataFrame({"high": [100], "low": [99], "close": [99.5]})
        result = calc_atr(df, 14)
        assert result == 0.0

    def test_non_nan_returns(self, sample_df):
        result = calc_atr(sample_df, 14)
        assert not np.isnan(result)
        assert np.isfinite(result)


class TestCalcATRSeries:
    def test_basic_atr_series(self, sample_df):
        result = calc_atr_series(sample_df, 14)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_zero_period(self, sample_df):
        result = calc_atr_series(sample_df, 0)
        assert result.isna().all()

    def test_none_df(self):
        result = calc_atr_series(None, 14)
        assert result.empty

    def test_series_end_values_match_atr(self, sample_df):
        series = calc_atr_series(sample_df, 14)
        scalar = calc_atr(sample_df, 14)
        assert abs(series.iloc[-1] - scalar) < 0.01

    def test_backtest_atr_matches_wilder(self):
        # H4 parity: the backtest vol-filter ATR (TR smoothed with Wilder ewm)
        # must agree with the live calc_atr_series Wilder implementation, so the
        # volatility filter and entry sizing use the same ATR (regression guard).
        rng = np.random.default_rng(0)
        n = 300
        close = np.cumsum(rng.normal(0, 1, n)) + 100
        high = close + np.abs(rng.normal(0, 0.5, n))
        low = close - np.abs(rng.normal(0, 0.5, n))
        df = pd.DataFrame({"high": high, "low": low, "close": close})
        period = 14
        tr = pd.DataFrame({
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift()).abs(),
            "lc": (df["low"] - df["close"].shift()).abs(),
        }).max(axis=1)
        atr_base = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        wilder = calc_atr_series(df, period)
        assert np.allclose(atr_base.iloc[-50:].values, wilder.iloc[-50:].values, rtol=1e-6)


class TestCalcRSI:
    def test_basic_rsi(self, sample_df):
        result = calc_rsi(sample_df, 14)
        assert 0 <= result <= 100

    def test_zero_period(self, sample_df):
        result = calc_rsi(sample_df, 0)
        assert result == 50.0

    def test_none_df(self):
        result = calc_rsi(None, 14)
        assert result == 50.0

    def test_insufficient_data(self, small_df):
        result = calc_rsi(small_df, 100)
        assert result == 50.0

    def test_up_trend_high_rsi(self, trending_up_df):
        result = calc_rsi(trending_up_df, 14)
        assert result > 50

    def test_constant_price_returns_100(self):
        df = pd.DataFrame({"close": [100.0] * 30})
        result = calc_rsi(df, 14)
        assert result == 100.0

    def test_always_up(self):
        df = pd.DataFrame({"close": np.linspace(100, 110, 30)})
        result = calc_rsi(df, 14)
        assert result == 100.0

    def test_always_down(self):
        df = pd.DataFrame({"close": np.linspace(100, 90, 30)})
        result = calc_rsi(df, 14)
        assert result == 0.0

    def test_rsi_wilder_full_history(self):
        # Strong early uptrend, then a long flat stretch, then a single drop.
        # Canonical Wilder RSI must retain the early trend's influence across the
        # FULL history (the pre-fix version only used the last ~15 bars and would
        # collapse toward 0 here). Regression guard for agent audit B2.
        rises = np.arange(15)                 # 0..14 (strong uptrend)
        flat = np.full(26, 14.0)              # long flat stretch
        drop = np.array([13.0])               # single down move
        close = np.concatenate([rises, flat, drop])
        result = calc_rsi(pd.DataFrame({"close": close}), 14)
        assert 55.0 < result < 75.0


class TestCalcADX:
    def test_basic_adx(self, sample_df):
        result = calc_adx(sample_df, 14)
        assert 0 <= result <= 100
        assert isinstance(result, float)

    def test_zero_period(self, sample_df):
        result = calc_adx(sample_df, 0)
        assert result == 0.0

    def test_none_df(self):
        result = calc_adx(None, 14)
        assert result == 0.0

    def test_trending_market_has_adx(self, trending_up_df):
        result = calc_adx(trending_up_df, 14)
        assert result > 0

    def test_insufficient_data(self, small_df):
        result = calc_adx(small_df, 100)
        assert result == 0.0


class TestCalcADXSeries:
    def test_basic_adx_series(self, sample_df):
        result = calc_adx_series(sample_df, 14)
        assert len(result) == len(sample_df)

    def test_end_value_matches_scalar(self, sample_df):
        series = calc_adx_series(sample_df, 14)
        scalar = calc_adx(sample_df, 14)
        assert abs(series[-1] - scalar) < 0.01


class TestRSISeries:
    def test_basic_rsi_series(self, sample_df):
        result = rsi_series(sample_df, 14)
        assert len(result) == len(sample_df)
        non_nan = result[~np.isnan(result)]
        assert len(non_nan) > 0
        assert (non_nan >= 0).all()
        assert (non_nan <= 100).all()

    def test_zero_period(self, sample_df):
        result = rsi_series(sample_df, 0)
        assert len(result) == 0

    def test_none_df(self):
        result = rsi_series(None, 14)
        assert len(result) == 0

    def test_end_value_bounded(self, sample_df):
        series = rsi_series(sample_df, 14)
        last_val = series[~np.isnan(series)][-1]
        assert 0 <= last_val <= 100


class TestVIDYA:
    def test_basic_vidya(self, sample_df):
        result = calc_vidya(sample_df, 14)
        assert len(result) == len(sample_df)
        assert not result.isna().all()

    def test_via_dispatch(self, sample_df):
        direct = calc_vidya(sample_df, 14)
        dispatched = calc_ma(sample_df, 14, "vidya")
        pd.testing.assert_series_equal(direct, dispatched)

    def test_tracks_trend(self, trending_up_df):
        result = calc_vidya(trending_up_df, 14).dropna()
        assert result.iloc[-1] > result.iloc[0]

    def test_within_price_range(self, sample_df):
        result = calc_vidya(sample_df, 14).dropna()
        assert result.min() >= sample_df["low"].min() - 5
        assert result.max() <= sample_df["high"].max() + 5


class TestEfficiencyRatio:
    def test_trending_high_er(self):
        close = np.arange(0, 20, dtype=float)
        er = calc_efficiency_ratio(close, 10)
        assert er == pytest.approx(1.0)

    def test_choppy_low_er(self):
        close = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        er = calc_efficiency_ratio(close, 10)
        assert er < 0.2

    def test_insufficient_data(self):
        assert calc_efficiency_ratio(np.array([1.0, 2.0]), 10) == 0.0

    def test_flat_zero_volatility(self):
        close = np.full(15, 5.0)
        assert calc_efficiency_ratio(close, 10) == 0.0

    def test_bounded_0_1(self, sample_df):
        er = calc_efficiency_ratio(sample_df["close"].values, 10)
        assert 0.0 <= er <= 1.0


class TestMaSlope:
    def test_rising(self):
        ma = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert calc_ma_slope(ma, period=1) > 0

    def test_falling(self):
        ma = pd.Series([4.0, 3.0, 2.0, 1.0])
        assert calc_ma_slope(ma, period=1) < 0

    def test_flat_zero(self):
        ma = pd.Series([5.0, 5.0, 5.0])
        assert calc_ma_slope(ma, period=1) == 0.0

    def test_insufficient(self):
        assert calc_ma_slope(pd.Series([1.0]), period=1) == 0.0

    def test_returns_raw_price_delta(self):
        # calc_ma_slope returns the RAW price delta (not a ratio vs start), so
        # that ma_change / atr in calc_fused_regime_score is dimensionless.
        ma = pd.Series([100.0, 110.0])
        assert calc_ma_slope(ma, period=1) == pytest.approx(10.0)

    def test_delta_is_scale_dependent_not_normalized(self):
        # A ratio-based slope would give both series the same value (0.10).
        # Raw deltas must differ, proving no start-relative normalization.
        small = calc_ma_slope(pd.Series([1.0, 1.1]), period=1)
        large = calc_ma_slope(pd.Series([100.0, 110.0]), period=1)
        assert small == pytest.approx(0.1)
        assert large == pytest.approx(10.0)
        assert large != pytest.approx(small)

    def test_multi_period_delta(self):
        ma = pd.Series([100.0, 102.0, 105.0])
        assert calc_ma_slope(ma, period=2) == pytest.approx(5.0)


class TestFusedRegimeScore:
    def test_strong_trend_high_score(self):
        score = calc_fused_regime_score(adx=50, er=1.0, ma_change=1.0, atr=1.0)
        assert score > 90

    def test_chop_low_score(self):
        score = calc_fused_regime_score(adx=5, er=0.05, ma_change=0.0, atr=1.0)
        assert score < 20

    def test_bounded_0_100(self):
        for adx in (0, 25, 60):
            for er in (0.0, 0.5, 1.5):
                s = calc_fused_regime_score(adx, er, 0.5, 1.0)
                assert 0.0 <= s <= 100.0

    def test_weights_sum_to_full(self):
        score = calc_fused_regime_score(adx=50, er=1.0, ma_change=10.0, atr=1.0)
        assert score == pytest.approx(100.0)

    def test_adx_dominant_weight(self):
        adx_only = calc_fused_regime_score(adx=50, er=0.0, ma_change=0.0, atr=1.0)
        er_only = calc_fused_regime_score(adx=0, er=1.0, ma_change=0.0, atr=1.0)
        assert adx_only > er_only

    def test_zero_atr_slope_ignored(self):
        score = calc_fused_regime_score(adx=0, er=0.0, ma_change=5.0, atr=0.0)
        assert score == 0.0

    # --- Regression guards for the fused-score slope unit mismatch ---
    # calc_ma_slope used to return a dimensionless ratio which was then divided
    # by ATR (a price), so the units did not cancel. The slope term therefore
    # scaled inversely with instrument price: ~0 on high-priced symbols and
    # saturated at the cap on low-priced FX. The live regime gate stayed shut
    # on ~100% of bars for 7 of 8 symbols and the bot took zero trades.
    # These tests fail against that old behaviour.

    def test_slope_term_actually_contributes(self):
        """A realistic price-unit slope must move the score materially."""
        flat = calc_fused_regime_score(adx=25, er=0.30, ma_change=0.0, atr=10.0)
        sloped = calc_fused_regime_score(adx=25, er=0.30, ma_change=5.0, atr=10.0)
        # 5/10 * SLOPE_SCALE(2.0) = 1.0 -> full 20 points of the slope weight.
        assert sloped - flat == pytest.approx(20.0)

    def test_slope_scales_with_atr_ratio(self):
        """Score must depend on slope/ATR, not on absolute price magnitude."""
        # Same ratio (0.1) at wildly different price scales -> identical score.
        cheap = calc_fused_regime_score(adx=20, er=0.2, ma_change=0.05, atr=0.5)
        rich = calc_fused_regime_score(adx=20, er=0.2, ma_change=25.0, atr=250.0)
        assert cheap == pytest.approx(rich)

    def test_degenerate_atr_contributes_zero_not_maximum(self):
        """A zero/NaN ATR must yield NO slope credit, never a full 20 points.

        The old implementation divided by max(atr, 1e-10), so a degenerate ATR
        saturated the slope term and handed the symbol a free 20/20 instead of
        discounting it.
        """
        baseline = calc_fused_regime_score(adx=30, er=0.4, ma_change=0.0, atr=5.0)
        for bad_atr in (0.0, -1.0, float("nan")):
            s = calc_fused_regime_score(adx=30, er=0.4, ma_change=5.0, atr=bad_atr)
            assert s == pytest.approx(baseline)

    def test_nan_ma_change_contributes_zero(self):
        baseline = calc_fused_regime_score(adx=30, er=0.4, ma_change=0.0, atr=5.0)
        s = calc_fused_regime_score(adx=30, er=0.4, ma_change=float("nan"), atr=5.0)
        assert s == pytest.approx(baseline)

    def test_end_to_end_slope_pipeline(self):
        """calc_ma_slope output must be directly consumable by the fused score.

        Guards the contract between the two functions: a trending MA on a
        realistic price scale has to produce a non-trivial slope contribution.
        """
        ma = pd.Series([2000.0, 2004.0, 2008.0])  # gold-like, +4 per bar
        slope = calc_ma_slope(ma, period=1)
        atr = 20.0
        with_slope = calc_fused_regime_score(adx=25, er=0.3, ma_change=slope, atr=atr)
        without = calc_fused_regime_score(adx=25, er=0.3, ma_change=0.0, atr=atr)
        assert with_slope > without
        # 4/20 * 2.0 = 0.4 -> 8 of the 20 slope points.
        assert with_slope - without == pytest.approx(8.0)
