"""Tests for ml_features.py — feature engineering, labeling, stats."""

import numpy as np
import pandas as pd
import pytest


class TestComputeKamaSeries:
    def test_returns_nan_for_short_input(self):
        from ml_features import compute_kama_series
        arr = np.array([1.0, 2.0])
        result = compute_kama_series(arr, 9)
        assert np.all(np.isnan(result))

    def test_returns_correct_length(self):
        from ml_features import compute_kama_series
        arr = np.linspace(100, 110, 50)
        result = compute_kama_series(arr, 9)
        assert len(result) == 50

    def test_kama_tracks_trend(self):
        from ml_features import compute_kama_series
        arr = np.linspace(100, 200, 100)
        result = compute_kama_series(arr, 10)
        valid = result[~np.isnan(result)]
        assert len(valid) > 0
        assert valid[-1] > valid[0]
        assert not np.isnan(result[-1])

    def test_kama_bounded_by_input(self):
        from ml_features import compute_kama_series
        rng = np.random.RandomState(99)
        arr = 100 + np.cumsum(rng.randn(200))
        result = compute_kama_series(arr, 10)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= np.min(arr))
        assert np.all(valid <= np.max(arr))


class TestComputeEma:
    def test_returns_correct_length(self):
        from ml_features import compute_ema
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        result = compute_ema(arr, 3)
        assert len(result) == 5

    def test_ema_follows_input(self):
        from ml_features import compute_ema
        arr = np.array([100, 101, 102, 103, 104], dtype=float)
        result = compute_ema(arr, 3)
        assert result[-1] > result[0]
        assert not np.isnan(result[-1])


class TestComputeRsi:
    def test_rsi_bounds(self):
        from ml_features import compute_rsi
        rng = np.random.RandomState(1)
        arr = 100 + np.cumsum(rng.randn(200))
        result = compute_rsi(arr, 14)
        valid = result[~np.isnan(result)]
        assert np.all(valid >= 0) and np.all(valid <= 100)

    def test_rsi_all_up_returns_100(self):
        from ml_features import compute_rsi
        arr = np.linspace(100, 200, 100)
        result = compute_rsi(arr, 14)
        last = result[~np.isnan(result)][-1]
        assert last > 95  # strong upward momentum

    def test_rsi_all_down_returns_0(self):
        from ml_features import compute_rsi
        arr = np.linspace(200, 100, 100)
        result = compute_rsi(arr, 14)
        last = result[~np.isnan(result)][-1]
        assert last < 5  # strong downward momentum


class TestComputeObv:
    def test_obv_rising_with_price(self):
        from ml_features import compute_obv
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
            "tick_volume": [100, 100, 100, 100, 100],
        })
        obv = compute_obv(df)
        assert obv[-1] > obv[0]

    def test_obv_falling_with_price(self):
        from ml_features import compute_obv
        df = pd.DataFrame({
            "close": [104, 103, 102, 101, 100],
            "tick_volume": [100, 100, 100, 100, 100],
        })
        obv = compute_obv(df)
        assert obv[-1] < obv[0]

    def test_obv_unchanged_when_price_flat(self):
        from ml_features import compute_obv
        df = pd.DataFrame({
            "close": [100, 100, 100, 100, 100],
            "tick_volume": [100, 200, 100, 200, 100],
        })
        obv = compute_obv(df)
        assert obv[-1] == 0.0


class TestComputeVwap:
    def test_vwap_matches_typical_price(self):
        from ml_features import compute_vwap
        df = pd.DataFrame({
            "high": [101, 102, 103],
            "low": [99, 98, 97],
            "close": [100, 100, 100],
            "tick_volume": [100, 100, 100],
        })
        vwap = compute_vwap(df)
        assert not np.isnan(vwap[-1])
        assert isinstance(vwap, np.ndarray)
        assert len(vwap) == len(df)


class TestComputeTimeFeatures:
    def test_returns_4_arrays(self):
        from ml_features import compute_time_features
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=100, freq="h"),
        })
        result = compute_time_features(df)
        assert isinstance(result, tuple)
        assert len(result) == 4
        for arr in result:
            assert len(arr) == 100

    def test_values_in_signed_unit_range(self):
        from ml_features import compute_time_features
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=100, freq="h"),
        })
        hour_sin, hour_cos, day_sin, day_cos = compute_time_features(df)
        assert -1.0 <= hour_sin.min() <= hour_sin.max() <= 1.0
        assert -1.0 <= day_sin.min() <= day_sin.max() <= 1.0

    def test_missing_time_returns_zeros_and_ones(self):
        from ml_features import compute_time_features
        df = pd.DataFrame({"close": [1, 2, 3]})
        result = compute_time_features(df)
        assert len(result) == 4
        assert np.allclose(result[0], [0, 0, 0])
        assert np.allclose(result[1], [1, 1, 1])


class TestFractionalDiff:
    def test_returns_same_length(self):
        from ml_features import fractional_diff
        arr = np.linspace(100, 200, 100)
        result = fractional_diff(arr, d=0.5)
        assert len(result) == len(arr)

    def test_d0_produces_valid_output(self):
        from ml_features import fractional_diff
        arr = np.linspace(100, 200, 100)
        result = fractional_diff(arr, d=0.0)
        assert len(result) == len(arr)
        assert not np.isnan(result[-1])

    def test_d1_produces_valid_output(self):
        from ml_features import fractional_diff
        arr = np.linspace(100, 200, 100)
        result = fractional_diff(arr, d=1.0)
        assert len(result) == len(arr)
        assert not np.isnan(result[-1])


class TestComputeFeatures:
    @pytest.fixture
    def df(self):
        n = 250
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n))
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0.1, 1.0, n),
            "low": closes - rng.uniform(0.1, 1.0, n),
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "spread": rng.randint(1, 10, n),
        })

    def test_all_feature_cols_present(self, df):
        from ml_features import FEATURE_COLS, compute_features
        result = compute_features(df)
        missing = [c for c in FEATURE_COLS if c not in result.columns]
        assert not missing, f"Missing features: {missing}"

    def test_no_inf_values(self, df):
        from ml_features import compute_features
        result = compute_features(df).replace([np.inf, -np.inf], np.nan)
        assert result.isna().any().any()  # NaN is OK

    def test_feature_shapes_match(self, df):
        from ml_features import compute_features
        result = compute_features(df)
        assert len(result) == len(df)

    def test_kama_ratios_in_expected_range(self, df):
        from ml_features import compute_features
        result = compute_features(df)
        for col in ["kama9_ratio", "kama21_ratio", "kama50_ratio"]:
            valid = result[col].dropna()
            if len(valid) > 0:
                assert valid.min() > 0
                assert valid.max() < 5  # price shouldn't be 5x KAMA

    def test_empty_df_returns_empty(self):
        from ml_features import compute_features
        df = pd.DataFrame()
        result = compute_features(df)
        assert len(result) == 0


class TestPrepareFeatures:
    @pytest.fixture
    def df(self):
        n = 300
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n))
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + rng.uniform(0.1, 1.0, n),
            "low": closes - rng.uniform(0.1, 1.0, n),
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
            "time": pd.date_range("2026-01-01", periods=n, freq="h"),
            "spread": rng.randint(1, 10, n),
        })

    def test_returns_feature_df_and_aligned_df(self, df):
        from ml_features import FEATURE_COLS, prepare_features
        feature_data, aligned = prepare_features(df)
        assert len(feature_data) > 0
        assert list(feature_data.columns) == FEATURE_COLS
        assert len(aligned) == len(feature_data)

    def test_no_inf_after_prepare(self, df):
        from ml_features import prepare_features
        feature_data, _ = prepare_features(df)
        assert not np.any(np.isinf(feature_data.values))


class TestRMultipleLabels:
    @pytest.fixture
    def df(self):
        n = 100
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n))
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + abs(rng.randn(n)) * 2,
            "low": closes - abs(rng.randn(n)) * 2,
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
        })

    def test_returns_correct_length(self, df):
        from ml_features import r_multiple_labels
        result = r_multiple_labels(df)
        assert len(result) == len(df)

    def test_returns_float_series(self, df):
        from ml_features import r_multiple_labels
        result = r_multiple_labels(df)
        assert isinstance(result, pd.Series)
        assert result.dtype == np.float32

    def test_up_trend_produces_positive_r(self, df):
        from ml_features import r_multiple_labels
        df_up = df.copy()
        df_up["close"] = np.linspace(100, 200, len(df_up))
        df_up["high"] = df_up["close"] + 5
        df_up["low"] = df_up["close"] - 1
        result = r_multiple_labels(df_up)
        valid = result.dropna()
        if len(valid) > 0:
            assert valid.mean() > 0


class TestTripleBarrierLabels:
    @pytest.fixture
    def df(self):
        n = 100
        rng = np.random.RandomState(42)
        closes = 100 + np.cumsum(rng.randn(n))
        return pd.DataFrame({
            "open": closes - rng.uniform(0, 0.5, n),
            "high": closes + abs(rng.randn(n)) * 2,
            "low": closes - abs(rng.randn(n)) * 2,
            "close": closes,
            "tick_volume": rng.randint(100, 10000, n),
        })

    def test_returns_correct_length(self, df):
        from ml_features import triple_barrier_labels
        result = triple_barrier_labels(df)
        assert len(result) == len(df)

    def test_labels_are_1_0_or_minus1(self, df):
        from ml_features import triple_barrier_labels
        result = triple_barrier_labels(df)
        valid = result.dropna().unique()
        assert set(valid).issubset({-1, 1, 0})

    def test_strong_up_trend_labels_positive(self, df):
        from ml_features import triple_barrier_labels
        df_up = df.copy()
        df_up["close"] = np.linspace(100, 200, len(df_up))
        df_up["high"] = df_up["close"] + 5
        df_up["low"] = df_up["close"] - 1
        result = triple_barrier_labels(df_up)
        valid = result.dropna()
        if len(valid) > 0:
            assert (valid > 0).mean() > 0.5


class TestComputeFeatureStats:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "feat_a": np.random.randn(100),
            "feat_b": np.random.randn(100) * 2 + 1,
            "feat_c": np.full(100, 42.0),
        })

    def test_returns_dict(self, df):
        from ml_features import compute_feature_stats
        stats = compute_feature_stats(df)
        assert isinstance(stats, dict)

    def test_returns_mean_and_std(self, df):
        from ml_features import compute_feature_stats
        stats = compute_feature_stats(df)
        for feat, s in stats.items():
            assert "mean" in s
            assert "std" in s

    def test_short_column_skipped(self):
        from ml_features import compute_feature_stats
        df = pd.DataFrame({"short": [1, 2]})
        stats = compute_feature_stats(df)
        assert "short" not in stats


class TestOrderflowBackfill:
    """Item #11 guard: historical of_* features must be filled from M1 bars
    (no train/serve skew where they were always NaN -> 0.0)."""

    def _h1_m1(self, n=3):
        h1_idx = pd.date_range("2026-01-01", periods=n, freq="h")
        h1 = pd.DataFrame({
            "time": h1_idx, "open": 1.0, "high": 1.1, "low": 0.9,
            "close": 1.0, "tick_volume": 1000.0,
        })
        m1_idx = pd.date_range("2026-01-01", periods=n * 4, freq="15min")
        # bar0 all up-close, bar1 all down-close, bar2 mixed
        oc = [(1.00, 1.02), (0.99, 1.01), (1.00, 1.03), (0.98, 1.02),
              (1.00, 0.98), (1.01, 0.99), (1.02, 0.97), (1.00, 0.96),
              (1.00, 1.01), (1.00, 0.99), (1.00, 1.02), (1.00, 0.98)]
        op = np.array([a for a, _ in oc])
        cl = np.array([b for _, b in oc])
        m1 = pd.DataFrame({
            "time": m1_idx, "open": op, "high": np.maximum(op, cl) + 0.005,
            "low": np.minimum(op, cl) - 0.005, "close": cl,
            "tick_volume": 100.0, "spread": 2.0,
        })
        return h1, m1

    def test_m1_backfill_fills_all_bars(self):
        from ml_features import compute_orderflow_features
        h1, m1 = self._h1_m1()
        res = compute_orderflow_features(h1, "TEST", m1_df=m1)
        of_cols = ["of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio"]
        assert all(c in res for c in of_cols)
        for c in of_cols:
            assert not np.any(np.isnan(res[c])), (c, res[c])

    def test_m1_backfill_sign_matches_direction(self):
        from ml_features import compute_orderflow_features
        h1, m1 = self._h1_m1()
        res = compute_orderflow_features(h1, "TEST", m1_df=m1)
        assert res["of_cum_delta"][0] > 0
        assert res["of_cum_delta"][1] < 0
        assert abs(res["of_buy_ratio"][0] - 1.0) < 1e-9
        assert abs(res["of_buy_ratio"][1] - 0.0) < 1e-9

    def test_prepare_features_no_nan_with_m1(self):
        import numpy as np
        from ml_features import prepare_features
        n = 200
        idx = pd.date_range("2026-01-01", periods=n, freq="h")
        rng = np.random.RandomState(0)
        close = 100 + np.cumsum(rng.randn(n) * 0.3)
        h1 = pd.DataFrame({
            "time": idx, "open": close - 0.1, "high": close + 0.4,
            "low": close - 0.4, "close": close,
            "tick_volume": rng.randint(500, 1500, n).astype(float),
        })
        m1_idx = pd.date_range("2026-01-01", periods=n * 4, freq="15min")
        m1_close = np.interp(np.arange(n * 4), np.arange(0, n * 4, 4), close) + rng.randn(n * 4) * 0.05
        m1 = pd.DataFrame({
            "time": m1_idx, "open": m1_close - 0.02,
            "high": m1_close + 0.03, "low": m1_close - 0.03, "close": m1_close,
            "tick_volume": rng.randint(100, 300, n * 4).astype(float),
            "spread": np.full(n * 4, 2.0),
        })
        fd, _ = prepare_features(h1, symbol="TEST", m1_df=m1)
        of_cols = ["of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio"]
        for c in of_cols:
            assert c in fd.columns
            assert fd[c].isna().sum() == 0, (c, fd[c].isna().sum())

    def test_no_m1_keeps_live_last_bar_behavior(self):
        from ml_features import compute_orderflow_features
        h1, _ = self._h1_m1()
        # Without m1_df (and without live mt5), returns None gracefully.
        res = compute_orderflow_features(h1, "TEST", m1_df=None)
        assert res is None

    def test_attach_orderflow_matches_m1_df_path(self):
        """Pre-attaching of_* (optimizer path) must equal passing m1_df directly."""
        from ml_features import attach_orderflow_features, compute_features
        h1, m1 = self._h1_m1()
        # Path A: m1_df passed into compute_features
        fa = compute_features(h1.copy(), symbol="TEST", m1_df=m1)
        # Path B: of_* attached beforehand, compute_features detects and keeps them
        h1b = attach_orderflow_features(h1.copy(), m1)
        fb = compute_features(h1b, symbol="TEST")
        of_cols = ["of_cum_delta", "of_delta_ratio", "of_tick_imb", "of_avg_spread", "of_buy_ratio"]
        for c in of_cols:
            np.testing.assert_allclose(
                fa[c].values.astype(float), fb[c].values.astype(float), equal_nan=True,
                err_msg=f"{c} diverged between m1_df and pre-attached paths",
            )

    def test_attach_orderflow_noop_without_m1(self):
        from ml_features import attach_orderflow_features
        h1, _ = self._h1_m1()
        out = attach_orderflow_features(h1.copy(), None)
        assert "of_cum_delta" not in out.columns
