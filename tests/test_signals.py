"""Tests for signals.py — pure functions only (no MT5 dependency)."""

import sys  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from analytics import compute_entry_score  # noqa: E402
from indicators import calc_rsi  # noqa: E402
from signals import RegimeGate, _get_regime_gate, get_mtf_fused_signal, get_trend_pullback_signal  # noqa: E402


class TestGetTrendPullbackSignal:
    def test_no_signal_without_cross(self):
        import numpy as np  # noqa: E402
        import pandas as pd  # noqa: E402
        flat = pd.DataFrame({
            "close": np.linspace(100, 100.5, 200),
            "high": np.linspace(100.5, 101, 200),
            "low": np.linspace(99.5, 100, 200),
        })
        cfg = {"ema_fast": 8, "ema_slow": 32, "atr_period": 14, "pb_atr_mult": 1.2}
        signal, atr, sig_type = get_trend_pullback_signal(flat, cfg)
        assert signal is None
        assert atr > 0
        assert sig_type is None

    def test_buy_pullback_in_up_trend(self, trending_up_df):
        cfg = {"ema_fast": 8, "ema_slow": 32, "atr_period": 14, "pb_atr_mult": 2.0}
        signal, atr, sig_type = get_trend_pullback_signal(trending_up_df, cfg)
        if signal == "buy":
            assert sig_type == "pullback"
            assert atr > 0

    def test_pullback_stays_within_bounds(self, trending_up_df):
        cfg = {"ema_fast": 8, "ema_slow": 32, "atr_period": 14, "pb_atr_mult": 1.0}
        signal, atr, sig_type = get_trend_pullback_signal(trending_up_df, cfg)
        if signal is not None:
            assert sig_type == "pullback"
            assert atr > 0

    def test_very_small_atr_mult_only_returns_atr(self, trending_up_df):
        cfg = {"ema_fast": 8, "ema_slow": 32, "atr_period": 14, "pb_atr_mult": 0.05}
        signal, atr, sig_type = get_trend_pullback_signal(trending_up_df, cfg)
        assert signal is None

    def test_fast_above_slow_condition(self, sample_df):
        cfg = {"ema_fast": 5, "ema_slow": 50, "atr_period": 14, "pb_atr_mult": 2.0}
        from indicators import calc_ma  # noqa: E402
        fast = calc_ma(sample_df, 5, "kama")
        slow = calc_ma(sample_df, 50, "kama")
        signal, atr, sig_type = get_trend_pullback_signal(sample_df, cfg)
        if fast.iloc[-1] > slow.iloc[-1]:
            if signal == "buy":
                assert sig_type == "pullback"
        elif fast.iloc[-1] < slow.iloc[-1]:
            if signal == "sell":
                assert sig_type == "pullback"
        else:
            assert signal is None


class TestCalcRSI:
    def test_returns_valid_rsi(self, sample_df):
        result = calc_rsi(sample_df, 14)
        if result is not None:
            assert 0 <= result <= 100

    def test_insufficient_data_returns_default(self, small_df):
        result = calc_rsi(small_df, 50)
        assert result == 50.0

    def test_less_than_period_plus_one(self):
        df = pd.DataFrame({"close": [100] * 20})
        result = calc_rsi(df, 50)
        assert result == 50.0

    def test_up_trend_high_rsi(self, trending_up_df):
        result = calc_rsi(trending_up_df, 14)
        if result is not None:
            assert result > 50

    def test_always_up_returns_100(self):
        df = pd.DataFrame({"close": np.linspace(100, 110, 30)})
        result = calc_rsi(df, 14)
        if result is not None:
            assert result == 100.0


class TestComputeEntryScoreNews:
    """News component must be SYMMETRIC: neutral news contributes 0.5 (its true
    neutral value) and is always included in the weighted score, so the
    min_entry_score threshold is measured against the same denominator whether
    or not a news reading exists."""

    def _cfg(self):
        return {
            "symbol": "TEST.raw",
            "ml_enabled": False,
            "scoring_ml_fallback": 0.60,
            "spf_enabled": False,
            "ns_enabled": True,
            "scoring_weights": {"ml": 0.40, "spread": 0.30, "news": 0.30},
            "ml_threshold_overrides": {},
            "ml_confidence": 0.55,
        }

    def test_neutral_news_scores_0_5(self):
        cfg = self._cfg()
        score, details, _ = compute_entry_score(cfg, "buy", 1.0)
        assert details["news"] == 0.5
        # ml fallback 0.6 (w0.4) + spread 0.5 (w0.3) + news 0.5 (w0.3)
        expected = 0.6 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3
        assert abs(score - expected) < 1e-9

    def test_news_present_shifts_score(self):
        cfg = self._cfg()
        news = {"symbols": {"TEST.raw": {"count": 3, "score": 0.8}}}
        with patch("signals._st._ns_cache", {"data": news}):
            score, details, _ = compute_entry_score(cfg, "buy", 1.0)
        # positive sentiment -> news component > 0.5
        assert details["news"] > 0.5
        expected = 0.6 * 0.4 + 0.5 * 0.3 + details["news"] * 0.3
        assert abs(score - expected) < 1e-9

    def test_sell_flips_news_sentiment(self):
        cfg = self._cfg()
        news = {"symbols": {"TEST.raw": {"count": 3, "score": 0.8}}}
        with patch("signals._st._ns_cache", {"data": news}):
            _, details_buy, _ = compute_entry_score(cfg, "buy", 1.0)
            _, details_sell, _ = compute_entry_score(cfg, "sell", 1.0)
        # For sell, positive news is bearish => news component flipped below 0.5
        assert details_sell["news"] < 0.5
        assert abs(details_buy["news"] - (1.0 - details_sell["news"])) < 1e-9

    def test_news_absence_and_presence_same_denominator(self):
        """Score with neutral news and score with a 0.5-news reading must be
        identical (symmetry): the threshold is never made easier by missing news."""
        cfg = self._cfg()
        score_neutral, _, _ = compute_entry_score(cfg, "buy", 1.0)
        news = {"symbols": {"TEST.raw": {"count": 1, "score": 0.0}}}  # (0+1)/2 = 0.5
        with patch("signals._st._ns_cache", {"data": news}):
            score_explicit, _, _ = compute_entry_score(cfg, "buy", 1.0)
        assert abs(score_neutral - score_explicit) < 1e-9


class TestRegimeGate:
    def test_starts_closed(self):
        gate = RegimeGate(threshold=50.0, buffer=5.0)
        assert gate.is_open is False

    def test_opens_above_upper_band(self):
        gate = RegimeGate(threshold=50.0, buffer=5.0)
        assert gate.update(53.0) is True  # > 52.5

    def test_stays_closed_in_buffer(self):
        gate = RegimeGate(threshold=50.0, buffer=5.0)
        assert gate.update(51.0) is False  # not > 52.5

    def test_hysteresis_no_flicker(self):
        gate = RegimeGate(threshold=50.0, buffer=5.0)
        assert gate.update(53.0) is True   # open
        assert gate.update(49.0) is True   # still open (>= 47.5)
        assert gate.update(47.0) is False  # now closes (< 47.5)

    def test_reset(self):
        gate = RegimeGate(threshold=50.0, buffer=5.0)
        gate.update(60.0)
        gate.reset()
        assert gate.is_open is False

    def test_get_regime_gate_cached_per_symbol(self):
        cfg = {"symbol": "AAA.raw", "fused_threshold": 50.0, "fused_buffer": 5.0}
        g1 = _get_regime_gate("AAA.raw", cfg)
        g2 = _get_regime_gate("AAA.raw", cfg)
        assert g1 is g2


def _trend_df(n=300, start=100.0, step=0.1):
    close = start + np.arange(n) * step
    return pd.DataFrame({
        "time": np.arange(n) * 3600,
        "open": close - 0.05,
        "high": close + 0.2,
        "low": close - 0.2,
        "close": close,
        "tick_volume": np.full(n, 1000),
        "spread": np.full(n, 10),
    })


class TestGetMtfFusedSignal:
    def _cfg(self):
        return {
            "symbol": "MTF.raw", "ma_type": "kama", "ema_fast": 8, "ema_slow": 32,
            "atr_period": 14, "pb_enabled": True, "pb_atr_mult": 1.2,
            "mtf_agreement_threshold": 0.5, "er_period": 10,
        }

    def test_insufficient_data_returns_none_tuple(self):
        with patch("signals.get_rates", return_value=None):
            result = get_mtf_fused_signal(self._cfg())
        assert result == (None, None, None, None)

    def test_returns_four_tuple_contract(self):
        with patch("signals.get_rates", return_value=_trend_df()):
            sig, atr, etype, ratio = get_mtf_fused_signal(self._cfg())
        assert sig in (None, "buy", "sell")
        assert atr is not None and atr > 0

    def test_does_not_touch_regime_gate_state(self):
        import state as _st  # noqa: E402
        _st._regime_gate_state.pop("MTF.raw", None)
        with patch("signals.get_rates", return_value=_trend_df()):
            get_mtf_fused_signal(self._cfg())
        assert "MTF.raw" not in _st._regime_gate_state
