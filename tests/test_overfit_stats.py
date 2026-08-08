"""Tests for the multiple-testing diagnostics (Bailey / Lopez de Prado)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bot"))

from overfit_stats import (  # noqa: E402
    _norm_ppf,
    deflated_sharpe,
    effective_trials,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    summarize,
)


class TestNormPpf:
    @pytest.mark.parametrize(
        "p,expected",
        [(0.5, 0.0), (0.975, 1.959964), (0.99, 2.326348), (0.025, -1.959964)],
    )
    def test_matches_known_quantiles(self, p, expected):
        assert _norm_ppf(p) == pytest.approx(expected, abs=1e-5)

    def test_monotonic(self):
        vals = [_norm_ppf(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert vals == sorted(vals)


class TestExpectedMaxSharpe:
    def test_grows_with_trial_count(self):
        """More trials must raise the bar a result has to clear."""
        seq = [expected_max_sharpe(n, 1.0) for n in (10, 45, 100, 1000)]
        assert seq == sorted(seq)

    def test_reproduces_published_value(self):
        # Bailey & Lopez de Prado: N=10 skill-less trials yields an expected
        # maximum Sharpe of ~1.57 despite zero true edge.
        assert expected_max_sharpe(10, 1.0) == pytest.approx(1.57, abs=0.02)

    def test_zero_dispersion_gives_zero(self):
        assert expected_max_sharpe(100, 0.0) == 0.0

    def test_scales_linearly_with_dispersion(self):
        assert expected_max_sharpe(50, 2.0) == pytest.approx(2 * expected_max_sharpe(50, 1.0))


class TestEffectiveTrials:
    def _grid(self, emas, sls=(1.0, 1.5, 2.0), rrs=(2.0, 2.5), adxs=(22, 25), scs=(0.55, 0.60)):
        return [
            (f, s, sl, rr, adx, sc)
            for (f, s) in emas
            for sl in sls
            for rr in rrs
            for adx in adxs
            for sc in scs
        ]

    def test_same_ratio_ma_pairs_collapse(self):
        """MA pairs sharing a fast/slow ratio explore one signal, not many.

        Mirrors the published finding that a large lookback grid reduces to a
        handful of independent bets.
        """
        emas = [(6, 24), (8, 32), (10, 40), (12, 48), (15, 60)]  # all 1:4
        combos = self._grid(emas)
        eff = effective_trials(combos)
        assert eff < len(combos)
        # 1 distinct ratio x 3 SL x 2 RR x 2 ADX x 2 score
        assert eff == pytest.approx(24)

    def test_distinct_ratios_add_breadth(self):
        same = effective_trials(self._grid([(10, 40), (12, 48)]))          # both 1:4
        diff = effective_trials(self._grid([(10, 40), (10, 30)]))          # 1:4 and 1:3
        assert diff > same

    def test_never_exceeds_nominal(self):
        combos = self._grid([(10, 40)])
        assert effective_trials(combos) <= len(combos)

    def test_single_trial(self):
        assert effective_trials([(10, 40, 1.5, 2.0, 25, 0.6)]) == 1.0

    def test_malformed_input_falls_back_to_nominal(self):
        """A conservative fallback keeps the significance bar high."""
        assert effective_trials([("bad",), ("worse",)]) == 2.0


class TestDeflatedSharpe:
    def test_strong_result_few_trials_passes(self):
        assert deflated_sharpe(3.0, 0.5, n_trials=10, n_obs=500) > 0.95

    def test_weak_result_many_trials_fails(self):
        assert deflated_sharpe(1.0, 1.0, n_trials=1000, n_obs=50) < 0.5

    def test_more_trials_lowers_dsr(self):
        few = deflated_sharpe(2.0, 0.5, n_trials=10, n_obs=200)
        many = deflated_sharpe(2.0, 0.5, n_trials=5000, n_obs=200)
        assert many < few

    def test_bounded_probability(self):
        for n in (2, 10, 1000):
            v = deflated_sharpe(1.5, 0.6, n_trials=n, n_obs=100)
            assert 0.0 <= v <= 1.0

    def test_insufficient_observations(self):
        assert deflated_sharpe(2.0, 0.5, n_trials=10, n_obs=1) == 0.0


class TestPBO:
    def test_perfect_carryover_is_zero(self):
        r = np.random.default_rng(0).normal(size=200)
        assert probability_of_backtest_overfitting(r, r) == pytest.approx(0.0)

    def test_total_inversion_is_one(self):
        r = np.random.default_rng(0).normal(size=200)
        assert probability_of_backtest_overfitting(r, -r) == pytest.approx(1.0)

    def test_independent_scores_near_half(self):
        rng = np.random.default_rng(7)
        v = probability_of_backtest_overfitting(rng.normal(size=400), rng.normal(size=400))
        assert 0.25 < v < 0.75

    def test_too_few_trials_is_nan(self):
        assert np.isnan(probability_of_backtest_overfitting([1, 2], [1, 2]))


class TestSummarize:
    def test_empty_input(self):
        out = summarize([])
        assert out["n_trials"] == 0
        assert out["dsr"] == 0.0

    def test_keys_present(self):
        out = summarize([1.0, 2.0, 3.0, 4.0], oos_scores=[1.0, 2.0, 3.0, 4.0], n_obs=50)
        for k in ("n_trials", "effective_trials", "sr_std", "expected_max_sr", "dsr", "pbo"):
            assert k in out

    def test_defaults_to_nominal_without_combos(self):
        out = summarize([1.0, 2.0, 3.0], n_obs=10)
        assert out["effective_trials"] == 3.0

    def test_uses_combos_when_supplied(self):
        combos = [(10, 40, 1.5, 2.0, 25, 0.6), (12, 48, 1.5, 2.0, 25, 0.6)]  # same ratio
        out = summarize([1.0, 2.0], oos_scores=[1.0, 2.0], n_obs=10, combos=combos)
        assert out["effective_trials"] < 2.0
