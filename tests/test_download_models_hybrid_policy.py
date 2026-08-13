"""Tests for scripts/download_models.py hybrid gate-failure streak accounting.

Regression for the double-count bug: a single optimize release reached the
apply step twice (once when it arrived via --dispatch, again when an unrelated
train release re-triggered the apply of the on-disk failed-params.json), and
each apply bumped the failure streak — pushing symbols to the pause strike
after one gate failure. Streaks are now keyed to the optimize release tag
(LAST_TAG_KEY) so a re-applied release counts only once, and a fresh pass
resets the streak.

bot/auto_optimizer is stubbed at sys.modules level (the real module needs the
bot package + settings.ini); only the param-apply helpers are exercised.
"""

import json
import logging
import sys

import download_models as dm

LOGGER = logging.getLogger("test_download_models")


class _FakeAO:
    """Stand-in for bot/auto_optimizer: records calls, mutates a dict settings."""

    def __init__(self, portfolio):
        self.portfolio = portfolio
        self.calls = []

    def set_trading_enabled(self, symbol, enabled, settings):
        self.calls.append(("trading", symbol, enabled))
        settings.setdefault(symbol, {})["trading_enabled"] = enabled
        return True

    def update_symbol_strategy(self, symbol, rec, settings):
        self.calls.append(("params", symbol, rec))
        settings.setdefault(symbol, {}).update(rec)
        return True

    def load_portfolio(self):
        return list(self.portfolio), {}

    def write_settings(self, settings):
        self.calls.append(("write", dict(settings)))


def _install_fake_ao(monkeypatch, portfolio):
    fake = _FakeAO(portfolio)
    monkeypatch.setitem(sys.modules, "auto_optimizer", fake)
    return fake


def _write_params(tmp_path, passed=None, failed=None):
    if passed is not None:
        (tmp_path / "strategy-params.json").write_text(json.dumps(passed))
    if failed is not None:
        (tmp_path / "failed-params.json").write_text(json.dumps(failed))


def _setup(monkeypatch, tmp_path, portfolio):
    _install_fake_ao(monkeypatch, portfolio)
    monkeypatch.setattr(dm, "BASE_DIR", tmp_path)
    monkeypatch.setattr(dm, "STREAKS_FILE", tmp_path / ".symbol_streaks.json")


def _streak_file(tmp_path):
    return json.loads((tmp_path / ".symbol_streaks.json").read_text())


PASSED = {
    "BTCUSD.raw": {"ema_fast_period": 3, "ema_slow_period": 12,
                   "atr_sl_multiplier": 1.5, "risk_reward_ratio": 3.0,
                   "adx_trend_threshold": 30, "scoring_min_entry": 0.60},
}
# EURUSD passing release (its own optimized params, not the tightened ones).
PASSED_EURUSD = dict(PASSED, **{
    "EURUSD.raw": {"ema_fast_period": 12, "ema_slow_period": 48,
                   "atr_sl_multiplier": 1.0, "risk_reward_ratio": 1.5,
                   "adx_trend_threshold": 22, "scoring_min_entry": 0.55},
})
FAILED = {
    "EURUSD.raw": {"ema_fast_period": 12, "ema_slow_period": 48,
                   "atr_sl_multiplier": 1.0, "risk_reward_ratio": 1.5,
                   "adx_trend_threshold": 22, "scoring_min_entry": 0.55},
    "XAUUSD.raw": {"ema_fast_period": 10, "ema_slow_period": 40,
                   "atr_sl_multiplier": 2.0, "risk_reward_ratio": 2.5,
                   "adx_trend_threshold": 25, "scoring_min_entry": 0.60},
}
PORTFOLIO = ["BTCUSD.raw", "EURUSD.raw", "XAUUSD.raw"]


class TestFailedReleaseCountsOnce:
    def test_same_release_reapplied_does_not_bump_streak(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        _write_params(tmp_path, passed=PASSED, failed=FAILED)

        # 1st apply — release arrives: both fail, strike 1, tightened, trading.
        assert dm._apply_strategy_params(LOGGER, "optimize-A") is True
        s1 = _streak_file(tmp_path)
        assert s1["EURUSD.raw"] == 1 and s1["XAUUSD.raw"] == 1
        assert s1[dm.LAST_TAG_KEY] == "optimize-A"

        # 2nd apply of the SAME release (the bug: train-only change re-triggers
        # the apply). Streak must NOT advance to 2, no pause.
        assert dm._apply_strategy_params(LOGGER, "optimize-A") is True
        s2 = _streak_file(tmp_path)
        assert s2["EURUSD.raw"] == 1 and s2["XAUUSD.raw"] == 1
        assert s2[dm.LAST_TAG_KEY] == "optimize-A"

    def test_next_failing_release_advances_to_pause(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        _write_params(tmp_path, passed=PASSED, failed=FAILED)

        dm._apply_strategy_params(LOGGER, "optimize-A")
        dm._apply_strategy_params(LOGGER, "optimize-A")  # re-apply: no-op
        dm._apply_strategy_params(LOGGER, "optimize-B")  # genuinely new release

        s = _streak_file(tmp_path)
        assert s["EURUSD.raw"] == 2 and s["XAUUSD.raw"] == 2
        assert s[dm.LAST_TAG_KEY] == "optimize-B"

    def test_strike_two_pauses_entries(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        _write_params(tmp_path, passed=PASSED, failed=FAILED)
        dm._apply_strategy_params(LOGGER, "optimize-A")
        dm._apply_strategy_params(LOGGER, "optimize-B")

        # Strike 2: trading_enabled=false, params untouched (no score bump).
        (tmp_path / "settings.ini").write_text("")  # placeholder not needed
        fake = sys.modules["auto_optimizer"]
        trading = [c for c in fake.calls if c[0] == "trading"]
        assert ("trading", "EURUSD.raw", False) in trading
        assert ("trading", "XAUUSD.raw", False) in trading
        score_recs = [c for c in fake.calls if c[0] == "params" and c[1] == "EURUSD.raw"]
        # Only the strike-1 apply (optimize-A) touched params, tightened to 0.70.
        assert score_recs and score_recs[0][2]["score"] == 0.70

    def test_strike_one_tightens_entry(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        _write_params(tmp_path, passed=PASSED, failed=FAILED)
        dm._apply_strategy_params(LOGGER, "optimize-A")

        fake = sys.modules["auto_optimizer"]
        recs = {c[1]: c[2] for c in fake.calls if c[0] == "params"}
        # base 0.55 + 0.15 = 0.70, base 0.60 + 0.15 = 0.75, cap 0.90 respected.
        assert recs["EURUSD.raw"]["score"] == 0.70
        assert recs["XAUUSD.raw"]["score"] == 0.75
        trading = {c[1]: c[2] for c in fake.calls if c[0] == "trading"}
        assert trading["EURUSD.raw"] is True and trading["XAUUSD.raw"] is True


class TestPassResetsStreak:
    def test_passed_release_clears_streak_and_enables(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        # Prior state: EURUSD paused at strike 2 from an older release.
        (tmp_path / ".symbol_streaks.json").write_text(json.dumps({
            "EURUSD.raw": 2, dm.LAST_TAG_KEY: "optimize-A",
        }))
        # EURUSD itself passes this release — its streak resets, entries re-enable.
        _write_params(tmp_path, passed=PASSED_EURUSD)

        assert dm._apply_strategy_params(LOGGER, "optimize-B") is True
        s = _streak_file(tmp_path)
        assert "EURUSD.raw" not in s
        assert s[dm.LAST_TAG_KEY] == "optimize-B"
        fake = sys.modules["auto_optimizer"]
        trading = {c[1]: c[2] for c in fake.calls if c[0] == "trading"}
        assert trading["BTCUSD.raw"] is True and trading["EURUSD.raw"] is True

    def test_failure_after_pass_is_strike_one(self, monkeypatch, tmp_path):
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        (tmp_path / ".symbol_streaks.json").write_text(json.dumps({
            "EURUSD.raw": 2, dm.LAST_TAG_KEY: "optimize-A",
        }))
        _write_params(tmp_path, passed=PASSED_EURUSD)
        dm._apply_strategy_params(LOGGER, "optimize-B")  # pass resets to 0
        _write_params(tmp_path, passed=PASSED_EURUSD, failed={"EURUSD.raw": FAILED["EURUSD.raw"]})
        dm._apply_strategy_params(LOGGER, "optimize-C")  # next failure

        s = _streak_file(tmp_path)
        # Reset then one new failure: strike 1 (tighten), not strike 3 (pause).
        assert s["EURUSD.raw"] == 1
        fake = sys.modules["auto_optimizer"]
        trading = {c[1]: c[2] for c in fake.calls if c[0] == "trading"}
        assert trading["EURUSD.raw"] is True


class TestStreakFilePersistence:
    def test_load_handles_tag_legacy_strings_and_junk(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dm, "STREAKS_FILE", tmp_path / ".symbol_streaks.json")
        (tmp_path / ".symbol_streaks.json").write_text(json.dumps({
            "EURUSD.raw": 2,
            "XAUUSD.raw": "1",  # legacy string int from an old writer
            dm.LAST_TAG_KEY: "optimize-A",
            "junk": "not-a-count",
        }))
        s = dm._load_streaks()
        assert s["EURUSD.raw"] == 2
        assert s["XAUUSD.raw"] == 1
        assert s[dm.LAST_TAG_KEY] == "optimize-A"
        assert "junk" not in s

    def test_load_missing_file_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dm, "STREAKS_FILE", tmp_path / ".symbol_streaks.json")
        assert dm._load_streaks() == {}

    def test_load_corrupt_file_is_empty(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dm, "STREAKS_FILE", tmp_path / ".symbol_streaks.json")
        (tmp_path / ".symbol_streaks.json").write_text("{ not json")
        assert dm._load_streaks() == {}

    def test_save_records_tag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(dm, "STREAKS_FILE", tmp_path / ".symbol_streaks.json")
        dm._save_streaks({"EURUSD.raw": 1}, "optimize-A")
        assert _streak_file(tmp_path) == {"EURUSD.raw": 1, dm.LAST_TAG_KEY: "optimize-A"}


class TestNoTagPath:
    def test_failed_without_tag_still_counts(self, monkeypatch, tmp_path):
        # Defensive: if no release tag is available (e.g. caller without one),
        # the streak still bumps rather than silently skipping.
        _setup(monkeypatch, tmp_path, PORTFOLIO)
        _write_params(tmp_path, passed=PASSED, failed=FAILED)
        assert dm._apply_strategy_params(LOGGER, None) is True
        s = _streak_file(tmp_path)
        assert s["EURUSD.raw"] == 1 and "EURUSD.raw" in s
        assert dm.LAST_TAG_KEY not in s  # nothing recorded without a tag

    def test_apply_without_param_files_returns_false(self, tmp_path):
        assert dm._apply_strategy_params(LOGGER, "optimize-A") is False
