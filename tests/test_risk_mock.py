"""Mock-based tests for risk.py — uses monkeypatch on mt5_call."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import pytest  # noqa: E402

from tests.mock_mt5 import FakeAccountInfo, FakeSymbolInfo  # noqa: E402


class TestCalcPositionSize:
    @pytest.fixture(autouse=True)
    def setup_risk(self):
        import risk  # noqa: E402
        self.risk = risk

    def test_returns_default_when_no_account(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(self.risk, "mt5_call",
                            lambda fn, *a, _timeout=30, **kw: None)
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol == 0.0

    def test_returns_zero_on_zero_balance(self, basic_cfg, monkeypatch):
        zero_acc = MagicMock()
        zero_acc.balance = 0.0
        monkeypatch.setattr(
            self.risk, "mt5_call",
            lambda fn, *a, _timeout=30, **kw: zero_acc if "account_info" in str(fn) else FakeSymbolInfo())
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol == 0.0

    def test_returns_default_when_no_symbol_info(self, basic_cfg, monkeypatch):
        calls = {"account_info": 0, "symbol_info": 0}
        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                calls["account_info"] += 1
                return FakeAccountInfo()
            if "symbol_info" in str(fn):
                calls["symbol_info"] += 1
                return None
            return None
        monkeypatch.setattr(self.risk, "mt5_call", fake_mt5_call)
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol == 0.0

    def test_calculates_volume(self, basic_cfg, monkeypatch):
        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo(balance=100000.0)
            if "symbol_info" in str(fn):
                return FakeSymbolInfo(trade_tick_value=0.1, trade_tick_size=0.01,
                                     volume_step=0.01, volume_min=0.01, volume_max=100.0)
            return None
        monkeypatch.setattr(self.risk, "mt5_call", fake_mt5_call)
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol > 0
        expected = 100000 * 0.01 / max(100 * 0.1, 1e-10)
        expected = max(round(expected / 0.01, 0) * 0.01, 0.01)
        assert vol == pytest.approx(expected, rel=0.1)

    def test_respects_max_tail_risk(self, basic_cfg, monkeypatch):
        basic_cfg["max_tail_risk_pct"] = 0.01
        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo(balance=100000.0)
            if "symbol_info" in str(fn):
                return FakeSymbolInfo(trade_tick_value=0.1, trade_tick_size=0.01,
                                     volume_step=0.01, volume_min=0.01, volume_max=100.0)
            return None
        monkeypatch.setattr(self.risk, "mt5_call", fake_mt5_call)
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol == 0.0

    def test_skips_when_risk_ratio_exceeds_max(self, basic_cfg, monkeypatch):
        basic_cfg["max_risk_ratio"] = 0.1
        def fake_mt5_call(fn, *a, _timeout=30, **kw):
            if "account_info" in str(fn):
                return FakeAccountInfo(balance=100000.0)
            if "symbol_info" in str(fn):
                return FakeSymbolInfo(trade_tick_value=0.1, trade_tick_size=0.01,
                                     volume_step=200.0, volume_min=200.0, volume_max=500.0)
            return None
        monkeypatch.setattr(self.risk, "mt5_call", fake_mt5_call)
        vol = self.risk.calc_position_size(basic_cfg, 100)
        assert vol == 0.0


class TestCalcKellyMult:
    @pytest.fixture(autouse=True)
    def setup_risk(self):
        import risk  # noqa: E402
        self.risk = risk

    def test_disabled_returns_1(self, basic_cfg):
        basic_cfg["dr_enabled"] = False
        result = self.risk.calc_kelly_mult(basic_cfg)
        assert result == 1.0

    def test_no_trades_returns_min_mult(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(self.risk, "get_recent_trade_stats",
                            lambda cfg: (None, None, None))
        result = self.risk.calc_kelly_mult(basic_cfg)
        assert result == basic_cfg["dr_min_mult"]

    def test_all_wins_returns_positive(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(self.risk, "get_recent_trade_stats",
                            lambda cfg: (1.0, 100.0, 0.0))
        result = self.risk.calc_kelly_mult(basic_cfg)
        assert result >= basic_cfg["dr_min_mult"]

    def test_all_losses_returns_min_mult(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(self.risk, "get_recent_trade_stats",
                            lambda cfg: (0.0, 0.0, 100.0))
        result = self.risk.calc_kelly_mult(basic_cfg)
        assert result == basic_cfg["dr_min_mult"]


class TestCalcVolatilityMult:
    @pytest.fixture(autouse=True)
    def setup_risk(self):
        import risk  # noqa: E402
        self.risk = risk

    def test_disabled_returns_1(self, basic_cfg):
        basic_cfg["dr_vol_adjust"] = False
        result = self.risk.calc_volatility_mult(basic_cfg)
        assert result == 1.0

    def test_no_rates_returns_1(self, basic_cfg, monkeypatch):
        monkeypatch.setattr(self.risk, "get_rates",
                            lambda sym, tf, n: None)
        result = self.risk.calc_volatility_mult(basic_cfg)
        assert result == 1.0

    def test_short_rates_returns_1(self, basic_cfg, monkeypatch):
        import pandas as pd  # noqa: E402
        monkeypatch.setattr(
            self.risk, "get_rates",
            lambda sym, tf, n: pd.DataFrame({"close": [100] * 10}))
        result = self.risk.calc_volatility_mult(basic_cfg)
        assert result == 1.0
