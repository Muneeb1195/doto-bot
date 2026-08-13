"""Tests for dashboard.py — write_dashboard_state atomic write, payload shape."""

import json
import sys
from unittest.mock import MagicMock

sys.path.insert(0, "bot")

import pytest


class FakeAccountInfo:
    def __init__(self):
        self.balance = 500000.0
        self.equity = 501000.0
        self.profit = 1000.0
        self.margin = 50000.0
        self.margin_free = 450000.0
        self.server = "DOTOGlobal-Real"
        self.name = "Test User"
        self.currency = "PKR"


class FakeTerminalInfo:
    def __init__(self):
        self.connected = True


class FakePosition:
    def __init__(self, symbol="XAU500.raw", ptype=0, volume=0.1, price=1900.0,
                 sl=1880.0, tp=1920.0, profit=100.0, swap=-2.5, ticket=1001):
        self.symbol = symbol
        self.type = ptype  # 0 = buy
        self.volume = volume
        self.price_open = price
        self.sl = sl
        self.tp = tp
        self.profit = profit
        self.swap = swap
        self.ticket = ticket





@pytest.fixture
def mock_mt5_calls(monkeypatch):
    # Set up named sub-mocks so mt5_call can dispatch by mock_name
    mt5_mod = sys.modules["MetaTrader5"]

    def make_named(name, return_val):
        m = MagicMock(name=name)
        m.__name__ = name
        return m

    acc_info = make_named("account_info", FakeAccountInfo())
    acc_info.balance = 500000.0
    acc_info.equity = 501000.0
    acc_info.profit = 1000.0
    acc_info.margin = 50000.0
    acc_info.margin_free = 450000.0
    acc_info.server = "DOTOGlobal-Real"
    mt5_mod.account_info = acc_info

    term_info = make_named("terminal_info", FakeTerminalInfo())
    term_info.connected = True
    mt5_mod.terminal_info = term_info

    mt5_mod.ORDER_TYPE_BUY = 0
    mt5_mod.ORDER_TYPE_SELL = 1

    mt5_return_values = {
        "account_info": acc_info,
        "terminal_info": term_info,
    }

    def fake_mt5_call(func, *args, **kwargs):
        name = getattr(func, "__name__", "")
        if name in mt5_return_values:
            return mt5_return_values[name]
        return None

    monkeypatch.setattr("dashboard.mt5_call", fake_mt5_call)


class TestWriteDashboardState:
    def test_writes_state_file(self, mock_mt5_calls, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        write_dashboard_state([FakePosition()], {"XAU500.raw": "strong_trend"})
        state_file = tmp_path / "data" / "dashboard_state.json"
        assert state_file.exists()
        with open(state_file) as f:
            data = json.load(f)
        assert data["balance"] == 500000.0
        assert data["equity"] == 501000.0

    def test_contains_required_keys(self, mock_mt5_calls, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        write_dashboard_state([FakePosition()], {"XAU500.raw": "strong_trend"})
        with open(tmp_path / "data" / "dashboard_state.json") as f:
            data = json.load(f)
        required = ["balance", "equity", "profit", "margin_free", "positions",
                     "regimes", "positions_detail", "filters", "health"]
        for key in required:
            assert key in data, f"Missing key: {key}"

    def test_positions_detail_no_dead_fields(self, mock_mt5_calls, tmp_path):
        """The template never reads `ticket`; dead writer fields are a contract
        violation (tests/test_dashboard_contract.py enforces this end-to-end)."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        pos = FakePosition(symbol="BTCUSD.raw", ptype=1, ticket=2002)
        write_dashboard_state([pos], {})
        with open(tmp_path / "data" / "dashboard_state.json") as f:
            data = json.load(f)
        assert len(data["positions_detail"]) == 1
        pd_ = data["positions_detail"][0]
        assert pd_["symbol"] == "BTCUSD.raw"
        assert pd_["type"] == "sell"
        assert pd_["volume"] == 0.1
        assert "ticket" not in pd_

    def test_health_connected(self, mock_mt5_calls, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        write_dashboard_state([], {})
        with open(tmp_path / "data" / "dashboard_state.json") as f:
            data = json.load(f)
        assert data["health"]["connected"] is True
        assert data["health"]["server"] == "DOTOGlobal-Real"

    def test_atomic_write_creates_no_tmp_leftover(self, mock_mt5_calls, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        write_dashboard_state([], {})
        tmp_files = list(tmp_path.rglob("*.tmp"))
        assert len(tmp_files) == 0

    def test_does_not_crash_when_account_info_none(self, monkeypatch, tmp_path):
        def fake_mt5_call(func, *args, **kwargs):
            if hasattr(func, "__name__") and func.__name__ == "account_info":
                return None
            return FakeTerminalInfo()

        monkeypatch.setattr("dashboard.mt5_call", fake_mt5_call)
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        from dashboard import write_dashboard_state
        write_dashboard_state([], {})  # should not raise

    def test_regimes_copied_not_mutated(self, mock_mt5_calls, tmp_path):
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr("dashboard.DASHBOARD_STATE", tmp_path / "data" / "dashboard_state.json")
        (tmp_path / "data").mkdir(exist_ok=True)

        regimes = {"XAU500.raw": "strong_trend"}
        from dashboard import write_dashboard_state
        write_dashboard_state([], regimes)
        regimes["XAU500.raw"] = "ranging"
        with open(tmp_path / "data" / "dashboard_state.json") as f:
            data = json.load(f)
        assert data["regimes"]["XAU500.raw"] == "strong_trend"
