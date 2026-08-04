"""Tests for state.py save/load with temp files."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import json  # noqa: E402
import os  # noqa: E402

import pytest  # noqa: E402
import state as _st  # noqa: E402


@pytest.fixture
def tmp_state_file(tmp_path):
    original = _st.STATE_FILE
    _st.STATE_FILE = tmp_path / "bot_state.json"
    yield _st.STATE_FILE
    _st.STATE_FILE = original


class TestSaveBotState:
    def test_saves_scale_out_state(self, tmp_state_file):
        _st._scale_out_state.clear()
        _st._scale_out_state[1001] = {"step": 1, "entry_price": 2000.0}
        _st.save_bot_state()
        assert tmp_state_file.exists()
        data = json.loads(tmp_state_file.read_text())
        assert "1001" in data["scale_out_state"]
        assert data["scale_out_state"]["1001"]["step"] == 1

    def test_saves_chandelier_state(self, tmp_state_file):
        _st._chandelier_state.clear()
        _st._chandelier_state[1002] = {"ch_sl": 1990.0}
        _st.save_bot_state()
        data = json.loads(tmp_state_file.read_text())
        assert "1002" in data["chandelier_state"]
        assert data["chandelier_state"]["1002"]["ch_sl"] == 1990.0

    def test_saves_exec_bias_with_date(self, tmp_state_file):
        from datetime import date  # noqa: E402
        _st._exec_bias.clear()
        _st._exec_bias["XAU500.raw"] = {"bias": "bullish", "since": 12345.0, "date": date(2024, 1, 15)}
        _st.save_bot_state()
        data = json.loads(tmp_state_file.read_text())
        assert data["exec_bias"]["XAU500.raw"]["date"] == "2024-01-15"

    def test_saves_last_trade_time(self, tmp_state_file):
        _st._last_trade_time.clear()
        _st._last_trade_time["trend:XAU500.raw"] = 1000.0
        _st.save_bot_state()
        data = json.loads(tmp_state_file.read_text())
        assert data["last_trade_time"]["trend:XAU500.raw"] == 1000.0

    def test_saves_risk_flags(self, tmp_state_file):
        _st._tail_risk_triggered["XAU500.raw"] = True
        _st._tail_risk_cooldown["XAU500.raw"] = 999.0
        _st._circuit_breaker_triggered = True
        _st.save_bot_state()
        data = json.loads(tmp_state_file.read_text())
        assert data["tail_risk_triggered"]["XAU500.raw"] is True
        assert data["tail_risk_cooldown"]["XAU500.raw"] == 999.0
        assert data["circuit_breaker_triggered"] is True

    def test_empty_state_does_not_crash(self, tmp_state_file):
        _st._scale_out_state.clear()
        _st._chandelier_state.clear()
        _st._exec_bias.clear()
        _st._last_trade_time.clear()
        _st.save_bot_state()
        assert tmp_state_file.exists()


class TestLoadBotState:
    def test_loads_scale_out_state(self, tmp_state_file):
        data = {
            "scale_out_state": {"1001": {"step": 2, "entry_price": 2000.0}},
            "chandelier_state": {},
            "exec_bias": {},
            "last_trade_time": {},
            "tail_risk_triggered": {},
            "tail_risk_cooldown": {},
            "circuit_breaker_triggered": False,
        }
        tmp_state_file.write_text(json.dumps(data))
        _st._scale_out_state.clear()
        _st.load_bot_state()
        assert 1001 in _st._scale_out_state
        assert _st._scale_out_state[1001]["step"] == 2

    def test_loads_exec_bias_with_date(self, tmp_state_file):
        data = {
            "scale_out_state": {},
            "chandelier_state": {},
            "exec_bias": {"XAU500.raw": {"bias": "bearish", "since": 12345.0, "date": "2024-06-01"}},
            "last_trade_time": {},
            "tail_risk_triggered": {},
            "tail_risk_cooldown": {},
            "circuit_breaker_triggered": False,
        }
        tmp_state_file.write_text(json.dumps(data))
        _st._exec_bias.clear()
        _st.load_bot_state()
        assert "XAU500.raw" in _st._exec_bias
        assert str(_st._exec_bias["XAU500.raw"]["date"]) == "2024-06-01"

    def test_loads_risk_flags(self, tmp_state_file):
        data = {
            "scale_out_state": {},
            "chandelier_state": {},
            "exec_bias": {},
            "last_trade_time": {},
            "tail_risk_triggered": {"XAU500.raw": True},
            "tail_risk_cooldown": {"XAU500.raw": 500.0},
            "circuit_breaker_triggered": True,
        }
        tmp_state_file.write_text(json.dumps(data))
        _st.load_bot_state()
        assert _st._tail_risk_triggered.get("XAU500.raw") is True
        assert _st._tail_risk_cooldown.get("XAU500.raw") == 500.0
        assert _st._circuit_breaker_triggered is True

    def test_missing_file_does_not_crash(self, tmp_state_file):
        if tmp_state_file.exists():
            os.remove(tmp_state_file)
        _st.load_bot_state()  # should not raise

    def test_corrupted_file_does_not_crash(self, tmp_state_file):
        tmp_state_file.write_text("{invalid json")
        _st.load_bot_state()  # should not raise

    def test_empty_state_file_does_not_crash(self, tmp_state_file):
        tmp_state_file.write_text("{}")
        _st.load_bot_state()  # should not raise

    def test_restores_only_saved_keys(self, tmp_state_file):
        _st._tail_risk_triggered.clear()
        _st._circuit_breaker_triggered = False
        data = {"scale_out_state": {}, "chandelier_state": {}, "exec_bias": {},
                "last_trade_time": {}, "tail_risk_triggered": {"SOLUSD.raw": True},
                "tail_risk_cooldown": {"SOLUSD.raw": 100.0}, "circuit_breaker_triggered": False}
        tmp_state_file.write_text(json.dumps(data))
        _st.load_bot_state()
        assert _st._tail_risk_triggered.get("SOLUSD.raw") is True
        assert _st._tail_risk_cooldown.get("SOLUSD.raw") == 100.0
