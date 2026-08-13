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


class TestSchemaTable:
    """Every PERSISTED field round-trips save->load and appears in the payload
    (the schema table is the single source — no hand-written mirrors to drift)."""

    def test_every_persisted_key_in_payload(self, tmp_state_file):
        _st.save_bot_state()
        data = json.loads(tmp_state_file.read_text())
        for key, gname, kind, _ in _st.PERSISTED:
            assert key in data, f"PERSISTED field {key} ({gname}) missing from saved payload"

    def test_every_persisted_key_roundtrips(self, tmp_state_file):
        from datetime import date

        reps = {
            "_scale_out_state": {1001: {"step": 1, "entry_price": 2000.0}},
            "_chandelier_state": {1002: {"ch_sl": 1990.0}},
            "_exec_bias": {"XAU500.raw": {"bias": "bullish", "since": 1.0, "date": date(2024, 1, 15)}},
            "_last_trade_time": {"trend:XAU500.raw": 123.0},
            "_tail_risk_triggered": {"XAU500.raw": True},
            "_tail_risk_cooldown": {"XAU500.raw": 999.0},
            "_circuit_breaker_triggered": True,
            "_peak_balance": 54321.0,
            "_mr_consecutive_losses": {"XAU500.raw": 2},
            "_mr_last_loss_time": {"XAU500.raw": 111.0},
            "_dynamic_deviation": {"XAU500.raw": 80},
            "_daily_loss_hit": True,
            "_daily_realized_pnl": -4321.5,
            "_daily_realized_date": date(2024, 3, 5),
            "_pending_limits": {"XAU500.raw": {"ticket": 7}},
            "_imported_external_ids": {"a", "b"},
        }
        for gname, value in reps.items():
            if isinstance(value, (dict, set)):
                target = getattr(_st, gname)
                target.clear()
                target.update(value)
            else:
                setattr(_st, gname, value)
        _st.save_bot_state()
        for gname in reps:
            current = getattr(_st, gname)
            if isinstance(current, (dict, set)):
                current.clear()
            else:
                setattr(_st, gname, None if gname.endswith("_date") else 0)
        _st.load_bot_state()
        for gname, expected in reps.items():
            assert getattr(_st, gname) == expected, f"{gname} roundtrip mismatch"


class TestDailyRollover:
    def test_read_stale_date_is_zero_and_read_only(self):
        from datetime import date
        _st._daily_realized_pnl = -100.0
        _st._daily_realized_date = date(2020, 1, 1)
        assert _st.daily_realized_pnl_for(date(2020, 1, 2)) == 0.0
        assert _st._daily_realized_pnl == -100.0  # filters must NOT zero the counter
        assert _st._daily_realized_date == date(2020, 1, 1)

    def test_read_same_date_returns_pnl(self):
        from datetime import date
        _st._daily_realized_pnl = -100.0
        _st._daily_realized_date = date(2020, 1, 2)
        assert _st.daily_realized_pnl_for(date(2020, 1, 2)) == -100.0

    def test_roll_zeroes_on_new_day(self):
        from datetime import date
        _st._daily_realized_pnl = -100.0
        _st._daily_realized_date = date(2020, 1, 1)
        assert _st.roll_daily_realized_pnl(date(2020, 1, 2)) == 0.0
        assert _st._daily_realized_pnl == 0.0
        assert _st._daily_realized_date == date(2020, 1, 2)

    def test_roll_same_date_keeps_pnl(self):
        from datetime import date
        _st._daily_realized_pnl = -100.0
        _st._daily_realized_date = date(2020, 1, 2)
        assert _st.roll_daily_realized_pnl(date(2020, 1, 2)) == -100.0
