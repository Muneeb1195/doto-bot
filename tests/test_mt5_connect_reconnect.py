"""Tests for mt5_connect.py reconnect/timeout machinery.

Covers the highest-risk code paths identified in the audit:
- timeout triggers instance recreation
- busy-vs-dead ping distinction
- instance swap on reconnect
- thread-bound call error recovery
"""

import sys
import time
from concurrent.futures import TimeoutError as FutureTimeout
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, "bot")


class TestMt5CallTimeoutRecreate:
    """A timeout in mt5_call should trigger instance recreation."""

    def test_timeout_triggers_recreate(self, monkeypatch):
        """When the executor future times out, _recreate_instance is called."""
        import mt5_connect as mc
        fake_executor = MagicMock()
        fut = MagicMock()
        fut.result.side_effect = FutureTimeout("slow")
        fake_executor.submit.return_value = fut
        monkeypatch.setattr(mc, "_call_executor", fake_executor)
        monkeypatch.setattr(mc, "_executor_lock", __import__("threading").Lock())
        recreated = []
        monkeypatch.setattr(mc, "_recreate_instance", lambda: recreated.append(True) or True)
        def slow(*a, **kw):
            time.sleep(10)
            return 42
        result = mc.mt5_call(slow, _timeout=0.1)
        assert result is None
        assert len(recreated) == 1

    def test_thread_bound_error_triggers_recreate(self, monkeypatch):
        """Thread-bound calls that raise should trigger recreation."""
        import mt5_connect as mc
        recreated = []
        monkeypatch.setattr(mc, "_recreate_instance", lambda: recreated.append(True) or True)
        def boom(*a, **kw):
            raise ConnectionError("RPyC pipe broken")
        result = mc.mt5_call(boom)
        assert result is None
        assert len(recreated) == 1


class TestBusyVsDeadPing:
    """_mt5linux_ping must distinguish a busy server from a dead one."""

    def test_returns_busy_on_timeout(self, monkeypatch):
        """A timeout returns _PING_BUSY (server is alive but blocked)."""
        import mt5_connect as mc
        monkeypatch.setattr(mc, "_PING_BUSY", object())  # unique sentinel
        with patch("mt5linux.MetaTrader5") as MockMT5:
            inst = MockMT5.return_value
            inst.initialize.side_effect = Exception("timed out after 10s")
            result = mc._mt5linux_ping()
            assert result is mc._PING_BUSY

    def test_returns_none_on_connection_error(self, monkeypatch):
        """A connection error returns None (server is dead)."""
        import mt5_connect as mc
        with patch("mt5linux.MetaTrader5") as MockMT5:
            MockMT5.side_effect = OSError("Connection refused")
            result = mc._mt5linux_ping()
            assert result is None

    def test_returns_terminal_info_on_success(self, monkeypatch):
        """A successful ping returns the TerminalInfo object."""
        import mt5_connect as mc
        fake_info = MagicMock()
        fake_info.connected = True
        with patch("mt5linux.MetaTrader5") as MockMT5:
            inst = MockMT5.return_value
            inst.terminal_info.return_value = fake_info
            result = mc._mt5linux_ping()
            assert result is fake_info


class TestInstanceSwap:
    """_recreate_instance should swap _mt5_instance and mt5 atomically."""

    def test_recreate_swaps_instance(self, monkeypatch):
        """After recreation, _mt5_instance points to the new object."""
        import mt5_connect as mc
        new_inst = MagicMock()
        monkeypatch.setattr(mc, "_init_mt5linux", lambda: new_inst)
        monkeypatch.setattr(mc, "_executor_lock", __import__("threading").Lock())
        ok = mc._recreate_instance()
        assert ok is True
        assert mc._mt5_instance is new_inst
        assert mc.mt5 is new_inst

    def test_recreate_returns_false_on_failure(self, monkeypatch):
        """If _init_mt5linux fails, _recreate_instance returns False."""
        import mt5_connect as mc
        def fail():
            raise RuntimeError("server down")
        monkeypatch.setattr(mc, "_init_mt5linux", fail)
        monkeypatch.setattr(mc, "_executor_lock", __import__("threading").Lock())
        ok = mc._recreate_instance()
        assert ok is False


class TestEnsureMt5Connected:
    """ensure_mt5_connected should recover from a dead connection."""

    def test_recovers_when_instance_is_none(self, monkeypatch):
        """If _mt5_instance is None, it attempts init_mt5()."""
        import mt5_connect as mc
        monkeypatch.setattr(mc, "_mt5_instance", None)
        monkeypatch.setattr(mc, "init_mt5", lambda: None)
        fake_info = MagicMock()
        fake_info.connected = True
        monkeypatch.setattr(mc, "_mt5linux_ping", lambda: fake_info)
        ok = mc.ensure_mt5_connected({"symbols": ["EURUSD.raw"]})
        assert ok is True

    def test_returns_true_when_connected(self, monkeypatch):
        """If ping succeeds, returns True."""
        import mt5_connect as mc
        monkeypatch.setattr(mc, "_mt5_instance", MagicMock())
        fake_info = MagicMock()
        fake_info.connected = True
        monkeypatch.setattr(mc, "_mt5linux_ping", lambda: fake_info)
        ok = mc.ensure_mt5_connected({"symbols": ["EURUSD.raw"]})
        assert ok is True

    def test_returns_true_when_busy(self, monkeypatch):
        """If ping returns _PING_BUSY, returns True (skip cycle, don't reconnect)."""
        import mt5_connect as mc
        monkeypatch.setattr(mc, "_mt5_instance", MagicMock())
        monkeypatch.setattr(mc, "_PING_BUSY", object())
        monkeypatch.setattr(mc, "_mt5linux_ping", lambda: mc._PING_BUSY)
        ok = mc.ensure_mt5_connected({"symbols": ["EURUSD.raw"]})
        assert ok is True
