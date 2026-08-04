"""Tests for risk.py — pure logic + CSV-dependent functions."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import csv  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import risk  # noqa: E402
import state  # noqa: E402


class TestCalcKellyMult:
    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.trade_csv = Path(self.tmp_dir) / "trades.csv"
        risk.TRADE_CSV = self.trade_csv
        state.TRADE_CSV = self.trade_csv
        yield
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_trades(self, pnls):
        headers = ["ticket", "symbol", "type", "volume", "entry_price", "sl",
                    "tp", "open_time", "atr", "exit_price", "exit_time", "pnl",
                    "pips", "event"]
        with open(self.trade_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for i, pnl in enumerate(pnls):
                event = "CLOSE" if pnl != 0 else "OPEN"
                writer.writerow({
                    "ticket": i, "symbol": "XAU500.raw", "type": "buy",
                    "volume": "0.10", "entry_price": "2000.00", "sl": "1990.00",
                    "tp": "2020.00", "open_time": "2024-01-01 10:00:00",
                    "atr": "10.0", "exit_price": "2010.00",
                    "exit_time": "2024-01-01 12:00:00", "pnl": str(pnl),
                    "pips": "10.0", "event": event,
                })

    def test_no_trades_returns_min_mult(self, basic_cfg):
        assert risk.calc_kelly_mult(basic_cfg) == basic_cfg["dr_min_mult"]

    def test_less_than_10_trades_returns_min_mult(self, basic_cfg):
        self._write_trades([10, -5, 8])
        assert risk.calc_kelly_mult(basic_cfg) == basic_cfg["dr_min_mult"]

    def test_10_wins_returns_positive_kelly(self, basic_cfg):
        self._write_trades([10] * 15)
        result = risk.calc_kelly_mult(basic_cfg)
        assert result > 0
        assert result <= basic_cfg["dr_max_mult"]

    def test_10_losses_returns_min_mult(self, basic_cfg):
        self._write_trades([-10] * 15)
        result = risk.calc_kelly_mult(basic_cfg)
        assert result == basic_cfg["dr_min_mult"]

    def test_mixed_trades(self, basic_cfg):
        pnls = [10, -5, 8, -3, 12, -6, 9, -4, 7, -2, 11, -7, 6, -3, 10]
        self._write_trades(pnls)
        result = risk.calc_kelly_mult(basic_cfg)
        assert 0 <= result <= basic_cfg["dr_max_mult"]

    def test_disabled_kelly_returns_1(self, basic_cfg):
        basic_cfg["dr_enabled"] = False
        assert risk.calc_kelly_mult(basic_cfg) == 1.0

    def test_lookback_limits(self, basic_cfg):
        self._write_trades([10] * 100)
        result = risk.calc_kelly_mult(basic_cfg)
        assert result >= 0

    def test_open_trades_not_counted(self, basic_cfg):
        pnls = [10] * 10 + [0] * 5
        self._write_trades(pnls)
        result = risk.calc_kelly_mult(basic_cfg)
        assert result == basic_cfg["dr_min_mult"]

    def test_csv_does_not_exist(self, basic_cfg):
        risk.TRADE_CSV = Path("/nonexistent/trades.csv")
        assert risk.calc_kelly_mult(basic_cfg) == basic_cfg["dr_min_mult"]

    def test_kelly_bounded(self, basic_cfg):
        self._write_trades([10] * 15)
        result = risk.calc_kelly_mult(basic_cfg)
        assert basic_cfg["dr_min_mult"] <= result <= basic_cfg["dr_max_mult"]

    def test_equal_win_loss_high_winrate(self, basic_cfg):
        pnls = [10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10, -10, 10]
        self._write_trades(pnls)
        result = risk.calc_kelly_mult(basic_cfg)
        assert result > 0


class TestKellyRMultAggregation:
    """Regression guards for agent audit H3 (R-multiples), N1 (per-symbol),
    M6 (scale-out partials aggregated per ticket)."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.trade_csv = Path(self.tmp_dir) / "trades.csv"
        risk.TRADE_CSV = self.trade_csv
        state.TRADE_CSV = self.trade_csv
        yield
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_row(self, **kw):
        headers = ["ticket", "symbol", "type", "volume", "entry_price", "sl",
                   "tp", "entry_time", "atr", "exit_price", "exit_time", "pnl",
                   "pips", "event"]
        row = {h: "" for h in headers}
        row.update(kw)
        exists = self.trade_csv.exists() and self.trade_csv.stat().st_size > 0
        with open(self.trade_csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            if not exists:
                w.writeheader()
            w.writerow(row)

    def _fake_symbol_info(self):
        class SI:
            trade_tick_size = 0.01
            trade_tick_value = 1.0
            point = 0.01
        return SI()

    def test_partials_aggregated_per_ticket(self, basic_cfg, monkeypatch):
        # tick=0.01, point=0.01 -> sl 2000->1990 => distance=10, sl_value=1000,
        # vol 0.1 => risk=100. A win ticket: partial +1000, close +1000 => +2000
        # (R=20.0). A loss ticket: close -5000 => R=-50.0. Build 5 wins + 5 losses
        # (10 grouped trades, >=10 required) of symbol XAU500.raw, plus one OTHER
        # ticket that must be excluded by N1.
        for i in range(5):
            t = 100 + i
            self._write_row(ticket=t, symbol="XAU500.raw", type="buy", volume="0.1",
                            entry_price="2000.00", sl="1990.00", tp="2020.00",
                            entry_time=f"2024-01-{i+1:02d} 10:00:00", event="OPEN")
            self._write_row(ticket=t, symbol="XAU500.raw", exit_price="2010.00",
                            exit_time=f"2024-01-{i+1:02d} 11:00:00", pnl="1000.00", pips="10", event="PARTIAL")
            self._write_row(ticket=t, symbol="XAU500.raw", exit_price="2010.00",
                            exit_time=f"2024-01-{i+1:02d} 12:00:00", pnl="1000.00", pips="10", event="CLOSE")
        for i in range(5):
            t = 200 + i
            self._write_row(ticket=t, symbol="XAU500.raw", type="buy", volume="0.1",
                            entry_price="2000.00", sl="1990.00", tp="2020.00",
                            entry_time=f"2024-02-{i+1:02d} 10:00:00", event="OPEN")
            self._write_row(ticket=t, symbol="XAU500.raw", exit_price="1995.00",
                            exit_time=f"2024-02-{i+1:02d} 11:00:00", pnl="-5000.00", pips="-5", event="CLOSE")
        self._write_row(ticket=999, symbol="OTHER", type="buy", volume="0.1",
                        entry_price="2000.00", sl="1990.00",
                        entry_time="2024-03-01 10:00:00", event="OPEN")
        self._write_row(ticket=999, symbol="OTHER", exit_price="2010.00",
                        exit_time="2024-03-01 11:00:00", pnl="3000.00", pips="10", event="CLOSE")

        monkeypatch.setattr(risk, "mt5_call", lambda *a, **k: self._fake_symbol_info())
        wr, aw, al = risk.get_recent_trade_stats(basic_cfg)
        assert wr is not None
        # 10 XAU500.raw trades (5 win, 5 loss); OTHER excluded by N1.
        assert abs(wr - 0.5) < 1e-9
        assert abs(aw - 20.0) < 1e-9
        assert abs(al - 50.0) < 1e-9

    def test_legacy_currency_fallback(self, basic_cfg, monkeypatch):
        # No OPEN rows -> legacy per-row currency path (keeps old behaviour/tests).
        for i in range(9):
            self._write_row(ticket=10 + i, symbol="XAU500.raw", exit_price="2010.00",
                            exit_time=f"2024-01-{i+1:02d} 12:00:00", pnl="10.00", pips="10", event="CLOSE")
        self._write_row(ticket=99, symbol="XAU500.raw", exit_price="1995.00",
                        exit_time="2024-02-01 12:00:00", pnl="-5.00", pips="-5", event="CLOSE")
        monkeypatch.setattr(risk, "mt5_call", lambda *a, **k: self._fake_symbol_info())
        wr, aw, al = risk.get_recent_trade_stats(basic_cfg)
        assert wr is not None
        # 9 wins + 1 loss = 10 grouped trades (>=10 required).
        assert abs(wr - 9 / 10) < 1e-9
