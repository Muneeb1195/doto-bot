"""Tests for journal.py — uses mock_mt5 factories."""

import sys  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402

sys.modules["MetaTrader5"] = MagicMock()
sys.path.insert(0, "bot")

import csv  # noqa: E402

import pytest  # noqa: E402
import state as _st  # noqa: E402


@pytest.fixture
def trade_csv(tmp_path):
    original = _st.TRADE_CSV
    _st.TRADE_CSV = tmp_path / "trades.csv"
    yield _st.TRADE_CSV
    _st.TRADE_CSV = original


class TestJournalInit:
    def test_creates_csv_with_headers(self, trade_csv):
        from journal import journal_init  # noqa: E402
        journal_init()
        assert trade_csv.exists()
        with open(trade_csv, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == _st.TRADE_HEADERS

    def test_does_not_overwrite_existing(self, trade_csv):
        trade_csv.write_text("col1,col2\nval1,val2\n")
        from journal import journal_init  # noqa: E402
        journal_init()
        assert trade_csv.read_text() == "col1,col2\nval1,val2\n"


class TestJournalOpen:
    def test_appends_open_row(self, trade_csv):
        from journal import journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        rows = list(csv.DictReader(trade_csv.read_text().splitlines()))
        assert len(rows) == 1
        assert rows[0]["ticket"] == "1001"
        assert rows[0]["symbol"] == "XAU500.raw"
        assert rows[0]["event"] == "OPEN"

    def test_appends_multiple_opens(self, trade_csv):
        from journal import journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_open(1002, "BTCUSD.raw", "sell", 0.2, 50000.0, 50500.0, 48000.0, 3.0)
        rows = list(csv.DictReader(trade_csv.read_text().splitlines()))
        assert len(rows) == 2

    def test_uses_correct_format(self, trade_csv):
        from journal import journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        row = list(csv.DictReader(trade_csv.read_text().splitlines()))[0]
        assert float(row["entry_price"]) == pytest.approx(2000.0, abs=0.001)
        assert float(row["sl"]) == pytest.approx(1990.0, abs=0.001)
        assert float(row["tp"]) == pytest.approx(2020.0, abs=0.001)
        assert float(row["atr"]) == pytest.approx(2.5, abs=0.001)


class TestJournalClose:
    def test_appends_close_row(self, trade_csv):
        from journal import journal_close, journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0)
        rows = list(csv.DictReader(trade_csv.read_text().splitlines()))
        assert len(rows) == 2
        close_rows = [r for r in rows if r["event"] == "CLOSE"]
        assert len(close_rows) == 1
        assert close_rows[0]["ticket"] == "1001"

    def test_close_has_exit_data(self, trade_csv):
        from journal import journal_close, journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0)
        row = list(csv.DictReader(trade_csv.read_text().splitlines()))[1]
        assert float(row["exit_price"]) == pytest.approx(2010.0, abs=0.001)
        assert float(row["pnl"]) == pytest.approx(50.0, abs=0.01)
        assert float(row["pips"]) == pytest.approx(10.0, abs=0.1)

    def test_updates_daily_realized_pnl(self, trade_csv):
        from journal import journal_close, journal_init, journal_open  # noqa: E402
        _st._daily_realized_pnl = 0.0
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0)
        assert _st._daily_realized_pnl == 50.0

    def test_multiple_closes_accumulate_pnl(self, trade_csv):
        from journal import journal_close, journal_init, journal_open  # noqa: E402
        _st._daily_realized_pnl = 0.0
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0)
        journal_open(1002, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1002, 1990.0, -30.0, -10.0)
        assert _st._daily_realized_pnl == 20.0

    def test_custom_event_type(self, trade_csv):
        from journal import journal_close, journal_init, journal_open  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0, event="MR_NAKED_CLOSE")
        row = list(csv.DictReader(trade_csv.read_text().splitlines()))[1]
        assert row["event"] == "MR_NAKED_CLOSE"


class TestReconcileJournal:
    def test_no_action_when_no_orphans(self, trade_csv):
        from journal import journal_close, journal_init, journal_open, reconcile_journal  # noqa: E402
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_close(1001, 2010.0, 50.0, 10.0)
        before = len(list(csv.DictReader(trade_csv.read_text().splitlines())))
        reconcile_journal({1001})
        after = len(list(csv.DictReader(trade_csv.read_text().splitlines())))
        assert after == before

    def test_appends_manual_close_for_orphan(self, trade_csv, monkeypatch):
        from journal import journal_init, journal_open, reconcile_journal  # noqa: E402

        from tests.mock_mt5 import FakeMt5Module  # noqa: E402
        fake_mt5 = FakeMt5Module()
        import journal as j_mod  # noqa: E402
        monkeypatch.setattr(j_mod, "mt5", fake_mt5)
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        reconcile_journal(set())
        rows = list(csv.DictReader(trade_csv.read_text().splitlines()))
        assert len(rows) == 2
        assert rows[1]["event"] == "MANUAL_CLOSE"

    def test_multiple_orphans_all_closed(self, trade_csv, monkeypatch):
        from journal import journal_init, journal_open, reconcile_journal  # noqa: E402

        from tests.mock_mt5 import FakeMt5Module  # noqa: E402
        fake_mt5 = FakeMt5Module()
        import journal as j_mod  # noqa: E402
        monkeypatch.setattr(j_mod, "mt5", fake_mt5)
        journal_init()
        journal_open(1001, "XAU500.raw", "buy", 0.1, 2000.0, 1990.0, 2020.0, 2.5)
        journal_open(1002, "BTCUSD.raw", "sell", 0.2, 50000.0, 50500.0, 48000.0, 3.0)
        reconcile_journal(set())
        rows = list(csv.DictReader(trade_csv.read_text().splitlines()))
        manual_rows = [r for r in rows if r["event"] == "MANUAL_CLOSE"]
        assert len(manual_rows) == 2

    def test_missing_csv_does_not_crash(self):
        from journal import reconcile_journal  # noqa: E402
        reconcile_journal({1001})  # should not raise
