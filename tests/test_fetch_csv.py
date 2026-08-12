"""Tests for `optimize_params._fetch_csv_mode` (the `--fetch-csv` harvest).

Stubs the MT5 bridge (`mt5_connect.mt5` + `ensure_mt5_connected` +
`fetch_rates_paged`) and points BASE_DIR/CONFIG_DIR at a tmp dir, then verifies
the per-timeframe CSV writes (correct `<SYMBOL>_<TF>.csv` names, epoch-second
`time`, fixed column order) and the `[SYMBOL_POINTS]` persistence in
settings.ini — the contract the offline `--csv` loaders depend on.
"""

import configparser
from types import SimpleNamespace

import mt5_connect
import numpy as np
import optimize_params as op
import pandas as pd


class _FakeMT5:
    """Minimal bridge proxy: timeframe constants, symbol_info, H1 fetch."""

    TIMEFRAME_H1 = 16385
    TIMEFRAME_M15 = 15
    TIMEFRAME_M1 = 1

    def __init__(self, sinfo):
        self._sinfo = sinfo

    def symbol_select(self, *a):
        return True

    def symbol_info(self, symbol):
        return self._sinfo

    def copy_rates_range(self, symbol, tf, start, end):
        # Real MT5 returns a structured array; a DataFrame with epoch-second
        # time exercises the same pd.DataFrame(rates) + to_datetime(unit='s') path.
        n = 500
        end_ts = int(pd.Timestamp(end).timestamp())
        t = np.arange(end_ts - n * 3600, end_ts, 3600)
        close = 100.0 + np.cumsum(np.random.RandomState(1).normal(0, 0.5, n))
        open_ = np.concatenate([[close[0]], close[:-1]])
        return pd.DataFrame(
            {
                "time": t,
                "open": open_,
                "high": open_ + 0.3,
                "low": np.minimum(open_, close) - 0.3,
                "close": close,
                "tick_volume": np.full(n, 1000),
                "spread": np.full(n, 30.0),
            }
        )


def _paged_df(symbol, tf, start, end):
    """Stand-in for mt5_connect.fetch_rates_paged (datetime-indexed bars)."""
    n = 2000
    t = pd.date_range(end=pd.Timestamp(end), periods=n, freq="15min")
    close = np.linspace(100.0, 105.0, n)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            "time": t,
            "open": open_,
            "high": open_ + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "tick_volume": np.full(n, 500),
            "spread": np.full(n, 30.0),
        }
    )


def _stub_bridge(monkeypatch, tmp_path, sinfo, paged_fn):
    import config as _config

    monkeypatch.setattr(_config, "load_config", lambda: {"symbols": ["X.raw"]})
    monkeypatch.setattr(mt5_connect, "ensure_mt5_connected", lambda cfg: True)
    monkeypatch.setattr(mt5_connect, "fetch_rates_paged", paged_fn)
    monkeypatch.setattr(mt5_connect, "mt5", _FakeMT5(sinfo))
    monkeypatch.setattr(op, "BASE_DIR", tmp_path)
    monkeypatch.setattr(op, "CONFIG_DIR", tmp_path / "config")
    (tmp_path / "config").mkdir()
    # Isolate the settings write from the real repo settings.ini object.
    monkeypatch.setattr(op, "settings", configparser.ConfigParser())


class TestFetchCsvMode:
    def test_writes_all_three_tfs_and_symbol_points(self, monkeypatch, tmp_path):
        sinfo = SimpleNamespace(point=0.01, trade_tick_value=1.0, volume_step=0.01)
        _stub_bridge(monkeypatch, tmp_path, sinfo, _paged_df)

        op._fetch_csv_mode(SimpleNamespace(years=1), ["X.raw"])

        hist = tmp_path / "data" / "history"
        for tf in ("H1", "M15", "M1"):
            p = hist / f"X_raw_{tf}.csv"
            assert p.exists(), f"missing {p}"
            df = pd.read_csv(p)
            assert list(df.columns) == [
                "time", "open", "high", "low", "close", "tick_volume", "spread",
            ]
            assert df["time"].iloc[0] > 1_500_000_000  # epoch seconds
            assert df["time"].is_monotonic_increasing

        ini = tmp_path / "config" / "settings.ini"
        assert ini.exists(), "settings.ini not written"
        txt = ini.read_text().lower()  # configparser lowercases option keys
        assert "[symbol_points]" in txt
        assert "x.raw" in txt and "x.raw_tick" in txt and "x.raw_vstep" in txt
        assert "x.raw = 0.01" in txt and "x.raw_tick = 1.0" in txt

    def test_bars_written_without_symbol_info_but_no_symbol_points(
        self, monkeypatch, tmp_path
    ):
        # symbol_info -> None: bars are still written (usable offline with
        # default point/tick), but nothing is persisted to [SYMBOL_POINTS].
        _stub_bridge(monkeypatch, tmp_path, None, _paged_df)

        op._fetch_csv_mode(SimpleNamespace(years=1), ["X.raw"])

        hist = tmp_path / "data" / "history"
        assert (hist / "X_raw_H1.csv").exists()
        assert (hist / "X_raw_M15.csv").exists()
        assert not (tmp_path / "config" / "settings.ini").exists()

    def test_paged_failure_skips_only_m15_m1(self, monkeypatch, tmp_path):
        sinfo = SimpleNamespace(point=0.01, trade_tick_value=1.0, volume_step=0.01)
        # fetch_rates_paged returns None -> M15/M1 skipped, H1 still saved.
        _stub_bridge(monkeypatch, tmp_path, sinfo, lambda *a, **k: None)

        op._fetch_csv_mode(SimpleNamespace(years=1), ["X.raw"])

        hist = tmp_path / "data" / "history"
        assert (hist / "X_raw_H1.csv").exists()
        assert not (hist / "X_raw_M15.csv").exists()
        assert not (hist / "X_raw_M1.csv").exists()
        # The H1 write still persisted symbol info.
        ini = tmp_path / "config" / "settings.ini"
        assert ini.exists() and "x.raw = 0.01" in ini.read_text().lower()
