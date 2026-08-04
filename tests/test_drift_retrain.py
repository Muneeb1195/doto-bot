"""Tests for drift_retrain.py — label-param resolution and warm-start guards."""

import configparser
import sys
from unittest.mock import MagicMock

sys.modules.setdefault("MetaTrader5", MagicMock())
sys.path.insert(0, "bot")

import pytest  # noqa: E402

pytest.importorskip("xgboost")
pytest.importorskip("lightgbm")

from drift_retrain import _resolve_label_params, warmstart_model  # noqa: E402


class TestResolveLabelParams:
    def _cfg(self, text):
        cfg = configparser.ConfigParser()
        cfg.read_string(text)
        return cfg

    def test_per_symbol_override(self):
        cfg = self._cfg(
            "[STRATEGY]\natr_sl_multiplier = 1.0\nrisk_reward_ratio = 2.0\n"
            "[STRATEGY:XAU500.raw]\natr_sl_multiplier = 2.0\nrisk_reward_ratio = 3.0\n"
        )
        tp_atr, sl_atr = _resolve_label_params(cfg, "XAU500.raw")
        assert sl_atr == 2.0
        assert tp_atr == 6.0  # 2.0 * 3.0

    def test_falls_back_to_global(self):
        cfg = self._cfg("[STRATEGY]\natr_sl_multiplier = 1.5\nrisk_reward_ratio = 2.0\n")
        tp_atr, sl_atr = _resolve_label_params(cfg, "BTCUSD.raw")
        assert sl_atr == 1.5
        assert tp_atr == 3.0

    def test_default_when_missing(self):
        cfg = self._cfg("[OTHER]\nfoo = 1\n")
        tp_atr, sl_atr = _resolve_label_params(cfg, "ANY.raw")
        assert sl_atr == 1.0
        assert tp_atr == 2.0

    def test_partial_symbol_section_uses_global(self):
        cfg = self._cfg(
            "[STRATEGY]\natr_sl_multiplier = 1.0\nrisk_reward_ratio = 2.0\n"
            "[STRATEGY:GBPJPY.raw]\natr_sl_multiplier = 2.5\n"  # missing rr
        )
        tp_atr, sl_atr = _resolve_label_params(cfg, "GBPJPY.raw")
        assert sl_atr == 1.0  # falls through to global
        assert tp_atr == 2.0


class TestWarmstartGuards:
    def test_missing_model_returns_false(self):
        assert warmstart_model("NONEXISTENT_SYMBOL_XYZ.raw") is False
