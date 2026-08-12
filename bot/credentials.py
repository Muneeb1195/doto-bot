"""Single reader for MT5 login credentials + terminal settings.

Consolidates the previously duplicated settings+credentials readers in
scenario_analysis.py and tune_scaleout.py. Env vars override credentials.ini:

    MT5_ACCOUNT / MT5_PASSWORD / MT5_SERVER

Terminal path/timeout come from config/settings.ini [MT5].
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

# Default terminal path for native (Windows) installs; the Linux deployment
# rewrites [MT5] path to the Wine binary via deploy-linux.sh.
_DEFAULT_MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"


def load_credentials():
    """Return {account, password, server, path, timeout, portfolio_symbols}.

    Reads config/credentials.ini [LOGIN] with MT5_ACCOUNT / MT5_PASSWORD /
    MT5_SERVER env-var overrides, and config/settings.ini for the terminal
    path/timeout and portfolio. Raises RuntimeError if the login triple is
    missing entirely.
    """
    settings = configparser.ConfigParser()
    settings.read(CONFIG_DIR / "settings.ini")
    creds = configparser.ConfigParser()
    creds.read(CONFIG_DIR / "credentials.ini")
    login = creds["LOGIN"] if creds.has_section("LOGIN") else {}

    account_raw = os.getenv("MT5_ACCOUNT") or login.get("account")
    password = os.getenv("MT5_PASSWORD") or login.get("password")
    server = os.getenv("MT5_SERVER") or login.get("server")
    if not account_raw or not password or not server:
        raise RuntimeError(
            "MT5 credentials missing: set MT5_ACCOUNT/MT5_PASSWORD/MT5_SERVER "
            "env vars or fill [LOGIN] in config/credentials.ini"
        )
    return {
        "account": int(account_raw),
        "password": password,
        "server": server,
        "path": settings.get("MT5", "path", fallback=_DEFAULT_MT5_PATH),
        "timeout": int(settings.get("MT5", "timeout_ms", fallback=180000)),
        "portfolio_symbols": [s.strip() for s in settings.get("PORTFOLIO", "symbols", fallback="").split(",")],
    }
