"""Start MT5 terminal with proper cleanup for use by optimizer scripts."""

import platform
import subprocess
import sys
import time
from pathlib import Path

if platform.system() == "Linux":
    WINE_DIR = Path.home() / ".wine"
    MT5_PATH = str(WINE_DIR / "drive_c/Program Files/MetaTrader 5/terminal64.exe")
else:
    WINE_DIR = Path(__file__).resolve().parent.parent / "wine"
    MT5_PATH = str(WINE_DIR / "drive_c/Program Files/MetaTrader 5/terminal64.exe")

# Kill existing terminals
if platform.system() == "Linux":
    subprocess.run(["pkill", "-f", "terminal64.exe"], capture_output=True)
else:
    subprocess.run(["taskkill", "/F", "/IM", "terminal64.exe"], capture_output=True)
time.sleep(3)

# Clear corrupt lock files
appdata = (
    WINE_DIR
    / "drive_c/users"
    / (subprocess.run(["whoami"], capture_output=True, text=True).stdout.strip() or "root")
    / "AppData/Roaming/MetaQuotes/Terminal"
)
if appdata.exists():
    for d in appdata.iterdir():
        if d.is_dir():
            for f in d.iterdir():
                if f.name.startswith("."):
                    f.unlink(missing_ok=True)

print("Lock files cleared. Starting terminal...")
proc = (
    subprocess.Popen(["wine", MT5_PATH])
    if platform.system() == "Linux"
    else subprocess.Popen([MT5_PATH])
)
time.sleep(20)

import configparser  # noqa: E402

import MetaTrader5 as mt5  # noqa: E402

creds = configparser.ConfigParser()
creds.read(Path(__file__).resolve().parent.parent / "config" / "credentials.ini")

import os  # noqa: E402

ok = mt5.initialize(
    path=MT5_PATH,
    login=int(os.getenv("MT5_ACCOUNT") or creds["LOGIN"]["account"]),
    password=os.getenv("MT5_PASSWORD") or creds["LOGIN"]["password"],
    server=os.getenv("MT5_SERVER") or creds["LOGIN"]["server"],
    timeout=60000,
)
if ok:
    acc = mt5.account_info()
    print(f"Connected: balance={acc.balance} currency={acc.currency}")
else:
    print(f"Init failed: {mt5.last_error()}", file=sys.stderr)
    mt5.shutdown()
    proc.terminate()
    sys.exit(1)
