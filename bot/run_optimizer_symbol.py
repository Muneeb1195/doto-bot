"""Run optimizer for specific symbols by temporarily modifying settings.ini."""

import configparser
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SETTINGS = BASE / "config" / "settings.ini"
OPTIMIZER = BASE / "bot" / "optimizer.py"
WINEPREFIX = str(BASE / "wine")
PYTHON = str(BASE / "wine/drive_c/Program Files/Python312/python.exe")

symbols = sys.argv[1:] if len(sys.argv) > 1 else ["SPY.raw", "IWM.raw"]

cfg = configparser.ConfigParser()
cfg.read(SETTINGS)
orig = cfg["PORTFOLIO"]["symbols"]

for sym in symbols:
    print(f"\n{'=' * 60}")
    print(f"RUNNING OPTIMIZER FOR {sym}")
    print(f"{'=' * 60}\n")
    cfg["PORTFOLIO"]["symbols"] = sym
    with open(SETTINGS, "w") as f:
        cfg.write(f)

    result = subprocess.run(
        ["wine", PYTHON, "-u", str(OPTIMIZER)],
        cwd=str(BASE),
        capture_output=True,
        timeout=3600,
        env={"WINEPREFIX": WINEPREFIX},
    )
    print(result.stdout.decode("utf-8", errors="replace"))
    if result.stderr:
        print("STDERR:", result.stderr.decode("utf-8", errors="replace")[-2000:])

cfg["PORTFOLIO"]["symbols"] = orig
with open(SETTINGS, "w") as f:
    cfg.write(f)
print("Done.")
