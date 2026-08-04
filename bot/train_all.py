import subprocess
from datetime import datetime
from pathlib import Path

PYTHON = str(Path(__file__).resolve().parent.parent / "wine/drive_c/Program Files/Python312/python.exe")
ENV = {**dict(WINEPREFIX=str(Path(__file__).resolve().parent.parent / "wine"), WINEARCH="win64", DISPLAY=":99")}

symbols = [
    ("XAU500.raw", 1.5, 2.5),
    ("EURJPY.raw", 1.5, 2.0),
    ("NZDUSD.raw", 1.5, 2.0),
    ("USDJPY.raw", 1.5, 2.5),
    ("GBPJPY.raw", 1.5, 2.5),
    ("US500.raw", 1.5, 2.0),
    ("DOGUSD.raw", 1.0, 3.0),
    ("LTCUSD.raw", 1.0, 2.0),
]

for sym, sl, tp in symbols:
    print(f"\n{'=' * 60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training {sym} (SL={sl}, TP={tp})")
    print(f"{'=' * 60}")
    result = subprocess.run(
        [
            "wine",
            PYTHON,
            "-u",
            str(Path(__file__).resolve().parent / "train_model.py"),
            "--symbols",
            sym,
            "--sl-atr",
            str(sl),
            "--tp-atr",
            str(tp),
            "--max-hold",
            "12",
            "--years",
            "2",
            "--prune",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=ENV,
    )
    print(result.stdout)
    if result.stderr:
        lines = [ln for ln in result.stderr.split("\n") if "err:menubuilder" not in ln and "fixme" not in ln]
        if lines:
            print("\n".join(lines[-300:]))
    if result.returncode != 0:
        print(f"FAILED {sym} (rc={result.returncode})")
    else:
        print(f"DONE {sym}")
    # Check if the model was saved
    model_path = Path(__file__).resolve().parent.parent / f"models/model_{sym.replace('.', '_')}.pkl"
    if model_path.exists():
        print(f"  Model saved: {model_path} ({model_path.stat().st_size / 1024:.0f} KB)")
