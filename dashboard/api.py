import sys

if getattr(sys, "_base_executable", sys.executable) != sys.executable:
    sys._base_executable = sys.executable
import os

os.environ["JOBLIB_PARALLEL_BACKEND"] = "threading"

import csv
import json
import os
import re
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "data" / "dashboard_state.json"
TRADES_CSV = BASE_DIR / "logs" / "trades.csv"
LOG_DIR = BASE_DIR / "logs"

security = HTTPBasic()

_dashboard_user: str | None = os.getenv("DASHBOARD_USER")
_dashboard_pass: str | None = os.getenv("DASHBOARD_PASS")
if not _dashboard_user or not _dashboard_pass:
    raise RuntimeError(
        "DASHBOARD_USER and DASHBOARD_PASS environment variables must be set. "
        "Export them before starting the dashboard."
    )
_dashboard_user_s: str = _dashboard_user
_dashboard_pass_s: str = _dashboard_pass

def verify_auth(creds: HTTPBasicCredentials = Depends(security)):
    is_user = secrets.compare_digest(creds.username, _dashboard_user_s)
    is_pass = secrets.compare_digest(creds.password, _dashboard_pass_s)
    if not (is_user and is_pass):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Basic"},
        )

app = FastAPI(title="Doto MT5 Bot Dashboard", dependencies=[Depends(verify_auth)])
TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"

_cache = {"state": None, "trades": [], "state_mtime": 0.0, "trades_mtime": 0.0}

def _load_state_raw():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def _load_trades_raw():
    try:
        with open(TRADES_CSV) as f:
            reader = csv.DictReader(f)
            rows = []
            for row in reader:
                row["pnl"] = float(row["pnl"]) if row.get("pnl") else 0.0
                row["pips"] = float(row["pips"]) if row.get("pips") else 0.0
                rows.append(row)
            rows.sort(key=lambda r: r.get("exit_time", "") or "", reverse=True)
            return rows
    except (FileNotFoundError, KeyError):
        return []

def load_state():
    try:
        mtime = STATE_FILE.stat().st_mtime
    except FileNotFoundError:
        _cache["state"] = None
        _cache["state_mtime"] = 0.0
        return None
    if mtime != _cache["state_mtime"]:
        _cache["state"] = _load_state_raw()
        _cache["state_mtime"] = mtime
    return _cache["state"]

def load_trades(limit=500):
    try:
        mtime = TRADES_CSV.stat().st_mtime
    except FileNotFoundError:
        _cache["trades"] = []
        _cache["trades_mtime"] = 0.0
        return []
    if mtime != _cache["trades_mtime"]:
        _cache["trades"] = _load_trades_raw()
        _cache["trades_mtime"] = mtime
    return _cache["trades"][:limit]

@app.get("/api/state")
def get_state():
    return load_state() or {}

@app.get("/api/trades")
def get_trades():
    return load_trades()

@app.get("/api/log")
def get_log():
    try:
        log_path = LOG_DIR / "bot.log"
        if not log_path.exists():
            return {"lines": ["No bot.log found"]}
        pat = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \[(INFO|WARNING|ERROR)\]")
        # Tail only the last chunk of the file — the log grows to several MB and
        # is polled every few seconds, so reading the whole file each time is
        # wasteful (agent audit L1).
        tail_bytes = 262144
        size = log_path.stat().st_size
        with open(log_path, "rb") as f:
            if size > tail_bytes:
                f.seek(size - tail_bytes)
                f.readline()  # drop the partial first line
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        lines = [line for line in text.splitlines() if pat.match(line)][-60:]
        return {"lines": lines}
    except Exception as e:
        return {"lines": [f"Log error: {e}"]}

@app.get("/", response_class=HTMLResponse)
def index():
    if TEMPLATE_PATH.exists():
        return HTMLResponse(TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<html><body><h1>Template not found</h1></body></html>")
