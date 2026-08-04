#!/usr/bin/env python3
"""Headless service launcher for the Doto MT5 Bot.

Replaces the PowerShell ``while``-loop launchers so that no visible console
window is spawned for Interactive (user-session) scheduled tasks, which ignore
both the task ``Hidden`` flag and ``-WindowStyle Hidden``.

The launcher itself runs under ``pythonw.exe`` (GUI subsystem -> no console)
and spawns each child under ``pythonw.exe`` with ``CREATE_NO_WINDOW``, so the
entire service tree is invisible. This also reduces DWM/GPU compositing load,
which helps with display-driver TDR (Timeout Detection & Recovery) crashes.

Usage::

    pythonw.exe svc_launcher.py {bot|dashboard|news}
"""
import configparser
import contextlib
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO, "logs")
CREDENTIALS_FILE = os.path.join(REPO, "config", "credentials.ini")
RESTART_DELAY = 5
# Watchdog: when the scheduler ends the task (schtasks /End), kill the child we
# own so it does not orphan and keep trading after a restart spawns a new one.
_PID_DIR = LOG_DIR

# pythonw executable per service (news uses its own venv).
_PYTHON = {
    "bot": os.path.join(REPO, ".venv", "Scripts", "pythonw.exe"),
    "dashboard": os.path.join(REPO, ".venv", "Scripts", "pythonw.exe"),
    "news": os.path.join(REPO, ".venv_news", "Scripts", "pythonw.exe"),
}

# child argument list per service (relative to REPO cwd).
_TARGET = {
    "bot": ["bot/main.py"],
    "dashboard": ["-m", "uvicorn", "dashboard.api:app", "--host", "127.0.0.1", "--port", "8501"],
    "news": ["services/news_sentiment.py"],
}


def _dashboard_creds_from_file():
    cfg = configparser.ConfigParser()
    try:
        cfg.read(CREDENTIALS_FILE)
    except (configparser.Error, OSError):
        return None, None
    if cfg.has_section("DASHBOARD"):
        return (cfg.get("DASHBOARD", "user", fallback=None),
                cfg.get("DASHBOARD", "password", fallback=None))
    return None, None


def _build_env(name):
    env = os.environ.copy()
    if name != "dashboard":
        return env
    # Never hardcode dashboard credentials in source (agent audit H4). The
    # dashboard is bound to 127.0.0.1, but credentials must still not live in the
    # repo. Prefer already-exported env vars, else source them from the
    # git-ignored credentials.ini [DASHBOARD] section. If neither supplies them,
    # dashboard/api.py fails loudly on startup rather than running open.
    if not env.get("DASHBOARD_USER") or not env.get("DASHBOARD_PASS"):
        user, password = _dashboard_creds_from_file()
        if user and password:
            env["DASHBOARD_USER"] = user
            env["DASHBOARD_PASS"] = password
    return env


def _log(path, msg):
    try:
        with open(path, "ab") as fh:
            fh.write((time.strftime("%Y-%m-%d %H:%M:%S") + " " + msg + "\n").encode())
            fh.flush()
    except OSError:
        pass


def _kill_proc(pid):
    """Best-effort terminate a process tree on Windows."""
    if pid is None or pid <= 0:
        return
    with contextlib.suppress(OSError):
        # /T kills child processes, /F forces. Ignore "not found" (128/1).
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )


def _child_pid_file(name):
    return os.path.join(_PID_DIR, f"{name}.child.pid")


def _read_prev_child_pid(name):
    """Read the child PID recorded by the previous launcher instance."""
    try:
        with open(_child_pid_file(name), "r") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _write_child_pid(name, pid):
    try:
        with open(_child_pid_file(name), "w") as fh:
            fh.write(str(pid))
    except OSError:
        pass


def _kill_prev_child(name):
    """Kill the orphaned child left by a previous launcher instance.

    When the scheduler ends the task, the parent launcher is terminated but its
    child (bot/main.py, uvicorn, news) keeps running. The previous launcher
    recorded that child's PID here, so we kill it before starting a fresh one.
    """
    prev = _read_prev_child_pid(name)
    if prev is not None and prev != os.getpid():
        _log(_child_pid_file(name).replace(".child.pid", "_py.err"),
             f"[WATCHDOG] killing previous child pid={prev}")
        _kill_proc(prev)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in _PYTHON:
        sys.stderr.write("usage: svc_launcher.py {bot|dashboard|news}\n")
        sys.exit(2)

    name = sys.argv[1]
    py = _PYTHON[name]
    args = _TARGET[name]
    env = _build_env(name)

    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)

    out_path = os.path.join(LOG_DIR, f"{name}_py.out")
    err_path = os.path.join(LOG_DIR, f"{name}_py.err")

    # Kill any orphaned child from a previous launcher instance of this service
    # before we spawn our own, so a scheduled restart never leaves two copies
    # trading/serving simultaneously.
    _kill_prev_child(name)

    child_pid = [None]

    def _stop_child():
        pid = child_pid[0]
        if pid is not None:
            _log(err_path, f"[WATCHDOG] killing child pid={pid}")
            _kill_proc(pid)

    # Best-effort clean shutdown if the process exits normally. NOTE: Windows
    # Task Scheduler terminating a pythonw (GUI-subsystem) process does NOT
    # reliably deliver SIGTERM to Python's handler, so we do NOT depend on a
    # signal handler here. The reliable guard is the startup stale-child kill
    # above (driven by the .child.pid file) plus the scheduler's tree-kill of
    # the child's own process group on /End.
    try:
        import atexit
        atexit.register(_stop_child)
    except (ValueError, OSError):
        pass
        pass

    while True:
        _log(err_path, f"[INFO] Starting {name} ({py} {' '.join(args)})")
        try:
            with open(out_path, "ab") as fout, open(err_path, "ab") as ferr:
                proc = subprocess.Popen(
                    [py, *args],
                    cwd=REPO,
                    env=env,
                    stdout=fout,
                    stderr=ferr,
                    # Own process group so the watchdog tree-kill (/T) targets only
                    # this child, not the launcher itself.
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
                )
                child_pid[0] = proc.pid
                _write_child_pid(name, proc.pid)
                child_pid[0] = proc.pid
                proc.wait()
                code = proc.returncode
        except Exception as exc:  # noqa: BLE001 - launcher must survive child failures
            _log(err_path, f"[FATAL] {name} launcher error: {exc!r}")
            code = -1

        _log(err_path, f"[INFO] {name} exited (code {code}), restarting in {RESTART_DELAY}s...")
        time.sleep(RESTART_DELAY)


if __name__ == "__main__":
    main()
