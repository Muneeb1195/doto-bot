import logging
import time

_drift_warmstart_queue: dict[str, float] = {}
_drift_last_check: dict[str, float] = {}
_DRIFT_COOLDOWN = 3600


def schedule_warmstart(symbol):
    global _drift_warmstart_queue
    _drift_warmstart_queue[symbol] = time.time()
    logging.info(f"[{symbol}] Warm-start queued (will run next idle cycle)")


def consume_warmstart_queue():
    global _drift_warmstart_queue
    pending = list(_drift_warmstart_queue.keys())
    _drift_warmstart_queue = {}
    return pending
