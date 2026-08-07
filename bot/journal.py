"""Trade journal — CSV read/write/reconcile."""

import contextlib
import csv
import logging
import os
import time
from datetime import datetime

try:
    import MetaTrader5 as mt5
except ImportError:  # Linux: no native package, use the socket/RPyC bridge
    from mt5_connect import mt5
import state as _st
from mt5_connect import mt5_call
from state import TRADE_HEADERS

_JOURNAL_CACHE = {"mtime": 0, "rows": None}

# External-deal reconciliation fetches the full account deal history and rewrites
# the journal CSV, so it must NOT run on every 10s trade cycle. Throttle it to at
# most once every 5 minutes (agent audit M8). force=True bypasses the throttle
# (used on shutdown / explicit resync).
_RECONCILE_INTERVAL = 300.0
_last_external_reconcile = 0.0

# Seed/demo OPEN rows left over from earlier testing (improbable round prices
# / known test tickets). These are not real trades and are pruned on init so
# the journal reflects only genuine activity.
_SEED_TICKETS = {"1001", "1002", "25603336"}


def _clear_cache():
    _JOURNAL_CACHE["mtime"] = 0
    _JOURNAL_CACHE["rows"] = None


def _read_csv():
    path = _st.TRADE_CSV
    if not path.exists():
        return []
    try:
        mtime = path.stat().st_mtime
        if _JOURNAL_CACHE["rows"] is not None and mtime <= _JOURNAL_CACHE["mtime"]:
            return _JOURNAL_CACHE["rows"]
        rows = []
        skipped = 0
        # Read defensively row-by-row so a single corrupt/partial line (e.g.
        # from a process killed mid-append) is quarantined rather than
        # aborting the whole read. Returning a truncated-to-empty list here
        # would cause reconcile_journal to atomically overwrite (wipe) the
        # entire journal (see agent audit C2). We only keep well-formed rows.
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                return []
            ncols = len(header)
            line_no = 1
            while True:
                line_no += 1
                try:
                    raw = next(reader)
                except StopIteration:
                    break
                except csv.Error:
                    skipped += 1
                    continue
                # A truncated final append yields a short row; drop it.
                if len(raw) != ncols:
                    skipped += 1
                    logging.warning(f"journal: skipping malformed line {line_no} ({len(raw)} cols, expected {ncols})")
                    continue
                rows.append(dict(zip(header, raw)))
        if skipped:
            logging.warning(f"journal: quarantined {skipped} corrupt row(s) on read")
        _JOURNAL_CACHE["rows"] = rows
        _JOURNAL_CACHE["mtime"] = mtime
        return rows
    except Exception:
        # Signal an unreadable file distinctly from an empty one so callers
        # that rewrite the journal can bail out instead of wiping it.
        logging.warning("journal: read failed", exc_info=True)
        return None


def _prune_seed_rows():
    """Remove leftover demo/seed OPEN rows so history is clean."""
    path = _st.TRADE_CSV
    if not path.exists():
        return
    try:
        rows = _read_csv()
        if not rows:
            return
        kept = [r for r in rows if str(r.get("ticket", "")).strip() not in _SEED_TICKETS]
        if len(kept) == len(rows):
            return
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
            w.writeheader()
            w.writerows(kept)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        _clear_cache()
        logging.info(f"Pruned {len(rows) - len(kept)} seed/demo journal rows")
    except Exception:
        logging.warning("Seed row prune failed", exc_info=True)


def _reconcile_external_deals(force=False):
    """Backfill trades the bot did not place (manual / external EAs) so the
    journal reflects ALL account activity, not just bot trades. External deals
    are identified by magic != bot magic and recorded once with event=EXTERNAL.
    Bot trades are already journaled via journal_open/journal_close and are
    never double-counted.

    Throttled to at most once per _RECONCILE_INTERVAL seconds because it pulls
    the full deal history and rewrites the CSV; pass force=True to override.
    """
    global _last_external_reconcile
    now_mono = time.monotonic()
    if not force and (now_mono - _last_external_reconcile) < _RECONCILE_INTERVAL:
        return
    _last_external_reconcile = now_mono
    path = _st.TRADE_CSV
    if not path.exists():
        return
    try:
        rows = _read_csv()
        if rows is None:
            return
        known_tickets = {str(r.get("ticket", "")).strip() for r in rows if str(r.get("ticket", "")).strip()}
        bot_magic = _st.BOT_MAGIC
        deals = mt5_call(mt5.history_deals_get, 0, int(datetime.now().timestamp()) + 10, _timeout=10)
        if not deals:
            return
        # Group deals by position ticket, keep entry + exit.
        positions = {}
        for d in deals:
            t = str(getattr(d, "position_id", 0) or 0)
            if not t or t == "0":
                continue
            positions.setdefault(t, []).append(d)
        appended = []
        for ticket, dlist in positions.items():
            if ticket in known_tickets or ticket in _st._imported_external_ids:
                continue
            # Skip if any deal in this position is a bot trade (magic match).
            if any((getattr(d, "magic", None) == bot_magic) for d in dlist):
                continue
            entry = next((d for d in dlist if d.type in (0, 1)), dlist[0])
            # Find actual closing deal (opposite type to entry) instead of dlist[-1]
            close_type = 1 - entry.type  # buy→0 (sell to close), sell→1 (buy to close)
            closing = None
            for d in reversed(dlist):
                if d.type == close_type:
                    closing = d
                    break
            exit_deal = closing if closing else dlist[-1]
            net = sum(
                (getattr(d, "profit", 0) or 0) + (getattr(d, "commission", 0) or 0) + (getattr(d, "swap", 0) or 0)
                for d in dlist
            )
            sym = getattr(entry, "symbol", "")
            side = "buy" if entry.type == 0 else "sell"
            try:
                entry_price = float(entry.price)
            except (ValueError, TypeError):
                entry_price = 0.0
            try:
                exit_price = float(exit_deal.price)
            except (ValueError, TypeError):
                exit_price = 0.0
            vol = getattr(entry, "volume", 0.0)
            try:
                pips = (exit_price - entry_price) * (1 if side == "buy" else -1)
            except Exception:
                pips = 0.0
            etime = datetime.fromtimestamp(exit_deal.time).strftime("%Y-%m-%d %H:%M:%S")
            appended.append(
                {
                    "ticket": ticket,
                    "symbol": sym,
                    "type": side,
                    "volume": f"{vol:.2f}",
                    "entry_price": f"{entry_price:.5f}",
                    "sl": "",
                    "tp": "",
                    "entry_time": "",
                    "atr": "",
                    "exit_price": f"{exit_price:.5f}",
                    "exit_time": etime,
                    "pnl": f"{net:.2f}",
                    "pips": f"{pips:.1f}",
                    "event": "EXTERNAL",
                }
            )
        if not appended:
            return
        tmp = path.with_suffix(".csv.tmp")
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_HEADERS)
            w.writeheader()
            w.writerows(rows)
            w.writerows(appended)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # Only external deals CLOSED today should affect the daily-loss counter.
        # Adding all-time external PnL here (deals fetched from epoch 0) caused
        # false daily-loss halts. The set of already-backfilled external ids is
        # now persisted in bot_state.json (imported_external_ids) and reloaded
        # on startup, so same-day restarts no longer re-count the same deals
        # (agent audit C4 — fixed).
        today = datetime.now().date()
        _reset_daily_pnl_if_new_day()
        for a in appended:
            try:
                closed_today = datetime.strptime(a.get("exit_time", ""), "%Y-%m-%d %H:%M:%S").date() == today
            except (ValueError, TypeError):
                closed_today = False
            if closed_today:
                with contextlib.suppress(ValueError, TypeError):
                    _st._daily_realized_pnl += float(a.get("pnl", 0))
            _st._imported_external_ids.add(a.get("ticket", ""))
        _clear_cache()
        logging.info(f"Backfilled {len(appended)} external (non-bot) trade(s) into journal")
    except Exception:
        logging.warning("External deal reconciliation failed", exc_info=True)


def journal_init():
    if not _st.TRADE_CSV.exists():
        with open(_st.TRADE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TRADE_HEADERS)
    _prune_seed_rows()
    _clear_cache()


def journal_open(ticket, sym, side, vol, entry, sl, tp, atr):
    with open(_st.TRADE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                ticket,
                sym,
                side,
                vol,
                f"{entry:.5f}",
                f"{sl:.5f}",
                f"{tp:.5f}",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                f"{atr:.4f}",
                "",
                "",
                "",
                "",
                "OPEN",
            ]
        )
        f.flush()
        os.fsync(f.fileno())
    _clear_cache()


def _reset_daily_pnl_if_new_day():
    today = datetime.now().date()
    if _st._daily_realized_date != today:
        _st._daily_realized_pnl = 0.0
        _st._daily_realized_date = today


def journal_close(ticket, exit_price, pnl, pips, event="CLOSE"):
    _reset_daily_pnl_if_new_day()
    exit_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Copy entry-side fields from the OPEN row for a self-documenting close row
    entry_price = entry_time = volume = side = ""
    try:
        rows = _read_csv()
        if rows:
            for r in rows:
                if r.get("event") == "OPEN" and str(r.get("ticket", "")).strip() == str(ticket):
                    entry_price = r.get("entry_price", "")
                    entry_time = r.get("entry_time", "")
                    volume = r.get("volume", "")
                    side = r.get("type", "")
                    break
    except Exception:
        pass
    row = [
        str(ticket),
        "",
        side,
        volume,
        entry_price,
        "",
        "",
        entry_time,
        "",
        f"{exit_price:.5f}",
        exit_ts,
        f"{pnl:.2f}",
        f"{pips:.1f}",
        event,
    ]
    with open(_st.TRADE_CSV, "a", newline="") as f:
        csv.writer(f).writerow(row)
        f.flush()
        os.fsync(f.fileno())
    _st._daily_realized_pnl += pnl
    _clear_cache()


def reconcile_journal(active_tickets):
    if not _st.TRADE_CSV.exists():
        return
    rows = _read_csv()
    if not rows:
        return

    active_set = set(active_tickets)

    # A ticket is already "resolved" in the journal if it has any non-OPEN
    # row (CLOSE / SCALE_OUT / CHANDELIER / REVERSAL / MR_EXIT / MANUAL_CLOSE /
    # *NAKED_CLOSE). We must NOT append another MANUAL_CLOSE for those, or every
    # closed trade would be double-counted each cycle (see agent audit C1).
    resolved_tickets = set()
    for row in rows:
        ev = row.get("event", "")
        if ev and ev != "OPEN":
            with contextlib.suppress(ValueError, TypeError):
                resolved_tickets.add(int(row["ticket"]))

    # An OPEN row is an orphan only if its position is gone AND we have not
    # already recorded a close for it.
    orphan_tickets = set()
    for row in rows:
        is_open = row.get("event") == "OPEN" or row.get("pips") == "OPEN"
        if not is_open:
            continue
        try:
            ticket = int(row["ticket"])
        except (ValueError, TypeError):
            continue
        if ticket not in active_set and ticket not in resolved_tickets:
            orphan_tickets.add(ticket)

    if not orphan_tickets:
        return

    # Filter out old MANUAL_CLOSE rows for orphaned tickets, keep everything else
    kept = []
    for row in rows:
        try:
            ticket = int(row["ticket"])
        except (ValueError, TypeError):
            ticket = 0
        is_manual = row.get("event") == "MANUAL_CLOSE"
        if is_manual and ticket in orphan_tickets:
            continue  # drop stale MANUAL_CLOSE
        kept.append(row)

    exit_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for ticket in orphan_tickets:
        exit_price = 0.0
        pnl = 0.0
        pips = 0.0
        try:
            deals = mt5_call(mt5.history_deals_get, position=ticket, _timeout=10)
            if deals is not None and len(deals) > 0:
                close_deal = deals[-1]
                exit_price = close_deal.price
                pnl = close_deal.profit
                entry = 0.0
                mult = 1
                # Try to find the OPEN row for entry price
                for row in rows:
                    if int(row["ticket"]) == ticket and (row.get("event") == "OPEN" or row.get("pips") == "OPEN"):
                        try:
                            entry = float(row["entry_price"]) if row["entry_price"] else 0
                            mult = 1 if row["type"] == "buy" else -1
                        except (ValueError, TypeError):
                            pass
                        break
                pips = (close_deal.price - entry) * mult
        except Exception:
            logging.warning(f"Reconcile failed for ticket {ticket}, using defaults", exc_info=True)
        kept.append(
            {
                "ticket": str(ticket),
                "symbol": "",
                "type": "",
                "volume": "",
                "entry_price": "",
                "sl": "",
                "tp": "",
                "entry_time": "",
                "atr": "",
                "exit_price": f"{exit_price:.5f}",
                "exit_time": exit_ts,
                "pnl": f"{pnl:.2f}",
                "pips": f"{pips:.1f}",
                "event": "MANUAL_CLOSE",
            }
        )

    tmp = _st.TRADE_CSV.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_st.TRADE_HEADERS)
        writer.writeheader()
        writer.writerows(kept)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, _st.TRADE_CSV)
    _clear_cache()
