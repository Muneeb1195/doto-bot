#!/home/muneeb/doto-mt5-bot/.venv_news/bin/python3
"""News sentiment — Marketaux primary, RSS fallback."""
import sys

if getattr(sys, "_base_executable", sys.executable) != sys.executable:
    setattr(sys, "_base_executable", sys.executable)
import os

os.environ["JOBLIB_PARALLEL_BACKEND"] = "threading"

import json
import logging
import logging.handlers
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
STATE_FILE = DATA_DIR / "news_sentiment.json"
STATE_FILE_TMP = STATE_FILE.with_suffix(".json.tmp")
CREDENTIALS_FILE = BASE_DIR / "config" / "credentials.ini"

log_handler = logging.handlers.TimedRotatingFileHandler(
    LOG_DIR / "news_sentiment.log", when="midnight", backupCount=14, utc=True,
)
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.basicConfig(level=logging.INFO, handlers=[log_handler, logging.StreamHandler()])

def _load_portfolio_symbols():
    """Derive watched symbols from [PORTFOLIO] in settings.ini so the news
    service stays in sync with what the bot actually trades."""
    try:
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(BASE_DIR / "config" / "settings.ini")
        raw = cfg.get("PORTFOLIO", "symbols", fallback="")
        return [s.strip() for s in raw.split(",") if s.strip()]
    except Exception:
        logging.warning("Could not load [PORTFOLIO] symbols from settings.ini; using fallback")
        return ["BTCUSD.raw", "US30.raw", "GBPJPY.raw", "SOLUSD.raw", "XRPUSD.raw"]


SYMBOLS = _load_portfolio_symbols()

MARKETAUX_SYMBOL_MAP = {
    "XAU500.raw": "XAUUSD",
    "BTCUSD.raw": "CC:BTC",
    "NZDUSD.raw": "NZDUSD",
    "US30.raw": "DJIA",
    "GBPJPY.raw": "GBPJPY",
    "SOLUSD.raw": "CC:SOL",
    "XRPUSD.raw": "CC:XRP",
    "DOGUSD.raw": "CC:DOGE",
}

MARKETAUX_URL = "https://api.marketaux.com/v1/news/all"
POLL_INTERVAL = 900
WINDOW_HOURS = 6
TIMEOUT = 15

RSS_FEEDS = [
    ("https://finance.yahoo.com/news/rssindex", "equities"),
    ("https://www.investing.com/rss/news_1.rss", "forex"),
    ("https://www.investing.com/rss/news_11.rss", "commodities"),
    ("https://www.investing.com/rss/news_14.rss", "economy"),
    ("https://feeds.marketwatch.com/marketwatch/topstories/", "general"),
]


def load_api_key():
    try:
        import configparser
        creds = configparser.ConfigParser()
        creds.read(CREDENTIALS_FILE)
        return creds.get("MARKETAUX", "api_key", fallback="")
    except Exception:
        return os.getenv("MARKETAUX_API_KEY", "")


def fetch_marketaux(api_key, symbols_str):
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "api_token": api_key,
        "symbols": symbols_str,
        "filter_entities": "true",
        "limit": 50,
        "published_after": cutoff,
        "language": "en",
    }
    try:
        r = requests.get(MARKETAUX_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        articles = data.get("data", [])
        logging.info(f"Marketaux returned {len(articles)} articles for [{symbols_str}]")
        return articles
    except Exception as e:
        logging.warning(f"Marketaux fetch failed: {e}")
        return None


def parse_marketaux_symbols(article, symbol_map):
    matched = {}
    for entity in article.get("entities", []):
        raw_sym = entity.get("symbol", "").upper()
        es = entity.get("sentiment_score")
        for our_sym, mkt_sym in symbol_map.items():
            if mkt_sym.upper() == raw_sym or raw_sym.endswith(mkt_sym.upper()):
                matched[our_sym] = matched.get(our_sym, []) + [es if es is not None else 0.0]
    os_score = article.get("overall_sentiment_score")
    if os_score is not None:
        for our_sym, mkt_sym in symbol_map.items():
            headline = f"{article.get('title', '')} {article.get('description', '')}"
            if (
                mkt_sym.upper().lower() in headline.lower()
                or article.get('source', '').lower() in headline.lower()
            ) and our_sym not in matched:
                matched[our_sym] = [os_score]
    return matched


def fetch_rss(url):
    try:
        r = requests.get(url, timeout=TIMEOUT, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        return r.text
    except Exception as e:
        logging.warning(f"RSS fetch failed: {url} — {e}")
        return None


def parse_rss(xml_text):
    items = []
    try:
        root = ET.fromstring(xml_text)
        for item in root.iter("item"):
            title = item.findtext("title", "")
            desc = item.findtext("description", "")
            pub_date_str = item.findtext("pubDate", "")
            combined = f"{title} {desc}"
            pub_date = datetime.now()
            if pub_date_str:
                for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S", "%Y-%m-%dT%H:%M:%S"]:
                    try:
                        pub_date = datetime.strptime(pub_date_str.rsplit(" ", 1)[0], fmt)
                        break
                    except ValueError:
                        continue
            items.append({"text": combined, "date": pub_date, "title": title})
    except ET.ParseError as e:
        logging.warning(f"RSS parse error: {e}")
    return items


TICKER_MAP_RSS = {  # minimal fallback for RSS
    "xauusd": "XAU500.raw", "gold": "XAU500.raw", "xau": "XAU500.raw",
    "btc": "BTCUSD.raw", "bitcoin": "BTCUSD.raw",
    "nzdusd": "NZDUSD.raw",
    "dow jones": "US30.raw", "djia": "US30.raw", "us30": "US30.raw",
    "gbpjpy": "GBPJPY.raw",
    "sol": "SOLUSD.raw", "solana": "SOLUSD.raw",
    "xrp": "XRPUSD.raw", "ripple": "XRPUSD.raw",
    "doge": "DOGUSD.raw", "dogecoin": "DOGUSD.raw",
}


def match_tickers_rss(text):
    matched = set()
    lower = text.lower()
    for keyword, symbol in TICKER_MAP_RSS.items():
        import re
        if re.search(r"\b" + re.escape(keyword) + r"\b", lower):
            matched.add(symbol)
    return matched


def _parse_ts(published):
    """Best-effort parse of an ISO-8601 timestamp to a naive datetime."""
    if not published:
        return None
    try:
        dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (ValueError, TypeError):
        return None


def main():
    api_key = load_api_key()

    def marketaux_ready():
        return api_key and (not hasattr(marketaux_ready, "skip") or marketaux_ready.skip <= 0)

    accumulated = defaultdict(list)

    while True:
        now = datetime.now()
        cutoff = now - timedelta(hours=WINDOW_HOURS)

        # Prune entries older than the rolling window and dedupe against what is
        # already retained. Previously `accumulated` grew without bound and the
        # same article (returned by every poll) was appended repeatedly, which
        # leaked memory and inflated both the article count and the averaged
        # sentiment score (agent audit M7). Keys are per-symbol so an article
        # matching multiple symbols still counts once per symbol.
        for sym in list(accumulated.keys()):
            accumulated[sym] = [e for e in accumulated[sym] if e.get("ts", now) >= cutoff]
            if not accumulated[sym]:
                del accumulated[sym]
        seen_keys = {e["key"] for entries in accumulated.values() for e in entries}

        if marketaux_ready():
            symbols_str = ",".join(MARKETAUX_SYMBOL_MAP.values())
            articles = fetch_marketaux(api_key, symbols_str)
            if articles:
                for art in articles:
                    matched = parse_marketaux_symbols(art, MARKETAUX_SYMBOL_MAP)
                    art_id = art.get("uuid") or f"{art.get('title', '')}|{art.get('published_at', '')}"
                    ts = _parse_ts(art.get("published_at", "")) or now
                    for sym, scores in matched.items():
                        key = f"{sym}|mkt|{art_id}"
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        avg_s = sum(scores) / len(scores) if scores else 0.0
                        accumulated[sym].append({
                            "score": avg_s,
                            "title": art.get("title", ""),
                            "published": art.get("published_at", ""),
                            "ts": ts,
                            "key": key,
                        })
                if not any(accumulated.values()):
                    logging.info("Marketaux returned no matched entities")
                marketaux_ready.skip = 0
            else:
                logging.warning("Marketaux failed — will retry in ~45 min")
                marketaux_ready.skip = 3

        if not marketaux_ready() or not any(accumulated.values()):
            for url, source in RSS_FEEDS:
                xml = fetch_rss(url)
                if xml is None:
                    continue
                items = parse_rss(xml)
                for item in items:
                    if item["date"] < cutoff:
                        continue
                    tickers = match_tickers_rss(item["text"])
                    if not tickers:
                        continue
                    for sym in tickers:
                        key = f"{sym}|rss|{item['title']}"
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        accumulated[sym].append({
                            "score": 0.0,
                            "title": item["title"],
                            "published": "",
                            "ts": item["date"] or now,
                            "key": key,
                        })

        output_symbols = {}
        for sym in SYMBOLS:
            entries = accumulated.get(sym, [])
            if entries:
                scores = [e["score"] for e in entries]
                avg_score = sum(scores) / len(scores)
                output_symbols[sym] = {
                    "score": round(avg_score, 4),
                    "count": len(entries),
                    "updated": now.isoformat(),
                }
            else:
                output_symbols[sym] = {"score": 0.0, "count": 0, "updated": now.isoformat()}

        output = {
            "updated": now.isoformat(),
            "window_hours": WINDOW_HOURS,
            "symbols": output_symbols,
        }

        try:
            with open(STATE_FILE_TMP, "w") as f:
                json.dump(output, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(STATE_FILE_TMP, STATE_FILE)
            n_total = sum(v["count"] for v in output_symbols.values())
            logging.info(f"Wrote {len(output_symbols)} symbols ({n_total} articles) to {STATE_FILE}")
        except Exception as e:
            logging.error(f"Failed to write state: {e}")

        if hasattr(marketaux_ready, "skip") and marketaux_ready.skip > 0:
            marketaux_ready.skip -= 1

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
