# services/

## Responsibility
Standalone detached service — news sentiment polling (not part of the `bot/` import graph). Periodically fetches market news headlines, scores per-symbol sentiment, and publishes `data/news_sentiment.json` for live scoring and confidence adjustment. Runs as `systemd doto-news.service` on the home-server, consuming the live `[PORTFOLIO] symbols` and `config/credentials.ini` MARKETAUX key.

## Design
- **Isolated process + file contract**: Own `BASE_DIR:23` derived from `Path(__file__)`, own `.venv_news` (shebang `#!/.../.venv_news/bin/python3:1`, `JOBLIB_PARALLEL_BACKEND=threading:9`), own rotating log `logs/news_sentiment.log:30` (14 day UTC). No import of `bot/` — communicates solely via atomic JSON `data/news_sentiment.json:26` (`STATE_FILE_TMP:.json.tmp + flush/fsync/replace:294`).
- **Primary path — Marketaux** (`MARKETAUX_URL https://api.marketaux.com/v1/news/all:63`, `MARKETAUX_SYMBOL_MAP:52` normalizes `XAU500.raw→XAUUSD`, `BTCUSD.raw→CC:BTC` etc.): `fetch_marketaux:87` (`GET ?api_token&symbols&filter_entities&limit=50&published_after=cutoff&language=en`, `WINDOW_HOURS=6:65`, `TIMEOUT 15`) → `parse_marketaux_symbols:109` (match `entity.symbol` exact/suffix + `sentiment_score`, fallback `overall_sentiment + headline` substring) → `accumulated: defaultdict(list):203` keyed `"{sym}|mkt|{uuid|title|published}"`.
- **Fallback — RSS** (`RSS_FEEDS:68` Yahoo/Investing/MarketWatch, 5 feeds): `fetch_rss:129` (`requests + Mozilla/5.0`) → `parse_rss:139` (`xml.etree` `<item><title/description/pubDate>` loop, 3 date fmts) → `match_tickers_rss:174` (`TICKER_MAP_RSS:162` regex `\bkeyword\b` case-insensitive, e.g. `gold→XAU500.raw`, `btc/bitcoin→BTCUSD.raw`). Scores `0.0` (neutral) for RSS — only Marketaux carries `sentiment_score`.
- **Dedupe + windowing** (`main:197` loop `POLL_INTERVAL 900 s:64`): prune `accumulated[sym]` to `cutoff=now-WINDOW_HOURS` (fixes unbounded growth/leaked memory M7), `seen_keys` set across both feeds, `marketaux_ready.skip` back-off `3` cycles (~45 min) on failure `246`.
- **Self-sync portfolio** (`_load_portfolio_symbols:36` reads `[PORTFOLIO] symbols` from `settings.ini` fallback 5 symbols, executed at import `SYMBOLS= _load_portfolio_symbols():50` so symbol set tracks bot config without code change).
- **Output schema** `274`: `{"updated":ISO, "window_hours":6, "symbols":{"BTCUSD.raw":{"score":avg±1, "count":n, "updated":ISO}}}` — `avg_score = sum(scores)/len(scores)` per symbol, else `0.0/0`. Consumer `bot/state.load_news_sentiment:357` caches by `mtime`, `bot/analytics.compute_entry_score:377` maps `(score+1)/2 →news_score` then `1-news_score` for sells, weighted `news 30%`.
- **Credentials**: `load_api_key:77` (`config/credentials.ini [MARKETAUX] api_key` or `MARKETAUX_API_KEY` env), git-ignored.

## Flow
1. `main:197` loop → prune `accumulated` by `cutoff` → `seen_keys` snapshot.
2. If `marketaux_ready()` (key present, `skip==0`): build `symbols_str` from `MARKETAUX_SYMBOL_MAP.values()` → `fetch_marketaux` → for each article parse matched symbols → `ts=_parse_ts:184` (ISO `Z→+00:00`→utc→naive) → append `{score,title,published,ts,key}` if not dup. On empty: log `no matched entities`; on `None` set `skip=3`.
3. If no Marketaux or no accumulations: iterate `RSS_FEEDS` → `fetch_rss`→`parse_rss`→ drop `date<cutoff`→`match_tickers_rss`→ append `score 0.0` per ticker.
4. Build `output_symbols` for every `SYMBOLS` (always emitted, even zero) → atomic write `STATE_FILE` (info `Wrote N symbols (total articles)`).
5. Decrement `skip` → `sleep 900`.

## Integration
- **Produces:** `data/news_sentiment.json` — consumed by `bot/state.load_news_sentiment:357` (mtime cache) → `bot/analytics.compute_entry_score:381` (news 30%) and `bot/filters.apply_news_confidence_mult` (`news≥0.7→×1.10 cap1.5`, `≤0.3→×0.50`).
- **Depends on:** `config/settings.ini [PORTFOLIO]` (watched symbols), `config/credentials.ini [MARKETAUX]` (primary, fallback env), outbound `api.marketaux.com` + 5 RSS hosts.
- **Runtime:** `systemd doto-news.service` (`scripts/deploy-linux.sh` Phase 5) ExecStart `.venv_news/bin/python services/news_sentiment.py`, no CI ownership (home-server only).
- **Sibling:** `dashboard/` also file-contracts via `data/dashboard_state.json`; `services/__init__.py` empty.
