# Forex & gold data sources for the personal Vela backtest app

Research date: **17 November 2026**, tests run from the Linux box around **12:00–12:15 Asia/Jakarta (UTC+7)**. Markets were open (Tuesday). No paid accounts were created. Free-key signup options are documented but were not registered.

Goal: free, legal OHLC (and tick where possible) for **FX majors + XAUUSD**, usable from Python in a personal, non-commercial Streamlit app, with realtime charts and long history for backtests (including an unseen-data split).

---

## Recommendation

### (a) Zero-signup default (use this first)

| Role | Source | Why |
|---|---|---|
| **Long history backtests** (FX + **XAUUSD** spot, M1/H1/D1 and ticks) | **Dukascopy** via `dukascopy-python`, Jetta, or Freeserv | No key. Official metadata: EURUSD tick/M1/H1 from **2003-05-04**, daily from **1973**; XAUUSD tick/M1/H1 from **2003-05-05**. Live-tested from the box. |
| **Realtime / live chart** (FX + XAUUSD) | **biquote.io** REST + SignalR tick stream | No key, no signup. Live quotes with `quoteAgeSeconds` of 1–9s in tests. WebSocket is unmetered. **History is short** — do not use it for multi-year backtests. |
| **Easy majors fallback** | **yfinance** (`EURUSD=X`, etc.) | No key. EURUSD 1h actually returned **~2 years** (2024-01-31 → now); 1m last **7 days**; daily from **2003-12-01**. **No working XAUUSD spot ticker.** Unofficial Yahoo access — ToS risk, can break. |
| **Gold futures fallback** | yfinance `GC=F` | COMEX gold futures, not spot XAUUSD. Daily from 2000-08-30; 1h ~2 years; 1m ~7 days. |
| **Tokenized gold proxy (optional)** | Binance `PAXGUSDT` or Kraken `XAUTUSD` | True realtime crypto APIs, no key. **Not XAUUSD**: 24/7 including weekends, can premia/discount vs spot. PAXG history from 2020-08-28. |

Default wiring for the app: **Dukascopy for any backtest window**, **biquote for the live chart and the current unfinished bar**, yfinance only as a majors fallback if Dukascopy throttles.

Helper: `research_fx_sources.py` (`fetch_ohlc("dukascopy", ...)`, `fetch_ohlc("biquote", ...)`, `fetch_tick("biquote", ...)`).

### (b) Best free-signup option for true realtime + long intraday history

**OANDA v20 Practice API** (free demo account + personal token).

- Official REST candles (M1 through D, bid/ask/mid, **5,000 bars per request**, instrument history described as “Complete”; community/docs examples show EUR_USD from ~2009 and XAU_USD available).
- Official **streaming** prices at `wss/https://stream-fxpractice.oanda.com`.
- Legal, documented, Python samples: `v20` / [oanda/v20-python-samples](https://github.com/oanda/v20-python-samples).
- Not tested live here (401 without a token, as expected). Signup may be region-dependent; the user should check that a practice account is offered in Indonesia.
- Use practice host only (`api-fxpractice.oanda.com`). Do not place live trades from the backtest app.

**Strong complement, also free signup:** [Twelve Data](https://twelvedata.com/pricing) Basic plan — email key, **8 credits/min and 800/day**, forex included, claims **realtime forex**, `/time_series` up to **5,000** bars, official `twelvedata` Python client, trial WebSocket (8 symbols). The public `demo` key returned live EUR/USD 1h and 1min bars in our test; `XAU/USD` was rejected on the demo key (needs a real free key). Rate limits are too tight to brute-force thousands of strategy variants, but fine for loading a chosen symbol.

Keep **Dukascopy as the deep M1/tick archive** even after OANDA or Twelve Data is connected — it is still the best no-cost path to 20+ years of 1-minute and tick data.

---

## Comparison table

| Source | Realtime vs delayed | Intraday TFs | History (tested or documented) | Rate limits | Key / account? | Python | Box test 17 Nov 2026 | Legal / ToS |
|---|---|---|---|---|---|---|---|---|
| **Dukascopy Jetta + Freeserv + datafeed** | Historical (finished bars). Freeserv can return very recent bars. Not a tick stream. | Tick, M1, H1, D1 native; Freeserv also 5/10/15/30m, 4h | EURUSD H1/tick from 2003-05-04 (metadata + Freeserv H1 tested). Freeserv EURUSD M1 jumped to **2007-01-01** when asked for 2003–2006 (use Jetta for earlier official M1). XAUUSD M1/H1 from 2003-05-05 (tested). | Jetta ~8–10 req/s before throttle (503s seen after a burst). Freeserv was stable at ~1–2 req/s. | **No** | `dukascopy-python` (v1 Freeserv + v2 Jetta **both worked**), `dukascopy-downloader`, older `duka`, `tickterial` | **PASS**: H1/M1 EURUSD + XAUUSD, ticks (4,484 ticks in 1 hour via the library; 9,686 ticks decoded from a `.bi5` file) | Personal **non-commercial** use. Terms also ban redistribution, “construct a database”, and scrapers. Official CSV export exists. Use slowly, cache locally, personal use only. |
| **biquote.io** | **Near realtime** (age 1–9s). SignalR `/hubs/tick` | 1m, 5m, 15m, 30m, 1h, 4h, 1d | Short. Tested: 1h **169 bars (~11d)**, 1m **301 bars (~5h)**, 1d **500 bars from 2025-04-11**. `from=2020` and `from=2024` returned **0 bars**. | 15,000 REST/min/IP; WS unmetered | **No** | `requests`, official `biquote` / SignalR | **PASS** live EURUSD & XAUUSD ticks + OHLC | [Terms](https://biquote.io/terms.html) (30 Sep 2026): free personal **and commercial** in-app use; do not resell the raw feed. Data is **indicative MT5 broker** prices, not exchange-licensed. No SLA. |
| **yfinance / Yahoo** | Appeared nearly live (1m last bar 05:01 UTC when clock was ~05:02). Can be cached/delayed. | 1m, 2m, 5m, 15m, 30m, 60m/1h, 1d | EURUSD 1m ~**7d** (8,843 bars); 5m/15m **60d**; 1h **730d** (17,183 bars from 2024-01-31); daily from **2003-12-01**. `XAUUSD=X` **404 / empty**. `GC=F` works. | Unofficial; IP blocks possible | **No** | `yfinance` 1.8.0 | **PASS** majors; **FAIL** XAUUSD spot | Yahoo ToS restrict automated collection. Unofficial. Personal use only; can break or ban IPs. Do not redistribute. |
| **TrueFX webrates** | Live CSV snapshot (timestamps matched “now”) | Tick snapshot only (no OHLC API) | Historical downloads exist behind **login**; page did not list 2024–2025 files in the public HTML. Live feed: 9 majors, **no XAUUSD**. | Unknown; poll gently | Live: **no**. History: free login | raw `requests` | **PASS** live CSV; history page is membership-walled | Copyrighted. Historical ToS historically “internal use only”. |
| **HistData.com** | Offline files only (not realtime) | Tick + M1 (resample the rest) | Multi-year, many pairs + gold. Timestamps are **EST year-round (UTC−5), no DST**. Volume always 0. | Manual / slow automated download | **No** (form download). `histdata-fetcher` scrapes the public form. | `histdata-fetcher` | Site **up**. No full-year zip was downloaded (avoid hammering). | “Free… no warranty”. No licence text. Personal/educational use is the usual reading; do not redistribute. |
| **FXCM tickdata.fxcorporate.com** | Historical only | Tick (2019–2023 docs) | Docs: weekly `.csv.gz` 2019–2023, no gold, personal use / EULA | — | No (public URL) | scripts in `fxcm/MarketData` | **FAIL** on 17 Nov 2026: even `EURUSD/2023/1.csv.gz` returned an **HTML** page, not gzip. Treat as **unavailable**. | EULA; personal use. Irrelevant until the CDN works again. |
| **AlgoSpecial preview API** | Not a live feed (catalog lag) | M1–MN1 | Preview: **31-day window, 2,000 rows**, 60/min, 1,000/day, no key. Some months work (XAUUSD H1 Jun 2024; EURUSD H1 Oct 2025). **EURUSD H1 Nov 2026 = 0 rows**; Jan 2024 EURUSD H1 also 0. | 60/min, 1,000/day | **No** | `requests` | Partial. **Not current enough** to be the live or primary history source. | Free for research/backtests; don’t resell the feed. Full history is paid. |
| **Frankfurter / ECB** | Daily, lagged (latest was **2026-11-16** on the morning of the 17th) | Daily mid only | Long daily EUR crosses. Not tradable OHLC. | Generous | **No** | `requests` | **PASS** | Official public statistical rates. Fine legally; wrong data type for a trading backtest. |
| **Binance PAXGUSDT** | **True realtime** (crypto) | 1m–1d klines + WS | From **2020-08-28** (tested). 1,000 bars/request, paginate with `startTime`. | Standard Binance public limits | **No** | `requests` / python-binance | **PASS** 1h and 1m | Exchange public API. Tokenized gold, **not XAUUSD spot**. |
| **Kraken XAUTUSD / PAXGUSD** | **True realtime** | 1m–1d; **~720 bars max** per call | Daily `since=0` returned 432 days from **2025-09-12**. Shallow vs Binance. | Public API limits | **No** | `requests` / kraken-cli | **PASS** (last 1h close 3860.2 vs biquote XAUUSD ~3874 — close but not identical) | Exchange public API. Same proxy caveat. |
| **OANDA v20 Practice** | **True realtime stream** + authoritative candles | S5–M1–H1–D etc. | “Complete” per docs; 5,000 candles/request | Documented API limits | **Free practice account + token** | official `v20` | **401** without token (expected) | Official broker API. Best legal “realtime + history” option if signup works. Practice host only. |
| **Twelve Data Basic** | Claims **realtime forex** on the free plan | 1min–1month | Up to 5,000 bars/request; 20+ years advertised on paid marketing pages | **8 credits/min, 800/day**; WS “8 trial” | **Free email key** | official `twelvedata` | `apikey=demo`: **PASS** EUR/USD 1h + 1min; **401** on XAU/USD; no key → 401 | Official API ToS. Fine for personal use within the plan. |
| **Alpha Vantage** | Daily FX (demo last refresh **2026-11-16**). Intraday is **premium** | FX_DAILY free; FX_INTRADAY paid | Compact daily ~100 rows on demo | Free: **25 req/day** | Free key | `requests` / `alpha_vantage` | Demo **FX_DAILY PASS**; **FX_INTRADAY** told us to get a real key / is premium | Official. Too tight and no free FX intraday — skip. |
| **Finnhub** | — | Forex candles exist on paid plans | — | 60/min on equity free tier | Free key does **not** unlock forex OHLC | `finnhub-python` | **401** no key / demo token | Forex is a paid product. Skip unless paying. |
| **Polygon / Massive** | Free tier is not a usable FX stream | — | Forex is plan-dependent / typically paid | Free stocks often 5/min EOD | Key required | official clients | **401** “API Key was not provided” | Official. Do not assume a useful free FX tier. |
| **Tiingo FX** | Plan-dependent | Intraday on paid FX | — | Token plans e.g. 50/hour class | Token required (FX often an add-on) | `tiingo` | **403** “Please supply a token” | Official. Not usable without a key/plan. |
| **Financial Modeling Prep** | Free is mostly EOD | Limited | — | Free ~250/day | Key required | official | **401** invalid key | Official. Not useful as a free intraday FX feed. |
| **iTick** | Marketing: free personal forex K-lines | K-lines + ticks | Plan-dependent | Free about **5 REST/min** | Free token | REST | **401** missing API key | Personal-use free plan. Too slow as a primary backtest loader. |
| **cTrader Open API** | Realtime via official Open API | Trendbars (history 5 req/s) | Broker-dependent | 5 historical req/s | Free cTID + broker demo | `ctrader_open_api` (protobuf) | Not tested (account required) | Official. Heavier setup than OANDA. Valid backup. |
| **MetaTrader 5 Python** | Broker-live + tester history | Whatever the broker stores | Broker-dependent, often excellent M1 | Local terminal | Free demo broker + **Windows** terminal | official `MetaTrader5` | **Cannot import on this Linux box**. Windows-only. | Official, under the broker’s agreement. Best if the user already runs MT5 on Windows. |
| **TradingView unofficial** (`tvDatafeed`, scrapers) | Looks realtime, unofficial | Many | Often long, but unofficial | Ban risk | TV login often required | unofficial libs | **Not tested on purpose** | **ToS risk.** TradingView forbids scraping / automated collection; accounts can be banned. Do not use in this app. |
| **open.er-api.com / jsDelivr currency-api** | Daily mid, not a market feed | Daily | Daily only | Public CDN | No | `requests` | PASS as FX *rates*, not OHLC | Fine legally; wrong product for Vela. |
| **exchangerate.host** | — | — | — | — | Now appears productized | — | **FAIL**: URL returned a Swagger/portal HTML page, not rates | Don’t use. |

---

## What actually worked from the box (selected raw results)

All times below are converted to **Asia/Jakarta (UTC+7)** when they came in as UTC.

### Dukascopy — primary archive (PASS)

- `dukascopy-python` **v1 and v2** both returned 96 EURUSD H1 BID candles for 10–16 Nov 2026 (open 1.12219 → close 1.10246).
- v1 ticks for 16 Nov 2026 12:00–13:00 UTC: **4,484** bid/ask ticks.
- Classic `datafeed.dukascopy.com` `.bi5` tick file for 15 Oct 2026 12:00 UTC (month in the URL is **zero-based**): **9,686** ticks, bid 1.12071–1.12139, decoded with the standard `lzma` + 20-byte `>IIIff` record (EURUSD scale 100,000).
- Hourly/minute `.bi5` candle URLs also returned 200 (EURUSD H1 and M1; XAUUSD H1). Candle `.bi5` layout is OCLH, not OHLC — prefer Jetta or Freeserv, which already decode for you.
- Jetta requires `Origin: https://www.dukascopy.com`. Without it, CloudFront returns **403**. Instrument codes are `EUR-USD`, `XAU-USD`, not `EURUSD`.
- Freeserv `chart/json3`: EURUSD H1 on 5 May 2003 (1.12161…); XAUUSD H1 on 5 May 2003 (**$340.345**); XAUUSD M1 on 5 May 2003. EURUSD M1 requests for 2005–2006 were rewritten to **1 Jan 2007** — treat early EURUSD M1 as a Jetta job.
- After a burst of datafeed requests the CDN returned **503**. Back off (≤ ~5–8 req/s) and cache.

Libraries: [`dukascopy-python`](https://pypi.org/project/dukascopy-python/) (tested 6.0.0), [`dukascopy-downloader`](https://pypi.org/project/dukascopy-downloader/) (Parquet, rate-limited), older [`giuse88/duka`](https://github.com/giuse88/duka), [`tickterial`](https://pypi.org/project/tickterial/).

### biquote — primary live feed (PASS, short history)

- `GET /api/EURUSD` mid **1.10055**, age **9s**, source “MetaTrader 5 (Broker 1)”.
- `GET /api/XAUUSD` mid **3874.243**, age **1s**.
- 1h EURUSD: 169 bars from 6 Nov 2026 05:00 UTC; 1m: 301 bars (~5 hours). Daily: 500 bars from 11 Apr 2025.
- Docs: [biquote.io/docs](https://biquote.io/docs/). WS: `https://biquote.io/hubs/tick`.

### yfinance (PASS majors, FAIL spot gold)

- EURUSD=X 1h `period=730d`: 17,183 rows, 31 Jan 2024 → 17 Nov 2026 05:00 UTC (12:00 WIB).
- EURUSD=X 1m `period=7d`: 8,843 rows.
- EURUSD=X 1d max: 5,958 rows from **1 Dec 2003**.
- XAUUSD=X / XAU=X / GOLD=X: **empty, Yahoo 404**.
- GC=F 1h `730d`: 13,652 rows from 27 Jun 2024; 1d max from 30 Aug 2000; last daily close **3853** (16 Nov 2026) vs spot ~3874 — futures, not spot.

### Others

- Twelve Data `apikey=demo`: EUR/USD 1h and 1min **ok** (latest 1min bar 17 Nov 2026 16:04 — Twelve Data timestamps look exchange-local, not UTC; confirm TZ when wiring). XAU/USD demo **401**.
- Alpha Vantage demo FX_DAILY: compact daily, last refresh 16 Nov 2026. FX_INTRADAY demo refused.
- Binance PAXGUSDT: listing 28 Aug 2020; recent 1h ~3865.
- Kraken XAUTUSD last 1h close **3860.2** at 17 Nov 2026 05:00 UTC.
- Frankfurter latest: EURUSD **1.1014** on 16 Nov 2026 (previous ECB day).
- TrueFX live CSV: EUR/USD bid/ask 1.10057 / 1.10063 at the same moment.
- FXCM public tick CDN, Finnhub, Polygon, Tiingo, FMP, iTick, OANDA: no usable anonymous data.
- MetaTrader5 Python package: not importable on Linux.

---

## Suggested architecture for the Streamlit app

1. **Backtest request** (user picks symbol, TF, dates, unseen %): load (or incrementally update a local Parquet cache) from **Dukascopy**. Resample M1 → 5m/15m/1h/4h as needed so walk-forward always uses the same tick/M1 basis.
2. **Unseen split** is a date cut on that cached series — it does not depend on the vendor.
3. **Live chart**: subscribe to **biquote** SignalR (or poll `/api/{symbol}` every few seconds). Overlay the current unfinished bar on the Dukascopy history. Warn that live quotes are a different broker than the Dukascopy archive (spread/price mismatch is normal).
4. After the user creates an **OANDA practice** token, optionally switch both live prices and recent candles to OANDA so history and live quotes share one book. Still use Dukascopy for pre-2009 or tick-level work.
5. Never scrape TradingView. Never hit Dukascopy or HistData in a tight loop from the UI; download once, cache, “update” on demand.
6. Gold: default **XAUUSD on Dukascopy/biquote**. Offer `GC=F`, `PAXGUSDT`, `XAUTUSD` only as clearly labelled proxies.

---

## Legal short notes (personal, non-commercial app)

- **Dukascopy**: licensed for personal non-commercial use; no redistribution; anti-scraper wording. Cache for this one app, modest request rate, no public data dump.
- **biquote**: written permission for in-app use, including commercial; do not wrap it as a public data API. Indicative only.
- **Yahoo / yfinance**: unofficial, ToS-hostile to automation. Fallback only.
- **OANDA / Twelve Data / cTrader / MT5**: use their official ToS and stay on demo/practice if the user is only backtesting.
- **TradingView unofficial libs**: do not use.
- All of these are **indicative** OTC/broker series, not a consolidated exchange tape. Fees and slippage in the backtester still matter.

---

## Files

- `research_fx_sources.py` — tested helpers + `python research_fx_sources.py` smoke test.
- Do not store API keys in this repo. If OANDA or Twelve Data is added later, use environment variables (`OANDA_API_TOKEN`, `TWELVEDATA_API_KEY`).
