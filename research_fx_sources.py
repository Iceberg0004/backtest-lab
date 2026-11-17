#!/usr/bin/env python3
"""Tested helpers for FREE, legal-enough forex/gold OHLC used by the personal Vela app.

This module was written after live requests from the research box on
2026-11-17 (Asia/Jakarta). It does not require API keys.

Working zero-signup sources implemented here
--------------------------------------------
- dukascopy_jetta / dukascopy_freeserv : long FX + XAUUSD history (best archive)
- biquote                             : near-realtime FX + XAUUSD, short history
- yfinance                            : easy majors (not XAUUSD spot); unofficial
- binance_paxg / kraken_xaut          : tokenized-gold proxy, not XAUUSD spot
- frankfurter                         : official daily mid rates, not tradable OHLC

Key-based sources (OANDA, Twelve Data, etc.) are documented in DATA_SOURCES.md
and are not called here.

Usage::

    from research_fx_sources import fetch_ohlc, fetch_tick
    df = fetch_ohlc("dukascopy", "EURUSD", "1h", start="2026-11-10", end="2026-11-16")
    tick = fetch_tick("biquote", "XAUUSD")

Run a live smoke test::

    python research_fx_sources.py
"""

from __future__ import annotations

import json
import random
import string
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Iterable
from urllib.parse import urlencode, urlunsplit

import requests

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
)

# jetta.dukascopy.com is fronted by CloudFront and returns 403 unless the
# request looks like it came from dukascopy.com.
DUKASCOPY_HEADERS = {
    "User-Agent": UA,
    "Origin": "https://www.dukascopy.com",
    "Referer": "https://www.dukascopy.com/",
    "Accept": "application/json,text/javascript,*/*;q=0.8",
}

FREESERV_HEADERS = {
    "User-Agent": UA,
    "Referer": "https://freeserv.dukascopy.com/2.0/?path=chart/index&instrument=EUR/USD&period=60",
}

# Official jetta history starts (ms), fetched 2026-11-17 from /v1/instruments/{code}.
JETTA_CODES = {
    "EURUSD": "EUR-USD",
    "GBPUSD": "GBP-USD",
    "USDJPY": "USD-JPY",
    "AUDUSD": "AUD-USD",
    "USDCAD": "USD-CAD",
    "USDCHF": "USD-CHF",
    "NZDUSD": "NZD-USD",
    "EURGBP": "EUR-GBP",
    "EURJPY": "EUR-JPY",
    "GBPJPY": "GBP-JPY",
    "XAUUSD": "XAU-USD",
    "XAGUSD": "XAG-USD",
}

YFINANCE_TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "USDCHF": "USDCHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    # Yahoo has no working XAUUSD=X (tested 2026-11-17, 404). GC=F is COMEX futures.
    "GC=F": "GC=F",
    "XAUUSD_FUT": "GC=F",
}

# Yahoo interval limits observed 2026-11-17 (EURUSD=X):
#   1m  ~7 days; 5m/15m ~60 days; 1h ~730 days (period=730d);
#   1d  from 2003-12-01.
YFINANCE_INTERVALS = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "1d": "1d",
    "1w": "1wk",
}

BIQUOTE_INTERVALS = {"1m", "5m", "15m", "30m", "1h", "4h", "1d"}

FREESERV_INTERVALS = {
    "1m": "1MIN",
    "5m": "5MIN",
    "10m": "10MIN",
    "15m": "15MIN",
    "30m": "30MIN",
    "1h": "1HOUR",
    "4h": "4HOUR",
    "1d": "1DAY",
}


def _session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})
    return sess


def _norm_symbol(symbol: str) -> str:
    s = symbol.upper().replace("/", "").replace("-", "").replace("_", "").replace("=", "")
    if s.endswith("X") and len(s) == 7:  # EURUSDX from EURUSD=X
        s = s[:-1]
    aliases = {"XAUUSDX": "XAUUSD", "GC": "GC=F", "GCF": "GC=F", "GOLD": "XAUUSD"}
    return aliases.get(s, s)


def _parse_dt(value: str | datetime | None, default: datetime | None = None) -> datetime | None:
    if value is None:
        return default
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text.replace("Z", ""), fmt.replace("Z", ""))
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _to_frame(rows: list[dict[str, Any]]):
    if pd is None:
        return rows
    if not rows:
        df = pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        return df
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df


def _cumsum(values: Iterable[float]) -> list[float]:
    out = []
    total = 0.0
    for v in values:
        total += v
        out.append(total)
    return out


# ---------------------------------------------------------------------------
# Dukascopy Jetta (no key). Needs Origin: https://www.dukascopy.com
# ---------------------------------------------------------------------------
def fetch_dukascopy_jetta(
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    side: str = "BID",
    session: requests.Session | None = None,
):
    """Fetch finished Dukascopy candles via jetta.dukascopy.com.

    interval: 1m, 1h, 1d. Jetta serves native minute / hour / day files only;
    other timeframes should be resampled by the caller.

    History (official /v1/instruments metadata, 2026-11-17):
      EURUSD tick/M1/H1 from 2003-05-04, daily from 1973-03-01
      XAUUSD tick/M1/H1 from 2003-05-05, daily from 1999-06-03

    The current unfinished month (H1) / year (D1) may 400 until the period
    closes; use freeserv or biquote for the most recent bars.
    """
    sym = _norm_symbol(symbol)
    code = JETTA_CODES.get(sym)
    if not code:
        raise ValueError(f"Unsupported jetta symbol {symbol!r}. Known: {sorted(JETTA_CODES)}")
    start_dt = _parse_dt(start, datetime.now(timezone.utc) - timedelta(days=7))
    end_dt = _parse_dt(end, datetime.now(timezone.utc))
    assert start_dt and end_dt
    side = side.upper()
    if side not in {"BID", "ASK"}:
        raise ValueError("side must be BID or ASK")

    sess = session or _session()
    sess.headers.update(DUKASCOPY_HEADERS)

    rows: list[dict[str, Any]] = []
    cursor = start_dt
    while cursor <= end_dt:
        if interval == "1m":
            path = f"/v1/candles/minute/{code}/{side}/{cursor.year}/{cursor.month:02d}/{cursor.day:02d}"
            nxt = cursor + timedelta(days=1)
        elif interval == "1h":
            path = f"/v1/candles/hour/{code}/{side}/{cursor.year}/{cursor.month:02d}"
            if cursor.month == 12:
                nxt = cursor.replace(year=cursor.year + 1, month=1, day=1)
            else:
                nxt = cursor.replace(month=cursor.month + 1, day=1)
        elif interval == "1d":
            path = f"/v1/candles/day/{code}/{side}/{cursor.year}"
            nxt = cursor.replace(year=cursor.year + 1, month=1, day=1)
        else:
            raise ValueError("dukascopy_jetta interval must be 1m, 1h or 1d")

        url = urlunsplit(("https", "jetta.dukascopy.com", path, "", ""))
        resp = sess.get(url, timeout=30)
        if resp.status_code == 400 and "too late" in resp.text.lower():
            break
        if resp.status_code == 400 and "Unknown instrument" in resp.text:
            raise RuntimeError(f"jetta unknown instrument {code}: {resp.text}")
        if resp.status_code != 200:
            # unfinished / missing period
            cursor = nxt
            continue
        rows.extend(_decode_jetta_candles(resp.json()))
        cursor = nxt
        time.sleep(0.12)

    rows = [r for r in rows if start_dt <= r["time"] < end_dt + timedelta(days=1)]
    return _to_frame(rows)


def _decode_jetta_candles(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Decode Dukascopy jetta's delta-encoded candle payload."""
    ts0 = int(payload["timestamp"])
    shift = int(payload["shift"])
    mult = float(payload["multiplier"])
    times = _cumsum(payload.get("times") or [])
    # Price deltas: first absolute value + cumulative delta * multiplier.
    def series(first: float, deltas: list[int] | None) -> list[float]:
        base = round(first / mult)
        if not deltas:
            return [first]
        acc = []
        running = base
        for d in deltas:
            running += int(d)
            acc.append(running * mult)
        return acc

    opens = series(float(payload["open"]), payload.get("opens"))
    highs = series(float(payload["high"]), payload.get("highs"))
    lows = series(float(payload["low"]), payload.get("lows"))
    closes = series(float(payload["close"]), payload.get("closes"))
    volumes = payload.get("volumes") or [0] * len(opens)
    out = []
    n = min(len(opens), len(highs), len(lows), len(closes), len(times) or len(opens))
    for i in range(n):
        t_ms = ts0 + int(times[i] if times else i) * shift
        out.append(
            {
                "time": datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc),
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(volumes[i]) if i < len(volumes) else 0.0,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Dukascopy freeserv chart/json3 widget API (no key, good for recent bars)
# ---------------------------------------------------------------------------
def fetch_dukascopy_freeserv(
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    side: str = "BID",
    limit: int = 5000,
    session: requests.Session | None = None,
):
    """Fetch candles from freeserv.dukascopy.com chart/json3.

    Max limit is 30_000 per request (library default). For long ranges, page
    by bumping `start`. When `start` is before the first available bar,
    freeserv silently begins at the first bar it has.

    Observed from the box (2026-11-17):
      EURUSD H1 from 2003-05-05 works.
      EURUSD M1 requested in 2003-2006 jumped to 2007-01-01 (use jetta for
      earlier official M1, which metadata dates to 2003-05-04).
      XAUUSD H1 and M1 from 2003-05-05 work.
    """
    if interval not in FREESERV_INTERVALS:
        raise ValueError(f"unsupported freeserv interval {interval}. Use {sorted(FREESERV_INTERVALS)}")
    if limit < 1 or limit > 30000:
        raise ValueError("limit must be 1..30000")
    sym = _norm_symbol(symbol)
    if sym.endswith("USD") and len(sym) == 6:
        instrument = f"{sym[:3]}/{sym[3:]}"
    elif sym == "XAUUSD":
        instrument = "XAU/USD"
    elif "/" in symbol:
        instrument = symbol.upper()
    else:
        instrument = f"{sym[:3]}/{sym[3:]}" if len(sym) == 6 else symbol

    start_dt = _parse_dt(start, datetime.now(timezone.utc) - timedelta(days=10))
    end_dt = _parse_dt(end, datetime.now(timezone.utc))
    assert start_dt and end_dt
    offer = {"BID": "B", "ASK": "A"}.get(side.upper())
    if not offer:
        raise ValueError("side must be BID or ASK")

    sess = session or _session()
    rows: list[dict[str, Any]] = []
    cursor = start_dt
    # Page forward until we pass end or get a short batch.
    while cursor < end_dt and len(rows) < 200_000:
        jsonp = "_callbacks____" + "".join(random.choices(string.ascii_letters, k=9))
        params = {
            "path": "chart/json3",
            "splits": "true",
            "stocks": "true",
            "time_direction": "N",
            "jsonp": jsonp,
            "last_update": str(int(cursor.timestamp() * 1000)),
            "offer_side": offer,
            "instrument": instrument,
            "interval": FREESERV_INTERVALS[interval],
            "limit": str(limit),
        }
        resp = sess.get(
            "https://freeserv.dukascopy.com/2.0/index.php",
            params=params,
            headers=FREESERV_HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.text.removeprefix(f"{jsonp}(").removesuffix(");")
        payload = json.loads(text)
        if not payload:
            break
        batch = []
        for item in payload:
            if not item or len(item) < 5 or any(x is None for x in item[:5]):
                continue   # freeserv sends null rows for some instruments/minutes (e.g. US30)
            ts = datetime.fromtimestamp(item[0] / 1000, tz=timezone.utc)
            if ts > end_dt:
                continue
            batch.append(
                {
                    "time": ts,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]) if len(item) > 5 else 0.0,
                }
            )
        if not batch:
            break
        rows.extend(batch)
        last_t = batch[-1]["time"]
        if last_t <= cursor:
            break
        cursor = last_t + timedelta(milliseconds=1)
        if len(payload) < limit:
            break
        time.sleep(0.15)
    return _to_frame(rows)


# ---------------------------------------------------------------------------
# biquote (no key): live ticks + short OHLC history
# ---------------------------------------------------------------------------
def fetch_biquote(
    symbol: str,
    interval: str = "1h",
    limit: int = 1000,
    session: requests.Session | None = None,
):
    """OHLC from biquote.io. No key. Up to 1000 bars, but stored history is short.

    Observed 2026-11-17 from the box (even with limit=1000):
      EURUSD 1h: 169 bars (~11 days)
      EURUSD 1m: 301 bars (~5 hours)
      EURUSD 1d: 500 bars (from 2025-04-11)
      from/to for 2020 and 2024 returned 0 bars
    Source field: "MetaTrader 5 (Broker 1)". Indicative broker quotes.
    """
    if interval not in BIQUOTE_INTERVALS:
        raise ValueError(f"biquote interval must be one of {sorted(BIQUOTE_INTERVALS)}")
    sym = _norm_symbol(symbol)
    sess = session or _session()
    resp = sess.get(
        f"https://biquote.io/api/{sym}/ohlc",
        params={"interval": interval, "limit": int(limit)},
        timeout=30,
    )
    resp.raise_for_status()
    bars = resp.json().get("bars") or []
    rows = []
    for b in bars:
        if b.get("isOpen"):
            # keep, but callers doing backtests should drop the in-progress bar
            pass
        rows.append(
            {
                "time": _parse_dt(b["openTime"]),
                "open": float(b["open"]),
                "high": float(b["high"]),
                "low": float(b["low"]),
                "close": float(b["close"]),
                "volume": float(b.get("tickVolume") or b.get("volume") or 0),
                "is_open": bool(b.get("isOpen")),
            }
        )
    return _to_frame(rows)


def fetch_tick(source: str, symbol: str, session: requests.Session | None = None) -> dict[str, Any]:
    """Latest quote. source: biquote | truefx (majors only)."""
    source = source.lower()
    sym = _norm_symbol(symbol)
    sess = session or _session()
    if source == "biquote":
        resp = sess.get(f"https://biquote.io/api/{sym}", timeout=20)
        resp.raise_for_status()
        return resp.json()
    if source == "truefx":
        # Public CSV snapshot, no login. Majors only; no XAUUSD in the feed.
        resp = sess.get("https://webrates.truefx.com/rates/connect.html?f=csv", timeout=20)
        resp.raise_for_status()
        pair = f"{sym[:3]}/{sym[3:]}" if len(sym) == 6 else symbol
        for line in resp.text.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 9:
                continue
            if parts[0].replace("/", "") == sym or parts[0] == pair:
                # Format: EUR/USD,ts,bid1,bid2,ask1,ask2,low,high,open
                bid = float(parts[2] + parts[3])
                ask = float(parts[4] + parts[5])
                ts_ms = int(parts[1])
                return {
                    "symbol": parts[0],
                    "timestamp": datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(),
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2,
                    "low": float(parts[6]),
                    "high": float(parts[7]),
                    "open": float(parts[8]),
                    "source": "truefx",
                }
        raise KeyError(f"TrueFX snapshot has no {sym}")
    raise ValueError("source must be biquote or truefx")


# ---------------------------------------------------------------------------
# yfinance (no key, unofficial Yahoo). Not for XAUUSD spot.
# ---------------------------------------------------------------------------
def fetch_yfinance(
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    period: str | None = None,
):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("pip install yfinance") from exc
    sym = _norm_symbol(symbol)
    ticker = YFINANCE_TICKERS.get(sym)
    if not ticker:
        if sym == "XAUUSD":
            raise ValueError(
                "Yahoo has no working XAUUSD=X ticker (404 on 2026-11-17). "
                "Use Dukascopy or biquote for spot gold, or GC=F for COMEX futures."
            )
        ticker = f"{sym}=X"
    yf_interval = YFINANCE_INTERVALS.get(interval)
    if not yf_interval:
        raise ValueError(f"unsupported yfinance interval {interval}")
    kwargs: dict[str, Any] = {
        "interval": yf_interval,
        "progress": False,
        "auto_adjust": False,
        "threads": False,
    }
    if period:
        kwargs["period"] = period
    else:
        start_dt = _parse_dt(start, datetime.now(timezone.utc) - timedelta(days=30))
        end_dt = _parse_dt(end, datetime.now(timezone.utc) + timedelta(days=1))
        kwargs["start"] = start_dt.strftime("%Y-%m-%d")
        kwargs["end"] = end_dt.strftime("%Y-%m-%d")
    df = yf.download(ticker, **kwargs)
    if df is None or df.empty:
        return _to_frame([])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() if isinstance(c, tuple) else str(c).lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    df = df.reset_index()
    time_col = "datetime" if "datetime" in df.columns else df.columns[0]
    rows = []
    for rec in df.to_dict("records"):
        rows.append(
            {
                "time": rec[time_col] if getattr(rec[time_col], "tzinfo", None) else pd.Timestamp(rec[time_col], tz="UTC"),
                "open": float(rec.get("open", float("nan"))),
                "high": float(rec.get("high", float("nan"))),
                "low": float(rec.get("low", float("nan"))),
                "close": float(rec.get("close", float("nan"))),
                "volume": float(rec.get("volume") or 0),
            }
        )
    return _to_frame(rows)


# ---------------------------------------------------------------------------
# Tokenized gold proxies (crypto venues). Not the same as XAUUSD spot.
# ---------------------------------------------------------------------------
def fetch_binance_paxg(
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    limit: int = 1000,
    session: requests.Session | None = None,
):
    """PAXGUSDT klines. History from 2020-08-28, true 24/7 crypto hours (includes weekends)."""
    mapping = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h", "4h": "4h", "1d": "1d"}
    if interval not in mapping:
        raise ValueError(f"unsupported binance interval {interval}")
    sess = session or _session()
    params: dict[str, Any] = {"symbol": "PAXGUSDT", "interval": mapping[interval], "limit": min(int(limit), 1000)}
    if start:
        params["startTime"] = int(_parse_dt(start).timestamp() * 1000)  # type: ignore[union-attr]
    if end:
        params["endTime"] = int(_parse_dt(end).timestamp() * 1000)  # type: ignore[union-attr]
    resp = sess.get("https://api.binance.com/api/v3/klines", params=params, timeout=20)
    resp.raise_for_status()
    rows = []
    for k in resp.json():
        rows.append(
            {
                "time": datetime.fromtimestamp(k[0] / 1000, tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            }
        )
    return _to_frame(rows)


def fetch_kraken_xaut(
    interval: str = "1h",
    session: requests.Session | None = None,
):
    """XAUTUSD OHLC. Kraken returns at most ~720 bars per call (no deep pagination)."""
    mapping = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
    if interval not in mapping:
        raise ValueError(f"unsupported kraken interval {interval}")
    sess = session or _session()
    resp = sess.get(
        "https://api.kraken.com/0/public/OHLC",
        params={"pair": "XAUTUSD", "interval": mapping[interval]},
        timeout=20,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("error"):
        raise RuntimeError(body["error"])
    result = body["result"]
    key = next(k for k in result if k != "last")
    rows = []
    for k in result[key]:
        rows.append(
            {
                "time": datetime.fromtimestamp(k[0], tz=timezone.utc),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[6]),
            }
        )
    return _to_frame(rows)


# ---------------------------------------------------------------------------
# Official daily FX reference (ECB via Frankfurter). Not for intraday backtests.
# ---------------------------------------------------------------------------
def fetch_frankfurter_daily(
    base: str = "EUR",
    quote: str = "USD",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    session: requests.Session | None = None,
):
    start_dt = _parse_dt(start, datetime(2020, 1, 1, tzinfo=timezone.utc))
    end_dt = _parse_dt(end, datetime.now(timezone.utc))
    assert start_dt and end_dt
    sess = session or _session()
    url = f"https://api.frankfurter.app/{start_dt.date()}..{end_dt.date()}"
    resp = sess.get(url, params={"from": base.upper(), "to": quote.upper()}, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = []
    for day, rates in sorted((payload.get("rates") or {}).items()):
        px = float(rates[quote.upper()])
        rows.append(
            {
                "time": datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc),
                "open": px,
                "high": px,
                "low": px,
                "close": px,
                "volume": 0.0,
            }
        )
    return _to_frame(rows)


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------
def fetch_ohlc(
    source: str,
    symbol: str,
    interval: str = "1h",
    start: str | datetime | None = None,
    end: str | datetime | None = None,
    **kwargs,
):
    """Dispatch to a tested source. source is one of:

    dukascopy (alias: dukascopy_freeserv), dukascopy_jetta, biquote,
    yfinance, binance_paxg, kraken_xaut, frankfurter
    """
    source = source.lower().strip()
    if source in {"dukascopy", "dukascopy_freeserv", "duka", "freeserv"}:
        return fetch_dukascopy_freeserv(symbol, interval, start, end, **kwargs)
    if source in {"dukascopy_jetta", "jetta"}:
        return fetch_dukascopy_jetta(symbol, interval, start, end, **kwargs)
    if source == "biquote":
        return fetch_biquote(symbol, interval, **kwargs)
    if source in {"yfinance", "yahoo"}:
        return fetch_yfinance(symbol, interval, start, end, **kwargs)
    if source in {"binance", "binance_paxg", "paxg"}:
        return fetch_binance_paxg(interval, start, end, **kwargs)
    if source in {"kraken", "kraken_xaut", "xaut"}:
        return fetch_kraken_xaut(interval, **kwargs)
    if source in {"frankfurter", "ecb"}:
        base, quote = "EUR", "USD"
        sym = _norm_symbol(symbol)
        if len(sym) == 6:
            base, quote = sym[:3], sym[3:]
        return fetch_frankfurter_daily(base, quote, start, end, **kwargs)
    raise ValueError(f"unknown source {source!r}")


def smoke_test() -> list[dict[str, Any]]:
    """Small live tests. Be polite: few requests, short windows."""
    results = []

    def run(name, fn):
        t0 = time.perf_counter()
        try:
            out = fn()
            ms = int((time.perf_counter() - t0) * 1000)
            n = len(out)
            first = last = None
            if n and pd is not None:
                first = str(out["time"].iloc[0])
                last = str(out["time"].iloc[-1])
                close = float(out["close"].iloc[-1])
            elif n and isinstance(out, list):
                first, last = str(out[0]["time"]), str(out[-1]["time"])
                close = out[-1]["close"]
            else:
                close = None
            row = {"name": name, "ok": n > 0, "n": n, "latency_ms": ms, "first": first, "last": last, "last_close": close}
        except Exception as exc:  # noqa: BLE001 - report every source
            ms = int((time.perf_counter() - t0) * 1000)
            row = {"name": name, "ok": False, "n": 0, "latency_ms": ms, "error": repr(exc)}
        results.append(row)
        status = "OK" if row.get("ok") else "FAIL"
        print(f"[{status}] {name}: { {k: row[k] for k in row if k != 'name'} }")
        time.sleep(0.25)
        return row

    start = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run("dukascopy_freeserv EURUSD 1h", lambda: fetch_dukascopy_freeserv("EURUSD", "1h", start, end, limit=120))
    run("dukascopy_freeserv EURUSD 1m", lambda: fetch_dukascopy_freeserv("EURUSD", "1m", start, end, limit=200))
    run("dukascopy_jetta EURUSD 1h last month-ish", lambda: fetch_dukascopy_jetta("EURUSD", "1h", "2026-10-01", "2026-10-08"))
    run("dukascopy_freeserv XAUUSD 1h", lambda: fetch_dukascopy_freeserv("XAUUSD", "1h", start, end, limit=80))
    run("biquote EURUSD 1h", lambda: fetch_biquote("EURUSD", "1h", limit=50))
    run("biquote XAUUSD 1h", lambda: fetch_biquote("XAUUSD", "1h", limit=50))
    run("yfinance EURUSD 1h", lambda: fetch_yfinance("EURUSD", "1h", period="5d"))
    run("binance PAXGUSDT 1h", lambda: fetch_binance_paxg("1h", limit=10))
    run("kraken XAUTUSD 1h", lambda: fetch_kraken_xaut("1h"))
    run("frankfurter EURUSD daily", lambda: fetch_frankfurter_daily("EUR", "USD", "2026-11-01", "2026-11-16"))

    try:
        tick = fetch_tick("biquote", "EURUSD")
        results.append({"name": "biquote tick EURUSD", "ok": True, "bid": tick.get("bid"), "ask": tick.get("ask"), "age": tick.get("quoteAgeSeconds")})
        print(f"[OK] biquote tick EURUSD mid={tick.get('mid')} age_s={tick.get('quoteAgeSeconds')}")
    except Exception as exc:  # noqa: BLE001
        results.append({"name": "biquote tick EURUSD", "ok": False, "error": repr(exc)})
        print(f"[FAIL] biquote tick: {exc!r}")
    return results


if __name__ == "__main__":
    smoke_test()
