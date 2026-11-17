"""Crypto derivatives data (free, public) aligned to price bars WITHOUT lookahead.

- Taker buy volume (order-flow proxy): from Binance spot klines (field 'taker buy base volume'), full history.
- Funding rate: Binance Futures API, archive fallback (core/realism.funding_history).
- Open interest + long/short ratios: daily 'metrics' files on data.binance.vision (5-minute snapshots, from
  about Sep 2020 for BTC), topped up with the Futures API (which itself only keeps ~30 days).
- Liquidations: no free historical source works (the archive has no liquidation files, the API only streams
  live ones), so they are NOT included.

Alignment: a bar gets the last value observed strictly BEFORE the bar closes (open time + timeframe).
Taker volume belongs to the bar itself (known at its close)."""
from __future__ import annotations
import io, time, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from .data import CACHE_DIR, TF_MINUTES, BinanceSource

COLS = ["taker_buy", "fund", "oi", "lsr", "top_lsr"]
LABELS = {"taker": "Volume taker (order flow)", "fund": "Funding rate", "oi": "Open interest", "lsr": "Rasio long/short akun"}
_UA = {"User-Agent": "Mozilla/5.0 (backtest-lab personal research)"}
FAPI = "https://fapi.binance.com"
VISION = "https://data.binance.vision/data/futures/um/daily/metrics"


def _save(df, p):
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
    except OSError:
        pass


# ------------------------------------------------------------------ taker buy volume (spot klines)
def taker_buy(symbol, tf, start, progress=None):
    """Series of taker-buy base volume per bar (index = bar open time, UTC)."""
    p = CACHE_DIR / f"binance_taker_{symbol}_{tf}.parquet"
    old = pd.read_parquet(p)["tb"] if p.exists() else None
    src = BinanceSource()
    step = TF_MINUTES[tf] * 60_000
    s_ms = int(pd.Timestamp(start).timestamp() * 1000)
    parts = []

    def fetch(a, b=None):
        out, cur = [], a
        total = max(1, ((b or int(time.time() * 1000)) - a) // step)
        while True:
            prm = {"symbol": symbol, "interval": tf, "startTime": int(cur), "limit": 1000}
            if b:
                prm["endTime"] = int(b)
            rows = src._req(prm)
            if not rows:
                break
            out.extend((r[0], float(r[9])) for r in rows)
            cur = rows[-1][0] + step
            if progress:
                progress(min(1.0, len(out) / total))
            if len(rows) < 1000 or (b and cur >= b):
                break
        return pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): v for t, v in out}, dtype=float)

    if old is None or len(old) == 0:
        parts.append(fetch(s_ms))
    else:
        if s_ms < old.index[0].timestamp() * 1000 - step:
            parts.append(fetch(s_ms, int(old.index[0].timestamp() * 1000)))
        parts.append(fetch(int(old.index[-1].timestamp() * 1000)))
    s = pd.concat([x for x in [old] + parts if x is not None and len(x)]) if any(len(x) for x in parts) or old is not None else pd.Series(dtype=float)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s):
        _save(pd.DataFrame({"tb": s}), p)
    return s


# ------------------------------------------------------------------ open interest + long/short (archive + API)
def _vision_day(symbol, d, sess):
    try:
        r = sess.get(f"{VISION}/{symbol}/{symbol}-metrics-{d:%Y-%m-%d}.zip", timeout=20, headers=_UA)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        x = pd.read_csv(z.open(z.namelist()[0]), usecols=["create_time", "sum_open_interest_value", "count_long_short_ratio",
                                                         "sum_toptrader_long_short_ratio"])
        x["t"] = pd.to_datetime(x["create_time"], utc=True)
        x = x.sort_values("t")
        # keep the LAST snapshot of every hour, with its real time (used for the no-lookahead alignment)
        x = x.groupby(x["t"].dt.floor("h")).last()
        return pd.DataFrame({"oi": x["sum_open_interest_value"].to_numpy(float), "lsr": x["count_long_short_ratio"].to_numpy(float),
                             "top_lsr": x["sum_toptrader_long_short_ratio"].to_numpy(float)}, index=pd.DatetimeIndex(x["t"].to_numpy()).tz_localize("UTC") if x["t"].dt.tz is None else pd.DatetimeIndex(x["t"]))
    except Exception:
        return None


def _fapi_recent(symbol):
    """Last ~20 days at 1h from the Futures API (OI history there is limited to ~30 days)."""
    out = {}
    for path, col, key in (("/futures/data/openInterestHist", "oi", "sumOpenInterestValue"),
                           ("/futures/data/globalLongShortAccountRatio", "lsr", "longShortRatio"),
                           ("/futures/data/topLongShortPositionRatio", "top_lsr", "longShortRatio")):
        r = requests.get(FAPI + path, params={"symbol": symbol, "period": "1h", "limit": 500}, timeout=15, headers=_UA)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        rows = r.json()
        # API timestamps are the START of the 1h period -> value known at the end of it
        out[col] = pd.Series({pd.Timestamp(int(x["timestamp"]) + 3_600_000 - 1, unit="ms", tz="UTC"): float(x[key]) for x in rows})
    return pd.DataFrame(out)


def metrics(symbol, start="2020-09-01", progress=None):
    """(DataFrame oi/lsr/top_lsr indexed by observation time, note)."""
    p = CACHE_DIR / f"binance_metrics_{symbol}.parquet"
    old = pd.read_parquet(p) if p.exists() else None
    first = pd.Timestamp(start, tz="UTC").normalize()
    if old is not None and len(old):
        first = old.index[-1].normalize()
    days = pd.date_range(first, pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=1), freq="D")
    parts = []
    if len(days):
        sess = requests.Session()
        sess.mount("https://", requests.adapters.HTTPAdapter(pool_maxsize=16))
        done = [0]

        def job(d):
            r = _vision_day(symbol, d, sess)
            done[0] += 1
            return r
        with ThreadPoolExecutor(16) as ex:
            futs = [ex.submit(job, d) for d in days]
            for i, f in enumerate(futs):
                x = f.result()
                if x is not None and len(x):
                    parts.append(x)
                if progress and i % 25 == 0:
                    progress(min(1.0, (i + 1) / len(days)))
    note = "arsip data.binance.vision"
    df = pd.concat([x for x in [old] + parts if x is not None and len(x)]) if (parts or old is not None) else pd.DataFrame(columns=["oi", "lsr", "top_lsr"])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df):
        _save(df, p)
    try:
        rec = _fapi_recent(symbol)
        rec = rec[rec.index > (df.index[-1] if len(df) else rec.index[0] - pd.Timedelta(seconds=1))]
        if len(rec):
            df = pd.concat([df, rec]).sort_index()
        note += " + Binance Futures API (hari-hari terakhir)"
    except Exception as e:
        note += f" (API futures tidak bisa diakses dari server ini: {e}; 1-2 hari terakhir belum ada)"
    return df, note


# ------------------------------------------------------------------ alignment + enrich
def align(bar_t_ms, tf_ms, obs_index, values):
    """For every bar: last value observed strictly before the bar close."""
    if len(obs_index) == 0:
        return np.full(len(bar_t_ms), np.nan)
    ot = pd.DatetimeIndex(obs_index).as_unit("ms").asi8
    pos = np.searchsorted(ot, bar_t_ms + tf_ms, side="left") - 1      # obs time < close
    out = np.full(len(bar_t_ms), np.nan)
    ok = pos >= 0
    out[ok] = np.asarray(values, float)[pos[ok]]
    return out


def enrich(df, symbol, tf, progress=None):
    """Adds taker_buy / fund / oi / lsr / top_lsr columns (NaN where not available). Returns (df, info)."""
    from .realism import funding_series
    df = df.copy()
    t = np.asarray(pd.DatetimeIndex(df.index).as_unit("ms").asi8, np.int64)
    tf_ms = TF_MINUTES[tf] * 60_000
    info = {}
    try:
        tb = taker_buy(symbol, tf, df.index[0], progress=(lambda f: progress(0.3 * f, "Volume taker")) if progress else None)
        df["taker_buy"] = tb.reindex(df.index).to_numpy(float)
        info["taker"] = dict(ok=True, frm=tb.index[0] if len(tb) else None, note="Binance spot klines")
    except Exception as e:
        df["taker_buy"] = np.nan
        info["taker"] = dict(ok=False, note=f"gagal: {e}")
    try:
        if progress:
            progress(0.35, "Funding rate")
        s, note = funding_series(symbol)
        df["fund"] = align(t, tf_ms, s.index, s.to_numpy())
        info["fund"] = dict(ok=len(s) > 0, frm=s.index[0] if len(s) else None, note=note)
    except Exception as e:
        df["fund"] = np.nan
        info["fund"] = dict(ok=False, note=f"gagal: {e}")
    try:
        m, note = metrics(symbol, progress=(lambda f: progress(0.4 + 0.6 * f, "Open interest & long/short (arsip harian)")) if progress else None)
        if len(m):
            m = m.where(m > 0)                       # a few archive snapshots are 0 = missing
        for c in ("oi", "lsr", "top_lsr"):
            df[c] = align(t, tf_ms, m.index, m[c].to_numpy()) if len(m) else np.nan
        # gaps longer than 1 day in the archive -> do not carry old values forward
        if len(m):
            ot = pd.DatetimeIndex(m.index).as_unit("ms").asi8
            pos = np.searchsorted(ot, t + tf_ms, side="left") - 1
            stale = (pos < 0) | ((t + tf_ms) - ot[np.maximum(pos, 0)] > 86_400_000)
            df.loc[stale, ["oi", "lsr", "top_lsr"]] = np.nan
        info["oi"] = dict(ok=len(m) > 0, frm=m.index[0] if len(m) else None, note=note)
        info["lsr"] = info["oi"]
    except Exception as e:
        for c in ("oi", "lsr", "top_lsr"):
            df[c] = np.nan
        info["oi"] = info["lsr"] = dict(ok=False, note=f"gagal: {e}")
    for k in ("taker", "fund", "oi"):
        col = {"taker": "taker_buy"}.get(k, k)
        info[k]["coverage"] = float(np.isfinite(df[col].to_numpy(float)).mean())
    return df, info
