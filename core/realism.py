"""Realism add-ons:
- intrabar(): when stop-loss AND take-profit are both touched inside one candle, look at 1-minute
  candles of that candle to see which was hit first (instead of always assuming the stop).
- funding_carry(): Binance USDT-M perpetual funding history -> holding cost per bar (crypto).
- swap_carry(): FX/gold overnight swap approximation from user-set yearly rates.
"""
from __future__ import annotations
import copy, io, zipfile
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests
from . import engine as E
from .data import TF_MINUTES, BinanceSource, CACHE_DIR

_UA = {"User-Agent": "BacktestLab/1.0 (personal research)"}


# ------------------------------------------------------------------ intrabar SL/TP order
def _ltf_for(tf):
    m = TF_MINUTES[tf]
    if m <= 1:
        return None
    if m <= 1000:
        return "1m"
    if m <= 5000:
        return "5m"
    return None


def _fetch_ltf_binance(symbol, ts, tf, ltf):
    n = TF_MINUTES[tf] // TF_MINUTES[ltf]
    rows = BinanceSource()._req({"symbol": symbol, "interval": ltf, "startTime": int(ts.timestamp() * 1000), "limit": n})
    end_ms = int(ts.timestamp() * 1000) + TF_MINUTES[tf] * 60_000
    rows = [r for r in rows if r[0] < end_ms]
    return np.array([[float(r[2]), float(r[3])] for r in rows]) if rows else None


def _fetch_ltf_dukascopy(symbol, ts, tf, ltf):
    from .data import fx, _throttle
    _throttle()
    end = ts + pd.Timedelta(minutes=TF_MINUTES[tf])
    d = fx.fetch_dukascopy_freeserv(symbol, ltf, ts.to_pydatetime(), end.to_pydatetime(),
                                    limit=TF_MINUTES[tf] // TF_MINUTES[ltf])
    if d is None or len(d) == 0:
        return None
    d = d[(d.time >= ts) & (d.time < end)]
    return d[["high", "low"]].to_numpy(float) if len(d) else None


def _first_hit(hl, direction, sl, tp):
    """1 = target first, -1 = stop first (also when both inside the same 1m candle: conservative), 0 = unknown."""
    if hl is None:
        return 0
    for hi, lo in hl:
        if direction == 1:
            hs, ht = lo <= sl, hi >= tp
        else:
            hs, ht = hi >= sl, lo <= tp
        if hs:
            return -1
        if ht:
            return 1
    return 0


def intrabar(fam, p, cc, start, end, cfg, source, symbol, tf, max_bars=250, rounds=4, progress=None):
    """Re-runs the strategy with the true SL/TP order for ambiguous candles. Returns before/after metrics."""
    ltf = _ltf_for(tf)
    base, btr = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
    info = dict(ltf=ltf, before=base, after=base, n_amb=int(btr["amb"].sum()), n_checked=0, n_tp_first=0, n_unknown=0,
                capped=False, error=None)
    if ltf is None or source not in ("binance", "dukascopy"):
        info["error"] = "Timeframe/sumber ini tidak didukung untuk cek candle 1 menit."
        return info, btr
    fetch = _fetch_ltf_binance if source == "binance" else _fetch_ltf_dukascopy
    cc2 = copy.copy(cc)
    cc2.resolve = np.zeros(len(cc.c), np.int8)
    ltf_data = {}
    tr = btr
    for _ in range(rounds):
        amb_idx = [int(i) for i, a in zip(tr["ex"], tr["amb"]) if a]
        need = [i for i in amb_idx if i not in ltf_data]
        if len(ltf_data) + len(need) > max_bars:
            need = need[: max(0, max_bars - len(ltf_data))]
            info["capped"] = True
        if need:
            workers = 8 if source == "binance" else 1
            def job(i):
                try:
                    return i, fetch(symbol, cc.index[i], tf, ltf)
                except Exception as e:  # network trouble -> keep the conservative rule for this candle
                    info["error"] = f"{type(e).__name__}: {e}"
                    return i, None
            with ThreadPoolExecutor(max_workers=workers) as ex:
                for k, (i, hl) in enumerate(ex.map(job, need)):
                    ltf_data[i] = hl
                    if progress:
                        progress(min(1.0, len(ltf_data) / max(1, min(max_bars, len(amb_idx)))))
        new_res = np.zeros(len(cc.c), np.int8)
        n_tp = n_unk = 0
        for i, a, d, sl, tp in zip(tr["ex"], tr["amb"], tr["dir"], tr["sl"], tr["tp"]):
            if not a:
                continue
            h = _first_hit(ltf_data.get(int(i)), d, sl, tp)
            if h == 1:
                new_res[i] = 1; n_tp += 1
            elif h == 0:
                n_unk += 1
        # keep earlier decisions for candles no longer ambiguous in this path (harmless: only used if ambiguous)
        merged = np.maximum(cc2.resolve, new_res)
        changed = not np.array_equal(merged, cc2.resolve)
        cc2.resolve = merged
        m, tr = E.run(fam, p, cc2, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        info.update(after=m, n_checked=len(ltf_data), n_tp_first=n_tp, n_unknown=n_unk)
        if not changed:
            break
    info["resolve"] = cc2.resolve
    return info, tr


# ------------------------------------------------------------------ funding (crypto perpetuals)
DEFAULT_FUNDING = 0.0001  # 0.01% per 8h = Binance baseline rate


def _funding_fapi(symbol, start_ms):
    out, cur = [], start_ms
    for _ in range(60):
        r = requests.get("https://fapi.binance.com/fapi/v1/fundingRate",
                         params={"symbol": symbol, "startTime": cur, "limit": 1000}, timeout=15, headers=_UA)
        if r.status_code != 200:
            raise RuntimeError(f"fapi HTTP {r.status_code}")
        rows = r.json()
        if not rows:
            break
        out += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in rows]
        cur = rows[-1]["fundingTime"] + 1
        if len(rows) < 1000:
            break
    return out


def _funding_vision(symbol, start, end):
    """Monthly zip files on data.binance.vision (public market-data archive; works where fapi is geo-blocked)."""
    months = pd.period_range(start.tz_convert(None).to_period("M"), end.tz_convert(None).to_period("M"), freq="M")
    def get(mo):
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{symbol}-fundingRate-{mo}.zip"
        try:
            r = requests.get(url, timeout=20, headers=_UA)
            if r.status_code != 200:
                return []
            z = zipfile.ZipFile(io.BytesIO(r.content))
            d = pd.read_csv(z.open(z.namelist()[0]))
            return list(zip(d["calc_time"].astype("int64"), d["last_funding_rate"].astype(float)))
        except Exception:
            return []
    with ThreadPoolExecutor(max_workers=8) as ex:
        parts = list(ex.map(get, months))
    return [x for p in parts for x in p]


def funding_history(symbol, start, end):
    """Returns (Series of funding rates indexed by UTC time, source note)."""
    cache = CACHE_DIR / f"funding_{symbol}.parquet"
    old = None
    if cache.exists():
        try:
            old = pd.read_parquet(cache)["rate"]
        except Exception:
            old = None
    rows, note = [], ""
    s_ms = int(start.timestamp() * 1000) if old is None or len(old) == 0 else int(old.index[-1].timestamp() * 1000) + 1
    try:
        rows = _funding_fapi(symbol, s_ms)
        note = "Binance Futures API"
    except Exception as e:
        vs = start if old is None or len(old) == 0 else old.index[-1]
        rows = _funding_vision(symbol, vs, end)
        note = f"arsip data.binance.vision (API futures diblokir di server ini: {e}; bulan berjalan belum tersedia di arsip)"
    s = pd.Series({pd.Timestamp(t, unit="ms", tz="UTC"): r for t, r in rows}, dtype=float)
    if old is not None:
        s = pd.concat([old, s])
    s = s[~s.index.duplicated(keep="last")].sort_index()
    if len(s):
        try:
            pd.DataFrame({"rate": s}).to_parquet(cache)
        except OSError:
            pass
    return s, note


def funding_series(symbol):
    return funding_history(symbol, pd.Timestamp("2019-09-01", tz="UTC"), pd.Timestamp.now(tz="UTC"))


def funding_carry(cc, tf, s, note=""):
    """Per-bar funding for a LONG position (short gets the opposite). Bars before the perpetual existed,
    or not covered by the data, use the 0.01%/8h baseline. Returns (carry_long, carry_short, info)."""
    idx = cc.index
    carry = np.zeros(len(idx))
    covered_from = s.index[0] if len(s) else None
    covered_to = s.index[-1] if len(s) else None
    if len(s):
        pos = np.searchsorted(pd.DatetimeIndex(idx).as_unit("ms").asi8, pd.DatetimeIndex(s.index).as_unit("ms").asi8, side="right") - 1   # bar that contains the funding time
        ok = pos >= 0
        np.add.at(carry, pos[ok], s.to_numpy()[ok])
    # baseline where we have no history: 3 payments per day spread over bars
    per_bar = DEFAULT_FUNDING * 3 * TF_MINUTES[tf] / 1440
    miss = np.ones(len(idx), bool)
    if len(s):
        miss = (idx < covered_from) | (idx > covered_to + pd.Timedelta(hours=8))
    carry[miss] = per_bar
    info = dict(note=note, covered_from=covered_from, covered_to=covered_to, n=len(s), share_default=float(miss.mean()),
                mean_8h=float(s.mean()) if len(s) else DEFAULT_FUNDING)
    return carry, -carry, info


def swap_carry(cc, long_pct_year, short_pct_year):
    """FX/gold swap approximation: yearly cost in % of position (positive = you pay, negative = you receive),
    charged pro-rata to the real time each bar is held (so weekends are counted, like triple-swap Wednesdays)."""
    t = pd.DatetimeIndex(cc.index).as_unit("ms").asi8.astype(float) / 1e3
    dt_days = np.diff(np.r_[t, t[-1] + (t[-1] - t[-2] if len(t) > 1 else 0)]) / 86400
    cl = long_pct_year / 100 / 365 * dt_days
    cs = short_pct_year / 100 / 365 * dt_days
    return cl, cs
