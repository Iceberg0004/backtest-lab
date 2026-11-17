"""Data layer (pluggable). Every source returns a UTC-indexed DataFrame with
columns open, high, low, close, volume. Data is cached to parquet and only new
candles are fetched on refresh (incremental), so the app always works on live data."""
from __future__ import annotations
import os, time, datetime as dt
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
import requests

def _pick_cache_dir() -> Path:
    """Prefer a writable dir. Streamlit Community Cloud (and similar hosts) have
    ephemeral disks; /tmp is always writable. VELA_CACHE overrides the default."""
    candidates = []
    env = os.environ.get("VELA_CACHE")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/tmp/vela_cache"))
    candidates.append(Path(__file__).resolve().parent.parent / "data_cache")
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            test = cand / ".write_test"
            test.write_bytes(b"ok")
            test.unlink(missing_ok=True)
            return cand
        except OSError:
            continue
    cand = Path("/tmp/vela_cache")
    cand.mkdir(parents=True, exist_ok=True)
    return cand


CACHE_DIR = _pick_cache_dir()
COLS = ["open", "high", "low", "close", "volume"]

TF_MINUTES = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120,
              "4h": 240, "6h": 360, "12h": 720, "1d": 1440, "1w": 10080}


def _cache_path(source, symbol, tf):
    safe = symbol.replace("/", "_").replace("=", "_").replace("^", "_")
    return CACHE_DIR / f"{source}_{safe}_{tf}.parquet"


def _load_cache(p):
    if p.exists():
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return None


def _merge_save(p, old, new):
    df = new if old is None else pd.concat([old, new])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[COLS].astype(float)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p)
    except OSError:
        pass  # cache is optional; a fresh fetch still works without it
    return df


class DataSource:
    key = "base"
    label = "base"
    timeframes: list = []
    def symbols(self):
        return []
    def limits_note(self, tf):
        return ""
    def get(self, symbol, tf, start: pd.Timestamp | None = None, refresh=True) -> pd.DataFrame:
        raise NotImplementedError
    # crypto vs forex helpers
    def pip_size(self, symbol):
        return None


# ---------------------------------------------------------------- Binance
class BinanceSource(DataSource):
    key = "binance"
    label = "Crypto - Binance (gratis, realtime)"
    timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
    # api.binance.com is geo-blocked from many US hosts (HTTP 451/403), including
    # Streamlit Community Cloud. Prefer the public market-data host first, then
    # fall back automatically on 403/418/451 or any other failure.
    HOSTS = [
        "https://data-api.binance.vision",
        "https://api.binance.com",
        "https://api.binance.us",
        "https://api1.binance.com",
        "https://api2.binance.com",
        "https://api3.binance.com",
    ]
    _BLOCKED_STATUS = {403, 418, 451}
    _UA = {"User-Agent": "BacktestLab/1.0 (personal research)"}
    _hosts_resolved = None  # list of hosts that responded, then the rest as last resort
    LISTING = pd.Timestamp("2017-08-17", tz="UTC")

    FALLBACK = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "PAXGUSDT", "TRXUSDT", "LINKUSDT",
                "AVAXUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT", "SUIUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "UNIUSDT",
                "ATOMUSDT", "ETCUSDT", "FILUSDT", "XLMUSDT", "HBARUSDT", "AAVEUSDT", "INJUSDT", "TONUSDT", "PEPEUSDT", "SHIBUSDT"]
    _STABLE = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "AEUR", "USDE", "XUSD", "RLUSD", "USD1", "BFUSD", "U", "RWUSD",
               "EURI", "PYUSD", "USDS", "GBP", "TRY", "BRL"}
    _sym_cache = [0.0, None]

    def symbols(self, top=80):
        """Most traded Binance USDT pairs (24h quote volume), refreshed every 6 hours; fixed list if offline."""
        now = time.time()
        if self._sym_cache[1] and now - self._sym_cache[0] < 6 * 3600:
            return self._sym_cache[1]
        out = None
        try:
            r = requests.get("https://data-api.binance.vision/api/v3/ticker/24hr", params={"type": "MINI"}, timeout=8, headers=self._UA)
            if r.status_code == 200:
                rows = []
                for d in r.json():
                    sym = d.get("symbol", "")
                    if not sym.endswith("USDT") or not sym.isascii() or not sym.isalnum():
                        continue
                    base = sym[:-4]
                    if base in self._STABLE or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                        continue
                    if base.endswith("B") and base not in ("BNB", "ARB", "TRB"):
                        continue  # tokenized stocks (NVDAB, MUB, INTCB...): short history, stock-market hours
                    try:
                        qv = float(d.get("quoteVolume") or 0)
                        if float(d.get("closeTime") or 0) < (now - 3 * 86400) * 1000:
                            continue  # delisted / halted
                    except ValueError:
                        continue
                    rows.append((qv, sym))
                rows.sort(reverse=True)
                out = [s_ for _, s_ in rows[:top]]
                for s_ in self.FALLBACK[:8]:       # keep the classics on top
                    if s_ in out:
                        out.remove(s_)
                out = self.FALLBACK[:8] + out
        except Exception as e:
            print(f"[data] Binance symbol list failed: {type(e).__name__}: {e}", flush=True)
        if not out:
            return self.FALLBACK
        self._sym_cache[:] = [now, out]
        return out

    def limits_note(self, tf):
        return ("Binance: riwayat lengkap sejak koin listing (BTC sejak Agustus 2017) untuk semua timeframe. "
                "Unduhan pertama bisa makan waktu (1h dari 2017 kira-kira 30-60 detik), selanjutnya hanya candle baru yang diambil.")

    def _resolve_hosts(self):
        """Probe hosts once so a blocked/timing-out api.binance.com is not retried on every page."""
        if BinanceSource._hosts_resolved is not None:
            return BinanceSource._hosts_resolved
        def _probe(h):
            try:
                r = requests.get(
                    h + "/api/v3/klines",
                    params={"symbol": "BTCUSDT", "interval": "1h", "limit": 1},
                    timeout=6,
                    headers=self._UA,
                )
                if r.status_code == 200:
                    print(f"[data] Binance OK: {h}", flush=True)
                    return h, True, f"HTTP {r.status_code}"
                reason = "geo-blocked/forbidden" if r.status_code in self._BLOCKED_STATUS else f"HTTP {r.status_code}"
                print(f"[data] Binance skip {h}: {reason}", flush=True)
                return h, False, f"HTTP {r.status_code}"
            except Exception as e:
                print(f"[data] Binance skip {h}: {type(e).__name__}: {e}", flush=True)
                return h, False, f"{type(e).__name__}: {e}"

        good, bad = [], []
        # Preserve HOSTS order among successes so the preferred public API stays first when it works.
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.HOSTS)) as ex:
            futs = {ex.submit(_probe, h): h for h in self.HOSTS}
            for fut in as_completed(futs):
                h, ok, _msg = fut.result()
                results[h] = ok
        for h in self.HOSTS:
            (good if results.get(h) else bad).append(h)
        BinanceSource._hosts_resolved = good + bad
        if not good:
            print("[data] WARNING: no Binance host responded during probe; will still try all of them", flush=True)
        return BinanceSource._hosts_resolved

    def _req(self, params):
        last = None
        for h in self._resolve_hosts():
            try:
                r = requests.get(h + "/api/v3/klines", params=params, timeout=15, headers=self._UA)
                if r.status_code == 200:
                    return r.json()
                last = f"{h} {r.status_code} {r.text[:200]}"
                if r.status_code in self._BLOCKED_STATUS:
                    print(f"[data] Binance {h} returned HTTP {r.status_code} (blocked); trying next host", flush=True)
            except Exception as e:  # network error -> next host
                last = f"{h} {e}"
        raise RuntimeError(f"Binance request gagal: {last}")

    def _fetch(self, symbol, tf, start_ms, end_ms=None, progress=None):
        out = []
        cur = start_ms
        step = TF_MINUTES[tf] * 60_000
        now_ms = int(time.time() * 1000)
        total = max(1, ((end_ms or now_ms) - start_ms) // step)
        while True:
            params = {"symbol": symbol, "interval": tf, "startTime": int(cur), "limit": 1000}
            if end_ms:
                params["endTime"] = int(end_ms)
            rows = self._req(params)
            if not rows:
                break
            out.extend(rows)
            cur = rows[-1][0] + step
            if progress:
                progress(min(1.0, len(out) / total))
            if len(rows) < 1000 or (end_ms and cur >= end_ms):
                break
        if not out:
            return pd.DataFrame(columns=COLS)
        df = pd.DataFrame(out, columns=["t", "open", "high", "low", "close", "volume", "ct", "q", "n", "tb", "tq", "i"])
        df.index = pd.to_datetime(df["t"], unit="ms", utc=True)
        return df[COLS].astype(float)

    def get(self, symbol, tf, start=None, refresh=True, progress=None):
        p = _cache_path(self.key, symbol, tf)
        old = _load_cache(p)
        start = pd.Timestamp(start, tz="UTC") if start is not None and pd.Timestamp(start).tzinfo is None else start
        start = start or self.LISTING
        new_parts = []
        if old is None or len(old) == 0:
            new_parts.append(self._fetch(symbol, tf, int(start.timestamp() * 1000), progress=progress))
        else:
            if start < old.index[0] - pd.Timedelta(minutes=TF_MINUTES[tf]):
                new_parts.append(self._fetch(symbol, tf, int(start.timestamp() * 1000),
                                             int(old.index[0].timestamp() * 1000), progress=progress))
            if refresh:  # incremental: re-fetch from last cached candle (it may have been unfinished)
                new_parts.append(self._fetch(symbol, tf, int(old.index[-1].timestamp() * 1000)))
        new = pd.concat([x for x in new_parts if len(x)]) if any(len(x) for x in new_parts) else None
        df = _merge_save(p, old, new) if new is not None else old
        if df is None:
            raise RuntimeError("Tidak ada data dari Binance untuk simbol ini.")
        return df[df.index >= start]


# ---------------------------------------------------------------- Yahoo (forex, gold, stocks)
FX_SYMBOLS = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X", "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X", "USDCHF": "USDCHF=X", "NZDUSD": "NZDUSD=X",
    "XAUUSD (Gold futures GC=F)": "GC=F",
}
YF_INTERVAL = {"1m": ("1m", "7d"), "5m": ("5m", "60d"), "15m": ("15m", "60d"), "30m": ("30m", "60d"),
               "1h": ("60m", "730d"), "4h": ("60m", "730d"), "1d": ("1d", "max"), "1w": ("1wk", "max")}


def pip_size_for(symbol: str):
    s = symbol.upper()
    if s in POINT_SIZE:
        return POINT_SIZE[s]   # indices / oil: 1 "pip" = 1 point (SPX500 0.1, oil 0.01)
    if "GC=F" in s or "XAU" in s:
        return 0.1          # gold: 1 pip = $0.10 (convention used here)
    if "XAG" in s:
        return 0.01         # silver
    if "JPY" in s:
        return 0.01
    if "=X" in s or (len(s) == 6 and s.isalpha() and not s.endswith("USDT")):
        return 0.0001
    return None


class YahooSource(DataSource):
    key = "yahoo"
    label = "Forex & Emas - Yahoo Finance (cadangan)"
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]

    def symbols(self):
        return list(FX_SYMBOLS.keys())

    def limits_note(self, tf):
        lim = {"1m": "hanya 7 hari terakhir", "5m": "hanya 60 hari terakhir", "15m": "hanya 60 hari terakhir",
               "30m": "hanya 60 hari terakhir", "1h": "hanya +- 730 hari (2 tahun) terakhir",
               "4h": "dibentuk dari data 1h, jadi juga +- 2 tahun terakhir", "1d": "sejak 2003 (forex)",
               "1w": "sejak 2003 (forex)"}[tf]
        return (f"Yahoo Finance {tf}: {lim}. Data yang sudah pernah diambil disimpan, jadi kalau aplikasi dipakai terus, "
                "riwayatnya bertambah sendiri. Emas memakai GC=F (futures COMEX), harganya sedikit beda dengan XAUUSD spot.")

    def ticker(self, symbol):
        return FX_SYMBOLS.get(symbol, symbol)

    def pip_size(self, symbol):
        return pip_size_for(self.ticker(symbol))

    def _fetch(self, tk, tf, period):
        import yfinance as yf
        itv, maxp = YF_INTERVAL[tf]
        df = yf.download(tk, period=period or maxp, interval=itv, progress=False, auto_adjust=False,
                         multi_level_index=False, threads=False)
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=COLS)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        idx = pd.to_datetime(df.index)
        df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df = df.dropna()
        if tf == "4h":
            df = df.resample("4h", origin="epoch").agg({"open": "first", "high": "max", "low": "min",
                                                        "close": "last", "volume": "sum"}).dropna()
        return df.astype(float)

    def get(self, symbol, tf, start=None, refresh=True, progress=None):
        tk = self.ticker(symbol)
        p = _cache_path(self.key, tk, tf)
        old = _load_cache(p)
        new = None
        if old is None or len(old) == 0:
            new = self._fetch(tk, tf, None)
        elif refresh:
            age_days = (pd.Timestamp.now(tz="UTC") - old.index[-1]).days + 2
            _, maxp = YF_INTERVAL[tf]
            per = f"{min(age_days, 7 if tf == '1m' else 59 if tf in ('5m','15m','30m') else 729)}d" if maxp != "max" else f"{max(age_days, 5)}d"
            new = self._fetch(tk, tf, per)
        if new is not None and len(new) == 0 and old is None:
            raise RuntimeError(f"Yahoo tidak mengembalikan data untuk {tk} {tf}.")
        df = _merge_save(p, old, new) if new is not None and len(new) else old
        if progress:
            progress(1.0)
        if start is not None:
            start = pd.Timestamp(start)
            start = start.tz_localize("UTC") if start.tzinfo is None else start
            df = df[df.index >= start]
        return df


# ---------------------------------------------------------------- Dukascopy (forex & gold default)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import research_fx_sources as fx  # tested helpers (Dukascopy freeserv, biquote)

_last_call = [0.0]


def _throttle(min_gap=0.2):  # ~5 requests/second max, avoids Dukascopy 503s
    gap = time.time() - _last_call[0]
    if gap < min_gap:
        time.sleep(min_gap - gap)
    _last_call[0] = time.time()


# display name -> Dukascopy instrument (indices / oil). Index CFD history starts ~2012, oil ~2010-2011.
DUKA_INSTRUMENT = {"US30": "USA30.IDX/USD", "NAS100": "USATECH.IDX/USD", "SPX500": "USA500.IDX/USD", "GER40": "DEU.IDX/EUR",
                   "UK100": "GBR.IDX/GBP", "JPN225": "JPN.IDX/JPY", "WTI": "LIGHT.CMD/USD", "BRENT": "BRENT.CMD/USD"}
POINT_SIZE = {"US30": 1.0, "NAS100": 1.0, "SPX500": 0.1, "GER40": 1.0, "UK100": 1.0, "JPN225": 1.0, "WTI": 0.01, "BRENT": 0.01}
DUKA_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD", "XAUUSD", "XAGUSD",
                "EURJPY", "GBPJPY", "EURGBP", "EURAUD", "EURCAD", "EURCHF", "EURNZD", "GBPAUD", "GBPCAD", "GBPCHF", "GBPNZD",
                "AUDJPY", "AUDCAD", "AUDCHF", "AUDNZD", "CADJPY", "CHFJPY", "NZDJPY", "NZDCAD", "CADCHF"] + list(DUKA_INSTRUMENT)


class DukascopySource(DataSource):
    key = "dukascopy"
    label = "Forex & Emas - Dukascopy (default, riwayat sejak 2003, live dari biquote)"
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    START = pd.Timestamp("2003-05-04", tz="UTC")

    def symbols(self):
        return DUKA_SYMBOLS

    def limits_note(self, tf):
        extra = " Untuk 1m disarankan mulai dari beberapa bulan/tahun terakhir saja (datanya sangat besar)." if tf in ("1m", "5m") else ""
        return ("Dukascopy: riwayat sejak 2003 (EURUSD, XAUUSD, dll; indeks sejak ±2012, minyak sejak ±2011), harga BID, dipakai untuk backtest dan data unseen. "
                "Candle yang sedang berjalan + harga live diambil dari biquote.io (broker MT5), jadi bisa beda sedikit "
                "dengan arsip Dukascopy. Itu normal." + extra)

    def pip_size(self, symbol):
        return pip_size_for(symbol)

    def _fetch(self, symbol, tf, start, progress=None):
        _throttle()
        sess = fx._session()
        orig_get = sess.get
        def get(*a, **k):  # throttle every page request
            _throttle()
            return orig_get(*a, **k)
        sess.get = get
        frames, cur = [], start
        while True:  # helper stops at 200k rows per call -> continue from the last bar
            d = fx.fetch_dukascopy_freeserv(DUKA_INSTRUMENT.get(symbol.upper(), symbol), tf, cur.to_pydatetime(), None, limit=30000, session=sess)
            if len(d) == 0:
                break
            frames.append(d)
            if progress:
                span = (pd.Timestamp.now(tz="UTC") - start).total_seconds()
                progress(min(0.99, (d.time.iloc[-1] - start).total_seconds() / max(span, 1)))
            if len(d) < 200_000:
                break
            cur = d.time.iloc[-1] + pd.Timedelta(minutes=TF_MINUTES[tf])
        if not frames:
            return pd.DataFrame(columns=COLS)
        df = pd.concat(frames).set_index("time")[COLS].astype(float)
        df = df[~df.index.duplicated(keep="last")]
        # drop flat filler bars (weekends / no ticks)
        df = df[~((df.volume == 0) & (df.high == df.low))]
        if progress:
            progress(1.0)
        return df

    def get(self, symbol, tf, start=None, refresh=True, progress=None):
        p = _cache_path(self.key, symbol, tf)
        old = _load_cache(p)
        if start is not None:
            start = pd.Timestamp(start)
            start = start.tz_localize("UTC") if start.tzinfo is None else start
        start = max(start or self.START, self.START)
        parts = []
        if old is None or len(old) == 0:
            parts.append(self._fetch(symbol, tf, start, progress))
        else:
            if start < old.index[0] - pd.Timedelta(days=3):
                parts.append(self._fetch(symbol, tf, start, progress))  # older history requested
            if refresh:
                parts.append(self._fetch(symbol, tf, old.index[-1] - pd.Timedelta(minutes=TF_MINUTES[tf])))
        new = pd.concat([x for x in parts if len(x)]) if any(len(x) for x in parts) else None
        df = _merge_save(p, old, new) if new is not None else old
        if df is None:
            raise RuntimeError("Dukascopy tidak mengembalikan data.")
        return df[df.index >= start]

    def live_tail(self, symbol, tf):
        """Recent bars incl. the still-forming one from biquote (chart only, never used for backtests)."""
        if symbol.upper() in DUKA_INSTRUMENT:
            return None, None   # biquote has no matching index/oil feed; chart uses Dukascopy only
        btf = tf if tf in fx.BIQUOTE_INTERVALS else "1h"
        d = fx.fetch_biquote(symbol, btf, limit=300)
        if len(d) == 0:
            return None, None
        d = d.set_index("time")
        tick = None
        try:
            tick = fx.fetch_tick("biquote", symbol)
        except Exception:
            pass
        return d[COLS + ["is_open"]], tick


# ---------------------------------------------------------------- OANDA v20 practice (optional key)
class OandaSource(DataSource):
    """Optional. Free OANDA practice account token in env OANDA_API_TOKEN (or the UI field).
    Written from the public v20 docs; NOT tested on the box (no token)."""
    key = "oanda"
    label = "Forex & Emas - OANDA practice (butuh token gratis)"
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
    G = {"1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30", "1h": "H1", "4h": "H4", "1d": "D"}

    def symbols(self):
        return DUKA_SYMBOLS[:8]

    def limits_note(self, tf):
        return ("OANDA practice: butuh token dari akun demo gratis (isi OANDA_API_TOKEN). Maks 5000 candle per request. "
                "Bagian ini belum diuji karena belum ada token.")

    def pip_size(self, symbol):
        return pip_size_for(symbol)

    def get(self, symbol, tf, start=None, refresh=True, progress=None, max_pages=40):
        tok = os.environ.get("OANDA_API_TOKEN")
        if not tok:
            raise RuntimeError("OANDA_API_TOKEN belum diisi.")
        inst = f"{symbol[:3]}_{symbol[3:]}"
        p = _cache_path(self.key, symbol, tf)
        old = _load_cache(p)
        cur = (old.index[-1] if old is not None and len(old) else pd.Timestamp(start or "2010-01-01", tz="UTC"))
        frames = []
        for _ in range(max_pages):
            r = requests.get(f"https://api-fxpractice.oanda.com/v3/instruments/{inst}/candles",
                             headers={"Authorization": f"Bearer {tok}"},
                             params={"granularity": self.G[tf], "price": "M", "count": 5000,
                                     "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ")}, timeout=20)
            r.raise_for_status()
            cs = r.json().get("candles", [])
            if not cs:
                break
            d = pd.DataFrame([dict(t=c["time"], open=float(c["mid"]["o"]), high=float(c["mid"]["h"]),
                                   low=float(c["mid"]["l"]), close=float(c["mid"]["c"]), volume=float(c["volume"]))
                              for c in cs])
            d.index = pd.to_datetime(d.pop("t"), utc=True)
            frames.append(d[COLS])
            if len(cs) < 5000:
                break
            cur = d.index[-1]
            _throttle()
        new = pd.concat(frames) if frames else None
        df = _merge_save(p, old, new) if new is not None else old
        if df is None:
            raise RuntimeError("OANDA tidak mengembalikan data.")
        return df


# ---------------------------------------------------------------- Twelve Data (optional, free API key)
class TwelveDataSource(DataSource):
    """Optional. Needs a free API key in env TWELVEDATA_API_KEY (800 req/day on free plan).
    NOTE: written against the public docs; not tested on the box (no key)."""
    key = "twelvedata"
    label = "Forex & Emas - Twelve Data (butuh API key gratis)"
    timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    TD_TF = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1day", "1w": "1week"}
    MAP = {"EURUSD": "EUR/USD", "GBPUSD": "GBP/USD", "USDJPY": "USD/JPY", "AUDUSD": "AUD/USD", "USDCAD": "USD/CAD",
           "USDCHF": "USD/CHF", "NZDUSD": "NZD/USD", "XAUUSD": "XAU/USD"}

    def symbols(self):
        return list(self.MAP.keys())

    def limits_note(self, tf):
        return ("Twelve Data: butuh API key gratis (isi TWELVEDATA_API_KEY). Paket gratis 800 request/hari, "
                "maks 5000 candle per request. Bagian ini belum diuji karena belum ada API key.")

    def pip_size(self, symbol):
        return pip_size_for(symbol)

    def get(self, symbol, tf, start=None, refresh=True, progress=None, pages=6):
        key = os.environ.get("TWELVEDATA_API_KEY")
        if not key:
            raise RuntimeError("TWELVEDATA_API_KEY belum diisi.")
        p = _cache_path(self.key, symbol, tf)
        old = _load_cache(p)
        frames, end = [], None
        for _ in range(pages if old is None else 1):
            params = {"symbol": self.MAP.get(symbol, symbol), "interval": self.TD_TF[tf], "outputsize": 5000,
                      "apikey": key, "timezone": "UTC", "order": "ASC"}
            if end:
                params["end_date"] = end
            j = requests.get("https://api.twelvedata.com/time_series", params=params, timeout=20).json()
            if j.get("status") != "ok":
                break
            d = pd.DataFrame(j["values"])
            d.index = pd.to_datetime(d["datetime"], utc=True)
            if "volume" not in d:
                d["volume"] = 0
            frames.append(d[COLS].astype(float))
            end = d.index[0].strftime("%Y-%m-%d %H:%M:%S")
            time.sleep(8)  # free plan: 8 req/min
        new = pd.concat(frames) if frames else None
        df = _merge_save(p, old, new) if new is not None else old
        if df is None:
            raise RuntimeError("Twelve Data tidak mengembalikan data.")
        return df


SOURCES = {s.key: s for s in [BinanceSource(), DukascopySource(), YahooSource(), OandaSource(), TwelveDataSource()]}
KEY_SOURCES = {"oanda": "OANDA_API_TOKEN", "twelvedata": "TWELVEDATA_API_KEY"}  # off unless a key is given


def closed_only(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Drop the last candle if it has not closed yet (backtests use closed candles only)."""
    if len(df) == 0:
        return df
    end = df.index[-1] + pd.Timedelta(minutes=TF_MINUTES.get(tf, 60))
    if end > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def probe_sources():
    """Log which live data endpoints are reachable (shown in app logs)."""
    print(f"[data] cache dir = {CACHE_DIR} (writable check already done)", flush=True)
    try:
        hosts = BinanceSource()._resolve_hosts()
        print(f"[data] Binance host order: {hosts}", flush=True)
    except Exception as e:
        print(f"[data] Binance probe failed: {type(e).__name__}: {e}", flush=True)
    try:
        sess = fx._session()
        # one lightweight freeserv page (EURUSD 1h, last few bars)
        end = pd.Timestamp.now(tz="UTC")
        start = end - pd.Timedelta(hours=12)
        d = fx.fetch_dukascopy_freeserv("EURUSD", "1h", start.to_pydatetime(), None, limit=5, session=sess)
        print(f"[data] Dukascopy OK: {len(d)} bars, last={d.time.iloc[-1] if len(d) else 'empty'}", flush=True)
    except Exception as e:
        print(f"[data] Dukascopy probe failed: {type(e).__name__}: {e}", flush=True)
    try:
        b = fx.fetch_biquote("EURUSD", "1h", limit=3)
        tick = fx.fetch_tick("biquote", "EURUSD")
        mid = tick.get("mid") if isinstance(tick, dict) else None
        print(f"[data] biquote OK: {len(b)} bars, tick mid={mid}", flush=True)
    except Exception as e:
        print(f"[data] biquote probe failed: {type(e).__name__}: {e}", flush=True)
