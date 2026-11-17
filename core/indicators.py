"""Indicator definitions written to match TradingView Pine v5 built-ins:
ta.sma, ta.ema (SMA seed), ta.rma (SMA seed), ta.rsi, ta.atr, ta.stdev (biased), ta.highest/lowest."""
import numpy as np
from numba import njit


@njit(cache=True)
def _ewm_seeded(x, n, alpha):
    out = np.full(x.shape[0], np.nan)
    cnt = 0
    s = 0.0
    started = False
    prev = np.nan
    for i in range(x.shape[0]):
        v = x[i]
        if not started:
            if np.isnan(v):
                cnt = 0
                s = 0.0
                continue
            cnt += 1
            s += v
            if cnt == n:
                prev = s / n
                out[i] = prev
                started = True
        else:
            if np.isnan(v):
                out[i] = prev
                continue
            prev = alpha * v + (1 - alpha) * prev
            out[i] = prev
    return out


def ema(x, n):
    return _ewm_seeded(np.asarray(x, float), int(n), 2.0 / (n + 1))


def rma(x, n):
    return _ewm_seeded(np.asarray(x, float), int(n), 1.0 / n)


def sma(x, n):
    import pandas as pd  # NaN-aware: any NaN in the window -> NaN (same as Pine ta.sma)
    return pd.Series(np.asarray(x, float)).rolling(int(n), min_periods=int(n)).mean().to_numpy()


def ma(kind, x, n):
    return ema(x, n) if kind == "EMA" else sma(x, n)


def stdev(x, n):  # population (Pine ta.stdev biased=true)
    m = sma(x, n)
    m2 = sma(np.asarray(x, float) ** 2, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def rsi(close, n):
    close = np.asarray(close, float)
    d = np.diff(close, prepend=np.nan)
    up = np.where(np.isnan(d), np.nan, np.maximum(d, 0))
    dn = np.where(np.isnan(d), np.nan, np.maximum(-d, 0))
    au, ad = rma(up, n), rma(dn, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ad == 0, 100.0, np.where(au == 0, 0.0, 100 - 100 / (1 + au / ad)))
    r[np.isnan(au) | np.isnan(ad)] = np.nan
    return r


def atr(high, low, close, n=14):
    high, low, close = (np.asarray(a, float) for a in (high, low, close))
    pc = np.roll(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - pc), np.abs(low - pc)))
    tr[0] = high[0] - low[0]
    return rma(tr, n)


def rolling_max(x, n):
    import pandas as pd
    return pd.Series(x).rolling(n).max().to_numpy()


def rolling_min(x, n):
    import pandas as pd
    return pd.Series(x).rolling(n).min().to_numpy()


def shift(x, k=1):
    out = np.empty_like(x, dtype=float)
    out[:k] = np.nan
    out[k:] = x[:-k]
    return out


def _arr(a, b):
    a = np.asarray(a, float)
    return a, np.broadcast_to(np.asarray(b, float), a.shape).astype(float)


def crossover(a, b):
    a, b = _arr(a, b)
    a1, b1 = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a > b) & (a1 <= b1)


def crossunder(a, b):
    a, b = _arr(a, b)
    a1, b1 = shift(a), shift(b)
    with np.errstate(invalid="ignore"):
        return (a < b) & (a1 >= b1)


# ---------------------------------------------------------------- added for regime filter + new families
# Every formula below is written out explicitly in the Pine export too (core/pine.py),
# so Python and Pine compute exactly the same numbers.
def adx(high, low, close, atr14, n=14):
    """ADX (Wilder) with DM smoothed by rma and divided by ATR(14), same as the Pine block in pine.py."""
    high, low = np.asarray(high, float), np.asarray(low, float)
    d_up = high - shift(high)
    d_dn = shift(low) - low
    with np.errstate(invalid="ignore"):
        p_dm = np.where(np.isnan(d_up), np.nan, np.where((d_up > d_dn) & (d_up > 0), d_up, 0.0))
        m_dm = np.where(np.isnan(d_dn), np.nan, np.where((d_dn > d_up) & (d_dn > 0), d_dn, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        p_di = 100 * rma(p_dm, n) / atr14
        m_di = 100 * rma(m_dm, n) / atr14
        s = p_di + m_di
        dx = np.abs(p_di - m_di) / np.where(s == 0, 1.0, s)
    return 100 * rma(dx, n)


@njit(cache=True)
def _supertrend(hl2, close, atr, f):
    n = close.shape[0]
    up = np.full(n, np.nan); dn = np.full(n, np.nan); d = np.zeros(n)
    pu = np.nan; pd_ = np.nan; dirn = 0.0
    for i in range(n):
        a = atr[i]
        if not np.isnan(a):
            bu = hl2[i] - f * a
            bd = hl2[i] + f * a
            c1 = close[i - 1] if i > 0 else np.nan
            if np.isnan(pu) or c1 <= pu:
                pu = bu
            else:
                pu = max(bu, pu)
            if np.isnan(pd_) or c1 >= pd_:
                pd_ = bd
            else:
                pd_ = min(bd, pd_)
            if dirn == 0.0:
                dirn = 1.0
            elif dirn == 1.0 and close[i] < pu:
                dirn = -1.0
            elif dirn == -1.0 and close[i] > pd_:
                dirn = 1.0
        up[i] = pu; dn[i] = pd_; d[i] = dirn
    return up, dn, d


def supertrend(high, low, close, atr_n, f):
    """Returns (lower band, upper band, direction 1/-1/0). Mirrors the explicit Pine version."""
    high, low, close = (np.asarray(a, float) for a in (high, low, close))
    return _supertrend((high + low) / 2, close, np.asarray(atr_n, float), float(f))


def stoch_rsi_k(close, rsi_len, stoch_len, smooth=3):
    r = rsi(close, rsi_len)
    hh, ll = rolling_max(r, stoch_len), rolling_min(r, stoch_len)
    rng = hh - ll
    with np.errstate(invalid="ignore", divide="ignore"):
        st = np.where(rng == 0, np.nan, 100 * (r - ll) / rng)
    k = np.array(sma(st, smooth), dtype=float)
    w = rsi_len + stoch_len + smooth + 1  # warm-up gate (also in Pine) so na-handling can never differ
    k[:w] = np.nan
    return k


def zscore(close, n):
    m, sd = sma(close, n), stdev(close, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.array(np.where(sd == 0, np.nan, (np.asarray(close, float) - m) / sd), dtype=float)
    z[:n] = np.nan
    return z


# ---------------------------------------------------------------- v3 helpers (all mirrored 1:1 in Pine)
@njit(cache=True)
def wma(x, n):
    """Pine ta.wma: weights n..1 (newest heaviest); any NaN in the window -> NaN."""
    m = x.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        s = 0.0; norm = 0.0; bad = False
        for k in range(n):
            v = x[i - k]
            if np.isnan(v):
                bad = True
                break
            w = (n - k) * n
            norm += w
            s += v * w
        if not bad:
            out[i] = s / norm
    return out


@njit(cache=True)
def linreg(x, n):
    """Least-squares line over the last n values, value at the newest bar (= Pine ta.linreg(src, n, 0)),
    computed with the same explicit loop that the Pine export uses."""
    m = x.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        sx = 0.0; sy = 0.0; sxx = 0.0; sxy = 0.0; bad = False
        for k in range(n):
            y = x[i - k]
            if np.isnan(y):
                bad = True
                break
            xx = float(n - 1 - k)
            sx += xx; sy += y; sxx += xx * xx; sxy += xx * y
        if not bad:
            den = n * sxx - sx * sx
            if den != 0:
                slope = (n * sxy - sx * sy) / den
                inter = (sy - slope * sx) / n
                out[i] = inter + slope * (n - 1)
    return out


@njit(cache=True)
def mean_dev(x, mean, n):
    m = x.shape[0]
    out = np.full(m, np.nan)
    for i in range(n - 1, m):
        s = 0.0
        for k in range(n):
            s += abs(x[i - k] - mean[i])
        out[i] = s / n
    return out


@njit(cache=True)
def kama(c, n, k1, k2):
    m = c.shape[0]
    out = np.empty(m)
    prev = np.nan
    for i in range(m):
        if np.isnan(prev):
            prev = c[i]
        elif i > n:
            chg = abs(c[i] - c[i - n])
            vol = 0.0
            for k in range(n):
                vol += abs(c[i - k] - c[i - k - 1])
            er = 0.0 if vol == 0 else chg / vol
            sc = (er * k1 + k2) ** 2
            prev = prev + sc * (c[i] - prev)
        out[i] = prev
    return out


@njit(cache=True)
def psar(h, l, c, start, inc, mx):
    m = c.shape[0]
    d = np.zeros(m); sar = np.full(m, np.nan)
    s = np.nan; e = np.nan; a = np.nan; dr = 0
    for i in range(m):
        if i == 1:
            dr = 1 if c[1] >= c[0] else -1
            s = l[0] if dr == 1 else h[0]
            e = h[1] if dr == 1 else l[1]
            a = start
        elif i > 1:
            s = s + a * (e - s)
            if dr == 1:
                s = min(s, l[i - 1], l[i - 2])
                if l[i] < s:
                    dr = -1; s = e; e = l[i]; a = start
                elif h[i] > e:
                    e = h[i]; a = min(a + inc, mx)
            else:
                s = max(s, h[i - 1], h[i - 2])
                if h[i] > s:
                    dr = 1; s = e; e = h[i]; a = start
                elif l[i] < e:
                    e = l[i]; a = min(a + inc, mx)
        d[i] = dr; sar[i] = s
    return d, sar


@njit(cache=True)
def heikin_counts(o, h, l, c):
    m = c.shape[0]
    hc = (o + h + l + c) / 4
    ho = np.empty(m)
    g = np.zeros(m); r = np.zeros(m)
    for i in range(m):
        ho[i] = (o[i] + c[i]) / 2 if i == 0 else (ho[i - 1] + hc[i - 1]) / 2
        if hc[i] > ho[i]:
            g[i] = (g[i - 1] if i > 0 else 0) + 1
        if hc[i] < ho[i]:
            r[i] = (r[i - 1] if i > 0 else 0) + 1
    return g, r, ho, hc


@njit(cache=True)
def aroon(h, l, n):
    m = h.shape[0]
    up = np.full(m, np.nan); dn = np.full(m, np.nan)
    for i in range(n, m):
        hv = h[i]; hi = 0; lv = l[i]; li = 0
        for k in range(1, n + 1):
            if h[i - k] > hv:
                hv = h[i - k]; hi = k
            if l[i - k] < lv:
                lv = l[i - k]; li = k
        up[i] = 100.0 * (n - hi) / n
        dn[i] = 100.0 * (n - li) / n
    return up, dn


@njit(cache=True)
def nr_setup(h, l, c, setup, win):
    """Setup bar high/low carried forward + bars since setup (for NR / inside-bar breakouts)."""
    m = c.shape[0]
    sh = np.full(m, np.nan); sl = np.full(m, np.nan); since = np.full(m, 1000000.0)
    ch = np.nan; cl = np.nan; s = 1000000.0
    for i in range(m):
        if setup[i]:
            ch = h[i]; cl = l[i]; s = 0.0
        else:
            s += 1
        sh[i] = ch; sl[i] = cl; since[i] = s
    return sh, sl, since


@njit(cache=True)
def opening_range(h, l, hr, day, s, ln):
    m = h.shape[0]
    oh = np.full(m, np.nan); ol = np.full(m, np.nan)
    ch = np.nan; cl = np.nan
    for i in range(m):
        inw = hr[i] >= s and hr[i] < s + ln
        prev_in = False
        if i > 0:
            prev_in = hr[i - 1] >= s and hr[i - 1] < s + ln
        if inw and ((not prev_in) or day[i] != day[i - 1]):
            ch = h[i]; cl = l[i]
        elif inw:
            ch = max(ch, h[i]); cl = min(cl, l[i])
        oh[i] = ch; ol[i] = cl
    return oh, ol


@njit(cache=True)
def divergence(h, l, r, R):
    """Regular RSI divergence on confirmed pivots (R bars each side). Signal on the confirmation bar."""
    m = h.shape[0]
    bull = np.zeros(m, np.bool_); bear = np.zeros(m, np.bool_)
    ppl = np.nan; prl = np.nan; pph = np.nan; prh = np.nan
    W = 2 * R + 1
    for i in range(W - 1, m):
        lo = l[i]; hi = h[i]
        for k in range(1, W):
            lo = min(lo, l[i - k]); hi = max(hi, h[i - k])
        if l[i - R] == lo:
            if l[i - R] < ppl and r[i - R] > prl:
                bull[i] = True
            ppl = l[i - R]; prl = r[i - R]
        if h[i - R] == hi:
            if h[i - R] > pph and r[i - R] < prh:
                bear[i] = True
            pph = h[i - R]; prh = r[i - R]
    return bull, bear


@njit(cache=True)
def htf_trend(t_ms, c, period_ms, n):
    """Higher-timeframe EMA trend without lookahead: an HTF candle is only used after it has closed
    (detected on the first lower-timeframe bar of the next HTF period). Returns (up, down) booleans."""
    m = c.shape[0]
    up = np.zeros(m, np.bool_); dn = np.zeros(m, np.bool_)
    a = 2.0 / (n + 1)
    e = np.nan; s = 0.0; cnt = 0; last = np.nan
    for i in range(m):
        if i > 0 and (t_ms[i] // period_ms) != (t_ms[i - 1] // period_ms):
            x = c[i - 1]
            last = x
            cnt += 1
            if cnt <= n:
                s += x
                if cnt == n:
                    e = s / n
            else:
                e = a * x + (1 - a) * e
        if not np.isnan(e):
            up[i] = last > e
            dn[i] = last < e
    return up, dn


def rolling_sum(x, n):
    import pandas as pd
    return pd.Series(np.asarray(x, float)).rolling(int(n), min_periods=int(n)).sum().to_numpy()
