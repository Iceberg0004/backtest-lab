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
