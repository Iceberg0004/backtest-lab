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
