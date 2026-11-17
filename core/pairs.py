"""Two-asset tools: pairs trading / stat-arb, portfolio rebalancing, and DCA vs lump-sum buy & hold.
All decisions use only past data: the hedge ratio and the cointegration test use the in-sample part only,
the z-score uses a rolling window, and a signal at a candle's close earns from the NEXT candle."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def align(a: pd.DataFrame, b: pd.DataFrame):
    j = pd.concat([a["close"].rename("a"), b["close"].rename("b")], axis=1, join="inner").dropna()
    return j[(j.a > 0) & (j.b > 0)]


def adf_t(x):
    """Engle-Granger step 2: ADF t-statistic (no lags, with constant) of the residual series."""
    x = np.asarray(x, float)
    dx = np.diff(x); xl = x[:-1]
    X = np.c_[np.ones(len(xl)), xl]
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    res = dx - X @ beta
    s2 = res @ res / max(len(dx) - 2, 1)
    cov = s2 * np.linalg.inv(X.T @ X)
    return float(beta[1] / math.sqrt(cov[1, 1]))


ADF_5PCT = -3.34   # Engle-Granger 2-variable critical value (MacKinnon), 5 %


def half_life(x):
    x = np.asarray(x, float)
    dx = np.diff(x); xl = x[:-1] - x[:-1].mean()
    b = np.polyfit(xl, dx, 1)[0]
    return float(-math.log(2) / b) if b < 0 else float("inf")


def pairs_fit(j, split):
    la, lb = np.log(j.a.to_numpy()), np.log(j.b.to_numpy())
    beta, alpha = np.polyfit(lb[:split], la[:split], 1)
    spread = la - beta * lb - alpha
    return dict(beta=float(beta), alpha=float(alpha), spread=spread, adf=adf_t(spread[:split]), hl=half_life(spread[:split]))


def pairs_run(j, fit, n, z_in, z_out, z_stop, start, end, cost_a, cost_b, bars_per_year):
    """Long spread = long A, short beta-weighted B (dollar amounts 1 and |beta|, normalised to gross 1).
    cost_a / cost_b: cost per side as a fraction of the traded amount."""
    s = pd.Series(fit["spread"])
    mu, sd = s.rolling(n).mean(), s.rolling(n).std(ddof=0)
    z = ((s - mu) / sd).to_numpy()
    ra = np.r_[0, np.diff(np.log(j.a.to_numpy()))]; rb = np.r_[0, np.diff(np.log(j.b.to_numpy()))]
    b = abs(fit["beta"]); wa, wb = 1 / (1 + b), b / (1 + b)
    pos = 0; eq = 1.0; eqs = np.ones(end - start); trades = []; cur = 0.0; nb = 0
    for t in range(start, end):
        r = pos * (wa * ra[t] - wb * rb[t] * np.sign(fit["beta"]))
        eq *= math.exp(r) if abs(r) < 5 else 1.0
        cur += r
        zt = z[t]
        newpos = pos
        if not np.isnan(zt):
            if pos == 0:
                if zt > z_in:
                    newpos = -1
                elif zt < -z_in:
                    newpos = 1
            elif (pos == 1 and (zt >= -z_out or zt < -z_stop)) or (pos == -1 and (zt <= z_out or zt > z_stop)):
                newpos = 0
        if newpos != pos:
            turn = abs(newpos - pos)
            eq *= (1 - turn * (wa * cost_a + wb * cost_b))
            if pos != 0:
                trades.append(cur - (wa * cost_a + wb * cost_b) * 2); cur = 0.0
            pos = newpos
        eqs[t - start] = eq
    peak = np.maximum.accumulate(eqs)
    tr = np.array(trades)
    w, l = tr[tr > 0].sum(), -tr[tr < 0].sum()
    return dict(eq=eqs, z=z[start:end], m=dict(net=float(eqs[-1] - 1), maxdd=float((1 - eqs / peak).max()), trades=len(tr),
                winrate=float((tr > 0).mean()) if len(tr) else 0.0, pf=float(w / l) if l > 0 else (float("inf") if w > 0 else 0.0)))


def pairs_search(j, split, cost_a, cost_b, bpy, is_start=0):
    fit = pairs_fit(j, split)
    best = None; rows = []
    for n in (50, 100, 200, 400):
        for zi in (1.5, 2.0, 2.5):
            for zo in (0.0, 0.5):
                for zs in (3.5, 5.0):
                    r = pairs_run(j, fit, n, zi, zo, zs, max(is_start, n), split, cost_a, cost_b, bpy)["m"]
                    sc = r["net"] / max(r["maxdd"], 0.05) if r["trades"] >= 5 else -1e9
                    rows.append((sc, dict(n=n, z_in=zi, z_out=zo, z_stop=zs), r))
    rows.sort(key=lambda x: -x[0])
    return fit, rows


# ------------------------------------------------------------------ rebalancing and DCA benchmarks
def rebalance(j, w_a, every_bars, band, cost_a, cost_b, start, end):
    """Two-asset portfolio: target weight w_a in A. Rebalance every N bars (0 = never) or when the weight drifts more than
    `band` (0 = off). Compared with buy & hold of the same starting weights."""
    pa, pb = j.a.to_numpy(), j.b.to_numpy()
    va, vb = w_a, 1 - w_a
    ha, hb = w_a, 1 - w_a
    eq = np.ones(end - start); bh = np.ones(end - start); n_reb = 0; cost_paid = 0.0
    for t in range(start + 1, end):
        va *= pa[t] / pa[t - 1]; vb *= pb[t] / pb[t - 1]
        ha *= pa[t] / pa[t - 1]; hb *= pb[t] / pb[t - 1]
        tot = va + vb
        drift = abs(va / tot - w_a)
        if (every_bars > 0 and (t - start) % every_bars == 0) or (band > 0 and drift > band):
            move = abs(va - w_a * tot)
            c = move * cost_a + move * cost_b
            tot -= c; cost_paid += c
            va, vb = w_a * tot, (1 - w_a) * tot; n_reb += 1
        eq[t - start] = va + vb; bh[t - start] = ha + hb
    def stats(e):
        pk = np.maximum.accumulate(e)
        return dict(net=float(e[-1] - 1), maxdd=float((1 - e / pk).max()))
    return dict(eq=eq, bh=bh, m=stats(eq), m_bh=stats(bh), n_reb=n_reb, cost=cost_paid)


def dca_benchmark(close: pd.Series, every_days=7, cost=0.001):
    """Buy the same amount every N days vs. putting everything in on day 1. Returns value per 1 unit invested."""
    s = close.dropna()
    days = (s.index - s.index[0]).total_seconds() / 86400
    buy = np.r_[True, np.diff(np.floor(days / every_days)) > 0]
    n_buys = int(buy.sum())
    units = np.cumsum(np.where(buy, (1 - cost) / n_buys / s.to_numpy(), 0.0))
    invested = np.cumsum(np.where(buy, 1 / n_buys, 0.0))
    value = units * s.to_numpy()
    lump = (1 - cost) * s.to_numpy() / s.iloc[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        dd_dca = 1 - value / np.maximum.accumulate(np.maximum(value, invested))
    return dict(index=s.index, value=value, invested=invested, lump=lump, n_buys=n_buys,
                dca_final=float(value[-1]), lump_final=float(lump[-1]), avg_price=float(1 / units[-1] * (1 - cost)) if units[-1] > 0 else float("nan"),
                dd_lump=float((1 - lump / np.maximum.accumulate(lump)).max()), dd_dca=float(np.nanmax(dd_dca)))
