"""Trade and equity analytics (all from the actual simulated trades; times shown in WIB, UTC+7)."""
from __future__ import annotations
import numpy as np
import pandas as pd

WIB = "Asia/Jakarta"
DAYS = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def daily_equity(eq, cc, start):
    de = cc.day_end
    de = de[(de >= start) & (de < start + len(eq))]
    return pd.Series(eq[de - start], index=cc.index[de])


def monthly_table(eq, cc, start):
    d = daily_equity(eq, cc, start)
    m = d.resample("ME").last()
    r = m / m.shift(1).fillna(1.0) - 1
    t = pd.DataFrame({"y": r.index.year, "m": r.index.month, "r": r.values * 100})
    pv = t.pivot(index="y", columns="m", values="r").reindex(columns=range(1, 13))
    pv.columns = MONTHS
    y = d.resample("YE").last()
    pv["Tahun"] = (y / y.shift(1).fillna(1.0) - 1).values * 100
    return pv


def by_time(tr, cc):
    t = cc.index[np.maximum(tr["ent"], 0)].tz_convert(WIB)
    df = pd.DataFrame({"jam": t.hour, "hari": t.dayofweek, "ret": tr["ret"] * 100})
    def agg(g):
        return pd.Series({"Trades": len(g), "Win rate %": (g.ret > 0).mean() * 100, "Rata2/trade %": g.ret.mean(), "Total %": g.ret.sum()})
    h = df.groupby("jam")[["ret"]].apply(agg)
    d = df.groupby("hari")[["ret"]].apply(agg)
    d.index = [DAYS[i] for i in d.index]
    return h, d


def mae_mfe(tr, cc):
    """Worst (MAE) and best (MFE) open move of each trade, % of entry price, from bar highs/lows."""
    n = len(tr["ent"])
    mae = np.zeros(n); mfe = np.zeros(n)
    for k in range(n):
        a, b = int(tr["ent"][k]), int(tr["ex"][k])
        b = max(b, a)
        hi = cc.h[a:b + 1].max(); lo = cc.l[a:b + 1].min(); ep = tr["ep"][k]
        if tr["dir"][k] > 0:
            mae[k], mfe[k] = (lo / ep - 1), (hi / ep - 1)
        else:
            mae[k], mfe[k] = (1 - hi / ep), (1 - lo / ep)
    return pd.DataFrame({"MAE %": mae * 100, "MFE %": mfe * 100, "Hasil %": tr["ret"] * 100,
                         "Menang": np.where(tr["ret"] > 0, "Menang", "Kalah")})


def streaks(ret):
    w = ret > 0
    runs = []
    if len(w):
        cut = np.flatnonzero(np.diff(w.astype(int))) + 1
        for seg in np.split(w, cut):
            runs.append((bool(seg[0]), len(seg)))
    wins = [l for s, l in runs if s]; losses = [l for s, l in runs if not s]
    return dict(max_win=max(wins, default=0), max_loss=max(losses, default=0),
                avg_win=float(np.mean(wins)) if wins else 0.0, avg_loss=float(np.mean(losses)) if losses else 0.0,
                loss_runs=pd.Series(losses).value_counts().sort_index() if losses else pd.Series(dtype=int))


def durations_hours(tr, cc):
    return (tr["ex"] - tr["ent"] + 1) * cc.tf_min / 60.0


def rolling_sharpe(eq, cc, start, window_days=90):
    d = daily_equity(eq, cc, start).pct_change().dropna()
    return (d.rolling(window_days).mean() / d.rolling(window_days).std() * np.sqrt(cc.days_per_year)).dropna()


def rolling_pf(tr, cc, window=50):
    r = pd.Series(tr["ret"], index=cc.index[np.minimum(tr["ex"], len(cc.index) - 1)])
    gp = r.clip(lower=0).rolling(window).sum(); gl = (-r.clip(upper=0)).rolling(window).sum()
    return (gp / gl.replace(0, np.nan)).dropna()


def underwater(eq, cc, start, max_points=4000):
    peak = np.maximum.accumulate(eq)
    uw = (eq / peak - 1) * 100
    step = max(1, len(uw) // max_points)
    return pd.Series(uw[::step], index=cc.index[start:start + len(eq)][::step])


def benchmark(eq, cc, start):
    """Strategy vs buy & hold of the same market over the same bars."""
    c = cc.c[start:start + len(eq)]
    bh = c / c[0]
    ds = daily_equity(eq, cc, start).pct_change().dropna()
    db = daily_equity(bh, cc, start).pct_change().dropna()
    j = ds.index.intersection(db.index)
    ds, db = ds[j], db[j]
    beta = float(np.cov(ds, db)[0, 1] / db.var()) if len(j) > 10 and db.var() > 0 else np.nan
    corr = float(ds.corr(db)) if len(j) > 10 else np.nan
    bpk = np.maximum.accumulate(bh)
    return dict(bh=bh, bh_net=float(bh[-1] - 1), bh_dd=float((1 - bh / bpk).max()), strat_net=float(eq[-1] / eq[0] - 1),
                corr=corr, beta=beta)
