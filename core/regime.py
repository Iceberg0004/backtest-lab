"""Market regime detection (opt-in): k-means on volatility + trend features.
Features are causal (rolling windows up to bar i). The scaler and k-means are fitted on in-sample bars
only, then every bar (including the unseen part) is assigned to the nearest regime centre."""
from __future__ import annotations
import numpy as np
from . import engine as E, indicators as ind


def regime_features(cc):
    c = cc.c
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.r_[np.nan, np.diff(np.log(c))]
        import pandas as pd
        vol = pd.Series(lr).rolling(48, min_periods=48).std().to_numpy()
        e = ind.ema(c, 50)
        slope = (e - ind.shift(e, 20)) / (20 * cc.atr)          # EMA50 move per bar, in ATR units
        adx = ind.adx(cc.h, cc.l, c, cc.atr)
    return np.column_stack([np.log(vol), slope, adx / 100])


def detect(cc, is_start, split, k=3, seed=0):
    """Returns (labels int8 [-1 = not enough data], names list, centres table rows)."""
    from sklearn.cluster import KMeans
    F = regime_features(cc)
    ok = np.isfinite(F).all(axis=1)
    tr = ok.copy(); tr[:is_start] = False; tr[split:] = False
    idx = np.flatnonzero(tr)
    step = max(1, len(idx) // 50000)
    mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-12
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit((F[idx[::step]] - mu) / sd)
    lab = np.full(len(cc.c), -1, np.int8)
    lab[ok] = km.predict((F[ok] - mu) / sd)
    cen = km.cluster_centers_ * sd + mu                     # back to real units
    # order regimes by volatility, then give plain names
    order = np.argsort(cen[:, 0])
    remap = np.empty(k, np.int8); remap[order] = np.arange(k)
    lab[ok] = remap[lab[ok]]
    cen = cen[order]
    tiers = {2: ["Tenang", "Bergejolak"], 3: ["Tenang", "Normal", "Bergejolak"], 4: ["Sangat tenang", "Tenang", "Ramai", "Bergejolak"]}[k]
    names = []
    for r in range(k):
        s = cen[r, 1]
        d = "naik" if s > 0.02 else "turun" if s < -0.02 else "datar"
        names.append(f"{tiers[r]} · tren {d}")
    rows = [dict(Regime=names[r], **{"Volatilitas/candle %": float(np.exp(cen[r, 0]) * 100), "Arah tren (ATR/candle)": float(cen[r, 1]),
                                    "ADX": float(cen[r, 2] * 100), "Porsi in-sample %": float((lab[idx] == r).mean() * 100),
                                    "Porsi unseen %": float((lab[split:] == r).mean() * 100)}) for r in range(k)]
    return lab, names, rows


def perf_by_regime(tr, lab, names):
    """Trades grouped by the regime at their signal bar."""
    j = np.maximum(tr["ent"] - 1, 0)
    rg = lab[j]
    out = []
    for r, nm in enumerate(names):
        m = rg == r
        ret = tr["ret"][m]
        gp, gl = ret[ret > 0].sum(), -ret[ret < 0].sum()
        out.append({"Regime": nm, "Trades": int(m.sum()), "Win rate %": float((ret > 0).mean() * 100) if m.any() else np.nan,
                    "Profit factor": float(gp / gl) if gl > 0 else (np.inf if gp > 0 else np.nan),
                    "Net % (dijumlah)": float(ret.sum() * 100), "Rata2/trade %": float(ret.mean() * 100) if m.any() else np.nan})
    return out


def gated_signals(fam, p, cc, lab, regimes):
    le, lx, se, sx, lines = E.signals(fam, p, cc)
    ok = np.isin(lab, list(regimes))
    return le & ok, lx, se & ok, sx, lines


def combine(assign, cc, start, end, cfg, direction):
    """assign: {regime: (fam, params) or None}. Each strategy trades only when its regime is active at the signal
    bar (open trades are managed normally). Daily returns are added (each strategy uses the full size, so with
    different strategies more than one position can be open at the same time)."""
    tot = None; trades = 0
    groups = {}                     # same strategy in several regimes = ONE strategy gated on all of them
    for r, sp in assign.items():
        if sp is not None:
            k = (sp[0], tuple(sorted(sp[1].items())))
            groups.setdefault(k, [sp, []])[1].append(r)
    for (sp, regs) in groups.values():
        fam, p = sp
        m, tr = E.run(fam, p, cc, start, end, direction, cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0), daily=True,
                      sig=gated_signals(fam, p, cc, cc._regime, regs))
        tot = tr["daily"] if tot is None else tot + tr["daily"]
        trades += m["trades"]
    if tot is None:
        return None
    eq = np.cumprod(1 + tot)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return dict(daily=tot, equity=eq, net=float(eq[-1] - 1), maxdd=float((1 - eq / peak).max()), trades=trades,
                sharpe=float(tot.mean() / tot.std() * np.sqrt(cc.days_per_year)) if tot.std() > 0 else 0.0)


def auto_assign(cands, cc, lab, names, is_start, split, cfg, direction, min_trades=15):
    """For each regime pick the candidate with the best IN-SAMPLE profit factor inside that regime."""
    best = {r: (None, 0.0) for r in range(len(names))}
    for fam, p in cands:
        _, tr = E.run(fam, p, cc, is_start, split, direction, cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        for row_r, row in enumerate(perf_by_regime(tr, lab, names)):
            pf = row["Profit factor"]
            if row["Trades"] >= min_trades and np.isfinite(pf) and pf > max(1.1, best[row_r][1]):
                best[row_r] = ((fam, p), pf)
    return {r: v[0] for r, v in best.items()}
