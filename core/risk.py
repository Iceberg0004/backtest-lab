"""Position sizing and a prop-firm challenge simulator.

Sizing works on the strategy's own trades (signals and exits unchanged). Each trade gets a size f
(position value as a multiple of equity); its P&L is f x the 1x trade return (costs and carry included).
Equity is rebuilt bar by bar, so drawdowns inside trades and daily results are visible."""
from __future__ import annotations
import numpy as np
from . import engine as E

MODES = {"fixed": "Ukuran tetap (x modal)", "risk": "Risiko tetap % per trade (pakai jarak stop loss)",
         "vol": "Target volatilitas", "kelly": "Kelly pecahan (dibatasi)"}


def trade_sizes(mode, fam, p, cc, tr, value, max_exp, kelly_ret=None, kelly_frac=0.25):
    """Size of every trade. value = x modal (fixed) / risk % (risk) / target annual vol % (vol)."""
    j = np.maximum(tr["ent"] - 1, 0)
    ep = tr["ep"]
    if mode == "fixed":
        f = np.full(len(j), float(value))
    elif mode == "risk":
        RL, RS = E.stop_dist(p, cc)
        dist = np.where(tr["dir"] == 1, RL[j], RS[j]) / ep   # initial stop distance as a share of the entry price
        f = (value / 100.0) / np.maximum(dist, 1e-9)
    elif mode == "vol":
        bar_vol = cc.atr[j] / ep                        # ~ typical move per bar
        ann = bar_vol * np.sqrt(cc.bars_per_year)
        f = (value / 100.0) / np.maximum(ann, 1e-9)
    elif mode == "kelly":
        r = np.asarray(kelly_ret if kelly_ret is not None else tr["ret"], float)
        k = r.mean() / r.var() if len(r) > 5 and r.var() > 0 else 0.0
        f = np.full(len(j), max(0.0, k * kelly_frac))
    else:
        raise ValueError(mode)
    return np.clip(np.nan_to_num(f, nan=0.0), 0.0, float(max_exp))


def sized_equity(tr, eq1, start, f, compound=True):
    """Bar-by-bar equity with sizes f. eq1 = 1x mark-to-market equity of the same run (index 0 = bar `start`)."""
    n = len(eq1)
    E_ = np.ones(n)
    cur = 1.0                                           # equity before the current trade
    last = 0
    for k in range(len(tr["ent"])):
        a, b = int(tr["ent"][k]) - start, int(tr["ex"][k]) - start
        if a < 1 or a >= n:
            continue
        E_[last:a] = cur
        m0 = eq1[a - 1]
        seg = eq1[a:b] / m0 - 1                         # open trade, still at 1x
        E_[a:b] = cur * (1 + f[k] * seg) if compound else cur + f[k] * seg
        r = tr["ret"][k]
        cur = cur * (1 + f[k] * r) if compound else cur + f[k] * r
        last = min(b, n)
        if b < n:
            E_[b] = cur
        if cur <= 0:
            E_[a:] = max(cur, 0.0); cur = 0.0; last = n
            break
    E_[last:] = cur
    return E_


def daily_stats(E_, cc, start):
    """(day_return, day_low_vs_day_start) for every calendar day that has bars."""
    de = cc.day_end
    de = de[(de >= start) & (de < start + len(E_))] - start
    close = E_[de]
    prev = np.r_[1.0, close[:-1]]
    lows = np.minimum.reduceat(E_, np.r_[0, de[:-1] + 1]) if len(de) else np.array([])
    return close / prev - 1, lows / prev - 1, prev


def metrics_from_equity(E_, n_trades, days_per_year, years):
    peak = np.maximum.accumulate(E_)
    dd = float((1 - E_ / np.maximum(peak, 1e-12)).max())
    net = float(E_[-1] - 1)
    cagr = (E_[-1]) ** (1 / max(years, 1e-9)) - 1 if E_[-1] > 0 else -1.0
    return dict(net=net, maxdd=dd, cagr=float(cagr), trades=int(n_trades), ret_dd=net / dd if dd > 0 else 0.0)


# ------------------------------------------------------------------ prop firm
PRESETS = {
    "Contoh 2 fase (mirip aturan prop firm populer)": dict(phases=[10.0, 5.0], daily=5.0, total=10.0, min_days=4, max_days=60),
    "Contoh 1 fase (lebih ketat)": dict(phases=[10.0], daily=4.0, total=6.0, min_days=5, max_days=60),
    "Atur sendiri": None,
}


def _run_phase(rets, lows, target, daily, total, min_days, max_days, start_eq_idx=0):
    """Walk through days. Daily limit: day low (intraday) vs. day-start equity, measured in % of the INITIAL balance.
    Returns (status, days_used): status 1 pass, 0 out of time, -1 fail (daily), -2 fail (total)."""
    eq = 1.0; active = 0
    for d in range(min(len(rets), max_days)):
        low = eq * (1 + lows[d])
        if eq - low >= daily / 100:                 # loss within the day, vs. initial balance (= 1)
            return -1, d + 1
        if low <= 1 - total / 100:
            return -2, d + 1
        eq = eq * (1 + rets[d])
        if rets[d] != 0:
            active += 1
        if eq >= 1 + target / 100 and active >= min_days:
            return 1, d + 1
    return 0, min(len(rets), max_days)


def prop_monte_carlo(day_ret, day_low, rules, runs=2000, block=5, seed=1):
    """Block bootstrap of real daily results (keeps short streaks). Each run plays phase 1 then phase 2 ..."""
    rng = np.random.default_rng(seed)
    nd = len(day_ret)
    if nd < 20:
        return None
    need = rules["max_days"] * len(rules["phases"]) + block
    outcomes = {"pass_all": 0, "fail_daily": 0, "fail_total": 0, "timeout": 0}
    phase_pass = np.zeros(len(rules["phases"]))
    days_to_pass = []
    for _ in range(runs):
        starts = rng.integers(0, nd - block + 1, size=need // block + 1)
        idx = (starts[:, None] + np.arange(block)).ravel()
        r, lo = day_ret[idx], day_low[idx]
        pos = 0; ok = True; used = 0
        for ph, tgt in enumerate(rules["phases"]):
            st, d = _run_phase(r[pos:], lo[pos:], tgt, rules["daily"], rules["total"], rules["min_days"], rules["max_days"])
            pos += d; used += d
            if st == 1:
                phase_pass[ph] += 1
                continue
            ok = False
            outcomes["fail_daily" if st == -1 else "fail_total" if st == -2 else "timeout"] += 1
            break
        if ok:
            outcomes["pass_all"] += 1
            days_to_pass.append(used)
    res = {k: v / runs for k, v in outcomes.items()}
    res["phase_pass"] = (phase_pass / runs).tolist()
    res["median_days"] = float(np.median(days_to_pass)) if days_to_pass else None
    res["runs"] = runs
    return res


def prop_history(day_ret, day_low, day_index, rules):
    """Start a challenge on the first day of every month in the data (no resampling)."""
    import pandas as pd
    idx = pd.DatetimeIndex(day_index)
    starts = np.flatnonzero(np.r_[True, idx.month[1:] != idx.month[:-1]])
    out = []
    for s in starts:
        pos = s; ok = True; used = 0; why = "lolos"
        for tgt in rules["phases"]:
            st, d = _run_phase(day_ret[pos:], day_low[pos:], tgt, rules["daily"], rules["total"], rules["min_days"], rules["max_days"])
            pos += d; used += d
            if st != 1:
                ok = False
                why = {-1: "gagal: rugi harian", -2: "gagal: rugi total", 0: "waktu habis"}[st]
                break
        if pos > len(day_ret) and ok:
            why = "data habis"
        out.append(dict(start=idx[s], result=why, days=used))
    return out
