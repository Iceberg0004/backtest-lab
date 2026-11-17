"""Robustness, tail risk and parameter search for the multi-leg modes in core/position.py.

Honesty rules used here:
  * parameters are chosen on in-sample data only; the unseen part is only used as the exam;
  * Monte Carlo 'risk of ruin' re-runs the SAME settings on many synthetic price paths built from random blocks of the
    real history (block bootstrap keeps volatility clusters and trends), or, for martingale, on reshuffled blocks of
    the strategy's own trades;
  * the report card is capped hard when an account blew up anywhere or the ruin probability is not tiny, so a 95 %
    win-rate martingale cannot get a good grade just because the losing streak did not happen in this sample."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from . import engine as E, position as PS, robust as R

MART_FAMS = ["rsi_mr", "zscore", "bb_squeeze", "willr", "cci", "lrc", "engulf", "ma_cross", "donchian"]
PYR_FAMS = ["donchian", "atr_ch", "supertrend", "keltner", "ma_cross"]

SPACE = {
    "grid": dict(lo_pct=[5, 10, 15, 20, 30, 40], hi_pct=[5, 10, 15, 20, 30, 40], grids=[5, 10, 20, 30, 50, 80], geo=[0, 1],
                 gdir=["long", "short", "neutral"], alloc=[0.5, 1.0, 2.0, 3.0], sl_out=[0, 5, 10], recenter=[0, 1],
                 btp=[0, 5, 10], bsl=[0, 10, 20]),
    "dca": dict(ddir=["long", "short"], start=["always", "rsi", "trend"], base=[5, 10, 20], so_size=[5, 10, 20], n_so=[2, 3, 4, 5, 6, 8, 10],
                dev=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0], sscale=[1.0, 1.2, 1.5, 2.0], vscale=[1.0, 1.5, 2.0], tp=[0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
                sl=[0, 5, 10, 20]),
    "mart": dict(base=[1, 2, 5, 10], mult=[1.5, 2.0, 2.5, 3.0], max_steps=[2, 3, 4, 5, 6, 8], anti=[0, 1]),
    "pyr": dict(pdir=["long", "short", "both"], risk=[0.5, 1.0, 2.0], stop_k=[1.5, 2.0, 3.0], add_k=[0.5, 1.0], max_u=[1, 2, 3, 4, 5]),
    "hedge": dict(size=[10, 20, 50], tp=[0.5, 1.0, 2.0, 3.0], btp=[1, 2, 5], bsl=[5, 10, 20]),
    "zone": dict(start=["trend", "candle"], base=[5, 10, 20], zone=[0.5, 1.0, 2.0, 3.0], tp=[1.0, 2.0, 4.0], max_legs=[2, 3, 4, 5, 6, 8]),
}


def _pick(v, rng):
    x = v[int(rng.integers(len(v)))]
    return x.item() if hasattr(x, "item") else x


def sample(mode, rng, direction="both", fam=None):
    p = {k: _pick(v, rng) for k, v in SPACE[mode].items()}
    if mode in ("mart", "pyr"):
        f = fam or _pick(MART_FAMS if mode == "mart" else PYR_FAMS, rng)
        p["fam"] = f
        fp = E.sample_params(f, rng)
        p["fparams"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in fp.items()}
    _fix_dir(mode, p, direction)
    return p


def _fix_dir(mode, p, direction):
    if direction in ("long", "short"):
        if mode == "grid":
            p["gdir"] = direction
        elif mode == "dca":
            p["ddir"] = direction
        elif mode == "pyr":
            p["pdir"] = direction
        elif mode == "mart":
            p["mdir"] = direction


def mutate(mode, p, rng, direction="both"):
    q = {k: (dict(v) if isinstance(v, dict) else v) for k, v in p.items()}
    keys = list(SPACE[mode])
    for _ in range(int(rng.integers(1, 3))):
        k = keys[int(rng.integers(len(keys)))]
        vals = SPACE[mode][k]
        if q.get(k) in vals:
            i = vals.index(q[k]) + int(rng.choice([-1, 1]))
            q[k] = vals[min(max(i, 0), len(vals) - 1)]
        else:
            q[k] = _pick(vals, rng)
    if mode in ("mart", "pyr") and rng.random() < 0.4:
        fp = E.mutate(q["fam"], {**PS.fam_params(q["fam"]), **q["fparams"]}, rng)
        q["fparams"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in fp.items()}
    _fix_dir(mode, q, direction)
    return q


def objective(m, min_trades=10):
    """In-sample score: return per unit of (intrabar) drawdown; a blown account or too few trades is rejected."""
    if m["blown"] or m["trades"] < min_trades or m["net"] <= 0:
        return -1e9 + m["net"]
    return m["net"] / max(m["maxdd"], 0.05)


def search(mode, cc, is_start, split, n, cf, ca, lev, so, n_var=500, algo="random", seed=7, time_budget=90, direction="both",
           fam=None, min_trades=10, progress=None):
    rng = np.random.default_rng(seed)
    t0 = time.time(); seen = set(); rows = []
    for i in range(n_var):
        if time.time() - t0 > time_budget:
            break
        if algo == "evo" and i >= n_var // 3 and rows:
            top = sorted(rows, key=lambda r: -r["score"])[:10]
            p = mutate(mode, top[int(rng.integers(len(top)))]["p"], rng, direction)
        else:
            p = sample(mode, rng, direction, fam)
        key = repr(sorted(p.items()))
        if key in seen:
            continue
        seen.add(key)
        try:
            m = PS.run(mode, p, cc, is_start, split, cf, ca, lev, so)["m"]
        except Exception:
            continue
        rows.append(dict(p=p, is_m=m, score=objective(m, min_trades)))
        if progress and i % 20 == 0:
            progress(min(1.0, max(i / n_var, (time.time() - t0) / time_budget)), f"{i + 1} variasi diuji")
    rows.sort(key=lambda r: -r["score"])
    top = rows[:20]
    for r in top:
        r["oos_m"] = PS.run(mode, r["p"], cc, split, n, cf, ca, lev, so)["m"]
    return dict(rows=top, tested=len(rows), seconds=time.time() - t0, ok_is=sum(r["score"] > -1e8 for r in rows))


# ------------------------------------------------------------------ Monte Carlo risk of ruin
def synth_cache(cc, a, b, horizon, block, rng, like):
    """Synthetic OHLC path of `horizon` bars from random blocks of real bars [a, b) (returns relative to previous close)."""
    o, h, l, c = cc.o, cc.h, cc.l, cc.c
    pc = np.r_[c[a - 1] if a > 0 else o[a], c[a:b - 1]]
    ro, rh, rl, rc = o[a:b] / pc, h[a:b] / pc, l[a:b] / pc, c[a:b] / pc
    L = b - a
    idx = []
    while len(idx) < horizon:
        s = int(rng.integers(0, max(1, L - block)))
        idx.extend(range(s, min(s + block, L)))
    idx = np.array(idx[:horizon])
    cum = c[a - 1 if a > 0 else a] * np.cumprod(rc[idx])
    prev = np.r_[c[a - 1 if a > 0 else a], cum[:-1]]
    df = pd.DataFrame({"open": prev * ro[idx], "high": prev * rh[idx], "low": prev * rl[idx], "close": cum, "volume": 1.0},
                      index=like[:horizon])
    df["high"] = df[["open", "high", "close"]].max(axis=1); df["low"] = df[["open", "low", "close"]].min(axis=1)
    s = E.Cache(df)
    # carry: keep the average real cost per bar
    s.carry_long = np.full(horizon, float(np.mean(cc.carry_long[a:b]))); s.carry_short = np.full(horizon, float(np.mean(cc.carry_short[a:b])))
    return s


def mc_ruin(mode, p, cc, a, b, cf, ca, lev, so, n_paths=200, years=1.0, block_days=5, seed=11, progress=None, base=None):
    rng = np.random.default_rng(seed)
    bpd = cc.bars_per_year / 365.25
    horizon = int(min(b - a, years * cc.bars_per_year))
    warm = 300
    out = dict(dd=[], net=[], blown=[], under=[])
    if mode == "mart":
        r = base or PS.run_mart(p, cc, a, b, cf, ca, lev, so)
        ent, ex, ret, mae = r["trades_raw"]
        if len(ret) < 5:
            return None
        per_year = len(ret) / max((b - a) / cc.bars_per_year, 1e-9)
        k = max(5, int(per_year * years))
        for i in range(n_paths):
            idx = []
            while len(idx) < k:
                s = int(rng.integers(0, max(1, len(ret) - 5)))
                idx.extend(range(s, min(s + 5, len(ret))))
            idx = np.array(idx[:k])
            eqs, eqmin, size, steps, blown, capped = PS._mart_seq(ret[idx], mae[idx], p["base"] / 100, float(p["mult"]), int(p["max_steps"]),
                                                                  bool(p["anti"]), bool(p.get("compound", False)), float(lev), float(so))
            peak = np.maximum.accumulate(np.r_[1.0, eqs])[1:]
            out["dd"].append(float(np.max(1 - np.minimum(eqmin, eqs) / peak)) if len(eqs) else 0.0)
            out["net"].append(max(float(eqs[-1] - 1), -1.0) if len(eqs) else 0.0)
            out["blown"].append(blown >= 0)
            out["under"].append(np.nan)
    else:
        like = cc.index[a:a + horizon + warm] if a + horizon + warm <= len(cc.index) else pd.date_range(cc.index[a], periods=horizon + warm, freq=f"{cc.tf_min}min")
        for i in range(n_paths):
            s = synth_cache(cc, a, b, horizon + warm, int(block_days * bpd), rng, like)
            r = PS.run(mode, p, s, warm, horizon + warm, cf, ca, lev, so)
            m = r["m"]
            out["dd"].append(min(1.0, m["maxdd"])); out["net"].append(m["net"]); out["blown"].append(m["blown"]); out["under"].append(m["under_days"])
            if progress and i % 10 == 0:
                progress((i + 1) / n_paths, f"Simulasi {i + 1}/{n_paths}")
    dd = np.array(out["dd"]); net = np.array(out["net"]); bl = np.array(out["blown"])
    return dict(n=len(dd), years=years, p_ruin=float(bl.mean()), p_dd50=float((dd >= 0.5).mean()), dd50=float(np.median(dd)),
                dd95=float(np.quantile(dd, 0.95)), dd99=float(np.quantile(dd, 0.99)), net5=float(np.quantile(net, 0.05)),
                net50=float(np.median(net)), p_loss=float((net < 0).mean()),
                under50=float(np.nanmedian(out["under"])) if np.isfinite(out["under"]).any() else None)


# ------------------------------------------------------------------ robustness + grade
def jitter(mode, p, cc, a, b, cf, ca, lev, so, k=8, seed=3):
    rng = np.random.default_rng(seed); ok = 0; done = 0
    for _ in range(k):
        q = mutate(mode, p, rng)
        try:
            m = PS.run(mode, q, cc, a, b, cf, ca, lev, so)["m"]
        except Exception:
            continue
        done += 1
        ok += int(m["net"] > 0 and not m["blown"])
    return ok / done if done else None


def tests(mode, p, cc, is_start, split, n, cf, ca, lev, so, max_dd=0.3, min_pf=1.1, mc=None, do_jitter=True):
    is_r = PS.run(mode, p, cc, is_start, split, cf, ca, lev, so)
    oos_r = PS.run(mode, p, cc, split, n, cf, ca, lev, so)
    c2 = PS.run(mode, p, cc, split, n, cf * 2, ca * 2, lev, so)["m"]
    om = oos_r["m"]
    t = dict(is_m=is_r["m"], oos_m=om, cost2x=c2,
             oos_pass=bool(om["net"] > 0 and om["maxdd"] <= max_dd and not om["blown"] and om["trades"] >= 5 and om["pf"] >= min_pf),
             jitter=jitter(mode, p, cc, split, n, cf, ca, lev, so) if do_jitter else None, mc=mc)
    t["grade"] = grade(mode, t)
    return t, is_r, oos_r


def grade(mode, t):
    gi = dict(oos_pass=t["oos_pass"], cost2x=bool(t["cost2x"]["net"] > 0 and not t["cost2x"]["blown"]), jitter=t.get("jitter"))
    score, letter, parts = R.grade(gi)
    if score is None:
        return None, "-", []
    why = []
    cap = 100.0
    blown = t["is_m"]["blown"] or t["oos_m"]["blown"]
    if blown:
        cap = min(cap, 20); why.append("akun kena margin call (stop-out) di data ini")
    mc = t.get("mc")
    if mc:
        pr = mc["p_ruin"]
        if pr > 0.10:
            cap = min(cap, 30); why.append(f"peluang kena margin call {pr * 100:.0f}% dalam {mc['years']:g} tahun (Monte Carlo)")
        elif pr > 0.05:
            cap = min(cap, 45); why.append(f"peluang kena margin call {pr * 100:.0f}% (Monte Carlo)")
        elif pr > 0.01:
            cap = min(cap, 60); why.append(f"peluang kena margin call {pr * 100:.1f}% (Monte Carlo)")
        if mc["dd95"] > 0.5:
            cap = min(cap, 49); why.append(f"5% simulasi turun lebih dari {mc['dd95'] * 100:.0f}%")
    elif mode in PS.TAIL_RISK:
        cap = min(cap, 64); why.append("tes Monte Carlo (risiko margin call) belum dijalankan")
    if max(t["is_m"]["maxdd"], t["oos_m"]["maxdd"]) > 0.5:
        cap = min(cap, 45); why.append("drawdown pernah lebih dari 50%")
    if mode in PS.TAIL_RISK and not (gi["oos_pass"] and gi["cost2x"] and (t.get("jitter") or 0) >= 0.8 and mc and mc["p_ruin"] <= 0.01):
        cap = min(cap, 79)
        if not why:
            why.append("strategi berisiko ekor: nilai A hanya kalau semua tes dan Monte Carlo lolos")
    score = min(score, cap)
    letter = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "E"
    return score, letter, why
