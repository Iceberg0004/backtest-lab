"""v7 strategy families (TradingView / retail classics). Same contract as core/families.py:
Python signals and a Pine block that compute EXACTLY the same raw signals (checked by tools/pinecheck)."""
from __future__ import annotations
import math
import numpy as np
from numba import njit
from . import indicators as ind
from .families import fam, _cx, _cu, T4, R_

NAN = float("nan")


def _b(x):
    return np.nan_to_num(np.asarray(x, float), nan=0).astype(np.bool_)


def _prev(b):
    b = np.asarray(b, bool)
    return np.r_[False, b[:-1]]


def _ev(cond):
    """True on the bar where a condition becomes true."""
    c = _b(cond)
    return c & ~_prev(c)


def _pmax(a, b):     # Pine math.max: na if any input is na
    return NAN if (a != a or b != b) else (a if a > b else b)


def _pmin(a, b):
    return NAN if (a != a or b != b) else (a if a < b else b)


def _sh(x, k=1):
    return ind.shift(np.asarray(x, float), k)


def _roc(c, n):
    s = _sh(c, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (c - s) / s


def _days(cc):
    """(group id per bar, first index of each group) for UTC days."""
    d = cc.day
    first = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    g = np.cumsum(np.r_[True, d[1:] != d[:-1]]) - 1
    return g, first


def _groups(key):
    first = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    g = np.cumsum(np.r_[True, key[1:] != key[:-1]]) - 1
    return g, first


def _prev_hl(cc, key):
    """High / low of the previous group (day or week) for every bar."""
    g, first = _groups(key)
    hm = np.maximum.reduceat(cc.h, first); lm = np.minimum.reduceat(cc.l, first)
    ph = np.where(g >= 1, hm[np.maximum(g - 1, 0)], np.nan); pl = np.where(g >= 1, lm[np.maximum(g - 1, 0)], np.nan)
    return ph, pl


PINE_DAY = ["xxD = math.floor(time / 86400000)", "xxNewDay = bar_index == 0 or xxD != xxD[1]", "xxHr = math.floor(time / 3600000) % 24"]


def _pday(px):
    return [l.replace("xx", px) for l in PINE_DAY]


# ------------------------------------------------------------------ momentum oscillators
@fam("trix", "Momentum (TRIX)", {"trix_len": (8, 30), "trix_sig": [5, 9, 12], "trend": T4})
class _:
    def py(p, cc):
        n, s = p["trix_len"], p["trix_sig"]
        def f():
            e3 = ind.ema(ind.ema(ind.ema(cc.c, n), n), n)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = 100 * (e3 - _sh(e3)) / _sh(e3)
            return t, ind.ema(t, s)
        t, sg = cc.get(("trix", n, s), f)
        up, dn = _cx(t, sg), _cu(t, sg)
        return up, dn, dn, up, {}

    def pine(p):
        n, s = p["trix_len"], p["trix_sig"]
        return [f"trE3 = ta.ema(ta.ema(ta.ema(close, {n}), {n}), {n})", "trX = 100 * (trE3 - trE3[1]) / trE3[1]", f"trS = ta.ema(trX, {s})",
                "longEntry = ta.crossover(trX, trS)", "longExit = ta.crossunder(trX, trS)", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Beli saat TRIX({p['trix_len']}) memotong ke atas signal {p['trix_sig']}; short saat memotong ke bawah"


@fam("kst", "Momentum (KST, Know Sure Thing)", {"kst_sig": [6, 9, 12], "trend": T4})
class _:
    def py(p, cc):
        def f():
            c = cc.c
            r1 = ind.sma(_roc(c, 10), 10); r2 = ind.sma(_roc(c, 15), 10); r3 = ind.sma(_roc(c, 20), 10); r4 = ind.sma(_roc(c, 30), 15)
            return r1 + 2 * r2 + 3 * r3 + 4 * r4
        k = cc.get(("kst",), f)
        sg = cc.get(("kst_s", p["kst_sig"]), lambda: ind.sma(k, p["kst_sig"]))
        up, dn = _cx(k, sg), _cu(k, sg)
        return up, dn, dn, up, {}

    def pine(p):
        L = []
        for i, (n, m) in enumerate([(10, 10), (15, 10), (20, 10), (30, 15)], 1):
            L.append(f"ksR{i} = ta.sma(100 * (close - close[{n}]) / close[{n}], {m})")
        return L + ["ksK = ksR1 + 2 * ksR2 + 3 * ksR3 + 4 * ksR4", f"ksS = ta.sma(ksK, {p['kst_sig']})",
                    "longEntry = ta.crossover(ksK, ksS)", "longExit = ta.crossunder(ksK, ksS)", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Beli saat KST (ROC 10/15/20/30) memotong ke atas signal {p['kst_sig']}; short saat memotong ke bawah"


@fam("elder", "Elder Ray (bull / bear power)", {"er_len": (8, 30), "trend": T4})
class _:
    def py(p, cc):
        e = cc.get(("ema", "c", p["er_len"]), lambda: ind.ema(cc.c, p["er_len"]))
        with np.errstate(invalid="ignore"):
            up = e > _sh(e); dn = e < _sh(e)
            bull = cc.h - e; bear = cc.l - e
            cl = up & (bear < 0) & (bear > _sh(bear)); cs = dn & (bull > 0) & (bull < _sh(bull))
        return _ev(cl), _b(dn), _ev(cs), _b(up), {}

    def pine(p):
        return [f"erE = ta.ema(close, {p['er_len']})", "erUp = erE > erE[1]", "erDn = erE < erE[1]", "erBull = high - erE", "erBear = low - erE",
                "erCL = erUp and erBear < 0 and erBear > erBear[1]", "erCS = erDn and erBull > 0 and erBull < erBull[1]",
                "longEntry = erCL and not erCL[1]", "longExit = erDn", "shortEntry = erCS and not erCS[1]", "shortExit = erUp"]

    def desc(p):
        return (f"Elder Ray: beli saat EMA {p['er_len']} naik dan bear power negatif tapi menguat; keluar saat EMA turun. "
                "Short kebalikannya")


@fam("ao", "Awesome Oscillator (Bill Williams)", {"ao_s": [34, 55], "trend": T4})
class _:
    def py(p, cc):
        def f():
            m = (cc.h + cc.l) / 2
            return ind.sma(m, 5) - ind.sma(m, p["ao_s"])
        a = cc.get(("ao", p["ao_s"]), f)
        up, dn = _cx(a, 0.0), _cu(a, 0.0)
        return up, dn, dn, up, {}

    def pine(p):
        return [f"aoV = ta.sma((high + low) / 2, 5) - ta.sma((high + low) / 2, {p['ao_s']})",
                "longEntry = ta.crossover(aoV, 0.0)", "longExit = ta.crossunder(aoV, 0.0)", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Beli saat Awesome Oscillator (SMA 5 - SMA {p['ao_s']} dari median) naik melewati 0; short saat turun melewati 0"


@fam("fisher", "Fisher Transform (Ehlers)", {"fi_len": (7, 20), "fi_th": [0.0, 1.0, 1.5], "trend": T4})
class _:
    def py(p, cc):
        n = p["fi_len"]
        def f():
            m = (cc.h + cc.l) / 2
            hi, lo = ind.rolling_max(m, n), ind.rolling_min(m, n)
            out = np.zeros(len(m)); v = 0.0; fi = 0.0
            for i in range(len(m)):
                if hi[i] == hi[i]:
                    x = (m[i] - lo[i]) / (hi[i] - lo[i]) if hi[i] > lo[i] else 0.5
                    v = 0.66 * (x - 0.5) + 0.67 * v
                    v = 0.999 if v > 0.999 else (-0.999 if v < -0.999 else v)
                    fi = 0.5 * math.log((1 + v) / (1 - v)) + 0.5 * fi
                out[i] = fi
            return out
        fi = cc.get(("fisher", n), f)
        f1 = _sh(fi); th = p["fi_th"]
        with np.errstate(invalid="ignore"):
            up = _cx(fi, f1) & (fi < -th); dn = _cu(fi, f1) & (fi > th)
        return up, _cu(fi, f1), dn, _cx(fi, f1), {}

    def pine(p):
        n, th = p["fi_len"], R_(float(p["fi_th"]))
        return [f"fiM = (high + low) / 2", f"fiHi = ta.highest(fiM, {n})", f"fiLo = ta.lowest(fiM, {n})", "var float fiV = 0.0", "var float fiF = 0.0",
                "if not na(fiHi)",
                "    fiX = fiHi > fiLo ? (fiM - fiLo) / (fiHi - fiLo) : 0.5",
                "    fiV := 0.66 * (fiX - 0.5) + 0.67 * fiV",
                "    fiV := fiV > 0.999 ? 0.999 : fiV < -0.999 ? -0.999 : fiV",
                "    fiF := 0.5 * math.log((1 + fiV) / (1 - fiV)) + 0.5 * fiF",
                f"longEntry = ta.crossover(fiF, fiF[1]) and fiF < -{th}", "longExit = ta.crossunder(fiF, fiF[1])",
                f"shortEntry = ta.crossunder(fiF, fiF[1]) and fiF > {th}", "shortExit = ta.crossover(fiF, fiF[1])"]

    def desc(p):
        return f"Fisher Transform({p['fi_len']}): beli saat berbalik naik di bawah -{p['fi_th']}, short saat berbalik turun di atas {p['fi_th']}"


# ------------------------------------------------------------------ more moving averages
def _alma_w(n, off=0.85, sig=6.0):
    m = off * (n - 1); s = n / sig
    w = [math.exp(-((k - m) ** 2) / (2 * s * s)) for k in range(n)]
    norm = 0.0
    for x in w:
        norm += x
    return w, norm


@njit(cache=True)
def _alma_loop(c, n, w, norm):
    out = np.full(len(c), np.nan)
    for i in range(n - 1, len(c)):
        acc = 0.0
        for k in range(n):
            acc += c[i - (n - 1 - k)] * w[k]
        out[i] = acc / norm
    return out


def _alma(c, n):
    w, norm = _alma_w(n)
    return _alma_loop(np.asarray(c, float), int(n), np.asarray(w, float), float(norm))


T3A = 0.7
T3C = (-T3A ** 3, 3 * T3A ** 2 + 3 * T3A ** 3, -6 * T3A ** 2 - 3 * T3A - 3 * T3A ** 3, 1 + 3 * T3A + T3A ** 3 + 3 * T3A ** 2)


def _ma3(kind, c, n):
    if kind == "ZLEMA":
        lag = (n - 1) // 2
        return ind.ema(c + (c - _sh(c, lag)), n)
    if kind == "ALMA":
        return _alma(c, n)
    if kind == "T3":
        e1 = ind.ema(c, n); e2 = ind.ema(e1, n); e3 = ind.ema(e2, n); e4 = ind.ema(e3, n); e5 = ind.ema(e4, n); e6 = ind.ema(e5, n)
        return T3C[0] * e6 + T3C[1] * e5 + T3C[2] * e4 + T3C[3] * e3
    base = ind.ema(c, n)          # McGinley dynamic, started from the EMA
    out = np.full(len(c), np.nan); md = NAN
    for i in range(len(c)):
        if md != md:
            md = base[i]
        else:
            r = c[i] / md
            md = md + (c[i] - md) / (n * (r * r * r * r))
        out[i] = md
    return out


def _ma3_pine(kind, n, v):
    if kind == "ZLEMA":
        lag = (n - 1) // 2
        return [f"{v} = ta.ema(close + (close - close[{lag}]), {n})"]
    if kind == "ALMA":
        w, norm = _alma_w(n)
        return [f"var float[] {v}W = array.from({', '.join(R_(x) for x in w)})", f"float {v} = na",
                f"if bar_index >= {n - 1}", f"    float {v}A = 0.0", f"    for k = 0 to {n - 1}",
                f"        {v}A += close[{n - 1} - k] * array.get({v}W, k)", f"    {v} := {v}A / {R_(norm)}"]
    if kind == "T3":
        return [f"{v}1 = ta.ema(close, {n})", f"{v}2 = ta.ema({v}1, {n})", f"{v}3 = ta.ema({v}2, {n})", f"{v}4 = ta.ema({v}3, {n})",
                f"{v}5 = ta.ema({v}4, {n})", f"{v}6 = ta.ema({v}5, {n})",
                f"{v} = {R_(T3C[0])} * {v}6 + {R_(T3C[1])} * {v}5 + {R_(T3C[2])} * {v}4 + {R_(T3C[3])} * {v}3"]
    return [f"{v}E = ta.ema(close, {n})", f"var float {v} = na",
            f"if na({v})", f"    {v} := {v}E", "else", f"    {v}R = close / {v}", f"    {v} := {v} + (close - {v}) / ({n} * ({v}R * {v}R * {v}R * {v}R))"]


@fam("ma3", "Trend following (ZLEMA / ALMA / T3 / McGinley cross)", {"ma3_type": ["ZLEMA", "ALMA", "T3", "MCG"], "fast": (5, 40), "slow": (20, 150), "trend": T4})
class _:
    def py(p, cc):
        k = p["ma3_type"]
        f = cc.get(("ma3", k, p["fast"]), lambda: _ma3(k, cc.c, p["fast"]))
        s = cc.get(("ma3", k, p["slow"]), lambda: _ma3(k, cc.c, p["slow"]))
        up, dn = _cx(f, s), _cu(f, s)
        return up, dn, dn, up, {f"{k} {p['fast']}": f, f"{k} {p['slow']}": s}

    def pine(p):
        k = p["ma3_type"]
        return _ma3_pine(k, p["fast"], "m3F") + _ma3_pine(k, p["slow"], "m3S") + [
            "longEntry = ta.crossover(m3F, m3S)", "longExit = ta.crossunder(m3F, m3S)", "shortEntry = longExit", "shortExit = longEntry",
            f"plot(m3F, '{k} {p['fast']}', color=color.orange)", f"plot(m3S, '{k} {p['slow']}', color=color.blue)"]

    def desc(p):
        nm = {"MCG": "McGinley"}.get(p["ma3_type"], p["ma3_type"])
        return f"Beli saat {nm} {p['fast']} memotong ke atas {nm} {p['slow']}; short saat memotong ke bawah"


@fam("ribbon", "Triple MA / MA ribbon", {"rb_mode": [0, 1], "fast": [5, 8, 10], "mid": [13, 21], "slow": [34, 50, 55], "trend": T4})
class _:
    def py(p, cc):
        E_ = lambda n: cc.get(("ema", "c", n), lambda: ind.ema(cc.c, n))
        lens = [p["fast"], p["mid"], p["slow"]] if p["rb_mode"] == 0 else [8, 13, 21, 34, 55, 89]
        es = [E_(n) for n in lens]
        with np.errstate(invalid="ignore"):
            al = np.ones(len(cc.c), bool); ash = np.ones(len(cc.c), bool)
            for a, b in zip(es[:-1], es[1:]):
                al &= a > b; ash &= a < b
        f, m = (es[0], es[1]) if p["rb_mode"] == 0 else (es[0], es[2])
        return _ev(al), _cu(f, m), _ev(ash), _cx(f, m), {}

    def pine(p):
        lens = [p["fast"], p["mid"], p["slow"]] if p["rb_mode"] == 0 else [8, 13, 21, 34, 55, 89]
        L = [f"rbE{i} = ta.ema(close, {n})" for i, n in enumerate(lens)]
        L.append("rbAL = " + " and ".join(f"rbE{i} > rbE{i + 1}" for i in range(len(lens) - 1)))
        L.append("rbAS = " + " and ".join(f"rbE{i} < rbE{i + 1}" for i in range(len(lens) - 1)))
        m = 1 if p["rb_mode"] == 0 else 2
        return L + ["longEntry = rbAL and not rbAL[1]", f"longExit = ta.crossunder(rbE0, rbE{m})",
                    "shortEntry = rbAS and not rbAS[1]", f"shortExit = ta.crossover(rbE0, rbE{m})"]

    def desc(p):
        if p["rb_mode"] == 0:
            return f"Beli saat EMA {p['fast']} > {p['mid']} > {p['slow']} (baru tersusun); keluar saat EMA {p['fast']} turun di bawah {p['mid']}. Short kebalikannya"
        return "MA ribbon: beli saat EMA 8/13/21/34/55/89 tersusun naik; keluar saat EMA 8 di bawah EMA 21. Short kebalikannya"


# ------------------------------------------------------------------ TradingView trailing-stop systems
@fam("utbot", "UT Bot (ATR trailing stop)", {"ut_key": [1.0, 1.5, 2.0, 3.0], "ut_atr": [5, 10, 14], "trend": T4})
class _:
    def py(p, cc):
        k, n = p["ut_key"], p["ut_atr"]
        def f():
            a = ind.atr(cc.h, cc.l, cc.c, n); c = cc.c
            ts = np.full(len(c), np.nan); prev = NAN
            for i in range(len(c)):
                pv = 0.0 if prev != prev else prev
                nl = k * a[i]
                c1 = c[i - 1] if i > 0 else NAN
                if c[i] > pv and c1 > pv:
                    v = _pmax(pv, c[i] - nl)
                elif c[i] < pv and c1 < pv:
                    v = _pmin(pv, c[i] + nl)
                elif c[i] > pv:
                    v = c[i] - nl
                else:
                    v = c[i] + nl
                ts[i] = v; prev = v
            return ts
        ts = cc.get(("utbot", k, n), f)
        up, dn = _cx(cc.c, ts), _cu(cc.c, ts)
        return up, dn, dn, up, {"UT stop": ts}

    def pine(p):
        return [f"utA = ta.atr({p['ut_atr']})", f"utL = {R_(float(p['ut_key']))} * utA", "var float utT = na", "utP = nz(utT[1], 0)",
                "utT := close > utP and close[1] > utP ? math.max(utP, close - utL) : close < utP and close[1] < utP ? math.min(utP, close + utL) : close > utP ? close - utL : close + utL",
                "longEntry = ta.crossover(close, utT)", "longExit = ta.crossunder(close, utT)", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"UT Bot: beli saat harga menembus ke atas trailing stop {p['ut_key']} x ATR({p['ut_atr']}); short saat menembus ke bawah"


@fam("ssl", "SSL channel", {"ssl_len": (5, 30), "trend": T4})
class _:
    def py(p, cc):
        n = p["ssl_len"]
        def f():
            sh, sl = ind.sma(cc.h, n), ind.sma(cc.l, n)
            hv = np.zeros(len(cc.c)); prev = 0.0
            for i in range(len(cc.c)):
                v = 1.0 if cc.c[i] > sh[i] else (-1.0 if cc.c[i] < sl[i] else prev)
                hv[i] = v; prev = v
            return hv
        hv = cc.get(("ssl", n), f)
        h1 = _sh(hv)
        up = (hv == 1) & (h1 == -1); dn = (hv == -1) & (h1 == 1)
        return up, dn, dn, up, {}

    def pine(p):
        n = p["ssl_len"]
        return [f"sslH = ta.sma(high, {n})", f"sslL = ta.sma(low, {n})", "var float sslV = 0.0",
                "sslV := close > sslH ? 1.0 : close < sslL ? -1.0 : sslV",
                "longEntry = sslV == 1 and sslV[1] == -1", "longExit = sslV == -1 and sslV[1] == 1", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"SSL channel {p['ssl_len']}: beli saat close menembus SMA high (garis berbalik naik); short saat menembus SMA low"


@fam("qqe", "QQE (RSI + trailing band)", {"qq_rsi": [6, 14], "qq_k": [3.0, 4.236], "trend": T4})
class _:
    def py(p, cc):
        n, k = p["qq_rsi"], p["qq_k"]
        def f():
            W = 2 * n - 1
            rm = ind.ema(ind.rsi(cc.c, n), 5)
            ar = np.abs(_sh(rm) - rm)
            dar = ind.ema(ind.ema(ar, W), W) * k
            m = len(rm); lb = np.zeros(m); sb = np.zeros(m); tr = np.ones(m)
            plb = 0.0; psb = 0.0; ppsb = NAN; pplb = NAN; ptr = 1.0
            for i in range(m):
                r0 = rm[i]; r1 = rm[i - 1] if i > 0 else NAN
                nl = r0 - dar[i]; ns = r0 + dar[i]
                l_ = _pmax(plb, nl) if (r1 > plb and r0 > plb) else nl
                s_ = _pmin(psb, ns) if (r1 < psb and r0 < psb) else ns
                # ta.cross(a, b) with a = rsiMa, b = shortband[1] / longband[1]
                cu_ = (r0 > psb and r1 <= ppsb) or (r0 < psb and r1 >= ppsb)
                cl_ = (plb > r0 and pplb <= r1) or (plb < r0 and pplb >= r1)
                t = 1.0 if cu_ else (-1.0 if cl_ else ptr)
                lb[i] = l_; sb[i] = s_; tr[i] = t
                pplb, ppsb = plb, psb
                plb, psb, ptr = l_, s_, t
            return tr
        t = cc.get(("qqe", n, k), f)
        t1 = _sh(t)
        up = (t == 1) & (t1 == -1); dn = (t == -1) & (t1 == 1)
        return up, dn, dn, up, {}

    def pine(p):
        n, k = p["qq_rsi"], R_(float(p["qq_k"]))
        W = 2 * n - 1
        return [f"qqR = ta.ema(ta.rsi(close, {n}), 5)", "qqA = math.abs(qqR[1] - qqR)", f"qqD = ta.ema(ta.ema(qqA, {W}), {W}) * {k}",
                "var float qqLB = 0.0", "var float qqSB = 0.0", "var float qqT = 1.0",
                "qqNL = qqR - qqD", "qqNS = qqR + qqD",
                "qqPL = qqLB", "qqPS = qqSB",
                "qqLB := qqR[1] > qqPL and qqR > qqPL ? math.max(qqPL, qqNL) : qqNL",
                "qqSB := qqR[1] < qqPS and qqR < qqPS ? math.min(qqPS, qqNS) : qqNS",
                "qqCU = (qqR > qqPS and qqR[1] <= qqPS[1]) or (qqR < qqPS and qqR[1] >= qqPS[1])",
                "qqCL = (qqPL > qqR and qqPL[1] <= qqR[1]) or (qqPL < qqR and qqPL[1] >= qqR[1])",
                "qqT := qqCU ? 1.0 : qqCL ? -1.0 : qqT",
                "longEntry = qqT == 1 and qqT[1] == -1", "longExit = qqT == -1 and qqT[1] == 1", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"QQE (RSI {p['qq_rsi']}, faktor {p['qq_k']}): beli saat tren QQE berbalik naik; short saat berbalik turun"


@fam("wt", "WaveTrend (LazyBear)", {"wt_n1": [9, 10], "wt_n2": [21], "wt_ob": [0, 53, 60], "trend": T4})
class _:
    def py(p, cc):
        n1, n2 = p["wt_n1"], p["wt_n2"]
        def f():
            ap = (cc.h + cc.l + cc.c) / 3
            esa = ind.ema(ap, n1); d = ind.ema(np.abs(ap - esa), n1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ci = (ap - esa) / (0.015 * d)
            w1 = ind.ema(ci, n2)
            return w1, ind.sma(w1, 4)
        w1, w2 = cc.get(("wt", n1, n2), f)
        ob = p["wt_ob"]
        with np.errstate(invalid="ignore"):
            up = _cx(w1, w2) & (w1 < -ob); dn = _cu(w1, w2) & (w1 > ob)
        return up, _cu(w1, w2), dn, _cx(w1, w2), {}

    def pine(p):
        n1, n2, ob = p["wt_n1"], p["wt_n2"], p["wt_ob"]
        return ["wtAp = (high + low + close) / 3", f"wtEsa = ta.ema(wtAp, {n1})", f"wtD = ta.ema(math.abs(wtAp - wtEsa), {n1})",
                "wtCi = (wtAp - wtEsa) / (0.015 * wtD)", f"wt1 = ta.ema(wtCi, {n2})", "wt2 = ta.sma(wt1, 4)",
                f"longEntry = ta.crossover(wt1, wt2) and wt1 < -{ob}", "longExit = ta.crossunder(wt1, wt2)",
                f"shortEntry = ta.crossunder(wt1, wt2) and wt1 > {ob}", "shortExit = ta.crossover(wt1, wt2)"]

    def desc(p):
        return f"WaveTrend: beli saat garis WT memotong ke atas di bawah -{p['wt_ob']}; short saat memotong ke bawah di atas {p['wt_ob']}"


@fam("chx", "Chandelier Exit (sebagai sinyal masuk)", {"ce_len": [14, 22], "ce_mult": [2.0, 3.0], "trend": T4})
class _:
    def py(p, cc):
        n, k = p["ce_len"], p["ce_mult"]
        def f():
            c = cc.c
            a = k * ind.atr(cc.h, cc.l, c, n)
            lsr = ind.rolling_max(c, n) - a; ssr = ind.rolling_min(c, n) + a
            m = len(c); d = np.ones(m)
            pls = NAN; pss = NAN; pd_ = 1.0
            for i in range(m):
                c1 = c[i - 1] if i > 0 else NAN
                lp = lsr[i] if pls != pls else pls
                sp = ssr[i] if pss != pss else pss
                ls = _pmax(lsr[i], lp) if c1 > lp else lsr[i]
                ss = _pmin(ssr[i], sp) if c1 < sp else ssr[i]
                dv = 1.0 if c[i] > sp else (-1.0 if c[i] < lp else pd_)
                d[i] = dv; pls, pss, pd_ = ls, ss, dv
            return d
        d = cc.get(("chx", n, k), f)
        d1 = _sh(d)
        up = (d == 1) & (d1 == -1); dn = (d == -1) & (d1 == 1)
        return up, dn, dn, up, {}

    def pine(p):
        n, k = p["ce_len"], R_(float(p["ce_mult"]))
        return [f"ceA = {k} * ta.atr({n})", f"ceLR = ta.highest(close, {n}) - ceA", f"ceSR = ta.lowest(close, {n}) + ceA",
                "var float ceL = na", "var float ceS = na", "var float ceD = 1.0",
                "ceLP = na(ceL) ? ceLR : ceL", "ceSP = na(ceS) ? ceSR : ceS",
                "ceL := close[1] > ceLP ? math.max(ceLR, ceLP) : ceLR", "ceS := close[1] < ceSP ? math.min(ceSR, ceSP) : ceSR",
                "ceD := close > ceSP ? 1.0 : close < ceLP ? -1.0 : ceD",
                "longEntry = ceD == 1 and ceD[1] == -1", "longExit = ceD == -1 and ceD[1] == 1", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Chandelier Exit ({p['ce_len']}, {p['ce_mult']} x ATR): beli saat arah berbalik naik; short saat berbalik turun"


@fam("rngf", "Range Filter", {"rf_len": [50, 100], "rf_mult": [2.0, 3.0], "trend": T4})
class _:
    def py(p, cc):
        n, k = p["rf_len"], p["rf_mult"]
        def f():
            c = cc.c
            av = ind.ema(np.abs(c - _sh(c)), n)
            r = ind.ema(av, 2 * n - 1) * k
            m = len(c); d = np.zeros(m); pf = NAN; pd_ = 0.0
            for i in range(m):
                pv = 0.0 if pf != pf else pf
                if c[i] > pv:
                    x = c[i] - r[i]; fl = pv if x < pv else x
                else:
                    x = c[i] + r[i]; fl = pv if x > pv else x
                dv = 1.0 if fl > pf else (-1.0 if fl < pf else pd_)
                d[i] = dv; pf, pd_ = fl, dv
            return d
        d = cc.get(("rngf", n, k), f)
        d1 = _sh(d)
        up = (d == 1) & (d1 == -1); dn = (d == -1) & (d1 == 1)
        return up, dn, dn, up, {}

    def pine(p):
        n, k = p["rf_len"], R_(float(p["rf_mult"]))
        return [f"rfAv = ta.ema(math.abs(close - close[1]), {n})", f"rfR = ta.ema(rfAv, {2 * n - 1}) * {k}",
                "var float rfF = na", "var float rfD = 0.0", "rfPF = rfF", "rfPV = nz(rfF, 0.0)",
                "rfF := close > rfPV ? (close - rfR < rfPV ? rfPV : close - rfR) : (close + rfR > rfPV ? rfPV : close + rfR)",
                "rfD := rfF > rfPF ? 1.0 : rfF < rfPF ? -1.0 : rfD",
                "longEntry = rfD == 1 and rfD[1] == -1", "longExit = rfD == -1 and rfD[1] == 1", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Range Filter ({p['rf_len']}, {p['rf_mult']}): beli saat filter berbalik naik; short saat berbalik turun"


def _gauss_coef(per, poles=4):
    beta = (1 - math.cos(2 * math.pi / per)) / (math.pow(1.414, 2.0 / poles) - 1)
    al = -beta + math.sqrt(beta * beta + 2 * beta)
    x = 1 - al
    return al ** 4, 4 * x, 6 * x ** 2, 4 * x ** 3, x ** 4


@fam("gauss", "Gaussian channel", {"ga_per": [100, 144, 200], "ga_mult": [1.0, 1.414, 2.0], "trend": T4})
class _:
    def py(p, cc):
        per, k = p["ga_per"], p["ga_mult"]
        def f():
            a4, b1, b2, b3, b4 = _gauss_coef(per)
            c, h, l = cc.c, cc.h, cc.l
            def filt(x):
                out = np.zeros(len(x)); f1 = f2 = f3 = f4 = 0.0
                for i in range(len(x)):
                    v = a4 * x[i] + b1 * f1 - b2 * f2 + b3 * f3 - b4 * f4
                    out[i] = v; f4, f3, f2, f1 = f3, f2, f1, v
                return out
            c1 = _sh(c)
            tr = np.where(np.isnan(c1), h - l, np.maximum(np.maximum(h - l, np.abs(h - c1)), np.abs(l - c1)))
            return filt(c), filt(tr)
        fv, ft = cc.get(("gauss", per), f)
        hb = fv + k * ft; lb = fv - k * ft
        return _cx(cc.c, hb), _cu(cc.c, fv), _cu(cc.c, lb), _cx(cc.c, fv), {"Gauss": fv}

    def pine(p):
        a4, b1, b2, b3, b4 = (R_(x) for x in _gauss_coef(p["ga_per"]))
        k = R_(float(p["ga_mult"]))
        L = ["gaTr = na(close[1]) ? high - low : math.max(math.max(high - low, math.abs(high - close[1])), math.abs(low - close[1]))"]
        for v, src in (("gaF", "close"), ("gaT", "gaTr")):
            L += [f"var float {v}1 = 0.0", f"var float {v}2 = 0.0", f"var float {v}3 = 0.0", f"var float {v}4 = 0.0",
                  f"{v} = {a4} * {src} + {b1} * {v}1 - {b2} * {v}2 + {b3} * {v}3 - {b4} * {v}4",
                  f"{v}4 := {v}3", f"{v}3 := {v}2", f"{v}2 := {v}1", f"{v}1 := {v}"]
        return L + [f"gaHb = gaF + {k} * gaT", f"gaLb = gaF - {k} * gaT",
                    "longEntry = ta.crossover(close, gaHb)", "longExit = ta.crossunder(close, gaF)",
                    "shortEntry = ta.crossunder(close, gaLb)", "shortExit = ta.crossover(close, gaF)"]

    def desc(p):
        return f"Gaussian channel ({p['ga_per']}, {p['ga_mult']} x range): beli saat tembus band atas, keluar saat di bawah garis tengah; short kebalikannya"


# ------------------------------------------------------------------ candle patterns
@fam("candles", "Pola candle (star, 3 soldiers, harami, tweezer)", {"cp_kind": [0, 1, 2, 3], "trend": [0, 50, 100, 200]})
class _:
    def py(p, cc):
        o, h, l, c = cc.o, cc.h, cc.l, cc.c
        o1, h1, l1, c1, o2, c2 = _sh(o), _sh(h), _sh(l), _sh(c), _sh(o, 2), _sh(c, 2)
        k = p["cp_kind"]
        with np.errstate(invalid="ignore"):
            if k == 0:
                bull = (c2 < o2) & (np.abs(c1 - o1) < 0.5 * np.abs(c2 - o2)) & (c > o) & (c > (o2 + c2) / 2)
                bear = (c2 > o2) & (np.abs(c1 - o1) < 0.5 * np.abs(c2 - o2)) & (c < o) & (c < (o2 + c2) / 2)
            elif k == 1:
                bull = (c > o) & (c1 > o1) & (c2 > o2) & (c > c1) & (c1 > c2) & (o > o1) & (o1 > o2)
                bear = (c < o) & (c1 < o1) & (c2 < o2) & (c < c1) & (c1 < c2) & (o < o1) & (o1 < o2)
            elif k == 2:
                bull = (c1 < o1) & (c > o) & (c < o1) & (o > c1)
                bear = (c1 > o1) & (c < o) & (o < c1) & (c > o1)
            else:
                bull = (c1 < o1) & (c > o) & (np.abs(l - l1) <= 0.1 * cc.atr)
                bear = (c1 > o1) & (c < o) & (np.abs(h - h1) <= 0.1 * cc.atr)
        bull, bear = _b(bull), _b(bear)
        return bull, bear, bear, bull, {}

    def pine(p):
        k = p["cp_kind"]
        if k == 0:
            L = ["cpBull = close[2] < open[2] and math.abs(close[1] - open[1]) < 0.5 * math.abs(close[2] - open[2]) and close > open and close > (open[2] + close[2]) / 2",
                 "cpBear = close[2] > open[2] and math.abs(close[1] - open[1]) < 0.5 * math.abs(close[2] - open[2]) and close < open and close < (open[2] + close[2]) / 2"]
        elif k == 1:
            L = ["cpBull = close > open and close[1] > open[1] and close[2] > open[2] and close > close[1] and close[1] > close[2] and open > open[1] and open[1] > open[2]",
                 "cpBear = close < open and close[1] < open[1] and close[2] < open[2] and close < close[1] and close[1] < close[2] and open < open[1] and open[1] < open[2]"]
        elif k == 2:
            L = ["cpBull = close[1] < open[1] and close > open and close < open[1] and open > close[1]",
                 "cpBear = close[1] > open[1] and close < open and open < close[1] and close > open[1]"]
        else:
            L = ["cpAtr = ta.atr(14)",
                 "cpBull = close[1] < open[1] and close > open and math.abs(low - low[1]) <= 0.1 * cpAtr",
                 "cpBear = close[1] > open[1] and close < open and math.abs(high - high[1]) <= 0.1 * cpAtr"]
        return L + ["longEntry = cpBull", "longExit = cpBear", "shortEntry = cpBear", "shortExit = cpBull"]

    def desc(p):
        return {0: "Morning star (beli) / evening star (short)", 1: "Three white soldiers (beli) / three black crows (short)",
                2: "Harami bullish (beli) / bearish (short)", 3: "Tweezer bottom (beli) / tweezer top (short), low/high sama ±0.1 ATR"}[p["cp_kind"]]


@fam("gapfill", "Gap fill (tutup gap)", {"gp_k": [0.5, 1.0, 2.0], "trend": T4})
class _:
    def py(p, cc):
        k = p["gp_k"]
        o, c = cc.o, cc.c
        c1, a1 = _sh(c), _sh(cc.atr)
        with np.errstate(invalid="ignore"):
            le = _b(o < c1 - k * a1); se = _b(o > c1 + k * a1)
        lx = np.zeros(len(c), bool); sx = np.zeros(len(c), bool); rl = NAN; rs = NAN
        for i in range(len(c)):
            if le[i]:
                rl = c1[i]
            if se[i]:
                rs = c1[i]
            lx[i] = c[i] >= rl; sx[i] = c[i] <= rs
        return le, lx, se, sx, {}

    def pine(p):
        k = R_(float(p["gp_k"]))
        return ["gpAtr = ta.atr(14)", f"gpL = open < close[1] - {k} * gpAtr[1]", f"gpS = open > close[1] + {k} * gpAtr[1]",
                "var float gpRL = na", "var float gpRS = na", "if gpL", "    gpRL := close[1]", "if gpS", "    gpRS := close[1]",
                "longEntry = gpL", "longExit = close >= gpRL", "shortEntry = gpS", "shortExit = close <= gpRS"]

    def desc(p):
        return f"Gap fill: harga buka gap > {p['gp_k']} x ATR dari close sebelumnya, masuk berlawanan arah gap, keluar saat gap tertutup"


# ------------------------------------------------------------------ session / level breakouts
@fam("lvl_bo", "Breakout level (high/low kemarin, open mingguan, range Asia, high/low minggu lalu)", {"lv_kind": [0, 1, 2, 3], "trend": T4}, intraday=True)
class _:
    def py(p, cc):
        k = p["lv_kind"]; c = cc.c
        if k in (0, 3):
            key = cc.day if k == 0 else (cc.day + 3) // 7
            ph, pl = cc.get(("prevhl", k), lambda: _prev_hl(cc, key))
            up, dn = _cx(c, ph), _cu(c, pl)
            return up, dn, dn, up, {"High lalu": ph, "Low lalu": pl}
        if k == 1:
            def f():
                g, first = _groups((cc.day + 3) // 7)
                return cc.o[first][g]
            wo = cc.get(("wopen",), f)
            up, dn = _cx(c, wo), _cu(c, wo)
            return up, dn, dn, up, {"Open mingguan": wo}
        def g_():
            m = len(c); ah = np.full(m, np.nan); al = np.full(m, np.nan); h_ = NAN; l_ = NAN
            d = cc.day; hr = cc.hour
            for i in range(m):
                if i == 0 or d[i] != d[i - 1]:
                    h_ = NAN; l_ = NAN
                if hr[i] < 7:
                    h_ = cc.h[i] if h_ != h_ else max(h_, cc.h[i]); l_ = cc.l[i] if l_ != l_ else min(l_, cc.l[i])
                ah[i] = h_; al[i] = l_
            return ah, al
        ah, al = cc.get(("asia",), g_)
        win = (cc.hour >= 7) & (cc.hour < 16)
        ext = cc.hour >= 20
        up, dn = _cx(c, ah) & win, _cu(c, al) & win
        return up, ext | dn, dn, ext | up, {"Asia high": ah, "Asia low": al}

    def pine(p):
        k = p["lv_kind"]
        L = _pday("lv")
        if k in (0, 3):
            if k == 0:
                L += ["lvNew = lvNewDay"]
            else:
                L += ["lvW = math.floor((lvD + 3) / 7)", "lvNew = bar_index == 0 or lvW != lvW[1]"]
            L += ["var float lvCH = na", "var float lvCL = na", "var float lvPH = na", "var float lvPL = na",
                  "if lvNew", "    lvPH := lvCH", "    lvPL := lvCL", "    lvCH := high", "    lvCL := low",
                  "else", "    lvCH := math.max(lvCH, high)", "    lvCL := math.min(lvCL, low)",
                  "longEntry = ta.crossover(close, lvPH)", "longExit = ta.crossunder(close, lvPL)", "shortEntry = longExit", "shortExit = longEntry"]
        elif k == 1:
            L += ["lvW = math.floor((lvD + 3) / 7)", "var float lvWO = na", "if bar_index == 0 or lvW != lvW[1]", "    lvWO := open",
                  "longEntry = ta.crossover(close, lvWO)", "longExit = ta.crossunder(close, lvWO)", "shortEntry = longExit", "shortExit = longEntry"]
        else:
            L += ["var float lvAH = na", "var float lvAL = na", "if lvNewDay", "    lvAH := na", "    lvAL := na",
                  "if lvHr < 7", "    lvAH := na(lvAH) ? high : math.max(lvAH, high)", "    lvAL := na(lvAL) ? low : math.min(lvAL, low)",
                  "lvWin = lvHr >= 7 and lvHr < 16", "lvExt = lvHr >= 20",
                  "lvUp = ta.crossover(close, lvAH) and lvWin", "lvDn = ta.crossunder(close, lvAL) and lvWin",
                  "longEntry = lvUp", "longExit = lvExt or lvDn", "shortEntry = lvDn", "shortExit = lvExt or lvUp"]
        return L

    def desc(p):
        return {0: "Beli saat tembus high kemarin, short saat tembus low kemarin", 1: "Beli saat harga naik melewati open minggu ini, short saat turun melewatinya",
                2: "Breakout range Asia (00-07 UTC) antara 07-16 UTC, tutup setelah 20 UTC", 3: "Beli saat tembus high minggu lalu, short saat tembus low minggu lalu"}[p["lv_kind"]]


@fam("odrive", "Opening drive (momentum awal sesi)", {"od_s": [7, 13], "od_n": [1, 2], "od_k": [0.5, 1.0], "od_hold": [3, 6]}, intraday=True)
class _:
    def py(p, cc):
        S, N, k = p["od_s"], p["od_n"], p["od_k"]
        E_ = min(S + N + p["od_hold"], 24)
        def f():
            m = len(cc.c); so = np.full(m, np.nan); v = NAN
            for i in range(m):
                if i == 0 or cc.day[i] != cc.day[i - 1]:
                    v = NAN
                if cc.hour[i] == S:
                    v = cc.o[i]
                so[i] = v
            return so
        so = cc.get(("sopen", S), f)
        mv = cc.c - so; at = cc.atr; hr = cc.hour
        with np.errstate(invalid="ignore"):
            sig = hr == S + N - 1
            le = sig & (mv > k * at); se = sig & (mv < -k * at)
        ext = (hr >= E_ - 1) | (hr < S)
        return _b(le), ext, _b(se), ext, {}

    def pine(p):
        S, N, k = p["od_s"], p["od_n"], R_(float(p["od_k"]))
        E_ = min(S + N + p["od_hold"], 24)
        return _pday("od") + ["odAtr = ta.atr(14)", "var float odO = na", "if odNewDay", "    odO := na", f"if odHr == {S}", "    odO := open",
                              "odMv = close - odO", f"odSig = odHr == {S + N - 1}",
                              f"longEntry = odSig and odMv > {k} * odAtr", f"shortEntry = odSig and odMv < -{k} * odAtr",
                              f"odExt = odHr >= {E_ - 1} or odHr < {S}", "longExit = odExt", "shortExit = odExt"]

    def desc(p):
        s = {7: "London", 13: "New York"}[p["od_s"]]
        return f"Opening drive {s}: kalau {p['od_n']} jam pertama bergerak > {p['od_k']} x ATR, ikut arahnya, tahan ±{p['od_hold']} jam"


@fam("lwvb", "Larry Williams volatility breakout", {"lw_k": [0.3, 0.5, 0.7], "trend": T4}, intraday=True)
class _:
    def py(p, cc):
        k = p["lw_k"]
        def f():
            ph, pl = cc.get(("prevhl", 0), lambda: _prev_hl(cc, cc.day))
            g, first = _groups(cc.day)
            return cc.o[first][g], ph - pl
        do, pr = cc.get(("lwv",), f)
        up = do + k * pr; dn = do - k * pr
        ext = cc.hour >= 23
        return _cx(cc.c, up), ext, _cu(cc.c, dn), ext, {}

    def pine(p):
        k = R_(float(p["lw_k"]))
        return _pday("lw") + ["var float lwCH = na", "var float lwCL = na", "var float lwPH = na", "var float lwPL = na", "var float lwO = na",
                              "if lwNewDay", "    lwPH := lwCH", "    lwPL := lwCL", "    lwCH := high", "    lwCL := low", "    lwO := open",
                              "else", "    lwCH := math.max(lwCH, high)", "    lwCL := math.min(lwCL, low)",
                              "lwR = lwPH - lwPL", f"lwUp = lwO + {k} * lwR", f"lwDn = lwO - {k} * lwR",
                              "longEntry = ta.crossover(close, lwUp)", "shortEntry = ta.crossunder(close, lwDn)",
                              "longExit = lwHr >= 23", "shortExit = lwHr >= 23"]

    def desc(p):
        return f"Larry Williams: beli saat harga > open hari ini + {p['lw_k']} x range kemarin, short saat < open - {p['lw_k']} x range; tutup akhir hari"


@fam("poc", "Volume POC kemarin (kasar)", {"pc_mode": [0, 1], "trend": T4}, intraday=True, needs_volume=True)
class _:
    def py(p, cc):
        def f():
            m = len(cc.c); out = np.full(m, np.nan); poc = NAN; mv = NAN; px = NAN
            for i in range(m):
                if i == 0 or cc.day[i] != cc.day[i - 1]:
                    poc = px; mv = cc.v[i]; px = cc.c[i]
                elif cc.v[i] > mv:
                    mv = cc.v[i]; px = cc.c[i]
                out[i] = poc
            return out
        poc = cc.get(("poc",), f)
        c = cc.c
        if p["pc_mode"] == 0:
            up, dn = _cx(c, poc), _cu(c, poc)
            return up, dn, dn, up, {"POC kemarin": poc}
        return _cu(c, poc - cc.atr), _cx(c, poc), _cx(c, poc + cc.atr), _cu(c, poc), {"POC kemarin": poc}

    def pine(p):
        L = _pday("pc") + ["var float pcPoc = na", "var float pcMv = na", "var float pcPx = na",
                           "if pcNewDay", "    pcPoc := pcPx", "    pcMv := volume", "    pcPx := close",
                           "else if volume > pcMv", "    pcMv := volume", "    pcPx := close"]
        if p["pc_mode"] == 0:
            return L + ["longEntry = ta.crossover(close, pcPoc)", "longExit = ta.crossunder(close, pcPoc)", "shortEntry = longExit", "shortExit = longEntry"]
        return L + ["pcAtr = ta.atr(14)", "longEntry = ta.crossunder(close, pcPoc - pcAtr)", "longExit = ta.crossover(close, pcPoc)",
                    "shortEntry = ta.crossover(close, pcPoc + pcAtr)", "shortExit = ta.crossunder(close, pcPoc)"]

    def desc(p):
        b = "harga candle dengan volume terbesar kemarin (POC kasar, bukan volume profile penuh)"
        return (f"Beli saat tembus ke atas {b}, short saat tembus ke bawah" if p["pc_mode"] == 0
                else f"Mean reversion ke {b}: beli 1 ATR di bawahnya, keluar saat kembali ke POC")


# ------------------------------------------------------------------ volume divergences
def _vsrc(cc, k):
    def f():
        h, l, c, v = cc.h, cc.l, cc.c, cc.v
        if k == 0:
            d = c - _sh(c)
            with np.errstate(invalid="ignore"):
                return np.cumsum(np.where(d > 0, v, np.where(d < 0, -v, 0.0)))
        with np.errstate(invalid="ignore", divide="ignore"):
            mf = np.where(h == l, 0.0, ((c - l) - (h - c)) / (h - l) * v)
        if k == 1:
            return np.cumsum(mf)
        with np.errstate(invalid="ignore", divide="ignore"):
            return ind.sma(mf, 20) / ind.sma(v, 20)
    return cc.get(("vsrc", k), f)


@fam("vdiv", "Divergence volume (OBV / A/D / CMF)", {"vd_src": [0, 1, 2], "dv_piv": (3, 6), "trend": T4}, needs_volume=True)
class _:
    def py(p, cc):
        r = _vsrc(cc, p["vd_src"])
        bull, bear = cc.get(("vdiv", p["vd_src"], p["dv_piv"]), lambda: ind.divergence(cc.h, cc.l, np.asarray(r, float), p["dv_piv"]))
        return bull, bear, bear, bull, {}

    def pine(p):
        R = p["dv_piv"]; W = 2 * R + 1; k = p["vd_src"]
        if k == 0:
            L = ["var float vdS = 0.0", "vdS += close > close[1] ? volume : close < close[1] ? -volume : 0.0"]
        else:
            L = ["vdMf = high == low ? 0.0 : ((close - low) - (high - close)) / (high - low) * volume"]
            L += ["var float vdS = 0.0", "vdS += vdMf"] if k == 1 else ["vdS = ta.sma(vdMf, 20) / ta.sma(volume, 20)"]
        return L + [f"vdLo = ta.lowest(low, {W})", f"vdHi = ta.highest(high, {W})",
                    "var float vdPl = na", "var float vdPr = na", "var float vdPh = na", "var float vdPhr = na",
                    "bool vdBull = false", "bool vdBear = false",
                    f"if bar_index >= {W - 1} and low[{R}] == vdLo",
                    f"    vdBull := low[{R}] < vdPl and vdS[{R}] > vdPr", f"    vdPl := low[{R}]", f"    vdPr := vdS[{R}]",
                    f"if bar_index >= {W - 1} and high[{R}] == vdHi",
                    f"    vdBear := high[{R}] > vdPh and vdS[{R}] < vdPhr", f"    vdPh := high[{R}]", f"    vdPhr := vdS[{R}]",
                    "longEntry = vdBull", "longExit = vdBear", "shortEntry = vdBear", "shortExit = vdBull"]

    def desc(p):
        s = {0: "OBV", 1: "Accumulation/Distribution", 2: "Chaikin Money Flow (20)"}[p["vd_src"]]
        return f"Divergence {s}: harga low lebih rendah tapi {s} lebih tinggi → beli (pivot {p['dv_piv']}); short kebalikannya"


# ------------------------------------------------------------------ simple classics
@fam("bbpb", "Bollinger %B (mean reversion)", {"pb_len": (10, 40), "pb_k": [2.0, 2.5], "pb_exit": [0.5, 1.0], "trend": T4})
class _:
    def py(p, cc):
        n, k, x = p["pb_len"], p["pb_k"], p["pb_exit"]
        def f():
            m = ind.sma(cc.c, n); sd = ind.stdev(cc.c, n)
            up = m + k * sd; dn = m - k * sd
            with np.errstate(invalid="ignore", divide="ignore"):
                return (cc.c - dn) / (up - dn)
        pb = cc.get(("pb", n, k), f)
        return _cx(pb, 0.0), _cx(pb, x), _cu(pb, 1.0), _cu(pb, 1.0 - x), {}

    def pine(p):
        n, k, x = p["pb_len"], R_(float(p["pb_k"])), R_(float(p["pb_exit"]))
        return [f"pbM = ta.sma(close, {n})", f"pbSd = ta.stdev(close, {n})", f"pbUp = pbM + {k} * pbSd", f"pbDn = pbM - {k} * pbSd",
                "pbB = (close - pbDn) / (pbUp - pbDn)",
                "longEntry = ta.crossover(pbB, 0.0)", f"longExit = ta.crossover(pbB, {x})",
                "shortEntry = ta.crossunder(pbB, 1.0)", f"shortExit = ta.crossunder(pbB, 1.0 - {x})"]

    def desc(p):
        return f"Bollinger %B ({p['pb_len']}, {p['pb_k']}): beli saat %B kembali naik di atas 0, keluar di {p['pb_exit']}; short kebalikannya"


@fam("dcmid", "Donchian middle line", {"dc_len": (10, 60), "trend": T4})
class _:
    def py(p, cc):
        n = p["dc_len"]
        mid = cc.get(("dcmid", n), lambda: (ind.rolling_max(cc.h, n) + ind.rolling_min(cc.l, n)) / 2)
        up, dn = _cx(cc.c, mid), _cu(cc.c, mid)
        return up, dn, dn, up, {"Donchian tengah": mid}

    def pine(p):
        n = p["dc_len"]
        return [f"dcM = (ta.highest(high, {n}) + ta.lowest(low, {n})) / 2", "longEntry = ta.crossover(close, dcM)",
                "longExit = ta.crossunder(close, dcM)", "shortEntry = longExit", "shortExit = longEntry"]

    def desc(p):
        return f"Beli saat close naik melewati garis tengah Donchian {p['dc_len']}; short saat turun melewatinya"


@fam("roc", "Momentum / Rate of Change", {"roc_len": (5, 50), "roc_th": [0.0, 1.0, 2.0], "trend": T4})
class _:
    def py(p, cc):
        r = cc.get(("roc", p["roc_len"]), lambda: _roc(cc.c, p["roc_len"]))
        th = p["roc_th"]
        return _cx(r, th), _cu(r, 0.0), _cu(r, -th), _cx(r, 0.0), {}

    def pine(p):
        n, th = p["roc_len"], R_(float(p["roc_th"]))
        return [f"rcV = 100 * (close - close[{n}]) / close[{n}]", f"longEntry = ta.crossover(rcV, {th})", "longExit = ta.crossunder(rcV, 0.0)",
                f"shortEntry = ta.crossunder(rcV, -{th})", "shortExit = ta.crossover(rcV, 0.0)"]

    def desc(p):
        return f"Beli saat ROC({p['roc_len']}) naik di atas {p['roc_th']}%, keluar saat di bawah 0; short kebalikannya"


@fam("rsi2", "RSI-2 (Connors)", {"r2_lo": [5, 10, 15], "r2_ma": [5, 10], "trend": [0]})
class _:
    def py(p, cc):
        r = cc.get(("rsi", 2), lambda: ind.rsi(cc.c, 2))
        t = cc.get(("sma", 200), lambda: ind.sma(cc.c, 200))
        e = cc.get(("sma", p["r2_ma"]), lambda: ind.sma(cc.c, p["r2_ma"]))
        lo = p["r2_lo"]; c = cc.c
        with np.errstate(invalid="ignore"):
            return _b((r < lo) & (c > t)), _b(c > e), _b((r > 100 - lo) & (c < t)), _b(c < e), {}

    def pine(p):
        lo = p["r2_lo"]
        return ["r2R = ta.rsi(close, 2)", "r2T = ta.sma(close, 200)", f"r2E = ta.sma(close, {p['r2_ma']})",
                f"longEntry = r2R < {lo} and close > r2T", "longExit = close > r2E",
                f"shortEntry = r2R > {100 - lo} and close < r2T", "shortExit = close < r2E"]

    def desc(p):
        return f"RSI-2 Connors: beli saat RSI(2) < {p['r2_lo']} di atas SMA 200, keluar saat close > SMA {p['r2_ma']}; short kebalikannya"


# ------------------------------------------------------------------ v7.1: relative strength vs a benchmark (BTC)
def _bench(cc):
    return cc.x["bench"]


@fam("rs_bench", "Kekuatan relatif vs BTC (rasio harga)", {"rs_fast": (5, 30), "rs_slow": (20, 150), "rs_mode": [0, 0, 1], "trend": T4},
     needs={"bench"})
class _:
    def py(p, cc):
        f, s = p["rs_fast"], p["rs_slow"]
        def ratio():
            with np.errstate(invalid="ignore", divide="ignore"):
                return cc.c / _bench(cc)
        r = cc.get(("rs_ratio",), ratio)
        ef = cc.get(("rs_ema", f), lambda: ind.ema(r, f))
        if p["rs_mode"] == 0:          # ratio EMA fast vs slow: the coin starts to beat / lag BTC
            es = cc.get(("rs_ema", s), lambda: ind.ema(r, s))
            return _cx(ef, es), _cu(ef, es), _cu(ef, es), _cx(ef, es), {}
        # mode 1: ratio momentum (rate of change of the ratio over rs_slow bars) crosses 0
        def roc():
            with np.errstate(invalid="ignore", divide="ignore"):
                o = np.full(len(ef), NAN); o[s:] = 100 * (ef[s:] - ef[:-s]) / ef[:-s]
            return o
        m = cc.get(("rs_roc", f, s), roc)
        return _cx(m, 0.0), _cu(m, 0.0), _cu(m, 0.0), _cx(m, 0.0), {}

    def pine(p):
        f, s = p["rs_fast"], p["rs_slow"]
        out = ["rsR = close / benchC", f"rsF = ta.ema(rsR, {f})"]
        if p["rs_mode"] == 0:
            out += [f"rsS = ta.ema(rsR, {s})", "longEntry = ta.crossover(rsF, rsS)", "longExit = ta.crossunder(rsF, rsS)",
                    "shortEntry = ta.crossunder(rsF, rsS)", "shortExit = ta.crossover(rsF, rsS)"]
        else:
            out += [f"rsM = 100 * (rsF - rsF[{s}]) / rsF[{s}]", "longEntry = ta.crossover(rsM, 0.0)", "longExit = ta.crossunder(rsM, 0.0)",
                    "shortEntry = ta.crossunder(rsM, 0.0)", "shortExit = ta.crossover(rsM, 0.0)"]
        return out

    def desc(p):
        if p["rs_mode"] == 0:
            return (f"Beli saat koin mulai lebih kuat dari BTC (EMA {p['rs_fast']} rasio harga koin/BTC naik di atas EMA {p['rs_slow']}), "
                    "short saat mulai lebih lemah")
        return f"Beli saat momentum rasio koin/BTC ({p['rs_slow']} candle) berubah positif, short saat berubah negatif"
