"""Strategy families, bar-by-bar simulator (numba), metrics, search loop,
walk-forward consistency, parameter stability and Monte Carlo."""
from __future__ import annotations
import math, time
import numpy as np
import pandas as pd
from numba import njit
from . import indicators as ind

# ------------------------------------------------------------------ families
EXIT_SPACE = {"sl_atr": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "tp_atr": [0.0, 1.5, 2.0, 3.0, 4.0, 6.0],
              "trail_atr": [0.0, 0.0, 2.0, 3.0, 4.0]}

FAMILIES = {
    "ma_cross": {"name": "Trend following (MA cross)",
                 "space": {"ma_type": ["EMA", "SMA"], "fast": (5, 60), "slow": (20, 250), "trend": [0, 0, 100, 200]}},
    "rsi_mr": {"name": "Mean reversion (RSI)",
               "space": {"rsi_len": (5, 30), "lower": (15, 40), "upper": (60, 85), "exit_level": [50, 55, 60, 65],
                         "trend": [0, 0, 100, 200]}},
    "donchian": {"name": "Breakout (Donchian)",
                 "space": {"entry_len": (10, 120), "exit_len": (5, 60), "trend": [0, 0, 100, 200]}},
    "bb_squeeze": {"name": "Volatility compression (Bollinger squeeze)",
                   "space": {"bb_len": (10, 50), "bb_mult": [1.5, 2.0, 2.5], "sq_len": (50, 200),
                             "sq_factor": [0.6, 0.7, 0.8, 0.9]}},
    "macd": {"name": "Momentum (MACD cross)",
             "space": {"fast": (5, 20), "slow": (21, 60), "signal": (5, 15), "trend": [0, 0, 100, 200]}},
    "supertrend": {"name": "Trend following (Supertrend)",
                   "space": {"st_len": (7, 30), "st_mult": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], "trend": [0, 0, 100, 200]}},
    "keltner": {"name": "Breakout (Keltner channel)",
                "space": {"kc_len": (10, 80), "kc_mult": [1.0, 1.5, 2.0, 2.5, 3.0], "trend": [0, 0, 100, 200]}},
    "stochrsi": {"name": "Pullback searah tren (Stochastic RSI)",
                 "space": {"rsi_len": (7, 21), "stoch_len": (7, 21), "k_lo": [10, 15, 20, 25], "k_hi": [75, 80, 85, 90],
                           "trend": [50, 100, 200]}},
    "zscore": {"name": "Mean reversion (Z-score)",
               "space": {"z_len": (20, 120), "z_entry": [1.5, 2.0, 2.5, 3.0], "z_exit": [0.0, 0.0, 0.5], "trend": [0, 0, 100, 200]}},
}
# searchable market-regime filter (applies to every family): 0 = off, 1 = only trade when trending (ADX >= thr),
# 2 = only trade when ranging (ADX < thr)
REGIME_SPACE = {"regime": [0, 0, 1, 2], "adx_thr": [20, 25]}
REGIME_LABEL = {0: "tanpa filter", 1: "hanya saat pasar trending", 2: "hanya saat pasar sideways"}
INT_KEYS = {"fast", "slow", "trend", "rsi_len", "lower", "upper", "exit_level", "entry_len", "exit_len",
            "bb_len", "sq_len", "signal", "st_len", "kc_len", "stoch_len", "k_lo", "k_hi", "z_len", "regime", "adx_thr"}


def sample_params(fam, rng):
    p = {}
    for k, v in FAMILIES[fam]["space"].items():
        p[k] = rng.choice(v) if isinstance(v, list) else int(rng.integers(v[0], v[1] + 1))
        if isinstance(p[k], np.generic):
            p[k] = p[k].item()
    for k, v in REGIME_SPACE.items():
        p[k] = int(rng.choice(v))
    for k, v in EXIT_SPACE.items():
        p[k] = float(rng.choice(v))
    if fam in ("ma_cross", "macd") and p["fast"] >= p["slow"] * 0.8:
        p["slow"] = int(max(p["slow"], p["fast"] * 1.5 + 2))
    if fam == "donchian" and p["exit_len"] > p["entry_len"]:
        p["exit_len"] = max(3, p["entry_len"] // 2)
    return p


class Cache:
    """Per-dataset indicator cache so thousands of variants stay fast."""
    def __init__(self, df):
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.atr = ind.atr(self.h, self.l, self.c, 14)
        self.d = {}
        self.index = df.index
        # last bar of every UTC day -> daily returns for Sharpe / DSR / PBO / portfolio
        day = np.asarray(df.index.normalize().asi8)
        self.day_end = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
        span_years = max((df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
        self.days_per_year = len(self.day_end) / span_years
        self.bars_per_year = len(df) / span_years
        # optional holding costs per bar (fraction of position): funding / swap. 0 = off.
        self.carry_long = np.zeros(len(df))
        self.carry_short = np.zeros(len(df))
        # intrabar resolution: 1 = take-profit was hit first in this bar (from lower-timeframe data)
        self.resolve = np.zeros(len(df), np.int8)

    def get(self, key, fn):
        if key not in self.d:
            self.d[key] = fn()
        return self.d[key]


def _trend(cc, n):
    if not n:
        return None
    return cc.get(("ema", "c", n), lambda: ind.ema(cc.c, n))


def signals(fam, p, cc: Cache):
    """Returns (long_entry, long_exit, short_entry, short_exit, lines) evaluated at bar close."""
    c = cc.c
    lines = {}
    tr = _trend(cc, p.get("trend", 0))
    tl = np.ones_like(c, bool) if tr is None else (c > tr)
    ts = np.ones_like(c, bool) if tr is None else (c < tr)
    if tr is not None:
        lines[f"Trend EMA {p['trend']}"] = tr
    if fam == "ma_cross":
        f = cc.get((p["ma_type"], p["fast"]), lambda: ind.ma(p["ma_type"], c, p["fast"]))
        s = cc.get((p["ma_type"], p["slow"]), lambda: ind.ma(p["ma_type"], c, p["slow"]))
        up, dn = ind.crossover(f, s), ind.crossunder(f, s)
        le, lx, se, sx = up & tl, dn, dn & ts, up
        lines[f"{p['ma_type']} {p['fast']}"] = f
        lines[f"{p['ma_type']} {p['slow']}"] = s
    elif fam == "rsi_mr":
        r = cc.get(("rsi", p["rsi_len"]), lambda: ind.rsi(c, p["rsi_len"]))
        le = ind.crossover(r, p["lower"]) & tl
        lx = ind.crossover(r, p["exit_level"])
        se = ind.crossunder(r, p["upper"]) & ts
        sx = ind.crossunder(r, 100 - p["exit_level"])
        lines["_rsi"] = r
    elif fam == "donchian":
        hh = cc.get(("hh", p["entry_len"]), lambda: ind.shift(ind.rolling_max(cc.h, p["entry_len"])))
        ll = cc.get(("ll", p["entry_len"]), lambda: ind.shift(ind.rolling_min(cc.l, p["entry_len"])))
        xh = cc.get(("hh", p["exit_len"]), lambda: ind.shift(ind.rolling_max(cc.h, p["exit_len"])))
        xl = cc.get(("ll", p["exit_len"]), lambda: ind.shift(ind.rolling_min(cc.l, p["exit_len"])))
        with np.errstate(invalid="ignore"):
            le, lx, se, sx = (c > hh) & tl, c < xl, (c < ll) & ts, c > xh
        lines[f"Donchian high {p['entry_len']}"] = hh
        lines[f"Donchian low {p['entry_len']}"] = ll
    elif fam == "bb_squeeze":
        L, m = p["bb_len"], p["bb_mult"]
        basis = cc.get(("sma", L), lambda: ind.sma(c, L))
        sd = cc.get(("sd", L), lambda: ind.stdev(c, L))
        up, lo = basis + m * sd, basis - m * sd
        with np.errstate(invalid="ignore", divide="ignore"):
            bbw = (up - lo) / basis
        thr = ind.sma(bbw, p["sq_len"]) * p["sq_factor"]
        with np.errstate(invalid="ignore"):
            sq = (bbw < thr).astype(float)
        recent = ind.shift(ind.rolling_max(sq, 5)) == 1
        le = ind.crossover(c, up) & recent
        lx = ind.crossunder(c, basis)
        se = ind.crossunder(c, lo) & recent
        sx = ind.crossover(c, basis)
        lines[f"BB upper {L}/{m}"] = up
        lines[f"BB basis {L}"] = basis
        lines[f"BB lower {L}/{m}"] = lo
    elif fam == "macd":
        ef = cc.get(("ema", "c", p["fast"]), lambda: ind.ema(c, p["fast"]))
        es = cc.get(("ema", "c", p["slow"]), lambda: ind.ema(c, p["slow"]))
        mline = ef - es
        sig = ind.ema(mline, p["signal"])
        up, dn = ind.crossover(mline, sig), ind.crossunder(mline, sig)
        le, lx, se, sx = up & tl, dn, dn & ts, up
        lines["_macd"] = mline
        lines["_macd_signal"] = sig
    elif fam == "supertrend":
        a = cc.get(("atr", p["st_len"]), lambda: ind.atr(cc.h, cc.l, c, p["st_len"]))
        up_b, dn_b, d = cc.get(("st", p["st_len"], p["st_mult"]), lambda: ind.supertrend(cc.h, cc.l, c, a, p["st_mult"]))
        d1 = ind.shift(d)
        flip_up, flip_dn = (d == 1) & (d1 == -1), (d == -1) & (d1 == 1)
        le, lx, se, sx = flip_up & tl, flip_dn, flip_dn & ts, flip_up
        lines["Supertrend"] = np.where(d == 1, up_b, np.where(d == -1, dn_b, np.nan))
    elif fam == "keltner":
        basis = cc.get(("ema", "c", p["kc_len"]), lambda: ind.ema(c, p["kc_len"]))
        up_b, lo_b = basis + p["kc_mult"] * cc.atr, basis - p["kc_mult"] * cc.atr
        le = ind.crossover(c, up_b) & tl
        lx = ind.crossunder(c, basis)
        se = ind.crossunder(c, lo_b) & ts
        sx = ind.crossover(c, basis)
        lines[f"KC upper {p['kc_len']}/{p['kc_mult']}"] = up_b
        lines[f"KC basis {p['kc_len']}"] = basis
        lines[f"KC lower {p['kc_len']}/{p['kc_mult']}"] = lo_b
    elif fam == "stochrsi":
        k = cc.get(("srsi", p["rsi_len"], p["stoch_len"]), lambda: ind.stoch_rsi_k(c, p["rsi_len"], p["stoch_len"]))
        le = ind.crossover(k, p["k_lo"]) & tl
        lx = ind.crossunder(k, p["k_hi"])
        se = ind.crossunder(k, p["k_hi"]) & ts
        sx = ind.crossover(k, p["k_lo"])
        lines["_stoch"] = k
    elif fam == "zscore":
        z = cc.get(("z", p["z_len"]), lambda: ind.zscore(c, p["z_len"]))
        le = ind.crossover(z, -p["z_entry"]) & tl
        lx = ind.crossover(z, p["z_exit"])
        se = ind.crossunder(z, p["z_entry"]) & ts
        sx = ind.crossunder(z, -p["z_exit"])
        lines["_z"] = z
    else:
        raise ValueError(fam)
    rg = p.get("regime", 0)
    if rg:
        a = cc.get(("adx",), lambda: ind.adx(cc.h, cc.l, c, cc.atr))
        with np.errstate(invalid="ignore"):
            ok = (a >= p["adx_thr"]) if rg == 1 else (a < p["adx_thr"])
        le, se = le & ok, se & ok
        lines["_adx"] = a
    f = lambda a: np.nan_to_num(a, nan=0).astype(np.bool_)
    return f(le), f(lx), f(se), f(sx), lines


def describe(fam, p, direction="both"):
    d = {"long": "hanya Long (beli)", "short": "hanya Short (jual)", "both": "Long dan Short"}[direction]
    tr = f", hanya searah tren (harga di atas/bawah EMA {p['trend']})" if p.get("trend") else ""
    if fam == "ma_cross":
        core = (f"Beli saat {p['ma_type']} {p['fast']} memotong ke atas {p['ma_type']} {p['slow']}{tr}; "
                f"jual/short saat memotong ke bawah")
    elif fam == "rsi_mr":
        core = (f"Beli saat RSI({p['rsi_len']}) naik kembali di atas {p['lower']} (jenuh jual){tr}, keluar saat RSI > {p['exit_level']}; "
                f"short saat RSI turun di bawah {p['upper']}, keluar saat RSI < {100 - p['exit_level']}")
    elif fam == "donchian":
        core = (f"Beli saat harga menembus high {p['entry_len']} candle{tr}, keluar saat tembus low {p['exit_len']} candle; "
                f"short kebalikannya")
    elif fam == "bb_squeeze":
        core = (f"Tunggu Bollinger({p['bb_len']}, {p['bb_mult']}) menyempit (lebar < {p['sq_factor']} x rata-rata {p['sq_len']} candle), "
                f"lalu beli saat harga tembus band atas / short saat tembus band bawah; keluar di garis tengah")
    elif fam == "macd":
        core = (f"Beli saat MACD({p['fast']},{p['slow']},{p['signal']}) memotong ke atas signal{tr}; "
                f"short saat memotong ke bawah")
    elif fam == "supertrend":
        core = f"Beli saat Supertrend({p['st_len']}, {p['st_mult']}) berbalik naik{tr}; short saat berbalik turun"
    elif fam == "keltner":
        core = (f"Beli saat harga tembus band atas Keltner (EMA {p['kc_len']} + {p['kc_mult']} x ATR){tr}, keluar saat "
                f"kembali di bawah EMA; short kebalikannya")
    elif fam == "stochrsi":
        core = (f"Searah tren EMA {p['trend']}: beli saat Stoch RSI({p['rsi_len']},{p['stoch_len']}) naik di atas {p['k_lo']} "
                f"(selesai koreksi), keluar saat turun di bawah {p['k_hi']}; short kebalikannya")
    else:
        core = (f"Beli saat Z-score({p['z_len']}) naik kembali di atas -{p['z_entry']}{tr}, keluar saat Z > {p['z_exit']}; "
                f"short saat turun kembali di bawah {p['z_entry']}, keluar saat Z < -{p['z_exit']}")
    if p.get("regime"):
        core += f"; {REGIME_LABEL[p['regime']]} (ADX {'≥' if p['regime'] == 1 else '<'} {p['adx_thr']})"
    ex = f". Stop loss {p['sl_atr']} x ATR"
    ex += f", take profit {p['tp_atr']} x ATR" if p["tp_atr"] > 0 else ", tanpa take profit tetap"
    ex += f", trailing stop {p['trail_atr']} x ATR" if p["trail_atr"] > 0 else ""
    return f"{core}{ex}. Arah: {d}."


# ------------------------------------------------------------------ simulator
@njit(cache=True)
def simulate(o, h, l, c, atr, le, lx, se, sx, start, end, allow_long, allow_short,
             sl_k, tp_k, tr_k, cost_frac, cost_abs, want_equity, lev, carry_l, carry_s, resolve):
    """Signals are evaluated at the close of bar j and filled at the OPEN of bar j+1
    (same as Pine strategy() default). Stops/targets use high/low of each bar; if both
    are touched in one bar, the stop is assumed first (conservative) unless resolve[i] == 1
    (lower-timeframe data showed the target was reached first).
    carry_l / carry_s: holding cost per bar as a fraction of the position (funding / swap),
    charged at each bar close while the position is open (negative = you receive it)."""
    n = end - start
    eq_arr = np.empty(n if want_equity else 1)
    tr_ent = np.empty(n, np.int64); tr_ex = np.empty(n, np.int64); tr_dir = np.empty(n, np.int64)
    tr_ep = np.empty(n); tr_xp = np.empty(n); tr_ret = np.empty(n); tr_reason = np.empty(n, np.int64)
    tr_amb = np.zeros(n, np.int64); tr_sl = np.empty(n); tr_tp = np.empty(n); tr_carry = np.zeros(n)
    nt = 0
    pos = 0; ep = 0.0; ei = 0; stop = np.nan; tp = np.nan; acc = 0.0
    eq = 1.0; peak = 1.0; maxdd = 0.0
    if want_equity:
        eq_arr[0] = 1.0
    for i in range(start + 1, end):
        j = i - 1
        # ---- 1) orders from the previous bar's close, filled at this bar's open
        new_dir = 0
        exit_now = False
        if pos == 1:
            if allow_short and se[j]:
                exit_now = True; new_dir = -1
            elif lx[j]:
                exit_now = True
        elif pos == -1:
            if allow_long and le[j]:
                exit_now = True; new_dir = 1
            elif sx[j]:
                exit_now = True
        else:
            if allow_long and le[j]:
                new_dir = 1
            elif allow_short and se[j]:
                new_dir = -1
        if exit_now:
            px = o[i]
            if pos == 1:
                r = (px * (1 - cost_frac) - cost_abs - (ep * (1 + cost_frac) + cost_abs)) / ep
            else:
                r = ((ep * (1 - cost_frac) - cost_abs) - (px * (1 + cost_frac) + cost_abs)) / ep
            r = (r - acc) * lev
            eq *= (1 + r)
            tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = 0
            tr_sl[nt] = stop; tr_tp[nt] = tp; tr_carry[nt] = acc
            nt += 1
            pos = 0
        if new_dir != 0 and not np.isnan(atr[j]):
            pos = new_dir; ep = o[i]; ei = i; acc = 0.0
            stop = c[j] - pos * sl_k * atr[j]
            tp = c[j] + pos * tp_k * atr[j] if tp_k > 0 else np.nan
        # ---- 2) intrabar stop / target
        if pos != 0:
            hit = False; px = 0.0; reason = 0; amb = 0
            if pos == 1:
                hs = l[i] <= stop
                ht = tp_k > 0 and h[i] >= tp
                if hs and ht and o[i] > stop:
                    amb = 1
                if hs and not (amb == 1 and resolve[i] == 1):
                    hit = True; px = min(o[i], stop); reason = 1
                elif ht:
                    hit = True; px = max(o[i], tp); reason = 2
            else:
                hs = h[i] >= stop
                ht = tp_k > 0 and l[i] <= tp
                if hs and ht and o[i] < stop:
                    amb = 1
                if hs and not (amb == 1 and resolve[i] == 1):
                    hit = True; px = max(o[i], stop); reason = 1
                elif ht:
                    hit = True; px = min(o[i], tp); reason = 2
            if hit:
                if pos == 1:
                    r = (px * (1 - cost_frac) - cost_abs - (ep * (1 + cost_frac) + cost_abs)) / ep
                else:
                    r = ((ep * (1 - cost_frac) - cost_abs) - (px * (1 + cost_frac) + cost_abs)) / ep
                r = (r - acc) * lev
                eq *= (1 + r)
                tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = reason
                tr_amb[nt] = amb; tr_sl[nt] = stop; tr_tp[nt] = tp; tr_carry[nt] = acc
                nt += 1
                pos = 0
        # ---- 3) bar close: carry, mark-to-market equity, drawdown, trailing update
        if pos == 1:
            acc += carry_l[i]
            m = eq * (1 + lev * ((c[i] * (1 - cost_frac) - cost_abs - (ep * (1 + cost_frac) + cost_abs)) / ep - acc))
            if tr_k > 0 and not np.isnan(atr[i]):
                stop = max(stop, h[i] - tr_k * atr[i])
        elif pos == -1:
            acc += carry_s[i]
            m = eq * (1 + lev * (((ep * (1 - cost_frac) - cost_abs) - (c[i] * (1 + cost_frac) + cost_abs)) / ep - acc))
            if tr_k > 0 and not np.isnan(atr[i]):
                stop = min(stop, l[i] + tr_k * atr[i])
        else:
            m = eq
        if m > peak:
            peak = m
        dd = 1 - m / peak
        if dd > maxdd:
            maxdd = dd
        if want_equity:
            eq_arr[i - start] = m
    if pos != 0:  # close open trade at last close
        i = end - 1; px = c[i]
        if pos == 1:
            r = (px * (1 - cost_frac) - cost_abs - (ep * (1 + cost_frac) + cost_abs)) / ep
        else:
            r = ((ep * (1 - cost_frac) - cost_abs) - (px * (1 + cost_frac) + cost_abs)) / ep
        r = (r - acc) * lev
        eq *= (1 + r)
        tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = 3
        tr_sl[nt] = stop; tr_tp[nt] = tp; tr_carry[nt] = acc
        nt += 1
    return (eq, maxdd, nt, tr_ent[:nt], tr_ex[:nt], tr_dir[:nt], tr_ep[:nt], tr_xp[:nt], tr_ret[:nt], tr_reason[:nt], eq_arr,
            tr_amb[:nt], tr_sl[:nt], tr_tp[:nt], tr_carry[:nt])


def metrics(rets, eq, maxdd):
    n = len(rets)
    if n == 0:
        return dict(trades=0, net=0.0, winrate=0.0, pf=0.0, maxdd=maxdd, avg=0.0, ret_dd=0.0)
    g = rets[rets > 0].sum(); ls = -rets[rets < 0].sum()
    pf = g / ls if ls > 0 else (99.0 if g > 0 else 0.0)
    net = eq - 1
    return dict(trades=int(n), net=float(net), winrate=float((rets > 0).mean()), pf=float(min(pf, 99.0)),
                maxdd=float(maxdd), avg=float(rets.mean()), ret_dd=float(net / max(maxdd, 1e-3)))


def daily_returns(cc, start, end, equity):
    """Daily returns from a bar equity curve that starts at bar `start` (value 1.0)."""
    de = cc.day_end[(cc.day_end > start) & (cc.day_end < end)]
    if len(de) == 0:
        return np.zeros(0)
    v = equity[de - start]
    return np.diff(np.r_[1.0, v]) / np.r_[1.0, v[:-1]]


def sharpe(dr):
    """Non-annualised Sharpe of a return series (0 when flat)."""
    if len(dr) < 2:
        return 0.0
    sd = dr.std(ddof=1)
    return float(dr.mean() / sd) if sd > 1e-12 else 0.0


def run(fam, p, cc, start, end, direction, cost_frac, cost_abs, want_equity=False, sig=None, lev=1.0, daily=False):
    le, lx, se, sx, _ = sig if sig is not None else signals(fam, p, cc)
    res = simulate(cc.o, cc.h, cc.l, cc.c, cc.atr, le, lx, se, sx, int(start), int(end),
                   direction in ("long", "both"), direction in ("short", "both"),
                   float(p["sl_atr"]), float(p["tp_atr"]), float(p["trail_atr"]), float(cost_frac), float(cost_abs),
                   want_equity or daily, float(lev), cc.carry_long, cc.carry_short, cc.resolve)
    eq, maxdd, nt, ent, ex, dr, ep, xp, ret, reason, eqa, amb, sl, tp, carry = res
    m = metrics(ret, eq, maxdd)
    tr = dict(ent=ent, ex=ex, dir=dr, ep=ep, xp=xp, ret=ret, reason=reason, equity=eqa, amb=amb, sl=sl, tp=tp, carry=carry)
    if daily:
        d = daily_returns(cc, start, end, eqa)
        tr["daily"] = d
        m["sharpe_d"] = sharpe(d)
        m["sharpe"] = m["sharpe_d"] * math.sqrt(cc.days_per_year)
    return m, tr


# ------------------------------------------------------------------ robustness
def wf_consistency(tr, start, end, k):
    """Split the in-sample period into k equal parts; share of parts where the strategy made money."""
    if k < 2:
        return 1.0, []
    edges = np.linspace(start, end, k + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        msk = (tr["ex"] >= a) & (tr["ex"] < b)
        r = tr["ret"][msk]
        out.append(float(np.prod(1 + r) - 1) if len(r) else 0.0)
    return float(np.mean([x > 0 for x in out])), out


def neighbours(fam, p):
    out = []
    for k, v in FAMILIES[fam]["space"].items():
        if isinstance(v, tuple):
            for f in (0.85, 1.15):
                q = dict(p); nv = int(round(p[k] * f))
                if nv == p[k]:
                    nv = p[k] + (1 if f > 1 else -1)
                q[k] = max(2, nv)
                out.append(q)
    for k in ("sl_atr",):
        for d in (-0.5, 0.5):
            q = dict(p); q[k] = max(0.5, p[k] + d); out.append(q)
    if p["tp_atr"] > 0:
        for d in (-0.5, 0.5):
            q = dict(p); q["tp_atr"] = max(0.5, p["tp_atr"] + d); out.append(q)
    return out


def stability(fam, p, cc, start, end, direction, cf, ca, lev=1.0):
    pfs = []
    for q in neighbours(fam, p):
        if fam in ("ma_cross", "macd") and q["fast"] >= q["slow"]:
            continue
        m, _ = run(fam, q, cc, start, end, direction, cf, ca, lev=lev)
        pfs.append(m["pf"] if m["trades"] >= 5 else 0)
    pfs = np.array(pfs)
    return (float((pfs > 1.0).mean()) if len(pfs) else 0.0), (float(np.median(pfs)) if len(pfs) else 0.0)


def monte_carlo(rets, runs=1000, method="bootstrap", seed=7, skip=0.1):
    """bootstrap = resample trades with replacement; shuffle = random order;
    skip = randomly miss `skip` share of trades (missed signals / downtime), then random order."""
    rng = np.random.default_rng(seed)
    rets = np.asarray(rets, float)
    n = len(rets)
    if n < 2:
        return None
    if method == "bootstrap":
        idx = rng.integers(0, n, size=(runs, n))
        samp = rets[idx]
    elif method == "skip":
        samp = np.array([rng.permutation(rets) for _ in range(runs)])
        samp[rng.random((runs, n)) < skip] = 0.0  # a skipped trade = flat
    else:
        samp = np.array([rng.permutation(rets) for _ in range(runs)])
    paths = np.cumprod(1 + samp, axis=1)
    paths = np.hstack([np.ones((runs, 1)), paths])
    peaks = np.maximum.accumulate(paths, axis=1)
    dds = (1 - paths / peaks).max(axis=1)
    finals = paths[:, -1] - 1
    bands = {q: np.percentile(paths, q, axis=0) for q in (5, 25, 50, 75, 95)}
    return dict(paths=paths, dds=dds, finals=finals, bands=bands)


# ------------------------------------------------------------------ search
def search(df, cfg, progress=None, cc=None):
    """cfg: families, n_variants, time_budget, direction, oos_pct, folds, cost_frac, cost_abs,
    min_pf, min_wr, max_dd, min_trades, min_wf, min_stab, oos_min_pf, seed"""
    t0 = time.time()
    cc = cc if cc is not None else Cache(df)
    n = len(df)
    split = int(n * (1 - cfg["oos_pct"] / 100.0))
    warm = 260  # bars of indicator warm-up
    is_start = min(warm, split // 4)
    rng = np.random.default_rng(cfg.get("seed", 42))
    fams = cfg["families"]
    rows = []
    seen = set()
    near = []
    tested = 0
    trial_sr = []            # daily Sharpe of EVERY trial (for the Deflated Sharpe Ratio)
    pbo_keep = cfg.get("pbo_trials", 800)
    trial_daily = []         # daily returns of the first `pbo_keep` trials (random sample, for PBO / CSCV)
    for k in range(cfg["n_variants"]):
        if time.time() - t0 > cfg["time_budget"]:
            break
        fam = fams[k % len(fams)]
        p = sample_params(fam, rng)
        key = (fam, tuple(sorted(p.items())))
        if key in seen:
            continue
        seen.add(key)
        m, tr = run(fam, p, cc, is_start, split, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0),
                    daily=True)
        tested += 1
        trial_sr.append(m["sharpe_d"])
        if len(trial_daily) < pbo_keep:
            trial_daily.append(tr["daily"].astype(np.float32))
        ok = (m["pf"] >= cfg["min_pf"] and m["winrate"] >= cfg["min_wr"] and m["maxdd"] <= cfg["max_dd"]
              and m["trades"] >= cfg["min_trades"])
        wf, _ = wf_consistency(tr, is_start, split, cfg["folds"])
        if ok:
            rows.append(dict(fam=fam, params=p, is_m=m, wf=wf, is_pass=True))
        elif m["trades"] >= cfg["min_trades"] and m["net"] > 0:
            near.append(dict(fam=fam, params=p, is_m=m, wf=wf, is_pass=False))
        if progress and k % 25 == 0:
            progress(k / cfg["n_variants"], tested, len(rows))
    t_search = time.time() - t0
    # ---- stage 2: robustness on the best in-sample candidates (ranked by IS return/DD only)
    rows.sort(key=lambda r: r["is_m"]["ret_dd"], reverse=True)
    n_is_pass = len(rows)
    rows = rows[: cfg.get("max_deep", 150)]
    near.sort(key=lambda r: r["is_m"]["ret_dd"], reverse=True)
    rows += near[: max(0, 20 - len(rows))]  # show 'near misses' (clearly flagged) when few pass
    out = []
    for r in rows:
        stab, stab_pf = stability(r["fam"], r["params"], cc, is_start, split, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        om, _ = run(r["fam"], r["params"], cc, split, n, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        ism = r["is_m"]
        oos_ok = om["pf"] >= cfg["oos_min_pf"] and om["maxdd"] <= cfg["max_dd"] and om["trades"] >= max(3, cfg["min_trades"] * cfg["oos_pct"] / max(1, 100 - cfg["oos_pct"]) * 0.3)
        overfit = (om["trades"] > 0 and om["pf"] < 0.6 * ism["pf"]) or om["net"] < 0
        passed = r["is_pass"] and oos_ok and r["wf"] >= cfg["min_wf"] and stab >= cfg["min_stab"] and not overfit
        out.append(dict(fam=r["fam"], params=r["params"], is_m=ism, oos_m=om, wf=r["wf"], stab=stab,
                        stab_pf=stab_pf, overfit=overfit, passed=passed, is_pass=r["is_pass"], oos_ok=oos_ok))
    meta = dict(tested=tested, is_pass=n_is_pass, seconds=time.time() - t0, search_seconds=t_search,
                split=split, is_start=is_start, n=n, split_time=df.index[split], start_time=df.index[is_start],
                end_time=df.index[-1], trial_sr=np.array(trial_sr),
                trial_daily=np.vstack(trial_daily) if trial_daily else np.zeros((0, 0), np.float32))
    return out, meta, cc
