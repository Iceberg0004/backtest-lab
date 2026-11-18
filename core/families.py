"""Strategy families. Each family has: a search space, a Python signal function and a Pine block
that compute EXACTLY the same raw signals (checked by tools/pinecheck with PineTS).
Raw signals = (longEntry, longExit, shortEntry, shortExit) before any filter; filters, exits and the
trend EMA are added generically in engine.py / pine.py."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from . import indicators as ind

R_ = repr  # exact float literal for Pine


def _bool(a):
    return np.nan_to_num(a, nan=0).astype(np.bool_)


def _cx(a, b):
    return ind.crossover(a, b)


def _cu(a, b):
    return ind.crossunder(a, b)


F = {}


def fam(key, name, space, intraday=False, needs_volume=False, sub=None, needs=None, fixed=None):
    def deco(cls):
        F[key] = dict(name=name, space=space, py=cls.py, pine=cls.pine, desc=cls.desc, intraday=intraday,
                      needs_volume=needs_volume, sub=sub, needs=set(needs or ()), fixed=dict(fixed or {}))
        return cls
    return deco


T4 = [0, 0, 100, 200]

# ------------------------------------------------------------------ original 9 (unchanged logic)
@fam("ma_cross", "Trend following (MA cross)", {"ma_type": ["EMA", "SMA"], "fast": (5, 60), "slow": (20, 250), "trend": T4})
class _:
    def py(p, cc):
        f = cc.get((p["ma_type"], p["fast"]), lambda: ind.ma(p["ma_type"], cc.c, p["fast"]))
        s = cc.get((p["ma_type"], p["slow"]), lambda: ind.ma(p["ma_type"], cc.c, p["slow"]))
        up, dn = _cx(f, s), _cu(f, s)
        return up, dn, dn, up, {f"{p['ma_type']} {p['fast']}": f, f"{p['ma_type']} {p['slow']}": s}

    def pine(p):
        f = "ta.ema" if p["ma_type"] == "EMA" else "ta.sma"
        return [f"fastMa = {f}(close, {p['fast']})", f"slowMa = {f}(close, {p['slow']})",
                "up = ta.crossover(fastMa, slowMa)", "dn = ta.crossunder(fastMa, slowMa)",
                "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up",
                f"plot(fastMa, '{p['ma_type']} {p['fast']}', color=color.orange)", f"plot(slowMa, '{p['ma_type']} {p['slow']}', color=color.blue)"]

    def desc(p):
        return f"Beli saat {p['ma_type']} {p['fast']} memotong ke atas {p['ma_type']} {p['slow']}; jual/short saat memotong ke bawah"


@fam("rsi_mr", "Mean reversion (RSI)", {"rsi_len": (5, 30), "lower": (15, 40), "upper": (60, 85), "exit_level": [50, 55, 60, 65], "trend": T4}, sub="_rsi")
class _:
    def py(p, cc):
        r = cc.get(("rsi", p["rsi_len"]), lambda: ind.rsi(cc.c, p["rsi_len"]))
        return _cx(r, p["lower"]), _cx(r, p["exit_level"]), _cu(r, p["upper"]), _cu(r, 100 - p["exit_level"]), {"_rsi": r}

    def pine(p):
        return [f"r = ta.rsi(close, {p['rsi_len']})", f"longEntry = ta.crossover(r, {p['lower']})", f"longExit = ta.crossover(r, {p['exit_level']})",
                f"shortEntry = ta.crossunder(r, {p['upper']})", f"shortExit = ta.crossunder(r, {100 - p['exit_level']})"]

    def desc(p):
        return (f"Beli saat RSI({p['rsi_len']}) naik kembali di atas {p['lower']}, keluar saat RSI > {p['exit_level']}; "
                f"short saat RSI turun di bawah {p['upper']}, keluar saat RSI < {100 - p['exit_level']}")


@fam("donchian", "Breakout (Donchian)", {"entry_len": (10, 120), "exit_len": (5, 60), "trend": T4})
class _:
    def py(p, cc):
        hh = cc.get(("hh", p["entry_len"]), lambda: ind.shift(ind.rolling_max(cc.h, p["entry_len"])))
        ll = cc.get(("ll", p["entry_len"]), lambda: ind.shift(ind.rolling_min(cc.l, p["entry_len"])))
        xh = cc.get(("hh", p["exit_len"]), lambda: ind.shift(ind.rolling_max(cc.h, p["exit_len"])))
        xl = cc.get(("ll", p["exit_len"]), lambda: ind.shift(ind.rolling_min(cc.l, p["exit_len"])))
        c = cc.c
        with np.errstate(invalid="ignore"):
            return c > hh, c < xl, c < ll, c > xh, {f"Donchian high {p['entry_len']}": hh, f"Donchian low {p['entry_len']}": ll}

    def pine(p):
        return [f"hh = ta.highest(high, {p['entry_len']})[1]", f"ll = ta.lowest(low, {p['entry_len']})[1]",
                f"xh = ta.highest(high, {p['exit_len']})[1]", f"xl = ta.lowest(low, {p['exit_len']})[1]",
                "longEntry = close > hh", "longExit = close < xl", "shortEntry = close < ll", "shortExit = close > xh",
                "plot(hh, 'Donchian high', color=color.teal)", "plot(ll, 'Donchian low', color=color.teal)"]

    def desc(p):
        return f"Beli saat harga menembus high {p['entry_len']} candle, keluar saat tembus low {p['exit_len']} candle; short kebalikannya"


@fam("bb_squeeze", "Volatility compression (Bollinger squeeze)", {"bb_len": (10, 50), "bb_mult": [1.5, 2.0, 2.5], "sq_len": (50, 200), "sq_factor": [0.6, 0.7, 0.8, 0.9]})
class _:
    def py(p, cc):
        L, m, c = p["bb_len"], p["bb_mult"], cc.c
        basis = cc.get(("sma", L), lambda: ind.sma(c, L))
        sd = cc.get(("sd", L), lambda: ind.stdev(c, L))
        up, lo = basis + m * sd, basis - m * sd
        with np.errstate(invalid="ignore", divide="ignore"):
            bbw = (up - lo) / basis
        thr = ind.sma(bbw, p["sq_len"]) * p["sq_factor"]
        with np.errstate(invalid="ignore"):
            sq = (bbw < thr).astype(float)
        recent = ind.shift(ind.rolling_max(sq, 5)) == 1
        return (_cx(c, up) & recent, _cu(c, basis), _cu(c, lo) & recent, _cx(c, basis),
                {f"BB upper {L}/{m}": up, f"BB basis {L}": basis, f"BB lower {L}/{m}": lo})

    def pine(p):
        return [f"basis = ta.sma(close, {p['bb_len']})", f"dev = {p['bb_mult']} * ta.stdev(close, {p['bb_len']})",
                "upper = basis + dev", "lower = basis - dev", "bbw = (upper - lower) / basis",
                f"thr = ta.sma(bbw, {p['sq_len']}) * {p['sq_factor']}", "sq = bbw < thr ? 1.0 : 0.0",
                "recent = nz(ta.highest(sq, 5)[1]) == 1",
                "bbXU = ta.crossover(close, upper)", "bbXD = ta.crossunder(close, lower)",
                "longEntry = bbXU and recent", "longExit = ta.crossunder(close, basis)",
                "shortEntry = bbXD and recent", "shortExit = ta.crossover(close, basis)",
                "plot(upper, 'BB upper', color=color.purple)", "plot(basis, 'BB basis', color=color.gray)", "plot(lower, 'BB lower', color=color.purple)"]

    def desc(p):
        return (f"Tunggu Bollinger({p['bb_len']}, {p['bb_mult']}) menyempit (lebar < {p['sq_factor']} x rata-rata {p['sq_len']} candle), "
                f"lalu beli saat harga tembus band atas / short saat tembus band bawah; keluar di garis tengah")


@fam("macd", "Momentum (MACD cross)", {"fast": (5, 20), "slow": (21, 60), "signal": (5, 15), "trend": T4}, sub="_macd")
class _:
    def py(p, cc):
        ef = cc.get(("ema", "c", p["fast"]), lambda: ind.ema(cc.c, p["fast"]))
        es = cc.get(("ema", "c", p["slow"]), lambda: ind.ema(cc.c, p["slow"]))
        ml = ef - es
        sg = ind.ema(ml, p["signal"])
        up, dn = _cx(ml, sg), _cu(ml, sg)
        return up, dn, dn, up, {"_macd": ml, "_macd_signal": sg}

    def pine(p):
        return [f"macdLine = ta.ema(close, {p['fast']}) - ta.ema(close, {p['slow']})", f"sigLine = ta.ema(macdLine, {p['signal']})",
                "up = ta.crossover(macdLine, sigLine)", "dn = ta.crossunder(macdLine, sigLine)",
                "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up"]

    def desc(p):
        return f"Beli saat MACD({p['fast']},{p['slow']},{p['signal']}) memotong ke atas signal; short saat memotong ke bawah"


@fam("supertrend", "Trend following (Supertrend)", {"st_len": (7, 30), "st_mult": [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], "trend": T4})
class _:
    def py(p, cc):
        a = cc.get(("atr", p["st_len"]), lambda: ind.atr(cc.h, cc.l, cc.c, p["st_len"]))
        ub, db, d = cc.get(("st", p["st_len"], p["st_mult"]), lambda: ind.supertrend(cc.h, cc.l, cc.c, a, p["st_mult"]))
        d1 = ind.shift(d)
        fu, fd = (d == 1) & (d1 == -1), (d == -1) & (d1 == 1)
        return fu, fd, fd, fu, {"Supertrend": np.where(d == 1, ub, np.where(d == -1, db, np.nan))}

    def pine(p):
        return [f"stAtr = ta.atr({p['st_len']})", f"stBu = hl2 - {p['st_mult']} * stAtr", f"stBd = hl2 + {p['st_mult']} * stAtr",
                "var float stUp = na", "var float stDn = na", "var int stDir = 0",
                "if not na(stAtr)",
                "    stUp := na(stUp) or close[1] <= stUp ? stBu : math.max(stBu, stUp)",
                "    stDn := na(stDn) or close[1] >= stDn ? stBd : math.min(stBd, stDn)",
                "    stDir := stDir == 0 ? 1 : stDir == 1 and close < stUp ? -1 : stDir == -1 and close > stDn ? 1 : stDir",
                "flipUp = stDir == 1 and stDir[1] == -1", "flipDn = stDir == -1 and stDir[1] == 1",
                "longEntry = flipUp", "longExit = flipDn", "shortEntry = flipDn", "shortExit = flipUp",
                "plot(stDir == 1 ? stUp : stDir == -1 ? stDn : na, 'Supertrend', color=stDir == 1 ? color.green : color.red, style=plot.style_linebr)"]

    def desc(p):
        return f"Beli saat Supertrend({p['st_len']}, {p['st_mult']}) berbalik naik; short saat berbalik turun"


@fam("keltner", "Breakout (Keltner channel)", {"kc_len": (10, 80), "kc_mult": [1.0, 1.5, 2.0, 2.5, 3.0], "trend": T4})
class _:
    def py(p, cc):
        b = cc.get(("ema", "c", p["kc_len"]), lambda: ind.ema(cc.c, p["kc_len"]))
        u, lo = b + p["kc_mult"] * cc.atr, b - p["kc_mult"] * cc.atr
        return _cx(cc.c, u), _cu(cc.c, b), _cu(cc.c, lo), _cx(cc.c, b), {"KC upper": u, "KC basis": b, "KC lower": lo}

    def pine(p):
        return [f"kcBasis = ta.ema(close, {p['kc_len']})", f"kcUp = kcBasis + {p['kc_mult']} * atr", f"kcLo = kcBasis - {p['kc_mult']} * atr",
                "longEntry = ta.crossover(close, kcUp)", "longExit = ta.crossunder(close, kcBasis)",
                "shortEntry = ta.crossunder(close, kcLo)", "shortExit = ta.crossover(close, kcBasis)",
                "plot(kcUp, 'KC upper', color=color.teal)", "plot(kcBasis, 'KC basis', color=color.gray)", "plot(kcLo, 'KC lower', color=color.teal)"]

    def desc(p):
        return f"Beli saat harga tembus band atas Keltner (EMA {p['kc_len']} + {p['kc_mult']} x ATR), keluar saat kembali di bawah EMA; short kebalikannya"


@fam("stochrsi", "Pullback searah tren (Stochastic RSI)", {"rsi_len": (7, 21), "stoch_len": (7, 21), "k_lo": [10, 15, 20, 25], "k_hi": [75, 80, 85, 90], "trend": [50, 100, 200]}, sub="_stoch")
class _:
    def py(p, cc):
        k = cc.get(("srsi", p["rsi_len"], p["stoch_len"]), lambda: ind.stoch_rsi_k(cc.c, p["rsi_len"], p["stoch_len"]))
        return _cx(k, p["k_lo"]), _cu(k, p["k_hi"]), _cu(k, p["k_hi"]), _cx(k, p["k_lo"]), {"_stoch": k}

    def pine(p):
        w = p["rsi_len"] + p["stoch_len"] + 3 + 1
        return [f"srR = ta.rsi(close, {p['rsi_len']})", f"srH = ta.highest(srR, {p['stoch_len']})", f"srL = ta.lowest(srR, {p['stoch_len']})",
                "srRng = srH - srL", "srSt = srRng == 0 ? na : 100 * (srR - srL) / srRng", "srK0 = ta.sma(srSt, 3)",
                f"srK = bar_index < {w} ? na : srK0",
                f"longEntry = ta.crossover(srK, {p['k_lo']})", f"longExit = ta.crossunder(srK, {p['k_hi']})",
                f"shortEntry = ta.crossunder(srK, {p['k_hi']})", f"shortExit = ta.crossover(srK, {p['k_lo']})"]

    def desc(p):
        return f"Searah tren: beli saat Stoch RSI({p['rsi_len']},{p['stoch_len']}) naik di atas {p['k_lo']}, keluar saat turun di bawah {p['k_hi']}; short kebalikannya"


@fam("zscore", "Mean reversion (Z-score)", {"z_len": (20, 120), "z_entry": [1.5, 2.0, 2.5, 3.0], "z_exit": [0.0, 0.0, 0.5], "trend": T4}, sub="_z")
class _:
    def py(p, cc):
        z = cc.get(("z", p["z_len"]), lambda: ind.zscore(cc.c, p["z_len"]))
        return _cx(z, -p["z_entry"]), _cx(z, p["z_exit"]), _cu(z, p["z_entry"]), _cu(z, -p["z_exit"]), {"_z": z}

    def pine(p):
        return [f"zM = ta.sma(close, {p['z_len']})", f"zSd = ta.stdev(close, {p['z_len']})",
                f"zV = bar_index < {p['z_len']} or zSd == 0 ? na : (close - zM) / zSd",
                f"longEntry = ta.crossover(zV, -{p['z_entry']})", f"longExit = ta.crossover(zV, {p['z_exit']})",
                f"shortEntry = ta.crossunder(zV, {p['z_entry']})", f"shortExit = ta.crossunder(zV, -{p['z_exit']})",
                f"plot(zM, 'Z-score mean {p['z_len']}', color=color.gray)"]

    def desc(p):
        return f"Beli saat Z-score({p['z_len']}) naik kembali di atas -{p['z_entry']}, keluar saat Z > {p['z_exit']}; short kebalikannya"


# ------------------------------------------------------------------ v3 families
K1, K2 = 2.0 / 3 - 2.0 / 31, 2.0 / 31


def _ma2_py(kind, c, n, cc):
    if kind == "HMA":
        return cc.get(("hma", n), lambda: ind.wma(2 * ind.wma(c, n // 2) - ind.wma(c, n), int(math.floor(math.sqrt(n)))))
    if kind == "TEMA":
        def f():
            e1 = ind.ema(c, n); e2 = ind.ema(e1, n); e3 = ind.ema(e2, n)
            return 3 * e1 - 3 * e2 + e3
        return cc.get(("tema", n), f)
    return cc.get(("kama", n), lambda: ind.kama(c, n, K1, K2))


def _ma2_pine(kind, n, v):
    if kind == "HMA":
        return [f"{v} = ta.wma(2 * ta.wma(close, {n // 2}) - ta.wma(close, {n}), {int(math.floor(math.sqrt(n)))})"]
    if kind == "TEMA":
        return [f"{v}e1 = ta.ema(close, {n})", f"{v}e2 = ta.ema({v}e1, {n})", f"{v}e3 = ta.ema({v}e2, {n})",
                f"{v} = 3 * {v}e1 - 3 * {v}e2 + {v}e3"]
    return [f"var float {v} = na",
            f"if na({v})", f"    {v} := close",
            f"else if bar_index > {n}",
            f"    {v}Chg = math.abs(close - close[{n}])", f"    float {v}Vol = 0.0",
            f"    for k = 0 to {n - 1}", f"        {v}Vol += math.abs(close[k] - close[k + 1])",
            f"    {v}Er = {v}Vol == 0 ? 0.0 : {v}Chg / {v}Vol",
            f"    {v}Sc = math.pow({v}Er * {R_(K1)} + {R_(K2)}, 2)",
            f"    {v} := {v} + {v}Sc * (close - {v})"]


@fam("ma2", "Trend following (Hull / TEMA / KAMA cross)", {"ma2_type": ["HMA", "TEMA", "KAMA"], "fast": (5, 40), "slow": (20, 150), "trend": T4})
class _:
    def py(p, cc):
        f = _ma2_py(p["ma2_type"], cc.c, p["fast"], cc)
        s = _ma2_py(p["ma2_type"], cc.c, p["slow"], cc)
        up, dn = _cx(f, s), _cu(f, s)
        return up, dn, dn, up, {f"{p['ma2_type']} {p['fast']}": f, f"{p['ma2_type']} {p['slow']}": s}

    def pine(p):
        return (_ma2_pine(p["ma2_type"], p["fast"], "mFast") + _ma2_pine(p["ma2_type"], p["slow"], "mSlow") +
                ["up = ta.crossover(mFast, mSlow)", "dn = ta.crossunder(mFast, mSlow)",
                 "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up",
                 f"plot(mFast, '{p['ma2_type']} {p['fast']}', color=color.orange)", f"plot(mSlow, '{p['ma2_type']} {p['slow']}', color=color.blue)"])

    def desc(p):
        return f"Beli saat {p['ma2_type']} {p['fast']} memotong ke atas {p['ma2_type']} {p['slow']}; short saat memotong ke bawah"


@fam("ichimoku", "Trend following (Ichimoku)", {"t_len": (7, 12), "k_len": (20, 30), "s_len": (40, 60)})
class _:
    def py(p, cc):
        h, l, c = cc.h, cc.l, cc.c
        mid = lambda n: (ind.rolling_max(h, n) + ind.rolling_min(l, n)) / 2
        tk, kj = mid(p["t_len"]), mid(p["k_len"])
        sa, sb = ind.shift((tk + kj) / 2, p["k_len"]), ind.shift(mid(p["s_len"]), p["k_len"])
        top, bot = np.maximum(sa, sb), np.minimum(sa, sb)
        up, dn = _cx(tk, kj), _cu(tk, kj)
        with np.errstate(invalid="ignore"):
            return up & (c > top), dn, dn & (c < bot), up, {"Tenkan": tk, "Kijun": kj, "Cloud A": sa, "Cloud B": sb}

    def pine(p):
        return [f"ichT = (ta.highest(high, {p['t_len']}) + ta.lowest(low, {p['t_len']})) / 2",
                f"ichK = (ta.highest(high, {p['k_len']}) + ta.lowest(low, {p['k_len']})) / 2",
                "ichA0 = (ichT + ichK) / 2", f"ichB0 = (ta.highest(high, {p['s_len']}) + ta.lowest(low, {p['s_len']})) / 2",
                f"ichA = ichA0[{p['k_len']}]", f"ichB = ichB0[{p['k_len']}]",
                "ichTop = math.max(ichA, ichB)", "ichBot = math.min(ichA, ichB)",
                "up = ta.crossover(ichT, ichK)", "dn = ta.crossunder(ichT, ichK)",
                "longEntry = up and close > ichTop", "longExit = dn", "shortEntry = dn and close < ichBot", "shortExit = up",
                "plot(ichT, 'Tenkan', color=color.blue)", "plot(ichK, 'Kijun', color=color.maroon)"]

    def desc(p):
        return f"Ichimoku ({p['t_len']}/{p['k_len']}/{p['s_len']}): beli saat Tenkan memotong ke atas Kijun dan harga di atas awan; short kebalikannya"


@fam("psar", "Trend following (Parabolic SAR)", {"ps_start": [0.01, 0.02, 0.03], "ps_inc": [0.01, 0.02, 0.03], "ps_max": [0.1, 0.2, 0.3], "trend": T4})
class _:
    def py(p, cc):
        d, s = cc.get(("psar", p["ps_start"], p["ps_inc"], p["ps_max"]), lambda: ind.psar(cc.h, cc.l, cc.c, p["ps_start"], p["ps_inc"], p["ps_max"]))
        d1 = ind.shift(d)
        fu, fd = (d == 1) & (d1 == -1), (d == -1) & (d1 == 1)
        return fu, fd, fd, fu, {"Parabolic SAR": s}

    def pine(p):
        S, I, M = R_(float(p["ps_start"])), R_(float(p["ps_inc"])), R_(float(p["ps_max"]))
        return ["var float psS = na", "var float psE = na", "var float psA = na", "var int psD = 0",
                "if bar_index == 1",
                "    psD := close >= close[1] ? 1 : -1", "    psS := psD == 1 ? low[1] : high[1]",
                "    psE := psD == 1 ? high : low", f"    psA := {S}",
                "else if bar_index > 1",
                "    psS := psS + psA * (psE - psS)",
                "    if psD == 1",
                "        psS := math.min(psS, low[1], low[2])",
                "        if low < psS",
                f"            psD := -1", "            psS := psE", "            psE := low", f"            psA := {S}",
                "        else if high > psE",
                f"            psE := high", f"            psA := math.min(psA + {I}, {M})",
                "    else",
                "        psS := math.max(psS, high[1], high[2])",
                "        if high > psS",
                f"            psD := 1", "            psS := psE", "            psE := high", f"            psA := {S}",
                "        else if low < psE",
                f"            psE := low", f"            psA := math.min(psA + {I}, {M})",
                "flipUp = psD == 1 and psD[1] == -1", "flipDn = psD == -1 and psD[1] == 1",
                "longEntry = flipUp", "longExit = flipDn", "shortEntry = flipDn", "shortExit = flipUp",
                "plot(psS, 'Parabolic SAR', color=color.yellow, style=plot.style_cross)"]

    def desc(p):
        return f"Beli saat Parabolic SAR ({p['ps_start']}, {p['ps_inc']}, {p['ps_max']}) pindah ke bawah harga; short saat pindah ke atas"


@fam("heikin", "Trend following (Heikin-Ashi)", {"ha_n": (1, 3), "trend": T4})
class _:
    def py(p, cc):
        g, r, ho, hc = cc.get(("ha",), lambda: ind.heikin_counts(cc.o, cc.h, cc.l, cc.c))
        return g == p["ha_n"], r >= 1, r == p["ha_n"], g >= 1, {}

    def pine(p):
        return ["haC = (open + high + low + close) / 4", "var float haO = na", "var int haG = 0", "var int haR = 0",
                "haO := na(haO) ? (open + close) / 2 : (haO + haC[1]) / 2",
                "haG := haC > haO ? haG + 1 : 0", "haR := haC < haO ? haR + 1 : 0",
                f"longEntry = haG == {p['ha_n']}", "longExit = haR >= 1", f"shortEntry = haR == {p['ha_n']}", "shortExit = haG >= 1"]

    def desc(p):
        return f"Beli setelah {p['ha_n']} candle Heikin-Ashi hijau berturut-turut, keluar saat ada candle merah; short kebalikannya"


@fam("cci", "Momentum / reversal (CCI)", {"cci_len": (10, 50), "cci_lvl": [100, 150, 200], "cci_mode": [0, 1], "trend": T4}, sub="_cci")
class _:
    def py(p, cc):
        n = p["cci_len"]
        def f():
            tp = (cc.h + cc.l + cc.c) / 3
            m = ind.sma(tp, n)
            md = ind.mean_dev(tp, np.asarray(m, float), n)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(md == 0, np.nan, (tp - m) / (0.015 * md))
            v = np.array(v, dtype=float); v[:n] = np.nan
            return v
        v = cc.get(("cci", n), f)
        L = p["cci_lvl"]
        if p["cci_mode"] == 0:
            return _cx(v, -L), _cx(v, 0), _cu(v, L), _cu(v, 0), {"_cci": v}
        return _cx(v, L), _cu(v, 0), _cu(v, -L), _cx(v, 0), {"_cci": v}

    def pine(p):
        n, L = p["cci_len"], p["cci_lvl"]
        out = ["cTp = (high + low + close) / 3", f"cM = ta.sma(cTp, {n})", "float cMd = 0.0",
               f"for k = 0 to {n - 1}", "    cMd += math.abs(cTp[k] - cM)", f"cMd := cMd / {n}",
               f"cciV = bar_index < {n} or cMd == 0 ? na : (cTp - cM) / (0.015 * cMd)"]
        if p["cci_mode"] == 0:
            out += [f"longEntry = ta.crossover(cciV, -{L})", "longExit = ta.crossover(cciV, 0)", f"shortEntry = ta.crossunder(cciV, {L})", "shortExit = ta.crossunder(cciV, 0)"]
        else:
            out += [f"longEntry = ta.crossover(cciV, {L})", "longExit = ta.crossunder(cciV, 0)", f"shortEntry = ta.crossunder(cciV, -{L})", "shortExit = ta.crossover(cciV, 0)"]
        return out

    def desc(p):
        if p["cci_mode"] == 0:
            return f"CCI({p['cci_len']}) balik arah: beli saat naik kembali di atas -{p['cci_lvl']}, keluar di 0; short kebalikannya"
        return f"CCI({p['cci_len']}) momentum: beli saat tembus +{p['cci_lvl']}, keluar saat turun di bawah 0; short kebalikannya"


@fam("willr", "Mean reversion (Williams %R)", {"wr_len": (7, 30), "wr_lo": [-80, -85, -90], "wr_hi": [-20, -15, -10], "trend": T4}, sub="_wr")
class _:
    def py(p, cc):
        n = p["wr_len"]
        def f():
            hh, ll = ind.rolling_max(cc.h, n), ind.rolling_min(cc.l, n)
            rng = hh - ll
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.array(np.where(rng == 0, np.nan, -100 * (hh - cc.c) / rng), dtype=float)
        w = cc.get(("wr", n), f)
        return _cx(w, p["wr_lo"]), _cx(w, p["wr_hi"]), _cu(w, p["wr_hi"]), _cu(w, p["wr_lo"]), {"_wr": w}

    def pine(p):
        n = p["wr_len"]
        return [f"wrH = ta.highest(high, {n})", f"wrL = ta.lowest(low, {n})", "wrRng = wrH - wrL",
                "wrV = wrRng == 0 ? na : -100 * (wrH - close) / wrRng",
                f"longEntry = ta.crossover(wrV, {p['wr_lo']})", f"longExit = ta.crossover(wrV, {p['wr_hi']})",
                f"shortEntry = ta.crossunder(wrV, {p['wr_hi']})", f"shortExit = ta.crossunder(wrV, {p['wr_lo']})"]

    def desc(p):
        return f"Williams %R({p['wr_len']}): beli saat naik dari bawah {p['wr_lo']}, keluar di atas {p['wr_hi']}; short kebalikannya"


@fam("mfi", "Mean reversion (Money Flow Index)", {"mfi_len": (7, 30), "mfi_lo": [10, 20, 30], "mfi_hi": [70, 80, 90], "trend": T4}, needs_volume=True, sub="_mfi")
class _:
    def py(p, cc):
        n = p["mfi_len"]
        def f():
            tp = (cc.h + cc.l + cc.c) / 3
            rmf = tp * cc.v
            d = tp - ind.shift(tp)
            with np.errstate(invalid="ignore"):
                eps = tp * 1e-12        # equal typical prices can differ by 1 float ulp; treat as "no change" (same in Pine)
                pos = np.where(d > eps, rmf, 0.0); neg = np.where(d < -eps, rmf, 0.0)
            sp, sn = ind.rolling_sum(pos, n), ind.rolling_sum(neg, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(sn == 0, 100.0, 100 - 100 / (1 + sp / sn))
            v = np.array(v, dtype=float); v[:n] = np.nan
            return v
        v = cc.get(("mfi", n), f)
        return _cx(v, p["mfi_lo"]), _cx(v, 50), _cu(v, p["mfi_hi"]), _cu(v, 50), {"_mfi": v}

    def pine(p):
        n = p["mfi_len"]
        return ["mfTp = (high + low + close) / 3", "mfRaw = mfTp * volume", "mfD = mfTp - mfTp[1]",
                "mfEps = mfTp * 1e-12", "mfPos = mfD > mfEps ? mfRaw : 0.0", "mfNeg = mfD < -mfEps ? mfRaw : 0.0",
                f"mfSp = math.sum(mfPos, {n})", f"mfSn = math.sum(mfNeg, {n})",
                f"mfiV = bar_index < {n} ? na : mfSn == 0 ? 100.0 : 100 - 100 / (1 + mfSp / mfSn)",
                f"longEntry = ta.crossover(mfiV, {p['mfi_lo']})", "longExit = ta.crossover(mfiV, 50)",
                f"shortEntry = ta.crossunder(mfiV, {p['mfi_hi']})", "shortExit = ta.crossunder(mfiV, 50)"]

    def desc(p):
        return f"MFI({p['mfi_len']}) (RSI dengan volume): beli saat naik dari bawah {p['mfi_lo']}, keluar di 50; short dari atas {p['mfi_hi']}"


@fam("obv", "Volume trend (OBV)", {"obv_len": (10, 100), "trend": T4}, needs_volume=True)
class _:
    def py(p, cc):
        def f():
            d = cc.c - ind.shift(cc.c)
            with np.errstate(invalid="ignore"):
                sv = np.where(d > 0, cc.v, np.where(d < 0, -cc.v, 0.0))
            return np.cumsum(sv)
        ob = cc.get(("obv",), f)
        e = cc.get(("obv_ema", p["obv_len"]), lambda: ind.ema(ob, p["obv_len"]))
        up, dn = _cx(ob, e), _cu(ob, e)
        return up, dn, dn, up, {}

    def pine(p):
        return ["var float obvV = 0.0", "obvV += close > close[1] ? volume : close < close[1] ? -volume : 0.0",
                f"obvE = ta.ema(obvV, {p['obv_len']})", "up = ta.crossover(obvV, obvE)", "dn = ta.crossunder(obvV, obvE)",
                "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up"]

    def desc(p):
        return f"Beli saat OBV (volume kumulatif) memotong ke atas EMA {p['obv_len']}-nya; short saat memotong ke bawah"


@fam("vol_bo", "Breakout + lonjakan volume", {"entry_len": (10, 60), "vmult": [1.5, 2.0, 3.0], "exit_len": (5, 30), "trend": T4}, needs_volume=True)
class _:
    def py(p, cc):
        hh = ind.shift(ind.rolling_max(cc.h, p["entry_len"])); ll = ind.shift(ind.rolling_min(cc.l, p["entry_len"]))
        xh = ind.shift(ind.rolling_max(cc.h, p["exit_len"])); xl = ind.shift(ind.rolling_min(cc.l, p["exit_len"]))
        vm = cc.get(("vsma", 20), lambda: ind.sma(cc.v, 20))
        c = cc.c
        with np.errstate(invalid="ignore"):
            vok = cc.v > p["vmult"] * vm
            return (c > hh) & vok, c < xl, (c < ll) & vok, c > xh, {"Breakout high": hh, "Breakout low": ll}

    def pine(p):
        return [f"vbH = ta.highest(high, {p['entry_len']})[1]", f"vbL = ta.lowest(low, {p['entry_len']})[1]",
                f"vbXH = ta.highest(high, {p['exit_len']})[1]", f"vbXL = ta.lowest(low, {p['exit_len']})[1]",
                "vbVm = ta.sma(volume, 20)", f"vbOk = volume > {p['vmult']} * vbVm",
                "longEntry = close > vbH and vbOk", "longExit = close < vbXL", "shortEntry = close < vbL and vbOk", "shortExit = close > vbXH"]

    def desc(p):
        return f"Beli saat harga tembus high {p['entry_len']} candle DAN volume > {p['vmult']}x rata-rata; keluar saat tembus low {p['exit_len']} candle; short kebalikannya"


@fam("atr_ch", "Breakout (ATR channel)", {"ac_len": (10, 100), "ac_mult": [1.0, 1.5, 2.0, 2.5, 3.0], "trend": T4})
class _:
    def py(p, cc):
        b = cc.get(("sma", p["ac_len"]), lambda: ind.sma(cc.c, p["ac_len"]))
        u, lo = b + p["ac_mult"] * cc.atr, b - p["ac_mult"] * cc.atr
        return _cx(cc.c, u), _cu(cc.c, b), _cu(cc.c, lo), _cx(cc.c, b), {"ATR ch upper": u, "ATR ch basis": b, "ATR ch lower": lo}

    def pine(p):
        return [f"acB = ta.sma(close, {p['ac_len']})", f"acU = acB + {p['ac_mult']} * atr", f"acL = acB - {p['ac_mult']} * atr",
                "longEntry = ta.crossover(close, acU)", "longExit = ta.crossunder(close, acB)",
                "shortEntry = ta.crossunder(close, acL)", "shortExit = ta.crossover(close, acB)",
                "plot(acU, 'ATR ch upper', color=color.teal)", "plot(acL, 'ATR ch lower', color=color.teal)"]

    def desc(p):
        return f"Beli saat harga tembus SMA {p['ac_len']} + {p['ac_mult']} x ATR, keluar saat kembali di bawah SMA; short kebalikannya"


@fam("pivot", "Breakout support/resistance (pivot)", {"pv_left": (2, 10), "pv_right": (2, 10), "trend": T4})
class _:
    def py(p, cc):
        L, R = p["pv_left"], p["pv_right"]
        def levels(x, roll):
            ref = ind.shift(x, R)
            cond = roll(x, L + R + 1) == ref
            lv = np.where(cond, ref, np.nan)
            import pandas as pd
            return pd.Series(lv).ffill().to_numpy()
        res = cc.get(("pvh", L, R), lambda: levels(cc.h, ind.rolling_max))
        sup = cc.get(("pvl", L, R), lambda: levels(cc.l, ind.rolling_min))
        return _cx(cc.c, res), _cu(cc.c, sup), _cu(cc.c, sup), _cx(cc.c, res), {"Resistance": res, "Support": sup}

    def pine(p):
        L, R = p["pv_left"], p["pv_right"]
        return [f"pvHc = high[{R}] == ta.highest(high, {L + R + 1})", f"pvLc = low[{R}] == ta.lowest(low, {L + R + 1})",
                "var float pvRes = na", "var float pvSup = na",
                f"if pvHc", f"    pvRes := high[{R}]", f"if pvLc", f"    pvSup := low[{R}]",
                "longEntry = ta.crossover(close, pvRes)", "longExit = ta.crossunder(close, pvSup)",
                "shortEntry = ta.crossunder(close, pvSup)", "shortExit = ta.crossover(close, pvRes)",
                "plot(pvRes, 'Resistance', color=color.red, style=plot.style_stepline)", "plot(pvSup, 'Support', color=color.green, style=plot.style_stepline)"]

    def desc(p):
        return f"Beli saat harga tembus resistance terakhir (pivot {p['pv_left']}/{p['pv_right']}); short saat tembus support"


@fam("nr7", "Breakout setelah candle sempit (NR / inside bar)", {"nr_len": [4, 5, 6, 7], "inside": [0, 1], "nr_win": [3, 5, 10], "trend": T4})
class _:
    def py(p, cc):
        h, l, c = cc.h, cc.l, cc.c
        rng = h - l
        with np.errstate(invalid="ignore"):
            setup = (rng == ind.rolling_min(rng, p["nr_len"]))
            if p["inside"]:
                setup = setup & (h < ind.shift(h)) & (l > ind.shift(l))
        sh, sl, since = ind.nr_setup(h, l, c, _bool(setup), p["nr_win"])
        ok = (since >= 1) & (since <= p["nr_win"])
        up, dn = _cx(c, sh), _cu(c, sl)
        return up & ok, dn, dn & ok, up, {"Setup high": sh, "Setup low": sl}

    def pine(p):
        ins = "(high < high[1] and low > low[1])" if p["inside"] else "true"
        return ["nrR = high - low", f"nrMin = ta.lowest(nrR, {p['nr_len']})", f"nrIn = {ins}", "nrSet = nrR == nrMin and nrIn",
                "var float nrH = na", "var float nrL = na", "var float nrS = 1000000.0",
                "if nrSet", "    nrH := high", "    nrL := low", "    nrS := 0.0", "else", "    nrS += 1",
                f"nrOk = nrS >= 1 and nrS <= {p['nr_win']}",
                "nrUp = ta.crossover(close, nrH)", "nrDn = ta.crossunder(close, nrL)",
                "longEntry = nrUp and nrOk", "longExit = nrDn", "shortEntry = nrDn and nrOk", "shortExit = nrUp"]

    def desc(p):
        k = "inside bar + " if p["inside"] else ""
        return f"Tunggu {k}candle tersempit dari {p['nr_len']} candle, lalu beli saat harga tembus high-nya (maks {p['nr_win']} candle kemudian); short saat tembus low"


@fam("engulf", "Pola candle engulfing", {"trend": [50, 100, 200]})
class _:
    def py(p, cc):
        o, c = cc.o, cc.c
        o1, c1 = ind.shift(o), ind.shift(c)
        with np.errstate(invalid="ignore"):
            bull = (c > o) & (c1 < o1) & (c >= o1) & (o <= c1)
            bear = (c < o) & (c1 > o1) & (c <= o1) & (o >= c1)
        return bull, bear, bear, bull, {}

    def pine(p):
        return ["bullE = close > open and close[1] < open[1] and close >= open[1] and open <= close[1]",
                "bearE = close < open and close[1] > open[1] and close <= open[1] and open >= close[1]",
                "longEntry = bullE", "longExit = bearE", "shortEntry = bearE", "shortExit = bullE"]

    def desc(p):
        return "Beli saat muncul bullish engulfing searah tren; short saat bearish engulfing; keluar saat pola kebalikannya"


@fam("orb", "Opening range breakout (sesi)", {"or_start": [0, 7, 13], "or_len": [1, 2], "or_hold": [4, 6, 8], "trend": T4}, intraday=True)
class _:
    def py(p, cc):
        S, Ln = p["or_start"], p["or_len"]
        E = min(S + Ln + p["or_hold"], 24)
        hr, day = cc.hour, cc.day
        oh, ol = cc.get(("orb", S, Ln), lambda: ind.opening_range(cc.h, cc.l, hr.astype(np.float64), day.astype(np.float64), S, Ln))
        tradable = (hr >= S + Ln) & (hr < E - 1)
        ext = (hr >= E - 1) | (hr < S)
        up, dn = _cx(cc.c, oh), _cu(cc.c, ol)
        return up & tradable, ext, dn & tradable, ext, {"Range high": oh, "Range low": ol}

    def pine(p):
        S, Ln = p["or_start"], p["or_len"]
        E = min(S + Ln + p["or_hold"], 24)
        return ["orHr = math.floor(time / 3600000) % 24", "orDay = math.floor(time / 86400000)",
                f"orIn = orHr >= {S} and orHr < {S + Ln}", f"orInP = orHr[1] >= {S} and orHr[1] < {S + Ln}",
                "var float orH = na", "var float orL = na",
                "if orIn and (not orInP or orDay != orDay[1])", "    orH := high", "    orL := low",
                "else if orIn", "    orH := math.max(orH, high)", "    orL := math.min(orL, low)",
                f"orTr = orHr >= {S + Ln} and orHr < {E - 1}", f"orEx = orHr >= {E - 1} or orHr < {S}",
                "orUp = ta.crossover(close, orH)", "orDn = ta.crossunder(close, orL)",
                "longEntry = orUp and orTr", "longExit = orEx", "shortEntry = orDn and orTr", "shortExit = orEx",
                "plot(orH, 'Range high', color=color.green, style=plot.style_stepline)", "plot(orL, 'Range low', color=color.red, style=plot.style_stepline)"]

    def desc(p):
        S, Ln = p["or_start"], p["or_len"]
        E = min(S + Ln + p["or_hold"], 24)
        return (f"Catat high/low {Ln} jam pertama mulai {S:02d}:00 UTC ({(S + 7) % 24:02d}:00 WIB), beli saat harga tembus high-nya, "
                f"short saat tembus low-nya; semua posisi ditutup sekitar {E:02d}:00 UTC")


@fam("aroon", "Trend following (Aroon)", {"ar_len": (10, 50), "trend": T4})
class _:
    def py(p, cc):
        up, dn = cc.get(("aroon", p["ar_len"]), lambda: ind.aroon(cc.h, cc.l, p["ar_len"]))
        a, b = _cx(up, dn), _cu(up, dn)
        return a, b, b, a, {}

    def pine(p):
        n = p["ar_len"]
        return ["float arHv = high", "int arHi = 0", "float arLv = low", "int arLi = 0",
                f"for k = 1 to {n}", "    if high[k] > arHv", "        arHv := high[k]", "        arHi := k",
                "    if low[k] < arLv", "        arLv := low[k]", "        arLi := k",
                f"arUp = bar_index < {n} ? na : 100.0 * ({n} - arHi) / {n}", f"arDn = bar_index < {n} ? na : 100.0 * ({n} - arLi) / {n}",
                "up = ta.crossover(arUp, arDn)", "dn = ta.crossunder(arUp, arDn)",
                "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up"]

    def desc(p):
        return f"Beli saat Aroon Up({p['ar_len']}) memotong ke atas Aroon Down; short kebalikannya"


@fam("vortex", "Trend following (Vortex)", {"vx_len": (7, 30), "trend": T4})
class _:
    def py(p, cc):
        n = p["vx_len"]
        def f():
            h, l, c = cc.h, cc.l, cc.c
            l1, h1, c1 = ind.shift(l), ind.shift(h), ind.shift(c)
            vp, vm = np.abs(h - l1), np.abs(l - h1)
            tr = np.maximum(h - l, np.maximum(np.abs(h - c1), np.abs(l - c1)))
            tr[0] = np.nan
            st = ind.rolling_sum(tr, n)
            with np.errstate(invalid="ignore", divide="ignore"):
                a, b = ind.rolling_sum(vp, n) / st, ind.rolling_sum(vm, n) / st
            a = np.array(a, dtype=float); b = np.array(b, dtype=float)
            a[:n + 1] = np.nan; b[:n + 1] = np.nan
            return a, b
        a, b = cc.get(("vortex", n), f)
        up, dn = _cx(a, b), _cu(a, b)
        return up, dn, dn, up, {}

    def pine(p):
        n = p["vx_len"]
        return ["vxP = math.abs(high - low[1])", "vxM = math.abs(low - high[1])",
                "vxTr = math.max(high - low, math.max(math.abs(high - close[1]), math.abs(low - close[1])))",
                f"vxS = math.sum(vxTr, {n})",
                f"vxIp = bar_index <= {n} ? na : math.sum(vxP, {n}) / vxS", f"vxIm = bar_index <= {n} ? na : math.sum(vxM, {n}) / vxS",
                "up = ta.crossover(vxIp, vxIm)", "dn = ta.crossunder(vxIp, vxIm)",
                "longEntry = up", "longExit = dn", "shortEntry = dn", "shortExit = up"]

    def desc(p):
        return f"Beli saat Vortex VI+({p['vx_len']}) memotong ke atas VI-; short kebalikannya"


def _lr_pine(src, n, v):
    return [f"float {v}Sx = 0.0", f"float {v}Sy = 0.0", f"float {v}Sxx = 0.0", f"float {v}Sxy = 0.0",
            f"for k = 0 to {n - 1}", f"    {v}X = {n - 1} - k", f"    {v}Y = {src}[k]",
            f"    {v}Sx += {v}X", f"    {v}Sy += {v}Y", f"    {v}Sxx += {v}X * {v}X", f"    {v}Sxy += {v}X * {v}Y",
            f"{v}Den = {n} * {v}Sxx - {v}Sx * {v}Sx",
            f"{v}Sl = ({n} * {v}Sxy - {v}Sx * {v}Sy) / {v}Den",
            f"{v} = bar_index < {n - 1} ? na : ({v}Sy - {v}Sl * {v}Sx) / {n} + {v}Sl * {n - 1}"]


@fam("sqz", "Squeeze momentum", {"sq_len2": (10, 30), "kc_m2": [1.0, 1.5, 2.0], "trend": T4}, sub="_sqz")
class _:
    def py(p, cc):
        n, km = p["sq_len2"], p["kc_m2"]
        c = cc.c
        def f():
            b = ind.sma(c, n); sd = ind.stdev(c, n)
            on = ((b - 2.0 * sd) > (b - km * cc.atr)) & ((b + 2.0 * sd) < (b + km * cc.atr))
            mid = ((ind.rolling_max(cc.h, n) + ind.rolling_min(cc.l, n)) / 2 + b) / 2
            mom = ind.linreg(np.asarray(c - mid, float), n)
            return on, mom
        on, mom = cc.get(("sqz", n, km), f)
        on1 = np.r_[False, on[:-1]]
        with np.errstate(invalid="ignore"):
            rel = on1 & ~on
            return rel & (mom > 0), _cu(mom, 0), rel & (mom < 0), _cx(mom, 0), {"_sqz": mom}

    def pine(p):
        n, km = p["sq_len2"], p["kc_m2"]
        return ([f"sqB = ta.sma(close, {n})", f"sqSd = ta.stdev(close, {n})",
                 f"sqOn = (sqB - 2.0 * sqSd) > (sqB - {km} * atr) and (sqB + 2.0 * sqSd) < (sqB + {km} * atr)",
                 f"sqMid = ((ta.highest(high, {n}) + ta.lowest(low, {n})) / 2 + sqB) / 2", "sqSrc = close - sqMid"]
                + _lr_pine("sqSrc", n, "sqMom") +
                ["sqRel = nz(sqOn[1], false) and not sqOn",
                 "sqXD = ta.crossunder(sqMom, 0)", "sqXU = ta.crossover(sqMom, 0)",
                 "longEntry = sqRel and sqMom > 0", "longExit = sqXD", "shortEntry = sqRel and sqMom < 0", "shortExit = sqXU"])

    def desc(p):
        return f"Squeeze momentum ({p['sq_len2']}, KC {p['kc_m2']}): saat Bollinger keluar dari Keltner, beli kalau momentum positif / short kalau negatif; keluar saat momentum ganti tanda"


@fam("lrc", "Mean reversion (linear regression channel)", {"lr_len": (20, 150), "lr_mult": [1.5, 2.0, 2.5, 3.0], "trend": T4})
class _:
    def py(p, cc):
        n, m = p["lr_len"], p["lr_mult"]
        lr = cc.get(("lr", n), lambda: ind.linreg(cc.c, n))
        sd = cc.get(("sd", n), lambda: ind.stdev(cc.c, n))
        u, lo = lr + m * sd, lr - m * sd
        return _cx(cc.c, lo), _cx(cc.c, lr), _cu(cc.c, u), _cu(cc.c, lr), {"LR": lr, "LR upper": u, "LR lower": lo}

    def pine(p):
        n, m = p["lr_len"], p["lr_mult"]
        return (_lr_pine("close", n, "lrV") +
                [f"lrSd = ta.stdev(close, {n})", f"lrU = lrV + {m} * lrSd", f"lrL = lrV - {m} * lrSd",
                 "longEntry = ta.crossover(close, lrL)", "longExit = ta.crossover(close, lrV)",
                 "shortEntry = ta.crossunder(close, lrU)", "shortExit = ta.crossunder(close, lrV)",
                 "plot(lrV, 'LR', color=color.gray)", "plot(lrU, 'LR upper', color=color.purple)", "plot(lrL, 'LR lower', color=color.purple)"])

    def desc(p):
        return f"Saluran regresi linear ({p['lr_len']}, ±{p['lr_mult']} stdev): beli saat harga masuk lagi dari bawah, keluar di garis tengah; short kebalikannya"


@fam("rsi_div", "Divergence RSI", {"rsi_len": (7, 21), "dv_piv": (3, 6), "exit_level": [60, 65, 70], "trend": T4}, sub="_rsi")
class _:
    def py(p, cc):
        r = cc.get(("rsi", p["rsi_len"]), lambda: ind.rsi(cc.c, p["rsi_len"]))
        bull, bear = cc.get(("div", p["rsi_len"], p["dv_piv"]), lambda: ind.divergence(cc.h, cc.l, np.asarray(r, float), p["dv_piv"]))
        return bull, _cx(r, p["exit_level"]), bear, _cu(r, 100 - p["exit_level"]), {"_rsi": r}

    def pine(p):
        R = p["dv_piv"]; W = 2 * R + 1
        return [f"dvR = ta.rsi(close, {p['rsi_len']})", f"dvLo = ta.lowest(low, {W})", f"dvHi = ta.highest(high, {W})",
                "var float dvPl = na", "var float dvPr = na", "var float dvPh = na", "var float dvPhr = na",
                "bool dvBull = false", "bool dvBear = false",
                f"if bar_index >= {W - 1} and low[{R}] == dvLo",
                f"    dvBull := low[{R}] < dvPl and dvR[{R}] > dvPr", f"    dvPl := low[{R}]", f"    dvPr := dvR[{R}]",
                f"if bar_index >= {W - 1} and high[{R}] == dvHi",
                f"    dvBear := high[{R}] > dvPh and dvR[{R}] < dvPhr", f"    dvPh := high[{R}]", f"    dvPhr := dvR[{R}]",
                f"dvXU = ta.crossover(dvR, {p['exit_level']})", f"dvXD = ta.crossunder(dvR, {100 - p['exit_level']})",
                "longEntry = dvBull", "longExit = dvXU", "shortEntry = dvBear", "shortExit = dvXD"]

    def desc(p):
        return (f"Beli saat harga membuat low lebih rendah tapi RSI({p['rsi_len']}) low lebih tinggi (bullish divergence, pivot {p['dv_piv']}), "
                f"keluar saat RSI > {p['exit_level']}; short kebalikannya")


# ------------------------------------------------------------------ crypto derivatives (Binance data, NOT exportable to Pine)
def _no_pine(p):
    return None


def _flow(cc, n):
    def f():
        d = 2 * cc.x["taker_buy"] - cc.v
        with np.errstate(invalid="ignore", divide="ignore"):
            return ind.sma(d, n) / ind.sma(cc.v, n)
    return cc.get(("flow", n), f)


@fam("fund_fade", "Funding ekstrem (lawan kerumunan)", {"z_len": [100, 300, 600], "z_thr": [1.5, 2.0, 2.5, 3.0], "exit_z": [0.0, 0.5], "trend": T4},
     needs=("fund",))
class _:
    def py(p, cc):
        x = cc.x["fund"]

        def zf():
            with np.errstate(invalid="ignore", divide="ignore"):
                return (x - ind.sma(x, p["z_len"])) / ind.stdev(x, p["z_len"])
        z = cc.get(("fundz", p["z_len"]), zf)
        return (_cu(z, -p["z_thr"]), _cx(z, -p["exit_z"]), _cx(z, p["z_thr"]), _cu(z, p["exit_z"]), {})

    pine = staticmethod(_no_pine)

    def desc(p):
        return (f"Short saat funding rate jauh di atas biasanya (z-score > {p['z_thr']} dari {p['z_len']} candle, banyak trader long membayar mahal), "
                f"keluar saat z < {p['exit_z']}; long kebalikannya (funding sangat negatif)")


@fam("oi_div", "Open interest vs harga", {"n": (6, 72), "mode": [0, 1], "thr": [1.0, 2.0, 4.0], "trend": T4}, needs=("oi",))
class _:
    def py(p, cc):
        n = p["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            oc = cc.get(("oichg", n), lambda: cc.x["oi"] / ind.shift(cc.x["oi"], n) - 1) * 100
        hh = cc.get(("hh", n), lambda: ind.shift(ind.rolling_max(cc.h, n)))
        ll = cc.get(("ll", n), lambda: ind.shift(ind.rolling_min(cc.l, n)))
        up, dn = cc.c > hh, cc.c < ll
        with np.errstate(invalid="ignore"):
            if p["mode"] == 0:      # price breaks out while OI falls = positions closing, move likely to fade
                le, se = _bool(dn & (oc < -p["thr"])), _bool(up & (oc < -p["thr"]))
            else:                   # breakout WITH rising OI = new money behind the move
                le, se = _bool(up & (oc > p["thr"])), _bool(dn & (oc > p["thr"]))
        return le, se, se, le, {}

    pine = staticmethod(_no_pine)

    def desc(p):
        if p["mode"] == 0:
            return (f"Lawan breakout {p['n']} candle saat open interest turun > {p['thr']}% (posisi ditutup, gerakan cenderung lemah)")
        return f"Ikut breakout {p['n']} candle saat open interest naik > {p['thr']}% (uang baru masuk)"


@fam("cvd_flow", "Order flow taker (CVD)", {"n": (6, 96), "thr": [0.02, 0.05, 0.1], "mode": [0, 1], "trend": T4}, needs=("taker",))
class _:
    def py(p, cc):
        n = p["n"]
        fr = _flow(cc, n)
        with np.errstate(invalid="ignore"):
            if p["mode"] == 1:      # follow aggressive flow
                le, se = _cx(fr, p["thr"]), _cu(fr, -p["thr"])
            else:                   # divergence: new n-bar low but buyers dominate (and vice versa)
                hh = cc.get(("hh", n), lambda: ind.shift(ind.rolling_max(cc.h, n)))
                ll = cc.get(("ll", n), lambda: ind.shift(ind.rolling_min(cc.l, n)))
                le, se = _bool((cc.c < ll) & (fr > p["thr"])), _bool((cc.c > hh) & (fr < -p["thr"]))
        return le, se, se, le, {}

    pine = staticmethod(_no_pine)

    def desc(p):
        if p["mode"] == 1:
            return f"Ikut order flow: beli saat delta taker {p['n']} candle > {p['thr'] * 100:.0f}% volume, short saat < -{p['thr'] * 100:.0f}%"
        return (f"Divergence order flow: beli saat harga di low {p['n']} candle tapi pembeli agresif dominan (> {p['thr'] * 100:.0f}% volume); "
                "short kebalikannya")


# ------------------------------------------------------------------ v6: more classic retail strategies (all with Pine twins)
def _fb(a):
    return np.nan_to_num(np.asarray(a, float), nan=0).astype(np.bool_)


@fam("macd_div", "Divergence MACD", {"fast": (5, 20), "slow": (21, 60), "signal": (5, 15), "dv_piv": (3, 6), "trend": T4}, sub="_macd")
class _:
    def py(p, cc):
        ef = cc.get(("ema", "c", p["fast"]), lambda: ind.ema(cc.c, p["fast"]))
        es = cc.get(("ema", "c", p["slow"]), lambda: ind.ema(cc.c, p["slow"]))
        ml = ef - es
        sg = ind.ema(ml, p["signal"])
        bull, bear = ind.divergence(cc.h, cc.l, np.asarray(ml, float), p["dv_piv"])
        return bull, _cu(ml, sg), bear, _cx(ml, sg), {"_macd": ml, "_macd_signal": sg}

    def pine(p):
        R = p["dv_piv"]; W = 2 * R + 1
        return [f"mdL = ta.ema(close, {p['fast']}) - ta.ema(close, {p['slow']})", f"mdS = ta.ema(mdL, {p['signal']})",
                f"mdLo = ta.lowest(low, {W})", f"mdHi = ta.highest(high, {W})",
                "var float mdPl = na", "var float mdPr = na", "var float mdPh = na", "var float mdPhr = na",
                "bool mdBull = false", "bool mdBear = false",
                f"if bar_index >= {W - 1} and low[{R}] == mdLo",
                f"    mdBull := low[{R}] < mdPl and mdL[{R}] > mdPr", f"    mdPl := low[{R}]", f"    mdPr := mdL[{R}]",
                f"if bar_index >= {W - 1} and high[{R}] == mdHi",
                f"    mdBear := high[{R}] > mdPh and mdL[{R}] < mdPhr", f"    mdPh := high[{R}]", f"    mdPhr := mdL[{R}]",
                "mdXU = ta.crossover(mdL, mdS)", "mdXD = ta.crossunder(mdL, mdS)",
                "longEntry = mdBull", "longExit = mdXD", "shortEntry = mdBear", "shortExit = mdXU"]

    def desc(p):
        return (f"Beli saat harga membuat low lebih rendah tapi MACD({p['fast']},{p['slow']}) low lebih tinggi (bullish divergence, "
                f"pivot {p['dv_piv']}); keluar saat MACD memotong ke bawah signal({p['signal']}). Short kebalikannya")


@fam("pinbar", "Pola candle pin bar", {"pb_k": [2.0, 3.0], "pb_wr": [0.5, 0.66], "pb_n": [5, 10, 20], "trend": T4})
class _:
    def py(p, cc):
        o, h, l, c = cc.o, cc.h, cc.l, cc.c
        rg = h - l; body = np.abs(c - o)
        lw = np.minimum(o, c) - l; uw = h - np.maximum(o, c)
        ll = cc.get(("llc", p["pb_n"]), lambda: ind.rolling_min(l, p["pb_n"]))
        hh = cc.get(("hhc", p["pb_n"]), lambda: ind.rolling_max(h, p["pb_n"]))
        tol = c * 1e-10   # exact ties (wick = k x body) are decided by 1-ulp rounding; treat them the same way as Pine
        with np.errstate(invalid="ignore"):
            bull = (rg > 0) & (lw >= p["pb_k"] * body - tol) & (lw >= p["pb_wr"] * rg - tol) & (l == ll)
            bear = (rg > 0) & (uw >= p["pb_k"] * body - tol) & (uw >= p["pb_wr"] * rg - tol) & (h == hh)
        return bull, bear, bear, bull, {}

    def pine(p):
        return ["pbR = high - low", "pbBody = math.abs(close - open)", "pbLw = math.min(open, close) - low", "pbUw = high - math.max(open, close)",
                f"pbLL = ta.lowest(low, {p['pb_n']})", f"pbHH = ta.highest(high, {p['pb_n']})",
                "pbTol = close * 1e-10",
                f"pbBull = pbR > 0 and pbLw >= {R_(p['pb_k'])} * pbBody - pbTol and pbLw >= {R_(p['pb_wr'])} * pbR - pbTol and low == pbLL",
                f"pbBear = pbR > 0 and pbUw >= {R_(p['pb_k'])} * pbBody - pbTol and pbUw >= {R_(p['pb_wr'])} * pbR - pbTol and high == pbHH",
                "longEntry = pbBull", "longExit = pbBear", "shortEntry = pbBear", "shortExit = pbBull"]

    def desc(p):
        return (f"Beli saat pin bar bullish (ekor bawah ≥ {p['pb_k']:g}x badan dan ≥ {p['pb_wr'] * 100:.0f}% panjang candle) di low terendah "
                f"{p['pb_n']} candle; short saat pin bar bearish di high tertinggi")


@fam("doji", "Pola candle doji (pembalikan)", {"dj_t": [0.05, 0.1], "dj_n": [5, 10, 20], "trend": T4})
class _:
    def py(p, cc):
        o, h, l, c = cc.o, cc.h, cc.l, cc.c
        rg = h - l
        with np.errstate(invalid="ignore"):
            dj = (rg > 0) & (np.abs(c - o) <= p["dj_t"] * rg + c * 1e-10)
        ll = cc.get(("llc", p["dj_n"]), lambda: ind.rolling_min(l, p["dj_n"]))
        hh = cc.get(("hhc", p["dj_n"]), lambda: ind.rolling_max(h, p["dj_n"]))
        dj1 = _fb(ind.shift(dj.astype(float)))
        with np.errstate(invalid="ignore"):
            bull = dj1 & (ind.shift(l) == ind.shift(ll)) & (c > ind.shift(h))
            bear = dj1 & (ind.shift(h) == ind.shift(hh)) & (c < ind.shift(l))
        return bull, bear, bear, bull, {}

    def pine(p):
        return ["djR = high - low", f"djDoji = djR > 0 and math.abs(close - open) <= {R_(p['dj_t'])} * djR + close * 1e-10",
                f"djLL = ta.lowest(low, {p['dj_n']})", f"djHH = ta.highest(high, {p['dj_n']})",
                "djBull = nz(djDoji[1], false) and low[1] == djLL[1] and close > high[1]",
                "djBear = nz(djDoji[1], false) and high[1] == djHH[1] and close < low[1]",
                "longEntry = djBull", "longExit = djBear", "shortEntry = djBear", "shortExit = djBull"]

    def desc(p):
        return (f"Doji (badan ≤ {p['dj_t'] * 100:.0f}% panjang candle) di low terendah {p['dj_n']} candle, lalu candle berikutnya tutup "
                f"di atas high doji: beli. Short kebalikannya")


@fam("fib", "Pullback Fibonacci retracement", {"fb_n": (20, 200), "fb_f": [0.382, 0.5, 0.618, 0.786], "trend": T4})
class _:
    def py(p, cc):
        n, f = p["fb_n"], p["fb_f"]
        H = cc.get(("hhc", n), lambda: ind.rolling_max(cc.h, n))
        L = cc.get(("llc", n), lambda: ind.rolling_min(cc.l, n))
        with np.errstate(invalid="ignore"):
            bsH = ind.barssince(_fb(cc.h >= H)); bsL = ind.barssince(_fb(cc.l <= L))
            up = bsH < bsL; dn = bsL < bsH
            lvL = H - f * (H - L); lvS = L + f * (H - L)
            le = up & (cc.l <= lvL) & (cc.c > lvL)
            se = dn & (cc.h >= lvS) & (cc.c < lvS)
            lx = cc.h >= ind.shift(H); sx = cc.l <= ind.shift(L)
        return le, lx, se, sx, {f"Fib {f:g} (long)": np.where(up, lvL, np.nan), f"Fib {f:g} (short)": np.where(dn, lvS, np.nan)}

    def pine(p):
        n, f = p["fb_n"], R_(p["fb_f"])
        return [f"fbH = ta.highest(high, {n})", f"fbL = ta.lowest(low, {n})",
                "fbBsH = ta.barssince(high >= fbH)", "fbBsL = ta.barssince(low <= fbL)",
                "fbUp = fbBsH < fbBsL", "fbDn = fbBsL < fbBsH",
                f"fbLvL = fbH - {f} * (fbH - fbL)", f"fbLvS = fbL + {f} * (fbH - fbL)",
                "longEntry = fbUp and low <= fbLvL and close > fbLvL", "shortEntry = fbDn and high >= fbLvS and close < fbLvS",
                "longExit = high >= fbH[1]", "shortExit = low <= fbL[1]",
                "plot(fbUp ? fbLvL : na, 'Fib long', color=color.teal, style=plot.style_linebr)",
                "plot(fbDn ? fbLvS : na, 'Fib short', color=color.maroon, style=plot.style_linebr)"]

    def desc(p):
        return (f"Swing {p['fb_n']} candle. Kalau high terbaru lebih baru dari low (kaki naik), beli saat harga turun ke retracement "
                f"{p['fb_f'] * 100:.1f}% lalu tutup di atasnya; target high sebelumnya. Short kebalikannya")


@fam("vwap_mr", "VWAP reversion (band harian)", {"vw_k": [1.0, 1.5, 2.0, 2.5], "vw_min": [3, 6], "trend": T4}, intraday=True, needs_volume=True)
class _:
    def py(p, cc):
        vw, sd, nb = cc.get(("vwap_day",), lambda: ind.vwap_day(cc.h, cc.l, cc.c, cc.v, cc.day.astype(np.float64)))
        lo, hi = vw - p["vw_k"] * sd, vw + p["vw_k"] * sd
        ok = nb >= p["vw_min"]
        return ok & _cx(cc.c, lo), _cx(cc.c, vw), ok & _cu(cc.c, hi), _cu(cc.c, vw), {"VWAP": vw, "VWAP bawah": lo, "VWAP atas": hi}

    def pine(p):
        k = R_(p["vw_k"])
        return ["vwDay = math.floor(time / 86400000)",
                "var float vwSPV = 0.0", "var float vwSV = 0.0", "var float vwSP2V = 0.0", "var int vwK = 0",
                "if bar_index == 0 or vwDay != vwDay[1]", "    vwSPV := 0.0", "    vwSV := 0.0", "    vwSP2V := 0.0", "    vwK := 0",
                "vwTp = (high + low + close) / 3",
                "vwSPV := vwSPV + vwTp * volume", "vwSV := vwSV + volume", "vwSP2V := vwSP2V + vwTp * vwTp * volume", "vwK := vwK + 1",
                "vwV = vwSV > 0 ? vwSPV / vwSV : na", "vwVar = vwSV > 0 ? vwSP2V / vwSV - vwV * vwV : na",
                "vwSd = vwSV > 0 ? (vwVar > 0 ? math.sqrt(vwVar) : 0.0) : na",
                f"vwLo = vwV - {k} * vwSd", f"vwHi = vwV + {k} * vwSd",
                f"longEntry = vwK >= {p['vw_min']} and ta.crossover(close, vwLo)", "longExit = ta.crossover(close, vwV)",
                f"shortEntry = vwK >= {p['vw_min']} and ta.crossunder(close, vwHi)", "shortExit = ta.crossunder(close, vwV)",
                "plot(vwV, 'VWAP', color=color.orange)", "plot(vwLo, 'VWAP bawah', color=color.gray)", "plot(vwHi, 'VWAP atas', color=color.gray)"]

    def desc(p):
        return (f"VWAP harian (mulai ulang 00:00 UTC = 07:00 WIB). Beli saat harga kembali naik ke atas VWAP − {p['vw_k']:g} deviasi, "
                f"keluar di VWAP; short kebalikannya. Hanya setelah {p['vw_min']} candle pertama hari itu")


@fam("fvg", "SMC: fair value gap (retest)", {"fv_w": [5, 10, 20, 50], "trend": T4})
class _:
    def py(p, cc):
        bull, bear = cc.get(("fvg", p["fv_w"]), lambda: ind.fvg_signals(cc.h, cc.l, cc.c, p["fv_w"]))
        return bull, bear, bear, bull, {}

    def pine(p):
        W = p["fv_w"]
        return ["var float fbT = na", "var float fbB = na", "var int fbA = -1", "var float fsT = na", "var float fsB = na", "var int fsA = -1",
                "bool fvBull = false", "bool fvBear = false",
                "if fbA >= 0", f"    if bar_index - fbA > {W} or close < fbB", "        fbA := -1",
                "    else if low <= fbT and close > fbB", "        fvBull := true", "        fbA := -1",
                "if fsA >= 0", f"    if bar_index - fsA > {W} or close > fsT", "        fsA := -1",
                "    else if high >= fsB and close < fsT", "        fvBear := true", "        fsA := -1",
                "if bar_index >= 2", "    if low > high[2]", "        fbT := low", "        fbB := high[2]", "        fbA := bar_index",
                "    if high < low[2]", "        fsT := low[2]", "        fsB := high", "        fsA := bar_index",
                "longEntry = fvBull", "longExit = fvBear", "shortEntry = fvBear", "shortExit = fvBull"]

    def desc(p):
        return (f"Fair value gap bullish = celah antara high 2 candle lalu dan low candle sekarang. Beli saat harga kembali masuk celah itu "
                f"dan tutup di atas dasarnya (maks {p['fv_w']} candle setelah celah terbentuk). Short kebalikannya")


@fam("bos", "SMC: break of structure / order block", {"bs_r": (2, 6), "bs_mode": [0, 1], "bs_w": [10, 20, 50], "trend": T4})
class _:
    def py(p, cc):
        bull, bear = cc.get(("bos", p["bs_r"], p["bs_mode"], p["bs_w"]),
                            lambda: ind.bos_signals(cc.o, cc.h, cc.l, cc.c, p["bs_r"], p["bs_mode"], p["bs_w"]))
        return bull, bear, bear, bull, {}

    def pine(p):
        R = p["bs_r"]; Wd = 2 * R + 1; W = p["bs_w"]
        L = [f"bsHi = ta.highest(high, {Wd})", f"bsLo = ta.lowest(low, {Wd})", "var float bsSH = na", "var float bsSL = na",
             "var float obH = na", "var float obL = na", "var int obA = -1", "var float sbH = na", "var float sbL = na", "var int sbA = -1",
             "bool bsBull = false", "bool bsBear = false",
             "if obA >= 0", f"    if bar_index - obA > {W} or close < obL", "        obA := -1",
             "    else if low <= obH and close > obL", "        bsBull := true", "        obA := -1",
             "if sbA >= 0", f"    if bar_index - sbA > {W} or close > sbH", "        sbA := -1",
             "    else if high >= sbL and close < sbH", "        bsBear := true", "        sbA := -1",
             f"if bar_index >= {Wd - 1}", f"    if high[{R}] == bsHi", f"        bsSH := high[{R}]", f"    if low[{R}] == bsLo", f"        bsSL := low[{R}]",
             "if bar_index >= 1",
             "    bsUp = close > bsSH and close[1] <= bsSH", "    bsDn = close < bsSL and close[1] >= bsSL",
             "    if bsUp"]
        if p["bs_mode"] == 0:
            L += ["        bsBull := true"]
        else:
            L += ["        for k = 0 to 20", "            if k > bar_index", "                break",
                  "            if close[k] < open[k]", "                obH := high[k]", "                obL := low[k]", "                obA := bar_index",
                  "                break"]
        L += ["        bsSH := na", "    if bsDn"]
        if p["bs_mode"] == 0:
            L += ["        bsBear := true"]
        else:
            L += ["        for k = 0 to 20", "            if k > bar_index", "                break",
                  "            if close[k] > open[k]", "                sbH := high[k]", "                sbL := low[k]", "                sbA := bar_index",
                  "                break"]
        L += ["        bsSL := na", "longEntry = bsBull", "longExit = bsBear", "shortEntry = bsBear", "shortExit = bsBull"]
        return L

    def desc(p):
        if p["bs_mode"] == 0:
            return (f"Break of structure: beli saat harga tutup menembus swing high terakhir (swing dikonfirmasi {p['bs_r']} candle tiap sisi); "
                    f"short saat menembus swing low terakhir")
        return (f"Order block: setelah break of structure naik, tandai candle merah terakhir sebelum tembusan; beli saat harga kembali ke "
                f"candle itu dan tutup di atas low-nya (maks {p['bs_w']} candle). Short kebalikannya")


@fam("sweep", "SMC: liquidity sweep (stop hunt)", {"sw_n": (10, 100), "trend": T4})
class _:
    def py(p, cc):
        n = p["sw_n"]
        ll = ind.shift(cc.get(("llc", n), lambda: ind.rolling_min(cc.l, n)))
        hh = ind.shift(cc.get(("hhc", n), lambda: ind.rolling_max(cc.h, n)))
        with np.errstate(invalid="ignore"):
            bull = (cc.l < ll) & (cc.c > ll)
            bear = (cc.h > hh) & (cc.c < hh)
        return bull, bear, bear, bull, {f"Low {n}": ll, f"High {n}": hh}

    def pine(p):
        n = p["sw_n"]
        return [f"swLL = ta.lowest(low, {n})[1]", f"swHH = ta.highest(high, {n})[1]",
                "swBull = low < swLL and close > swLL", "swBear = high > swHH and close < swHH",
                "longEntry = swBull", "longExit = swBear", "shortEntry = swBear", "shortExit = swBull"]

    def desc(p):
        return (f"Beli saat harga sempat menembus low terendah {p['sw_n']} candle (menyapu stop loss) tapi tutup kembali di atasnya; "
                f"short kebalikannya di high tertinggi")


@fam("season_hour", "Musiman: jam & hari tertentu", {"se_h": (0, 23), "se_hold": [1, 2, 4, 8], "se_d": [-1, 0, 1, 2, 3, 4], "se_s": [1, -1]},
     intraday=True)
class _:
    def py(p, cc):
        on = (cc.hour == p["se_h"]) & ((p["se_d"] < 0) | (cc.dow == p["se_d"]))
        off = cc.hour == (p["se_h"] + p["se_hold"]) % 24
        z = np.zeros(len(cc.c), np.bool_)
        return (on if p["se_s"] == 1 else z), off, (on if p["se_s"] == -1 else z), off, {}

    def pine(p):
        dcond = "true" if p["se_d"] < 0 else f"seDow == {p['se_d']}"
        return ["seHr = math.floor(time / 3600000) % 24", "seDow = (math.floor(time / 86400000) + 3) % 7",
                f"seOn = seHr == {p['se_h']} and {dcond}", f"seOff = seHr == {(p['se_h'] + p['se_hold']) % 24}",
                f"longEntry = {'seOn' if p['se_s'] == 1 else 'false'}", "longExit = seOff",
                f"shortEntry = {'seOn' if p['se_s'] == -1 else 'false'}", "shortExit = seOff"]

    def desc(p):
        days = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        d = "setiap hari" if p["se_d"] < 0 else f"hanya hari {days[p['se_d']]}"
        return (f"{'Beli' if p['se_s'] == 1 else 'Short'} setelah candle jam {p['se_h']:02d}:00 UTC ({(p['se_h'] + 7) % 24:02d}:00 WIB) tutup, "
                f"{d}; tutup setelah ±{p['se_hold']} jam")


def _dom_after(t_ms, days):
    return np.asarray(pd.to_datetime(t_ms + int(days) * 86_400_000, unit="ms").day)


@fam("season_tom", "Musiman: pergantian bulan", {"tm_last": [0, 1, 2, 3], "tm_first": [1, 2, 3, 5], "tm_s": [1, -1]})
class _:
    def py(p, cc):
        d0 = cc.get(("dom", 0), lambda: _dom_after(cc.t_ms, 0))
        dN = cc.get(("dom", p["tm_last"]), lambda: _dom_after(cc.t_ms, p["tm_last"]))
        tom = ((dN < d0) if p["tm_last"] > 0 else np.zeros(len(d0), np.bool_)) | (d0 <= p["tm_first"])
        prev = np.r_[False, tom[:-1]]
        ent = tom & ~prev; ext = ~tom
        z = np.zeros(len(d0), np.bool_)
        return (ent if p["tm_s"] == 1 else z), ext, (ent if p["tm_s"] == -1 else z), ext, {}

    def pine(p):
        # day of month from Unix time without timezone functions (H. Hinnant's civil-from-days), UTC like the app
        L = ["seDom(t) =>", "    z = math.floor(t / 86400000) + 719468", "    era = math.floor(z / 146097)", "    doe = z - era * 146097",
             "    yoe = math.floor((doe - math.floor(doe / 1460) + math.floor(doe / 36524) - math.floor(doe / 146096)) / 365)",
             "    doy = doe - (365 * yoe + math.floor(yoe / 4) - math.floor(yoe / 100))", "    mp = math.floor((5 * doy + 2) / 153)",
             "    doy - math.floor((153 * mp + 2) / 5) + 1",
             "tmD0 = seDom(time)"]
        last = f"seDom(time + {p['tm_last']} * 86400000) < tmD0" if p["tm_last"] > 0 else "false"
        L += [f"tmOn = {last} or tmD0 <= {p['tm_first']}", "tmEnt = tmOn and not nz(tmOn[1], false)",
              f"longEntry = {'tmEnt' if p['tm_s'] == 1 else 'false'}", "longExit = not tmOn",
              f"shortEntry = {'tmEnt' if p['tm_s'] == -1 else 'false'}", "shortExit = not tmOn"]
        return L

    def desc(p):
        a = f"{p['tm_last']} hari terakhir bulan sampai " if p["tm_last"] > 0 else "awal bulan sampai "
        return f"{'Beli' if p['tm_s'] == 1 else 'Short'} dari {a}hari ke-{p['tm_first']} bulan berikutnya (efek pergantian bulan), lalu keluar"


BH_FIXED = {"sl_mode": 1, "sl_pct": 99.0, "tp_pct": 0.0, "tp_atr": 0.0, "trail_atr": 0.0, "tp1_r": 0.0, "be_r": 0.0, "max_bars": 0,
            "exit_sig": 0, "regime": 0, "sess": 0, "dow": 0, "htf_m": 0, "vol": 0, "rsi_f": 0, "volf": 0, "fund_f": 0, "flow_f": 0, "oi_f": 0}


@fam("bh", "Patokan: beli & tahan", {"trend": [0]}, fixed=BH_FIXED)
class _:
    def py(p, cc):
        n = len(cc.c)
        return np.ones(n, np.bool_), np.zeros(n, np.bool_), np.zeros(n, np.bool_), np.zeros(n, np.bool_), {}

    def pine(p):
        return ["longEntry = true", "longExit = false", "shortEntry = false", "shortExit = false"]

    def desc(p):
        return "Patokan: beli di awal dan tahan (masuk lagi kalau kena stop). Strategi lain sebaiknya mengalahkan ini setelah biaya"


@fam("carry", "Carry funding (terima funding)", {"cr_n": [24, 72, 168], "cr_thr": [0.00005, 0.0001, 0.0002], "trend": T4}, needs=("fund",))
class _:
    def py(p, cc):
        m = cc.get(("fund_sma", p["cr_n"]), lambda: ind.sma(np.nan_to_num(cc.x["fund"], nan=0.0), p["cr_n"]))
        thr = p["cr_thr"]
        with np.errstate(invalid="ignore"):
            se = _cx(m, thr); sx = _cu(m, thr / 2)
            le = _cu(m, -thr); lx = _cx(m, -thr / 2)
        return le, lx, se, sx, {}

    pine = staticmethod(_no_pine)

    def desc(p):
        return (f"Short perpetual saat rata-rata funding {p['cr_n']} candle > {p['cr_thr'] * 100:.3f}% (short menerima funding), long saat "
                f"< −{p['cr_thr'] * 100:.3f}%; keluar saat funding kembali normal")
from . import families_v7  # noqa: E402,F401  (v7 families register themselves in F)
