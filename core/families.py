"""Strategy families. Each family has: a search space, a Python signal function and a Pine block
that compute EXACTLY the same raw signals (checked by tools/pinecheck with PineTS).
Raw signals = (longEntry, longExit, shortEntry, shortExit) before any filter; filters, exits and the
trend EMA are added generically in engine.py / pine.py."""
from __future__ import annotations
import math
import numpy as np
from . import indicators as ind

R_ = repr  # exact float literal for Pine


def _bool(a):
    return np.nan_to_num(a, nan=0).astype(np.bool_)


def _cx(a, b):
    return ind.crossover(a, b)


def _cu(a, b):
    return ind.crossunder(a, b)


F = {}


def fam(key, name, space, intraday=False, needs_volume=False, sub=None, needs=None):
    def deco(cls):
        F[key] = dict(name=name, space=space, py=cls.py, pine=cls.pine, desc=cls.desc, intraday=intraday,
                      needs_volume=needs_volume, sub=sub, needs=set(needs or ()))
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
                pos = np.where(d > 0, rmf, 0.0); neg = np.where(d < 0, rmf, 0.0)
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
                "mfPos = mfD > 0 ? mfRaw : 0.0", "mfNeg = mfD < 0 ? mfRaw : 0.0",
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
