"""Complete SL / TP / trailing / trade-management toolkit (v7).

Every stop-loss and take-profit type is turned into per-bar price arrays at the SIGNAL bar j
(the order fills at the open of bar j+1, like Pine strategy()). The simulator in engine.py only
sees these arrays, so new types never change the simulator. Each array has a Pine twin
(pine_lines) that computes the same numbers; tools/pinecheck tests them.

R = distance from the signal close to the initial stop. Partial targets, breakeven, R-multiple
targets, step trailing and "trail after X R" are all measured in R.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from . import indicators as ind

CAPITAL = 10000.0
SL_TYPES = {0: "ATR x", 1: "% harga", 2: "pips / poin tetap", 3: "swing low/high terakhir + buffer", 4: "pivot / struktur terakhir + buffer",
            5: "channel Donchian (ikut bergerak)", 6: "band Bollinger", 7: "band Keltner", 8: "low/high candle sinyal + buffer",
            9: "chandelier (ikut bergerak)", 10: "Parabolic SAR (ikut bergerak)", 11: "garis Supertrend (ikut bergerak)",
            12: "volatilitas (standar deviasi)", 13: "maks rugi per trade (uang)", 14: "tanpa stop loss"}
TP_TYPES = {0: "satuan yang sama dengan SL (ATR / %)", 1: "kelipatan R (risk:reward)", 2: "ATR x", 3: "% harga", 4: "pips / poin tetap",
            5: "swing high/low (S&R) terdekat", 6: "pivot terakhir", 7: "Fibonacci extension", 8: "band seberang (Bollinger)",
            9: "tanpa TP (hanya trailing / sinyal)"}
TRAIL_TYPES = {0: "ATR dari high/low candle", 1: "chandelier (dari puncak sejak masuk)", 2: "% dari puncak", 3: "Parabolic SAR",
               4: "Supertrend", 5: "garis MA (EMA)", 6: "step (naik per X R)", 7: "channel Donchian"}
EOD_TYPES = {0: "tidak", 1: "tutup di akhir hari (00:00 UTC)", 2: "tutup saat New York tutup (21:00 UTC)", 3: "tutup saat London tutup (16:00 UTC)"}
EOD_MIN = {1: 1440, 2: 21 * 60, 3: 16 * 60}
REASONS = {0: "Sinyal", 1: "Stop loss", 2: "Take profit", 3: "Akhir data", 4: "Waktu habis", 5: "Tutup sesi / akhir pekan", 6: "Time stop (tidak untung)"}

# search space (added to engine.EXIT_SPACE). Weighted lists: repeated values are sampled more often.
SPACE = {
    "sl_mode": [0, 0, 0, 1, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    "sl_pip": [10.0, 20.0, 30.0, 50.0, 100.0, 200.0, 500.0], "sl_n": [5, 10, 20, 55], "sl_buf": [0.0, 0.2, 0.5, 1.0],
    "sl_money": [50.0, 100.0, 200.0, 300.0, 500.0],
    "tp_mode": [0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "tp_rr": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0], "tp_pip": [20.0, 50.0, 100.0, 200.0, 500.0], "tp_n": [10, 20, 50], "tp_fib": [1.272, 1.618, 2.0],
    "tp1_f": [0.25, 0.33, 0.5, 0.5], "tp2_r": [0.0, 0.0, 0.0, 2.0, 3.0], "tp2_f": [0.25, 0.33], "be_off": [0.0, 0.0, 0.1, 0.25],
    "trail_mode": [0, 0, 1, 1, 2, 3, 4, 5, 6, 7], "trail_pct": [1.0, 2.0, 3.0, 5.0], "trail_ma": [10, 20, 50], "trail_step": [0.5, 1.0],
    "trail_n": [10, 20], "trail_act": [0.0, 0.0, 0.0, 1.0, 2.0],
    "ts_bars": [0, 0, 0, 0, 6, 12, 24], "eod": [0, 0, 0, 0, 0, 1, 2], "wkend": [0, 0, 0, 1],
    "cool": [0, 0, 0, 3, 6, 12], "max_day": [0, 0, 0, 1, 2, 3], "dloss": [0.0, 0.0, 0.0, 2.0, 3.0], "dprof": [0.0, 0.0, 0.0, 0.0, 3.0, 5.0],
    "reent": [0, 0, 1],
}
DEFAULTS = {"sl_pip": 50.0, "sl_n": 10, "sl_buf": 0.0, "sl_money": 100.0, "tp_mode": 0, "tp_rr": 2.0, "tp_pip": 100.0, "tp_n": 20, "tp_fib": 1.618,
            "tp1_f": 0.5, "tp2_r": 0.0, "tp2_f": 0.25, "be_off": 0.0, "trail_pct": 2.0, "trail_ma": 20, "trail_step": 1.0, "trail_n": 20,
            "trail_act": 0.0, "ts_bars": 0, "eod": 0, "wkend": 0, "cool": 0, "max_day": 0, "dloss": 0.0, "dprof": 0.0, "reent": 0}
INT_KEYS = {"sl_n", "tp_mode", "tp_n", "trail_ma", "trail_n", "ts_bars", "eod", "wkend", "cool", "max_day", "reent"}
PIVOT_LR = 3          # pivots used by SL/TP "pivot" types: 3 bars left, 3 bars right (confirmed 3 bars later)
BB_LEN, BB_K, KC_LEN, KC_K, CH_LEN, SD_LEN, ST_LEN, ST_SL_F = 20, 2.0, 20, 2.0, 22, 20, 10, 3.0
PSAR = (0.02, 0.02, 0.2)


def pip_size(price):
    """Pip / point size class from the price level: FX majors 0.0001, JPY pairs and silver 0.01, gold 0.1, indices/crypto 1."""
    p = float(price)
    if not math.isfinite(p) or p <= 0:
        return 1.0
    if p < 20:
        return 0.0001
    if p < 1000:
        return 0.01
    if p < 5000:
        return 0.1
    return 1.0


def fix(p):
    """Keep the exit settings consistent (called from engine.fix_params)."""
    q = p
    if q.get("tp1_r", 0) <= 0:
        q["tp2_r"] = 0.0
    if q.get("tp2_r", 0) > 0 and q["tp2_r"] <= q.get("tp1_r", 0):
        q["tp2_r"] = 0.0
    if q.get("tp_mode", 0) == 1:
        if q.get("tp1_r", 0) >= q.get("tp_rr", 2.0):
            q["tp1_r"] = 0.0; q["tp2_r"] = 0.0
        if q.get("tp2_r", 0) >= q.get("tp_rr", 2.0):
            q["tp2_r"] = 0.0
    if q.get("tp1_f", 0.5) + (q.get("tp2_f", 0.25) if q.get("tp2_r", 0) > 0 else 0) >= 0.95:
        q["tp2_f"] = 0.25; q["tp1_f"] = min(q.get("tp1_f", 0.5), 0.5)
    if q.get("tp_mode", 0) == 9 and q.get("trail_atr", 0) <= 0 and not q.get("exit_sig", 0) and q.get("max_bars", 0) <= 0:
        q["exit_sig"] = 1          # "no TP" needs some way out besides the stop
    return q


# ------------------------------------------------------------------ python levels
def _get(cc, key, fn):
    return cc.get(("x7",) + key, fn)


def _pivot_levels(cc):
    L = R = PIVOT_LR
    def lv(x, roll):
        ref = ind.shift(x, R)
        cond = roll(x, L + R + 1) == ref
        return pd.Series(np.where(cond, ref, np.nan)).ffill().to_numpy()
    return _get(cc, ("pvh",), lambda: lv(cc.h, ind.rolling_max)), _get(cc, ("pvl",), lambda: lv(cc.l, ind.rolling_min))


def _bb(cc):
    def f():
        m = ind.sma(cc.c, BB_LEN); sd = ind.stdev(cc.c, BB_LEN)
        return m + BB_K * sd, m - BB_K * sd
    return _get(cc, ("bb",), f)


def _psar(cc):
    return _get(cc, ("psar",), lambda: ind.psar(cc.h, cc.l, cc.c, *PSAR)[1])


def _st(cc, f):
    def g():
        a = ind.atr(cc.h, cc.l, cc.c, ST_LEN)
        ub, db, d = ind.supertrend(cc.h, cc.l, cc.c, a, f)
        return np.where(d == 1, ub, np.where(d == -1, db, np.nan))
    return _get(cc, ("st", float(f)), g)


def levels(p, cc, lev=1.0):
    """Returns dict of arrays (length n) used by the simulator:
    RL/RS  initial stop distance for a long/short signal at bar j (R), NaN = cannot trade yet
    tpL/tpS final target price (NaN = none), stL/stS trailing SL line (NaN = none),
    trL/trS trailing-mode line (array trailing types), fx = force exit at next open, nb = no new entries."""
    c, h, l, atr = cc.c, cc.h, cc.l, cc.atr
    n = len(c)
    sm = int(p.get("sl_mode", 0))
    nan = np.full(n, np.nan)
    stL = stS = nan
    with np.errstate(invalid="ignore"):
        if sm == 0:
            RL = p["sl_atr"] * atr; RS = RL
        elif sm == 1:
            RL = p["sl_pct"] * cc.pct_unit; RS = RL
        elif sm == 2:
            RL = np.where(np.isnan(atr), np.nan, p["sl_pip"] * pip_size(c[-1])); RS = RL
        elif sm in (3, 5):
            N = int(p["sl_n"])
            lo = _get(cc, ("min", N), lambda: ind.rolling_min(l, N)); hi = _get(cc, ("max", N), lambda: ind.rolling_max(h, N))
            b = p["sl_buf"] if sm == 3 else 0.0
            RL = c - (lo - b * atr); RS = (hi + b * atr) - c
            if sm == 5:
                stL, stS = lo, hi
        elif sm == 4:
            ph, pl = _pivot_levels(cc)
            RL = c - (pl - p["sl_buf"] * atr); RS = (ph + p["sl_buf"] * atr) - c
        elif sm == 6:
            up, dn = _bb(cc)
            RL = c - dn; RS = up - c
        elif sm == 7:
            m = _get(cc, ("ema", KC_LEN), lambda: ind.ema(c, KC_LEN))
            RL = c - (m - KC_K * atr); RS = (m + KC_K * atr) - c
        elif sm == 8:
            RL = c - (l - p["sl_buf"] * atr); RS = (h + p["sl_buf"] * atr) - c
        elif sm == 9:
            hi = _get(cc, ("max", CH_LEN), lambda: ind.rolling_max(h, CH_LEN)); lo = _get(cc, ("min", CH_LEN), lambda: ind.rolling_min(l, CH_LEN))
            stL = hi - p["sl_atr"] * atr; stS = lo + p["sl_atr"] * atr
            RL = c - stL; RS = stS - c
        elif sm == 10:
            s = _psar(cc)
            RL = c - s; RS = s - c
            stL = stS = s
        elif sm == 11:
            s = _st(cc, ST_SL_F)
            RL = c - s; RS = s - c
            stL = stS = s
        elif sm == 12:
            sd = _get(cc, ("sd", SD_LEN), lambda: ind.stdev(c, SD_LEN))
            RL = p["sl_atr"] * sd; RS = RL
        elif sm == 13:
            RL = c * (p["sl_money"] / CAPITAL) / float(lev); RS = RL
        else:   # 14 = no stop loss; R (for partial targets / breakeven) = 1 ATR
            RL = atr * 1.0; RS = RL
        if sm not in (0, 1, 2, 12, 13, 14):
            fb = p["sl_atr"] * atr                  # level on the wrong side / not ready -> ATR stop
            RL = np.where(np.isnan(RL) | (RL <= 0), fb, RL); RS = np.where(np.isnan(RS) | (RS <= 0), fb, RS)
        # ---- final target
        tm = int(p.get("tp_mode", 0))
        if tm == 0:
            if sm == 1:
                k, u = p["tp_pct"], cc.pct_unit
            else:
                k, u = p["tp_atr"], atr
            tpL = c + k * u if k > 0 else nan; tpS = c - k * u if k > 0 else nan
        elif tm == 1:
            tpL = c + p["tp_rr"] * RL; tpS = c - p["tp_rr"] * RS
        elif tm == 2:
            tpL = c + p["tp_atr"] * atr if p["tp_atr"] > 0 else nan; tpS = c - p["tp_atr"] * atr if p["tp_atr"] > 0 else nan
        elif tm == 3:
            tpL = c + p["tp_pct"] * cc.pct_unit if p["tp_pct"] > 0 else nan; tpS = c - p["tp_pct"] * cc.pct_unit if p["tp_pct"] > 0 else nan
        elif tm == 4:
            ps = pip_size(c[-1])
            tpL = c + p["tp_pip"] * ps; tpS = c - p["tp_pip"] * ps
        elif tm == 5:
            N = int(p["tp_n"])
            tpL = ind.shift(_get(cc, ("max", N), lambda: ind.rolling_max(h, N))); tpS = ind.shift(_get(cc, ("min", N), lambda: ind.rolling_min(l, N)))
        elif tm == 6:
            tpL, tpS = _pivot_levels(cc)
        elif tm == 7:
            N = int(p["tp_n"])
            hi = _get(cc, ("max", N), lambda: ind.rolling_max(h, N)); lo = _get(cc, ("min", N), lambda: ind.rolling_min(l, N))
            tpL = lo + p["tp_fib"] * (hi - lo); tpS = hi - p["tp_fib"] * (hi - lo)
        elif tm == 8:
            tpL, tpS = _bb(cc)
        else:
            tpL = tpS = nan
        tpL = np.where(tpL > c, tpL, np.nan); tpS = np.where(tpS < c, tpS, np.nan)
        # ---- array trailing types
        trL = trS = nan
        tmode = int(p.get("trail_mode", 0))
        if p.get("trail_atr", 0) > 0:
            if tmode == 3:
                trL = trS = _psar(cc)
            elif tmode == 4:
                trL = trS = _st(cc, p["trail_atr"])
            elif tmode == 5:
                N = int(p["trail_ma"]); trL = trS = _get(cc, ("ema", N), lambda: ind.ema(c, N))
            elif tmode == 7:
                N = int(p["trail_n"])
                trL = _get(cc, ("min", N), lambda: ind.rolling_min(l, N)); trS = _get(cc, ("max", N), lambda: ind.rolling_max(h, N))
    fx, nb = session_flags(p, cc)
    return dict(RL=np.ascontiguousarray(RL, float), RS=np.ascontiguousarray(RS, float), tpL=np.ascontiguousarray(tpL, float),
                tpS=np.ascontiguousarray(tpS, float), stL=np.ascontiguousarray(stL, float), stS=np.ascontiguousarray(stS, float),
                trL=np.ascontiguousarray(trL, float), trS=np.ascontiguousarray(trS, float), fx=fx, nb=nb)


def session_flags(p, cc):
    eod, wk = int(p.get("eod", 0)), int(p.get("wkend", 0))
    if not eod and not wk:
        z = _get(cc, ("zero",), lambda: np.zeros(len(cc.c), np.bool_))
        return z, z
    def f():
        m0 = (cc.t_ms // 60000) % 1440
        m1 = m0 + cc.tf_min
        fx = np.zeros(len(cc.c), np.bool_); nb = np.zeros(len(cc.c), np.bool_)
        if eod:
            S = EOD_MIN[eod]
            fx |= (m0 < S) & (m1 >= S)
        if wk:
            fri = cc.dow == 4
            w = fri & (m0 < 1200) & (m1 >= 1200)
            fx |= w
            nb |= w | (fri & (m0 >= 1200)) | (cc.dow >= 5)
        return fx, nb
    return _get(cc, ("sess", eod, wk), f)


def trail_args(p):
    """(numba trail code, k): 0 none, 1 ATR from bar high/low, 2 chandelier since entry, 3 % from peak (k = 1 - pct/100),
    4 array line, 5 step (k = step in R)."""
    if p.get("trail_atr", 0) <= 0:
        return 0, 0.0
    m = int(p.get("trail_mode", 0))
    if m == 0:
        return 1, float(p["trail_atr"])
    if m == 1:
        return 2, float(p["trail_atr"])
    if m == 2:
        return 3, float(1.0 - p["trail_pct"] / 100.0)
    if m == 6:
        return 5, float(p["trail_step"])
    return 4, 0.0


# ------------------------------------------------------------------ text
def _n(x):
    return f"{x:g}"


def describe(p, lev=1.0):
    sm, tm = int(p.get("sl_mode", 0)), int(p.get("tp_mode", 0))
    sl = {0: f"Stop loss {_n(p['sl_atr'])} x ATR", 1: f"Stop loss {_n(p['sl_pct'])}%", 2: f"Stop loss {_n(p['sl_pip'])} pips/poin",
          3: f"Stop loss di bawah swing low {p['sl_n']} candle (di atas swing high untuk short) + {_n(p['sl_buf'])} x ATR",
          4: f"Stop loss di pivot terakhir ({PIVOT_LR}/{PIVOT_LR}) + {_n(p['sl_buf'])} x ATR",
          5: f"Stop loss di channel Donchian {p['sl_n']} (ikut naik/turun)", 6: f"Stop loss di band Bollinger ({BB_LEN}, {_n(BB_K)})",
          7: f"Stop loss di band Keltner (EMA {KC_LEN} ± {_n(KC_K)} ATR)", 8: f"Stop loss di low/high candle sinyal + {_n(p['sl_buf'])} x ATR",
          9: f"Stop loss chandelier ({CH_LEN} candle, {_n(p['sl_atr'])} x ATR, ikut bergerak)", 10: "Stop loss di Parabolic SAR (ikut bergerak)",
          11: f"Stop loss di garis Supertrend ({ST_LEN}, {_n(ST_SL_F)}) (ikut bergerak)",
          12: f"Stop loss {_n(p['sl_atr'])} x standar deviasi {SD_LEN} candle", 13: f"Stop loss maks rugi {_n(p['sl_money'])} per {CAPITAL:,.0f} modal",
          14: "TANPA stop loss (berbahaya)"}[sm]
    if sm not in (0, 1, 2, 12, 13, 14):
        sl += f" (kalau level tidak valid: {_n(p['sl_atr'])} x ATR)"
    if tm == 0:
        k = p["tp_pct"] if sm == 1 else p["tp_atr"]
        tp = (f"take profit {_n(k)}{'%' if sm == 1 else ' x ATR'}" if k > 0 else "tanpa take profit tetap")
    else:
        tp = {1: f"take profit {_n(p['tp_rr'])}R (risk:reward 1:{_n(p['tp_rr'])})", 2: f"take profit {_n(p['tp_atr'])} x ATR",
              3: f"take profit {_n(p['tp_pct'])}%", 4: f"take profit {_n(p['tp_pip'])} pips/poin",
              5: f"take profit di swing high/low {p['tp_n']} candle sebelumnya", 6: "take profit di pivot terakhir",
              7: f"take profit Fibonacci extension {_n(p['tp_fib'])} dari swing {p['tp_n']} candle", 8: "take profit di band Bollinger seberang",
              9: "tanpa take profit"}[tm]
        if tm in (5, 6, 7, 8):
            tp += " (kalau level tidak di depan harga: tanpa TP)"
    out = [sl, tp]
    if p.get("tp1_r", 0) > 0:
        out.append(f"TP1 {int(round(p['tp1_f'] * 100))}% di {_n(p['tp1_r'])}R")
        if p.get("tp2_r", 0) > 0:
            out.append(f"TP2 {int(round(p['tp2_f'] * 100))}% di {_n(p['tp2_r'])}R")
    if p.get("be_r", 0) > 0:
        out.append(f"stop ke harga masuk{' + ' + _n(p['be_off']) + 'R' if p.get('be_off', 0) > 0 else ''} setelah {_n(p['be_r'])}R")
    if p.get("trail_atr", 0) > 0:
        m = int(p.get("trail_mode", 0))
        t = {0: f"trailing {_n(p['trail_atr'])} x ATR", 1: f"trailing chandelier {_n(p['trail_atr'])} x ATR",
             2: f"trailing {_n(p['trail_pct'])}% dari puncak", 3: "trailing Parabolic SAR", 4: f"trailing Supertrend ({ST_LEN}, {_n(p['trail_atr'])})",
             5: f"trailing EMA {p['trail_ma']}", 6: f"trailing step {_n(p['trail_step'])}R", 7: f"trailing Donchian {p['trail_n']}"}[m]
        if p.get("trail_act", 0) > 0:
            t += f" (aktif setelah untung {_n(p['trail_act'])}R)"
        out.append(t)
    if p.get("max_bars", 0) > 0:
        out.append(f"keluar paksa setelah {p['max_bars']} candle")
    if p.get("ts_bars", 0) > 0:
        out.append(f"keluar kalau belum untung setelah {p['ts_bars']} candle")
    if p.get("eod", 0):
        out.append(EOD_TYPES[p["eod"]])
    if p.get("wkend", 0):
        out.append("tutup sebelum akhir pekan (Jumat 20:00 UTC)")
    if p.get("exit_sig", 0) == 1:
        out.append("keluar hanya saat sinyal kebalikan")
    g = []
    if p.get("cool", 0):
        g.append(f"jeda {p['cool']} candle setelah rugi")
    if p.get("max_day", 0):
        g.append(f"maks {p['max_day']} trade per hari")
    if p.get("dloss", 0) > 0:
        g.append(f"berhenti hari itu kalau rugi {_n(p['dloss'])}%")
    if p.get("dprof", 0) > 0:
        g.append(f"berhenti hari itu kalau untung {_n(p['dprof'])}%")
    if p.get("reent", 0):
        g.append("tidak masuk lagi ke arah yang sama sampai ada sinyal kebalikan")
    s = ", ".join(out)
    if g:
        s += ". Aturan: " + ", ".join(g)
    return s


# ------------------------------------------------------------------ Pine twin of levels()
def _lit(x):
    return repr(float(x))


def _pine_psar(px):
    S, I, M = (_lit(v) for v in PSAR)
    return [f"var float {px}S = na", f"var float {px}E = na", f"var float {px}A = na", f"var int {px}D = 0",
            "if bar_index == 1",
            f"    {px}D := close >= close[1] ? 1 : -1", f"    {px}S := {px}D == 1 ? low[1] : high[1]",
            f"    {px}E := {px}D == 1 ? high : low", f"    {px}A := {S}",
            "else if bar_index > 1",
            f"    {px}S := {px}S + {px}A * ({px}E - {px}S)",
            f"    if {px}D == 1",
            f"        {px}S := math.min({px}S, low[1], low[2])",
            f"        if low < {px}S",
            f"            {px}D := -1", f"            {px}S := {px}E", f"            {px}E := low", f"            {px}A := {S}",
            f"        else if high > {px}E",
            f"            {px}E := high", f"            {px}A := math.min({px}A + {I}, {M})",
            "    else",
            f"        {px}S := math.max({px}S, high[1], high[2])",
            f"        if high > {px}S",
            f"            {px}D := 1", f"            {px}S := {px}E", f"            {px}E := high", f"            {px}A := {S}",
            f"        else if low < {px}E",
            f"            {px}E := low", f"            {px}A := math.min({px}A + {I}, {M})"]


def _pine_st(px, f):
    return [f"{px}Atr = ta.atr({ST_LEN})", f"{px}Bu = hl2 - {_lit(f)} * {px}Atr", f"{px}Bd = hl2 + {_lit(f)} * {px}Atr",
            f"var float {px}Up = na", f"var float {px}Dn = na", f"var int {px}Dir = 0",
            f"if not na({px}Atr)",
            f"    {px}Up := na({px}Up) or close[1] <= {px}Up ? {px}Bu : math.max({px}Bu, {px}Up)",
            f"    {px}Dn := na({px}Dn) or close[1] >= {px}Dn ? {px}Bd : math.min({px}Bd, {px}Dn)",
            f"    {px}Dir := {px}Dir == 0 ? 1 : {px}Dir == 1 and close < {px}Up ? -1 : {px}Dir == -1 and close > {px}Dn ? 1 : {px}Dir",
            f"{px}Line = {px}Dir == 1 ? {px}Up : {px}Dir == -1 ? {px}Dn : na"]


def pine_lines(p, lev=1.0, pip=1.0, tf_min=60):
    """Pine code that defines the same per-bar series as levels(): xRL, xRS, xTpL, xTpS, xStL, xStS, xTrL, xTrS, xFx, xNb."""
    sm, tm = int(p.get("sl_mode", 0)), int(p.get("tp_mode", 0))
    L = ["// ---- stop loss / take profit (jarak R dari close candle sinyal)"]
    need = set()
    if sm == 4 or tm == 6:
        need.add("pv")
    if sm == 6 or tm == 8:
        need.add("bb")
    if sm == 10 or (p.get("trail_atr", 0) > 0 and int(p.get("trail_mode", 0)) == 3):
        need.add("ps")
    if "pv" in need:
        R_ = PIVOT_LR
        L += [f"xPvHc = high[{R_}] == ta.highest(high, {2 * R_ + 1})", f"xPvLc = low[{R_}] == ta.lowest(low, {2 * R_ + 1})",
              "var float xPvH = na", "var float xPvL = na", "if xPvHc", f"    xPvH := high[{R_}]", "if xPvLc", f"    xPvL := low[{R_}]"]
    if "bb" in need:
        L += [f"xBbM = ta.sma(close, {BB_LEN})", f"xBbSd = ta.stdev(close, {BB_LEN})",
              f"xBbUp = xBbM + {_lit(BB_K)} * xBbSd", f"xBbDn = xBbM - {_lit(BB_K)} * xBbSd"]
    if "ps" in need:
        L += _pine_psar("xPs")
    sa, sp_ = _lit(p["sl_atr"]), _lit(p["sl_pct"])
    L += ["float xStL = na", "float xStS = na"]
    if sm == 0:
        L += [f"float xRL = {sa} * atr", "float xRS = xRL"]
    elif sm == 1:
        L += [f"float xRL = {sp_} * (close * 0.01)", "float xRS = xRL"]
    elif sm == 2:
        L += [f"float xRL = na(atr) ? na : {_lit(p['sl_pip'] * pip)}", "float xRS = xRL"]
    elif sm in (3, 5):
        N = int(p["sl_n"]); b = _lit(p["sl_buf"] if sm == 3 else 0.0)
        L += [f"xLo = ta.lowest(low, {N})", f"xHi = ta.highest(high, {N})",
              f"float xRL = close - (xLo - {b} * atr)", f"float xRS = (xHi + {b} * atr) - close"]
        if sm == 5:
            L += ["xStL := xLo", "xStS := xHi"]
    elif sm == 4:
        b = _lit(p["sl_buf"])
        L += [f"float xRL = close - (xPvL - {b} * atr)", f"float xRS = (xPvH + {b} * atr) - close"]
    elif sm == 6:
        L += ["float xRL = close - xBbDn", "float xRS = xBbUp - close"]
    elif sm == 7:
        L += [f"xKcM = ta.ema(close, {KC_LEN})", f"float xRL = close - (xKcM - {_lit(KC_K)} * atr)", f"float xRS = (xKcM + {_lit(KC_K)} * atr) - close"]
    elif sm == 8:
        b = _lit(p["sl_buf"])
        L += [f"float xRL = close - (low - {b} * atr)", f"float xRS = (high + {b} * atr) - close"]
    elif sm == 9:
        L += [f"xChH = ta.highest(high, {CH_LEN})", f"xChL = ta.lowest(low, {CH_LEN})",
              f"xStL := xChH - {sa} * atr", f"xStS := xChL + {sa} * atr", "float xRL = close - xStL", "float xRS = xStS - close"]
    elif sm == 10:
        L += ["float xRL = close - xPsS", "float xRS = xPsS - close", "xStL := xPsS", "xStS := xPsS"]
    elif sm == 11:
        L += _pine_st("xSs", ST_SL_F) + ["float xRL = close - xSsLine", "float xRS = xSsLine - close", "xStL := xSsLine", "xStS := xSsLine"]
    elif sm == 12:
        L += [f"xSd = ta.stdev(close, {SD_LEN})", f"float xRL = {sa} * xSd", "float xRS = xRL"]
    elif sm == 13:
        L += [f"float xRL = close * {_lit(p['sl_money'] / CAPITAL)} / {_lit(lev)}", "float xRS = xRL"]
    else:
        L += ["float xRL = atr * 1.0", "float xRS = xRL"]
    if sm not in (0, 1, 2, 12, 13, 14):
        L += [f"xRL := na(xRL) or xRL <= 0 ? {sa} * atr : xRL", f"xRS := na(xRS) or xRS <= 0 ? {sa} * atr : xRS"]
    # final target
    if tm == 0:
        k = p["tp_pct"] if sm == 1 else p["tp_atr"]
        u = "(close * 0.01)" if sm == 1 else "atr"
        L += ([f"float xTpL = close + {_lit(k)} * {u}", f"float xTpS = close - {_lit(k)} * {u}"] if k > 0 else ["float xTpL = na", "float xTpS = na"])
    elif tm == 1:
        L += [f"float xTpL = close + {_lit(p['tp_rr'])} * xRL", f"float xTpS = close - {_lit(p['tp_rr'])} * xRS"]
    elif tm == 2:
        k = p["tp_atr"]
        L += ([f"float xTpL = close + {_lit(k)} * atr", f"float xTpS = close - {_lit(k)} * atr"] if k > 0 else ["float xTpL = na", "float xTpS = na"])
    elif tm == 3:
        k = p["tp_pct"]
        L += ([f"float xTpL = close + {_lit(k)} * (close * 0.01)", f"float xTpS = close - {_lit(k)} * (close * 0.01)"] if k > 0 else ["float xTpL = na", "float xTpS = na"])
    elif tm == 4:
        L += [f"float xTpL = close + {_lit(p['tp_pip'] * pip)}", f"float xTpS = close - {_lit(p['tp_pip'] * pip)}"]
    elif tm == 5:
        N = int(p["tp_n"])
        L += [f"float xTpL = ta.highest(high, {N})[1]", f"float xTpS = ta.lowest(low, {N})[1]"]
    elif tm == 6:
        L += ["float xTpL = xPvH", "float xTpS = xPvL"]
    elif tm == 7:
        N = int(p["tp_n"]); f = _lit(p["tp_fib"])
        L += [f"xFbH = ta.highest(high, {N})", f"xFbL = ta.lowest(low, {N})",
              f"float xTpL = xFbL + {f} * (xFbH - xFbL)", f"float xTpS = xFbH - {f} * (xFbH - xFbL)"]
    elif tm == 8:
        L += ["float xTpL = xBbUp", "float xTpS = xBbDn"]
    else:
        L += ["float xTpL = na", "float xTpS = na"]
    L += ["xTpL := xTpL > close ? xTpL : na", "xTpS := xTpS < close ? xTpS : na"]
    # array trailing lines
    L += ["float xTrL = na", "float xTrS = na"]
    if p.get("trail_atr", 0) > 0:
        m = int(p.get("trail_mode", 0))
        if m == 3:
            L += ["xTrL := xPsS", "xTrS := xPsS"]
        elif m == 4:
            L += _pine_st("xTs", p["trail_atr"]) + ["xTrL := xTsLine", "xTrS := xTsLine"]
        elif m == 5:
            L += [f"xTrL := ta.ema(close, {int(p['trail_ma'])})", "xTrS := xTrL"]
        elif m == 7:
            N = int(p["trail_n"])
            L += [f"xTrL := ta.lowest(low, {N})", f"xTrS := ta.highest(high, {N})"]
    # session flags (UTC from the candle open time)
    eod, wk = int(p.get("eod", 0)), int(p.get("wkend", 0))
    L += ["xDay = math.floor(time / 86400000)", "bool xFx = false", "bool xNb = false"]
    if eod or wk:
        L += ["xM0 = math.floor(time / 60000) % 1440", f"xM1 = xM0 + {int(tf_min)}", "xDow = (xDay + 3) % 7"]
        if eod:
            S = EOD_MIN[eod]
            L += [f"xFx := xFx or (xM0 < {S} and xM1 >= {S})"]
        if wk:
            L += ["xWk = xDow == 4 and xM0 < 1200 and xM1 >= 1200", "xFx := xFx or xWk",
                  "xNb := xWk or (xDow == 4 and xM0 >= 1200) or xDow >= 5"]
    return L
