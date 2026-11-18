"""Strategy families, bar-by-bar simulator (numba), metrics, search loop,
walk-forward consistency, parameter stability and Monte Carlo."""
from __future__ import annotations
import math, os, time
import numpy as np
import pandas as pd
from numba import njit
from . import indicators as ind
from . import exits as X

# ------------------------------------------------------------------ families (see core/families.py)
from .families import F as FAMILIES

# exits (all searchable). sl_mode 0 = ATR based, 1 = percent of price
EXIT_SPACE = {"sl_mode": [0, 0, 0, 1], "sl_atr": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0], "tp_atr": [0.0, 1.5, 2.0, 3.0, 4.0, 6.0],
              "sl_pct": [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0], "tp_pct": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0],
              "trail_atr": [0.0, 0.0, 2.0, 3.0, 4.0], "trail_mode": [0, 1],
              "tp1_r": [0.0, 0.0, 0.0, 1.0, 1.5], "be_r": [0.0, 0.0, 0.0, 1.0, 1.5, 2.0],
              "max_bars": [0, 0, 0, 12, 24, 48, 96], "exit_sig": [0, 0, 1]}
EXIT_SPACE.update(X.SPACE)        # v7: all SL / TP / trailing / trade-management types (see core/exits.py)
TP1_FRAC = 0.5
# searchable market-regime filter (every family)
REGIME_SPACE = {"regime": [0, 0, 1, 2], "adx_thr": [20, 25]}
REGIME_LABEL = {0: "tanpa filter", 1: "hanya saat pasar trending", 2: "hanya saat pasar sideways"}
# combo builder: confirmation filters added to the entry signal
FILTER_SPACE = {"sess": [0, 0, 0, 1, 2, 3, 4], "dow": [0, 0, 0, 1, 2, 3], "htf_m": [0, 0, 240, 1440], "htf_len": [20, 50, 100],
                "vol": [0, 0, 1, 2], "rsi_f": [0, 0, 1], "volf": [0, 0, 1],
                # crypto derivatives (only used when the data is loaded; removed otherwise, see drop_unavailable)
                "fund_f": [0, 0, 0, 1], "flow_f": [0, 0, 0, 1], "oi_f": [0, 0, 0, 1, 2]}
DERIV_FILTERS = {"fund_f": "fund", "flow_f": "taker", "oi_f": "oi"}
SESS = {1: (0, 8, "Asia"), 2: (7, 16, "London"), 3: (13, 21, "New York"), 4: (13, 16, "London+NY")}
DOW = {1: "Senin-Jumat saja", 2: "tanpa Senin", 3: "tanpa Jumat"}
DEFAULTS = {"regime": 0, "adx_thr": 20, "sl_mode": 0, "sl_atr": 2.0, "tp_atr": 0.0, "trail_atr": 0.0, "sl_pct": 1.0, "tp_pct": 0.0, "trail_mode": 0, "tp1_r": 0.0, "be_r": 0.0,
            "max_bars": 0, "exit_sig": 0, "sess": 0, "dow": 0, "htf_m": 0, "htf_len": 50, "vol": 0, "rsi_f": 0, "volf": 0, "trend": 0,
            "fund_f": 0, "flow_f": 0, "oi_f": 0}
DEFAULTS.update(X.DEFAULTS)
INT_KEYS = {"regime", "adx_thr", "sl_mode", "trail_mode", "max_bars", "exit_sig", "sess", "dow", "htf_m", "htf_len", "vol", "rsi_f", "volf",
            "fund_f", "flow_f", "oi_f"} | X.INT_KEYS


def full_params(p):
    q = dict(DEFAULTS)
    q.update(p)
    return q


def allowed_families(fams, tf_min, has_volume=True, avail=None):
    out = []
    for f in fams:
        d = FAMILIES.get(f)
        if d is None:
            continue
        if d.get("needs") and not d["needs"] <= set(avail or ()):
            continue
        if d["intraday"] and tf_min > 60:
            continue
        if d["needs_volume"] and not has_volume:
            continue
        out.append(f)
    return out


def drop_unavailable(p, cc):
    """Switch off derivative filters whose data is not loaded for this market."""
    av = getattr(cc, "avail", set())
    for k, need in DERIV_FILTERS.items():
        if p.get(k) and need not in av:
            p[k] = 0
    return p


def uses_deriv(fam, p):
    """True if the strategy needs crypto derivatives data (cannot be exported to Pine)."""
    return bool(FAMILIES[fam].get("needs")) or any(full_params(p).get(k) for k in DERIV_FILTERS)


def fix_params(fam, p, tf_min=60, combo=False):
    """Keep sampled/mutated parameters consistent."""
    for k, v in FAMILIES[fam]["space"].items():
        if isinstance(v, tuple):
            p[k] = int(min(max(int(p[k]), v[0]), v[1]))
    if fam in ("ma_cross", "macd", "ma2", "macd_div") and p["fast"] >= p["slow"] * 0.8:
        p["slow"] = int(max(p["slow"], p["fast"] * 1.5 + 2))
    if fam in ("donchian", "vol_bo") and p["exit_len"] > p["entry_len"]:
        p["exit_len"] = max(3, p["entry_len"] // 2)
    # partial TP only makes sense below the final target
    if p.get("tp_mode", 0) == 0 and p.get("sl_mode", 0) in (0, 1):
        tp_u = p["tp_atr"] if p.get("sl_mode", 0) == 0 else p["tp_pct"]
        sl_u = p["sl_atr"] if p.get("sl_mode", 0) == 0 else p["sl_pct"]
        if p.get("tp1_r", 0) > 0 and tp_u > 0 and p["tp1_r"] * sl_u >= tp_u:
            p["tp1_r"] = 0.0
    X.fix(p)
    if not combo:
        for k in FILTER_SPACE:
            p[k] = DEFAULTS[k]
    if tf_min >= 1440:
        p["sess"] = 0
    if p.get("htf_m", 0) and p["htf_m"] <= tf_min:
        p["htf_m"] = 0
    p.update(FAMILIES[fam].get("fixed") or {})   # e.g. buy & hold: no exits, no filters
    return p


def sample_params(fam, rng, tf_min=60, combo=False):
    p = {}
    for k, v in FAMILIES[fam]["space"].items():
        p[k] = rng.choice(v) if isinstance(v, list) else int(rng.integers(v[0], v[1] + 1))
    for sp in (REGIME_SPACE, EXIT_SPACE, FILTER_SPACE):
        for k, v in sp.items():
            p[k] = rng.choice(v)
    for k in list(p):
        if isinstance(p[k], np.generic):
            p[k] = p[k].item()
        if k in INT_KEYS:
            p[k] = int(p[k])
        elif k in EXIT_SPACE:
            p[k] = float(p[k])
    return fix_params(fam, p, tf_min, combo)


def mutate(fam, p, rng, tf_min=60, combo=False, n_changes=None):
    """Evolutionary search: copy a good parameter set and change 1-3 settings a little."""
    q = dict(p)
    space = dict(FAMILIES[fam]["space"]); space.update(REGIME_SPACE); space.update(EXIT_SPACE)
    if combo:
        space.update(FILTER_SPACE)
    keys = list(space)
    for k in rng.choice(keys, size=n_changes or int(rng.integers(1, 4)), replace=False):
        v = space[k]
        if isinstance(v, tuple):
            q[k] = int(round(q[k] * rng.uniform(0.8, 1.2) + rng.choice([-1, 1])))
        else:
            vals = sorted(set(v), key=lambda x: (str(type(x)), x))
            if q.get(k) in vals and rng.random() < 0.7 and len(vals) > 1:
                i = vals.index(q[k]) + int(rng.choice([-1, 1]))
                q[k] = vals[min(max(i, 0), len(vals) - 1)]
            else:
                q[k] = rng.choice(vals)
            if isinstance(q[k], np.generic):
                q[k] = q[k].item()
    for k in list(q):
        if k in INT_KEYS:
            q[k] = int(q[k])
        elif k in EXIT_SPACE:
            q[k] = float(q[k])
    return fix_params(fam, q, tf_min, combo)


class Cache:
    """Per-dataset indicator cache so thousands of variants stay fast."""
    def __init__(self, df):
        self.o, self.h, self.l, self.c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.v = df["volume"].to_numpy(float) if "volume" in df else np.zeros(len(df))
        self.has_volume = bool(np.nansum(self.v) > 0)
        self.atr = ind.atr(self.h, self.l, self.c, 14)
        self.d = {}
        self.index = df.index
        self.t_ms = np.asarray(pd.DatetimeIndex(df.index).as_unit("ms").asi8, np.int64)
        self.hour = (self.t_ms // 3_600_000) % 24
        self.day = self.t_ms // 86_400_000
        self.dow = (self.day + 3) % 7  # 0 = Monday (UTC)
        self.tf_min = int(np.median(np.diff(self.t_ms[: min(len(df), 500)])) // 60000) if len(df) > 2 else 60
        day = np.asarray(df.index.normalize().asi8)
        self.day_end = np.flatnonzero(np.r_[day[1:] != day[:-1], True])
        span_years = max((df.index[-1] - df.index[0]).total_seconds() / (365.25 * 86400), 1e-9)
        self.days_per_year = len(self.day_end) / span_years
        self.bars_per_year = len(df) / span_years
        self.carry_long = np.zeros(len(df))
        self.carry_short = np.zeros(len(df))
        self.resolve = np.zeros(len(df), np.int8)
        self.pct_unit = self.c * 0.01
        # crypto derivatives columns (see core/deriv.py), NaN where not available
        self.x = {}
        for col in ("taker_buy", "fund", "oi", "lsr", "top_lsr"):
            if col in df.columns:
                a = df[col].to_numpy(float)
                if np.isfinite(a).any():
                    self.x[col] = a
        self.avail = set()
        if "taker_buy" in self.x and self.has_volume:
            self.avail.add("taker")
        for col in ("fund", "oi", "lsr"):
            if col in self.x:
                self.avail.add(col)

    # indicator arrays are cached, but capped in bytes (LRU) so long histories fit in ~1 GB of RAM
    MAX_BYTES = int(float(os.environ.get("VELA_INDICATOR_CACHE_MB", "32")) * 1e6)

    def get(self, key, fn):
        d = self.d
        if key in d:
            v = d.pop(key)
            d[key] = v          # mark as recently used
            return v
        v = fn()
        nb = sum(getattr(x, "nbytes", 0) for x in v) if isinstance(v, tuple) else getattr(v, "nbytes", 0)
        self._bytes = getattr(self, "_bytes", 0) + nb
        d[key] = v
        while self._bytes > self.MAX_BYTES and len(d) > 1:
            k0 = next(iter(d))
            v0 = d.pop(k0)
            self._bytes -= sum(getattr(x, "nbytes", 0) for x in v0) if isinstance(v0, tuple) else getattr(v0, "nbytes", 0)
        return v


def _trend(cc, n):
    if not n:
        return None
    return cc.get(("ema", "c", n), lambda: ind.ema(cc.c, n))


def flow_ratio(cc, n):
    """Taker delta over n bars as a share of volume: +1 = all aggressive buying, -1 = all selling."""
    def f():
        d = 2 * cc.x["taker_buy"] - cc.v
        with np.errstate(invalid="ignore", divide="ignore"):
            return ind.sma(d, n) / ind.sma(cc.v, n)
    return cc.get(("flow", n), f)


def fund_z(cc, n):
    def f():
        x = cc.x["fund"]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (x - ind.sma(x, n)) / ind.stdev(x, n)
    return cc.get(("fundz", n), f)


def oi_change(cc, n):
    def f():
        with np.errstate(invalid="ignore", divide="ignore"):
            return cc.x["oi"] / ind.shift(cc.x["oi"], n) - 1
    return cc.get(("oichg", n), f)


def filter_masks(p, cc):
    """Entry confirmations: trend EMA, ADX regime, session, day of week, higher-timeframe trend,
    volatility regime, RSI side, volume. Returns (long_ok, short_ok, lines)."""
    n = len(cc.c)
    L = np.ones(n, bool); S = np.ones(n, bool); lines = {}
    with np.errstate(invalid="ignore"):
        tr = _trend(cc, p.get("trend", 0))
        if tr is not None:
            L &= cc.c > tr; S &= cc.c < tr
            lines[f"Trend EMA {p['trend']}"] = tr
        rg = p.get("regime", 0)
        if rg:
            a = cc.get(("adx",), lambda: ind.adx(cc.h, cc.l, cc.c, cc.atr))
            ok = (a >= p["adx_thr"]) if rg == 1 else (a < p["adx_thr"])
            L &= ok; S &= ok
            lines["_adx"] = a
        if p.get("sess", 0):
            a, b, _ = SESS[p["sess"]]
            ok = (cc.hour >= a) & (cc.hour < b)
            L &= ok; S &= ok
        dw = p.get("dow", 0)
        if dw:
            ok = cc.dow < 5 if dw == 1 else (cc.dow != 0 if dw == 2 else cc.dow != 4)
            L &= ok; S &= ok
        if p.get("htf_m", 0):
            up, dn = cc.get(("htf", p["htf_m"], p["htf_len"]), lambda: ind.htf_trend(cc.t_ms, cc.c, p["htf_m"] * 60000, p["htf_len"]))
            L &= up; S &= dn
        vv = p.get("vol", 0)
        if vv:
            r = cc.get(("volr",), lambda: cc.atr / ind.sma(cc.atr, 100))
            ok = (r > 1) if vv == 1 else (r < 1)
            L &= ok; S &= ok
        if p.get("rsi_f", 0):
            r14 = cc.get(("rsi", 14), lambda: ind.rsi(cc.c, 14))
            L &= r14 > 50; S &= r14 < 50
        if p.get("volf", 0):
            vm = cc.get(("vsma", 20), lambda: ind.sma(cc.v, 20))
            ok = cc.v > vm
            L &= ok; S &= ok
        av = getattr(cc, "avail", set())
        if p.get("fund_f", 0) and "fund" in av:        # skip crowded side: no long when funding is extreme high
            z = fund_z(cc, 300)
            L &= z < 2; S &= z > -2
        if p.get("flow_f", 0) and "taker" in av:       # aggressive flow on the same side over the last 24 bars
            fr = flow_ratio(cc, 24)
            L &= fr > 0; S &= fr < 0
        if p.get("oi_f", 0) and "oi" in av:            # 1 = open interest rising (new money), 2 = falling
            oc = oi_change(cc, 24)
            ok = (oc > 0) if p["oi_f"] == 1 else (oc < 0)
            L &= ok; S &= ok
    return L, S, lines


def signals(fam, p, cc: Cache):
    """Returns (long_entry, long_exit, short_entry, short_exit, lines) evaluated at bar close."""
    p = full_params(p)
    le, lx, se, sx, lines = FAMILIES[fam]["py"](p, cc)
    f = lambda a: np.nan_to_num(np.asarray(a, float), nan=0).astype(np.bool_)
    le, lx, se, sx = f(le), f(lx), f(se), f(sx)
    if p.get("exit_sig", 0) == 1:   # exit only on the opposite raw signal
        lx, sx = se.copy(), le.copy()
    L, S, fl = filter_masks(p, cc)
    lines.update(fl)
    return le & L, lx, se & S, sx, lines


def describe(fam, p, direction="both"):
    p = full_params(p)
    d = {"long": "hanya Long (beli)", "short": "hanya Short (jual)", "both": "Long dan Short"}[direction]
    core = FAMILIES[fam]["desc"](p)
    f = []
    if p.get("trend"):
        f.append(f"searah tren EMA {p['trend']}")
    if p.get("regime"):
        f.append(f"{REGIME_LABEL[p['regime']]} (ADX {'≥' if p['regime'] == 1 else '<'} {p['adx_thr']})")
    if p.get("sess"):
        a, b, nm = SESS[p["sess"]]
        f.append(f"sesi {nm} ({a:02d}-{b:02d} UTC)")
    if p.get("dow"):
        f.append(DOW[p["dow"]])
    if p.get("htf_m"):
        f.append(f"tren {'4 jam' if p['htf_m'] == 240 else 'harian'} searah (EMA {p['htf_len']})")
    if p.get("vol"):
        f.append("hanya saat volatilitas tinggi" if p["vol"] == 1 else "hanya saat volatilitas rendah")
    if p.get("rsi_f"):
        f.append("RSI(14) di sisi yang sama (>50 long, <50 short)")
    if p.get("volf"):
        f.append("volume di atas rata-rata 20 candle")
    if p.get("fund_f"):
        f.append("tidak ikut sisi yang ramai (funding z-score < 2 untuk long, > -2 untuk short)")
    if p.get("flow_f"):
        f.append("order flow taker 24 candle searah (lebih banyak beli agresif untuk long)")
    if p.get("oi_f"):
        f.append("open interest naik 24 candle" if p["oi_f"] == 1 else "open interest turun 24 candle")
    flt = (" Konfirmasi: " + ", ".join(f) + ".") if f else ""
    ex = " " + X.describe(p)
    return f"{core}.{flt}{ex}. Arah: {d}."


def stop_dist(p, cc, lev=1.0):
    """Initial stop distance (price units) for a long / short signal at every bar."""
    lv = X.levels(full_params(p), cc, lev)
    return lv["RL"], lv["RS"]


# ------------------------------------------------------------------ simulator
@njit(cache=True)
def _rr(pos, ep, px, cf, ca):
    if pos == 1:
        return (px * (1 - cf) - ca - (ep * (1 + cf) + ca)) / ep
    return ((ep * (1 - cf) - ca) - (px * (1 + cf) + ca)) / ep


@njit(cache=True)
def _tmax(stop, lv, pos):
    """Move a stop only in the trade's favour (NaN stop = no stop yet)."""
    if np.isnan(lv):
        return stop
    if np.isnan(stop):
        return lv
    return max(stop, lv) if pos == 1 else min(stop, lv)


@njit(cache=True)
def simulate(o, h, l, c, atr, le, lx, se, sx, start, end, allow_long, allow_short,
             RL, RS, tpL, tpS, stL, stS, trL, trS, fx, nb, day,
             sl_none, tp1_r, tp1_f, tp2_r, tp2_f, be_r, be_off, tr_code, tr_k, tr_act, max_bars, ts_bars,
             cool, max_day, dloss, dprof, reent,
             cost_frac, cost_abs, want_equity, lev, carry_l, carry_s, resolve):
    """Bar-by-bar simulator (v7).
    - Decisions use bar j = i-1 (closed); orders fill at the OPEN of bar i (same as Pine strategy() default).
    - Initial stop = c[j] -/+ R (RL / RS arrays from core/exits.py), final target tpL / tpS (NaN = none),
      partial targets TP1 / TP2 at tp1_r / tp2_r x R closing tp1_f / tp2_f of the ORIGINAL size.
    - Intrabar order: stop first (conservative) unless resolve[i] == 1 (1-minute data showed the targets first),
      then TP1, TP2, final target.
    - Bar close: carry, mark-to-market, SL line (stL/stS, e.g. PSAR / Supertrend / Donchian stop), trailing mode
      (tr_code 1 ATR from bar high/low, 2 chandelier since entry, 3 % from peak, 4 line trL/trS, 5 step per tr_k R;
      only after the trade gained tr_act R), breakeven after be_r R (stop to entry + be_off R).
    - Exits at next open: opposite signal (reverse), exit signal, max_bars, time stop (not in profit after ts_bars),
      fx (end of day / session / weekend).
    - Entry rules: nb blocks entries, cooldown after a loss, max entries per day, daily loss limit / profit target
      (realised, incl. costs and leverage, by UTC day of the signal bar), reent = no re-entry in the same direction
      until an opposite signal."""
    n = end - start
    eq_arr = np.empty(n if want_equity else 1)
    tr_ent = np.empty(n, np.int64); tr_ex = np.empty(n, np.int64); tr_dir = np.empty(n, np.int64)
    tr_ep = np.empty(n); tr_xp = np.empty(n); tr_ret = np.empty(n); tr_reason = np.empty(n, np.int64)
    tr_amb = np.zeros(n, np.int64); tr_sl = np.empty(n); tr_tp = np.empty(n); tr_carry = np.zeros(n)
    tr_p1b = np.full(n, -1, np.int64); tr_p1p = np.full(n, np.nan); tr_p2b = np.full(n, -1, np.int64); tr_p2p = np.full(n, np.nan)
    tr_sl0 = np.empty(n)
    nt = 0
    pos = 0; ep = 0.0; ei = 0; stop = np.nan; stop0 = np.nan; tp = np.nan; t1 = np.nan; t2 = np.nan; bel = np.nan; acc = 0.0
    R = 0.0; cref = 0.0
    q = 1.0; real = 0.0; p1b = -1; p1p = np.nan; p2b = -1; p2p = np.nan; hh = -np.inf; ll = np.inf; be_done = False
    dkey = -1; dret = 0.0; ekey = -1; ecnt = 0; last_loss = -1000000000; block_dir = 0
    eq = 1.0; peak = 1.0; maxdd = 0.0
    if want_equity:
        eq_arr[0] = 1.0
    for i in range(start + 1, end):
        j = i - 1
        new_dir = 0
        exit_now = False
        xr = 0
        # entry gate from the state BEFORE anything happens at this open
        if block_dir == 1 and se[j]:
            block_dir = 0
        elif block_dir == -1 and le[j]:
            block_dir = 0
        gate = not nb[j]
        if cool > 0 and j - last_loss < cool:
            gate = False
        if max_day > 0 and day[j] == ekey and ecnt >= max_day:
            gate = False
        if dkey == day[j]:
            if dloss > 0 and dret <= -dloss:
                gate = False
            if dprof > 0 and dret >= dprof:
                gate = False
        timeup = max_bars > 0 and pos != 0 and (j - ei + 1) >= max_bars
        tstop = ts_bars > 0 and pos != 0 and (j - ei + 1) >= ts_bars and pos * (c[j] - ep) <= 0
        if pos == 1:
            if allow_short and se[j]:
                exit_now = True; new_dir = -1
            elif lx[j]:
                exit_now = True
            elif timeup:
                exit_now = True; xr = 4
            elif tstop:
                exit_now = True; xr = 6
            elif fx[j]:
                exit_now = True; xr = 5
        elif pos == -1:
            if allow_long and le[j]:
                exit_now = True; new_dir = 1
            elif sx[j]:
                exit_now = True
            elif timeup:
                exit_now = True; xr = 4
            elif tstop:
                exit_now = True; xr = 6
            elif fx[j]:
                exit_now = True; xr = 5
        else:
            if allow_long and le[j]:
                new_dir = 1
            elif allow_short and se[j]:
                new_dir = -1
        if new_dir != 0 and (not gate or new_dir == block_dir or fx[j]):
            new_dir = 0
        if exit_now:
            px = o[i]
            r = (real + q * _rr(pos, ep, px, cost_frac, cost_abs) - acc) * lev
            eq *= (1 + r)
            tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = xr
            tr_sl[nt] = stop; tr_tp[nt] = tp; tr_carry[nt] = acc; tr_p1b[nt] = p1b; tr_p1p[nt] = p1p; tr_p2b[nt] = p2b; tr_p2p[nt] = p2p
            tr_sl0[nt] = stop0
            nt += 1
            if day[i] != dkey:
                dkey = day[i]; dret = 0.0
            dret += r
            if r < 0:
                last_loss = i
            if reent == 1:
                block_dir = pos
            pos = 0
        if new_dir != 0:
            Rj = RL[j] if new_dir == 1 else RS[j]
            if not np.isnan(atr[j]) and not np.isnan(Rj):
                pos = new_dir; ep = o[i]; ei = i; acc = 0.0; q = 1.0; real = 0.0; p1b = -1; p1p = np.nan; p2b = -1; p2p = np.nan
                hh = -np.inf; ll = np.inf; be_done = False
                R = Rj; cref = c[j]
                stop0 = c[j] - pos * R
                stop = np.nan if sl_none else stop0
                tp = tpL[j] if pos == 1 else tpS[j]
                t1 = c[j] + pos * tp1_r * R if tp1_r > 0 else np.nan
                t2 = c[j] + pos * tp2_r * R if tp2_r > 0 else np.nan
                bel = c[j] + pos * be_r * R if be_r > 0 else np.nan
                if day[j] != ekey:
                    ekey = day[j]; ecnt = 0
                ecnt += 1
        # ---- intrabar stop / partial targets / final target
        if pos != 0:
            has_tp = not np.isnan(tp)
            if pos == 1:
                hs = l[i] <= stop
                h1 = p1b < 0 and tp1_r > 0 and h[i] >= t1
                h2 = p2b < 0 and tp2_r > 0 and h[i] >= t2
                ht = has_tp and h[i] >= tp
                gap = o[i] <= stop
            else:
                hs = h[i] >= stop
                h1 = p1b < 0 and tp1_r > 0 and l[i] <= t1
                h2 = p2b < 0 and tp2_r > 0 and l[i] <= t2
                ht = has_tp and l[i] <= tp
                gap = o[i] >= stop
            amb = 1 if (hs and (h1 or h2 or ht) and not gap) else 0
            targets_first = amb == 1 and resolve[i] == 1
            closed = False
            reason = 0; px = 0.0
            if hs and not targets_first:
                px = min(o[i], stop) if pos == 1 else max(o[i], stop)
                reason = 1; closed = True
            else:
                if h1:
                    p1p = max(o[i], t1) if pos == 1 else min(o[i], t1)
                    real += tp1_f * _rr(pos, ep, p1p, cost_frac, cost_abs)
                    q = 1.0 - tp1_f; p1b = i
                if h2 and p1b >= 0:
                    p2p = max(o[i], t2) if pos == 1 else min(o[i], t2)
                    real += tp2_f * _rr(pos, ep, p2p, cost_frac, cost_abs)
                    q = 1.0 - tp1_f - tp2_f; p2b = i
                if ht:
                    px = max(o[i], tp) if pos == 1 else min(o[i], tp)
                    reason = 2; closed = True
                elif hs:
                    px = stop
                    reason = 1; closed = True
            if closed:
                r = (real + q * _rr(pos, ep, px, cost_frac, cost_abs) - acc) * lev
                eq *= (1 + r)
                tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = reason
                tr_amb[nt] = amb; tr_sl[nt] = stop; tr_tp[nt] = tp
                tr_carry[nt] = acc; tr_p1b[nt] = p1b; tr_p1p[nt] = p1p; tr_p2b[nt] = p2b; tr_p2p[nt] = p2p; tr_sl0[nt] = stop0
                nt += 1
                if day[i] != dkey:
                    dkey = day[i]; dret = 0.0
                dret += r
                if r < 0:
                    last_loss = i
                if reent == 1:
                    block_dir = pos
                pos = 0
        # ---- bar close: carry, mark-to-market, SL line, trailing, breakeven
        if pos != 0:
            if pos == 1:
                acc += carry_l[i] * q
                hh = max(hh, h[i])
                fav = hh - cref
                if not np.isnan(stL[i]) and stL[i] < c[i]:
                    stop = _tmax(stop, stL[i], 1)
            else:
                acc += carry_s[i] * q
                ll = min(ll, l[i])
                fav = cref - ll
                if not np.isnan(stS[i]) and stS[i] > c[i]:
                    stop = _tmax(stop, stS[i], -1)
            m = eq * (1 + lev * (real + q * _rr(pos, ep, c[i], cost_frac, cost_abs) - acc))
            if tr_code > 0 and (tr_act <= 0 or fav >= tr_act * R):
                lv = np.nan
                if tr_code == 1:
                    if not np.isnan(atr[i]):
                        lv = (h[i] - tr_k * atr[i]) if pos == 1 else (l[i] + tr_k * atr[i])
                elif tr_code == 2:
                    if not np.isnan(atr[i]):
                        lv = (hh - tr_k * atr[i]) if pos == 1 else (ll + tr_k * atr[i])
                elif tr_code == 3:
                    lv = hh * tr_k if pos == 1 else ll * (2.0 - tr_k)
                elif tr_code == 4:
                    x = trL[i] if pos == 1 else trS[i]
                    if not np.isnan(x) and ((pos == 1 and x < c[i]) or (pos == -1 and x > c[i])):
                        lv = x
                else:
                    st_ = tr_k * R
                    k_ = math.floor(fav / st_) if st_ > 0 else 0.0
                    if k_ >= 1:
                        lv = stop0 + pos * k_ * st_
                stop = _tmax(stop, lv, pos)
            if be_r > 0 and not be_done and ((pos == 1 and h[i] >= bel) or (pos == -1 and l[i] <= bel)):
                stop = _tmax(stop, ep + pos * be_off * R, pos); be_done = True
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
        r = (real + q * _rr(pos, ep, px, cost_frac, cost_abs) - acc) * lev
        eq *= (1 + r)
        tr_ent[nt] = ei; tr_ex[nt] = i; tr_dir[nt] = pos; tr_ep[nt] = ep; tr_xp[nt] = px; tr_ret[nt] = r; tr_reason[nt] = 3
        tr_sl[nt] = stop; tr_tp[nt] = tp; tr_carry[nt] = acc; tr_p1b[nt] = p1b; tr_p1p[nt] = p1p; tr_p2b[nt] = p2b; tr_p2p[nt] = p2p
        tr_sl0[nt] = stop0
        nt += 1
    return (eq, maxdd, nt, tr_ent[:nt], tr_ex[:nt], tr_dir[:nt], tr_ep[:nt], tr_xp[:nt], tr_ret[:nt], tr_reason[:nt], eq_arr,
            tr_amb[:nt], tr_sl[:nt], tr_tp[:nt], tr_carry[:nt], tr_p1b[:nt], tr_p1p[:nt], tr_p2b[:nt], tr_p2p[:nt], tr_sl0[:nt])


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
    q = full_params(p)
    lv = X.levels(q, cc, lev)
    trc, trk = X.trail_args(q)
    tp2 = float(q["tp2_r"]) if q["tp1_r"] > 0 else 0.0
    res = simulate(cc.o, cc.h, cc.l, cc.c, cc.atr, le, lx, se, sx, int(start), int(end),
                   direction in ("long", "both"), direction in ("short", "both"),
                   lv["RL"], lv["RS"], lv["tpL"], lv["tpS"], lv["stL"], lv["stS"], lv["trL"], lv["trS"], lv["fx"], lv["nb"], cc.day,
                   int(q["sl_mode"]) == 14, float(q["tp1_r"]), float(q["tp1_f"]), tp2, float(q["tp2_f"]), float(q["be_r"]), float(q["be_off"]),
                   int(trc), float(trk), float(q["trail_act"]), int(q["max_bars"]), int(q["ts_bars"]),
                   int(q["cool"]), int(q["max_day"]), float(q["dloss"]) / 100.0, float(q["dprof"]) / 100.0, int(q["reent"]),
                   float(cost_frac), float(cost_abs), want_equity or daily, float(lev), cc.carry_long, cc.carry_short, cc.resolve)
    (eq, maxdd, nt, ent, ex, dr, ep, xp, ret, reason, eqa, amb, sl_, tp_, carry, p1b, p1p, p2b, p2p, sl0) = res
    m = metrics(ret, eq, maxdd)
    tr = dict(ent=ent, ex=ex, dir=dr, ep=ep, xp=xp, ret=ret, reason=reason, equity=eqa, amb=amb, sl=sl_, tp=tp_, carry=carry,
              p1b=p1b, p1p=p1p, p2b=p2b, p2p=p2p, sl0=sl0)
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
    """Nearby parameter sets (each number nudged ~15%, stop/target nudged) used for the stability check."""
    p = full_params(p)
    out = []
    for k, v in FAMILIES[fam]["space"].items():
        if isinstance(v, tuple):
            for f in (0.85, 1.15):
                q = dict(p); nv = int(round(p[k] * f))
                if nv == p[k]:
                    nv = p[k] + (1 if f > 1 else -1)
                q[k] = max(2, nv)
                out.append(q)
    sk, tk, step = ("sl_pct", "tp_pct", 0.25) if p["sl_mode"] == 1 else ("sl_atr", "tp_atr", 0.5)
    for d in (-1, 1):
        q = dict(p); q[sk] = max(step, p[sk] + d * step * max(1, p[sk] // 2)); out.append(q)
    if p[tk] > 0:
        for d in (-1, 1):
            q = dict(p); q[tk] = max(step, p[tk] + d * step * max(1, p[tk] // 2)); out.append(q)
    return out


def stability(fam, p, cc, start, end, direction, cf, ca, lev=1.0):
    pfs = []
    for q in neighbours(fam, p):
        if fam in ("ma_cross", "macd", "ma2") and q["fast"] >= q["slow"]:
            continue
        if fam in ("donchian", "vol_bo") and q["exit_len"] > q["entry_len"]:
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
    min_pf, min_wr, max_dd, min_trades, min_wf, min_stab, oos_min_pf, seed,
    mode ("ketat" | "vela"), algo ("random" | "evo"), combo (bool)."""
    t0 = time.time()
    cc = cc if cc is not None else Cache(df)
    n = len(df)
    split = int(n * (1 - cfg["oos_pct"] / 100.0))
    warm = 260  # bars of indicator warm-up
    is_start = min(warm, split // 4)
    rng = np.random.default_rng(cfg.get("seed", 42))
    tf_min = cfg.get("tf_min", cc.tf_min)
    combo = bool(cfg.get("combo", False))
    mode = cfg.get("mode", "ketat")
    algo = cfg.get("algo", "random")
    fams = allowed_families(cfg["families"], tf_min, cc.has_volume, getattr(cc, "avail", set()))
    skipped_fams = [f for f in cfg["families"] if f not in fams]
    if not fams:
        raise ValueError("Tidak ada keluarga strategi yang cocok untuk timeframe / data ini.")
    lev = cfg.get("lev", 1.0)
    rows, near, seen = [], [], set()
    tested = 0
    trial_sr = []            # daily Sharpe of EVERY trial (Deflated Sharpe Ratio counts all of them)
    pbo_keep = cfg.get("pbo_trials", 800)
    trial_daily = []
    elite = []               # (score, fam, params) for the evolutionary search
    warmup = max(200, int(cfg["n_variants"] * 0.25))
    n_elite = 20
    k = 0; misses = 0
    while k < cfg["n_variants"] and misses < 5000:
        if time.time() - t0 > cfg["time_budget"]:
            break
        if algo == "evo" and tested >= warmup and elite and rng.random() < 0.7:
            _, fam, par = elite[int(rng.integers(0, len(elite)))]
            p = drop_unavailable(mutate(fam, par, rng, tf_min, combo), cc)
        else:
            fam = fams[k % len(fams)]
            p = drop_unavailable(sample_params(fam, rng, tf_min, combo), cc)
        key = (fam, tuple(sorted(p.items())))
        if key in seen:
            misses += 1
            continue
        seen.add(key); k += 1
        m, tr = run(fam, p, cc, is_start, split, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=lev, daily=True)
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
        if algo == "evo" and m["trades"] >= cfg["min_trades"] and m["net"] > 0:
            # fitness on in-sample data only (OOS never feeds the search)
            score = m["ret_dd"] * min(1.0, m["pf"] / max(cfg["min_pf"], 1e-9))
            if len(elite) < n_elite or score > elite[-1][0]:
                elite.append((score, fam, p))
                elite.sort(key=lambda e: e[0], reverse=True)
                del elite[n_elite:]
        if progress and k % 25 == 0:
            progress(min(1.0, k / cfg["n_variants"]), tested, len(rows))
    t_search = time.time() - t0
    rows.sort(key=lambda r: r["is_m"]["ret_dd"], reverse=True)
    n_is_pass = len(rows)
    # diversity: max 3 variants with the same signal settings (differing only in exits/filters) go to stage 2,
    # so the evolutionary search cannot fill the list with near-copies
    max_deep = cfg.get("max_deep", 300 if mode == "vela" else 150)
    keep, cnt = [], {}
    for r in rows:
        ck = (r["fam"], tuple((k, r["params"][k]) for k in FAMILIES[r["fam"]]["space"]))
        if cnt.get(ck, 0) >= 3:
            continue
        cnt[ck] = cnt.get(ck, 0) + 1
        keep.append(r)
        if len(keep) >= max_deep:
            break
    rows = keep
    near.sort(key=lambda r: r["is_m"]["ret_dd"], reverse=True)
    rows += near[: max(0, 20 - len(rows))]
    out = []
    for r in rows:
        if time.time() - t0 > cfg["time_budget"] * 2.5 and out:
            break   # hard time cap on stage 2 (cloud)
        om, _ = run(r["fam"], r["params"], cc, split, n, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=lev)
        ism = r["is_m"]
        min_oos_tr = max(3, cfg["min_trades"] * cfg["oos_pct"] / max(1, 100 - cfg["oos_pct"]) * 0.3)
        oos_ok = om["pf"] >= cfg["oos_min_pf"] and om["maxdd"] <= cfg["max_dd"] and om["trades"] >= min_oos_tr
        if mode == "vela":
            stab, stab_pf, overfit = float("nan"), float("nan"), False
            passed = r["is_pass"] and oos_ok
        else:
            stab, stab_pf = stability(r["fam"], r["params"], cc, is_start, split, cfg["direction"], cfg["cost_frac"],
                                      cfg["cost_abs"], lev=lev)
            overfit = (om["trades"] > 0 and om["pf"] < 0.6 * ism["pf"]) or om["net"] < 0
            passed = r["is_pass"] and oos_ok and r["wf"] >= cfg["min_wf"] and stab >= cfg["min_stab"] and not overfit
        out.append(dict(fam=r["fam"], params=r["params"], is_m=ism, oos_m=om, wf=r["wf"], stab=stab,
                        stab_pf=stab_pf, overfit=overfit, passed=passed, is_pass=r["is_pass"], oos_ok=oos_ok))
    meta = dict(tested=tested, is_pass=n_is_pass, seconds=time.time() - t0, search_seconds=t_search,
                split=split, is_start=is_start, n=n, split_time=df.index[split], start_time=df.index[is_start],
                end_time=df.index[-1], trial_sr=np.array(trial_sr), mode=mode, algo=algo, combo=combo,
                skipped_families=skipped_fams, n_deep=len(out),
                trial_daily=np.vstack(trial_daily) if trial_daily else np.zeros((0, 0), np.float32))
    return out, meta, cc
