"""Multi-leg position engine: grid, DCA / safety orders (averaging down), martingale / anti-martingale,
pyramiding (Turtle-style scale-in), hedging (long + short at the same time) and zone recovery,
with basket close by total profit/loss, realistic costs, margin and stop-out (account blow-up) detection.

Account model (same for every mode):
  * start capital 1.0 (shown as 10,000 in the UI), cash accounting: equity = cash + net qty x price;
  * every fill pays cost_frac x notional + cost_abs x qty (spread/slippage/commission from the sidebar);
  * swap / funding: carry_long / carry_short (fraction of notional per bar, from realism.py) charged at each bar close;
  * margin used = gross notional / max leverage. If equity <= stop_out x margin used (or <= 0) at ANY price point of a
    bar, every leg is closed at that price: a margin call / stop-out. That is reported as an account blow-up.
Inside a candle the price path is assumed O -> L -> H -> C for green candles and O -> H -> L -> C for red ones
(the usual convention; a 1-minute check is not done here). Gaps fill limit orders at the open."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from numba import njit

from . import engine as E, indicators as ind

MODES = {
    "grid": "Grid trading",
    "dca": "DCA / safety order (averaging down, gaya 3Commas)",
    "mart": "Martingale / anti-martingale",
    "pyr": "Pyramiding / scale-in (gaya Turtle)",
    "hedge": "Hedging (long + short bersamaan)",
    "zone": "Zone recovery",
}
TAIL_RISK = {"grid", "dca", "mart", "hedge", "zone"}   # modes whose profits come with hidden tail risk


# ------------------------------------------------------------------ shared account helpers
@njit(cache=True)
def _path(o, h, l, c):
    p = np.empty(4)
    p[0] = o
    if c >= o:
        p[1] = l; p[2] = h
    else:
        p[1] = h; p[2] = l
    p[3] = c
    return p


@njit(cache=True)
def _fee(q, px, cf, ca):
    return abs(q) * px * cf + abs(q) * ca


# ------------------------------------------------------------------ GRID
@njit(cache=True)
def _grid(o, h, l, c, cl, cs, start, end, lo, hi, n, geo, mode, order_frac, lev, so, cf, ca, sl_out, recenter, btp, bsl):
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    tr_b = np.zeros(200000, np.int64); tr_p = np.zeros(200000); nt = 0
    P = np.empty(n + 1)
    kind = np.zeros(n, np.int8)     # +1 long slot (buy P[i] -> sell P[i+1]), -1 short slot (short P[i+1] -> cover P[i])
    state = np.zeros(n, np.int8)    # 0 waiting entry, 1 holding
    q = np.zeros(n); epx = np.zeros(n); efee = np.zeros(n)
    cash = 1.0; QL = 0.0; QS = 0.0
    blown = -1; stop_outs = 0; active = True
    cyc_eq = 1.0
    x0 = o[start]
    lo_r = lo / x0; hi_r = hi / x0
    rebuild = True
    for t in range(start, end):
        if rebuild:
            rebuild = False
            x = o[t]
            if t > start:
                lo = x * lo_r; hi = x * hi_r
            for k in range(n + 1):
                if geo:
                    P[k] = lo * (hi / lo) ** (k / n)
                else:
                    P[k] = lo + (hi - lo) * k / n
            for k in range(n):
                if mode == 0:
                    kind[k] = 1
                elif mode == 1:
                    kind[k] = -1
                else:
                    kind[k] = 1 if P[k + 1] <= x else (-1 if P[k] >= x else 1)
                q[k] = order_frac / (P[k] if kind[k] == 1 else P[k + 1])
                state[k] = 0
                # Binance-style start: slots on the "wrong" side of price are filled at market at the start
                if (kind[k] == 1 and P[k] >= x and mode != 2) or (kind[k] == -1 and P[k + 1] <= x and mode != 2):
                    state[k] = 1; epx[k] = x; f = _fee(q[k], x, cf, ca); efee[k] = f; cash -= f
                    if kind[k] == 1:
                        cash -= q[k] * x; QL += q[k]
                    else:
                        cash += q[k] * x; QS += q[k]
            cyc_eq = cash + (QL - QS) * x
        pts = _path(o[t], h[t], l[t], c[t])
        prev = c[t - 1] if t > start else o[t]
        mn = 1e300
        for j in range(4):
            b = pts[j]
            a = prev if j == 0 else pts[j - 1]
            gap = j == 0
            if active and b != a:
                for k in range(n):
                    if kind[k] == 1:
                        if state[k] == 0 and b < a and P[k] < a and P[k] >= b:
                            px = b if gap else P[k]
                            f = _fee(q[k], px, cf, ca); cash -= q[k] * px + f; QL += q[k]; state[k] = 1; epx[k] = px; efee[k] = f
                        elif state[k] == 1 and b > a and P[k + 1] > a and P[k + 1] <= b:
                            px = b if gap else P[k + 1]
                            f = _fee(q[k], px, cf, ca); cash += q[k] * px - f; QL -= q[k]; state[k] = 0
                            tr_b[nt] = t; tr_p[nt] = q[k] * (px - epx[k]) - f - efee[k]; nt += 1
                    else:
                        if state[k] == 0 and b > a and P[k + 1] > a and P[k + 1] <= b:
                            px = b if gap else P[k + 1]
                            f = _fee(q[k], px, cf, ca); cash += q[k] * px - f; QS += q[k]; state[k] = 1; epx[k] = px; efee[k] = f
                        elif state[k] == 1 and b < a and P[k] < a and P[k] >= b:
                            px = b if gap else P[k]
                            f = _fee(q[k], px, cf, ca); cash -= q[k] * px + f; QS -= q[k]; state[k] = 0
                            tr_b[nt] = t; tr_p[nt] = q[k] * (epx[k] - px) - f - efee[k]; nt += 1
            eqx = cash + (QL - QS) * b
            G = QL + QS
            mn = min(mn, eqx)
            out = sl_out > 0 and (b < lo * (1 - sl_out) or b > hi * (1 + sl_out))
            basket = (btp > 0 and eqx - cyc_eq >= btp) or (bsl > 0 and eqx - cyc_eq <= -bsl)
            stopped = G > 0 and (eqx <= 0 or eqx <= so * G * b / lev)
            if active and (stopped or out or basket):
                # close everything at b
                for k in range(n):
                    if state[k] == 1:
                        f = _fee(q[k], b, cf, ca)
                        if kind[k] == 1:
                            cash += q[k] * b - f; pnl = q[k] * (b - epx[k]) - f - efee[k]
                        else:
                            cash -= q[k] * b + f; pnl = q[k] * (epx[k] - b) - f - efee[k]
                        tr_b[nt] = t; tr_p[nt] = pnl; nt += 1
                        state[k] = 0
                QL = 0.0; QS = 0.0
                eqx = cash; mn = min(mn, eqx)
                if stopped:
                    stop_outs += 1; blown = t; active = False
                elif out and not recenter:
                    active = False
                else:
                    rebuild = True
                break
        # carry at close
        if QL > 0 or QS > 0:
            cash -= QL * c[t] * cl[t] + QS * c[t] * cs[t]
        e = cash + (QL - QS) * c[t]
        eq_c[t - start] = e; eq_m[t - start] = min(mn, e)
        expo[t - start] = (QL + QS) * c[t] / max(e, 1e-12) if e > 0 else 0.0
        cnt = 0
        for k in range(n):
            cnt += state[k]
        legs[t - start] = cnt
        if cash + (QL - QS) * c[t] <= 0 and blown < 0:
            blown = t; active = False
        if not active and not rebuild and QL == 0 and QS == 0 and blown >= 0:
            for r in range(t - start + 1, N):
                eq_c[r] = e; eq_m[r] = e
            break
        if nt > 199000:
            break
    return eq_c, eq_m, expo, legs, tr_b[:nt], tr_p[:nt], blown, stop_outs


# ------------------------------------------------------------------ DCA / SAFETY ORDERS
@njit(cache=True)
def _dca(o, h, l, c, cl, cs, start, end, startsig, d, base, so_f, n_so, dev, sscale, vscale, tp, sl, lev, so, cf, ca, cool, btp, bsl):
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    tr_b = np.zeros(100000, np.int64); tr_p = np.zeros(100000); tr_n = np.zeros(100000, np.int64); nt = 0
    lvl = np.empty(max(n_so, 1)); vol = np.empty(max(n_so, 1))
    cash = 1.0; Q = 0.0; cost_in = 0.0; nfill = 0; inpos = False; base_px = 0.0; last_exit = -10 ** 9
    blown = -1; stop_outs = 0; deal_fee = 0.0; deal_cash0 = 1.0
    for t in range(start, end):
        if not inpos and blown < 0 and t > start and startsig[t - 1] and t - last_exit > cool:
            x = o[t]; base_px = x
            q0 = base / x; f = _fee(q0, x, cf, ca)
            deal_cash0 = cash
            cash -= d * q0 * x + f; Q = d * q0; cost_in = q0 * x; nfill = 0; inpos = True; deal_fee = f
            cum = 0.0; step = dev
            for k in range(n_so):
                cum += step; step *= sscale
                lvl[k] = x * (1 - d * cum)
                vol[k] = so_f * vscale ** k
        pts = _path(o[t], h[t], l[t], c[t])
        mn = 1e300
        if inpos:
            for j in range(4):
                b = pts[j]
                # safety orders (adverse direction)
                while nfill < n_so and ((d == 1 and b <= lvl[nfill]) or (d == -1 and b >= lvl[nfill])):
                    px = b if j == 0 else lvl[nfill]
                    qk = vol[nfill] / lvl[nfill]
                    f = _fee(qk, px, cf, ca)
                    cash -= d * qk * px + f; Q += d * qk; cost_in += qk * px; nfill += 1; deal_fee += f
                avg = cost_in / abs(Q)
                tpx = avg * (1 + d * tp)
                slx = avg * (1 - d * sl) if sl > 0 else (-1.0 if d == 1 else 1e300)
                eqx = cash + Q * b
                mn = min(mn, eqx)
                hit_tp = (d == 1 and b >= tpx) or (d == -1 and b <= tpx)
                hit_sl = sl > 0 and ((d == 1 and b <= slx) or (d == -1 and b >= slx))
                basket = (btp > 0 and eqx - deal_cash0 >= btp) or (bsl > 0 and eqx - deal_cash0 <= -bsl)
                G = abs(Q)
                stopped = eqx <= 0 or eqx <= so * G * b / lev
                if hit_tp or hit_sl or basket or stopped:
                    if stopped:
                        px = b
                    elif hit_sl:
                        px = b if j == 0 else slx
                    elif hit_tp:
                        px = b if j == 0 else tpx
                    else:
                        px = b
                    f = _fee(Q, px, cf, ca)
                    cash += Q * px - f
                    tr_b[nt] = t; tr_p[nt] = cash - deal_cash0; tr_n[nt] = nfill; nt += 1
                    Q = 0.0; inpos = False; last_exit = t
                    mn = min(mn, cash)
                    if stopped:
                        stop_outs += 1; blown = t
                    break
        if inpos:
            cash -= (Q * c[t] * cl[t]) if Q > 0 else (-Q * c[t] * cs[t])
        e = cash + Q * c[t]
        eq_c[t - start] = e; eq_m[t - start] = min(mn, e)
        expo[t - start] = abs(Q) * c[t] / e if e > 0 else 0.0
        legs[t - start] = (1 + nfill) if inpos else 0
        if blown >= 0:
            for r in range(t - start + 1, N):
                eq_c[r] = e; eq_m[r] = e
            break
        if nt > 99000:
            break
    return eq_c, eq_m, expo, legs, tr_b[:nt], tr_p[:nt], tr_n[:nt], blown, stop_outs


# ------------------------------------------------------------------ PYRAMIDING (Turtle style)
@njit(cache=True)
def _pyr(o, h, l, c, atr, cl, cs, start, end, le, lx, se, sx, allow_l, allow_s, risk, stop_k, add_k, max_u, lev, so, cf, ca):
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    tr_b = np.zeros(100000, np.int64); tr_p = np.zeros(100000); tr_n = np.zeros(100000, np.int64); nt = 0
    cash = 1.0; Q = 0.0; d = 0; units = 0; last_add = 0.0; stop = 0.0; nATR = 0.0; cash0 = 1.0
    blown = -1; stop_outs = 0
    for t in range(start, end):
        e_open = cash + Q * o[t]
        # signal from the previous closed bar
        if d != 0 and t > start and ((d == 1 and lx[t - 1]) or (d == -1 and sx[t - 1])):
            f = _fee(Q, o[t], cf, ca); cash += Q * o[t] - f
            tr_b[nt] = t; tr_p[nt] = cash - cash0; tr_n[nt] = units; nt += 1
            Q = 0.0; d = 0; units = 0
        if d == 0 and t > start and not math.isnan(atr[t - 1]) and atr[t - 1] > 0:
            go = 1 if (allow_l and le[t - 1]) else (-1 if (allow_s and se[t - 1]) else 0)
            if go != 0:
                e0 = cash
                nATR = atr[t - 1]
                qu = risk * e0 / (stop_k * nATR)
                qu = min(qu, e0 * lev / o[t] / max_u)
                if qu > 0:
                    cash0 = cash
                    f = _fee(qu, o[t], cf, ca); cash -= go * qu * o[t] + f; Q = go * qu; d = go; units = 1
                    last_add = o[t]; stop = o[t] - go * stop_k * nATR
        pts = _path(o[t], h[t], l[t], c[t])
        mn = 1e300
        for j in range(4):
            b = pts[j]
            if d != 0:
                # add units every add_k x N in our favour (stop moves with the last add)
                while units < max_u and ((d == 1 and b >= last_add + add_k * nATR) or (d == -1 and b <= last_add - add_k * nATR)):
                    px = last_add + d * add_k * nATR
                    if j == 0:
                        px = b
                    qu = abs(Q) / units
                    f = _fee(qu, px, cf, ca); cash -= d * qu * px + f; Q += d * qu; units += 1
                    last_add = px; stop = px - d * stop_k * nATR
                hit = (d == 1 and b <= stop) or (d == -1 and b >= stop)
                eqx = cash + Q * b
                stopped = eqx <= 0 or eqx <= so * abs(Q) * b / lev
                if hit or stopped:
                    px = b if (j == 0 or stopped) else stop
                    f = _fee(Q, px, cf, ca); cash += Q * px - f
                    tr_b[nt] = t; tr_p[nt] = cash - cash0; tr_n[nt] = units; nt += 1
                    Q = 0.0; d = 0; units = 0
                    if stopped:
                        stop_outs += 1; blown = t
                mn = min(mn, cash + Q * b)
        if Q != 0:
            cash -= (Q * c[t] * cl[t]) if Q > 0 else (-Q * c[t] * cs[t])
        e = cash + Q * c[t]
        eq_c[t - start] = e; eq_m[t - start] = min(mn, e)
        expo[t - start] = abs(Q) * c[t] / e if e > 0 else 0.0
        legs[t - start] = units
        if blown >= 0:
            for r in range(t - start + 1, N):
                eq_c[r] = e; eq_m[r] = e
            break
    return eq_c, eq_m, expo, legs, tr_b[:nt], tr_p[:nt], tr_n[:nt], blown, stop_outs


# ------------------------------------------------------------------ HEDGING (long + short legs at the same time)
@njit(cache=True)
def _hedge(o, h, l, c, cl, cs, start, end, size, tp, btp, bsl, lev, so, cf, ca):
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    tr_b = np.zeros(200000, np.int64); tr_p = np.zeros(200000); tr_n = np.zeros(200000, np.int64); nt = 0
    # open legs stored in a small pool
    M = 400
    lq = np.zeros(M); lp = np.zeros(M); lf = np.zeros(M); used = np.zeros(M, np.bool_)
    cash = 1.0; blown = -1; stop_outs = 0; cyc = 1.0; need_open = True
    for t in range(start, end):
        if need_open and blown < 0:
            x = o[t]
            for dd in (1, -1):
                for k in range(M):
                    if not used[k]:
                        qq = size / x; f = _fee(qq, x, cf, ca)
                        used[k] = True; lq[k] = dd * qq; lp[k] = x; lf[k] = f; cash -= dd * qq * x + f
                        break
            cyc = cash + _net(lq, used) * x
            need_open = False
        pts = _path(o[t], h[t], l[t], c[t])
        mn = 1e300
        for j in range(4):
            b = pts[j]
            # take-profit of single legs, each one re-opened at the same price (keeps the hedge on)
            for k in range(M):
                if used[k]:
                    dd = 1.0 if lq[k] > 0 else -1.0
                    tpx = lp[k] * (1 + dd * tp)
                    if (dd > 0 and b >= tpx) or (dd < 0 and b <= tpx):
                        px = b if j == 0 else tpx
                        f = _fee(lq[k], px, cf, ca); cash += lq[k] * px - f
                        tr_b[nt] = t; tr_p[nt] = lq[k] * (px - lp[k]) - f - lf[k]; tr_n[nt] = 1; nt += 1
                        qq = size / px; f2 = _fee(qq, px, cf, ca)
                        lq[k] = dd * qq; lp[k] = px; lf[k] = f2; cash -= dd * qq * px + f2
            Qn = _net(lq, used); G = _gross(lq, used)
            eqx = cash + Qn * b
            mn = min(mn, eqx)
            basket = (btp > 0 and eqx - cyc >= btp) or (bsl > 0 and eqx - cyc <= -bsl)
            stopped = G > 0 and (eqx <= 0 or eqx <= so * G * b / lev)
            if basket or stopped:
                for k in range(M):
                    if used[k]:
                        f = _fee(lq[k], b, cf, ca); cash += lq[k] * b - f
                        tr_b[nt] = t; tr_p[nt] = lq[k] * (b - lp[k]) - f - lf[k]; tr_n[nt] = 1; nt += 1
                        used[k] = False
                mn = min(mn, cash)
                if stopped:
                    stop_outs += 1; blown = t
                else:
                    need_open = True
                break
        ql = 0.0; qs = 0.0
        for k in range(M):
            if used[k]:
                if lq[k] > 0:
                    ql += lq[k]
                else:
                    qs -= lq[k]
        cash -= ql * c[t] * cl[t] + qs * c[t] * cs[t]
        e = cash + (ql - qs) * c[t]
        eq_c[t - start] = e; eq_m[t - start] = min(mn, e)
        expo[t - start] = (ql + qs) * c[t] / e if e > 0 else 0.0
        legs[t - start] = _count(used)
        if blown >= 0:
            for r in range(t - start + 1, N):
                eq_c[r] = e; eq_m[r] = e
            break
        if nt > 199000:
            break
    return eq_c, eq_m, expo, legs, tr_b[:nt], tr_p[:nt], tr_n[:nt], blown, stop_outs


@njit(cache=True)
def _net(lq, used):
    s = 0.0
    for k in range(len(lq)):
        if used[k]:
            s += lq[k]
    return s


@njit(cache=True)
def _gross(lq, used):
    s = 0.0
    for k in range(len(lq)):
        if used[k]:
            s += abs(lq[k])
    return s


@njit(cache=True)
def _count(used):
    s = 0
    for k in range(len(used)):
        if used[k]:
            s += 1
    return s


# ------------------------------------------------------------------ ZONE RECOVERY
@njit(cache=True)
def _zone(o, h, l, c, cl, cs, start, end, startsig, base, zone, tp, max_legs, lev, so, cf, ca, cool):
    """Open in the signal direction. If price moves `zone` against the basket, open the opposite side with a size chosen
    so that reaching the new take-profit closes the WHOLE basket with the original target profit. After max_legs the
    basket is closed at the next zone break (the loss is taken)."""
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    tr_b = np.zeros(100000, np.int64); tr_p = np.zeros(100000); tr_n = np.zeros(100000, np.int64); nt = 0
    lq = np.zeros(64); lp = np.zeros(64); nl = 0
    cash = 1.0; cash0 = 1.0; blown = -1; stop_outs = 0; last_exit = -10 ** 9
    up = 0.0; dn = 0.0; dcur = 0; target = 0.0
    for t in range(start, end):
        if nl == 0 and blown < 0 and t > start and startsig[t - 1] != 0 and t - last_exit > cool:
            x = o[t]; dcur = int(startsig[t - 1])
            q0 = base / x; f = _fee(q0, x, cf, ca)
            cash0 = cash
            cash -= dcur * q0 * x + f; lq[0] = dcur * q0; lp[0] = x; nl = 1
            target = q0 * x * tp
            if dcur == 1:
                up = x * (1 + tp); dn = x * (1 - zone)
            else:
                dn = x * (1 - tp); up = x * (1 + zone)
            anchor_hi = x if dcur == 1 else x * (1 + zone)
        pts = _path(o[t], h[t], l[t], c[t])
        mn = 1e300
        for j in range(4):
            b = pts[j]
            if nl > 0:
                Qn = 0.0; G = 0.0
                for k in range(nl):
                    Qn += lq[k]; G += abs(lq[k])
                eqx = cash + Qn * b
                mn = min(mn, eqx)
                stopped = eqx <= 0 or eqx <= so * G * b / lev
                win = (dcur == 1 and b >= up) or (dcur == -1 and b <= dn)
                flip = (dcur == 1 and b <= dn) or (dcur == -1 and b >= up)
                if stopped or win or (flip and nl >= max_legs):
                    px = b
                    if not stopped and j > 0:
                        px = up if (win and dcur == 1) or (flip and dcur == -1) else dn
                    f = 0.0
                    for k in range(nl):
                        f += _fee(lq[k], px, cf, ca)
                    cash += Qn * px - f
                    tr_b[nt] = t; tr_p[nt] = cash - cash0; tr_n[nt] = nl; nt += 1
                    nl = 0; last_exit = t
                    if stopped:
                        stop_outs += 1; blown = t
                    mn = min(mn, cash)
                    continue
                if flip:
                    px = b if j == 0 else (dn if dcur == 1 else up)
                    dnew = -dcur
                    tpx = px * (1 + dnew * tp)
                    pnl_exist = 0.0
                    for k in range(nl):
                        pnl_exist += lq[k] * (tpx - lp[k])
                    qn = (target - pnl_exist) / (px * tp)
                    if qn <= 0:
                        qn = base / px
                    f = _fee(qn, px, cf, ca)
                    cash -= dnew * qn * px + f
                    lq[nl] = dnew * qn; lp[nl] = px; nl += 1
                    dcur = dnew
                    if dcur == 1:
                        up = px * (1 + tp); dn = px * (1 - zone)
                    else:
                        dn = px * (1 - tp); up = px * (1 + zone)
        ql = 0.0; qs = 0.0
        for k in range(nl):
            if lq[k] > 0:
                ql += lq[k]
            else:
                qs -= lq[k]
        cash -= ql * c[t] * cl[t] + qs * c[t] * cs[t]
        e = cash + (ql - qs) * c[t]
        eq_c[t - start] = e; eq_m[t - start] = min(mn, e)
        expo[t - start] = (ql + qs) * c[t] / e if e > 0 else 0.0
        legs[t - start] = nl
        if blown >= 0:
            for r in range(t - start + 1, N):
                eq_c[r] = e; eq_m[r] = e
            break
    return eq_c, eq_m, expo, legs, tr_b[:nt], tr_p[:nt], tr_n[:nt], blown, stop_outs


# ------------------------------------------------------------------ MARTINGALE on a rule strategy's trades
@njit(cache=True)
def _mae(h, l, ent, ex, dr, ep):
    out = np.zeros(len(ent))
    for k in range(len(ent)):
        a, b = ent[k], ex[k]
        if dr[k] == 1:
            m = l[a:b + 1].min()
            out[k] = max(0.0, (ep[k] - m) / ep[k])
        else:
            m = h[a:b + 1].max()
            out[k] = max(0.0, (m - ep[k]) / ep[k])
    return out


@njit(cache=True)
def _mart_seq(ret, mae, base, mult, max_steps, anti, compound, lev, so):
    """Size of trade k = base x mult^step (fraction of start capital, or of current equity when compound).
    Martingale: step+1 after a loss, reset after a win; anti-martingale the other way round.
    After max_steps the sequence resets (the loss is accepted). A trade that would need more margin than the
    account has is capped at the account's maximum leverage."""
    n = len(ret)
    eq = 1.0; step = 0
    eqs = np.zeros(n); eqmin = np.zeros(n); size = np.zeros(n); steps = np.zeros(n, np.int64)
    blown = -1; capped = 0
    for k in range(n):
        s = base * mult ** step * (eq if compound else 1.0)
        if s > eq * lev:
            s = eq * lev; capped += 1
        size[k] = s; steps[k] = step
        worst = eq - s * mae[k]
        if worst <= 0 or worst <= so * s / lev:
            eq = so * s / lev if so * s / lev < eq else 0.0
            eq = max(eq, 0.0)
            eqs[k] = eq; eqmin[k] = min(worst, eq)
            blown = k
            break
        eqmin[k] = worst
        eq += s * ret[k]
        eqs[k] = eq
        win = ret[k] > 0
        if (not anti and not win) or (anti and win):
            step = step + 1 if step < max_steps else 0
        else:
            step = 0
        if eq <= 0:
            blown = k
            break
    m = n if blown < 0 else blown + 1
    return eqs[:m], eqmin[:m], size[:m], steps[:m], blown, capped


# ------------------------------------------------------------------ public API
def fam_params(fam, params=None):
    """Full parameter set for a rule family: engine defaults, then the middle of each search range, then `params`."""
    p = dict(E.DEFAULTS)
    for k, v in E.FAMILIES[fam]["space"].items():
        if isinstance(v, tuple):
            m = (v[0] + v[1]) / 2
            p[k] = int(round(m)) if isinstance(v[0], int) and isinstance(v[1], int) else m
        else:
            p[k] = v[len(v) // 2] if k != "trend" else 0
    p.update(params or {})
    return p


def base_signals(cc, fam, params):
    p = fam_params(fam, params)
    return E.signals(fam, p, cc)


def start_signal(cc, kind, direction=1, rsi_len=14, rsi_lvl=30):
    """Deal start condition for DCA / zone recovery: 'always' (new deal right after the previous one) or 'rsi'."""
    n = len(cc.c)
    if kind == "rsi":
        r = cc.get(("rsi", rsi_len), lambda: ind.rsi(cc.c, rsi_len))
        with np.errstate(invalid="ignore"):
            s = ind.crossover(r, rsi_lvl) if direction == 1 else ind.crossunder(r, 100 - rsi_lvl)
        return np.nan_to_num(s, nan=0).astype(np.bool_)
    if kind == "trend":
        e = cc.get(("EMA", 200), lambda: ind.ema(cc.c, 200))
        with np.errstate(invalid="ignore"):
            return np.nan_to_num((cc.c > e) if direction == 1 else (cc.c < e), nan=0).astype(np.bool_)
    return np.ones(n, np.bool_)


def _carry(cc):
    return np.asarray(cc.carry_long, float), np.asarray(cc.carry_short, float)


def run(mode, p, cc, start, end, cost_frac, cost_abs, lev=10.0, stop_out=0.5):
    """Runs one multi-leg backtest. Returns a result dict (see summarize)."""
    o, h, l, c = cc.o, cc.h, cc.l, cc.c
    cl, cs = _carry(cc)
    start, end = int(start), int(end)
    cf, ca, lev, so = float(cost_frac), float(cost_abs), float(lev), float(stop_out)
    tr_n = None
    if mode == "grid":
        x0 = o[start]
        lo, hi = x0 * (1 - p["lo_pct"] / 100), x0 * (1 + p["hi_pct"] / 100)
        n = int(p["grids"])
        res = _grid(o, h, l, c, cl, cs, start, end, lo, hi, n, bool(p["geo"]), {"long": 0, "short": 1, "neutral": 2}[p["gdir"]],
                    p["alloc"] / n, lev, so, cf, ca, p["sl_out"] / 100, bool(p["recenter"]),
                    p["btp"] / 100, p["bsl"] / 100)
        eq_c, eq_m, expo, legs, tb, tp_, blown, sos = res
    elif mode == "dca":
        d = 1 if p["ddir"] == "long" else -1
        sig = start_signal(cc, p["start"], d)
        res = _dca(o, h, l, c, cl, cs, start, end, sig, d, p["base"] / 100, p["so_size"] / 100, int(p["n_so"]), p["dev"] / 100,
                   p["sscale"], p["vscale"], p["tp"] / 100, p["sl"] / 100, lev, so, cf, ca, int(p.get("cool", 0)),
                   p.get("btp", 0) / 100, p.get("bsl", 0) / 100)
        eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos = res
    elif mode == "pyr":
        le, lx, se, sx, _ = base_signals(cc, p["fam"], p["fparams"])
        res = _pyr(o, h, l, c, cc.atr, cl, cs, start, end, le, lx, se, sx, p["pdir"] in ("long", "both"), p["pdir"] in ("short", "both"),
                   p["risk"] / 100, p["stop_k"], p["add_k"], int(p["max_u"]), lev, so, cf, ca)
        eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos = res
    elif mode == "hedge":
        res = _hedge(o, h, l, c, cl, cs, start, end, p["size"] / 100, p["tp"] / 100, p["btp"] / 100, p["bsl"] / 100, lev, so, cf, ca)
        eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos = res
    elif mode == "zone":
        if p["start"] == "trend":
            e = cc.get(("EMA", 200), lambda: ind.ema(c, 200))
            with np.errstate(invalid="ignore"):
                sig = np.where(c > e, 1, np.where(c < e, -1, 0)).astype(np.int64)
        else:
            sig = np.where(c >= o, 1, -1).astype(np.int64)
        res = _zone(o, h, l, c, cl, cs, start, end, sig, p["base"] / 100, p["zone"] / 100, p["tp"] / 100, int(p["max_legs"]), lev, so, cf, ca,
                    int(p.get("cool", 0)))
        eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos = res
    elif mode == "mart":
        return run_mart(p, cc, start, end, cost_frac, cost_abs, lev, stop_out)
    else:
        raise ValueError(mode)
    return summarize(mode, cc, start, end, eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos)


def run_mart(p, cc, start, end, cost_frac, cost_abs, lev=10.0, stop_out=0.5, trades=None):
    """Martingale / anti-martingale sizing on top of a rule strategy (any family, its own exits)."""
    if trades is None:
        fp = fam_params(p["fam"], p["fparams"])
        _, tr = E.run(p["fam"], fp, cc, start, end, p.get("mdir", "both"), cost_frac, cost_abs, lev=1.0)
        ent, ex, dr, ep, ret = (np.asarray(tr[k]) for k in ("ent", "ex", "dir", "ep", "ret"))
        mae = _mae(cc.h, cc.l, ent.astype(np.int64), ex.astype(np.int64), dr.astype(np.int64), ep.astype(float)) if len(ent) else np.zeros(0)
    else:
        ent, ex, ret, mae = trades
    eqs, eqmin, size, steps, blown, capped = _mart_seq(np.asarray(ret, float), np.asarray(mae, float), p["base"] / 100, float(p["mult"]),
                                                       int(p["max_steps"]), bool(p["anti"]), bool(p.get("compound", False)),
                                                       float(lev), float(stop_out))
    N = end - start
    eq_c = np.full(N, np.nan); eq_m = np.full(N, np.nan); expo = np.zeros(N); legs = np.zeros(N)
    cur = 1.0; last = 0
    for k in range(len(eqs)):
        a, b = int(ent[k]) - start, int(ex[k]) - start
        eq_c[last:b] = cur
        eq_m[last:b] = cur
        eq_m[max(a, 0):b + 1] = np.fmin(eq_m[max(a, 0):b + 1], eqmin[k])
        expo[max(a, 0):b + 1] = size[k] / max(cur, 1e-12)
        legs[max(a, 0):b + 1] = steps[k] + 1
        cur = eqs[k]; last = b
    eq_c[last:] = cur
    eq_m[last:] = np.fmin(np.nan_to_num(eq_m[last:], nan=cur), cur)
    tb = np.asarray(ex[:len(eqs)], np.int64)
    pnl = np.diff(np.r_[1.0, eqs])
    blown_bar = int(ex[blown]) if blown >= 0 else -1
    r = summarize("mart", cc, start, end, eq_c, eq_m, expo, legs, tb, pnl, steps + 1, blown_bar, int(blown >= 0))
    r["capped"] = int(capped)
    r["max_step"] = int(steps.max()) if len(steps) else 0
    r["trades_raw"] = (np.asarray(ent), np.asarray(ex), np.asarray(ret), np.asarray(mae))
    return r


def summarize(mode, cc, start, end, eq_c, eq_m, expo, legs, tb, tp_, tr_n, blown, sos):
    eq_c = pd.Series(eq_c).ffill().fillna(1.0).to_numpy()
    eq_m = np.fmin(pd.Series(eq_m).fillna(pd.Series(eq_c)).to_numpy(), eq_c)
    peak = np.maximum.accumulate(np.r_[1.0, eq_c])[1:]
    dd = np.where(peak > 0, 1 - eq_m / peak, 0.0)
    maxdd = float(np.clip(dd.max(), 0, 1)) if len(dd) else 0.0
    under = eq_c < peak - 1e-12
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    bpd = cc.bars_per_year / 365.25
    pnl = np.asarray(tp_, float)
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    net = float(eq_c[-1] - 1) if len(eq_c) else 0.0
    years = max((end - start) / cc.bars_per_year, 1e-9)
    cagr = (max(eq_c[-1], 1e-9)) ** (1 / years) - 1 if len(eq_c) else 0.0
    rets = np.diff(np.log(np.maximum(eq_c[np.r_[0:len(eq_c):max(1, int(round(bpd)))]], 1e-12)))
    sharpe = float(rets.mean() / rets.std() * math.sqrt(365.25)) if len(rets) > 5 and rets.std() > 0 else 0.0
    m = dict(net=net, cagr=float(cagr), maxdd=maxdd, trades=int(len(pnl)), winrate=float((pnl > 0).mean()) if len(pnl) else 0.0,
             pf=float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0),
             avg=float(pnl.mean()) if len(pnl) else 0.0, worst=float(pnl.min()) if len(pnl) else 0.0,
             max_expo=float(np.nanmax(expo)) if len(expo) else 0.0, max_legs=int(np.nanmax(legs)) if len(legs) else 0,
             under_days=float(longest / bpd), under_share=float(under.mean()) if len(under) else 0.0,
             blown=bool(blown >= 0), blown_at=(cc.index[blown] if blown >= 0 else None), stop_outs=int(sos), sharpe=sharpe,
             ret_dd=float(net / maxdd) if maxdd > 0 else 0.0)
    return dict(mode=mode, m=m, eq=eq_c, eq_min=eq_m, expo=np.asarray(expo), legs=np.asarray(legs), dd=dd,
                tr_bar=np.asarray(tb), tr_pnl=pnl, tr_legs=(np.asarray(tr_n) if tr_n is not None else None), start=start, end=end)
