"""v7 "Bandingkan exit": keep the ENTRY of a strategy fixed and try many exit combinations (all SL / TP / trailing /
partial / rule types). Ranked by the unseen (out-of-sample) part, but the in-sample result is always shown next to it:
picking the best exit by its unseen result is itself a selection, so treat the top row as optimistic."""
import numpy as np
from . import engine as E, exits as X

EXIT_KEYS = set(E.EXIT_SPACE) | {"exit_sig"}


def _base(p):
    q = E.full_params(p)
    for k in EXIT_KEYS:
        if k in E.DEFAULTS:
            q[k] = E.DEFAULTS[k]
    return q


def combos(fam, p, n=120, seed=11, tf_min=60):
    """Curated grid (every SL type with R targets / its natural TP, the trailing types, partial + breakeven, rules)
    followed by random exit samples. The original exit is always row 0."""
    rng = np.random.default_rng(seed)
    out = [("Exit asli", E.full_params(p))]
    base = _base(p)
    def add(lbl, **kw):
        q = dict(base); q.update(kw); out.append((lbl, E.fix_params(fam, q, tf_min)))
    for sm in range(14):
        for rr in (1.5, 2.0, 3.0):
            add(f"SL {X.SL_TYPES[sm]} · TP {rr:g}R", sl_mode=sm, tp_mode=1, tp_rr=rr)
    for tm, kw in ((2, dict(tp_atr=3.0)), (3, dict(tp_pct=2.0)), (5, {}), (6, {}), (7, dict(tp_fib=1.618)), (7, dict(tp_fib=1.272)), (8, {})):
        add(f"SL ATR 2 · TP {X.TP_TYPES[tm]}" + (f" {kw.get('tp_fib')}" if "tp_fib" in kw else ""), sl_mode=0, sl_atr=2.0, tp_mode=tm, **kw)
    for tmode in range(8):
        add(f"SL ATR 2 · trailing {X.TRAIL_TYPES[tmode]}", sl_mode=0, sl_atr=2.0, tp_mode=9, trail_atr=3.0, trail_mode=tmode)
        add(f"SL ATR 2 · trailing {X.TRAIL_TYPES[tmode]} aktif setelah 1R", sl_mode=0, sl_atr=2.0, tp_mode=9, trail_atr=3.0,
            trail_mode=tmode, trail_act=1.0)
    add("SL ATR 2 · TP1 1R 50% · BE · TP 3R", sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=3.0, tp1_r=1.0, tp1_f=0.5, be_r=1.0)
    add("SL ATR 2 · TP1 1R / TP2 2R / TP3 4R", sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=4.0, tp1_r=1.0, tp1_f=0.33, tp2_r=2.0, tp2_f=0.33)
    add("SL ATR 2 · TP 2R · tutup akhir hari", sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=2.0, eod=1)
    add("SL ATR 2 · TP 2R · time stop 12 candle", sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=2.0, ts_bars=12)
    add("SL ATR 2 · TP 2R · jeda 6 candle setelah rugi", sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=2.0, cool=6)
    add("Hanya sinyal kebalikan (tanpa SL/TP)", sl_mode=14, tp_mode=9, exit_sig=1)
    seen = {tuple(sorted(q.items())) for _, q in out}
    tries = 0
    while len(out) < n and tries < n * 20:
        tries += 1
        q = dict(base)
        for k, v in E.EXIT_SPACE.items():
            v_ = rng.choice(v)
            q[k] = v_.item() if hasattr(v_, "item") else v_
        for k in list(q):
            if k in E.INT_KEYS:
                q[k] = int(q[k])
            elif k in E.EXIT_SPACE:
                q[k] = float(q[k])
        q = E.fix_params(fam, q, tf_min)
        key = tuple(sorted(q.items()))
        if key in seen:
            continue
        seen.add(key); out.append(("Acak: " + X.describe(q, 1.0)[:90], q))
    return out[:max(n, 1)]


def compare(fam, p, cc, is_start, split, n, direction, cf, ca, lev=1.0, n_combos=120, min_trades=5, progress=None):
    rows = []
    cl = combos(fam, p, n_combos, tf_min=cc.tf_min)
    for i, (lbl, q) in enumerate(cl):
        q = E.drop_unavailable(q, cc)
        im, itr = E.run(fam, q, cc, is_start, split, direction, cf, ca, lev=lev)
        wf, _ = E.wf_consistency(itr, is_start, split, 4)
        om, _ = E.run(fam, q, cc, split, n, direction, cf, ca, lev=lev)
        rows.append(dict(label=lbl, params=q, is_m=im, oos_m=om, wf=float(wf), sl=int(q["sl_mode"]), tp=int(q.get("tp_mode", 0)),
                         ok=im["trades"] >= min_trades and om["trades"] >= max(3, min_trades // 3)))
        if progress and i % 5 == 0:
            progress((i + 1) / len(cl))
    def key(r):
        om = r["oos_m"]
        both = om["net"] > 0 and r["is_m"]["net"] > 0
        return (r["ok"], both and r["wf"] >= 0.5, both, om["ret_dd"])
    first = rows[0]
    rest = sorted(rows[1:], key=key, reverse=True)
    return [first] + rest
