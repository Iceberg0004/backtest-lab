"""Build parity test cases: same candles -> Python engine trades vs exported Pine (run by PineTS).
Covers every family (random parameters, filters and exits) plus targeted exit cases.
Run from the repo root:  python tools/pinecheck/make_case.py [n_random_per_family] && cd tools/pinecheck && node run_all.mjs"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.data import SOURCES, closed_only
from core import engine as E
from core.pine import to_pine

HERE = os.path.dirname(os.path.abspath(__file__))
for f in glob.glob(os.path.join(HERE, "case*.json")):
    os.remove(f)
N = 3000
PER = int(sys.argv[1]) if len(sys.argv) > 1 else 1
DATA = {}


def data(src, sym):
    if (src, sym) not in DATA:
        df = closed_only(SOURCES[src].get(sym, "1h", refresh=False), "1h")
        if src == "binance" and sym != "BTCUSDT":     # benchmark close for relative-strength families
            b = closed_only(SOURCES["binance"].get("BTCUSDT", "1h", refresh=False), "1h")["close"].rename("bench")
            df = df.join(b, how="left")
        df = df.iloc[-N:]
        DATA[(src, sym)] = (df, E.Cache(df))
    return DATA[(src, sym)]


def expect_strategy(tr, p):
    """Python trades -> TradingView closed-trade records. Each partial TP is its own record. A stop-out also fills every
    partial-TP order still pending (those orders carry the stop too) = one extra record each.
    Known: in ~4% of multi-partial cases TradingView merges/skips a leg; the indicator() twin stays exact."""
    out = []
    for k in range(len(tr["ent"])):
        rec = (int(tr["ent"][k]), int(tr["ex"][k]), float(tr["ep"][k]), float(tr["xp"][k]), int(tr["amb"][k]))
        done = 0
        for b_, p_ in (("p1b", "p1p"), ("p2b", "p2p")):
            if tr[b_][k] >= 0:
                out.append((int(tr["ent"][k]), int(tr[b_][k]), float(tr["ep"][k]), float(tr[p_][k]), 0)); done += 1
        if tr["reason"][k] != 3:
            out.append(rec)
            if tr["reason"][k] == 1:
                out += [rec] * (int(tr["npart"][k]) - done)
    out.sort(key=lambda r: (r[1], r[0]))
    return out


cases = []
rng = np.random.default_rng(2026)
mk = ["binance:BTCUSDT", "dukascopy:EURUSD", "dukascopy:XAUUSD"]
for fam in E.FAMILIES:
    if set(E.FAMILIES[fam].get("needs") or ()) - E.PINE_OK_NEEDS:      # derivatives families have no Pine version
        continue
    if "bench" in (E.FAMILIES[fam].get("needs") or ()):
        for r in range(max(2, PER)):
            df, cc = data("binance", "ETHUSDT")
            for _ in range(50):
                p = E.drop_unavailable(E.sample_params(fam, rng, 60, combo=True), cc)
                _, tr = E.run(fam, p, cc, 0, len(df), "both", 0.0, 0.0)
                if len(tr["ent"]) >= 5:
                    break
            cases.append(("binance", "ETHUSDT", fam, p, "both"))
        continue
    for r in range(PER):
        src, sym = mk[(len(cases)) % 3].split(":")
        df, cc = data(src, sym)
        for _ in range(50):
            p = E.drop_unavailable(E.sample_params(fam, rng, 60, combo=True), cc)
            _, tr = E.run(fam, p, cc, 0, len(df), "both", 0.0, 0.0)
            if len(tr["ent"]) >= 5:
                break
        cases.append((src, sym, fam, p, "both"))
# targeted exit cases (all features forced on)
base = {"ma_type": "EMA", "fast": 12, "slow": 48, "trend": 0}
exits = [
    dict(sl_mode=0, sl_atr=2.0, tp_atr=4.0, tp1_r=1.0, be_r=1.0, trail_atr=0.0, max_bars=0),
    dict(sl_mode=1, sl_pct=1.0, tp_pct=3.0, tp1_r=1.0, be_r=0.0, trail_atr=0.0, max_bars=24, sl_atr=2.0, tp_atr=0.0),
    dict(sl_mode=0, sl_atr=2.0, tp_atr=0.0, trail_atr=3.0, trail_mode=1, be_r=1.5, max_bars=48),
    dict(sl_mode=0, sl_atr=1.5, tp_atr=3.0, trail_atr=2.0, trail_mode=0, max_bars=12, exit_sig=1),
    dict(sl_mode=1, sl_pct=0.5, tp_pct=2.0, tp1_r=1.5, be_r=1.0, trail_atr=2.0, trail_mode=1, max_bars=96),
]
# v7: every SL type, TP type, trailing type and trade rule
v7 = [dict(sl_mode=m, sl_n=10, sl_buf=0.5, sl_pip=200.0, sl_money=150.0, sl_atr=2.0, tp_mode=1, tp_rr=2.0) for m in range(2, 15)]
v7 += [dict(sl_mode=0, sl_atr=2.0, tp_mode=t, tp_rr=2.5, tp_atr=3.0, tp_pct=2.0, tp_pip=300.0, tp_n=20, tp_fib=1.618) for t in range(1, 10)]
v7 += [dict(sl_mode=0, sl_atr=2.5, tp_mode=9, trail_atr=2.0, trail_mode=m, trail_pct=2.0, trail_ma=20, trail_step=1.0, trail_n=10,
            trail_act=a) for m in range(2, 8) for a in (0.0, 1.0)]
v7 += [dict(sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=4.0, tp1_r=1.0, tp1_f=0.33, tp2_r=2.0, tp2_f=0.33, be_r=1.0, be_off=0.25),
       dict(sl_mode=3, sl_n=10, sl_buf=0.2, tp_mode=1, tp_rr=3.0, tp1_r=1.5, tp1_f=0.5, ts_bars=12, eod=1),
       dict(sl_mode=0, sl_atr=1.5, tp_mode=1, tp_rr=2.0, cool=6, max_day=1, reent=1, wkend=1),
       dict(sl_mode=1, sl_pct=0.5, tp_mode=3, tp_pct=1.0, dloss=1.0, dprof=1.5, eod=2, max_bars=24),
       dict(sl_mode=0, sl_atr=2.0, tp_mode=1, tp_rr=3.0, eod=3, wkend=1, cool=3)]
for k, ex in enumerate(v7):
    src, sym = mk[k % 3].split(":")
    cases.append((src, sym, "ma_cross", dict(base, fast=8, slow=30, **ex), "both"))
for k, ex in enumerate(exits):
    src, sym = mk[k % 3].split(":")
    cases.append((src, sym, "ma_cross", dict(base, **ex), "both"))
cases.append(("binance", "BTCUSDT", "rsi_mr", {"rsi_len": 14, "lower": 35, "upper": 65, "exit_level": 55, "trend": 0, "sl_atr": 2.0, "tp_atr": 3.0,
              "trail_atr": 0.0, "sess": 2, "dow": 3, "htf_m": 240, "htf_len": 20, "vol": 0, "rsi_f": 0, "volf": 1}, "both"))
cases.append(("dukascopy", "EURUSD", "donchian", {"entry_len": 30, "exit_len": 10, "trend": 100, "sl_atr": 2.0, "tp_atr": 0.0, "trail_atr": 0.0,
              "htf_m": 1440, "htf_len": 20, "rsi_f": 1, "regime": 1, "adx_thr": 20}, "short"))

for k, (src, sym, fam, p, d) in enumerate(cases):
    df, cc = data(src, sym)
    _, tr = E.run(fam, p, cc, 0, len(df), d, 0.0, 0.0)
    strat, indi = to_pine(fam, p, d, "t", sym, "1h", 0.0, 1.0, "test", [("IS PF", "1.23"), ("Grade", "C")], pip=E.X.pip_size(cc.c[-1]), tf_min=cc.tf_min,
                          bench_vals=cc.x["bench"].tolist() if "bench" in cc.x else None)
    indi += '\nplot(goLong ? 1 : 0, "xL")\nplot(goShort ? 1 : 0, "xS")\nplot(not na(tradePnl) ? 1 : 0, "xX")\n'
    json.dump(dict(fam=fam, sym=sym, params=E.full_params(p), candles=[dict(open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
                                 openTime=int(t.timestamp() * 1000), closeTime=int(t.timestamp() * 1000) + 3599999)
                            for t, r in df.iterrows()],
                   indicator=indi, strategy=strat, strat_expect=expect_strategy(tr, p),
                   py=dict(ent=tr["ent"].tolist(), ex=tr["ex"].tolist(), dir=tr["dir"].tolist(), reason=tr["reason"].tolist(),
                           ep=tr["ep"].tolist(), xp=tr["xp"].tolist())),
              open(os.path.join(HERE, f"case{k}.json"), "w"))
    print(k, sym, fam, len(tr["ent"]), "python trades", flush=True)
