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
        df = closed_only(SOURCES[src].get(sym, "1h", refresh=False), "1h").iloc[-N:]
        DATA[(src, sym)] = (df, E.Cache(df))
    return DATA[(src, sym)]


def expect_strategy(tr, p):
    """Python trades -> TradingView closed-trade records. A partial TP1 is its own record; while TP1 is still
    pending, a stop-out fills BOTH exit orders (the 50% TP1 order also carries the stop) = two records."""
    out = []
    tp1_on = E.full_params(p)["tp1_r"] > 0
    for k in range(len(tr["ent"])):
        part = tr["p1b"][k] >= 0
        rec = (int(tr["ent"][k]), int(tr["ex"][k]), float(tr["ep"][k]), float(tr["xp"][k]), int(tr["amb"][k]))
        if part:
            out.append((int(tr["ent"][k]), int(tr["p1b"][k]), float(tr["ep"][k]), float(tr["p1p"][k]), 0))
        if tr["reason"][k] != 3:
            out.append(rec)
            if tp1_on and not part and tr["reason"][k] == 1:
                out.append(rec)
    out.sort(key=lambda r: (r[1], r[0]))
    return out


cases = []
rng = np.random.default_rng(2026)
mk = ["binance:BTCUSDT", "dukascopy:EURUSD", "dukascopy:XAUUSD"]
for fam in E.FAMILIES:
    if E.FAMILIES[fam].get("needs"):      # derivatives families have no Pine version
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
    strat, indi = to_pine(fam, p, d, "t", sym, "1h", 0.0, 1.0, "test", [("IS PF", "1.23"), ("Grade", "C")])
    indi += '\nplot(goLong ? 1 : 0, "xL")\nplot(goShort ? 1 : 0, "xS")\nplot(not na(tradePnl) ? 1 : 0, "xX")\n'
    json.dump(dict(fam=fam, sym=sym, params=E.full_params(p), candles=[dict(open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
                                 openTime=int(t.timestamp() * 1000), closeTime=int(t.timestamp() * 1000) + 3599999)
                            for t, r in df.iterrows()],
                   indicator=indi, strategy=strat, strat_expect=expect_strategy(tr, p),
                   py=dict(ent=tr["ent"].tolist(), ex=tr["ex"].tolist(), dir=tr["dir"].tolist(), reason=tr["reason"].tolist(),
                           ep=tr["ep"].tolist(), xp=tr["xp"].tolist())),
              open(os.path.join(HERE, f"case{k}.json"), "w"))
    print(k, sym, fam, len(tr["ent"]), "python trades", flush=True)
