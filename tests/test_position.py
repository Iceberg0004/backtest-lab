"""Unit tests for core/position.py on synthetic prices with known answers. Run: python tests/test_position.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from core import engine as E, position as PS


def make(close, spread=0.0):
    close = np.asarray(close, float)
    idx = pd.date_range("2024-01-01", periods=len(close), freq="1h", tz="UTC")
    o = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": o, "high": np.maximum(o, close) + spread, "low": np.minimum(o, close) - spread, "close": close,
                       "volume": 1.0}, index=idx)
    return E.Cache(df)


def check(name, cond, info=""):
    print(("OK  " if cond else "FAIL"), name, info)
    if not cond:
        check.failed = True
check.failed = False

# 1) grid on a price that oscillates and returns to the start: every round trip wins, final equity > 1
x = 100 + 8 * np.sin(np.linspace(0, 12 * np.pi, 600))
cc = make(x)
r = PS.run("grid", dict(lo_pct=10, hi_pct=10, grids=8, geo=0, gdir="neutral", alloc=1.0, sl_out=0, recenter=0, btp=0, bsl=0), cc, 1, len(x), 0, 0)
check("grid: round trips all profitable", r["m"]["trades"] > 10 and (r["tr_pnl"] > 0).all(), f"{r['m']['trades']} trades")
check("grid: profit when price comes back", r["eq"][-1] > 1.0, f"eq {r['eq'][-1]:.4f}")

# 2) DCA long: drop 10% then recover -> safety orders fill, deal closes at TP from the average price
x = np.r_[np.linspace(100, 90, 50), np.linspace(90, 100, 50)]
cc = make(x)
p = dict(ddir="long", start="always", base=10, so_size=10, n_so=3, dev=2.0, sscale=1.0, vscale=1.0, tp=1.0, sl=0)
r = PS.run("dca", p, cc, 1, len(x), 0, 0)
# levels: 98, 96, 94 (from base price at bar 1 open); avg of 4 equal-notional orders
b0 = cc.o[1]; lv = [b0 * (1 - 0.02 * k) for k in (1, 2, 3)]
q = [0.1 / b0] + [0.1 / v for v in lv]; avg = (0.1 * 4) / sum(q)
exp_pnl = sum(q) * avg * 1.01 - 0.4
check("dca: first deal used 3 safety orders", r["tr_legs"] is not None and r["tr_legs"][0] == 3, str(r["tr_legs"][:3]))
check("dca: first deal profit = TP from average", abs(r["tr_pnl"][0] - exp_pnl) < 1e-9, f"{r['tr_pnl'][0]:.6f} vs {exp_pnl:.6f}")

# 3) stop-out: huge DCA with leverage on a -90% crash must blow up the account
x = np.linspace(100, 10, 300)
cc = make(x)
p = dict(ddir="long", start="always", base=200, so_size=300, n_so=5, dev=5, sscale=1.2, vscale=2, tp=2, sl=0)
r = PS.run("dca", p, cc, 1, len(x), 0, 0, lev=20, stop_out=0.5)
check("dca: blow-up detected", r["m"]["blown"] and r["m"]["stop_outs"] == 1, f"eq {r['eq'][-1]:.3f}")

# 4) martingale sequence arithmetic
eqs, eqmin, size, steps, blown, capped = PS._mart_seq(np.array([-0.01, -0.01, 0.01]), np.zeros(3), 1.0, 2.0, 5, False, False, 100.0, 0.5)
check("martingale: sizes 1,2,4 and equity 1.01", np.allclose(size, [1, 2, 4]) and abs(eqs[-1] - 1.01) < 1e-12, f"{size} {eqs}")
eqs, eqmin, size, steps, blown, capped = PS._mart_seq(np.array([0.01, 0.01, -0.01]), np.zeros(3), 1.0, 2.0, 5, True, False, 100.0, 0.5)
check("anti-martingale: sizes 1,2,4", np.allclose(size, [1, 2, 4]), f"{size}")
eqs, eqmin, size, steps, blown, capped = PS._mart_seq(np.array([-0.5, 0.1]), np.array([1.2, 0.0]), 1.0, 2.0, 5, False, False, 10.0, 0.5)
check("martingale: adverse move beyond margin -> blown", blown == 0, f"blown={blown}")

# 5) zone recovery: down move flips to short, then continued drop reaches the short TP -> basket closes with the target
x = np.r_[np.linspace(100, 98.9, 20), np.linspace(98.9, 96.0, 40)]
cc = make(x)
sig = np.ones(len(x), np.int64)
res = PS._zone(cc.o, cc.h, cc.l, cc.c, np.zeros(len(x)), np.zeros(len(x)), 1, len(x), sig, 0.1, 0.01, 0.02, 5, 100.0, 0.5, 0.0, 0.0, 0)
pnl, legs = res[5], res[6]
x0 = cc.o[1]
check("zone: basket closed with target profit after one flip", len(pnl) >= 1 and legs[0] == 2 and abs(pnl[0] - 0.1 * 0.02) < 1e-9,
      f"pnl {pnl[:2]} legs {legs[:2]}")

# 6) no lookahead: cutting the future does not change the past
rng = np.random.default_rng(1)
x = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 3000)))
cc_full, cc_cut = make(x), make(x[:2000])
for mode, p in (("grid", dict(lo_pct=15, hi_pct=15, grids=10, geo=1, gdir="long", alloc=1.0, sl_out=0, recenter=1, btp=0, bsl=0)),
                ("dca", dict(ddir="long", start="rsi", base=10, so_size=10, n_so=4, dev=1.5, sscale=1.3, vscale=1.5, tp=1.5, sl=0)),
                ("hedge", dict(size=20, tp=1.0, btp=3, bsl=10)),
                ("zone", dict(start="trend", base=10, zone=1.0, tp=2.0, max_legs=5)),
                ("pyr", dict(fam="donchian", fparams=dict(entry_len=55, exit_len=20, trend=0), pdir="both", risk=1, stop_k=2, add_k=0.5, max_u=4)),
                ("mart", dict(fam="rsi_mr", fparams={}, base=5, mult=2.0, max_steps=4, anti=0))):
    a = PS.run(mode, p, cc_full, 300, 3000, 0.001, 0)["eq"][:1500]
    b = PS.run(mode, p, cc_cut, 300, 2000, 0.001, 0)["eq"][:1500]
    check(f"no lookahead: {mode}", np.allclose(a, b), f"max diff {np.nanmax(np.abs(a - b)):.2e}")
sys.exit(1 if check.failed else 0)
