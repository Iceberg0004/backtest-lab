"""Builds multi_case.json: candles + Pine exports of DCA / martingale / grid, plus the Python engine's deals for DCA."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import numpy as np, pandas as pd
from core import engine as E, position as PS, pine_multi as PM
src = sorted(f for f in os.listdir(".") if f.startswith("case") and f.endswith(".json"))
d = json.load(open(next(f for f in src if json.load(open(f))["sym"] == "BTCUSDT")))
c = pd.DataFrame(d["candles"])
idx = pd.to_datetime(c["openTime"], unit="ms", utc=True) if "openTime" in c else pd.date_range("2024-01-01", periods=len(c), freq="1h", tz="UTC")
df = pd.DataFrame({k: c[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")}, index=idx)
cc = E.Cache(df)
p_dca = dict(ddir="long", start="rsi", base=10, so_size=10, n_so=4, dev=1.5, sscale=1.3, vscale=1.5, tp=1.5, sl=0, cool=0, btp=0, bsl=0)
r = PS.run("dca", p_dca, cc, 1, len(df), 0.0, 0.0, lev=10, stop_out=0.0)
p_mart = dict(fam="rsi_mr", fparams=dict(rsi_len=14, lower=30, upper=70, exit_level=55, trend=0), base=5, mult=2.0, max_steps=4, anti=0, mdir="both")
p_grid = dict(lo_pct=10, hi_pct=10, grids=10, geo=1)
out = dict(candles=d["candles"], dca=PM.to_pine("dca", p_dca, "BTCUSDT", "1h", 0, 0.0), mart=PM.to_pine("mart", p_mart, "BTCUSDT", "1h", 0, 0.0),
           grid=PM.to_pine("grid", p_grid, "BTCUSDT", "1h", float(df.close.iloc[-1]), 0.0),
           py_dca=dict(deals=int(len(r["tr_pnl"])), exit_bars=[int(x) for x in r["tr_bar"]], pnl=[float(x) for x in r["tr_pnl"]], legs=[int(x) for x in r["tr_legs"]]))
json.dump(out, open("multi_case.json", "w"))
print("python DCA deals", out["py_dca"]["deals"], "first exits", out["py_dca"]["exit_bars"][:8])
