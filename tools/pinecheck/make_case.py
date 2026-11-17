"""Build parity test cases: same candles -> Python engine trades vs exported Pine (run by PineTS).
Run from the repo root:  python tools/pinecheck/make_case.py && cd tools/pinecheck && node run_all.mjs"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.data import SOURCES, closed_only
from core import engine as E
from core.pine import to_pine

X = {"regime": 0, "adx_thr": 20}
cases = [
    ("binance", "BTCUSDT", "ma_cross", {"ma_type": "EMA", "fast": 20, "slow": 100, "trend": 200, "sl_atr": 2.0, "tp_atr": 3.0, "trail_atr": 0.0, **X}, "both"),
    ("binance", "BTCUSDT", "rsi_mr", {"rsi_len": 14, "lower": 30, "upper": 70, "exit_level": 55, "trend": 0, "sl_atr": 2.0, "tp_atr": 0.0, "trail_atr": 3.0, **X}, "both"),
    ("binance", "BTCUSDT", "donchian", {"entry_len": 55, "exit_len": 20, "trend": 0, "sl_atr": 3.0, "tp_atr": 0.0, "trail_atr": 0.0, "regime": 1, "adx_thr": 25}, "long"),
    ("dukascopy", "EURUSD", "bb_squeeze", {"bb_len": 20, "bb_mult": 2.0, "sq_len": 100, "sq_factor": 0.8, "sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 0.0, **X}, "both"),
    ("dukascopy", "EURUSD", "macd", {"fast": 12, "slow": 26, "signal": 9, "trend": 100, "sl_atr": 1.5, "tp_atr": 2.0, "trail_atr": 2.0, "regime": 2, "adx_thr": 20}, "both"),
    ("binance", "BTCUSDT", "supertrend", {"st_len": 10, "st_mult": 3.0, "trend": 0, "sl_atr": 3.0, "tp_atr": 0.0, "trail_atr": 0.0, **X}, "both"),
    ("dukascopy", "XAUUSD", "supertrend", {"st_len": 14, "st_mult": 2.0, "trend": 200, "sl_atr": 2.0, "tp_atr": 4.0, "trail_atr": 3.0, "regime": 1, "adx_thr": 20}, "both"),
    ("binance", "BTCUSDT", "keltner", {"kc_len": 20, "kc_mult": 2.0, "trend": 100, "sl_atr": 2.0, "tp_atr": 3.0, "trail_atr": 0.0, **X}, "both"),
    ("dukascopy", "XAUUSD", "stochrsi", {"rsi_len": 14, "stoch_len": 14, "k_lo": 20, "k_hi": 80, "trend": 200, "sl_atr": 2.0, "tp_atr": 3.0, "trail_atr": 0.0, **X}, "both"),
    ("binance", "BTCUSDT", "stochrsi", {"rsi_len": 10, "stoch_len": 12, "k_lo": 15, "k_hi": 85, "trend": 50, "sl_atr": 1.5, "tp_atr": 0.0, "trail_atr": 2.0, "regime": 2, "adx_thr": 25}, "both"),
    ("dukascopy", "EURUSD", "zscore", {"z_len": 50, "z_entry": 2.0, "z_exit": 0.0, "trend": 0, "sl_atr": 3.0, "tp_atr": 0.0, "trail_atr": 0.0, **X}, "both"),
    ("binance", "BTCUSDT", "zscore", {"z_len": 30, "z_entry": 2.5, "z_exit": 0.5, "trend": 200, "sl_atr": 2.0, "tp_atr": 2.0, "trail_atr": 0.0, "regime": 2, "adx_thr": 20}, "both"),
]
N = 4000
for k, (src, sym, fam, p, d) in enumerate(cases):
    df = closed_only(SOURCES[src].get(sym, "1h", refresh=False), "1h").iloc[-N:]
    cc = E.Cache(df)
    _, tr = E.run(fam, p, cc, 0, len(df), d, 0.0, 0.0)
    strat, indi = to_pine(fam, p, d, "t", sym, "1h", 0.0, 1.0, "test", [("IS PF", "1.23"), ("Grade", "C")])
    indi += '\nplot(goLong ? 1 : 0, "xL")\nplot(goShort ? 1 : 0, "xS")\nplot(not na(tradePnl) ? 1 : 0, "xX")\n'
    json.dump(dict(fam=fam, sym=sym, candles=[dict(open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume,
                                 openTime=int(t.timestamp() * 1000), closeTime=int(t.timestamp() * 1000) + 3599999)
                            for t, r in df.iterrows()],
                   indicator=indi, strategy=strat,
                   py=dict(ent=tr["ent"].tolist(), ex=tr["ex"].tolist(), dir=tr["dir"].tolist(), reason=tr["reason"].tolist(),
                           ep=tr["ep"].tolist(), xp=tr["xp"].tolist())),
              open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"case{k}.json"), "w"))
    print(k, sym, fam, p.get("regime"), len(tr["ent"]), "python trades")
