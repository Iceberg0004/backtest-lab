"""Current signal of a saved strategy on the latest data. Read-only: nothing is ever sent to a broker."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import engine as E

TAIL = 5000          # bars used (enough warm-up for every indicator, small memory)


def _last_sig(le, lx, se, sx, i):
    if le[i]:
        return "BELI (long)"
    if se[i]:
        return "JUAL (short)"
    if lx[i] or sx[i]:
        return "Tutup posisi"
    return "Tidak ada"


def state(fam, p, direction, closed: pd.DataFrame, raw: pd.DataFrame | None, cost_frac, cost_abs, lev=1.0, n_last=10):
    p = E.full_params(p)
    d = closed.iloc[-TAIL:]
    cc = E.Cache(d)
    le, lx, se, sx, _ = E.signals(fam, p, cc)
    if direction == "long":
        se = np.zeros_like(se)
    elif direction == "short":
        le = np.zeros_like(le)
    n = len(d)
    start = min(260, n // 4)
    m, tr = E.run(fam, p, cc, start, n, direction, cost_frac, cost_abs, lev=lev, sig=(le, lx, se, sx, {}))
    out = dict(bar_time=d.index[-1], close=float(d["close"].iloc[-1]), closed_signal=_last_sig(le, lx, se, sx, n - 1))
    # open position = last trade that only ended because the data ended
    pos = None
    if len(tr["ent"]) and tr["reason"][-1] == 3:
        k = len(tr["ent"]) - 1
        pos = dict(arah="Long" if tr["dir"][k] > 0 else "Short", masuk=d.index[min(tr["ent"][k], n - 1)], harga_masuk=float(tr["ep"][k]),
                   stop=float(tr["sl"][k]) if np.isfinite(tr["sl"][k]) and tr["sl"][k] > 0 else None,
                   target=float(tr["tp"][k]) if np.isfinite(tr["tp"][k]) and tr["tp"][k] > 0 else None,
                   pnl=float(tr["dir"][k] * (out["close"] / tr["ep"][k] - 1)))
    out["position"] = pos
    # the still-forming candle (provisional: can change until it closes)
    out["forming_signal"] = None
    if raw is not None and len(raw) and raw.index[-1] > d.index[-1]:
        dr = pd.concat([d.iloc[-(TAIL - 1):], raw[raw.index > d.index[-1]].iloc[-1:][d.columns.intersection(raw.columns)]])
        cr = E.Cache(dr)
        a, b, c_, e = E.signals(fam, p, cr)[:4]
        if direction == "long":
            c_ = np.zeros_like(c_)
        elif direction == "short":
            a = np.zeros_like(a)
        out["forming_signal"] = _last_sig(a, b, c_, e, len(dr) - 1)
        out["forming_time"] = dr.index[-1]
    rows = []
    for k in range(max(0, len(tr["ent"]) - n_last), len(tr["ent"])):
        rows.append({"Sinyal (candle tutup)": d.index[max(tr["ent"][k] - 1, 0)], "Arah": "Long" if tr["dir"][k] > 0 else "Short",
                     "Harga masuk": float(tr["ep"][k]),
                     "Keluar": d.index[min(tr["ex"][k], n - 1)] if tr["reason"][k] != 3 else None,
                     "Hasil %": float(tr["ret"][k] * 100) if tr["reason"][k] != 3 else None,
                     "Status": "masih terbuka" if tr["reason"][k] == 3 else ["Sinyal", "Stop loss", "Take profit", "Akhir data", "Waktu habis"][tr["reason"][k]]})
    out["last"] = pd.DataFrame(rows[::-1])
    out["recent_m"] = m
    return out
