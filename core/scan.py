"""Scan many markets / timeframes with the same search settings and rank them in one table (time-budgeted)."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
from . import engine as E, robust as R
from .data import closed_only, TF_MINUTES

# how much history a scan uses per timeframe (days); keeps a multi-market scan fast on a small server
SCAN_DAYS = {"5m": 180, "15m": 365, "30m": 730, "1h": 1460, "2h": 2190, "4h": 2920, "1d": 5000}


def scan_markets(jobs, load_fn, cfg_base, cost_fn, total_budget, progress=None, min_bars=1500):
    """jobs: list of (source, symbol, tf). load_fn(source, symbol, tf, start) -> DataFrame.
    cost_fn(source, symbol, price) -> (cost_frac, cost_abs). Returns (rows, results)."""
    t0 = time.time()
    rows, results = [], {}
    for k, (src, sym, tf) in enumerate(jobs):
        left = total_budget - (time.time() - t0)
        if left < 3:
            rows.append(dict(Pasar=sym, TF=tf, Status="⏱️ dilewati (waktu habis)"))
            continue
        per = max(5.0, left / max(1, len(jobs) - k))
        if progress:
            progress(k / len(jobs), f"{sym} {tf}: mengambil data...")
        t1 = time.time()
        try:
            start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=SCAN_DAYS.get(tf, 1460))
            df = closed_only(load_fn(src, sym, tf, start), tf)
            df = df[df.index >= start]
        except Exception as e:
            rows.append(dict(Pasar=sym, TF=tf, Status=f"❌ data gagal: {str(e)[:80]}"))
            continue
        if len(df) < min_bars:
            rows.append(dict(Pasar=sym, TF=tf, Candle=len(df), Status="➖ data terlalu sedikit"))
            continue
        px = float(df.close.mean())
        cf, ca = cost_fn(src, sym, px)
        cfg = dict(cfg_base, cost_frac=cf, cost_abs=ca, pine_cost=cf * 100 + ca / px * 100, time_budget=per * 0.5, tf_min=TF_MINUTES.get(tf, 60))
        if progress:
            progress(k / len(jobs), f"{sym} {tf}: mencari strategi ({len(df):,} candle)...")
        try:
            out, meta, cc = E.search(df, cfg)
        except ValueError as e:
            rows.append(dict(Pasar=sym, TF=tf, Candle=len(df), Status=f"➖ {e}"))
            continue
        n_eff = None
        pbo = None
        try:
            n_eff = R.effective_trials(meta.get("trial_daily"))
            pbo = R.pbo_cscv(meta.get("trial_daily"))
        except Exception:
            pass
        meta.pop("trial_daily", None)
        passed = [o for o in out if o["passed"]]
        for o in passed[:20]:
            _, itr = E.run(o["fam"], o["params"], cc, meta["is_start"], meta["split"], cfg["direction"], cf, ca,
                           lev=cfg.get("lev", 1.0), daily=True)
            o["dsr"] = R.deflated_sharpe(itr["daily"], meta["trial_sr"], n_eff)["dsr"]
        best = max(passed[:20], key=lambda o: (o.get("dsr") or 0, o["oos_m"]["pf"])) if passed else None
        key = f"{src}|{sym}|{tf}"
        results[key] = dict(out=out, meta=meta, cfg=cfg, symbol=sym, tf=tf, source=src, id=f"scan_{key}_{time.time():.0f}",
                            start=str(df.index[0].date()), carry_info=None,
                            search_tests=dict(n_eff=n_eff, pbo=pbo))
        row = dict(Pasar=sym, TF=tf, Candle=len(df), Diuji=meta["tested"], **{"Lolos IS": meta["is_pass"], "Lolos tes": len(passed)},
                   Status="✅ ada yang lolos" if passed else "❌ tidak ada yang lolos",
                   **{"PBO %": (pbo["pbo"] * 100) if pbo else np.nan}, Detik=round(time.time() - t1, 1), key=key)
        if best is not None:
            row.update({"Strategi terbaik": E.FAMILIES[best["fam"]]["name"], "IS PF": best["is_m"]["pf"], "OOS PF": best["oos_m"]["pf"],
                        "OOS Net %": best["oos_m"]["net"] * 100, "OOS DD %": best["oos_m"]["maxdd"] * 100,
                        "OOS Trades": best["oos_m"]["trades"], "DSR %": (best.get("dsr") or 0) * 100,
                        "best_idx": out.index(best)})
        rows.append(row)
    if progress:
        progress(1.0, f"Selesai dalam {time.time() - t0:.0f} detik")
    rows.sort(key=lambda r: (0 if (r.get("Lolos tes") or 0) > 0 else 1, -(r.get("DSR %") or -1), -(r.get("OOS PF") or 0)))
    return rows, results
