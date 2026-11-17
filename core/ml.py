"""Machine-learning signal mode (opt-in, scikit-learn imported lazily).

- Features: causal indicator values at each bar's close (no future data).
- Labels: triple barrier (Lopez de Prado): from the close of bar i, which is hit first within H bars:
  upper barrier (+pt x ATR), lower barrier (-sl x ATR) or time. If both are touched in one candle the
  trade counts as a loss (same conservative rule as the backtester).
- Validation: purged walk-forward CV on the in-sample part only (training rows whose label window
  reaches into the test fold are removed, plus an embargo gap). The unseen (OOS) part is never used
  for fitting or for choosing the threshold.
- Trading: enter when P(win) >= threshold, exit with the same barriers (SL/TP/time) in the normal simulator.
"""
from __future__ import annotations
import time
import numpy as np
from numba import njit
from . import engine as E, indicators as ind, robust as R

FEATURE_LABELS = {
    "fundz": "Funding z-score (300 candle)", "fund": "Funding rate", "flow6": "Order flow taker 6 candle",
    "flow24": "Order flow taker 24 candle", "oi6": "Perubahan open interest 6 candle", "oi24": "Perubahan open interest 24 candle",
    "lsr": "Rasio long/short akun", "lsr24": "Perubahan rasio long/short 24 candle",
    "r1": "Return 1 candle", "r3": "Return 3 candle", "r6": "Return 6 candle", "r12": "Return 12 candle",
    "r24": "Return 24 candle", "r48": "Return 48 candle", "atrp": "ATR % harga", "atr_rel": "ATR vs rata-rata 100",
    "rsi7": "RSI 7", "rsi14": "RSI 14", "rsi28": "RSI 28", "adx": "ADX 14", "bbw": "Lebar Bollinger 20",
    "z20": "Z-score 20", "z50": "Z-score 50", "macd": "MACD histogram", "d20": "Jarak ke EMA 20", "d50": "Jarak ke EMA 50",
    "d200": "Jarak ke EMA 200", "stoch": "Posisi di range 14", "don20": "Posisi di Donchian 20", "don55": "Posisi di Donchian 55",
    "rng": "Range candle / ATR", "hsin": "Jam (sin)", "hcos": "Jam (cos)", "dow": "Hari", "vrel": "Volume vs rata-rata 20",
}


class _F32(dict):
    """Stores every feature column as float32 right away (keeps peak memory low on long histories)."""
    def __setitem__(self, k, v):
        super().__setitem__(k, np.asarray(v, np.float32))


def features(cc):
    """(matrix float32 [n, k], names). Every column only uses data up to and including bar i."""
    c, h, l = cc.c, cc.h, cc.l
    f = _F32()
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in (1, 3, 6, 12, 24, 48):
            f[f"r{k}"] = c / ind.shift(c, k) - 1
        f["atrp"] = cc.atr / c
        f["atr_rel"] = cc.atr / ind.sma(cc.atr, 100)
        for n in (7, 14, 28):
            f[f"rsi{n}"] = ind.rsi(c, n) / 100
        f["adx"] = ind.adx(h, l, c, cc.atr) / 100
        sd = ind.stdev(c, 20); m = ind.sma(c, 20)
        f["bbw"] = 4 * sd / m
        f["z20"] = (c - m) / sd
        f["z50"] = (c - ind.sma(c, 50)) / ind.stdev(c, 50)
        mac = ind.ema(c, 12) - ind.ema(c, 26)
        f["macd"] = (mac - ind.ema(mac, 9)) / c
        for n in (20, 50, 200):
            f[f"d{n}"] = c / ind.ema(c, n) - 1
        hh, ll = ind.rolling_max(h, 14), ind.rolling_min(l, 14)
        f["stoch"] = (c - ll) / (hh - ll)
        for n in (20, 55):
            hh, ll = ind.rolling_max(h, n), ind.rolling_min(l, n)
            f[f"don{n}"] = (c - ll) / (hh - ll)
        f["rng"] = (h - l) / cc.atr
        if cc.tf_min < 1440:
            f["hsin"] = np.sin(2 * np.pi * cc.hour / 24)
            f["hcos"] = np.cos(2 * np.pi * cc.hour / 24)
        f["dow"] = cc.dow / 6.0
        av = getattr(cc, "avail", set())
        if av:
            from . import engine as _E
            if "fund" in av:
                f["fundz"] = _E.fund_z(cc, 300)
                f["fund"] = cc.x["fund"] * 1e4
            if "taker" in av:
                f["flow6"] = _E.flow_ratio(cc, 6)
                f["flow24"] = _E.flow_ratio(cc, 24)
            if "oi" in av:
                f["oi6"] = _E.oi_change(cc, 6)
                f["oi24"] = _E.oi_change(cc, 24)
            if "lsr" in av:
                f["lsr"] = np.log(cc.x["lsr"])
                f["lsr24"] = np.log(cc.x["lsr"] / ind.shift(cc.x["lsr"], 24))
        if cc.has_volume:
            f["vrel"] = np.log(np.maximum(cc.v, 1e-12) / np.maximum(ind.sma(cc.v, 20), 1e-12))
    names = list(f)
    X = np.empty((len(c), len(names)), np.float32)
    for j, k in enumerate(names):
        X[:, j] = f.pop(k)
    X[~np.isfinite(X)] = np.nan
    return X, names


@njit(cache=True)
def triple_barrier(c, h, l, atr, pt, sl, H, end):
    """long_win[i] = upper (+pt ATR) hit before lower (-sl ATR) within H bars; short_win[i] the mirror.
    valid[i] = the whole window is inside [i+1, end)."""
    n = c.shape[0]
    lw = np.zeros(n, np.int8); sw = np.zeros(n, np.int8); ok = np.zeros(n, np.bool_)
    for i in range(n):
        if i + H >= end or np.isnan(atr[i]):
            continue
        ok[i] = True
        a = atr[i]
        up_l = c[i] + pt * a; dn_l = c[i] - sl * a      # long: TP up, SL down
        dn_s = c[i] - pt * a; up_s = c[i] + sl * a      # short: TP down, SL up
        dl = 0; ds = 0
        for j in range(i + 1, i + H + 1):
            if dl == 0:
                hit_sl = l[j] <= dn_l
                hit_tp = h[j] >= up_l
                if hit_sl:
                    dl = -1
                elif hit_tp:
                    dl = 1
            if ds == 0:
                hit_sl = h[j] >= up_s
                hit_tp = l[j] <= dn_s
                if hit_sl:
                    ds = -1
                elif hit_tp:
                    ds = 1
            if dl != 0 and ds != 0:
                break
        lw[i] = 1 if dl == 1 else 0
        sw[i] = 1 if ds == 1 else 0
    return lw, sw, ok


def _model(kind, n):
    if kind == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=300))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=max(20, n // 200), l2_regularization=1.0, random_state=0)


def purged_folds(start, end, k, purge, embargo):
    """Expanding walk-forward folds on [start, end): yields (train_idx_end, test_start, test_end).
    Training rows are [start, test_start - purge - embargo)."""
    edges = np.linspace(start, end, k + 1).astype(int)
    for f in range(1, k):
        ts, te = edges[f], edges[f + 1]
        tr_end = ts - purge - embargo
        if tr_end - start < 500:
            continue
        yield tr_end, ts, te


def _fit_predict(kind, X, y, mask_tr, X_pred, step=1):
    idx = np.flatnonzero(mask_tr)[::step]
    if len(idx) < 200 or len(np.unique(y[idx])) < 2:
        return None, None
    m = _model(kind, len(idx))
    m.fit(X[idx], y[idx])
    return m, m.predict_proba(X_pred)[:, 1]


TOP_SHARES = (0.30, 0.20, 0.10, 0.05)   # trade only the top x% most confident in-sample predictions


def ml_search(cc, is_start, split, cfg, pt=2.0, sl=1.0, H=24, kind="hgb", folds=5, top_shares=TOP_SHARES,
              direction="both", progress=None, max_train=60000):
    """Returns dict with OOF (in-sample) and OOS results, chosen threshold, trial Sharpes, importance."""
    t0 = time.time()
    n = len(cc.c)
    X, names = features(cc)
    lw, sw, ok = triple_barrier(cc.c, cc.h, cc.l, cc.atr, float(pt), float(sl), int(H), int(split))
    feat_ok = ~np.isnan(X).all(axis=1)
    usable = ok & feat_ok
    usable[:is_start] = False
    step = max(1, int(usable.sum() // max_train))          # thin very long histories (memory / CPU on the cloud)
    embargo = max(H, int(0.01 * (split - is_start)))
    oof_l = np.full(n, np.nan); oof_s = np.full(n, np.nan)
    folds_used = list(purged_folds(is_start, split, folds, H, embargo))
    for fi, (tr_end, ts, te) in enumerate(folds_used):
        m_tr = usable.copy(); m_tr[tr_end:] = False
        if direction in ("long", "both"):
            _, p = _fit_predict(kind, X, lw, m_tr, X[ts:te], step)
            if p is not None:
                oof_l[ts:te] = p
        if direction in ("short", "both"):
            _, p = _fit_predict(kind, X, sw, m_tr, X[ts:te], step)
            if p is not None:
                oof_s[ts:te] = p
        if progress:
            progress((fi + 1) / (len(folds_used) + 2), f"Validasi silang {fi + 1}/{len(folds_used)}")
    # final models: all in-sample rows whose label window ends before the unseen part
    m_tr = usable.copy()
    p_l = np.full(n, np.nan); p_s = np.full(n, np.nan)
    ml = ms = None
    if direction in ("long", "both"):
        ml, p = _fit_predict(kind, X, lw, m_tr, X[split:], step)
        if p is not None:
            p_l[split:] = p
    if direction in ("short", "both"):
        ms, p = _fit_predict(kind, X, sw, m_tr, X[split:], step)
        if p is not None:
            p_s[split:] = p
    if progress:
        progress((len(folds_used) + 1) / (len(folds_used) + 2), "Memilih ambang & menguji di data unseen")
    oof_start = folds_used[0][1] if folds_used else split
    exitp = dict(sl_mode=0, sl_atr=float(sl), tp_atr=float(pt), trail_atr=0.0, max_bars=int(H))
    lev, cf, ca = cfg.get("lev", 1.0), cfg["cost_frac"], cfg["cost_abs"]
    z = np.zeros(n, bool)

    def sig(pl, ps, thr):
        a = np.nan_to_num(pl, nan=0.0); b = np.nan_to_num(ps, nan=0.0)
        le = (a >= thr) & (a >= b) if direction in ("long", "both") else z
        se = (b >= thr) & (b > a) if direction in ("short", "both") else z
        return le, z, se, z, {}

    # thresholds = quantiles of the out-of-fold (in-sample) probabilities, so they adapt to the base win rate
    pool = np.concatenate([oof_l[~np.isnan(oof_l)], oof_s[~np.isnan(oof_s)]])
    thresholds = sorted({round(float(np.quantile(pool, 1 - q)), 4) for q in top_shares}) if len(pool) else [0.5]
    trials, trial_sr = [], []
    for thr in thresholds:
        m, tr = E.run("ma_cross", exitp, cc, oof_start, split, "both", cf, ca, sig=sig(oof_l, oof_s, thr), lev=lev, daily=True)
        trials.append((thr, m, tr)); trial_sr.append(m["sharpe_d"])
    good = [t for t in trials if t[1]["trades"] >= max(10, cfg.get("min_trades", 50) // 3)]
    best = max(good or trials, key=lambda t: (t[1]["ret_dd"] if t[1]["net"] > 0 else t[1]["net"]))
    thr = best[0]
    sig_all = sig(np.where(np.isnan(oof_l), p_l, oof_l), np.where(np.isnan(oof_s), p_s, oof_s), thr)
    om, otr = E.run("ma_cross", exitp, cc, split, n, "both", cf, ca, sig=sig_all, lev=lev, daily=True, want_equity=True)
    im, itr = best[1], best[2]
    # base rates (how often a random entry wins) to judge the model honestly
    base_l = float(lw[usable].mean()) if usable.any() else float("nan")
    base_s = float(sw[usable].mean()) if usable.any() else float("nan")
    auc = {}
    try:
        from sklearn.metrics import roc_auc_score
        mm = ~np.isnan(oof_l) & usable
        if mm.sum() > 100 and len(np.unique(lw[mm])) == 2:
            auc["long_cv"] = float(roc_auc_score(lw[mm], oof_l[mm]))
        mm = ~np.isnan(oof_s) & usable
        if mm.sum() > 100 and len(np.unique(sw[mm])) == 2:
            auc["short_cv"] = float(roc_auc_score(sw[mm], oof_s[mm]))
        lw2, sw2, ok2 = triple_barrier(cc.c, cc.h, cc.l, cc.atr, float(pt), float(sl), int(H), int(n))
        mm = ok2 & ~np.isnan(p_l); mm[:split] = False
        if mm.sum() > 100 and len(np.unique(lw2[mm])) == 2:
            auc["long_oos"] = float(roc_auc_score(lw2[mm], p_l[mm]))
        mm = ok2 & ~np.isnan(p_s); mm[:split] = False
        if mm.sum() > 100 and len(np.unique(sw2[mm])) == 2:
            auc["short_oos"] = float(roc_auc_score(sw2[mm], p_s[mm]))
    except Exception:
        pass
    imp = importance(ml or ms, X, lw if ml is not None else sw, usable, names, oof_start, split)
    dsr = R.deflated_sharpe(itr["daily"], np.array(trial_sr), None)
    return dict(threshold=thr, trials=[(t[0], t[1]) for t in trials], is_m=im, oos_m=om, oos_tr=otr, is_tr=itr,
                trial_sr=np.array(trial_sr), dsr=dsr, oos_psr=R.psr(otr["daily"]), auc=auc, base=(base_l, base_s),
                importance=imp, names=names, oof_start=oof_start, exitp=exitp, signals=sig_all, step=step,
                n_train=int(usable.sum() // step), seconds=time.time() - t0, kind=kind, pt=pt, sl=sl, H=H, folds=len(folds_used))


def importance(model, X, y, usable, names, a, b, n=4000):
    """Permutation importance (drop in ROC AUC when one feature is shuffled) on in-sample rows."""
    if model is None:
        return []
    try:
        from sklearn.inspection import permutation_importance
        idx = np.flatnonzero(usable[a:b]) + a
        if len(idx) < 200:
            return []
        rng = np.random.default_rng(0)
        idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
        r = permutation_importance(model, X[idx], y[idx], scoring="roc_auc", n_repeats=3, random_state=0)
        order = np.argsort(-r.importances_mean)
        return [(names[i], float(r.importances_mean[i])) for i in order]
    except Exception:
        return []


# ------------------------------------------------------------------ meta-labeling
def meta_label(fam, p, cc, is_start, split, cfg, direction, kind="auto", folds=4, thresholds=(0.45, 0.5, 0.55, 0.6, 0.65)):
    """Filter an existing rule strategy: a model predicts whether each rule signal becomes a winning trade.
    Trained on in-sample trades only (purged walk-forward CV), applied unchanged to the unseen part."""
    n = len(cc.c)
    X, names = features(cc)
    le, lx, se, sx, lines = E.signals(fam, p, cc)
    lev, cf, ca = cfg.get("lev", 1.0), cfg["cost_frac"], cfg["cost_abs"]
    base_m, btr = E.run(fam, p, cc, is_start, split, direction, cf, ca, lev=lev)
    j = btr["ent"] - 1                                  # signal bar of every in-sample trade
    keep = j >= 0
    j, dr, ret, ex = j[keep], btr["dir"][keep], btr["ret"][keep], btr["ex"][keep]
    if len(j) < 40:
        return dict(error=f"Trade in-sample terlalu sedikit ({len(j)}). Meta-label butuh minimal 40 trade.")
    Xt = np.column_stack([X[j], dr.astype(np.float32)])
    y = (ret > 0).astype(int)
    kind = ("hgb" if len(j) >= 600 else "logreg") if kind == "auto" else kind
    oof = np.full(len(j), np.nan)
    edges = np.linspace(0, len(j), folds + 1).astype(int)
    for f in range(1, folds):
        ts, te = edges[f], edges[f + 1]
        tr = np.flatnonzero(ex[:ts] < j[ts])            # purge: training trades must be closed before the test fold starts
        if len(tr) < 20 or len(np.unique(y[tr])) < 2:
            continue
        m = _model(kind, len(tr)); m.fit(Xt[tr], y[tr])
        oof[ts:te] = m.predict_proba(Xt[ts:te])[:, 1]
    final = _model(kind, len(j)); final.fit(Xt, y)
    # probability for every bar that has a rule signal (OOS uses the final model, IS uses out-of-fold values)
    prob = np.full(n, np.nan)
    sig_bars = np.flatnonzero(le | se)
    oos_bars = sig_bars[sig_bars >= split]
    if len(oos_bars):
        d = np.where(le[oos_bars], 1, -1).astype(np.float32)
        prob[oos_bars] = final.predict_proba(np.column_stack([X[oos_bars], d]))[:, 1]
    prob[j[~np.isnan(oof)]] = oof[~np.isnan(oof)]
    first = j[edges[1]] if len(j) > edges[1] else split

    def gated(thr):
        g = np.nan_to_num(prob, nan=-1.0) >= thr
        return le & g, lx, se & g, sx, lines

    trials = []
    for thr in thresholds:
        m, tr = E.run(fam, p, cc, first, split, direction, cf, ca, sig=gated(thr), lev=lev, daily=True)
        trials.append((thr, m))
    rule_is, _ = E.run(fam, p, cc, first, split, direction, cf, ca, lev=lev)
    good = [t for t in trials if t[1]["trades"] >= 10]
    thr = max(good or trials, key=lambda t: (t[1]["ret_dd"] if t[1]["net"] > 0 else t[1]["net"]))[0]
    meta_is = dict(trials)[thr]
    rule_oos, _ = E.run(fam, p, cc, split, n, direction, cf, ca, lev=lev)
    meta_oos, _ = E.run(fam, p, cc, split, n, direction, cf, ca, sig=gated(thr), lev=lev)
    auc = None
    try:
        from sklearn.metrics import roc_auc_score
        mm = ~np.isnan(oof)
        if mm.sum() > 30 and len(np.unique(y[mm])) == 2:
            auc = float(roc_auc_score(y[mm], oof[mm]))
    except Exception:
        pass
    return dict(threshold=thr, trials=trials, rule_is=rule_is, meta_is=meta_is, rule_oos=rule_oos, meta_oos=meta_oos,
                n_trades=len(j), kind=kind, auc_cv=auc, base_win=float(y.mean()), first=first)
