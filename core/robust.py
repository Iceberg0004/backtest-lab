"""Extra robustness / overfitting tests that go beyond a single in-sample vs unseen split:

- walk_forward():   true walk-forward optimisation (re-optimise each window, test on the next one)
- perm_test():      fixed strategy on shuffled-bar synthetic prices (Masters' bar permutation)
- perm_search_test(): the whole search re-run on shuffled prices (selection-bias test)
- jitter(), cost_stress(): parameter jitter and fee/slippage stress
- deflated_sharpe(): Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014), corrects for the number of trials
- pbo_cscv():       Probability of Backtest Overfitting via CSCV (Bailey, Borwein, Lopez de Prado, Zhu 2015)
- basket_test(), portfolio(): multi-asset validation and an uncorrelated portfolio of passing strategies
- grade():          one transparent 0-100 score built from the tests that were actually run
"""
from __future__ import annotations
import math, time
from itertools import combinations
from statistics import NormalDist
import numpy as np
import pandas as pd
from . import engine as E

_N = NormalDist()
EULER = 0.5772156649


# ------------------------------------------------------------------ helpers
def _ann(ret, bars, bars_per_year):
    """Annualised compound return of `ret` earned over `bars` bars."""
    yrs = bars / max(bars_per_year, 1e-9)
    if yrs <= 0 or ret <= -1:
        return -1.0 if ret <= -1 else 0.0
    return (1 + ret) ** (1 / yrs) - 1


def _quick_search(cc, start, end, cfg, n_variants, rng, min_trades):
    """Small random search on [start, end) that returns the best candidate (by return/drawdown)
    among those meeting the user's targets, or None. Same sampler and engine as the main search."""
    best, best_any = None, None
    seen = set()
    fams = E.allowed_families(cfg["families"], cc.tf_min, cc.has_volume, getattr(cc, "avail", set())) or list(cfg["families"])
    for k in range(n_variants):
        fam = fams[k % len(fams)]
        p = E.drop_unavailable(E.sample_params(fam, rng, cc.tf_min, bool(cfg.get("combo", False))), cc)
        key = (fam, tuple(sorted(p.items())))
        if key in seen:
            continue
        seen.add(key)
        m, _ = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        if m["trades"] < min_trades:
            continue
        if m["pf"] >= cfg["min_pf"] and m["maxdd"] <= cfg["max_dd"] and m["winrate"] >= cfg.get("min_wr", 0):
            if best is None or m["ret_dd"] > best[2]["ret_dd"]:
                best = (fam, p, m)
        if best_any is None or m["ret_dd"] > best_any[2]["ret_dd"]:
            best_any = (fam, p, m)
    return best, best_any


# ------------------------------------------------------------------ 1) walk-forward optimisation
def walk_forward(df, cfg, n_windows=5, mode="anchored", variants=600, train_frac=0.4, progress=None, cc=None, seed=11):
    """Re-optimise on each training window, then trade the chosen strategy (unchanged) on the next window.
    Returns per-window table, the stitched out-of-sample equity and the walk-forward efficiency (WFE):
    annualised OOS return / annualised in-sample return (Pardo). WFE >= 0.5 is usually considered healthy."""
    cc = cc if cc is not None else E.Cache(df)
    n = len(df)
    warm = min(260, n // 10)
    first_train = int(warm + (n - warm) * train_frac)
    edges = np.linspace(first_train, n, n_windows + 1).astype(int)
    train_len = first_train - warm
    is_len_main = max(1, (n - warm) * (1 - cfg["oos_pct"] / 100))
    rng = np.random.default_rng(seed)
    rows, eq_parts, idx_parts = [], [], []
    level = 1.0
    t0 = time.time()
    for w in range(n_windows):
        ta, tb = edges[w], edges[w + 1]
        tr_start = warm if mode == "anchored" else max(warm, ta - train_len)
        min_tr = max(10, int(cfg["min_trades"] * (ta - tr_start) / is_len_main))
        best, best_any = _quick_search(cc, tr_start, ta, cfg, variants, rng, min_tr)
        pick = best or best_any
        row = dict(window=w + 1, train_from=df.index[tr_start], train_to=df.index[ta - 1], test_from=df.index[ta],
                   test_to=df.index[tb - 1], chosen="(tidak ada, tidak trading)", is_net=0.0, is_pf=0.0, oos_net=0.0,
                   oos_pf=0.0, oos_dd=0.0, oos_trades=0, met_target=best is not None, fam=None, params=None)
        if pick is not None:
            fam, p, im = pick
            om, otr = E.run(fam, p, cc, ta, tb, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], want_equity=True,
                            lev=cfg.get("lev", 1.0))
            row.update(chosen=f"{E.FAMILIES[fam]['name']}", fam=fam, params=p, is_net=im["net"], is_pf=im["pf"],
                       oos_net=om["net"], oos_pf=om["pf"], oos_dd=om["maxdd"], oos_trades=om["trades"],
                       is_ann=_ann(im["net"], ta - tr_start, cc.bars_per_year), oos_ann=_ann(om["net"], tb - ta, cc.bars_per_year))
            eq = otr["equity"]
        else:
            row.update(is_ann=0.0, oos_ann=0.0)
            eq = np.ones(tb - ta)
        eq_parts.append(level * eq)
        idx_parts.append(df.index[ta:tb])
        level = level * eq[-1]
        rows.append(row)
        if progress:
            progress((w + 1) / n_windows, w + 1, time.time() - t0)
    equity = pd.Series(np.concatenate(eq_parts), index=idx_parts[0].append(idx_parts[1:]))
    peak = equity.cummax()
    is_ann = np.mean([r["is_ann"] for r in rows])
    oos_ann = np.mean([r["oos_ann"] for r in rows])
    wfe = oos_ann / is_ann if is_ann > 0 else float("nan")
    total = float(equity.iloc[-1] - 1)
    return dict(rows=rows, equity=equity, total=total, maxdd=float((1 - equity / peak).max()), wfe=float(wfe),
                is_ann=float(is_ann), oos_ann=float(oos_ann), oos_ann_total=_ann(total, n - first_train, cc.bars_per_year),
                pos_windows=float(np.mean([r["oos_net"] > 0 for r in rows])), mode=mode, variants=variants,
                seconds=time.time() - t0)


# ------------------------------------------------------------------ 2) permutation tests
def permute_bars(df, rng, keep=0):
    """Bar permutation (T. Masters): shuffle the open-gaps and the intrabar (high, low, close relative to open)
    moves separately. Same return distribution, but any real time structure (trend, cycles) is destroyed."""
    lo, lh, ll, lc = (np.log(df[k].to_numpy(float)) for k in ("open", "high", "low", "close"))
    n = len(df)
    s = max(1, keep)
    gap = lo[s:] - lc[s - 1:-1]
    rh, rl, rc = lh[s:] - lo[s:], ll[s:] - lo[s:], lc[s:] - lo[s:]
    p1, p2 = rng.permutation(n - s), rng.permutation(n - s)
    gap = gap[p2]; rh, rl, rc = rh[p1], rl[p1], rc[p1]
    no, nh, nl, nc = lo.copy(), lh.copy(), ll.copy(), lc.copy()
    prev = lc[s - 1]
    o = prev + np.cumsum(gap + np.r_[0.0, rc[:-1]])  # open_i = close_{i-1} + gap_i
    no[s:] = o; nh[s:] = o + rh; nl[s:] = o + rl; nc[s:] = o + rc
    out = pd.DataFrame({"open": np.exp(no), "high": np.exp(nh), "low": np.exp(nl), "close": np.exp(nc),
                        "volume": df["volume"].to_numpy(float)}, index=df.index)
    for col in ("taker_buy", "fund", "oi", "lsr", "top_lsr"):   # derivatives stay in time (like volume): the
        if col in df.columns:                                    # shuffle breaks any real price <-> data link
            out[col] = df[col].to_numpy(float)
    return out


def _cache_like(d, base=None, upto=None):
    """New indicator cache for (synthetic) prices that keeps the same holding costs as `base`."""
    cc = E.Cache(d)
    if base is not None:
        cc.carry_long = base.carry_long[:len(d)].copy()
        cc.carry_short = base.carry_short[:len(d)].copy()
    return cc


def perm_test(fam, p, df, cfg, start, n_perm=200, seed=3, progress=None, base=None):
    """Fixed strategy on n_perm shuffled price series. p-value = share of fake series where the strategy did
    at least as well as on the real one. Small p (< 0.05) = the edge needs the real price structure."""
    cc = _cache_like(df, base)
    real, _ = E.run(fam, p, cc, start, len(df), cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
    rng = np.random.default_rng(seed)
    nets = []
    for k in range(n_perm):
        d2 = permute_bars(df, rng, keep=start)
        m, _ = E.run(fam, p, _cache_like(d2, base), start, len(df), cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        nets.append(m["net"])
        if progress and k % 10 == 0:
            progress((k + 1) / n_perm)
    nets = np.array(nets)
    pval = (1 + int((nets >= real["net"]).sum())) / (n_perm + 1)
    return dict(real=real["net"], perm=nets, p=pval, n=n_perm)


def perm_search_test(df, cfg, is_start, split, n_perm=20, variants=300, seed=5, progress=None, base=None):
    """Selection-bias test: run the same small search on the real in-sample data and on n_perm shuffled copies.
    If the best result on fake data is often as good as on real data, the search is just finding noise."""
    rng = np.random.default_rng(seed)
    min_tr = cfg["min_trades"]

    def best_score(d):
        cc = _cache_like(d, base)
        _, best_any = _quick_search(cc, is_start, split, cfg, variants, np.random.default_rng(seed), min_tr)
        return best_any[2]["ret_dd"] if best_any else 0.0

    d_is = df.iloc[:split]
    real = best_score(d_is)
    fakes = []
    for k in range(n_perm):
        fakes.append(best_score(permute_bars(d_is, rng, keep=is_start)))
        if progress:
            progress((k + 1) / n_perm)
    fakes = np.array(fakes)
    return dict(real=real, perm=fakes, p=(1 + int((fakes >= real).sum())) / (n_perm + 1), n=n_perm, variants=variants)


# ------------------------------------------------------------------ 3) jitter and cost stress
def jitter(fam, p, cc, start, end, cfg, n=200, pct=0.2, seed=9):
    """Randomly nudge every numeric parameter by up to +/- pct and re-test. A robust strategy stays profitable."""
    rng = np.random.default_rng(seed)
    space = E.FAMILIES[fam]["space"]
    fl_keys = ("sl_atr", "tp_atr", "sl_pct", "tp_pct", "trail_atr", "tp1_r", "be_r", "sl_pip", "sl_buf", "sl_money", "tp_rr",
               "tp_pip", "tp2_r", "trail_pct", "trail_step", "trail_act")
    res = []
    for _ in range(n):
        q = E.full_params(p)
        for k, v in list(q.items()):
            sp = space.get(k)
            num_space = isinstance(sp, tuple) or (isinstance(sp, list) and all(isinstance(x, (int, float)) for x in sp))
            if not (num_space or k in fl_keys or k in ("trend", "adx_thr", "max_bars")):
                continue          # switches / categories (session, mode, MA type...) stay as they are
            if isinstance(v, bool) or v == 0:
                continue
            if isinstance(v, (int, np.integer)):
                lo = 2 if k not in ("trend", "lower", "upper", "exit_level", "k_lo", "k_hi", "adx_thr") else 1
                q[k] = max(lo, int(round(v * (1 + rng.uniform(-pct, pct)))))
            elif isinstance(v, float):
                q[k] = round(max(0.1, v * (1 + rng.uniform(-pct, pct))), 3)
        q = E.fix_params(fam, q, cc.tf_min, combo=True)
        m, _ = E.run(fam, q, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0))
        res.append((m["net"], m["pf"], m["maxdd"], m["trades"]))
    a = np.array(res)
    return dict(net=a[:, 0], pf=a[:, 1], dd=a[:, 2], trades=a[:, 3], share_profit=float((a[:, 0] > 0).mean()),
                share_pf=float((a[:, 1] >= 1.1).mean()), n=n, pct=pct)


def cost_stress(fam, p, cc, start, end, cfg, mults=(1, 2, 3)):
    out = []
    for k in mults:
        m, _ = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"] * k, cfg["cost_abs"] * k, lev=cfg.get("lev", 1.0))
        out.append(dict(mult=k, **m))
    return out


# ------------------------------------------------------------------ 4) overfitting probability
def _skew_kurt(x):
    x = np.asarray(x, float)
    sd = x.std()
    if sd < 1e-12 or len(x) < 3:
        return 0.0, 3.0
    z = (x - x.mean()) / sd
    return float((z ** 3).mean()), float((z ** 4).mean())


def psr(dr, sr_star=0.0):
    """Probabilistic Sharpe Ratio: probability the true (non-annualised) Sharpe is above sr_star."""
    dr = np.asarray(dr, float)
    T = len(dr)
    if T < 10:
        return float("nan")
    sr = E.sharpe(dr)
    g3, g4 = _skew_kurt(dr)
    den = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
    if den <= 0:
        return float("nan")
    return float(_N.cdf((sr - sr_star) * math.sqrt(T - 1) / math.sqrt(den)))


def _sr0(N, sd):
    """Expected maximum Sharpe among N independent trials whose estimated Sharpe ratios have std `sd`
    while their TRUE Sharpe is zero (False Strategy Theorem, Bailey & Lopez de Prado)."""
    if N < 2:
        return 0.0
    return sd * ((1 - EULER) * _N.inv_cdf(1 - 1 / N) + EULER * _N.inv_cdf(1 - 1 / (N * math.e)))


def effective_trials(M):
    """Effective number of independent trials from the correlation of the trials' daily returns
    (participation ratio of the eigenvalues). Many variants are near copies of each other
    (EMA 20/100 vs EMA 21/100), so the effective number is much smaller than the raw count."""
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[0] < 3:
        return None
    M = M[M.std(1) > 0]
    if len(M) < 3:
        return None
    ev = np.clip(np.linalg.eigvalsh(np.corrcoef(M)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def deflated_sharpe(dr, trial_sr, n_eff=None):
    """Deflated Sharpe Ratio = PSR measured against the Sharpe the best of N worthless trials would show by luck.
    Main value: N = effective number of independent trials, std of a Sharpe estimate under the null = 1/sqrt(T-1).
    Strict value: N = every variant tested and the empirical spread of all trials' Sharpe ratios (very harsh when
    many variants lose money because of costs). Both are returned; values close to 1 = edge survives the search."""
    dr = np.asarray(dr, float)
    trial_sr = np.asarray(trial_sr, float)
    N = len(trial_sr)
    T = len(dr)
    out = dict(sr=E.sharpe(dr), N=N, T=T, n_eff=n_eff, dsr=float("nan"), dsr_strict=float("nan"), sr0=float("nan"), sr0_strict=float("nan"))
    if T < 10 or N < 2:
        return out
    ne = max(2.0, min(float(N), n_eff if n_eff else float(N)))
    out["sr0"] = _sr0(ne, 1 / math.sqrt(T - 1))
    out["dsr"] = psr(dr, out["sr0"])
    out["sr0_strict"] = _sr0(N, math.sqrt(max(float(np.var(trial_sr, ddof=1)), 0)))
    out["dsr_strict"] = psr(dr, out["sr0_strict"])
    return out


def pbo_cscv(M, S=12):
    """M: (trials x days) daily returns of the tested variants over the in-sample period.
    Split days into S blocks; for every half/half combination pick the best variant on one half
    and look at its rank on the other half. PBO = share of combinations where the 'winner' ranks
    in the bottom half out-of-sample. < 0.2 good, > 0.5 = the selection process mostly picks luck."""
    M = np.asarray(M, float)
    if M.ndim != 2 or M.shape[0] < 10 or M.shape[1] < S * 10:
        return None
    N, T = M.shape
    blocks = np.array_split(np.arange(T), S)
    bs = np.stack([M[:, b].sum(1) for b in blocks])          # S x N
    bq = np.stack([(M[:, b] ** 2).sum(1) for b in blocks])    # S x N
    bc = np.array([len(b) for b in blocks], float)            # S
    combos = np.array([[1.0 if s in c else 0.0 for s in range(S)] for c in combinations(range(S), S // 2)])
    def sr(mask):
        cnt = mask @ bc
        s1 = mask @ bs; s2 = mask @ bq
        mean = s1 / cnt[:, None]
        var = np.maximum(s2 / cnt[:, None] - mean ** 2, 0)
        sd = np.sqrt(var)
        return np.where(sd > 1e-12, mean / np.where(sd > 1e-12, sd, 1), 0.0)
    is_sr, oos_sr = sr(combos), sr(1 - combos)
    star = is_sr.argmax(1)
    chosen = oos_sr[np.arange(len(combos)), star]
    below = (oos_sr < chosen[:, None]).sum(1)
    equal = (oos_sr == chosen[:, None]).sum(1)
    rank = below + (equal + 1) / 2.0                         # 1..N, ties averaged
    w = rank / (N + 1)
    lam = np.log(w / (1 - w))
    return dict(pbo=float((lam <= 0).mean()), logits=lam, n_trials=N, n_combos=len(combos), S=S)


# ------------------------------------------------------------------ 5) multi-asset + portfolio
def basket_test(fam, p, items, cfg, oos_pct):
    """items: list of (symbol, Cache, cost_frac, cost_abs). Fixed parameters on every asset (no re-optimising)."""
    rows = []
    for sym, cc, cf, ca in items:
        n = len(cc.c)
        st = min(260, n // 4)
        split = int(n * (1 - oos_pct / 100))
        fm, _ = E.run(fam, p, cc, st, n, cfg["direction"], cf, ca, lev=cfg.get("lev", 1.0))
        om, _ = E.run(fam, p, cc, split, n, cfg["direction"], cf, ca, lev=cfg.get("lev", 1.0))
        ok = fm["net"] > 0 and fm["pf"] >= 1.1 and fm["trades"] >= 20 and om["net"] > 0
        rows.append(dict(symbol=sym, trades=fm["trades"], net=fm["net"], pf=fm["pf"], maxdd=fm["maxdd"], winrate=fm["winrate"],
                         oos_net=om["net"], oos_pf=om["pf"], passed=ok, start=cc.index[st], end=cc.index[-1]))
    return rows


def portfolio(cands, cc, start, end, split, cfg, max_n=5, max_corr=0.5):
    """cands: list of (label, fam, params) ordered by preference (in-sample rank). Greedily keeps strategies
    whose daily returns correlate < max_corr with those already kept; equal weight, rebalanced daily."""
    series = {}
    for lab, fam, p in cands:
        _, tr = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0), daily=True)
        series[lab] = tr["daily"]
    de = cc.day_end[(cc.day_end > start) & (cc.day_end < end)]
    idx = cc.index[de]
    D = pd.DataFrame(series, index=idx)
    keep = []
    Dis = D[D.index < cc.index[split]]          # choose with in-sample correlations only (unseen part stays an exam)
    for lab in D.columns:
        if Dis[lab].std() == 0:
            continue
        if all(abs(Dis[lab].corr(Dis[k])) < max_corr for k in keep):
            keep.append(lab)
        if len(keep) >= max_n:
            break
    if not keep:
        return None
    sel = D[keep]
    comb = sel.mean(1)
    eq = (1 + comb).cumprod()
    dd = 1 - eq / eq.cummax()
    split_t = cc.index[split]
    def stats(r):
        if len(r) < 2:
            return dict(net=0.0, maxdd=0.0, sharpe=0.0)
        e = (1 + r).cumprod()
        return dict(net=float(e.iloc[-1] - 1), maxdd=float((1 - e / e.cummax()).max()),
                    sharpe=E.sharpe(r.to_numpy()) * math.sqrt(cc.days_per_year))
    ind_stats = {k: stats(sel[k]) for k in keep}
    return dict(keep=keep, corr=D.corr().loc[keep, keep] if len(keep) > 1 else D[keep].corr(), all_corr=D.corr(),
                equity=eq, dd=dd, comb=comb, single=sel, stats_all=stats(comb), stats_is=stats(comb[comb.index < split_t]),
                stats_oos=stats(comb[comb.index >= split_t]), ind_stats=ind_stats)


def similarity(cands, cc, start, end, cfg, thr=0.8):
    """Correlation of daily returns (in-sample window given by start/end), a cluster order for the heatmap and
    groups of near-duplicates (corr >= thr with the group's best member). cands ordered by preference."""
    series = {}
    for lab, fam, p in cands:
        _, tr = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg.get("lev", 1.0), daily=True)
        if np.std(tr["daily"]) > 0:
            series[lab] = tr["daily"]
    if len(series) < 2:
        return None
    D = pd.DataFrame(series)
    C = D.corr().fillna(0.0)
    labs = list(C.columns)
    # groups: walk in preference order; join the first group whose representative (its best member) has corr >= thr,
    # else start a new group. No chaining (A~B, B~C does not make A a twin of C); negative correlation is not a twin.
    Cv = C.to_numpy()
    reps, groups = [], {}
    for i in range(len(labs)):
        g = next((r for r in reps if Cv[i, r] >= thr), None)
        if g is None:
            reps.append(i); groups[i] = [labs[i]]
        else:
            groups[g].append(labs[i])
    groups = sorted(groups.values(), key=lambda g: labs.index(g[0]))
    # heatmap order: groups together, members in preference order
    order = [x for g in groups for x in g]
    return dict(corr=C.loc[order, order], groups=groups, n=len(labs))


# ------------------------------------------------------------------ 6) one clear score
def grade(tests: dict):
    """tests may contain: oos_pass(bool), dsr(0-1), pbo(0-1), perm_p(0-1), wfe(float), jitter(0-1 share profitable),
    basket(0-1 share of assets passing), cost2x(bool profitable at 2x costs). Only tests that were run count;
    the score is rescaled to 0-100 over those. Returns (score, letter, parts)."""
    W = dict(oos_pass=15, dsr=20, pbo=15, perm_p=15, wfe=10, jitter=10, basket=10, cost2x=5)
    parts = {}
    for k, w in W.items():
        v = tests.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if k == "oos_pass" or k == "cost2x":
            s = 1.0 if v else 0.0
        elif k == "dsr":
            s = min(1.0, max(0.0, (v - 0.5) / 0.45))            # 0.5 -> 0, 0.95 -> full
        elif k == "pbo":
            s = min(1.0, max(0.0, (0.6 - v) / 0.5))             # 0.6 -> 0, 0.1 -> full
        elif k == "perm_p":
            s = 1.0 if v <= 0.01 else 0.8 if v <= 0.05 else 0.4 if v <= 0.1 else 0.0
        elif k == "wfe":
            s = min(1.0, max(0.0, v / 0.6))                      # 0.6+ -> full
        else:
            s = min(1.0, max(0.0, float(v)))
        parts[k] = (s, w)
    if not parts:
        return None, "-", parts
    tot_w = sum(w for _, w in parts.values())
    score = 100 * sum(s * w for s, w in parts.values()) / tot_w
    letter = "A" if score >= 80 else "B" if score >= 65 else "C" if score >= 50 else "D" if score >= 35 else "E"
    return score, letter, parts
