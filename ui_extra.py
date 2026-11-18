"""Extra pages/tabs for Backtest Lab: report card (overfit score), robustness tests, multi-asset check,
realism (1-minute SL/TP check, funding/swap), walk-forward optimisation, portfolio, save & compare.
Heavy tests only run when their button is pressed; results are kept in the session."""
from __future__ import annotations
import json, math, time, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from core import engine as E, robust as R, realism as X

TZ = "Asia/Jakarta"
ss = st.session_state
PLOT = dict(template="plotly_dark", margin=dict(t=40, b=10, l=10, r=10))


def wib(idx):
    return pd.DatetimeIndex(idx).tz_convert(TZ)


def pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def tests_for(skey):
    return ss.setdefault("tests", {}).setdefault(skey, {})


# ------------------------------------------------------------------ after search (cheap, automatic)
def after_search(res, cc):
    """PBO for the whole search + DSR for every listed strategy. Runs once right after the search."""
    meta, cfg = res["meta"], res["cfg"]
    st_ = res.setdefault("search_tests", {})
    try:
        st_["n_eff"] = R.effective_trials(meta.get("trial_daily"))
    except Exception:
        st_["n_eff"] = None
    try:
        st_["pbo"] = R.pbo_cscv(meta.get("trial_daily"))
    except Exception as e:  # never break the search page because of an extra statistic
        st_["pbo"] = None
        st_["pbo_error"] = str(e)
    meta.pop("trial_daily", None)  # free memory (Streamlit Cloud has ~1 GB)
    for o in res["out"]:
        _, itr = E.run(o["fam"], o["params"], cc, meta["is_start"], meta["split"], cfg["direction"], cfg["cost_frac"],
                       cfg["cost_abs"], lev=cfg["lev"], daily=True)
        o["dsr"] = R.deflated_sharpe(itr["daily"], meta["trial_sr"], st_.get("n_eff"))["dsr"]


# ------------------------------------------------------------------ report card
def dsr_pick(d):
    """DSR used for the grade: the textbook strict version if the user switched it on."""
    d = d or {}
    return d.get("dsr_strict") if st.session_state.get("strict_dsr") else d.get("dsr")


def grade_inputs(o, res, t):
    st_ = res.get("search_tests", {})
    pbo = st_.get("pbo")
    wfo = st_.get("wfo")
    b = t.get("basket")
    return dict(oos_pass=bool(t.get("strict_pass", o["passed"])), dsr=dsr_pick(t.get("dsr")), pbo=pbo["pbo"] if pbo else None,
                perm_p=t["perm"]["p"] if "perm" in t else None,
                wfe=(wfo["wfe"] if wfo and not math.isnan(wfo["wfe"]) else (0.0 if wfo else None)),
                jitter=t["jitter"]["share_profit"] if "jitter" in t else None,
                basket=(np.mean([r["passed"] for r in b]) if b else None),
                cost2x=(t["stress"][1]["net"] > 0) if "stress" in t else None,
                cap_d=bool(o.get("overfit")) or o["oos_m"]["net"] < 0 or o["is_m"]["net"] < 0)


def render_report(ctx):
    o, res, cc, cfg = ctx["o"], ctx["res"], ctx["cc"], ctx["cfg"]
    t = tests_for(ctx["skey"])
    if "dsr" not in t:
        im, itr = E.run(o["fam"], o["params"], cc, ctx["is_start"], ctx["split"], cfg["direction"], cfg["cost_frac"],
                        cfg["cost_abs"], lev=cfg["lev"], daily=True)
        om, otr = E.run(o["fam"], o["params"], cc, ctx["split"], ctx["n"], cfg["direction"], cfg["cost_frac"],
                        cfg["cost_abs"], lev=cfg["lev"], daily=True)
        t["dsr"] = R.deflated_sharpe(itr["daily"], res["meta"]["trial_sr"], res.get("search_tests", {}).get("n_eff"))
        t["oos_psr"] = R.psr(otr["daily"])
        t["is_sharpe"], t["oos_sharpe"] = im.get("sharpe", 0.0), om.get("sharpe", 0.0)
    if "strict_pass" not in t:
        # the grade always uses the STRICT checks, also for results found in Vela mode
        stab = o["stab"]
        if stab != stab:
            stab, _ = E.stability(o["fam"], o["params"], cc, ctx["is_start"], ctx["split"], cfg["direction"], cfg["cost_frac"],
                                  cfg["cost_abs"], lev=cfg["lev"])
        ism, om = o["is_m"], o["oos_m"]
        overfit = (om["trades"] > 0 and om["pf"] < 0.6 * ism["pf"]) or om["net"] < 0
        t["strict_pass"] = bool(o["is_pass"] and o["oos_ok"] and o["wf"] >= cfg["min_wf"] and stab >= cfg["min_stab"] and not overfit)
        t["strict_stab"] = stab
    gi = grade_inputs(o, res, t)
    score, letter, parts = R.grade(gi)
    t["grade"] = (score, letter)
    pbo = res.get("search_tests", {}).get("pbo")
    colors = {"A": "#00c853", "B": "#64dd17", "C": "#ffd600", "D": "#ff9100", "E": "#ff1744"}
    done = len([k for k in parts if k != "cap_d"])
    st.markdown(
        f"<div style='padding:10px 14px;border-radius:8px;background:#1e1e1e;border-left:6px solid {colors.get(letter, '#888')}'>"
        f"<span style='font-size:1.6em;font-weight:700;color:{colors.get(letter, '#888')}'>Rapor ketahanan: {letter}</span> "
        f"<span style='font-size:1.2em'>· {score:.0f}/100</span><br><span style='color:#aaa'>Dihitung dari {done} dari 8 tes. "
        f"Tes lain bisa dijalankan di tab 'Uji ketahanan', 'Multi-aset' dan halaman 'Walk-forward'. "
        f"Ini hasil uji data lama, bukan jaminan dan bukan saran.</span></div>", unsafe_allow_html=True)
    if "cap_d" in parts:
        st.warning("Nilai dibatasi maksimal D: strategi ini rugi di data unseen / in-sample, atau ada tanda overfitting.")
    c1, c2, c3, c4 = st.columns(4)
    d = t["dsr"]
    ann = math.sqrt(ctx["cc"].days_per_year)
    ne = d.get("n_eff")
    c1.metric("Deflated Sharpe (DSR)", pct(d["dsr"], 0), help="Peluang hasilnya bukan kebetulan, SETELAH memperhitungkan "
              f"banyaknya variasi yang dicoba ({d['N']:,} variasi, setara ±{ne or d['N']:.0f} variasi yang benar-benar berbeda). "
              "Di atas 95% = bagus. Di bawah 50% = kemungkinan besar cuma untung-untungan.")
    c2.metric("Peluang overfit (PBO)", pct(pbo["pbo"], 0) if pbo else "-",
              help="Dari seluruh pencarian: seberapa sering 'juara' di separuh data ternyata di bawah rata-rata di separuh lainnya. "
                   "Di bawah 20% bagus, di atas 50% buruk.")
    c3.metric("Sharpe/tahun IS → unseen", f"{t['is_sharpe']:.2f} → {t['oos_sharpe']:.2f}")
    c4.metric("Peluang edge nyata di unseen (PSR)", pct(t["oos_psr"], 0),
              help="Probabilistic Sharpe Ratio di data unseen saja (1 strategi, tanpa koreksi jumlah percobaan).")
    st.caption(f"Sharpe in-sample strategi ini {d['sr'] * ann:.2f}/tahun. Dengan variasi sebanyak ini, Sharpe setinggi "
               f"±{d['sr0'] * ann:.2f}/tahun bisa muncul hanya karena kebetulan. Versi paling ketat (semua {d['N']:,} variasi dianggap "
               f"terpisah, sebaran Sharpe asli): batas {d['sr0_strict'] * ann:.2f}/tahun, DSR {pct(d['dsr_strict'], 0)}.")
    with st.expander("Rincian skor (apa saja yang dihitung)"):
        names = dict(oos_pass="Lolos tes dasar versi ketat (unseen, walk-forward, stabilitas, tidak overfit)", dsr="Deflated Sharpe Ratio",
                     pbo="PBO (CSCV) seluruh pencarian", perm_p="Tes harga acak (permutation)", wfe="Walk-forward efficiency",
                     jitter="Parameter digeser ±% tetap untung", basket="Lolos di aset lain", cost2x="Tetap untung saat biaya 2x")
        rows = []
        for k, nm in names.items():
            if k in parts:
                s_, w = parts[k]
                rows.append({"Tes": nm, "Nilai": _fmt_test(k, gi[k]), "Poin": f"{s_ * w:.1f} / {w}"})
            else:
                rows.append({"Tes": nm, "Nilai": "belum dijalankan", "Poin": "-"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        st.caption("Skor = jumlah poin dibagi poin maksimal dari tes yang sudah dijalankan. A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, "
                   "selain itu E. DSR: Bailey & López de Prado (2014). PBO: Bailey, Borwein, López de Prado & Zhu (2015), "
                   f"dihitung dari {pbo['n_trials'] if pbo else 0} variasi acak pertama, {pbo['S'] if pbo else 0} blok, "
                   f"{pbo['n_combos'] if pbo else 0} kombinasi.")


def _fmt_test(k, v):
    if k in ("oos_pass", "cost2x"):
        return "ya" if v else "tidak"
    if k == "perm_p":
        return f"p = {v:.3f}"
    if k == "wfe":
        return f"{v:.2f}"
    return pct(v, 0)


# ------------------------------------------------------------------ robustness tab
def render_robust(ctx):
    o, cc, cfg, dfc = ctx["o"], ctx["cc"], ctx["cfg"], ctx["dfc"]
    t = tests_for(ctx["skey"])
    st.markdown("Tes-tes ini mencoba **menjatuhkan** strategi. Strategi yang bagus tetap berdiri.")

    # --- cost stress (fast, automatic)
    if "stress" not in t:
        t["stress"] = R.cost_stress(o["fam"], o["params"], cc, ctx["is_start"], ctx["n"], cfg)
    st.markdown("##### 1. Biaya dinaikkan (fee + slippage + spread)")
    sdf = pd.DataFrame([{"Biaya": f"{r['mult']}x", "Net %": r["net"] * 100, "Profit factor": r["pf"], "Max DD %": r["maxdd"] * 100,
                         "Trades": r["trades"]} for r in t["stress"]])
    st.dataframe(sdf.style.format({"Net %": "{:.2f}", "Profit factor": "{:.2f}", "Max DD %": "{:.2f}"}), hide_index=True, width='stretch')
    ok2 = t["stress"][1]["net"] > 0
    (st.success if ok2 else st.warning)("Biaya 2x: " + ("masih untung." if ok2 else "sudah rugi. Edge-nya tipis, sangat tergantung biaya."))

    # --- permutation test (fixed strategy)
    st.markdown("##### 2. Tes harga acak (strategi ini)")
    st.caption("Urutan candle diacak ratusan kali (sebaran naik-turunnya sama, tapi pola/trennya hilang). Strategi lalu dijalankan "
               "di harga palsu itu. Kalau hasil di harga palsu sering sama bagusnya, berarti edge-nya bukan dari pola pasar yang nyata.")
    a, b = st.columns([1, 3])
    n_perm = a.selectbox("Jumlah acakan", [100, 200, 500], index=1, key=f"np_{ctx['skey']}")
    if b.button("▶️ Jalankan tes harga acak", key=f"bp_{ctx['skey']}"):
        bar = st.progress(0.0, text="Mengacak harga...")
        t["perm"] = R.perm_test(o["fam"], o["params"], dfc, cfg, ctx["is_start"], n_perm,
                                progress=lambda f: bar.progress(min(f, 1.0), text=f"Mengacak harga... {f * 100:.0f}%"), base=cc)
        bar.empty()
    if "perm" in t:
        pt = t["perm"]
        fig = go.Figure(go.Histogram(x=pt["perm"] * 100, nbinsx=40, marker_color="#78909c", name="Harga acak"))
        fig.add_vline(x=pt["real"] * 100, line_color="#ffd600", line_width=3, annotation_text="harga asli")
        fig.update_layout(title=f"Net % di {pt['n']} harga acak vs harga asli", height=300, **PLOT)
        st.plotly_chart(fig, width='stretch')
        msg = f"p-value = {pt['p']:.3f} (harga acak yang hasilnya ≥ harga asli: {(pt['perm'] >= pt['real']).mean() * 100:.1f}%)."
        (st.success if pt["p"] <= 0.05 else st.warning)(msg + (" Bagus: edge-nya butuh pola harga asli." if pt["p"] <= 0.05 else
                                                                " Belum meyakinkan: harga acak pun sering sama bagusnya."))

    # --- permutation of the whole search (selection bias)
    st.markdown("##### 3. Tes harga acak (seluruh pencarian)")
    st.caption("Pencarian kecil diulang di harga acak. Kalau 'juara' di harga acak sama hebatnya dengan juara di harga asli, "
               "berarti pencariannya cuma menemukan kebetulan. Lebih berat, butuh beberapa puluh detik.")
    a, b, c = st.columns([1, 1, 2])
    nps = a.selectbox("Jumlah acakan", [10, 20, 50], index=1, key=f"nps_{ctx['skey']}")
    nv = b.selectbox("Variasi per pencarian", [200, 300, 500], index=1, key=f"nv_{ctx['skey']}")
    if c.button("▶️ Jalankan tes pencarian acak", key=f"bps_{ctx['skey']}"):
        bar = st.progress(0.0, text="Mencari di harga acak...")
        t["perm_search"] = R.perm_search_test(dfc, cfg, ctx["is_start"], ctx["split"], nps, nv,
                                              progress=lambda f: bar.progress(min(f, 1.0), text=f"Mencari di harga acak... {f * 100:.0f}%"),
                                              base=cc)
        bar.empty()
    if "perm_search" in t:
        ps = t["perm_search"]
        fig = go.Figure(go.Histogram(x=ps["perm"], nbinsx=20, marker_color="#78909c"))
        fig.add_vline(x=ps["real"], line_color="#ffd600", line_width=3, annotation_text="harga asli")
        fig.update_layout(title="Return/DD terbaik (in-sample) per pencarian", height=280, **PLOT)
        st.plotly_chart(fig, width='stretch')
        (st.success if ps["p"] <= 0.1 else st.warning)(
            f"p-value = {ps['p']:.3f} ({ps['n']} acakan × {ps['variants']} variasi). "
            + ("Pencarian di harga asli jelas lebih bagus dari harga acak." if ps["p"] <= 0.1 else
               "Juara di harga acak sering sama bagusnya. Hati-hati, bisa jadi cuma kebetulan.")
            + (f" Catatan: dengan {ps['n']} acakan, p terkecil yang mungkin = {1 / (ps['n'] + 1):.3f}." ))

    # --- jitter
    st.markdown("##### 4. Parameter digeser acak")
    st.caption("Semua angka pengaturan digeser sedikit secara acak (misal EMA 20 jadi 18 atau 23). Strategi yang kuat tetap untung.")
    a, b, c = st.columns([1, 1, 2])
    jp = a.selectbox("Geser maksimal", [0.1, 0.2, 0.3], index=1, format_func=lambda x: f"±{x * 100:.0f}%", key=f"jp_{ctx['skey']}")
    jn = b.selectbox("Jumlah percobaan", [100, 200, 400], index=1, key=f"jn_{ctx['skey']}")
    if c.button("▶️ Jalankan tes geser parameter", key=f"bj_{ctx['skey']}"):
        with st.spinner("Menguji..."):
            t["jitter"] = R.jitter(o["fam"], o["params"], cc, ctx["is_start"], ctx["n"], cfg, jn, jp)
    if "jitter" in t:
        j = t["jitter"]
        fig = go.Figure(go.Histogram(x=j["pf"], nbinsx=40, marker_color="#26a69a"))
        fig.add_vline(x=1.0, line_color="#ef5350", annotation_text="PF 1 (impas)")
        fig.update_layout(title=f"Profit factor dari {j['n']} versi yang digeser ±{j['pct'] * 100:.0f}%", height=280, **PLOT)
        st.plotly_chart(fig, width='stretch')
        (st.success if j["share_profit"] >= 0.7 else st.warning)(
            f"{j['share_profit'] * 100:.0f}% versi tetap untung, {j['share_pf'] * 100:.0f}% punya PF ≥ 1.1, "
            f"median PF {np.median(j['pf']):.2f}.")


# ------------------------------------------------------------------ multi-asset tab
CRYPTO_BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
FX_BASKET = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "XAUUSD"]


def render_basket(ctx):
    o, cfg, res = ctx["o"], ctx["cfg"], ctx["res"]
    t = tests_for(ctx["skey"])
    src = res["source"]
    syms = ctx["symbols"]
    default = CRYPTO_BASKET if src == "binance" else FX_BASKET if src == "dukascopy" else syms[:5]
    st.markdown("Strategi yang **sama persis** (tanpa diubah) dijalankan di aset lain. Kalau cuma bagus di satu aset, "
                "bisa jadi itu kebetulan.")
    basket = st.multiselect("Aset yang diuji", sorted(set(syms) | set(default)), default=default,
                            key=f"bk_{ctx['skey']}")
    st.caption("Lolos = untung di seluruh periode, profit factor ≥ 1.1, minimal 20 trade, dan tetap untung di bagian data terbaru "
               f"({cfg['oos_pct']}% terakhir). Data aset baru diambil dulu, pertama kali bisa beberapa menit.")
    if st.button("▶️ Jalankan tes multi-aset", key=f"bb_{ctx['skey']}"):
        from concurrent.futures import ThreadPoolExecutor
        items, errs = [], []
        with st.spinner(f"Mengambil data {', '.join(basket)} (sekaligus, pertama kali bisa beberapa menit)..."):
            def job(s_):
                try:
                    return s_, ctx["load_symbol_raw"](s_), None
                except Exception as e:
                    return s_, None, str(e)
            with ThreadPoolExecutor(max_workers=3 if res["source"] == "dukascopy" else 5) as ex:
                got = list(ex.map(job, basket))
        for s_, d, err in got:
            if err:
                errs.append(f"{s_}: {err}"); continue
            if d is None or len(d) < 600:
                errs.append(f"{s_}: data terlalu sedikit"); continue
            cc = ctx["make_cache"](d, s_)
            cf, ca = ctx["cost_for"](s_, float(d.close.mean()))
            items.append((s_, cc, cf, ca))
        t["basket"] = R.basket_test(o["fam"], o["params"], items, cfg, cfg["oos_pct"])
        t["basket_err"] = errs
    if "basket" in t:
        rows = t["basket"]
        if rows:
            df = pd.DataFrame([{"Aset": r["symbol"], "Hasil": "✅ Lolos" if r["passed"] else "❌ Gagal", "Net %": r["net"] * 100,
                                "Profit factor": r["pf"], "Max DD %": r["maxdd"] * 100, "Win rate %": r["winrate"] * 100,
                                "Trades": r["trades"], "Net % data terbaru": r["oos_net"] * 100, "PF data terbaru": r["oos_pf"],
                                "Data dari": wib([r["start"]])[0].strftime("%d %b %Y")} for r in rows])
            st.dataframe(df.style.format({k: "{:.2f}" for k in ["Net %", "Profit factor", "Max DD %", "Win rate %", "Net % data terbaru", "PF data terbaru"]}),
                         hide_index=True, width='stretch')
            share = np.mean([r["passed"] for r in rows])
            (st.success if share >= 0.6 else st.warning)(f"Lolos di {sum(r['passed'] for r in rows)} dari {len(rows)} aset ({share * 100:.0f}%).")
        for e in t.get("basket_err", []):
            st.caption(f"⚠️ {e}")
        if res["source"] == "dukascopy" and (ctx["cfg"].get("carry") or {}).get("kind") == "swap":
            st.caption("Swap tidak dihitung untuk aset lain (tiap pair beda swap).")


# ------------------------------------------------------------------ realism tab
def render_realism(ctx):
    o, cc, cfg, res = ctx["o"], ctx["cc"], ctx["cfg"], ctx["res"]
    t = tests_for(ctx["skey"])
    st.markdown("##### 1. Stop loss vs take profit di candle yang sama")
    st.caption("Kalau dalam satu candle harga menyentuh stop loss DAN take profit, backtest biasa tidak tahu mana yang duluan. "
               "Aplikasi ini default menganggap stop loss duluan (paling aman). Tombol ini membuka candle 1 menit untuk candle-candle itu "
               "dan memakai urutan yang sebenarnya.")
    _pp = E.full_params(o["params"])
    if not E.has_target(_pp):
        st.info("Strategi ini tidak memakai take profit tetap, jadi masalah ini tidak terjadi.")
    elif st.button("▶️ Cek dengan candle 1 menit", key=f"bi_{ctx['skey']}"):
        bar = st.progress(0.0, text="Mengambil candle 1 menit...")
        info, _ = X.intrabar(o["fam"], o["params"], cc, ctx["is_start"], ctx["n"], cfg, res["source"], res["symbol"], res["tf"],
                             progress=lambda f: bar.progress(min(f, 1.0), text=f"Mengambil candle 1 menit... {f * 100:.0f}%"))
        info.pop("resolve", None)
        t["intrabar"] = info
        bar.empty()
    if "intrabar" in t:
        i = t["intrabar"]
        if i.get("error") and i["n_checked"] == 0 and i["n_amb"] > 0:
            st.warning(f"Tidak bisa mengambil candle kecil: {i['error']}")
        b, a = i["before"], i["after"]
        df = pd.DataFrame([{"Versi": "Aturan aman (SL duluan)", "Net %": b["net"] * 100, "PF": b["pf"], "Win rate %": b["winrate"] * 100, "Max DD %": b["maxdd"] * 100},
                           {"Versi": f"Urutan asli dari candle {i['ltf']}", "Net %": a["net"] * 100, "PF": a["pf"], "Win rate %": a["winrate"] * 100, "Max DD %": a["maxdd"] * 100}])
        st.dataframe(df.style.format({k: "{:.2f}" for k in ["Net %", "PF", "Win rate %", "Max DD %"]}), hide_index=True, width='stretch')
        st.caption(f"{i['n_amb']} trade berakhir di candle yang menyentuh SL dan TP sekaligus; {i['n_checked']} candle dicek; "
                   f"{i['n_tp_first']} ternyata TP duluan; {i['n_unknown']} tidak bisa dipastikan (tetap dianggap SL)."
                   + (" Dibatasi 250 candle agar cepat." if i.get("capped") else ""))
    st.markdown("##### 2. Biaya menginap (funding / swap)")
    car = cfg.get("carry") or {}
    if car.get("kind") == "funding":
        inf = ctx.get("carry_info") or {}
        st.write(f"Funding rate perpetual **aktif** · sumber: {inf.get('note', '-')} · rata-rata {inf.get('mean_8h', 0) * 100:.4f}% per 8 jam · "
                 f"{inf.get('share_default', 0) * 100:.0f}% candle memakai tarif dasar 0,01%/8 jam (sebelum ada data perpetual).")
    elif car.get("kind") == "swap":
        st.write(f"Swap **aktif**: Long {car['long']:+.2f}%/tahun, Short {car['short']:+.2f}%/tahun (positif = bayar, negatif = dapat).")
    else:
        st.write("Tidak aktif. Bisa dinyalakan di tab 'Cari Strategi' sebelum mencari.")
    if car:
        cc0 = ctx["cc_nocarry"]
        m0, tr0 = E.run(o["fam"], o["params"], cc0, ctx["is_start"], ctx["n"], cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
        m1, tr1 = E.run(o["fam"], o["params"], cc, ctx["is_start"], ctx["n"], cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
        st.dataframe(pd.DataFrame([{"Versi": "Tanpa biaya menginap", "Net %": m0["net"] * 100, "PF": m0["pf"], "Max DD %": m0["maxdd"] * 100},
                                   {"Versi": "Dengan biaya menginap", "Net %": m1["net"] * 100, "PF": m1["pf"], "Max DD %": m1["maxdd"] * 100}])
                     .style.format({k: "{:.2f}" for k in ["Net %", "PF", "Max DD %"]}), hide_index=True, width='stretch')
        st.caption(f"Total biaya menginap yang dibayar/diterima per trade, dijumlah: {tr1['carry'].sum() * 100:+.2f}% (positif = bayar).")


# ------------------------------------------------------------------ walk-forward page
def render_wfo(res, dfc, cc):
    cfg = res["cfg"]
    st.markdown("**Walk-forward sungguhan:** data dibagi beberapa jendela. Di tiap jendela, aplikasi mencari strategi terbaik "
                "HANYA dari data sebelumnya, lalu memakainya tanpa diubah di jendela berikutnya. Hasil semua jendela berikutnya "
                "disambung jadi satu kurva. Ini meniru kalau kamu benar-benar memakai aplikasi ini dari waktu ke waktu.")
    a, b, c, d = st.columns(4)
    nw = a.slider("Jumlah jendela uji", 3, 10, 5)
    mode = b.radio("Data latih", ["anchored", "rolling"], horizontal=True,
                   format_func={"anchored": "Makin panjang (anchored)", "rolling": "Geser (rolling)"}.get)
    nv = c.select_slider("Variasi per jendela", [300, 600, 1000, 2000], value=600)
    tf_ = d.slider("Data latih awal (%)", 20, 60, 40)
    st.caption(f"Memakai jenis strategi, target, biaya dan arah yang sama dengan pencarian terakhir ({res['symbol']} {res['tf']}). "
               "Pilihan di tiap jendela = return/drawdown terbaik yang memenuhi target. Kalau tidak ada yang memenuhi target, "
               "dipakai yang terbaik saja (ditandai).")
    if st.button("▶️ Jalankan walk-forward", type="primary"):
        bar = st.progress(0.0, text="Jendela 1...")
        w = R.walk_forward(dfc, cfg, nw, mode, nv, tf_ / 100, cc=cc,
                           progress=lambda f, k, s: bar.progress(min(f, 1.0), text=f"Jendela {k}/{nw} selesai · {s:.0f} detik"))
        bar.empty()
        res.setdefault("search_tests", {})["wfo"] = w
    w = res.get("search_tests", {}).get("wfo")
    if not w:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hasil gabungan semua jendela uji", pct(w["total"]), help="Return dari kurva yang disambung (hanya data yang belum pernah dilihat saat memilih).")
    c2.metric("Max drawdown gabungan", pct(w["maxdd"]))
    c3.metric("Walk-forward efficiency", "-" if math.isnan(w["wfe"]) else f"{w['wfe']:.2f}",
              help="Return tahunan di jendela uji dibagi return tahunan di data latih. ≥ 0.5 biasanya dianggap sehat; "
                   "negatif = strategi yang dipilih rugi saat dipakai.")
    c4.metric("Jendela uji yang untung", pct(w["pos_windows"], 0))
    eq = w["equity"]
    first = eq.index[0]
    bh = dfc.close[dfc.index >= first]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wib(eq.index), y=eq.values * 10000, name="Walk-forward (strategi berganti tiap jendela)", line=dict(color="#29b6f6")))
    fig.add_trace(go.Scatter(x=wib(bh.index), y=bh.values / bh.values[0] * 10000, name="Beli & tahan (pembanding)", line=dict(color="#888", dash="dot")))
    for r in w["rows"]:
        fig.add_vline(x=wib([r["test_from"]])[0], line_color="rgba(255,255,255,0.15)")
    fig.update_layout(title="Kurva gabungan jendela uji (modal awal 10.000)", height=380, **PLOT)
    st.plotly_chart(fig, width='stretch')
    rows = [{"Jendela": r["window"], "Latih": f"{wib([r['train_from']])[0]:%d %b %Y} – {wib([r['train_to']])[0]:%d %b %Y}",
             "Uji": f"{wib([r['test_from']])[0]:%d %b %Y} – {wib([r['test_to']])[0]:%d %b %Y}",
             "Strategi terpilih": r["chosen"] + ("" if r["met_target"] else " (tidak ada yang memenuhi target)"),
             "Net % latih": r["is_net"] * 100, "PF latih": r["is_pf"], "Net % uji": r["oos_net"] * 100, "PF uji": r["oos_pf"],
             "DD % uji": r["oos_dd"] * 100, "Trades uji": r["oos_trades"],
             "Detail": E.describe(r["fam"], r["params"], cfg["direction"]) if r["fam"] else "-"} for r in w["rows"]]
    st.dataframe(pd.DataFrame(rows).style.format({k: "{:.2f}" for k in ["Net % latih", "PF latih", "Net % uji", "PF uji", "DD % uji"]}),
                 hide_index=True, width='stretch')
    st.caption(f"{w['variants']} variasi per jendela · mode {w['mode']} · {w['seconds']:.0f} detik. "
               "Nilai WFE ikut masuk Rapor ketahanan untuk semua strategi dari pencarian ini.")


# ------------------------------------------------------------------ portfolio page
def render_portfolio(res, dfc, cc):
    cfg, meta = res["cfg"], res["meta"]
    st.markdown("Gabungkan beberapa strategi yang lolos dan **tidak bergerak bersamaan** (korelasi rendah). "
                "Modal dibagi rata. Biasanya kurvanya lebih halus daripada satu strategi saja.")
    a, b, c = st.columns(3)
    max_n = a.slider("Maksimal strategi", 2, 8, 4)
    max_corr = b.slider("Korelasi maksimal antar strategi", 0.2, 0.9, 0.5, 0.05)
    loose = c.checkbox("Ikutkan yang lolos unseen tapi gagal tes lain", value=False)
    cands = [(f"#{i + 1} {E.FAMILIES[o['fam']]['name']}", o["fam"], o["params"]) for i, o in enumerate(res["out"])
             if o["passed"] or (loose and o["oos_ok"] and not o["overfit"])]
    st.caption(f"{len(cands)} kandidat (urut dari hasil in-sample, data unseen tidak dipakai untuk memilih).")
    if len(cands) < 2:
        st.info("Butuh minimal 2 strategi yang lolos. Coba centang pilihan di atas, atau longgarkan target lalu cari lagi.")
        return
    if st.button("▶️ Buat portofolio", type="primary"):
        with st.spinner("Menghitung..."):
            res.setdefault("search_tests", {})["port"] = R.portfolio(cands, cc, meta["is_start"], len(dfc), meta["split"] if meta["split"] < len(dfc) else len(dfc) - 1,
                                                                     cfg, max_n, max_corr)
    p = res.get("search_tests", {}).get("port")
    if not p:
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategi terpilih", len(p["keep"]))
    c2.metric("Net % seluruh periode", pct(p["stats_all"]["net"]))
    c3.metric("Max DD gabungan", pct(p["stats_all"]["maxdd"]))
    c4.metric("Sharpe/tahun unseen", f"{p['stats_oos']['sharpe']:.2f}")
    fig = go.Figure()
    for k in p["keep"]:
        e = (1 + p["single"][k]).cumprod()
        fig.add_trace(go.Scatter(x=wib(e.index), y=e.values * 10000, name=k, line=dict(width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=wib(p["equity"].index), y=p["equity"].values * 10000, name="PORTOFOLIO", line=dict(width=3, color="#ffd600")))
    fig.add_vrect(x0=wib([meta["split_time"]])[0], x1=wib(p["equity"].index[-1:])[0], fillcolor="orange", opacity=0.08, line_width=0,
                  annotation_text="unseen")
    fig.update_layout(title="Equity (modal awal 10.000)", height=400, **PLOT)
    st.plotly_chart(fig, width='stretch')
    fig2 = go.Figure(go.Scatter(x=wib(p["dd"].index), y=-p["dd"].values * 100, fill="tozeroy", line=dict(color="#ef5350")))
    fig2.update_layout(title="Drawdown portofolio (%)", height=220, **PLOT)
    st.plotly_chart(fig2, width='stretch')
    if len(p["keep"]) > 1:
        cm = p["corr"]
        fig3 = go.Figure(go.Heatmap(z=cm.values, x=cm.columns, y=cm.index, zmin=-1, zmax=1, colorscale="RdBu_r",
                                    text=np.round(cm.values, 2), texttemplate="%{text}"))
        fig3.update_layout(title="Korelasi return harian", height=380, **PLOT)
        st.plotly_chart(fig3, width='stretch')
    rows = [{"Bagian": "Portofolio · in-sample", **_fmt_stats(p["stats_is"])}, {"Bagian": "Portofolio · unseen", **_fmt_stats(p["stats_oos"])}]
    rows += [{"Bagian": k, **_fmt_stats(v)} for k, v in p["ind_stats"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("Return harian strategi dirata-rata (modal dibagi rata, diseimbangkan tiap hari). Biaya rebalancing tidak dihitung.")


def _fmt_stats(s):
    return {"Net %": f"{s['net'] * 100:.2f}", "Max DD %": f"{s['maxdd'] * 100:.2f}", "Sharpe/tahun": f"{s['sharpe']:.2f}"}


# ------------------------------------------------------------------ save & compare
def strategy_json(o, res, t, extra=None):
    cfg = res["cfg"]
    d = dict(app="backtest-lab", version=2, saved_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
             name=f"{E.FAMILIES[o['fam']]['name']}", family=o["fam"], params=o["params"], direction=cfg["direction"],
             symbol=res["symbol"], tf=res["tf"], source=res["source"], start=str(res.get("start", "")),
             settings=dict(oos_pct=cfg["oos_pct"], cost_frac=cfg["cost_frac"], cost_abs=cfg["cost_abs"], lev=cfg["lev"],
                           carry=cfg.get("carry")),
             in_sample=o["is_m"], oos=o["oos_m"], passed=bool(o["passed"]),
             grade=list(t.get("grade", (None, None))), dsr=t.get("dsr", {}).get("dsr"))
    if extra:
        d.update(extra)
    return json.dumps(d, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))


def render_compare(load_any, cost_for_src, make_cache_fn=None):
    st.markdown("Simpan strategi sebagai file JSON (tombol ada di tab **Export** tiap strategi), lalu upload lagi kapan saja di sini. "
                "Penyimpanan di server gratis ini hilang saat aplikasi tidur, jadi file JSON di HP/laptopmu adalah cadangannya.")
    ups = st.file_uploader("Upload satu atau beberapa file strategi (.json)", type=["json"], accept_multiple_files=True)
    lst = ss.setdefault("compare", {})
    for u in ups or []:
        try:
            d = json.loads(u.getvalue().decode("utf-8"))
            if "family" not in d or "params" not in d:
                raise ValueError("bukan file strategi Backtest Lab")
            if d["family"] not in E.FAMILIES:
                raise ValueError(f"jenis strategi '{d['family']}' tidak dikenal")
            p = dict(d["params"])
            p.setdefault("regime", 0); p.setdefault("adx_thr", 20)
            d["params"] = p
            lst[f"{u.name}"] = d
        except Exception as e:
            st.error(f"{u.name}: {e}")
    if not lst:
        st.info("Belum ada strategi untuk dibandingkan. Tambahkan dari tab Export ('➕ Tambah ke perbandingan') atau upload file JSON.")
        return
    rm = st.multiselect("Hapus dari daftar", list(lst))
    if rm and st.button("Hapus yang dipilih"):
        for k in rm:
            lst.pop(k, None)
        st.rerun()
    if st.button("▶️ Uji ulang semua di data terbaru", type="primary"):
        out = {}
        bar = st.progress(0.0)
        for k, (name, d) in enumerate(lst.items()):
            bar.progress(k / len(lst), text=f"Menguji {name}...")
            try:
                df = load_any(d["source"], d["symbol"], d["tf"], d.get("start") or None, E.uses_deriv(d["family"], d["params"]))
                sett = d.get("settings", {})
                cc = make_cache_fn(df, {"carry": sett.get("carry")}, d["symbol"], d["tf"])[0] if make_cache_fn else E.Cache(df)
                n = len(df)
                st_i = min(260, n // 4)
                oos = sett.get("oos_pct", 30)
                if "cost_frac" in sett:
                    cf, ca = sett["cost_frac"], sett["cost_abs"]
                else:
                    cf, ca = cost_for_src(d["source"], d["symbol"], float(df.close.mean()))
                lev = sett.get("lev", 1.0)
                split = int(n * (1 - oos / 100))
                fm, ftr = E.run(d["family"], d["params"], cc, st_i, n, d["direction"], cf, ca, lev=lev, want_equity=True)
                om, _ = E.run(d["family"], d["params"], cc, split, n, d["direction"], cf, ca, lev=lev)
                out[name] = dict(full=fm, oos=om, eq=pd.Series(ftr["equity"], index=df.index[st_i:n]), d=d)
            except Exception as e:
                out[name] = dict(error=str(e), d=d)
        bar.empty()
        ss.compare_res = out
    cr = ss.get("compare_res", {})
    rows = []
    for name, d in lst.items():
        r = cr.get(name, {})
        g = d.get("grade") or [None, None]
        row = {"File": name, "Strategi": d.get("name", d["family"]), "Aset": f"{d['symbol']} {d['tf']}", "Arah": d["direction"],
               "Disimpan": str(d.get("saved_at", "-"))[:10], "Rapor saat disimpan": f"{g[1]} ({g[0]:.0f})" if g and g[0] is not None else "-",
               "PF unseen saat disimpan": d.get("oos", {}).get("pf")}
        if "full" in r:
            row.update({"Net % terbaru (semua)": r["full"]["net"] * 100, "PF terbaru": r["full"]["pf"], "DD % terbaru": r["full"]["maxdd"] * 100,
                        "Trades": r["full"]["trades"], "Net % data paling baru": r["oos"]["net"] * 100, "PF data paling baru": r["oos"]["pf"]})
        elif "error" in r:
            row["Catatan"] = r["error"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    if cr:
        fig = go.Figure()
        for name, r in cr.items():
            if "eq" in r:
                fig.add_trace(go.Scatter(x=wib(r["eq"].index), y=r["eq"].values * 10000, name=name))
        fig.update_layout(title="Equity tiap strategi di data terbaru (modal awal 10.000)", height=400, **PLOT)
        st.plotly_chart(fig, width='stretch')
        st.caption("'Data paling baru' = porsi unseen yang sama dengan saat disimpan, dihitung dari data hari ini.")
