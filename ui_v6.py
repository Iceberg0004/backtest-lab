"""v6 tab: multi-leg strategies (grid, DCA, martingale, pyramiding, hedging, zone recovery) plus pairs trading,
rebalancing and DCA benchmark. Plain, short Indonesian text."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core import engine as E, position as PS, posrobust as PR, pairs as PA

ss = st.session_state
CAP = 10_000
ALL_MODES = dict(PS.MODES, pairs="Pairs trading / stat-arb (2 aset)", rebal="Rebalancing & DCA patokan")
DEFAULTS = {
    "grid": dict(lo_pct=15, hi_pct=15, grids=20, geo=1, gdir="neutral", alloc=1.0, sl_out=0, recenter=1, btp=0, bsl=0),
    "dca": dict(ddir="long", start="rsi", base=10, so_size=10, n_so=5, dev=1.5, sscale=1.3, vscale=1.5, tp=1.5, sl=0, cool=0, btp=0, bsl=0),
    "mart": dict(fam="rsi_mr", fparams={}, base=5, mult=2.0, max_steps=5, anti=0, compound=0, mdir="both"),
    "pyr": dict(fam="donchian", fparams=dict(entry_len=55, exit_len=20, trend=0), pdir="both", risk=1.0, stop_k=2.0, add_k=0.5, max_u=4),
    "hedge": dict(size=20, tp=1.0, btp=2, bsl=10),
    "zone": dict(start="trend", base=10, zone=1.0, tp=2.0, max_legs=5, cool=0),
}
LBL = dict(net="Net P&L %", cagr="Per tahun %", maxdd="Max DD % (dalam candle)", trades="Transaksi / siklus", winrate="Win rate %",
           pf="Profit factor", worst="Rugi terburuk 1 siklus %", max_expo="Eksposur maks (x ekuitas)", max_legs="Posisi terbuka maks",
           under_days="Underwater terlama (hari)", under_share="Waktu underwater %", stop_outs="Margin call")


def _f(x, pct=False, d=1):
    if x is None or (isinstance(x, float) and (math.isnan(x))):
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return f"{x * 100:.{d}f}" if pct else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def mtable(cols: dict):
    rows = []
    for k, lab in LBL.items():
        r = {"": lab}
        for name, m in cols.items():
            v = m.get(k)
            if k in ("net", "cagr", "maxdd", "winrate", "under_share"):
                r[name] = _f(v, True)
            elif k == "worst":
                r[name] = _f(v * 100 if v is not None else None, False, 2)
            elif k == "stop_outs":
                r[name] = ("YA · " + str(m["blown_at"].tz_convert("Asia/Jakarta").strftime("%d %b %Y"))) if m.get("blown") else "Tidak"
            elif k in ("pf", "max_expo"):
                r[name] = _f(v, False, 2)
            elif k == "under_days":
                r[name] = _f(v, False, 0)
            else:
                r[name] = str(v)
        rows.append(r)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def banner(mode):
    if mode in PS.TAIL_RISK:
        st.error("⚠️ **Risiko ekor.** Grid, martingale, DCA averaging, hedging dan zone recovery sering terlihat untung terus (win rate tinggi) "
                 "sampai satu pergerakan besar menghabiskan akun. Nilai rapor di sini dipotong keras kalau pernah kena margin call "
                 "atau peluang margin call di Monte Carlo tidak kecil. Lihat kolom risiko, bukan profitnya saja.")


# ------------------------------------------------------------------ parameter widgets
def widgets(mode, ctx):
    k = f"mp_{mode}_"
    p0 = dict(DEFAULTS[mode]); p0.update(ss.get(f"mp_load_{mode}") or {})
    p = {}
    c = st.columns(4)
    if mode == "grid":
        p["gdir"] = c[0].selectbox("Arah grid", ["neutral", "long", "short"], index=["neutral", "long", "short"].index(p0["gdir"]), key=k + "gdir",
                                   format_func={"neutral": "Netral (beli di bawah, jual di atas)", "long": "Long", "short": "Short"}.get)
        p["lo_pct"] = c[1].number_input("Batas bawah (% di bawah harga awal)", 1.0, 90.0, float(p0["lo_pct"]), 1.0, key=k + "lo")
        p["hi_pct"] = c[2].number_input("Batas atas (% di atas harga awal)", 1.0, 300.0, float(p0["hi_pct"]), 1.0, key=k + "hi")
        p["grids"] = c[3].number_input("Jumlah grid", 2, 200, int(p0["grids"]), 1, key=k + "n")
        c = st.columns(4)
        p["geo"] = int(c[0].selectbox("Jarak grid", [1, 0], index=[1, 0].index(int(p0["geo"])), key=k + "geo",
                                      format_func={1: "Geometris (% sama)", 0: "Aritmetika (jarak harga sama)"}.get))
        p["alloc"] = c[1].number_input("Total modal di grid (x modal)", 0.1, 20.0, float(p0["alloc"]), 0.1, key=k + "alloc",
                                       help="1 = seluruh modal dibagi rata ke semua grid. Di atas 1 berarti memakai leverage.")
        p["sl_out"] = c[2].number_input("Tutup semua kalau keluar range lebih dari (%) · 0 = tidak", 0.0, 50.0, float(p0["sl_out"]), 1.0, key=k + "slo")
        p["recenter"] = int(c[3].checkbox("Pasang grid baru di harga sekarang setelah ditutup", bool(p0["recenter"]), key=k + "rc"))
    elif mode == "dca":
        p["ddir"] = c[0].selectbox("Arah", ["long", "short"], index=["long", "short"].index(p0["ddir"]), key=k + "dir")
        p["start"] = c[1].selectbox("Mulai deal", ["always", "rsi", "trend"], index=["always", "rsi", "trend"].index(p0["start"]), key=k + "st",
                                    format_func={"always": "Langsung setelah deal selesai", "rsi": "RSI(14) keluar dari jenuh jual/beli (30/70)",
                                                 "trend": "Harga di atas/bawah EMA 200"}.get)
        p["base"] = c[2].number_input("Base order (% modal)", 0.5, 500.0, float(p0["base"]), 0.5, key=k + "bo")
        p["so_size"] = c[3].number_input("Safety order pertama (% modal)", 0.5, 500.0, float(p0["so_size"]), 0.5, key=k + "so")
        c = st.columns(5)
        p["n_so"] = c[0].number_input("Jumlah safety order", 0, 30, int(p0["n_so"]), 1, key=k + "nso")
        p["dev"] = c[1].number_input("Jarak SO pertama (%)", 0.1, 30.0, float(p0["dev"]), 0.1, key=k + "dev")
        p["sscale"] = c[2].number_input("Step scale (jarak x)", 0.5, 3.0, float(p0["sscale"]), 0.05, key=k + "ss")
        p["vscale"] = c[3].number_input("Volume scale (ukuran x)", 0.5, 3.0, float(p0["vscale"]), 0.05, key=k + "vs",
                                        help="1 = averaging down biasa (ukuran sama). Di atas 1 = makin besar, mirip martingale.")
        p["tp"] = c[4].number_input("Take profit dari harga rata-rata (%)", 0.1, 50.0, float(p0["tp"]), 0.1, key=k + "tp")
        c = st.columns(4)
        p["sl"] = c[0].number_input("Stop loss dari harga rata-rata (%) · 0 = tidak ada", 0.0, 90.0, float(p0["sl"]), 0.5, key=k + "sl")
        p["cool"] = c[1].number_input("Jeda antar deal (candle)", 0, 500, int(p0.get("cool", 0)), 1, key=k + "cool")
        lv, cum, step, tot = [], 0.0, p["dev"], p["base"]
        for i in range(int(p["n_so"])):
            cum += step; step *= p["sscale"]; lv.append(cum); tot += p["so_size"] * p["vscale"] ** i
        c[2].metric("Turun maks yang ditampung", f"{lv[-1]:.1f}%" if lv else "-")
        c[3].metric("Modal maks 1 deal", f"{tot:.0f}% modal")
    elif mode in ("mart", "pyr"):
        fams = [f for f in E.FAMILIES if not E.FAMILIES[f]["needs"] - set(getattr(ctx["cc"], "avail", set())) and f != "bh"]
        src = ["manual"] + (["res"] if ctx.get("res_strategy") else [])
        use = c[0].selectbox("Sinyal masuk dari", src, key=k + "src",
                             format_func={"manual": "Pilih jenis strategi", "res": "Strategi terpilih di tab Hasil"}.get)
        if use == "res":
            f_, fp_ = ctx["res_strategy"]
            p["fam"], p["fparams"] = f_, dict(fp_)
            c[1].caption(E.describe(f_, fp_, "both")[:160])
        else:
            p["fam"] = c[1].selectbox("Jenis strategi", fams, index=fams.index(p0["fam"]) if p0["fam"] in fams else 0, key=k + "fam",
                                      format_func=lambda f: E.FAMILIES[f]["name"])
            p["fparams"] = p0["fparams"] if p["fam"] == p0["fam"] else {}
        if mode == "mart":
            p["anti"] = int(c[2].selectbox("Tipe", [0, 1], index=int(p0["anti"]), key=k + "anti",
                                           format_func={0: "Martingale (perbesar setelah rugi)", 1: "Anti-martingale (perbesar setelah untung)"}.get))
            p["mdir"] = c[3].selectbox("Arah", ["both", "long", "short"], index=["both", "long", "short"].index(p0.get("mdir", "both")), key=k + "md")
            c = st.columns(4)
            p["base"] = c[0].number_input("Ukuran awal (% modal)", 0.5, 500.0, float(p0["base"]), 0.5, key=k + "base")
            p["mult"] = c[1].number_input("Pengali ukuran", 1.0, 5.0, float(p0["mult"]), 0.1, key=k + "mult")
            p["max_steps"] = c[2].number_input("Langkah maks (lalu reset)", 1, 15, int(p0["max_steps"]), 1, key=k + "steps")
            p["compound"] = int(c[3].checkbox("Ukuran ikut ekuitas (compounding)", bool(p0.get("compound", 0)), key=k + "cmp"))
            st.caption(f"Ukuran terbesar setelah {p['max_steps']} kali rugi beruntun: {p['base'] * p['mult'] ** p['max_steps']:.0f}% modal.")
        else:
            p["pdir"] = c[2].selectbox("Arah", ["both", "long", "short"], index=["both", "long", "short"].index(p0["pdir"]), key=k + "pd")
            c = st.columns(4)
            p["risk"] = c[0].number_input("Risiko per unit (% ekuitas)", 0.1, 10.0, float(p0["risk"]), 0.1, key=k + "risk")
            p["stop_k"] = c[1].number_input("Stop (x ATR dari unit terakhir)", 0.5, 10.0, float(p0["stop_k"]), 0.5, key=k + "stopk")
            p["add_k"] = c[2].number_input("Tambah unit tiap (x ATR) searah", 0.25, 5.0, float(p0["add_k"]), 0.25, key=k + "addk")
            p["max_u"] = c[3].number_input("Unit maks (1 = tanpa piramida)", 1, 10, int(p0["max_u"]), 1, key=k + "maxu")
    elif mode == "hedge":
        p["size"] = c[0].number_input("Ukuran tiap sisi (% modal)", 1.0, 500.0, float(p0["size"]), 1.0, key=k + "size")
        p["tp"] = c[1].number_input("Take profit tiap posisi (%)", 0.1, 20.0, float(p0["tp"]), 0.1, key=k + "tp")
        st.caption("Buka long dan short bersamaan. Posisi yang kena take profit langsung dibuka lagi di harga itu, jadi selalu ada dua sisi.")
    elif mode == "zone":
        p["start"] = c[0].selectbox("Arah awal", ["trend", "candle"], index=["trend", "candle"].index(p0["start"]), key=k + "st",
                                    format_func={"trend": "Ikut EMA 200", "candle": "Ikut warna candle terakhir"}.get)
        p["base"] = c[1].number_input("Ukuran awal (% modal)", 0.5, 500.0, float(p0["base"]), 0.5, key=k + "base")
        p["zone"] = c[2].number_input("Lebar zona (%)", 0.1, 20.0, float(p0["zone"]), 0.1, key=k + "zone")
        p["tp"] = c[3].number_input("Target (%)", 0.1, 20.0, float(p0["tp"]), 0.1, key=k + "tp")
        c = st.columns(4)
        p["max_legs"] = c[0].number_input("Posisi maks (lalu terima rugi)", 2, 20, int(p0["max_legs"]), 1, key=k + "legs")
        p["cool"] = c[1].number_input("Jeda antar siklus (candle)", 0, 500, int(p0.get("cool", 0)), 1, key=k + "cool")
        st.caption("Kalau harga melawan sejauh zona, buka posisi berlawanan yang lebih besar supaya saat target tercapai semua posisi ditutup untung. "
                   "Ukuran membesar cepat: ini sumber risikonya.")
    if mode in ("grid", "dca", "hedge"):
        c = st.columns(2)
        p["btp"] = c[0].number_input("Tutup semua kalau total untung siklus ≥ (% modal) · 0 = tidak", 0.0, 500.0, float(p0.get("btp", 0)), 0.5, key=k + "btp")
        p["bsl"] = c[1].number_input("Tutup semua kalau total rugi siklus ≥ (% modal) · 0 = tidak", 0.0, 100.0, float(p0.get("bsl", 0)), 0.5, key=k + "bsl")
    return p


# ------------------------------------------------------------------ charts
def eq_chart(cc, r_is, r_oos, title):
    a, b = r_is["start"], r_oos["end"]
    x = pd.DatetimeIndex(cc.index[a:b]).tz_convert("Asia/Jakarta")
    eq = np.r_[r_is["eq"], r_oos["eq"] * r_is["eq"][-1]] * CAP
    eqm = np.r_[r_is["eq_min"], r_oos["eq_min"] * r_is["eq"][-1]] * CAP
    expo = np.r_[r_is["expo"], r_oos["expo"]]
    step = max(1, len(x) // 4000)
    f = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    f.add_trace(go.Scatter(x=x[::step], y=eq[::step], name="Ekuitas (candle tutup)", line=dict(color="#4fc3f7")), 1, 1)
    f.add_trace(go.Scatter(x=x[::step], y=eqm[::step], name="Ekuitas terendah dalam candle", line=dict(color="#ff8a80", width=1)), 1, 1)
    f.add_trace(go.Scatter(x=x[::step], y=expo[::step], name="Eksposur (x ekuitas)", line=dict(color="#ffd54f", width=1), fill="tozeroy"), 2, 1)
    f.add_vline(x=pd.Timestamp(cc.index[r_oos["start"]]).tz_convert("Asia/Jakarta"), line_dash="dash", line_color="#aaa")
    for r in (r_is, r_oos):
        if r["m"]["blown"]:
            f.add_vline(x=r["m"]["blown_at"].tz_convert("Asia/Jakarta"), line_color="red")
    f.update_layout(height=480, template="plotly_dark", margin=dict(l=10, r=10, t=40, b=10), title=title, legend=dict(orientation="h"))
    f.update_yaxes(type="log", row=1, col=1)
    st.plotly_chart(f, width="stretch")
    st.caption("Garis putus-putus = mulai data unseen. Garis merah = margin call. Ekuitas digabung (in-sample lalu unseen), modal awal 10.000.")


def grade_box(g):
    score, letter, why = g
    colors = {"A": "#00c853", "B": "#64dd17", "C": "#ffd600", "D": "#ff9100", "E": "#ff1744"}
    if score is None:
        return
    st.markdown(f"<div style='padding:8px 12px;border-radius:8px;background:#1e1e1e;border-left:6px solid {colors.get(letter, '#888')}'>"
                f"<b style='font-size:1.4em;color:{colors.get(letter, '#888')}'>Rapor: {letter}</b> · {score:.0f}/100 "
                f"<span style='color:#aaa'>(lolos unseen, biaya 2x, setelan tetangga, Monte Carlo)</span></div>", unsafe_allow_html=True)
    for w in why:
        st.caption(f"⛔ {w}")


# ------------------------------------------------------------------ main
def render_multi(ctx):
    cc, df = ctx["cc"], ctx["df"]
    n = len(df)
    split = int(n * (1 - ctx["oos_pct"] / 100))
    is_start = min(260, split // 4)
    st.caption(f"{ctx['symbol']} {ctx['tf']} · {n:,} candle · in-sample sampai {pd.Timestamp(cc.index[split]).tz_convert('Asia/Jakarta'):%d %b %Y}, "
               f"sesudahnya data unseen ({ctx['oos_pct']}%). Biaya dari sidebar. Hasil adalah uji data lama, bukan saran.")
    mode = st.selectbox("Jenis strategi", list(ALL_MODES), format_func=ALL_MODES.get, key="mp_mode")
    if mode == "pairs":
        return render_pairs(ctx, split)
    if mode == "rebal":
        return render_rebal(ctx, split)
    banner(mode)
    c = st.columns(3)
    lev = c[0].number_input("Leverage maks akun", 1.0, 500.0, float(ss.get("mp_lev", 10.0)), 1.0, key="mp_lev",
                            help="Margin = nilai posisi / leverage. Ini batas dari broker/exchange, bukan ukuran posisi.")
    so = c[1].number_input("Stop-out (margin level %)", 0.0, 200.0, float(ss.get("mp_so", 50.0)), 5.0, key="mp_so",
                           help="Kalau ekuitas turun di bawah persen ini dari margin terpakai, semua posisi ditutup paksa (margin call).") / 100
    c[2].caption("Urutan harga dalam 1 candle dianggap O→L→H→C (candle hijau) atau O→H→L→C (merah). Gap mengisi order di harga open.")
    p = widgets(mode, ctx)
    cf, ca = ctx["cf"], ctx["ca"]
    key = (mode, repr(sorted((k, str(v)) for k, v in p.items())), ctx["symbol"], ctx["tf"], n, lev, so, cf, ca)
    store = ss.setdefault("mp_res", {})
    b = st.columns(4)
    if b[0].button("▶️ Jalankan backtest", type="primary", key="mp_run"):
        with st.spinner("Menghitung..."):
            store.clear()
            t, r_is, r_oos = PR.tests(mode, p, cc, is_start, split, n, cf, ca, lev, so, ctx["max_dd"], ctx["min_pf"])
            store[key] = dict(t=t, r_is=r_is, r_oos=r_oos, p=p)
    npaths = b[1].select_slider("Jumlah simulasi", [100, 200, 400], value=200, key="mp_npaths")
    if b[2].button("🎲 Monte Carlo margin call", key="mp_mc"):
        if key not in store:
            t, r_is, r_oos = PR.tests(mode, p, cc, is_start, split, n, cf, ca, lev, so, ctx["max_dd"], ctx["min_pf"])
            store.clear(); store[key] = dict(t=t, r_is=r_is, r_oos=r_oos, p=p)
        bar = st.progress(0.0, "Simulasi...")
        mc = PR.mc_ruin(mode, p, cc, is_start, n, cf, ca, lev, so, n_paths=npaths, years=1.0,
                        progress=lambda f, t_: bar.progress(min(1.0, f), t_))
        bar.empty()
        s = store[key]; s["t"]["mc"] = mc; s["t"]["grade"] = PR.grade(mode, s["t"])
    if b[3].button("🌍 Uji di pasar lain", key="mp_basket"):
        if key not in store:
            t, r_is, r_oos = PR.tests(mode, p, cc, is_start, split, n, cf, ca, lev, so, ctx["max_dd"], ctx["min_pf"])
            store.clear(); store[key] = dict(t=t, r_is=r_is, r_oos=r_oos, p=p)
        store[key]["basket"] = basket(mode, p, ctx, lev, so)
    with st.expander("🔎 Cari setelan terbaik (hanya dari data in-sample)"):
        c = st.columns(3)
        nv = c[0].select_slider("Jumlah variasi", [100, 300, 500, 1000, 2000], value=500, key="mp_nv")
        algo = c[1].radio("Cara mencari", ["random", "evo"], horizontal=True, key="mp_algo",
                          format_func={"random": "Acak", "evo": "Evolusi"}.get)
        fam_fix = p.get("fam") if mode in ("mart", "pyr") and c[2].checkbox("Kunci jenis strategi di atas", value=False, key="mp_famfix") else None
        if st.button("🔎 Mulai cari", key="mp_search"):
            bar = st.progress(0.0, "Mencari...")
            sr = PR.search(mode, cc, is_start, split, n, cf, ca, lev, so, n_var=nv, algo=algo, time_budget=120, direction=ctx["direction"],
                           fam=fam_fix, progress=lambda f, t_: bar.progress(min(1.0, f), t_))
            bar.empty()
            ss["mp_search_res"] = (mode, ctx["symbol"], ctx["tf"], sr)
        got = ss.get("mp_search_res")
        if got and got[0] == mode and got[1] == ctx["symbol"] and got[2] == ctx["tf"]:
            sr = got[3]
            st.caption(f"{sr['tested']:,} variasi dalam {sr['seconds']:.0f} detik · {sr['ok_is']} untung di in-sample tanpa margin call. "
                       "Urutan dari in-sample (untung ÷ drawdown). Kolom unseen hanya ujian.")
            rows = []
            for i, r in enumerate(sr["rows"]):
                im, om = r["is_m"], r["oos_m"]
                rows.append({"#": i + 1, "IS net %": im["net"] * 100, "IS DD %": im["maxdd"] * 100, "IS margin call": "YA" if im["blown"] else "-",
                             "Unseen net %": om["net"] * 100, "Unseen DD %": om["maxdd"] * 100, "Unseen win %": om["winrate"] * 100,
                             "Unseen margin call": "YA" if om["blown"] else "-", "Eksposur maks": om["max_expo"],
                             "Setelan": ", ".join(f"{k}={v}" for k, v in r["p"].items() if k != "fparams")[:120]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                         column_config={k: st.column_config.NumberColumn(format="%.1f") for k in
                                        ["IS net %", "IS DD %", "Unseen net %", "Unseen DD %", "Unseen win %", "Eksposur maks"]})
            pick = st.number_input("Pakai setelan nomor", 1, max(1, len(sr["rows"])), 1, key="mp_pick")
            if st.button("Pakai setelan ini", key="mp_apply"):
                ss[f"mp_load_{mode}"] = sr["rows"][pick - 1]["p"]
                for k_ in [k_ for k_ in ss.keys() if str(k_).startswith(f"mp_{mode}_")]:
                    del ss[k_]
                st.rerun()
    s = store.get(key)
    if not s:
        st.caption("Atur setelan lalu klik 'Jalankan backtest'. Monte Carlo menjalankan setelan yang sama di 100-400 jalur harga acak "
                   "(potongan riwayat asli) selama 1 tahun dan menghitung berapa sering akun kena margin call.")
        return
    t = s["t"]
    grade_box(t["grade"])
    mtable({"In-sample": t["is_m"], "Unseen (ujian)": t["oos_m"], "Unseen, biaya 2x": t["cost2x"]})
    if t.get("jitter") is not None:
        st.caption(f"Setelan tetangga (diubah sedikit) yang tetap untung tanpa margin call di unseen: {t['jitter'] * 100:.0f}%.")
    if mode == "mart" and s["r_is"].get("capped"):
        st.caption(f"{s['r_is']['capped']} trade dibatasi karena margin tidak cukup untuk ukuran martingale penuh.")
    mc = t.get("mc")
    if mc:
        c = st.columns(5)
        c[0].metric("Peluang margin call (1 thn)", f"{mc['p_ruin'] * 100:.1f}%")
        c[1].metric("Peluang turun > 50%", f"{mc['p_dd50'] * 100:.0f}%")
        c[2].metric("Drawdown median / 95% / 99%", f"{mc['dd50'] * 100:.0f} / {mc['dd95'] * 100:.0f} / {mc['dd99'] * 100:.0f}%")
        c[3].metric("Hasil 1 thn: 5% terburuk / median", f"{mc['net5'] * 100:.0f}% / {mc['net50'] * 100:.0f}%")
        c[4].metric("Peluang rugi setahun", f"{mc['p_loss'] * 100:.0f}%")
        st.caption(f"{mc['n']} simulasi × {mc['years']:g} tahun dari potongan acak riwayat asli (blok 5 hari"
                   + (", martingale: urutan trade diacak per blok 5 trade" if mode == "mart" else "") + ").")
    eq_chart(cc, s["r_is"], s["r_oos"], f"{ALL_MODES[mode]} · {ctx['symbol']} {ctx['tf']}")
    if s.get("basket"):
        st.markdown("**Setelan yang sama di pasar lain (unseen)**")
        st.dataframe(pd.DataFrame(s["basket"]), hide_index=True, width="stretch")
    pine_block(mode, s["p"], ctx)


def basket(mode, p, ctx, lev, so):
    rows = []
    for sym in ctx["basket"]:
        try:
            d = ctx["load_other"](sym)
            c2, _ = ctx["make_cache"](d, dict(carry=ctx.get("carry")), sym, ctx["tf"])
            n = len(d); split = int(n * (1 - ctx["oos_pct"] / 100))
            cf, ca = ctx["cost_for"](ctx["source"], sym, float(d.close.iloc[-1]))
            m = PS.run(mode, p, c2, split, n, cf, ca, lev, so)["m"]
            rows.append({"Pasar": sym, "Net %": round(m["net"] * 100, 1), "Max DD %": round(m["maxdd"] * 100, 1), "Win %": round(m["winrate"] * 100),
                         "Transaksi": m["trades"], "Eksposur maks": round(m["max_expo"], 1), "Margin call": "YA" if m["blown"] else "-"})
        except Exception as e:
            rows.append({"Pasar": sym, "Net %": None, "Margin call": f"gagal: {str(e)[:60]}"})
    return rows


def pine_block(mode, p, ctx):
    from core import pine_multi
    code = pine_multi.to_pine(mode, p, ctx["symbol"], ctx["tf"], float(ctx["df"].close.iloc[-1]), ctx["pine_cost"])
    with st.expander("🌲 Export ke TradingView (Pine)"):
        if code is None:
            st.info(pine_multi.WHY_NOT.get(mode, "Mode ini belum bisa diekspor ke Pine."))
            return
        st.caption(pine_multi.NOTE.get(mode, ""))
        st.download_button("⬇️ Download .pine", code, file_name=f"{mode}_{ctx['symbol']}_{ctx['tf']}.pine", key="mp_pine_dl")
        st.code(code, language="javascript")


# ------------------------------------------------------------------ pairs & rebalancing
def _other(ctx, label, key):
    syms = [s for s in ctx["symbols"] if s != ctx["symbol"]]
    default = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT", "EURUSD": "GBPUSD", "GBPUSD": "EURUSD", "XAUUSD": "XAGUSD"}.get(ctx["symbol"], syms[0] if syms else None)
    return st.selectbox(label, syms, index=syms.index(default) if default in syms else 0, key=key)


def _frac_cost(ctx, sym, price):
    cf, ca = ctx["cost_for"](ctx["source"], sym, price)
    return cf + ca / price


def render_pairs(ctx, split_unused):
    st.caption("Pairs trading: cari dua pasar yang harganya 'terikat' (kointegrasi). Saat selisihnya (spread) melebar jauh dari rata-rata, "
               "beli yang murah dan jual yang mahal, tutup saat kembali normal. Rasio lindung nilai dan tes kointegrasi hanya dari in-sample.")
    other = _other(ctx, "Pasar kedua", "mp_pair_b")
    if not st.button("▶️ Uji pairs", type="primary", key="mp_pair_run"):
        got = ss.get("mp_pair_res")
        if not got or got[0] != (ctx["symbol"], other, ctx["tf"]):
            return
        res = got[1]
    else:
        with st.spinner("Mengambil data dan menghitung..."):
            b = ctx["load_other"](other)
            j = PA.align(ctx["df"], b)
            n = len(j); split = int(n * (1 - ctx["oos_pct"] / 100))
            ca_, cb_ = _frac_cost(ctx, ctx["symbol"], float(j.a.iloc[-1])), _frac_cost(ctx, other, float(j.b.iloc[-1]))
            fit, rows = PA.pairs_search(j, split, ca_, cb_, ctx["cc"].bars_per_year)
            sc, pp, im = rows[0]
            full = PA.pairs_run(j, fit, pp["n"], pp["z_in"], pp["z_out"], pp["z_stop"], pp["n"], n, ca_, cb_, ctx["cc"].bars_per_year)
            oos = PA.pairs_run(j, fit, pp["n"], pp["z_in"], pp["z_out"], pp["z_stop"], split, n, ca_, cb_, ctx["cc"].bars_per_year)
            c2 = PA.pairs_run(j, fit, pp["n"], pp["z_in"], pp["z_out"], pp["z_stop"], split, n, ca_ * 2, cb_ * 2, ctx["cc"].bars_per_year)
            res = dict(fit={k: v for k, v in fit.items() if k != "spread"}, pp=pp, im=im, om=oos["m"], c2=c2["m"],
                       idx=j.index[pp["n"]:], eq=full["eq"], z=full["z"], split_t=j.index[split], n=n)
            ss["mp_pair_res"] = ((ctx["symbol"], other, ctx["tf"]), res)
    f = res["fit"]
    coint = f["adf"] < PA.ADF_5PCT
    (st.success if coint else st.warning)(
        f"{'Terkointegrasi' if coint else 'TIDAK terkointegrasi'} di in-sample (ADF t = {f['adf']:.2f}, batas 5% = {PA.ADF_5PCT}). "
        f"Rasio lindung nilai β = {f['beta']:.3f}, half-life ±{f['hl']:.0f} candle." + ("" if coint else " Hasil pairs di bawah kemungkinan kebetulan."))
    pp = res["pp"]
    st.caption(f"Setelan terbaik in-sample: z-score {pp['n']} candle, masuk |z| > {pp['z_in']}, keluar |z| < {pp['z_out']}, stop |z| > {pp['z_stop']}.")
    rows = [{"": lab, "In-sample": _f(res["im"][k], k in ("net", "maxdd", "winrate"), 2), "Unseen": _f(res["om"][k], k in ("net", "maxdd", "winrate"), 2),
             "Unseen biaya 2x": _f(res["c2"][k], k in ("net", "maxdd", "winrate"), 2)}
            for k, lab in (("net", "Net %"), ("maxdd", "Max DD %"), ("trades", "Trade"), ("winrate", "Win rate %"), ("pf", "Profit factor"))]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    x = pd.DatetimeIndex(res["idx"]).tz_convert("Asia/Jakarta")
    step = max(1, len(x) // 4000)
    fg = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.04)
    fg.add_trace(go.Scatter(x=x[::step], y=res["eq"][::step] * CAP, name="Ekuitas"), 1, 1)
    fg.add_trace(go.Scatter(x=x[::step], y=res["z"][::step], name="Z-score spread", line=dict(width=1)), 2, 1)
    fg.add_vline(x=pd.Timestamp(res["split_t"]).tz_convert("Asia/Jakarta"), line_dash="dash", line_color="#aaa")
    fg.update_layout(height=420, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fg, width="stretch")


def render_rebal(ctx, split_unused):
    st.caption("Rebalancing: dua pasar dengan bobot tetap, dikembalikan ke bobot awal secara berkala. Dibandingkan dengan beli & tahan. "
               "Di bawahnya: DCA (beli rutin) vs beli sekaligus di pasar sekarang.")
    other = _other(ctx, "Pasar kedua", "mp_reb_b")
    c = st.columns(3)
    w = c[0].slider(f"Bobot {ctx['symbol']} (%)", 5, 95, 50, 5, key="mp_reb_w") / 100
    every = c[1].selectbox("Rebalance", ["d", "w", "m", "q", "0"], index=2, key="mp_reb_e",
                           format_func={"d": "Harian", "w": "Mingguan", "m": "Bulanan", "q": "3 bulanan", "0": "Hanya kalau melenceng"}.get)
    band = c[2].number_input("Atau kalau bobot melenceng lebih dari (%) · 0 = tidak", 0.0, 50.0, 0.0, 1.0, key="mp_reb_band") / 100
    dca_days = st.select_slider("DCA: beli setiap (hari)", [1, 7, 14, 30], value=7, key="mp_dca_days")
    if st.button("▶️ Hitung", type="primary", key="mp_reb_run"):
        with st.spinner("Menghitung..."):
            b = ctx["load_other"](other)
            j = PA.align(ctx["df"], b)
            bpd = ctx["cc"].bars_per_year / 365.25
            eb = {"d": 1, "w": 7, "m": 30, "q": 91, "0": 0}[every]
            ca_, cb_ = _frac_cost(ctx, ctx["symbol"], float(j.a.iloc[-1])), _frac_cost(ctx, other, float(j.b.iloc[-1]))
            r = PA.rebalance(j, w, int(eb * bpd), band, ca_, cb_, 0, len(j))
            d = PA.dca_benchmark(ctx["df"]["close"], dca_days, _frac_cost(ctx, ctx["symbol"], float(ctx["df"].close.iloc[-1])))
            ss["mp_reb_res"] = (other, j.index, r, d)
    got = ss.get("mp_reb_res")
    if not got or got[0] != other:
        return
    _, idx, r, d = got
    c = st.columns(4)
    c[0].metric("Rebalance: hasil", f"{r['m']['net'] * 100:+.0f}%", help=f"{r['n_reb']} kali rebalance, biaya {r['cost'] * 100:.1f}% modal")
    c[1].metric("Rebalance: max DD", f"{r['m']['maxdd'] * 100:.0f}%")
    c[2].metric("Beli & tahan: hasil", f"{r['m_bh']['net'] * 100:+.0f}%")
    c[3].metric("Beli & tahan: max DD", f"{r['m_bh']['maxdd'] * 100:.0f}%")
    x = pd.DatetimeIndex(idx).tz_convert("Asia/Jakarta"); step = max(1, len(x) // 4000)
    fg = go.Figure([go.Scatter(x=x[::step], y=r["eq"][::step] * CAP, name="Rebalance"), go.Scatter(x=x[::step], y=r["bh"][::step] * CAP, name="Beli & tahan")])
    fg.update_layout(height=320, template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10), yaxis_type="log", legend=dict(orientation="h"))
    st.plotly_chart(fg, width="stretch")
    c = st.columns(4)
    c[0].metric(f"DCA tiap {dca_days} hari: nilai akhir", f"{d['dca_final']:.2f}x", help=f"{d['n_buys']} kali beli, total modal 1x")
    c[1].metric("Beli sekaligus: nilai akhir", f"{d['lump_final']:.2f}x")
    c[2].metric("DCA: turun maks dari puncak", f"{d['dd_dca'] * 100:.0f}%")
    c[3].metric("Beli sekaligus: turun maks", f"{d['dd_lump'] * 100:.0f}%")
    st.caption("DCA membagi modal sama rata ke semua pembelian di periode ini. Beli sekaligus biasanya menang kalau harga naik dalam jangka panjang; "
               "DCA mengurangi risiko salah waktu masuk.")
