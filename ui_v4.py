"""v4 pages: analytics, sizing & prop firm, regime, meta-label, ML mode, live signals, HTML report.
Heavy parts only run after a button press (the free cloud has ~1 GB RAM)."""
from __future__ import annotations
import math, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from core import engine as E, robust as R, analytics as A, risk as K


def _dsr_pick(d):
    d = d or {}
    return d.get("dsr_strict") if st.session_state.get("strict_dsr") else d.get("dsr")

ss = st.session_state
TZ = "Asia/Jakarta"
GREEN, RED, BLUE = "#26a69a", "#ef5350", "#29b6f6"
REGIME_COLORS = ["#26a69a", "#ffd600", "#ef5350", "#ab47bc"]


def _trim():
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def wib(idx):
    return pd.DatetimeIndex(idx).tz_convert(TZ)


def _lay(fig, title, h=320):
    fig.update_layout(title=title, template="plotly_dark", height=h, margin=dict(t=40, b=10))
    return fig


def _pct(x, d=1):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:.{d}f}%"


def _mtable(d: dict):
    rows = {}
    for k, m in d.items():
        rows[k] = {"Trades": m.get("trades", 0), "Net %": m.get("net", 0) * 100, "Profit factor": m.get("pf", np.nan),
                   "Win rate %": m.get("winrate", np.nan) * 100, "Max DD %": m.get("maxdd", 0) * 100, "Return/DD": m.get("ret_dd", np.nan)}
    return pd.DataFrame(rows).T


def _show(df, fmt="{:.2f}"):
    st.dataframe(df.style.format(fmt, na_rep="-"), width='stretch')


# ================================================================== analytics
def analytics_figs(ftr, cc, start):
    eq = ftr["equity"]
    figs = {}
    mt = A.monthly_table(eq, cc, start)
    f = go.Figure(go.Heatmap(z=mt.values, x=list(mt.columns), y=[str(y) for y in mt.index], colorscale="RdYlGn", zmid=0,
                             text=np.vectorize(lambda v: "" if v != v else f"{v:.1f}")(mt.values), texttemplate="%{text}",
                             hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>", showscale=False))
    figs["monthly"] = _lay(f, "Return bulanan (%) · kolom terakhir = setahun", max(260, 26 * len(mt) + 80))
    bench = A.benchmark(eq, cc, start)
    step = max(1, len(eq) // 3000)
    x = wib(cc.index[start:start + len(eq)][::step])
    f = go.Figure([go.Scatter(x=x, y=eq[::step] * 100 - 100, name="Strategi", line=dict(color=BLUE)),
                   go.Scatter(x=x, y=bench["bh"][::step] * 100 - 100, name="Beli & tahan", line=dict(color="#9e9e9e"))])
    figs["bench"] = _lay(f, "Strategi vs beli & tahan (% sejak awal)", 330)
    uw = A.underwater(eq, cc, start)
    figs["underwater"] = _lay(go.Figure(go.Scatter(x=wib(uw.index), y=uw.values, fill="tozeroy", line=dict(color=RED))),
                              "Underwater: jarak dari puncak equity (%)", 260)
    return figs, mt, bench


def render_analytics(ctx, ftr):
    cc, start = ctx["cc"], ctx["is_start"]
    if not st.toggle("Tampilkan analitik", key="ana_on", help="Dimatikan dulu supaya aplikasi tetap ringan."):
        st.caption("Isi: return bulanan, jam & hari terbaik, MAE/MFE, kalah beruntun, lama trade, Sharpe & PF bergulir, "
                   "underwater, dan perbandingan dengan beli & tahan.")
        return
    if len(ftr["ret"]) < 5:
        st.info("Trade terlalu sedikit untuk analitik.")
        return
    figs, mt, bench = analytics_figs(ftr, cc, start)
    st.caption("Seluruh periode (in-sample + unseen), posisi sesuai pengaturan pencarian. Waktu dalam WIB.")
    st.plotly_chart(figs["monthly"], width='stretch')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Strategi (total)", _pct(bench["strat_net"])); c2.metric("Beli & tahan (total)", _pct(bench["bh_net"]))
    c3.metric("Korelasi harian dg pasar", f"{bench['corr']:.2f}" if bench["corr"] == bench["corr"] else "-")
    c4.metric("Beta", f"{bench['beta']:.2f}" if bench["beta"] == bench["beta"] else "-")
    st.plotly_chart(figs["bench"], width='stretch')
    st.plotly_chart(figs["underwater"], width='stretch')
    h, d = A.by_time(ftr, cc)
    a, b = st.columns(2)
    a.plotly_chart(_lay(go.Figure(go.Bar(x=h.index, y=h["Total %"], marker_color=np.where(h["Total %"] >= 0, GREEN, RED),
                                         customdata=np.c_[h["Trades"], h["Win rate %"]],
                                         hovertemplate="Jam %{x}: %{y:.1f}% · %{customdata[0]:.0f} trade · WR %{customdata[1]:.0f}%<extra></extra>")),
                        "Hasil per jam masuk (WIB, total %)", 300), width='stretch')
    b.plotly_chart(_lay(go.Figure(go.Bar(x=d.index, y=d["Total %"], marker_color=np.where(d["Total %"] >= 0, GREEN, RED),
                                         customdata=np.c_[d["Trades"], d["Win rate %"]],
                                         hovertemplate="%{x}: %{y:.1f}% · %{customdata[0]:.0f} trade · WR %{customdata[1]:.0f}%<extra></extra>")),
                        "Hasil per hari masuk (WIB, total %)", 300), width='stretch')
    mm = A.mae_mfe(ftr, cc)
    if len(mm) > 3000:
        mm = mm.sample(3000, random_state=1)
    a, b = st.columns(2)
    for col, key, title in ((a, "MAE %", "MAE: minus terdalam selama trade vs hasil"), (b, "MFE %", "MFE: untung tertinggi selama trade vs hasil")):
        f = go.Figure()
        for lab, colr in (("Menang", GREEN), ("Kalah", RED)):
            s = mm[mm.Menang == lab]
            f.add_trace(go.Scattergl(x=s[key], y=s["Hasil %"], mode="markers", name=lab, marker=dict(size=4, color=colr, opacity=0.6)))
        f.update_xaxes(title=key); f.update_yaxes(title="Hasil %")
        col.plotly_chart(_lay(f, title, 330), width='stretch')
    st.caption("MAE membantu menilai stop loss (trade menang yang sempat minus jauh = stop terlalu ketat bisa merusak). "
               "MFE membantu menilai target (trade kalah yang sempat untung besar = mungkin perlu ambil untung lebih awal).")
    sk = A.streaks(ftr["ret"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Menang beruntun terpanjang", sk["max_win"]); c2.metric("Kalah beruntun terpanjang", sk["max_loss"])
    du = A.durations_hours(ftr, cc)
    c3.metric("Lama trade median", f"{np.median(du):.1f} jam"); c4.metric("Lama trade terpanjang", f"{du.max():.0f} jam")
    a, b = st.columns(2)
    if len(sk["loss_runs"]):
        a.plotly_chart(_lay(go.Figure(go.Bar(x=sk["loss_runs"].index, y=sk["loss_runs"].values, marker_color=RED)),
                            "Berapa kali kalah beruntun N kali", 280), width='stretch')
    b.plotly_chart(_lay(go.Figure(go.Histogram(x=du, nbinsx=40, marker_color=BLUE)), "Sebaran lama trade (jam)", 280), width='stretch')
    rs, rp = A.rolling_sharpe(ftr["equity"], cc, start), A.rolling_pf(ftr, cc)
    a, b = st.columns(2)
    if len(rs):
        a.plotly_chart(_lay(go.Figure(go.Scatter(x=wib(rs.index), y=rs.values, line=dict(color=BLUE))), "Sharpe bergulir 90 hari (per tahun)", 280), width='stretch')
    if len(rp):
        b.plotly_chart(_lay(go.Figure(go.Scatter(x=wib(rp.index), y=rp.clip(upper=5).values, line=dict(color="#ffd600"))),
                            "Profit factor bergulir 50 trade (dipotong di 5)", 280), width='stretch')


# ================================================================== sizing & prop firm
def render_sizing(ctx):
    o, cc, cfg = ctx["o"], ctx["cc"], ctx["cfg"]
    fam, p = o["fam"], o["params"]
    if not st.toggle("Buka simulasi ukuran posisi & prop firm", key="sz_on", help="Dimatikan dulu supaya aplikasi tetap ringan."):
        st.caption("Isi: ukuran tetap, risiko % per trade, target volatilitas, Kelly pecahan, compounding, dan peluang lolos challenge prop firm.")
        return
    st.caption("Sinyal dan exit tetap sama. Yang diubah hanya besar posisi tiap trade. Ukuran 1x = nilai posisi sama dengan modal.")
    c1, c2, c3 = st.columns(3)
    mode = c1.selectbox("Cara menentukan ukuran", list(K.MODES), format_func=K.MODES.get, key="sz_mode")
    if mode == "fixed":
        val = c2.number_input("Ukuran posisi (x modal)", 0.05, 20.0, 1.0, 0.05, key="sz_fixed")
    elif mode == "risk":
        val = c2.number_input("Risiko per trade (% modal, kalau kena stop loss)", 0.05, 10.0, 1.0, 0.05, key="sz_risk")
        if E.full_params(p).get("sl_atr", 0) <= 0:
            st.warning("Strategi ini tidak punya stop loss, jadi mode risiko tetap tidak bisa dipakai.")
            return
    elif mode == "vol":
        val = c2.number_input("Target volatilitas per tahun (%)", 1.0, 100.0, 15.0, 1.0, key="sz_vol")
    else:
        val = 0
        kf = c2.select_slider("Bagian Kelly", [0.1, 0.25, 0.5], value=0.25, key="sz_kf",
                              help="Kelly penuh terlalu agresif. 0,25 = seperempat Kelly. Dihitung dari trade in-sample saja.")
    max_exp = c3.number_input("Batas ukuran maksimal (x modal)", 0.1, 20.0, 3.0, 0.1, key="sz_max",
                              help="Satu strategi = maksimal satu posisi sekaligus, jadi ini juga batas eksposur total.")
    c4, c5 = st.columns(2)
    comp = c4.toggle("Bunga berbunga (compounding)", value=True, key="sz_comp", help="Mati = ukuran dihitung dari modal awal terus.")
    period = c5.radio("Periode", ["Unseen (ujian)", "Seluruh periode"], horizontal=True, key="sz_per")
    start = ctx["split"] if period.startswith("Unseen") else ctx["is_start"]
    end = ctx["n"]
    m1, tr = E.run(fam, p, cc, start, end, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], want_equity=True, lev=1.0)
    if len(tr["ent"]) < 3:
        st.info("Trade terlalu sedikit di periode ini.")
        return
    _, tri = E.run(fam, p, cc, ctx["is_start"], ctx["split"], cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=1.0)
    f = K.trade_sizes(mode, fam, p, cc, tr, val, max_exp, kelly_ret=tri["ret"], kelly_frac=ss.get("sz_kf", 0.25))
    E1 = tr["equity"]
    Es = K.sized_equity(tr, E1, start, f, comp)
    yrs = max((cc.index[end - 1] - cc.index[start]).days / 365.25, 1e-6)
    base = K.metrics_from_equity(E1, len(f), cc.days_per_year, yrs)
    sz = K.metrics_from_equity(Es, len(f), cc.days_per_year, yrs)
    tbl = pd.DataFrame({"Ukuran 1x (pembanding)": base, "Pengaturan ini": sz}).T
    tbl["Ukuran median (x)"] = [1.0, float(np.median(f))]
    tbl["Ukuran maks (x)"] = [1.0, float(f.max())]
    tbl = tbl.rename(columns={"net": "Net %", "maxdd": "Max DD %", "cagr": "Per tahun %", "trades": "Trades", "ret_dd": "Return/DD"})
    for k in ("Net %", "Max DD %", "Per tahun %"):
        tbl[k] *= 100
    _show(tbl)
    if mode == "kelly" and f.max() == 0:
        st.warning("Kelly = 0: rata-rata trade in-sample tidak untung, jadi Kelly menyarankan tidak trading.")
    step = max(1, len(Es) // 3000)
    x = wib(cc.index[start:end][::step])
    st.plotly_chart(_lay(go.Figure([go.Scatter(x=x, y=E1[::step] * 10000, name="1x", line=dict(color="#9e9e9e")),
                                    go.Scatter(x=x, y=Es[::step] * 10000, name="Pengaturan ini", line=dict(color=BLUE))]),
                         "Equity (modal awal 10.000)", 340), width='stretch')
    if Es.min() <= 0:
        st.error("Dengan ukuran ini modal habis (equity mencapai 0).")

    st.markdown("#### 🏁 Simulasi challenge prop firm")
    st.caption("Aturan di bawah hanya CONTOH yang mirip aturan umum prop firm, bukan aturan resmi perusahaan mana pun. Selalu cek aturan resminya.")
    pre = st.selectbox("Aturan", list(K.PRESETS), key="pf_preset")
    base_r = K.PRESETS[pre] or dict(phases=[8.0], daily=5.0, total=10.0, min_days=3, max_days=30)
    c1, c2, c3, c4, c5 = st.columns(5)
    ph = c1.text_input("Target tiap fase (%)", ", ".join(f"{x:g}" for x in base_r["phases"]), key=f"pf_ph_{pre}",
                       help="Pisahkan dengan koma, misal 10, 5 untuk dua fase.")
    daily = c2.number_input("Rugi harian maks (%)", 0.5, 50.0, float(base_r["daily"]), 0.5, key=f"pf_d_{pre}")
    total = c3.number_input("Rugi total maks (%)", 1.0, 90.0, float(base_r["total"]), 0.5, key=f"pf_t_{pre}")
    mind = c4.number_input("Min hari trading", 0, 60, int(base_r["min_days"]), key=f"pf_m_{pre}")
    maxd = c5.number_input("Batas hari per fase", 5, 365, int(base_r["max_days"]), key=f"pf_x_{pre}",
                           help="Hari kalender yang punya data candle.")
    try:
        phases = [float(x) for x in ph.replace(";", ",").split(",") if x.strip()]
    except ValueError:
        st.error("Target fase harus angka, dipisah koma."); return
    rules = dict(phases=phases, daily=daily, total=total, min_days=mind, max_days=maxd)
    if st.button("▶️ Simulasikan challenge", key="pf_run"):
        dr, dl, _ = K.daily_stats(Es, cc, start)
        day_idx = cc.index[cc.day_end[(cc.day_end >= start) & (cc.day_end < end)]]
        ss["pf_res"] = dict(mc=K.prop_monte_carlo(dr, dl, rules), hist=K.prop_history(dr, dl, day_idx, rules), rules=rules,
                            key=(ctx["skey"], mode, val, max_exp, comp, period))
    r = ss.get("pf_res")
    if r and r["key"] == (ctx["skey"], mode, val, max_exp, comp, period):
        mc = r["mc"]
        if mc is None:
            st.info("Hari trading terlalu sedikit untuk simulasi.")
        else:
            c = st.columns(5)
            c[0].metric("Peluang lolos semua fase", _pct(mc["pass_all"]))
            c[1].metric("Gagal: rugi harian", _pct(mc["fail_daily"])); c[2].metric("Gagal: rugi total", _pct(mc["fail_total"]))
            c[3].metric("Waktu habis", _pct(mc["timeout"]))
            c[4].metric("Hari sampai lolos (median)", f"{mc['median_days']:.0f}" if mc["median_days"] else "-")
            st.caption(" · ".join(f"Lolos fase {i + 1}: {_pct(v)}" for i, v in enumerate(mc["phase_pass"])) +
                       f" · {mc['runs']:,} simulasi dari hasil harian asli (diambil acak per blok 5 hari).")
        h = pd.Series([x["result"] for x in r["hist"]]).value_counts()
        st.markdown("**Kalau mulai challenge di awal tiap bulan (data asli, tanpa acak):** " +
                    " · ".join(f"{k}: {v}x" for k, v in h.items()))
        st.caption("Rugi harian dihitung dari titik terendah equity di hari itu (pakai harga penutupan tiap candle), dibanding modal awal. "
                   "Hasil uji data lama, bukan jaminan lolos.")


# ================================================================== regime
def _regime_key(ctx, k):
    return (ctx["res"].get("id", "0"), ctx["n"], ctx["split"], k)


def render_regime(ctx):
    from core import regime as G
    o, cc, cfg, res = ctx["o"], ctx["cc"], ctx["cfg"], ctx["res"]
    st.caption("Membagi pasar jadi beberapa 'cuaca' (tenang/ramai, naik/turun) dari volatilitas dan tren. "
               "Model hanya belajar dari data in-sample, lalu dipakai apa adanya di data unseen. Tidak melihat masa depan.")
    k = st.radio("Jumlah regime", [2, 3, 4], index=1, horizontal=True, key="rg_k")
    key = _regime_key(ctx, k)
    store = ss.setdefault("regime", {})
    if st.button("🌦️ Deteksi regime", key="rg_run") or key in store:
        if key not in store:
            store.clear()
            with st.spinner("Mendeteksi regime..."):
                from core import isolate
                store[key] = isolate.run(G.detect, cc, ctx["is_start"], ctx["split"], k)
        lab, names, rows = store[key]
        cc._regime = lab
        _show(pd.DataFrame(rows).set_index("Regime"))
        nb = st.select_slider("Candle terakhir di chart", [500, 2000, 5000, 20000], value=2000, key="rg_nb")
        s0 = max(0, ctx["n"] - nb)
        step = max(1, (ctx["n"] - s0) // 4000)
        ii = np.arange(s0, ctx["n"], step)
        x = wib(cc.index[ii])
        f = go.Figure(go.Scatter(x=x, y=cc.c[ii], line=dict(color="#555", width=1), showlegend=False, hoverinfo="skip"))
        for r_, nm in enumerate(names):
            m = lab[ii] == r_
            f.add_trace(go.Scattergl(x=x[m], y=cc.c[ii][m], mode="markers", name=nm, marker=dict(size=3, color=REGIME_COLORS[r_])))
        f.add_vline(x=wib(cc.index[[ctx["split"]]])[0], line_dash="dash", line_color="orange")
        st.plotly_chart(_lay(f, "Harga diwarnai per regime (garis oranye = mulai unseen)", 380), width='stretch')
        st.markdown("**Hasil strategi ini per regime** (regime di candle sinyal)")
        _, itr = E.run(o["fam"], o["params"], cc, ctx["is_start"], ctx["split"], cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
        _, otr = E.run(o["fam"], o["params"], cc, ctx["split"], ctx["n"], cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
        a = pd.DataFrame(G.perf_by_regime(itr, lab, names)).set_index("Regime").add_prefix("IS ")
        b = pd.DataFrame(G.perf_by_regime(otr, lab, names)).set_index("Regime").add_prefix("Unseen ")
        _show(pd.concat([a, b], axis=1))

        st.markdown("#### 🔀 Strategi per regime")
        st.caption("Pilih strategi untuk tiap regime (atau tidak trading). 'Pilih otomatis' memakai profit factor in-sample di regime itu "
                   "(minimal 15 trade), jadi data unseen tetap jadi ujian yang jujur.")
        out = res["out"][:30]
        labels = ["— Tidak trading —"] + [f"{E.FAMILIES[x['fam']]['name']} #{i + 1}" for i, x in enumerate(out)]
        akey = f"rg_assign_{key}"
        if st.button("✨ Pilih otomatis (dari in-sample)", key="rg_auto"):
            with st.spinner("Membandingkan kandidat per regime..."):
                aa = G.auto_assign([(x["fam"], x["params"]) for x in out], cc, lab, names, ctx["is_start"], ctx["split"], cfg, cfg["direction"])
            for r_, sp in aa.items():
                ss[f"{akey}_{r_}"] = 0 if sp is None else 1 + next(i for i, x in enumerate(out) if x["params"] is sp[1])
        cols = st.columns(len(names))
        assign = {}
        me = int(ss.get("sel") or 0) if (ss.get("sel") or 0) < len(out) else 0
        for r_, nm in enumerate(names):
            ss.setdefault(f"{akey}_{r_}", min(me + 1, len(labels) - 1))
            j = cols[r_].selectbox(nm, range(len(labels)), format_func=lambda i: labels[i], key=f"{akey}_{r_}")
            assign[r_] = None if j == 0 else (out[j - 1]["fam"], out[j - 1]["params"])
        rows = {}
        for nm_, s_, e_ in (("In-sample", ctx["is_start"], ctx["split"]), ("Unseen", ctx["split"], ctx["n"])):
            cmb = G.combine(assign, cc, s_, e_, cfg, cfg["direction"])
            sm, _ = E.run(o["fam"], o["params"], cc, s_, e_, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"], daily=True)
            rows[f"{nm_}: gabungan per regime"] = dict(Trades=cmb["trades"], **{"Net %": cmb["net"] * 100, "Max DD %": cmb["maxdd"] * 100,
                                                       "Sharpe/tahun": cmb["sharpe"]}) if cmb else dict(Trades=0)
            rows[f"{nm_}: strategi ini saja"] = {"Trades": sm["trades"], "Net %": sm["net"] * 100, "Max DD %": sm["maxdd"] * 100,
                                                 "Sharpe/tahun": sm.get("sharpe", np.nan)}
        _show(pd.DataFrame(rows).T)
        st.caption("Tiap strategi berbeda bisa membuka posisinya sendiri, jadi kadang ada lebih dari satu posisi sekaligus. "
                   "Strategi gabungan ini belum bisa diekspor ke Pine.")


# ================================================================== meta-label
def render_meta(ctx):
    from core import ml
    o, cc, cfg = ctx["o"], ctx["cc"], ctx["cfg"]
    st.caption("Meta-label = model kecil yang menilai tiap sinyal strategi ini: diambil atau dilewati. Belajar hanya dari trade in-sample "
               "(validasi silang berurutan dengan jeda), lalu diuji di data unseen.")
    kind = st.radio("Model", ["auto", "hgb", "logreg"], horizontal=True, key="ml_meta_kind",
                    format_func={"auto": "Otomatis", "hgb": "Gradient boosting", "logreg": "Regresi logistik"}.get)
    key = (ctx["skey"], kind, ctx["n"])
    store = ss.setdefault("meta", {})
    if st.button("🤖 Latih meta-label", key="ml_meta_run"):
        store.clear()
        with st.spinner("Melatih model..."):
            from core import isolate
            store[key] = isolate.run(ml.meta_label, o["fam"], o["params"], cc, ctx["is_start"], ctx["split"], cfg, cfg["direction"], kind=kind)
        _trim()
    r = store.get(key)
    if not r:
        return
    if "error" in r:
        st.warning(r["error"]); return
    t = _mtable({"In-sample · strategi asli": r["rule_is"], "In-sample · + meta-label": r["meta_is"],
                 "Unseen · strategi asli": r["rule_oos"], "Unseen · + meta-label": r["meta_oos"]})
    _show(t)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ambang dipilih", f"{r['threshold']:.2f}"); c2.metric("AUC validasi silang", f"{r['auc_cv']:.3f}" if r["auc_cv"] else "-",
                                                                     help="0,5 = sama dengan menebak. Di atas 0,55 baru mulai berarti.")
    c3.metric("Win rate dasar (in-sample)", _pct(r["base_win"]))
    better = r["meta_oos"]["pf"] > r["rule_oos"]["pf"] and r["meta_oos"]["trades"] >= 20
    (st.success if better else st.info)(
        f"Di data unseen: PF {r['rule_oos']['pf']:.2f} → {r['meta_oos']['pf']:.2f}, trade {r['rule_oos']['trades']} → {r['meta_oos']['trades']}. " +
        ("Filter membantu di data unseen." if better else "Filter belum terbukti membantu (atau tradenya terlalu sedikit)."))
    st.caption(f"Model: {r['kind']} · {r['n_trades']} trade in-sample. Meta-label belum bisa diekspor ke Pine.")


# ================================================================== ML signal mode
def render_ml(res, dfc, cc):
    from core import ml
    cfg = res["cfg"]
    st.markdown("Model machine learning belajar dari indikator (momentum, volatilitas, tren, jam, dll.) untuk menebak apakah "
                "harga lebih dulu menyentuh **target** atau **stop loss** (triple-barrier). Hanya belajar dari data in-sample, "
                "dengan validasi silang berurutan yang diberi jeda supaya tidak bocor.")
    c1, c2, c3, c4, c5 = st.columns(5)
    kind = c1.selectbox("Model", ["hgb", "logreg"], key="ml_kind", format_func={"hgb": "Gradient boosting", "logreg": "Regresi logistik"}.get)
    pt = c2.number_input("Target (x ATR)", 0.5, 10.0, 2.0, 0.5, key="ml_pt")
    sl = c3.number_input("Stop loss (x ATR)", 0.5, 10.0, 1.0, 0.5, key="ml_sl")
    H = c4.number_input("Batas waktu (candle)", 4, 500, 24, 4, key="ml_H")
    direction = c5.selectbox("Arah", ["both", "long", "short"], index=["both", "long", "short"].index(cfg["direction"]), key="ml_dir",
                             format_func={"both": "Long & short", "long": "Long saja", "short": "Short saja"}.get)
    meta = res["meta"]
    n = len(dfc)
    split = int(np.searchsorted(dfc.index.values, np.datetime64(meta["split_time"].tz_convert(None))))
    is_start = min(260, split // 4)
    key = (res.get("id", "0"), n, kind, pt, sl, H, direction)
    store = ss.setdefault("ml", {})
    if st.button("🤖 Latih & uji model", type="primary", key="ml_run"):
        store.clear()
        bar = st.progress(0.0, "Menyiapkan fitur...")
        try:
            from core import isolate
            r = isolate.run(ml.ml_search, cc, is_start, split, cfg, pt=pt, sl=sl, H=int(H), kind=kind, direction=direction,
                            progress=lambda f, t: bar.progress(min(1.0, f), t))
            om2, _ = E.run("ma_cross", r["exitp"], cc, split, n, "both", cfg["cost_frac"] * 2, cfg["cost_abs"] * 2, sig=r["signals"], lev=cfg["lev"])
            r["cost2x"] = om2
            r.pop("signals", None); r.pop("is_tr", None)
            store[key] = r
        finally:
            bar.empty()
            _trim()
    r = store.get(key)
    if not r:
        st.caption("Butuh ±5-30 detik tergantung panjang data. Hasil hanya disimpan selama sesi ini.")
        return
    im, om = r["is_m"], r["oos_m"]
    min_oos_tr = max(3, cfg["min_trades"] * cfg["oos_pct"] / max(1, 100 - cfg["oos_pct"]) * 0.3)
    oos_pass = bool(om["net"] > 0 and om["pf"] >= cfg["oos_min_pf"] and om["maxdd"] <= cfg["max_dd"] and om["trades"] >= min_oos_tr)
    gi = dict(oos_pass=oos_pass, dsr=_dsr_pick(r["dsr"]), cost2x=r["cost2x"]["net"] > 0, cap_d=om["net"] < 0 or im["net"] < 0)
    score, letter, parts = R.grade(gi)
    colors = {"A": "#00c853", "B": "#64dd17", "C": "#ffd600", "D": "#ff9100", "E": "#ff1744"}
    st.markdown(f"<div style='padding:8px 12px;border-radius:8px;background:#1e1e1e;border-left:6px solid {colors.get(letter, '#888')}'>"
                f"<b style='font-size:1.4em;color:{colors.get(letter, '#888')}'>Rapor model ML: {letter}</b> · {score:.0f}/100 "
                f"<span style='color:#aaa'>(dari 3 tes: lolos unseen, DSR, biaya 2x)</span></div>", unsafe_allow_html=True)
    _show(_mtable({"In-sample (validasi silang)": im, "Unseen (ujian)": om, "Unseen, biaya 2x": r["cost2x"]}))
    best_rule = next((x for x in res["out"] if x["passed"]), res["out"][0] if res["out"] else None)
    if best_rule:
        bm, _ = E.run(best_rule["fam"], best_rule["params"], cc, split, n, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
        st.caption(f"Pembanding, strategi aturan teratas ({E.FAMILIES[best_rule['fam']]['name']}) di unseen yang sama: "
                   f"net {bm['net'] * 100:.1f}%, PF {bm['pf']:.2f}, DD {bm['maxdd'] * 100:.1f}%, {bm['trades']} trade.")
    auc = r["auc"]
    c = st.columns(4)
    c[0].metric("AUC long (CV → unseen)", f"{auc.get('long_cv', float('nan')):.3f} → {auc.get('long_oos', float('nan')):.3f}")
    c[1].metric("AUC short (CV → unseen)", f"{auc.get('short_cv', float('nan')):.3f} → {auc.get('short_oos', float('nan')):.3f}")
    c[2].metric("Peluang menang acak", f"L {_pct(r['base'][0], 0)} · S {_pct(r['base'][1], 0)}",
                help="Seberapa sering target tersentuh duluan kalau masuk di sembarang candle. Model harus lebih baik dari ini.")
    c[3].metric("DSR (ketat)" if st.session_state.get("strict_dsr") else "DSR", _pct(_dsr_pick(r["dsr"]), 0))
    st.caption("AUC 0,50 = sama dengan menebak. Model pasar biasanya hanya 0,52-0,56; selisih kecil itu sering habis dimakan biaya.")
    eq = r["oos_tr"]["equity"]
    if eq is not None and len(eq):
        x = wib(cc.index[split:split + len(eq)])
        st.plotly_chart(_lay(go.Figure(go.Scatter(x=x, y=eq * 10000, line=dict(color=BLUE))), "Equity di data unseen (modal 10.000)", 300), width='stretch')
    tt = pd.DataFrame([{"Ambang peluang": t_, "Trades": m["trades"], "Net %": m["net"] * 100, "PF": m["pf"], "Max DD %": m["maxdd"] * 100}
                       for t_, m in r["trials"]])
    st.markdown(f"**Ambang yang dicoba (in-sample)** · dipilih {r['threshold']:.3f}")
    _show(tt.set_index("Ambang peluang"))
    if r["importance"]:
        imp = pd.DataFrame(r["importance"], columns=["f", "v"]).head(15)[::-1]
        imp["f"] = imp["f"].map(lambda k_: ml.FEATURE_LABELS.get(k_, k_))
        st.plotly_chart(_lay(go.Figure(go.Bar(x=imp["v"], y=imp["f"], orientation="h", marker_color="#ab47bc")),
                             "Fitur paling berpengaruh (turunnya AUC kalau fitur diacak)", 420), width='stretch')
    st.caption(f"{r['n_train']:,} baris latih · {r['folds']} lipatan · {r['seconds']:.1f} detik. Model ML tidak bisa diekspor ke Pine "
               "(TradingView tidak bisa menjalankan model ini). Hasil uji data lama, bukan saran.")


# ================================================================== live signals
def render_live(load_closed, load_raw, cost_for, symbols_for):
    st.markdown("Sinyal terkini dari strategi yang kamu simpan. **Hanya informasi, tidak ada order otomatis.**")
    lst = ss.get("compare", {})
    if not lst:
        st.info("Belum ada strategi tersimpan. Tambahkan dari tab Export ('➕ Tambah ke perbandingan') atau upload JSON di 'Simpan & Bandingkan'.")
        return
    names = list(lst)
    pick = st.multiselect("Strategi", names, default=names[:6], key="lv_pick")
    extra = []
    srcs = {lst[k].get("source") for k in pick}
    if len(srcs) == 1 and pick:
        src = srcs.pop()
        extra = st.multiselect("Cek juga di pasar lain (opsional)", symbols_for(src), key="lv_extra")
    if not st.button("📡 Cek sinyal sekarang", type="primary", key="lv_run"):
        st.caption("Sinyal dihitung di candle yang sudah tutup. Sinyal candle yang masih berjalan bisa berubah sampai candle tutup.")
        if "lv_rows" not in ss:
            return
    else:
        from core import live as L
        rows, details = [], []
        with st.spinner("Mengambil data & menghitung..."):
            for k in pick:
                d = lst[k]
                for sym in [d["symbol"]] + [s for s in extra if s != d["symbol"]]:
                    try:
                        raw = load_raw(d["source"], sym, d["tf"], E.uses_deriv(d["family"], d["params"]))
                        closed = load_closed(d["source"], sym, d["tf"], raw)
                        price = float(closed["close"].iloc[-1])
                        cf, ca = cost_for(d["source"], sym, price)
                        s = L.state(d["family"], d["params"], d.get("direction", "both"), closed, raw, cf, ca, d.get("settings", {}).get("lev", 1.0))
                        pos = s["position"]
                        rows.append({"Strategi": k, "Pasar": sym, "TF": d["tf"],
                                     "Candle tutup terakhir (WIB)": wib([s["bar_time"]])[0].strftime("%d %b %H:%M"),
                                     "Sinyal candle tutup": s["closed_signal"],
                                     "Candle berjalan (sementara)": s["forming_signal"] or "-",
                                     "Posisi": pos["arah"] if pos else "Tidak ada",
                                     "Harga masuk": pos["harga_masuk"] if pos else None,
                                     "Stop": pos["stop"] if pos else None, "Target": pos["target"] if pos else None,
                                     "P&L posisi %": pos["pnl"] * 100 if pos else None})
                        details.append((f"{k} · {sym}", s["last"]))
                    except Exception as e:
                        rows.append({"Strategi": k, "Pasar": sym, "TF": d["tf"], "Sinyal candle tutup": f"gagal: {e}"})
        ss["lv_rows"], ss["lv_details"] = rows, details
        ss["lv_time"] = dt.datetime.now(dt.timezone(dt.timedelta(hours=7)))
    st.caption(f"Dicek {ss['lv_time']:%d %b %Y %H:%M} WIB.")
    st.dataframe(pd.DataFrame(ss["lv_rows"]), hide_index=True, width='stretch',
                 column_config={k: st.column_config.NumberColumn(format="%.5g") for k in ("Harga masuk", "Stop", "Target")} |
                 {"P&L posisi %": st.column_config.NumberColumn(format="%.2f")})
    for nm, tb in ss.get("lv_details", []):
        with st.expander(f"Sinyal terakhir · {nm}"):
            t2 = tb.copy()
            for c in ("Sinyal (candle tutup)", "Keluar"):
                if c in t2:
                    t2[c] = pd.to_datetime(t2[c], utc=True).dt.tz_convert(TZ).dt.strftime("%d %b %Y %H:%M")
            st.dataframe(t2, hide_index=True, width='stretch', column_config={"Hasil %": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Entry dijalankan di open candle berikutnya setelah sinyal (sama dengan backtest). Stop = posisi stop terakhir (termasuk trailing). "
               "Bukan saran trading.")


# ================================================================== HTML report
def report_html(ctx, ftr, im, om, fm, grade, pine, name):
    from core import report as P
    cc = ctx["cc"]
    figs, mt, bench = analytics_figs(ftr, cc, ctx["is_start"])
    eq = ftr["equity"]
    step = max(1, len(eq) // 3000)
    x = wib(cc.index[ctx["is_start"]:ctx["is_start"] + len(eq)][::step])
    fe = go.Figure(go.Scatter(x=x, y=eq[::step] * 10000, line=dict(color=BLUE), name="Equity"))
    fe.add_vline(x=wib(cc.index[[ctx["split"]]])[0], line_dash="dash", line_color="orange")
    _lay(fe, "Equity (modal 10.000) · garis oranye = mulai data unseen", 360)
    mtab = _mtable({"In-sample": im, "Unseen (ujian)": om, "Seluruh periode": fm})
    mt2 = mt.copy(); mt2.index.name = "Tahun"
    sk = A.streaks(ftr["ret"])
    extra = pd.DataFrame({"Nilai": [bench["bh_net"] * 100, bench["corr"], float(sk["max_loss"]), float(np.median(A.durations_hours(ftr, cc)))]},
                         index=["Beli & tahan, total %", "Korelasi harian dg pasar", "Kalah beruntun terpanjang", "Lama trade median (jam)"])
    res = ctx["res"]
    return P.build(f"{name} · {res['symbol']} {res['tf']}", E.describe(ctx["o"]["fam"], ctx["o"]["params"], ctx["cfg"]["direction"]),
                   grade, [("Metrik utama", mtab), ("Return bulanan %", mt2), ("Lain-lain", extra)],
                   [("Equity", fe), ("Return bulanan", figs["monthly"]), ("Strategi vs beli & tahan", figs["bench"]), ("Underwater", figs["underwater"])],
                   pine, [f"Biaya trading dihitung. Ukuran posisi {ctx['cfg']['lev']:g}x modal."])


# ================================================================== 2D parameter sensitivity
def _grid_vals(k, v, n=9):
    if k in ("sl_atr", "tp_atr", "sl_pct", "tp_pct", "trail_atr", "tp_rr", "sl_pip", "tp_pip", "sl_buf", "tp1_r", "be_r", "trail_act") \
            or isinstance(v, float):
        return sorted({round(float(x), 2) for x in np.linspace(max(0.25, v * 0.5), v * 1.5, n)})
    return sorted({int(round(x)) for x in np.linspace(max(2, v * 0.5), max(v * 1.5, v + 4), n)})


def render_sensitivity(ctx):
    o, cc, cfg = ctx["o"], ctx["cc"], ctx["cfg"]
    p = E.full_params(o["params"])
    keys = [k for k, v in E.FAMILIES[o["fam"]]["space"].items() if isinstance(v, tuple) and isinstance(p.get(k), (int, float))]
    keys += [k for k, v in E.FAMILIES[o["fam"]]["space"].items()
             if isinstance(v, list) and len(set(v)) > 1 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)
             and isinstance(p.get(k), (int, float)) and p.get(k) and k not in keys]
    keys += [k for k in E.exit_keys(p) + ["trail_atr", "tp1_r", "be_r", "trail_act"] if p.get(k, 0) > 0 and k not in keys]
    st.markdown("#### 🗺️ Peta sensitivitas parameter")
    if len(keys) < 2:
        st.caption("Strategi ini punya kurang dari 2 parameter angka untuk dipetakan.")
        return
    c1, c2, c3, c4 = st.columns(4)
    kx = c1.selectbox("Parameter sumbu X", keys, index=0, key="sens_x")
    ky = c2.selectbox("Parameter sumbu Y", [k for k in keys if k != kx], index=0, key="sens_y")
    metric = c3.selectbox("Ukuran", ["pf", "net", "ret_dd"], key="sens_m",
                          format_func={"pf": "Profit factor", "net": "Net %", "ret_dd": "Return/DD"}.get)
    per = c4.radio("Data", ["In-sample", "Unseen"], horizontal=True, key="sens_p")
    if not st.button("🗺️ Buat peta (81 backtest)", key="sens_run"):
        st.caption("Area hijau yang luas di sekitar titik pilihan = strategi tidak rapuh. Kalau hanya satu kotak yang hijau, kemungkinan kebetulan.")
        return
    xs, ys = _grid_vals(kx, p[kx]), _grid_vals(ky, p[ky])
    s_, e_ = (ctx["is_start"], ctx["split"]) if per == "In-sample" else (ctx["split"], ctx["n"])
    Z = np.full((len(ys), len(xs)), np.nan)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            q = dict(p); q[kx] = xv; q[ky] = yv
            m, _ = E.run(o["fam"], q, cc, s_, e_, cfg["direction"], cfg["cost_frac"], cfg["cost_abs"], lev=cfg["lev"])
            if m["trades"] >= 5:
                Z[i, j] = (m[metric] * 100 if metric == "net" else min(m[metric], 5.0) if metric == "pf" else m[metric])
    mid = 1.0 if metric == "pf" else 0.0
    f = go.Figure(go.Heatmap(z=Z, x=[str(x) for x in xs], y=[str(y) for y in ys], colorscale="RdYlGn", zmid=mid,
                             text=np.vectorize(lambda v: "" if v != v else f"{v:.2f}")(Z), texttemplate="%{text}"))
    f.add_trace(go.Scatter(x=[str(p[kx])] if str(p[kx]) in [str(x) for x in xs] else [], y=[str(p[ky])], mode="markers",
                           marker=dict(symbol="x", size=14, color="white"), name="Pilihan sekarang", showlegend=False))
    f.update_xaxes(title=kx); f.update_yaxes(title=ky)
    st.plotly_chart(_lay(f, f"{per}: {dict(pf='Profit factor (dipotong di 5)', net='Net %', ret_dd='Return/DD')[metric]}", 460), width='stretch')
    good = np.nanmean(Z > mid) if np.isfinite(Z).any() else 0
    st.caption(f"{good * 100:.0f}% kombinasi di peta ini {'untung' if metric != 'pf' else 'PF > 1'}. Kotak kosong = kurang dari 5 trade.")


# ================================================================== multi-timeframe scan matrix
def render_scan_matrix(rows):
    rows = [r for r in rows if r.get("fam_grades")]
    if not rows:
        return
    st.markdown("#### 🧮 Matriks rapor (pasar × timeframe × strategi)")
    st.caption("Nilai = rapor singkat 0-100 (lolos tes ketat versi dasar, Deflated Sharpe, PBO) dari variasi terbaik in-sample tiap jenis "
               "strategi. Semakin hijau semakin kuat. Ini versi cepat; rapor lengkap ada setelah dibuka di 'Hasil & Detail'.")
    mk = sorted({r["Pasar"] for r in rows}); tfs = [t for t in ["5m", "15m", "30m", "1h", "2h", "4h", "1d"] if any(r["TF"] == t for r in rows)]
    Z = np.full((len(mk), len(tfs)), np.nan); T = np.full(Z.shape, "", object)
    for r in rows:
        best = max(r["fam_grades"].items(), key=lambda kv: kv[1]["score"] or 0)
        i, j = mk.index(r["Pasar"]), tfs.index(r["TF"])
        Z[i, j] = best[1]["score"]; T[i, j] = f"{best[1]['letter']} · {E.FAMILIES[best[0]]['name'][:22]}"
    f = go.Figure(go.Heatmap(z=Z, x=tfs, y=mk, text=T, texttemplate="%{text}", colorscale="RdYlGn", zmin=0, zmax=100,
                             hovertemplate="%{y} %{x}: %{z:.0f}/100<br>%{text}<extra></extra>"))
    st.plotly_chart(_lay(f, "Strategi terbaik per pasar & timeframe", max(260, 46 * len(mk) + 90)), width='stretch')
    cols = [f"{r['Pasar']} {r['TF']}" for r in rows]
    fams = sorted({k for r in rows for k in r["fam_grades"]}, key=lambda k: -np.nanmean([r["fam_grades"].get(k, {}).get("score") or np.nan for r in rows]))
    Z2 = np.full((len(fams), len(rows)), np.nan); T2 = np.full(Z2.shape, "", object)
    for j, r in enumerate(rows):
        for i, k in enumerate(fams):
            g = r["fam_grades"].get(k)
            if g:
                Z2[i, j] = g["score"]; T2[i, j] = g["letter"] + ("✓" if g["passed"] else "")
    f2 = go.Figure(go.Heatmap(z=Z2, x=cols, y=[E.FAMILIES[k]["name"] for k in fams], text=T2, texttemplate="%{text}", colorscale="RdYlGn",
                              zmin=0, zmax=100, hovertemplate="%{y}<br>%{x}: %{z:.0f}/100<extra></extra>"))
    st.plotly_chart(_lay(f2, "Rapor tiap jenis strategi (✓ = lolos tes dasar)", max(320, 22 * len(fams) + 120)), width='stretch')
    st.caption("Jenis strategi yang hijau di banyak pasar/timeframe lebih bisa dipercaya daripada yang hanya bagus di satu tempat.")


# ================================================================== strategy similarity / clusters
def render_similarity(res, dfc, cc):
    cfg, meta = res["cfg"], res["meta"]
    st.markdown("#### 🔗 Kemiripan strategi")
    st.caption("Banyak hasil pencarian sebenarnya strategi yang hampir sama (return hariannya bergerak bersamaan). "
               "Di sini strategi dikelompokkan dari korelasi return harian di data in-sample saja.")
    a, b = st.columns(2)
    thr = a.slider("Dianggap kembar kalau korelasi ≥", 0.5, 0.95, 0.8, 0.05, key="sim_thr")
    nmax = b.select_slider("Jumlah strategi dibandingkan", [10, 20, 30, 40], value=20, key="sim_n")
    if st.button("🔗 Hitung kemiripan", key="sim_run"):
        out = res["out"]
        order = [i for i, o in enumerate(out) if o["passed"]] + [i for i, o in enumerate(out) if not o["passed"]]
        cands = [(f"#{i + 1} {E.FAMILIES[out[i]['fam']]['name'][:20]}", out[i]["fam"], out[i]["params"]) for i in order[:nmax]]
        with st.spinner("Menghitung korelasi..."):
            res.setdefault("search_tests", {})["sim"] = (thr, nmax, R.similarity(cands, cc, meta["is_start"], meta["split"], cfg, thr))
    got = res.get("search_tests", {}).get("sim")
    if not got or got[2] is None:
        if got:
            st.info("Strategi yang punya trade kurang dari 2.")
        return
    _, _, s = got
    groups = s["groups"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategi dibandingkan", s["n"]); c2.metric("Kelompok berbeda", len(groups))
    c3.metric("Duplikat (bisa dibuang)", s["n"] - len(groups))
    C = s["corr"]
    f = go.Figure(go.Heatmap(z=C.values, x=list(C.columns), y=list(C.index), zmin=-1, zmax=1, colorscale="RdBu_r",
                             hovertemplate="%{y}<br>%{x}<br>korelasi %{z:.2f}<extra></extra>"))
    st.plotly_chart(_lay(f, "Korelasi return harian in-sample (diurutkan per kelompok)", max(380, 18 * len(C) + 140)), width='stretch')
    rows = [{"Kelompok": k + 1, "Wakil (peringkat in-sample terbaik)": g[0], "Anggota": len(g), "Kembarannya": ", ".join(g[1:]) or "-"}
            for k, g in enumerate(groups)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption("Untuk portofolio, ambil satu wakil dari tiap kelompok. Tombol 'Buat portofolio' di atas memilih otomatis 3-5 strategi "
               "dengan korelasi rendah (juga dari data in-sample saja), lalu menunjukkan hasilnya di data unseen.")
