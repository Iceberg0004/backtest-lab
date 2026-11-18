"""v7 UI: pilih jenis SL/TP untuk pencarian + alat 'Bandingkan exit'."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from core import engine as E, exits as X, exitcmp as C

SL_ALL = [k for k in X.SL_TYPES if k != 14]


def render_exit_pick():
    """Expander di tab cari. Kembalikan None kalau semua jenis boleh (perilaku default)."""
    with st.expander("🎯 SL / TP lengkap (pilih jenis yang dicoba)"):
        st.caption(f"Default: semua dicoba ({len(X.SL_TYPES) - 1} jenis SL, {len(X.TP_TYPES)} jenis TP, {len(X.TRAIL_TYPES)} jenis trailing, "
                   "TP1/TP2 + breakeven, dan aturan trade). Batasi kalau kamu hanya mau jenis tertentu.")
        sl = st.multiselect("Jenis stop loss", list(X.SL_TYPES), default=SL_ALL, key="v7_sl", format_func=lambda k: X.SL_TYPES[k])
        if 14 in sl:
            st.warning("'Tanpa stop loss' berisiko besar: satu trade bisa rugi sangat dalam. Hanya untuk tes.")
        tp = st.multiselect("Jenis take profit", list(X.TP_TYPES), default=list(X.TP_TYPES), key="v7_tp", format_func=lambda k: X.TP_TYPES[k])
        c1, c2, c3 = st.columns(3)
        tr = c1.checkbox("Trailing stop", True, key="v7_trail")
        mu = c2.checkbox("TP1/TP2 + breakeven", True, key="v7_multi")
        ru = c3.checkbox("Aturan trade", True, key="v7_rules",
                         help="Time stop, tutup akhir hari / akhir pekan, jeda setelah rugi, maks trade per hari, batas rugi/untung harian, "
                              "tidak masuk ulang ke arah yang sama.")
    if not sl:
        sl = SL_ALL
    if not tp:
        tp = list(X.TP_TYPES)
    if sorted(sl) == SL_ALL and sorted(tp) == list(X.TP_TYPES) and tr and mu and ru:
        return None
    return {"sl": [int(x) for x in sl], "tp": [int(x) for x in tp], "trail": bool(tr), "multi": bool(mu), "rules": bool(ru)}


def no_sl_warning(p):
    if int(E.full_params(p).get("sl_mode", 0)) == 14:
        st.warning("⚠️ Strategi ini tanpa stop loss. Hasil bisa terlihat bagus, tapi satu trade bisa rugi sangat dalam.")


def render_exit_compare(ctx):
    o, cc, cfg = ctx["o"], ctx["cc"], ctx["cfg"]
    st.markdown("#### ⚖️ Bandingkan exit")
    st.caption("Sinyal masuk tetap sama, exit diganti-ganti (semua jenis SL, TP, trailing, TP1/TP2, aturan). "
               "Diurutkan dari hasil data unseen. Hati-hati: memilih yang terbaik di unseen juga termasuk 'memilih', "
               "jadi baris teratas cenderung terlalu bagus. Lebih aman pilih yang untung di in-sample DAN unseen.")
    key = f"v7cmp_{ctx['skey']}"
    n_c = st.select_slider("Jumlah kombinasi exit", [60, 120, 200], value=120, key=f"v7n_{ctx['skey']}")
    if st.button("▶️ Bandingkan exit", key=f"v7b_{ctx['skey']}"):
        bar = st.progress(0.0, text="Menguji kombinasi exit...")
        rows = C.compare(o["fam"], o["params"], cc, ctx["is_start"], ctx["split"], ctx["n"], cfg["direction"], cfg["cost_frac"],
                         cfg["cost_abs"], cfg["lev"], n_combos=n_c, min_trades=max(5, int(cfg.get("min_trades", 20)) // 2),
                         progress=lambda f: bar.progress(min(f, 1.0), text=f"Menguji kombinasi exit... {f * 100:.0f}%"))
        bar.empty()
        st.session_state[key] = [dict(label=r["label"], is_m=r["is_m"], oos_m=r["oos_m"], ok=r["ok"]) for r in rows]
    rows = st.session_state.get(key)
    if not rows:
        return
    tbl = pd.DataFrame([{"Exit": r["label"],
                         "IS Net %": r["is_m"]["net"] * 100, "IS PF": r["is_m"]["pf"], "IS trade": r["is_m"]["trades"],
                         "Unseen Net %": r["oos_m"]["net"] * 100, "Unseen PF": r["oos_m"]["pf"], "Unseen DD %": r["oos_m"]["maxdd"] * 100,
                         "Unseen trade": r["oos_m"]["trades"],
                         "Untung di dua-duanya": "✅" if r["is_m"]["net"] > 0 and r["oos_m"]["net"] > 0 and r["ok"] else ""} for r in rows])
    base = rows[0]["oos_m"]["ret_dd"]
    both = int((tbl["Untung di dua-duanya"] == "✅").iloc[1:].sum())
    beat = sum(1 for r in rows[1:] if r["ok"] and r["oos_m"]["ret_dd"] > base)
    st.write(f"Baris 1 = exit asli. Dari {len(rows) - 1} kombinasi: {both} untung di in-sample dan unseen, {beat} lebih baik dari exit asli di unseen.")
    st.dataframe(tbl, hide_index=True, width='stretch', height=380,
                 column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                ["IS Net %", "IS PF", "Unseen Net %", "Unseen PF", "Unseen DD %"]})
