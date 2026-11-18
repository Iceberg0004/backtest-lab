"""Backtest Lab pribadi (non-komersial) - meniru ALUR backtest seperti 'Quant' di Vela.
Jalankan: pip install -r requirements.txt && streamlit run app.py"""
import json, os, time, datetime as dt

# Streamlit Community Cloud needs a writable cache path. /tmp always works;
# VELA_CACHE can override it. Set these before importing numba-using modules.
os.environ.setdefault("VELA_CACHE", "/tmp/vela_cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys as _sys

# After a git pull, Streamlit re-runs this file but keeps our own modules (core/*, ui_*) imported from the OLD code.
# If any of them changed on disk, drop them all so the imports below load the new code together.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LOCAL = {n: m for n, m in list(_sys.modules.items()) if n != "__main__" and getattr(m, "__file__", None)
          and os.path.abspath(m.__file__).startswith(_HERE + os.sep)}
if any(getattr(m, "_bt_mtime", None) != os.path.getmtime(m.__file__) for m in _LOCAL.values() if os.path.exists(m.__file__)):
    for _n in _LOCAL:
        _sys.modules.pop(_n, None)

from core.data import SOURCES, KEY_SOURCES, closed_only, TF_MINUTES
from core import engine as E
from core.pine import to_pine, NoPine
from core.prompt import parse
from core import robust as R, realism as X
import ui_extra as U
import ui_v4 as V
import ui_v6 as V6
import ui_v7 as V7
for _m in list(_sys.modules.values()):   # remember the file version each local module was loaded from (see above)
    _f = getattr(_m, "__file__", None)
    if _f and os.path.abspath(_f).startswith(_HERE + os.sep) and not hasattr(_m, "_bt_mtime") and os.path.exists(_f):
        try:
            _m._bt_mtime = os.path.getmtime(_f)
        except Exception:
            pass
import threading

def _bg_probe():
    try:
        from core.data import probe_sources
        probe_sources()
    except Exception as e:
        print(f"[data] probe thread failed: {type(e).__name__}: {e}", flush=True)

threading.Thread(target=_bg_probe, daemon=True, name="probe-sources").start()

st.set_page_config(page_title="Backtest Lab", page_icon="📈", layout="wide")
st.markdown("""<style>
.stButton button, .stDownloadButton button {min-height: 2.8rem; font-size: 1.02rem;}
.stButton button[kind="primary"] {min-height: 3.4rem; font-size: 1.15rem; font-weight: 700;}
[data-testid="stMetricValue"] {font-size: 1.45rem;}
@media (max-width: 640px) {
  .block-container {padding: 0.6rem 0.6rem 3rem 0.6rem;}
  [data-testid="stMetricValue"] {font-size: 1.15rem;}
  .stTabs [data-baseweb="tab"] {padding: 0.35rem 0.5rem; font-size: 0.9rem;}
  h1, h2, h3 {font-size: 1.15rem !important;}
}
[data-testid="stDataFrame"] {overflow-x: auto;}
</style>""", unsafe_allow_html=True)
TZ = "Asia/Jakarta"
FX_SPREAD = {"EURUSD": 1.0, "GBPUSD": 1.5, "USDJPY": 1.2, "AUDUSD": 1.2, "USDCAD": 1.8, "USDCHF": 1.5,
             "NZDUSD": 1.8, "XAUUSD (Gold futures GC=F)": 3.0, "XAUUSD": 3.0, "XAGUSD": 2.5, "EURJPY": 1.5,
             "GBPJPY": 2.5, "EURGBP": 1.2, "EURAUD": 2.0, "EURCAD": 2.2, "EURCHF": 1.8, "EURNZD": 3.0, "GBPAUD": 2.8,
             "GBPCAD": 3.0, "GBPCHF": 2.8, "GBPNZD": 4.0, "AUDJPY": 1.8, "AUDCAD": 2.0, "AUDCHF": 2.0, "AUDNZD": 2.5,
             "CADJPY": 2.0, "CHFJPY": 2.5, "NZDJPY": 2.2, "NZDCAD": 2.5, "CADCHF": 2.5,
             # indices / oil: spread in points (SPX500 in 0.1 points, oil in cents)
             "US30": 2.5, "NAS100": 1.5, "SPX500": 5.0, "GER40": 1.5, "UK100": 1.5, "JPN225": 10.0, "WTI": 3.0, "BRENT": 3.0}
# how far back small timeframes may go (keeps memory/time sane on the free cloud)
MAX_DAYS = {"binance": {"1m": 120, "3m": 365, "5m": 1095, "15m": 2190},
            "dukascopy": {"1m": 120, "5m": 1095, "15m": 2190, "30m": 3650}}
DIR_LABEL = {"both": "Long & Short", "long": "Hanya Long", "short": "Hanya Short"}

ss = st.session_state
defaults = dict(source="binance", symbol="BTCUSDT", tf="1h", direction="both", oos_pct=30, folds=4,
                min_pf=1.3, min_wr=0.0, max_dd=30.0, min_trades=50, families=list(E.FAMILIES),
                n_variants=3000, time_budget=120, mode="ketat", algo="random", combo=False, min_wf=60, min_stab=60, oos_min_pf=1.1, lev=1.0,
                fee=0.10, slip=0.05, spread=1.0, fx_comm=0.0, fx_slip=0.2, refresh=15, prompt="",
                use_funding=False, use_deriv=False, use_swap=False, swap_long=2.0, swap_short=-0.5,
                crypto_preset="Binance spot (fee 0,1%)", fx_preset="Spread rata-rata (bawaan)", strict_dsr=False)
for k, v in defaults.items():
    ss.setdefault(k, v)
ss.families = [f for f in ss.families if f in E.FAMILIES]

# ---- v8: phone browsers drop the connection when minimized. Settings, the running job and the opened result are kept on
# the server under a token in the page URL (?s=...), so coming back continues where you were instead of starting over.
import core.jobs as J
PERSIST_KEYS = list(defaults) + ["start_date_saved", "custom_sym", "v7_pick_on", "v7_sl", "v7_tp", "v7_trail", "v7_multi", "v7_rules", "adv_tool",
                                 "scan_src", "scan_tfs", "scan_var", "scan_time"]
_tok = st.query_params.get("s")
if not _tok:
    _tok = J.new_token()
    st.query_params["s"] = _tok
if ss.get("_tok") != _tok:
    ss["_tok"] = _tok
    _saved = J.sess_load(_tok)
    for _k, _v in (_saved.get("settings") or {}).items():
        if _k in PERSIST_KEYS and _v is not None:
            ss[_k] = _v
    for _k in ("job_search", "res_job", "sel", "job_scan", "scan_job", "res_scan"):
        if _saved.get(_k) is not None:
            ss[_k] = _saved[_k]
    if ss.get("res_job") and not ss.get("res"):
        _r = J.result(ss["res_job"])
        if _r:
            ss.res = _r
    if ss.get("scan_job") and not ss.get("scan_rows"):
        _r = J.result(ss["scan_job"])
        if _r:
            ss.scan_rows, ss.scan_results = _r
    if ss.get("res_scan") and not ss.get("res"):
        _r = J.result(ss["res_scan"][0])
        if _r and ss["res_scan"][1] in _r[1]:
            ss.res = _r[1][ss["res_scan"][1]]
    ss.families = [f for f in ss.families if f in E.FAMILIES] or list(E.FAMILIES)


def _json_ok(v):
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, str)):
        return True
    if isinstance(v, (list, tuple)):
        return all(isinstance(x, (bool, int, float, str)) for x in v)
    return False


def _save_session():
    data = dict(settings={k: ss.get(k) for k in PERSIST_KEYS if k in ss and _json_ok(ss.get(k))})
    for k in ("job_search", "res_job", "sel", "job_scan", "scan_job", "res_scan"):
        data[k] = ss.get(k)
    J.sess_save(ss["_tok"], data)


@st.fragment(run_every=2)
def job_panel(jid, what):
    """Progress of a background job. Re-checks every 2 s; when it is done the whole page refreshes to show the result."""
    s_ = J.status(jid)
    if not s_ or s_["state"] not in ("antri", "jalan"):
        st.rerun()
        return
    st.progress(s_["frac"], text=s_["text"] or "Berjalan...")
    el = time.time() - (s_.get("started") or time.time())
    st.caption(f"⏳ {what} berjalan di server ({el:.0f} detik). Boleh minimize atau kunci HP: hasilnya tidak hilang. "
               "Kalau halaman sempat terputus, buka lagi link yang sama.")


# broker cost presets (EXAMPLES, always editable). Forex/gold/index: spread multiplier on the typical spread + commission.
CRYPTO_PRESETS = {"Binance spot (fee 0,1%)": (0.10, 0.05), "Binance spot bayar fee pakai BNB (0,075%)": (0.075, 0.05),
                  "Binance futures taker (0,05%)": (0.05, 0.03), "Binance futures maker (0,02%)": (0.02, 0.0), "Atur sendiri": None}
FX_PRESETS = {"Spread rata-rata (bawaan)": (1.0, 0.0), "Akun standar (contoh: spread lebar, tanpa komisi)": (1.6, 0.0),
              "Akun raw/ECN (contoh: spread tipis + komisi)": (0.35, 0.0035), "Atur sendiri": None}


def spread_for(sym):
    """Spread (pip) for a market: what you set for it in this session, else the typical spread x account preset."""
    mem = ss.get("cost_mem", {}).get(sym)
    if mem and "spread" in mem:
        return mem["spread"]
    mult = (FX_PRESETS.get(ss.get("fx_preset", "")) or (1.0, 0))[0]
    return round(FX_SPREAD.get(sym, 1.5) * mult, 2)


def _restore_costs():
    ss.spread = spread_for(ss.symbol)
    mem = ss.get("cost_mem", {}).get(ss.symbol, {})
    for k in ("swap_long", "swap_short"):
        if k in mem:
            ss[k] = mem[k]


def _apply_preset():
    if ss.source == "binance":
        v = CRYPTO_PRESETS.get(ss.crypto_preset)
        if v:
            ss.fee, ss.slip = v
    else:
        v = FX_PRESETS.get(ss.fx_preset)
        if v:
            ss.fx_comm = v[1]
            ss.get("cost_mem", {}).get(ss.symbol, {}).pop("spread", None)
            ss.spread = spread_for(ss.symbol)


def _remember_costs():
    if ss.source != "binance":
        ss.setdefault("cost_mem", {})[ss.symbol] = dict(spread=float(ss.spread), swap_long=float(ss.swap_long), swap_short=float(ss.swap_short))


def apply_prompt():
    got = parse(ss.prompt)
    for k, v in got.items():
        ss[k] = v
    if "source" in got and got["source"] != "binance":
        ss.spread = FX_SPREAD.get(got.get("symbol", ""), 1.5)
    ss.prompt_msg = got or "Tidak ada target yang terbaca dari kalimat itu."


def to_wib(idx):
    return pd.DatetimeIndex(idx).tz_convert(TZ)


# ------------------------------------------------------------------ market (main page) + extra data options (sidebar)
with st.sidebar:
    st.header("Data (lanjutan)")
    use_keys = st.checkbox("Aktifkan sumber dengan API key (OANDA / Twelve Data)", value=False)
    if use_keys:
        for k, env in KEY_SOURCES.items():
            v = st.text_input(f"{env}", value=os.environ.get(env, ""), type="password")
            if v:
                os.environ[env] = v.strip()
    custom = st.text_input("Simbol lain (opsional, contoh ETHUSDT / AAPL / ^GSPC)", key="custom_sym")
src_keys = [k for k in SOURCES if use_keys or k not in KEY_SOURCES]
if ss.source not in src_keys:
    ss.source = "binance"
def _on_source():
    ss.symbol = SOURCES[ss.source].symbols()[0]
    _restore_costs()
st.markdown("#### 📈 Backtest Lab")
with st.container(border=True):
    _m1, _m2, _m3 = st.columns([1.3, 1.3, 0.8])
    _m1.selectbox("Sumber data", src_keys, key="source", format_func=lambda k: SOURCES[k].label, on_change=_on_source)
    src = SOURCES[ss.source]
    syms = src.symbols()
    if not custom:
        _m2.selectbox("Pasar", syms if ss.symbol in syms else syms + [ss.symbol], key="symbol", on_change=_restore_costs)
    else:
        _m2.text_input("Pasar", value=custom.strip().upper(), disabled=True)
    symbol = custom.strip().upper() or ss.symbol
    if ss.tf not in src.timeframes:
        ss.tf = "1h"
    _m3.selectbox("Timeframe", src.timeframes, key="tf")
    tf = ss.tf
with st.sidebar:
    st.caption(src.limits_note(tf))
    default_start = {"1m": 30, "3m": 90, "5m": 180, "15m": 540}.get(tf) if ss.source == "binance" else \
        {"1m": 120, "5m": 730, "15m": 1825, "30m": 3650}.get(tf)
    first = dt.date(2017, 8, 17) if ss.source == "binance" else dt.date(2003, 5, 5)
    start_default = (dt.date.today() - dt.timedelta(days=default_start)) if default_start else first
    _sd = ss.get("start_date_saved")
    try:
        _sd = dt.date.fromisoformat(_sd) if _sd else None
    except Exception:
        _sd = None
    start_date = st.date_input("Data mulai dari", value=_sd or start_default, min_value=dt.date(2000, 1, 1))
    ss["start_date_saved"] = None if start_date == start_default else str(start_date)
    _cap = MAX_DAYS.get(ss.source, {}).get(tf)
    if _cap and start_date < dt.date.today() - dt.timedelta(days=_cap):
        start_date = dt.date.today() - dt.timedelta(days=_cap)
        st.caption(f"Timeframe {tf} dibatasi {_cap} hari terakhir (mulai {start_date:%d %b %Y}) supaya tidak kehabisan memori.")
    if ss.source == "binance":
        st.checkbox("📊 Pakai data derivatif (funding, open interest, long/short, order flow)", key="use_deriv",
                    help="Data gratis dari Binance Futures dan arsip data.binance.vision. Dipakai sebagai filter, jenis strategi baru dan fitur ML. "
                         "Pertama kali bisa 2-4 menit per koin (mengunduh riwayat open interest harian), berikutnya cepat. "
                         "Open interest & long/short tersedia sejak ±Sep 2020 (API Binance sendiri hanya menyimpan 30 hari, jadi riwayat lama "
                         "diambil dari arsip harian). Data likuidasi tidak ada sumber riwayat gratis, jadi tidak dipakai.")
with st.sidebar:
    st.slider("Auto-refresh chart (detik)", 5, 120, key="refresh")
    st.caption("Aplikasi pribadi, non-komersial. Backtest bukan jaminan hasil.")


@st.cache_data(ttl=120, show_spinner=False, max_entries=6)
def load_deriv(symbol, tf, start):
    from core import deriv as D
    base = SOURCES["binance"].get(symbol, tf, start=pd.Timestamp(start, tz="UTC"), refresh=False)
    e, info = D.enrich(base, symbol, tf)
    return e[D.COLS], info


@st.cache_data(ttl=60, show_spinner=False, max_entries=3)
def load_bench(tf, start):
    b = SOURCES["binance"].get("BTCUSDT", tf, start=pd.Timestamp(start, tz="UTC"), refresh=True)
    return b["close"].rename("bench")


@st.cache_data(ttl=5, show_spinner=False)
def load(source, symbol, tf, start, deriv=False):
    df = SOURCES[source].get(symbol, tf, start=pd.Timestamp(start, tz="UTC"), refresh=True)
    if source == "binance" and symbol != "BTCUSDT":
        try:      # v7.1: BTC close for the "relative strength vs BTC" strategy
            df = df.join(load_bench(tf, start), how="left")
        except Exception:
            pass
    if deriv and source == "binance":
        try:
            x, info = load_deriv(symbol, tf, start)
            df = df.join(x, how="left")
            df.attrs["deriv_info"] = info
        except Exception as e:  # never break the app because of the extra data
            df.attrs["deriv_info"] = {"error": str(e)}
    return df


def deriv_on():
    return bool(ss.get("use_deriv")) and ss.source == "binance"


def get_data(show=True):
    key = (ss.source, symbol, tf, str(start_date), deriv_on())
    try:
        if show:
            with st.spinner(f"Mengambil data {symbol} {tf} (pertama kali bisa agak lama, berikutnya hanya candle baru)" +
                            (" + data derivatif (pertama kali 2-4 menit)" if deriv_on() else "") + "..."):
                df = load(ss.source, symbol, tf, start_date, deriv_on())
        else:
            df = load(ss.source, symbol, tf, start_date, deriv_on())
        if df is None or not len(df):
            raise ValueError("data kosong")
        ss["_last_df"] = (key, df)   # one market only (same object as the loader cache, no extra copy)
        return df
    except Exception as e:
        last = ss.get("_last_df")
        if last and last[0] == key:   # a short hiccup of the free source: keep working with the data we already have
            st.caption(f"⚠️ Gagal mengambil candle terbaru ({str(e)[:80]}). Memakai data terakhir yang sudah ada.")
            return last[1]
        st.error(f"Gagal mengambil data: {e}")
        return None


def cost_params(price):
    """Returns (cost_frac per side, cost_abs per side in price units, pine commission %)."""
    if ss.source == "binance":
        cf = (ss.fee + ss.slip) / 100
        return cf, 0.0, ss.fee + ss.slip
    pip = src.pip_size(symbol) or 0.0001
    ca = (ss.spread / 2 + ss.fx_slip) * pip
    cf = ss.fx_comm / 100
    return cf, ca, ss.fx_comm + ca / price * 100


def cost_for(source, sym, price):
    """(cost_frac, cost_abs) for any symbol with the current cost settings (used by multi-asset / compare)."""
    if source == "binance":
        return (ss.fee + ss.slip) / 100, 0.0
    pip = SOURCES[source].pip_size(sym) or 0.0001
    return ss.fx_comm / 100, ((ss.spread if sym == symbol else spread_for(sym)) / 2 + ss.fx_slip) * pip


@st.cache_data(ttl=3600, show_spinner=False)
def get_funding(sym):
    return X.funding_series(sym)


def make_cache(dfc, cfg, sym, tf_):
    """Indicator cache + holding costs (funding / swap) as set for this search. Returns (cache, carry info)."""
    cc = E.Cache(dfc)
    car = (cfg or {}).get("carry") or {}
    info = None
    if car.get("kind") == "funding":
        try:
            s_, note = get_funding(sym)
        except Exception as e:
            s_, note = pd.Series(dtype=float), f"gagal mengambil funding ({e}); dipakai tarif dasar 0,01%/8 jam"
        cc.carry_long, cc.carry_short, info = X.funding_carry(cc, tf_, s_, note)
    elif car.get("kind") == "swap":
        cc.carry_long, cc.carry_short = X.swap_carry(cc, car["long"], car["short"])
    return cc, info


def shared_cache(dfc, cfg, sym, tf_):
    """One indicator cache per session for the same data + holding-cost settings (detail page, ML, walk-forward,
    portfolio all reuse it instead of building their own copy on every rerun; saves memory on the 1 GB cloud)."""
    key = (sym, tf_, len(dfc), str(dfc.index[-1]) if len(dfc) else "", repr((cfg or {}).get("carry")))
    got = ss.get("_shared_cc")
    if got and got[0] == key:
        return got[1], got[2]
    ss["_shared_cc"] = None
    cc, info = make_cache(dfc, cfg, sym, tf_)
    ss["_shared_cc"] = (key, cc, info)
    return cc, info


def carry_cfg():
    if ss.source == "binance" and ss.use_funding:
        return dict(kind="funding")
    if ss.source != "binance" and ss.use_swap:
        return dict(kind="swap", long=float(ss.swap_long), short=float(ss.swap_short))
    return None


# ------------------------------------------------------------------ charts
SUB_PANEL = {"_rsi": ("RSI", ("lower", "upper", "exit_level")), "_cci": ("CCI", ()), "_wr": ("Williams %R", ("wr_lo", "wr_hi")),
             "_mfi": ("MFI", ("mfi_lo", "mfi_hi")), "_sqz": ("Squeeze momentum", ())}
def candle_fig(df, fam=None, p=None, tr=None, cc=None, start_i=None, height=620, split_time=None):
    sub = df
    two = fam is not None and bool(E.FAMILIES[fam].get("sub"))
    fig = make_subplots(rows=2 if two else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.75, 0.25] if two else [1.0], vertical_spacing=0.03)
    x = to_wib(sub.index)
    fig.add_trace(go.Candlestick(x=x, open=sub.open, high=sub.high, low=sub.low, close=sub.close, name="Harga",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    if fam is not None:
        _, _, _, _, lines = E.signals(fam, p, cc)
        off = len(cc.c) - len(sub)
        for name, arr in lines.items():
            a = arr[off:]
            if name in SUB_PANEL:
                lab, lv_keys = SUB_PANEL[name]
                fig.add_trace(go.Scatter(x=x, y=a, name=lab, line=dict(color="#7e57c2")), row=2, col=1)
                for k_ in lv_keys:
                    if k_ in p:
                        fig.add_hline(y=p[k_], line_dash="dot", line_color="gray", row=2, col=1)
                if name == "_cci" and "cci_lvl" in p:
                    for lv in (-p["cci_lvl"], p["cci_lvl"]):
                        fig.add_hline(y=lv, line_dash="dot", line_color="gray", row=2, col=1)
            elif name == "_macd":
                fig.add_trace(go.Scatter(x=x, y=a, name="MACD", line=dict(color="#2962ff")), row=2, col=1)
            elif name == "_macd_signal":
                fig.add_trace(go.Scatter(x=x, y=a, name="Signal", line=dict(color="#ff6d00")), row=2, col=1)
            elif name == "_stoch":
                fig.add_trace(go.Scatter(x=x, y=a, name="Stoch RSI %K", line=dict(color="#7e57c2")), row=2, col=1)
                for lv in (p["k_lo"], p["k_hi"]):
                    fig.add_hline(y=lv, line_dash="dot", line_color="gray", row=2, col=1)
            elif name == "_z":
                fig.add_trace(go.Scatter(x=x, y=a, name="Z-score", line=dict(color="#7e57c2")), row=2, col=1)
                for lv in (-p["z_entry"], p["z_entry"]):
                    fig.add_hline(y=lv, line_dash="dot", line_color="gray", row=2, col=1)
            elif name == "_adx":
                continue
            else:
                fig.add_trace(go.Scatter(x=x, y=a, name=name, line=dict(width=1.3)), row=1, col=1)
        if tr is not None and len(tr["ent"]):
            ent, ex = tr["ent"] - off, tr["ex"] - off
            ok = ent >= 0
            idx = sub.index
            for d, col, lab, sym in ((1, "#00c853", "Long", "triangle-up"), (-1, "#ff1744", "Short", "triangle-down")):
                m = ok & (tr["dir"] == d)
                if m.any():
                    fig.add_trace(go.Scatter(x=to_wib(idx[ent[m]]), y=tr["ep"][m], mode="markers+text", name=lab,
                                             text=[lab] * int(m.sum()), textposition="bottom center" if d == 1 else "top center",
                                             marker=dict(symbol=sym, size=12, color=col)), row=1, col=1)
            m = (ex >= 0)
            if m.any():
                reason = np.array(["Exit", "SL", "TP", "Akhir", "Waktu", "Sesi", "TimeStop"])[tr["reason"][m]]
                txt = [f"{r} {v*100:+.2f}%" for r, v in zip(reason, tr["ret"][m])]
                fig.add_trace(go.Scatter(x=to_wib(idx[ex[m]]), y=tr["xp"][m], mode="markers+text", name="Exit",
                                         text=txt, textposition="top center", textfont=dict(size=10),
                                         marker=dict(symbol="x", size=9, color="#ffd600")), row=1, col=1)
    if split_time is not None and split_time >= sub.index[0]:
        fig.add_vrect(x0=pd.Timestamp(split_time).tz_convert(TZ), x1=x[-1], fillcolor="orange", opacity=0.07,
                      line_width=0, annotation_text="data unseen (OOS)", annotation_position="top left")
    if ss.source != "binance":  # hide forex weekend gap
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_layout(height=height, xaxis_rangeslider_visible=False, template="plotly_dark",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.04))
    return fig


def selected_strategy():
    res = ss.get("res")
    if not res or ss.get("sel") is None:
        return None
    return res["out"][ss.sel], res


# ------------------------------------------------------------------ tabs
st.title("📈 Backtest Lab")
st.caption(f"{symbol} · {tf} · sumber: {src.label} · waktu ditampilkan dalam WIB (UTC+7)")
t_search, t_res, t_live, t_hist, t_adv, t_help = st.tabs(
    ["🔬 Cari strategi", "📊 Hasil", "📈 Chart", "🕘 Riwayat", "🧰 Alat lanjutan", "📘 Panduan"])
ADV_TOOLS = {"scan": "🛰️ Scan banyak pasar sekaligus", "multi": "🧮 Grid · Martingale · DCA", "ml": "🤖 Machine learning",
             "wfo": "🔁 Walk-forward", "port": "🧺 Portofolio", "cmp": "💾 Simpan & bandingkan", "sig": "📡 Sinyal live"}
with t_adv:
    ADV = st.selectbox("Pilih alat", list(ADV_TOOLS), key="adv_tool", format_func=ADV_TOOLS.get,
                       help="Alat tambahan. Hanya alat yang dipilih yang dimuat, supaya aplikasi tetap ringan di HP.")

with t_live:
    nbars = st.select_slider("Jumlah candle di chart", [200, 500, 1000, 2000], value=500)

    @st.fragment(run_every=ss.refresh)
    def live():
        df = get_data(show=False)
        if df is None or len(df) == 0:
            st.warning("Belum ada data.")
            return
        tick = None
        if hasattr(src, "live_tail"):
            try:
                tail, tick = src.live_tail(symbol, tf)
                if tail is not None and tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d"):
                    add = tail[tail.index > df.index[-1]][["open", "high", "low", "close", "volume"]]
                    df = pd.concat([df, add])
            except Exception as e:
                st.caption(f"Harga live biquote tidak tersedia saat ini ({e}); chart memakai data Dukascopy saja.")
            st.caption("ℹ️ Candle yang sedang berjalan & harga live dari biquote.io (broker MT5). Bisa beda sedikit dengan arsip "
                       "Dukascopy yang dipakai untuk backtest. Itu normal.")
        last = df.iloc[-1]
        prev = df.iloc[-2]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{symbol} harga terakhir", f"{last.close:,.5g}", f"{(last.close/prev.close-1)*100:+.2f}%")
        c2.metric("Candle terakhir (WIB)", to_wib(df.index[-1:])[0].strftime("%d %b %H:%M"))
        c3.metric("Jumlah candle tersimpan", f"{len(df):,}")
        c4.metric("Diperbarui", pd.Timestamp.now(tz=TZ).strftime("%H:%M:%S WIB"))
        if tick:
            pip = src.pip_size(symbol) or 0.0001
            st.caption(f"Live biquote: bid {tick.get('bid')} · ask {tick.get('ask')} · spread {float(tick.get('spread') or 0)/pip:.1f} pip · "
                       f"umur quote {tick.get('quoteAgeSeconds')} detik · pasar {tick.get('marketState')}")
        sel = selected_strategy()
        fam = p = tr = cc = None
        if sel and sel[1]["symbol"] == symbol and sel[1]["tf"] == tf:
            o, res = sel
            dfc = df.copy()
            cc, _ = make_cache(dfc, res["cfg"], symbol, tf)
            cf, ca, _ = cost_params(float(dfc.close.iloc[-1]))
            _, tr = E.run(o["fam"], o["params"], cc, min(260, len(dfc) // 4), len(dfc), res["cfg"]["direction"], cf, ca,
                          lev=res["cfg"]["lev"])
            fam, p = o["fam"], o["params"]
            # live status: is the strategy currently in a trade?
            le, lx, se, sx, _ = E.signals(fam, p, cc)
            last_open = len(tr["ent"]) and tr["reason"][-1] == 3
            if last_open:
                d = "LONG" if tr["dir"][-1] == 1 else "SHORT"
                st.success(f"Strategi terpilih SAAT INI dalam posisi {d} sejak "
                           f"{to_wib(df.index[[tr['ent'][-1]]])[0]:%d %b %H:%M} WIB, harga masuk {tr['ep'][-1]:,.5g}, "
                           f"hasil berjalan {tr['ret'][-1]*100:+.2f}% (candle terakhir belum tutup ikut dihitung).")
            else:
                st.info("Strategi terpilih saat ini tidak punya posisi terbuka.")
            if le[-2] or se[-2]:
                st.warning("Ada sinyal baru di candle yang baru tutup.")
            st.caption(f"Garis & label dari strategi: {E.describe(fam, p, res['cfg']['direction'])}")
        else:
            st.caption("Pilih strategi di tab 'Hasil & Detail' untuk menampilkan garis indikator dan label Long/Short di chart ini.")
        sub = df.iloc[-nbars:]
        st.plotly_chart(candle_fig(sub, fam, p, tr, cc), width='stretch', key=f"live_{time.time()}")
    live()

with t_search:
    st.caption(f"Pasar **{symbol} · {tf}** · {len(ss.families)} jenis strategi · {ss.n_variants:,} variasi · "
               f"target PF ≥ {ss.min_pf:g}, drawdown ≤ {ss.max_dd:g}%. Ubah di 'Pengaturan lanjutan' kalau perlu.")
    with st.expander("⚙️ Pengaturan lanjutan (target, biaya, mode, jenis strategi, SL/TP)"):
        st.subheader("Tulis target dengan kalimat biasa (opsional)")
        c1, c2 = st.columns([4, 1])
        c1.text_input("Contoh: BTC 1h trend, profit factor 1.5+, drawdown di bawah 20%", key="prompt",
                      label_visibility="collapsed", placeholder="BTC 1h trend strategy, profit factor 1.5+, drawdown under 20%")
        c2.button("Terapkan", on_click=apply_prompt, width='stretch')
        if ss.get("prompt_msg"):
            st.caption(f"Terbaca: {ss.prompt_msg}")

        st.subheader("Pengaturan backtest")
        a, b, c = st.columns(3)
        with a:
            st.radio("Arah trade", ["both", "long", "short"], key="direction", format_func=DIR_LABEL.get, horizontal=True)
            st.slider("Data 'unseen' / out-of-sample (%)", 10, 50, key="oos_pct",
                      help="Bagian data paling baru yang TIDAK dipakai saat mencari. Strategi harus tetap bagus di sini.")
            st.slider("Jumlah periode walk-forward", 2, 8, key="folds",
                      help="Data in-sample dibagi beberapa periode; strategi harus untung di sebagian besar periode.")
            st.slider("Minimal periode yang untung (%)", 0, 100, key="min_wf")
        with b:
            st.number_input("Profit factor minimal", 1.0, 5.0, step=0.1, key="min_pf")
            st.number_input("Win rate minimal (%)", 0.0, 90.0, step=5.0, key="min_wr")
            st.number_input("Drawdown maksimal (%)", 1.0, 90.0, step=1.0, key="max_dd")
            st.number_input("Jumlah trade minimal (in-sample)", 5, 2000, step=10, key="min_trades")
            st.number_input("Profit factor minimal di data unseen", 0.5, 5.0, step=0.1, key="oos_min_pf")
        with c:
            if ss.source == "binance":
                st.selectbox("Preset biaya", list(CRYPTO_PRESETS), key="crypto_preset", on_change=_apply_preset,
                             help="Contoh tarif umum. Angka di bawah tetap bisa diubah.")
                st.number_input("Fee per transaksi (%)", 0.0, 1.0, step=0.01, key="fee")
                st.number_input("Slippage per transaksi (%)", 0.0, 1.0, step=0.01, key="slip")
            else:
                pip = src.pip_size(symbol) or 0.0001
                st.selectbox("Preset akun broker", list(FX_PRESETS), key="fx_preset", on_change=_apply_preset,
                             help="Contoh saja (bukan broker tertentu). Spread disimpan per pasar selama sesi ini dan tetap bisa diubah.")
                st.number_input(f"Spread {symbol} (pip) · 1 pip = {pip:g}", 0.0, 50.0, step=0.1, key="spread", on_change=_remember_costs)
                st.number_input("Slippage per sisi (pip)", 0.0, 20.0, step=0.1, key="fx_slip")
                st.number_input("Komisi per sisi (%)", 0.0, 0.5, step=0.001, format="%.3f", key="fx_comm")
            st.number_input("Ukuran posisi (x modal, 1 = tanpa leverage)", 0.1, 20.0, step=0.5, key="lev")
            st.checkbox("Rapor pakai DSR versi ketat (buku teks)", key="strict_dsr",
                        help="Anggap semua variasi yang dicoba benar-benar terpisah. Nilai rapor jadi lebih pelit. Bisa diubah kapan saja.")
            st.slider("Minimal parameter tetangga yang tetap untung (%)", 0, 100, key="min_stab",
                      help="Kalau EMA 11/36 bagus tapi 12/35 hancur, berarti cuma kebetulan.")
            if ss.source == "binance":
                st.checkbox("Hitung funding rate (seperti trading perpetual futures)", key="use_funding",
                            help="Biaya/pendapatan tiap 8 jam dari riwayat funding Binance. Long bayar kalau funding positif, short menerima.")
            else:
                st.checkbox("Hitung swap (biaya menginap)", key="use_swap",
                            help="Perkiraan swap broker per tahun, dihitung sesuai lama posisi dibuka (termasuk akhir pekan).")
                if ss.use_swap:
                    s1, s2 = st.columns(2)
                    s1.number_input("Swap Long (%/tahun, + = bayar)", -20.0, 20.0, step=0.25, key="swap_long", on_change=_remember_costs)
                    s2.number_input("Swap Short (%/tahun, + = bayar)", -20.0, 20.0, step=0.25, key="swap_short", on_change=_remember_costs)
                    st.caption("Lihat angka swap di spesifikasi kontrak brokermu. Contoh: kalau broker memotong 2% setahun untuk Long, isi 2.")
        st.multiselect("Jenis strategi yang dicari", list(E.FAMILIES), key="families", format_func=lambda k: E.FAMILIES[k]["name"])
        bad_tf = [E.FAMILIES[f]["name"] for f in ss.families if E.FAMILIES[f]["intraday"] and TF_MINUTES.get(tf, 60) > 60]
        if bad_tf:
            st.caption(f"Dilewati di timeframe ini (butuh timeframe ≤ 1 jam): {', '.join(bad_tf)}.")
        st.caption("Exit ikut dicoba otomatis: 14 jenis stop loss, 10 jenis take profit, TP1/TP2 + breakeven, 8 jenis trailing, "
                   "time stop, tutup akhir hari / akhir pekan, jeda setelah rugi, batas harian. Filter ADX juga ikut dicoba.")
        exit_pick = V7.render_exit_pick()
        m1, m2 = st.columns(2)
        m1.radio("Mode penilaian", ["ketat", "vela"], key="mode", horizontal=True,
                 format_func=lambda k: {"ketat": "Ketat (disarankan)", "vela": "Mode Vela (longgar)"}[k],
                 help="Ketat: harus lolos data unseen + walk-forward + stabilitas parameter + tidak overfit. "
                      "Vela: cukup lolos target di data lama dan data unseen.")
        m2.radio("Cara mencari", ["random", "evo"], key="algo", horizontal=True,
                 format_func=lambda k: {"random": "Acak", "evo": "Evolusi (lebih cepat ketemu)"}[k],
                 help="Evolusi: setelah 25% percobaan acak, variasi yang bagus di data lama diubah sedikit-sedikit. "
                      "Data unseen tidak pernah dipakai untuk memilih. Semua percobaan tetap dihitung di Deflated Sharpe.")
        st.checkbox("Kombinasi dengan filter konfirmasi (sesi, hari, tren timeframe besar, volatilitas, RSI, volume)", key="combo",
                    help="Sinyal masuk hanya diambil kalau filter konfirmasi juga setuju. Lebih banyak kombinasi = peluang kebetulan juga naik.")
        if ss.mode == "vela":
            st.warning("Mode Vela: hasil 'lolos' lebih mungkin cuma kebetulan (tanpa cek walk-forward & stabilitas). "
                       "Lihat kolom DSR % dan nilai (grade) sebelum percaya.")
        d1, d2 = st.columns(2)
        d1.select_slider("Jumlah variasi yang dicoba", [500, 1000, 2000, 3000, 5000, 10000, 20000], key="n_variants")
        d2.select_slider("Batas waktu (detik)", [30, 60, 120, 300, 600], key="time_budget")

        if deriv_on():
            st.caption("📊 Data derivatif aktif: 3 jenis strategi baru (funding ekstrem, open interest vs harga, order flow taker) dan "
                       "3 filter tambahan (funding, order flow, open interest) ikut dicari. Strategi yang memakainya tidak bisa diekspor ke Pine. "
                       "Open interest & long/short baru ada sejak ±Sep 2020, funding sejak Sep 2019; sebelum itu filter/strategi ini tidak memberi sinyal.")
    _force = False
    _jid0 = ss.get("job_search")
    _st0 = J.status(_jid0)
    if _st0 and _st0["state"] in ("antri", "jalan"):
        st.button("⏳ Pencarian sedang berjalan...", type="primary", width='stretch', disabled=True)
        _go = False
    else:
        _go = st.button("🚀 Mulai cari strategi", type="primary", width='stretch')
        if ss.get("res") and ss.get("res_job") == _jid0 and _st0 and _st0["state"] == "selesai":
            _force = st.button("🔁 Cari ulang dengan variasi acak baru", width='stretch',
                               help="Setelan sama tapi dicoba variasi lain. Tombol 'Mulai' dengan setelan sama memakai hasil yang sudah ada.")
    if _go or _force:
        df = get_data()
        if df is not None and ss.families:
            dfc = closed_only(df, tf)
            if len(dfc) < 1500:
                st.error(f"Data terlalu sedikit ({len(dfc)} candle). Pilih timeframe lebih besar atau tanggal mulai lebih awal.")
            else:
                cf, ca, pine_cost = cost_params(float(dfc.close.mean()))
                cfg = dict(families=ss.families, n_variants=ss.n_variants, time_budget=ss.time_budget,
                           direction=ss.direction, oos_pct=ss.oos_pct, folds=ss.folds, cost_frac=cf, cost_abs=ca,
                           min_pf=ss.min_pf, min_wr=ss.min_wr / 100, max_dd=ss.max_dd / 100, min_trades=ss.min_trades,
                           min_wf=ss.min_wf / 100, min_stab=ss.min_stab / 100, oos_min_pf=ss.oos_min_pf,
                           lev=ss.lev, pine_cost=pine_cost, carry=carry_cfg(),
                           mode=ss.mode, algo=ss.algo, combo=bool(ss.combo), tf_min=TF_MINUTES.get(tf, 60), deriv=deriv_on(),
                           exit_pick=exit_pick)
                _key = dict(cfg, symbol=symbol, tf=tf, source=ss.source, start=str(start_date), n=len(dfc),
                            nonce=(time.time() if _force else 0))
                cfg["seed"] = J.seed_of(_key)
                _jid = J.job_id("search", _key)
                if J.status(_jid) is None:
                    cc0, carry_info = make_cache(dfc, cfg, symbol, tf)      # needs the page (cached funding data) -> before the thread
                    _dinfo = df.attrs.get("deriv_info")
                    _sym, _tf, _src, _start = symbol, tf, ss.source, str(start_date)

                    def _work(prog, dfc=dfc, cfg=cfg, cc0=cc0, carry_info=carry_info):
                        out, meta, cc1 = E.search(dfc, cfg, lambda f, t_, n_: prog(f * 0.95, f"Diuji {t_:,} variasi · lolos target in-sample: {n_}"),
                                                  cc=cc0)
                        prog(0.96, "Menghitung Deflated Sharpe dan peluang overfit (PBO)...")
                        r_ = dict(out=out, meta=meta, cfg=cfg, symbol=_sym, tf=_tf, source=_src, id=_jid, start=_start,
                                  carry_info=carry_info, deriv_info=_dinfo)
                        U.after_search(r_, cc1)
                        return r_
                    J.start(_jid, _work, label=f"{symbol} {tf} · {len(cfg['families'])} jenis · {cfg['n_variants']:,} variasi",
                            kind="search", extra_meta=dict(symbol=symbol, tf=tf, source=ss.source))
                ss.job_search = _jid
                _save_session()
    _jid = ss.get("job_search")
    _st = J.status(_jid)
    if _st and _st["state"] in ("antri", "jalan"):
        job_panel(_jid, "Pencarian strategi")
    elif _st and _st["state"] == "gagal":
        st.error("Pencarian gagal: " + (_st.get("error") or "").split("\n")[0])
    elif _st and _st["state"] == "selesai" and ss.get("res_job") != _jid:
        _r = J.result(_jid)
        if _r:
            ss.res, ss.res_job, ss.sel = _r, _jid, None
    if ss.get("res") and ss.get("res_job") == _jid and _st and _st["state"] == "selesai":
        _r = ss.res
        npass = sum(o["passed"] for o in _r["out"])
        tests_txt = "unseen, walk-forward, stabilitas" if _r["cfg"]["mode"] == "ketat" else "data unseen saja (mode Vela)"
        st.success(f"Selesai ({_r['symbol']} {_r['tf']}): {_r['meta']['tested']:,} variasi diuji. {_r['meta']['is_pass']} lolos target "
                   f"in-sample, {npass} lolos tes ({tests_txt}). Buka tab '📊 Hasil'.")
        if npass == 0:
            st.info("Tidak ada yang lolos semua tes. Ini jujur: berarti target terlalu ketat atau pasar ini sulit. "
                    "Coba longgarkan target, ganti timeframe, atau tambah jenis strategi.")

def _open_scan(key):
    r = ss.scan_results[key]
    ss.res = r
    ss.res_job = None
    ss.res_scan = [ss.get("scan_job"), key] if ss.get("scan_job") else None
    ss.sel = None
    ss.source, ss.symbol, ss.tf = r["source"], r["symbol"], r["tf"]
    ss.open_msg = f"{r['symbol']} {r['tf']} dibuka. Lihat tab 'Hasil & Detail'."


if ADV == "scan":
    with t_adv:
        st.markdown("Cari strategi di **banyak pasar & timeframe sekaligus** dengan pengaturan yang sama seperti tab 'Cari Strategi' "
                    "(target, mode, cara mencari, jenis strategi, biaya). Hasilnya diurutkan dalam satu tabel.")
        sc1, sc2 = st.columns(2)
        scan_src = sc1.selectbox("Sumber", ["binance", "dukascopy"], format_func=lambda k: SOURCES[k].label, key="scan_src")
        all_syms = SOURCES[scan_src].symbols()
        dflt = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] if scan_src == "binance" else ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "US30", "NAS100"]
        scan_syms = sc1.multiselect("Pasar", all_syms, default=[x for x in dflt if x in all_syms], key=f"scan_syms_{scan_src}")
        scan_tfs = sc2.multiselect("Timeframe", ["5m", "15m", "30m", "1h", "4h", "1d"], default=["1h", "4h"], key="scan_tfs")
        sc3, sc4 = st.columns(2)
        scan_var = sc3.select_slider("Variasi per pasar", [300, 500, 1000, 2000, 3000], value=1000, key="scan_var")
        scan_time = sc4.select_slider("Batas waktu total (detik)", [60, 120, 300, 600, 900], value=300, key="scan_time")
        n_jobs = len(scan_syms) * len(scan_tfs)
        st.caption(f"{n_jobs} kombinasi pasar × timeframe. Riwayat yang dipakai: 5m 180 hari, 15m 1 tahun, 30m 2 tahun, 1h 4 tahun, "
                   "4h 8 tahun, 1d semua. Unduhan pertama tiap pasar butuh waktu; di server gratis pilih sedikit pasar dulu.")
        if ss.mode == "vela":
            st.warning("Mode Vela aktif: makin banyak pasar dicoba, makin besar peluang ada yang 'lolos' karena kebetulan.")
        _sj = ss.get("job_scan")
        _sst = J.status(_sj)
        _srun = bool(_sst and _sst["state"] in ("antri", "jalan"))
        if st.button("⏳ Scan sedang berjalan..." if _srun else "🛰️ Mulai scan", type="primary", disabled=n_jobs == 0 or _srun):
            base = dict(families=ss.families or list(E.FAMILIES), n_variants=int(scan_var), direction=ss.direction, oos_pct=ss.oos_pct,
                        folds=ss.folds, min_pf=ss.min_pf, min_wr=ss.min_wr / 100, max_dd=ss.max_dd / 100, min_trades=ss.min_trades,
                        min_wf=ss.min_wf / 100, min_stab=ss.min_stab / 100, oos_min_pf=ss.oos_min_pf, lev=ss.lev,
                        carry=None, mode=ss.mode, algo=ss.algo, combo=bool(ss.combo), pbo_trials=400)
            jobs = [(scan_src, s_, t_) for s_ in scan_syms for t_ in scan_tfs]
            _key = dict(base, jobs=jobs, budget=float(scan_time), day=str(dt.date.today()))
            base["seed"] = J.seed_of(_key)
            # cost settings frozen now (the background thread cannot read the page settings)
            _cs = dict(fee=ss.fee, slip=ss.slip, fx_comm=ss.fx_comm, fx_slip=ss.fx_slip,
                       spreads={s_: (ss.spread if s_ == symbol else spread_for(s_)) for s_ in scan_syms})

            def _cost(s0, s1, px, _cs=_cs):
                if s0 == "binance":
                    return (_cs["fee"] + _cs["slip"]) / 100, 0.0
                pip = SOURCES[s0].pip_size(s1) or 0.0001
                return _cs["fx_comm"] / 100, (_cs["spreads"].get(s1, 1.5) / 2 + _cs["fx_slip"]) * pip

            def _work(prog, jobs=jobs, base=base, budget=float(scan_time)):
                from core.scan import scan_markets
                return scan_markets(jobs, lambda s0, s1, s2, st_: SOURCES[s0].get(s1, s2, start=st_, refresh=True), base, _cost, budget,
                                    progress=lambda f, txt: prog(f, txt))
            _sj = J.job_id("scan", _key)
            J.start(_sj, _work, label=f"Scan {len(jobs)} pasar×TF ({scan_src})", kind="scan")
            ss.job_scan = _sj
            _save_session()
            _sst = J.status(_sj)
        if _sst and _sst["state"] in ("antri", "jalan"):
            job_panel(_sj, "Scan pasar")
        elif _sst and _sst["state"] == "gagal":
            st.error("Scan gagal: " + (_sst.get("error") or "").split("\n")[0])
        elif _sst and _sst["state"] == "selesai" and ss.get("scan_job") != _sj:
            _r = J.result(_sj)
            if _r:
                ss.scan_rows, ss.scan_results = _r
                ss.scan_job = _sj
        if ss.get("scan_rows"):
            tbl = pd.DataFrame(ss.scan_rows)
            show = [c for c in ["Pasar", "TF", "Status", "Lolos tes", "Lolos IS", "Diuji", "Strategi terbaik", "IS PF", "OOS PF", "OOS Net %",
                                "OOS DD %", "OOS Trades", "DSR %", "PBO %", "Candle", "Detik"] if c in tbl]
            st.dataframe(tbl[show], hide_index=True, width='stretch',
                         column_config={k: st.column_config.NumberColumn(format="%.2f") for k in ["IS PF", "OOS PF", "OOS Net %", "OOS DD %", "DSR %", "PBO %"]})
            st.caption("Urutan: yang punya strategi lolos dulu, lalu DSR (peluang bukan kebetulan) tertinggi, lalu PF data unseen. "
                       "'Strategi terbaik' = yang DSR-nya paling tinggi di antara yang lolos. Ini hasil uji data lama, bukan saran.")
            st.download_button("⬇️ Unduh tabel (CSV)", tbl[show].to_csv(index=False).encode("utf-8"), "scan_pasar.csv", "text/csv")
            V.render_scan_matrix(ss.scan_rows)
            keys = [r["key"] for r in ss.scan_rows if r.get("key")]
            if keys:
                o1, o2 = st.columns([3, 1])
                pick = o1.selectbox("Buka hasil pasar", keys, format_func=lambda k: " ".join(k.split("|")[1:]), key="scan_pick")
                o2.button("Buka di 'Hasil & Detail'", on_click=_open_scan, args=(pick,), width='stretch')
                if ss.get("open_msg"):
                    st.success(ss.open_msg)

with t_res:
    res = ss.get("res")
    if not res:
        st.info("Belum ada hasil. Jalankan pencarian di tab 'Cari Strategi'.")
    else:
        meta, cfg = res["meta"], res["cfg"]
        st.caption(f"{res['symbol']} {res['tf']} · in-sample {to_wib([meta['start_time']])[0]:%d %b %Y} – "
                   f"{to_wib([meta['split_time']])[0]:%d %b %Y} · data unseen sejak {to_wib([meta['split_time']])[0]:%d %b %Y %H:%M} WIB · "
                   f"{meta['tested']:,} variasi dalam {meta['seconds']:.1f} detik")
        if res["cfg"].get("deriv"):
            if not deriv_on():
                st.warning("Hasil ini dicari dengan data derivatif. Nyalakan lagi '📊 Pakai data derivatif' di sidebar, "
                           "kalau tidak strategi yang memakai funding/OI/order flow tidak akan punya trade.")
            di = res.get("deriv_info") or {}
            with st.expander("📊 Data derivatif yang dipakai"):
                if "error" in di:
                    st.error(f"Gagal memuat data derivatif: {di['error']}")
                for k_, lab_ in (("taker", "Volume taker (order flow)"), ("fund", "Funding rate"), ("oi", "Open interest & rasio long/short")):
                    v_ = di.get(k_)
                    if v_:
                        st.markdown(f"- **{lab_}**: " + (f"sejak {pd.Timestamp(v_['frm']).tz_convert(TZ):%d %b %Y}, " if v_.get("frm") is not None else "") +
                                    f"ada di {v_.get('coverage', 0) * 100:.0f}% candle · sumber: {v_.get('note', '-')}")
                st.caption("Nilai tiap candle = data terakhir yang sudah ada SEBELUM candle tutup (tidak melihat masa depan). "
                           "Data likuidasi tidak dipakai karena tidak ada sumber riwayat gratis.")
        rows = []
        for i, o in enumerate(res["out"]):
            im, om = o["is_m"], o["oos_m"]
            status = ("✅ Lolos semua" if cfg.get("mode", "ketat") == "ketat" else "✅ Lolos (mode Vela)") if o["passed"] else ("⚠️ Overfit" if o["overfit"] else ("➖ Hampir (gagal target IS)" if not o["is_pass"] else "❌ Gagal tes"))
            rows.append({"idx": i, "Status": status, "Nama": f"{E.FAMILIES[o['fam']]['name']} #{i+1}",
                         "IS Net %": im["net"] * 100, "IS PF": im["pf"], "IS WR %": im["winrate"] * 100,
                         "IS DD %": im["maxdd"] * 100, "IS Trades": im["trades"],
                         "OOS Net %": om["net"] * 100, "OOS PF": om["pf"], "OOS WR %": om["winrate"] * 100,
                         "OOS DD %": om["maxdd"] * 100, "OOS Trades": om["trades"],
                         "Walk-fwd %": o["wf"] * 100, "Stabil %": o["stab"] * 100,
                         "DSR %": (o.get("dsr") if o.get("dsr") == o.get("dsr") and o.get("dsr") is not None else 0) * 100,
                         "Deskripsi": E.describe(o["fam"], o["params"], cfg["direction"])})
        tbl = pd.DataFrame(rows)
        only = st.checkbox("Tampilkan hanya yang lolos semua tes", value=False)
        if only:
            tbl = tbl[tbl.Status.str.startswith("✅")]
        tbl = tbl.sort_values(["Status", "IS Net %"], ascending=[True, False], key=lambda s: s.map({"✅ Lolos semua": 0, "✅ Lolos (mode Vela)": 0}).fillna(1) if s.name == "Status" else s)
        st.caption("Urutan berdasarkan hasil in-sample (data unseen hanya dipakai sebagai ujian, bukan untuk memilih). Klik satu baris untuk melihat detail. "
                   "DSR % = peluang hasil in-sample bukan kebetulan setelah dikoreksi jumlah variasi yang dicoba.")
        pbo = res.get("search_tests", {}).get("pbo")
        if pbo:
            lvl = "rendah (bagus)" if pbo["pbo"] < 0.2 else "sedang" if pbo["pbo"] <= 0.5 else "tinggi (hati-hati)"
            st.caption(f"Peluang overfit seluruh pencarian ini (PBO, metode CSCV): **{pbo['pbo'] * 100:.0f}%**, {lvl}.")
        ev = st.dataframe(tbl.drop(columns=["idx"]), hide_index=True, width='stretch', on_select="rerun",
                          selection_mode="single-row", height=330,
                          column_config={k: st.column_config.NumberColumn(format="%.2f") for k in
                                         ["IS Net %", "IS PF", "IS WR %", "IS DD %", "OOS Net %", "OOS PF", "OOS WR %", "OOS DD %", "Walk-fwd %", "Stabil %", "DSR %"]})
        if ev.selection.rows:
            ss.sel = int(tbl.iloc[ev.selection.rows[0]]["idx"])
        if ss.get("sel") is None and len(tbl):
            ss.sel = int(tbl.iloc[0]["idx"])
        if ss.get("sel") is not None:
            o = res["out"][ss.sel]
            st.markdown(f"### {E.FAMILIES[o['fam']]['name']} #{ss.sel + 1}")
            _om, _im = o["oos_m"], o["is_m"]
            _g = U.tests_for(f"{res.get('id', '0')}_{ss.sel}").get("grade")
            with st.container(border=True):
                _k = st.columns(5)
                _k[0].metric("Rapor", _g[1] if _g and _g[0] is not None else "-", help="Nilai ketahanan A-E (lihat rincian di bawah).")
                _k[1].metric("Untung (unseen)", f"{_om['net'] * 100:+.1f}%")
                _k[2].metric("Drawdown (unseen)", f"{_om['maxdd'] * 100:.1f}%")
                _k[3].metric("Profit factor (unseen)", f"{_om['pf']:.2f}")
                _k[4].metric("Win rate (unseen)", f"{_om['winrate'] * 100:.0f}%")
                st.caption(f"'Unseen' = data terbaru yang tidak dipakai saat mencari ({_om['trades']} trade). "
                           f"Data lama (in-sample): {_im['net'] * 100:+.1f}%, PF {_im['pf']:.2f}, {_im['trades']} trade. "
                           "Hasil uji data lama, bukan saran.")
            st.write(E.describe(o["fam"], o["params"], cfg["direction"]))
            V7.no_sl_warning(o["params"])
            if o["overfit"]:
                st.error("⚠️ Tanda overfitting: hasil di data unseen jauh lebih buruk dari data in-sample. Jangan dipakai.")
            # recompute on the latest data (new candles since the search are also 'unseen')
            df = get_data(show=False)
            if df is None:
                st.stop()
            dfc = closed_only(df, res["tf"]) if res["symbol"] == symbol and res["tf"] == tf else None
            if dfc is None:
                st.warning(f"Pilih lagi {res['symbol']} {res['tf']} di sidebar untuk melihat detail strategi ini.")
                st.stop()
            cc, carry_info = shared_cache(dfc, cfg, res["symbol"], res["tf"])
            n = len(dfc)
            split = int(np.searchsorted(dfc.index.values, np.datetime64(meta["split_time"].tz_convert(None))))
            is_start = min(260, split // 4)
            cf, ca, lev = cfg["cost_frac"], cfg["cost_abs"], cfg["lev"]
            im, _ = E.run(o["fam"], o["params"], cc, is_start, split, cfg["direction"], cf, ca, lev=lev)
            om, otr = E.run(o["fam"], o["params"], cc, split, n, cfg["direction"], cf, ca, lev=lev)
            fm, ftr = E.run(o["fam"], o["params"], cc, is_start, n, cfg["direction"], cf, ca, want_equity=True, lev=lev)
            cmp = pd.DataFrame({"In-sample (dipakai mencari)": im, "Unseen / OOS (ujian)": om, "Seluruh periode": fm}).T
            cmp = cmp.rename(columns={"trades": "Trades", "net": "Net P&L %", "winrate": "Win rate %", "pf": "Profit factor",
                                      "maxdd": "Max DD %", "avg": "Rata2/trade %", "ret_dd": "Return/DD"})
            for k in ["Net P&L %", "Win rate %", "Max DD %", "Rata2/trade %"]:
                cmp[k] = cmp[k] * 100
            st.dataframe(cmp.style.format("{:.2f}"), width='stretch')
            stab_txt = (f"Stabilitas parameter: {o['stab']*100:.0f}% tetangga untung (median PF tetangga {o['stab_pf']:.2f})"
                        if o["stab"] == o["stab"] else "Stabilitas parameter: tidak dicek (mode Vela)")
            st.caption(f"Walk-forward: untung di {o['wf']*100:.0f}% periode · {stab_txt} · biaya dihitung · posisi {lev:g}x modal · "
                       "OOS di atas sudah termasuk candle baru sejak pencarian.")

            ctx = dict(o=o, res=res, cfg=cfg, cc=cc, dfc=dfc, n=n, split=split, is_start=is_start, skey=f"{res.get('id', '0')}_{ss.sel}",
                       carry_info=carry_info or res.get("carry_info"), symbols=SOURCES[res["source"]].symbols(),
                       cc_nocarry=E.Cache(dfc) if cfg.get("carry") else cc,
                       load_symbol_raw=lambda s_: closed_only(load(res["source"], s_, res["tf"], start_date, bool(cfg.get("deriv")) and E.uses_deriv(o["fam"], o["params"])), res["tf"]) if bool(cfg.get("deriv")) else closed_only(SOURCES[res["source"]].get(s_, res["tf"], start=pd.Timestamp(start_date, tz="UTC"),
                                                                                      refresh=True), res["tf"]),
                       make_cache=lambda d_, s_: make_cache(d_, cfg if res["source"] == "binance" else None, s_, res["tf"])[0],
                       cost_for=lambda s_, px: cost_for(res["source"], s_, px))
            U.render_report(ctx)
            d_chart, d_perf, d_ana, d_trades, d_sim, d_size, d_reg, d_meta, d_rob, d_bask, d_real, d_exp = st.tabs(
                ["Chart", "Performance", "📐 Analitik", "Trades", "Simulation (Monte Carlo)", "💰 Ukuran posisi & Prop firm", "🌦️ Regime",
                 "🤖 Meta-label", "🧪 Uji ketahanan", "🌍 Multi-aset", "🎯 Realisme", "Export & Simpan"])
            with d_ana:
                V.render_analytics(ctx, ftr)
                V.render_sensitivity(ctx)
            with d_size:
                V.render_sizing(ctx)
            with d_reg:
                V.render_regime(ctx)
            with d_meta:
                V.render_meta(ctx)
            with d_rob:
                U.render_robust(ctx)
                V7.render_exit_compare(ctx)
            with d_bask:
                U.render_basket(ctx)
            with d_real:
                U.render_realism(ctx)
            with d_chart:
                rng = st.select_slider("Candle terakhir yang ditampilkan", [300, 1000, 2000, 4000], value=1000)
                sub = dfc.iloc[-rng:]
                st.plotly_chart(candle_fig(sub, o["fam"], o["params"], ftr, cc, split_time=meta["split_time"]), width='stretch')
            with d_perf:
                eqs = pd.Series(ftr["equity"] * 10000, index=dfc.index[is_start:n])
                fig = go.Figure(go.Scatter(x=to_wib(eqs.index), y=eqs.values, name="Equity", line=dict(color="#29b6f6")))
                fig.add_vrect(x0=meta["split_time"].tz_convert(TZ), x1=to_wib(eqs.index[-1:])[0], fillcolor="orange", opacity=0.1,
                              line_width=0, annotation_text="unseen (OOS)")
                fig.update_layout(title="Equity curve (modal awal 10.000)", template="plotly_dark", height=380, margin=dict(t=40, b=10))
                st.plotly_chart(fig, width='stretch')
                per = st.radio("P&L per", ["Harian", "Mingguan", "Bulanan"], horizontal=True, index=1)
                rule = {"Harian": "1D", "Mingguan": "W-MON", "Bulanan": "ME"}[per]
                eqw = eqs.copy(); eqw.index = to_wib(eqw.index)
                pl = eqw.resample(rule).last().ffill().pct_change().dropna() * 100
                fig2 = go.Figure(go.Bar(x=pl.index, y=pl.values, marker_color=np.where(pl.values >= 0, "#26a69a", "#ef5350")))
                fig2.update_layout(title=f"P&L {per.lower()} (%)", template="plotly_dark", height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig2, width='stretch')
            with d_trades:
                idx = dfc.index
                log = pd.DataFrame({
                    "Masuk (WIB)": to_wib(idx[ftr["ent"]]).strftime("%Y-%m-%d %H:%M"),
                    "Keluar (WIB)": to_wib(idx[ftr["ex"]]).strftime("%Y-%m-%d %H:%M"),
                    "Arah": np.where(ftr["dir"] == 1, "Long", "Short"),
                    "Harga masuk": ftr["ep"], "Harga keluar": ftr["xp"],
                    "P&L %": ftr["ret"] * 100,
                    "SL awal": ftr["sl0"], "SL akhir": ftr["sl"], "TP": ftr["tp"],
                    "TP1 kena": [f"{v:.6g}" if b >= 0 else "-" for b, v in zip(ftr["p1b"], ftr["p1p"])],
                    "TP2 kena": [f"{v:.6g}" if b >= 0 else "-" for b, v in zip(ftr["p2b"], ftr["p2p"])],
                    "Alasan keluar": np.array([E.X.REASONS[i] for i in range(7)])[ftr["reason"]],
                    "Periode": np.where(ftr["ent"] >= split, "Unseen", "In-sample")})
                a1, a2, a3, a4 = st.columns(4)
                r = ftr["ret"]
                a1.metric("Trade", len(r)); a2.metric("Rata-rata menang", f"{r[r>0].mean()*100 if (r>0).any() else 0:.2f}%")
                a3.metric("Rata-rata kalah", f"{r[r<0].mean()*100 if (r<0).any() else 0:.2f}%")
                lens = (ftr["ex"] - ftr["ent"]) * TF_MINUTES[tf] / 60
                a4.metric("Lama trade rata-rata", f"{lens.mean() if len(lens) else 0:.1f} jam")
                st.dataframe(log.iloc[::-1], hide_index=True, width='stretch', height=400,
                             column_config={"P&L %": st.column_config.NumberColumn(format="%.2f")})
                st.download_button("⬇️ Download trade log (CSV)", log.to_csv(index=False).encode(), f"trades_{symbol}_{tf}.csv")
            with d_sim:
                mcol1, mcol2 = st.columns(2)
                runs = mcol1.select_slider("Jumlah simulasi", [200, 500, 1000, 2000, 5000], value=1000)
                method = mcol2.radio("Metode", ["bootstrap", "shuffle", "skip"], horizontal=True,
                                     format_func={"bootstrap": "Bootstrap (ambil acak dengan pengembalian)", "shuffle": "Acak urutan trade",
                                                  "skip": "Lewatkan 10% trade secara acak"}.get)
                mc = E.monte_carlo(ftr["ret"], runs, method)
                if mc is None:
                    st.info("Trade terlalu sedikit untuk simulasi.")
                else:
                    fig = go.Figure()
                    show = mc["paths"][:: max(1, runs // 100)]
                    for pth in show:
                        fig.add_trace(go.Scatter(y=pth * 10000, mode="lines", line=dict(width=0.6, color="rgba(120,144,156,0.25)"), showlegend=False))
                    for q, col in ((5, "#ef5350"), (50, "#ffffff"), (95, "#26a69a")):
                        fig.add_trace(go.Scatter(y=mc["bands"][q] * 10000, name=f"Persentil {q}", line=dict(width=2, color=col)))
                    fig.add_trace(go.Scatter(y=np.r_[1, np.cumprod(1 + ftr["ret"])] * 10000, name="Hasil asli", line=dict(width=2, color="#ffd600", dash="dot")))
                    fig.update_layout(title=f"{runs} jalur Monte Carlo (sumbu x = urutan trade)", template="plotly_dark", height=420, margin=dict(t=40, b=10))
                    st.plotly_chart(fig, width='stretch')
                    h = go.Figure(go.Histogram(x=mc["dds"] * 100, nbinsx=40, marker_color="#ef5350"))
                    h.update_layout(title="Sebaran max drawdown (%)", template="plotly_dark", height=300, margin=dict(t=40, b=10))
                    st.plotly_chart(h, width='stretch')
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("DD median", f"{np.median(mc['dds'])*100:.1f}%")
                    b2.metric("DD terburuk 5%", f"{np.percentile(mc['dds'], 95)*100:.1f}%")
                    b3.metric(f"Peluang DD > {cfg['max_dd']*100:.0f}%", f"{(mc['dds'] > cfg['max_dd']).mean()*100:.1f}%")
                    b4.metric("Peluang rugi di akhir", f"{(mc['finals'] < 0).mean()*100:.1f}%")
            with d_exp:
                note = (f"Hasil uji di aplikasi: in-sample PF {im['pf']:.2f}, DD {im['maxdd']*100:.1f}%; unseen PF {om['pf']:.2f}, "
                        f"DD {om['maxdd']*100:.1f}%, {om['trades']} trade.")
                tv_sym = res["symbol"].split(" ")[0]
                tt = U.tests_for(ctx["skey"])
                g = tt.get("grade", (None, "-"))
                trows = [("Aset", f"{res['symbol']} {res['tf']}"), ("IS PF / DD", f"{im['pf']:.2f} / {im['maxdd']*100:.1f}%"),
                         ("Unseen PF / DD", f"{om['pf']:.2f} / {om['maxdd']*100:.1f}%"), ("Unseen trades", str(om["trades"])),
                         ("DSR", U.pct(tt.get("dsr", {}).get("dsr"), 0)),
                         ("Rapor", f"{g[1]} ({g[0]:.0f}/100)" if g[0] is not None else "-")]
                try:
                    strat, indi = to_pine(o["fam"], o["params"], cfg["direction"], f"{E.FAMILIES[o['fam']]['name']} #{ss.sel+1}",
                                          tv_sym, res["tf"], cfg["pine_cost"], lev, note, trows,
                                          pip=E.X.pip_size(float(dfc.close.iloc[-1])), tf_min=TF_MINUTES.get(res["tf"], 60))
                except NoPine as e:
                    strat = indi = None
                    st.warning(str(e) + " Simpan sebagai JSON untuk dipakai di tab 'Sinyal Live'.")
                if strat:
                  st.markdown("**Cara pakai:** buka TradingView → Pine Editor → tempel kode → *Add to chart*. "
                            "Versi **indicator** menghitung posisi sendiri dan **sama persis** dengan aplikasi (label Long/Short/TP1/SL/TP/Exit + alert). "
                            "Versi **strategy** memakai mesin order TradingView: kalau stop loss dan take profit tersentuh di candle yang sama, "
                            "TradingView menebak urutannya sendiri (aplikasi menganggap stop loss duluan), jadi angkanya bisa sedikit beda. "
                            "Jam sesi, hari, dan tren timeframe besar dihitung dari waktu UTC candle.")
                  e1, e2 = st.columns(2)
                  e1.download_button("⬇️ Pine strategy (.pine)", strat, f"strategy_{tv_sym}_{res['tf']}.pine", width='stretch')
                  e2.download_button("⬇️ Pine indicator (.pine)", indi, f"indicator_{tv_sym}_{res['tf']}.pine", width='stretch')
                sj = U.strategy_json(o, res, tt)
                j1, j2 = st.columns(2)
                j1.download_button("💾 Simpan strategi (JSON)", sj, f"strategy_{tv_sym}_{res['tf']}_{o['fam']}_{ss.sel+1}.json", width='stretch',
                                   help="Simpan file ini di HP/laptop. Bisa di-upload lagi di tab 'Simpan & Bandingkan'.")
                if j2.button("➕ Tambah ke perbandingan", width='stretch'):
                    ss.setdefault("compare", {})[f"{tv_sym} {res['tf']} {E.FAMILIES[o['fam']]['name']} #{ss.sel+1}"] = json.loads(sj)
                    st.success("Ditambahkan. Buka tab 'Simpan & Bandingkan'.")
                st.caption("Pine sekarang juga punya tabel statistik di pojok chart dan alert untuk sinyal, stop loss, take profit dan posisi baru.")
                st.markdown("**📄 Laporan lengkap (HTML)**: chart, metrik, rapor dan kode Pine dalam satu file. Buka di browser HP/laptop.")
                rkey = f"rep_{ctx['skey']}_{n}"
                if st.button("Buat laporan HTML", key="mk_rep"):
                    ss["report"] = (rkey, V.report_html(ctx, ftr, im, om, fm, g, strat, f"{E.FAMILIES[o['fam']]['name']} #{ss.sel+1}"))
                if ss.get("report") and ss["report"][0] == rkey:
                    st.download_button("⬇️ Download laporan (.html)", ss["report"][1], f"laporan_{tv_sym}_{res['tf']}_{ss.sel+1}.html",
                                       mime="text/html", width='stretch')
                if strat:
                    with st.expander("Lihat kode Pine strategy"):
                        st.code(strat, language="javascript")
                    with st.expander("Lihat kode Pine indicator"):
                        st.code(indi, language="javascript")

def _res_data():
    res = ss.get("res")
    if not res:
        st.info("Jalankan pencarian di tab 'Cari Strategi' dulu.")
        return None
    if res["symbol"] != symbol or res["tf"] != tf:
        st.warning(f"Pilih lagi {res['symbol']} {res['tf']} di sidebar untuk halaman ini.")
        return None
    df = get_data(show=False)
    if df is None:
        return None
    dfc = closed_only(df, res["tf"])
    cc, _ = shared_cache(dfc, res["cfg"], res["symbol"], res["tf"])
    return res, dfc, cc


if ADV == "multi":
    with t_adv:
        if st.toggle("Buka alat grid / martingale / DCA / hedging / pairs", key="mp_on",
                     help="Dimuat hanya kalau dibuka supaya aplikasi tetap ringan."):
            _dfm = get_data(show=False)
            if _dfm is not None and len(_dfm) > 600:
                _dfm = closed_only(_dfm, tf)
                _ccm, _ = shared_cache(_dfm, dict(carry=carry_cfg()), symbol, tf)
                _cf, _ca, _pc = cost_params(float(_dfm.close.iloc[-1]))
                _r = ss.get("res")
                _rs = None
                if _r and ss.get("sel") is not None and _r["symbol"] == symbol and _r["tf"] == tf and ss.sel < len(_r["out"]):
                    _rs = (_r["out"][ss.sel]["fam"], _r["out"][ss.sel]["params"])
                V6.render_multi(dict(
                    df=_dfm, cc=_ccm, symbol=symbol, tf=tf, source=ss.source, cf=_cf, ca=_ca, pine_cost=_pc, oos_pct=ss.oos_pct,
                    max_dd=ss.max_dd / 100, min_pf=ss.oos_min_pf, direction=ss.direction, carry=carry_cfg(),
                    basket=[s_ for s_ in (U.CRYPTO_BASKET if ss.source == "binance" else U.FX_BASKET) if s_ != symbol][:4],
                    symbols=list(dict.fromkeys(src.symbols() + [symbol])), res_strategy=_rs,
                    load_other=lambda s_: closed_only(SOURCES[ss.source].get(s_, tf, start=pd.Timestamp(start_date, tz="UTC"), refresh=True), tf),
                    cost_for=cost_for, make_cache=make_cache))
            elif _dfm is not None:
                st.info("Data terlalu pendek (butuh lebih dari 600 candle).")

if ADV == "ml":
    with t_adv:
        got = _res_data()
        if got:
            V.render_ml(*got)

if ADV == "wfo":
    with t_adv:
        got = _res_data()
        if got:
            U.render_wfo(*got)

if ADV == "port":
    with t_adv:
        got = _res_data()
        if got:
            U.render_portfolio(*got)
            V.render_similarity(*got)

if ADV == "cmp":
    with t_adv:
        def _load_any(source, sym, tf_, start, deriv=False):
            st_ = pd.Timestamp(start).date() if start else start_date
            return closed_only(load(source, sym, tf_, st_, bool(deriv) and source == "binance"), tf_)
        U.render_compare(_load_any, cost_for, make_cache)

if ADV == "sig":
    with t_adv:
        def _lv_raw(source, sym, tf_, deriv=False):
            need = int(TF_MINUTES[tf_] * 5200 / 1440 * (1.0 if source == "binance" else 1.5)) + 3
            cap = MAX_DAYS.get(source, {}).get(tf_)
            need = min(need, cap) if cap else need
            return load(source, sym, tf_, dt.date.today() - dt.timedelta(days=need), bool(deriv) and source == "binance")
        V.render_live(lambda source, sym, tf_, raw: closed_only(raw, tf_), _lv_raw, cost_for, lambda s_: SOURCES[s_].symbols())

def _open_hist(jid):
    r = J.result(jid)
    if not r:
        ss.hist_msg = "Hasil ini sudah tidak ada (dihapus setelah 24 jam atau server dimulai ulang)."
        return
    if J.status(jid) and J._meta(jid).get("kind") == "scan":
        ss.scan_rows, ss.scan_results = r
        ss.scan_job = ss.job_scan = jid
        ss.adv_tool = "scan"
        ss.hist_msg = "Hasil scan dibuka di tab '🧰 Alat lanjutan'."
        return
    ss.res, ss.res_job, ss.job_search, ss.sel, ss.res_scan = r, jid, jid, None, None
    if r.get("source") in SOURCES:
        ss.source, ss.symbol, ss.tf = r["source"], r["symbol"], r["tf"]
    ss.hist_msg = f"{r['symbol']} {r['tf']} dibuka. Lihat tab '📊 Hasil'."


with t_hist:
    st.caption("Hasil pencarian & scan tersimpan di server selama 24 jam (bisa dibuka lagi walau HP sempat ditutup). "
               "Kalau server gratis dimulai ulang, riwayat ikut hilang.")
    if ss.get("hist_msg"):
        st.success(ss.pop("hist_msg"))
    _h = J.history(limit=25)
    if not _h:
        st.info("Belum ada riwayat.")
    for _m in _h:
        _c1, _c2 = st.columns([4, 1])
        _when = pd.Timestamp(_m.get("finished") or _m.get("started"), unit="s", tz="UTC").tz_convert(TZ).strftime("%d %b %H:%M")
        _c1.markdown(f"**{_m.get('label') or _m['id']}**  \n{'Scan' if _m.get('kind') == 'scan' else 'Pencarian'} · selesai {_when} WIB")
        _c2.button("Buka", key=f"h_{_m['id']}", on_click=_open_hist, args=(_m["id"],), width='stretch')

with t_help:
    st.markdown(open("README_ID.md", encoding="utf-8").read() if __import__("os").path.exists("README_ID.md") else "")


# give freed memory back to the OS after every run (the free cloud has ~1 GB RAM)
def _trim_memory():
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


try:
    _save_session()
except Exception:
    pass
if time.time() % 50 < 1:
    J.cleanup()
_trim_memory()
