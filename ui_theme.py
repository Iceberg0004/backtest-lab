"""v9 look & feel, built on open-source Streamlit components (credits in README):
- streamlit-extras (Apache-2.0)   metric card styling (pure CSS, no iframe)
- streamlit-shadcn-ui (MIT)       grade badge
- streamlit-lightweight-charts (MIT, TradingView lightweight-charts) candle chart with trade markers
Every helper falls back to plain Streamlit if a component is missing or fails, so the app never breaks because of the UI."""
from __future__ import annotations
import pandas as pd
import streamlit as st

try:
    from streamlit_extras.metric_cards import style_metric_cards as _smc
except Exception:          # pragma: no cover
    _smc = None
try:
    import streamlit_shadcn_ui as _ui
except Exception:          # pragma: no cover
    _ui = None
try:
    from streamlit_lightweight_charts import renderLightweightCharts as _lw
except Exception:          # pragma: no cover
    _lw = None

ACCENT = "#6366F1"
UP, DOWN = "#22C55E", "#EF4444"
BG, CARD, BORDER = "#0B1020", "#131A2E", "#26304D"

CSS = f"""<style>
.block-container {{padding-top: 1.2rem; max-width: 1280px;}}
.bt-hero {{background: linear-gradient(135deg, #1E1B4B 0%, #312E81 45%, #0F766E 100%); border-radius: 18px;
  padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 8px 28px rgba(0,0,0,.35);}}
.bt-hero h1 {{font-size: 1.55rem !important; margin: 0 !important; padding: 0 !important; color: #fff; letter-spacing: -.01em;}}
.bt-hero p {{margin: .25rem 0 0 0; color: #C7D2FE; font-size: .92rem;}}
.bt-pill {{display: inline-block; padding: .1rem .55rem; border-radius: 999px; background: rgba(255,255,255,.14);
  color: #E0E7FF; font-size: .75rem; margin-right: .3rem;}}
.stTabs [data-baseweb="tab-list"] {{gap: .35rem; overflow-x: auto; scrollbar-width: none; flex-wrap: nowrap;}}
.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {{display: none;}}
.stTabs [data-baseweb="tab"] {{background: {CARD}; border: 1px solid {BORDER}; border-radius: 999px; padding: .35rem .9rem;
  white-space: nowrap; height: auto;}}
.stTabs [aria-selected="true"] {{background: {ACCENT}; border-color: {ACCENT}; color: #fff;}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display: none;}}
[data-testid="stExpander"] details {{border-radius: 14px; border-color: {BORDER};}}
[data-testid="stVerticalBlockBorderWrapper"] {{border-radius: 16px;}}
.stButton button {{border-radius: 12px; transition: transform .05s ease;}}
.stButton button:active {{transform: scale(.98);}}
.stButton button[kind="primary"] {{background: linear-gradient(135deg, {ACCENT}, #8B5CF6); border: 0;
  box-shadow: 0 6px 18px rgba(99,102,241,.35);}}
[data-testid="stProgress"] > div > div > div > div {{background: linear-gradient(90deg, {ACCENT}, #22D3EE);}}
@media (max-width: 640px) {{
  .bt-hero {{padding: .8rem .9rem; border-radius: 14px;}}
  .bt-hero h1 {{font-size: 1.25rem !important;}}
  .bt-hero p {{font-size: .82rem;}}
  [data-testid="stHorizontalBlock"] {{gap: .4rem; flex-wrap: wrap !important;}}
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{min-width: calc(50% - .4rem) !important;
    flex: 1 1 calc(50% - .4rem) !important; width: calc(50% - .4rem) !important;}}
  .bt-hero .bt-tags, .bt-hero p {{display: none;}}
  .bt-hero {{padding: .5rem .8rem !important; border-radius: 12px !important;}}
  .bt-hero h1 {{font-size: 1.08rem !important;}}
  .st-key-mkt {{padding: .45rem .5rem !important;}}
  .st-key-mkt [data-testid="stHorizontalBlock"] {{flex-wrap: nowrap !important; gap: .3rem !important;}}
  .st-key-mkt [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{min-width: 0 !important; width: auto !important; flex: 1.3 1 0 !important;}}
  .st-key-mkt [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{flex: .8 1 0 !important;}}
  .st-key-mkt [data-testid="stWidgetLabel"] {{display: none;}}
  .st-key-mkt [data-baseweb="select"] > div {{min-height: 2.3rem; font-size: .85rem;}}
  .st-key-ctxcap {{display: none;}}
  .block-container {{padding-top: 3.2rem !important;}}   /* clear the fixed Streamlit header (it covered the title) */
  .bt-hero {{margin-bottom: .5rem;}}
  [data-testid="stMetricValue"] {{font-size: 1.05rem !important;}}
  [data-testid="stMetric"] {{padding: .45rem .55rem !important;}}
  [data-testid="stMetricLabel"] p {{font-size: .78rem;}}
  /* tester round: settings inputs one per row (50% columns were too narrow for number inputs / sliders) */
  .st-key-setx [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{min-width: 100% !important; width: 100% !important; flex: 1 1 100% !important;}}
  /* bigger touch targets: tabs, checkboxes, radios, expander headers */
  .stTabs [data-baseweb="tab"] {{min-height: 2.6rem;}}
  .stTabs [data-baseweb="tab-list"] {{overflow-x: auto; scrollbar-width: none;}}
  [data-testid="stCheckbox"] label, [data-testid="stRadio"] label {{min-height: 2.2rem; align-items: center;}}
  [data-testid="stExpander"] summary {{min-height: 2.8rem;}}
  .stButton button, .stDownloadButton button {{width: 100%;}}
  /* progress text readable on a phone */
  [data-testid="stProgress"] p {{font-size: .9rem; white-space: normal;}}
  /* tables never push the page sideways */
  [data-testid="stDataFrame"], [data-testid="stTable"] {{max-width: 100%; overflow-x: auto;}}
}}
/* Stop button stands out while a search runs */
[class*="st-key-stop_"] button {{background: #DC2626 !important; color: #fff !important; border: 0 !important; min-height: 3rem; font-weight: 700;}}
</style>"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)
    if _smc is not None:
        try:
            _smc(background_color=CARD, border_color=BORDER, border_left_color=ACCENT, border_radius_px=14, box_shadow=False)
        except Exception:
            pass


def hero(sub=""):
    st.markdown(
        f"""<div class="bt-hero"><h1>📈 Backtest Lab</h1>
<p>{sub or 'Cari & uji strategi trading dengan data asli. Hasil uji data lama, bukan saran.'}</p>
<div class="bt-tags" style="margin-top:.45rem"><span class="bt-pill">69 jenis strategi</span><span class="bt-pill">14 jenis SL</span>
<span class="bt-pill">Kode Pine</span><span class="bt-pill">Jalan di server</span></div></div>""",
        unsafe_allow_html=True)


_VARIANT = {"A": "default", "B": "default", "C": "secondary", "D": "outline", "E": "destructive"}


def grade_badge(letter, note="", key="grade"):
    """Shadcn badge for the grade; plain Streamlit badge if the component is unavailable."""
    if not letter or letter == "-":
        return
    label = f"Rapor {letter}" + (f" · {note}" if note else "")
    if _ui is not None:
        try:
            _ui.badges([(label, _VARIANT.get(letter[0], "secondary"))], key=key)
            return
        except Exception:
            pass
    color = {"A": "green", "B": "green", "C": "orange", "D": "red", "E": "red"}.get(letter[0], "gray")
    st.badge(label, color=color)


def has_tv():
    return _lw is not None


def _t(ts):
    # lightweight-charts shows times as UTC; shift to WIB so the axis reads in the user's time
    return int(pd.Timestamp(ts).tz_convert("Asia/Jakarta").tz_localize(None).timestamp()) if pd.Timestamp(ts).tzinfo \
        else int(pd.Timestamp(ts).timestamp()) + 7 * 3600


def tv_chart(df, tr=None, height=460, key="tv"):
    """TradingView-style candles + volume + trade markers. Returns False if the component is unavailable (caller falls back)."""
    if _lw is None or df is None or len(df) == 0:
        return False
    idx = df.index
    ts = [_t(x) for x in idx]
    candles = [dict(time=t, open=float(o), high=float(h), low=float(l), close=float(c))
               for t, o, h, l, c in zip(ts, df.open.values, df.high.values, df.low.values, df.close.values)]
    series = [dict(type="Candlestick", data=candles,
                   options=dict(upColor=UP, downColor=DOWN, borderVisible=False, wickUpColor=UP, wickDownColor=DOWN))]
    if "volume" in df and float(df.volume.fillna(0).sum()) > 0:
        series.append(dict(type="Histogram",
                           data=[dict(time=t, value=float(v), color="rgba(34,197,94,.35)" if c >= o else "rgba(239,68,68,.35)")
                                 for t, v, o, c in zip(ts, df.volume.fillna(0).values, df.open.values, df.close.values)],
                           options=dict(priceFormat=dict(type="volume"), priceScaleId=""),
                           priceScale=dict(scaleMargins=dict(top=0.82, bottom=0))))
    marks = []
    if tr is not None and len(tr.get("ent", [])):
        first = tr.get("_offset", 0)
        for k in range(len(tr["ent"])):
            e, x, d = int(tr["ent"][k]) - first, int(tr["ex"][k]) - first, int(tr["dir"][k])
            if 0 <= e < len(ts):
                marks.append(dict(time=ts[e], position="belowBar" if d == 1 else "aboveBar", color=UP if d == 1 else DOWN,
                                  shape="arrowUp" if d == 1 else "arrowDown", text="L" if d == 1 else "S"))
            if 0 <= x < len(ts) and tr["reason"][k] != 3:
                win = float(tr["ret"][k]) > 0
                marks.append(dict(time=ts[x], position="aboveBar" if d == 1 else "belowBar", color="#FACC15" if win else "#94A3B8",
                                  shape="circle", text=f"{float(tr['ret'][k]) * 100:+.1f}%"))
        marks.sort(key=lambda m: m["time"])
        series[0]["markers"] = marks
    chart = dict(height=height, layout=dict(background=dict(type="solid", color=BG), textColor="#CBD5E1"),
                 grid=dict(vertLines=dict(color="rgba(148,163,184,.08)"), horzLines=dict(color="rgba(148,163,184,.08)")),
                 crosshair=dict(mode=0), rightPriceScale=dict(borderColor=BORDER),
                 timeScale=dict(borderColor=BORDER, timeVisible=True, secondsVisible=False))
    try:
        _lw([dict(chart=chart, series=series)], key=key)
        return True
    except Exception:
        return False
