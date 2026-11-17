"""Self-contained HTML report (charts load plotly.js from the CDN)."""
from __future__ import annotations
import html
import datetime as dt
import pandas as pd

CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#111;color:#eee;max-width:1100px;margin:auto;padding:18px}
h1,h2{color:#fff} table{border-collapse:collapse;margin:8px 0;font-size:14px} td,th{border:1px solid #333;padding:4px 8px;text-align:right}
th{background:#222} td:first-child,th:first-child{text-align:left} .g{font-size:28px;font-weight:700;padding:8px 14px;border-radius:8px;
display:inline-block;background:#1e1e1e} pre{background:#1b1b1b;padding:10px;overflow:auto;font-size:12px;border-radius:6px}
.note{color:#aaa;font-size:13px}"""


def _table(df: pd.DataFrame, fmt="{:,.2f}"):
    def f(v):
        if isinstance(v, (int,)) and not isinstance(v, bool):
            return f"{v:,}"
        if isinstance(v, float):
            return "-" if v != v else fmt.format(v)
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in [df.index.name or ""] + list(df.columns))
    body = "".join("<tr>" + f"<td>{html.escape(str(i))}</td>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>"
                   for i, r in zip(df.index, df.itertuples(index=False)))
    return f"<table><tr>{head}</tr>{body}</table>"


def build(title, subtitle, grade=None, tables=(), figs=(), pine=None, notes=()):
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><p class='note'>{html.escape(subtitle)}<br>Dibuat {dt.datetime.now(dt.timezone(dt.timedelta(hours=7))):%d %b %Y %H:%M} WIB "
             "dengan Backtest Lab (aplikasi pribadi, non-komersial).</p>"]
    if grade and grade[0] is not None:
        parts.append(f"<div class='g'>Rapor ketahanan: {html.escape(str(grade[1]))} · {grade[0]:.0f}/100</div>")
    for name, df in tables:
        parts.append(f"<h2>{html.escape(name)}</h2>{_table(df)}")
    first = True
    for name, fig in figs:
        parts.append(f"<h2>{html.escape(name)}</h2>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False))
        first = False
    if pine:
        parts.append(f"<h2>Kode Pine Script</h2><pre>{html.escape(pine)}</pre>")
    for n_ in notes:
        parts.append(f"<p class='note'>{html.escape(n_)}</p>")
    parts.append("<p class='note'>Ini hasil uji data lama, bukan jaminan dan bukan saran investasi.</p></body></html>")
    return "".join(parts)
