"""v8.7: walk-forward, portfolio and multi-asset tests run as background jobs and survive a reconnect."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

TOK = f"xbg{int(time.time())}"

def new():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
    at.query_params["s"] = TOK; at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at

def wait(ids, limit=900):
    t0 = time.time()
    for jid in ids:
        while (J.status(jid) or {}).get("state") in ("antri", "jalan") and time.time() - t0 < limit:
            time.sleep(1)
        st = J.status(jid) or {}
        print("job", jid, st.get("state"), st.get("error", "")[:200], "after", round(time.time() - t0), "s")
        assert st.get("state") == "selesai", st

def set_by_label(widgets, label, value):
    w = next((x for x in widgets if getattr(x, "label", None) == label), None)
    assert w is not None, f"missing widget {label!r} among {[getattr(x, 'label', None) for x in widgets]}"
    w.set_value(value)
    return w

at = new()
at.session_state["families"] = ["ma_cross", "donchian", "keltner"]
at.session_state["n_variants"] = 500
at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
wait([at.session_state["job_search"]])
at.run(); at.run()
res = at.session_state["res"]
print("search passed", sum(o["passed"] for o in res["out"]), "of", len(res["out"]), "oos_ok", sum(o.get("oos_ok") for o in res["out"]))
# include weaker candidates so the portfolio has at least 2 strategies
for cb in at.checkbox:
    if cb.label.startswith("Ikutkan yang lolos unseen"):
        cb.check()
at.run()

# --- walk-forward (small: 3 windows x 300 variants) ---
at.selectbox(key="adv_tool").set_value("wfo").run()
set_by_label(at.slider, "Jumlah jendela uji", 3)
set_by_label(at.select_slider, "Variasi per jendela", 300)
at.run()
next(b for b in at.button if b.label.startswith("▶️ Jalankan walk-forward")).click(); at.run()
jobs = dict(at.session_state["tool_jobs"])
print("wfo job started", jobs)
del at                                          # phone minimized
at2 = new()
print("jobs restored", dict(at2.session_state.get("tool_jobs", {})) == jobs)
wait(jobs.values(), 700); at2.run(); at2.run()
w = at2.session_state["res"]["search_tests"].get("wfo")
print("wfo after reconnect", bool(w), "windows" , w and len(w.get("rows") or []),
      "total", w and round(w["total"] * 100, 2), "exceptions", [e.message[:200] for e in at2.exception])
assert w and len(w.get("rows") or []) >= 3 and not at2.exception

# --- portfolio ---
at2.selectbox(key="adv_tool").set_value("port").run()
for cb in at2.checkbox:
    if cb.label.startswith("Ikutkan yang lolos unseen"):
        if not cb.value:
            cb.check()
at2.run()
btn = next((b for b in at2.button if b.label.startswith("▶️ Buat portofolio")), None)
if btn is None:
    print("portfolio button missing (not enough candidates); skip")
    port_ok = False
else:
    btn.click(); at2.run()
    pj = dict(at2.session_state["tool_jobs"])
    print("port job", pj)
    del at2
    at2 = new()
    wait(pj.values(), 400); at2.run(); at2.run()
    p = at2.session_state["res"]["search_tests"].get("port")
    print("port after reconnect", bool(p), p and p.get("keep"), "exceptions", [e.message[:200] for e in at2.exception])
    assert p and p.get("keep") and not at2.exception
    port_ok = True

# --- multi-asset (2 symbols, cached data) ---
# basket lives in the strategy-detail 'Uji ketahanan' group, always on the Hasil page
# shrink the default 5-asset basket if the widget is visible
for ms in at2.multiselect:
    if ms.label == "Aset yang diuji":
        cur = list(ms.value)
        pick = [s for s in cur if s != at2.session_state["res"]["symbol"]][:2] or cur[:2]
        ms.set_value(pick); at2.run(); print("basket set", pick); break
btn = next((b for b in at2.button if b.label.startswith("▶️ Jalankan tes multi-aset")), None)
assert btn is not None, f"basket button missing :: {[b.label for b in at2.button]}"
btn.click(); at2.run()
bj = dict(at2.session_state["tool_jobs"])
print("basket job", bj)
del at2
at3 = new()
wait(bj.values(), 500); at3.run(); at3.run()
# basket result is copied from basket_job into tests[skey]
tests = at3.session_state["tests"]
rows, errs = None, None
for t in tests.values():
    if isinstance(t, dict) and t.get("basket") is not None:
        rows, errs = t["basket"], t.get("basket_err")
        break
print("basket after reconnect", rows and [(r["symbol"], r["passed"], round(r["net"] * 100, 1)) for r in rows],
      "errs", errs, "exceptions", [e.message[:200] for e in at3.exception])
assert rows and not at3.exception
print("EXTRA BG TEST PASSED", "port" if port_ok else "port-skipped")
