"""v8.7 follow-up: portfolio as a background job (needs >=2 passing strategies; use gold)."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

TOK = f"port{int(time.time())}"

def new():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
    at.query_params["s"] = TOK; at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at

def wait(ids, limit=700):
    t0 = time.time()
    for jid in ids:
        while (J.status(jid) or {}).get("state") in ("antri", "jalan") and time.time() - t0 < limit:
            time.sleep(1)
        st = J.status(jid) or {}
        print("job", jid, st.get("state"), st.get("error", "")[:200], "after", round(time.time() - t0), "s")
        assert st.get("state") == "selesai", st

at = new()
at.selectbox(key="source").set_value("dukascopy").run()
at.selectbox(key="symbol").set_value("XAUUSD").run()
at.selectbox(key="tf").set_value("1h").run()
at.session_state["families"] = ["ma_cross", "donchian", "keltner", "supertrend", "orb"]
at.session_state["n_variants"] = 800
at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
wait([at.session_state["job_search"]])
at.run(); at.run()
res = at.session_state["res"]
print("search passed", sum(o["passed"] for o in res["out"]), "oos_ok", sum(bool(o.get("oos_ok")) for o in res["out"]), "of", len(res["out"]))
for cb in at.checkbox:
    if cb.label.startswith("Ikutkan yang lolos unseen"):
        cb.check()
at.run()
at.selectbox(key="adv_tool").set_value("port").run()
btn = next((b for b in at.button if b.label.startswith("▶️ Buat portofolio")), None)
assert btn is not None, f"still no portfolio button; buttons={[b.label for b in at.button]}"
btn.click(); at.run()
jobs = dict(at.session_state["tool_jobs"])
print("port job started", jobs)
del at
at2 = new()
print("jobs restored", dict(at2.session_state.get("tool_jobs", {})) == jobs)
wait(jobs.values(), 400); at2.run(); at2.run()
p = at2.session_state["res"]["search_tests"].get("port")
print("port after reconnect", bool(p), p and p.get("keep"),
      p and {k: round(v, 3) for k, v in p["stats_all"].items()},
      "exceptions", [e.message[:200] for e in at2.exception])
assert p and p.get("keep") and not at2.exception
print("PORT BG TEST PASSED")
