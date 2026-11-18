"""v8.8: HTML report is built as a background job and comes back after a reconnect."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

TOK = f"rep{int(time.time())}"

def new():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
    at.query_params["s"] = TOK; at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at

def wait(ids, limit=400):
    t0 = time.time()
    for jid in ids:
        while (J.status(jid) or {}).get("state") in ("antri", "jalan") and time.time() - t0 < limit:
            time.sleep(1)
        st = J.status(jid) or {}
        print("job", jid, st.get("state"), st.get("error", "")[:200], "after", round(time.time() - t0), "s")
        assert st.get("state") == "selesai", st

at = new()
at.session_state["families"] = ["ma_cross", "donchian"]
at.session_state["n_variants"] = 300
at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
wait([at.session_state["job_search"]])
at.run(); at.run()
btn = next((b for b in at.button if b.label == "Buat laporan HTML" or b.label.startswith("Buat laporan HTML")), None)
assert btn is not None, f"report button missing :: {[b.label for b in at.button]}"
btn.click(); at.run()
jobs = dict(at.session_state["tool_jobs"])
print("report job started", jobs)
assert any(k.startswith("html_report|") for k in jobs)
del at
at2 = new()
print("jobs restored", dict(at2.session_state.get("tool_jobs", {})) == jobs)
wait(jobs.values(), 300); at2.run(); at2.run()
rep = at2.session_state.get("report")
html = rep[1] if rep else None
print("report after reconnect", bool(html), "bytes", html and len(html),
      "has equity", html and "Equity" in html, "exceptions", [e.message[:200] for e in at2.exception])
assert html and len(html) > 1000 and "Equity" in html and not at2.exception
# Streamlit AppTest lists download buttons separately from regular buttons
dls = [b.label for b in at2.download_button]
print("download buttons", dls)
assert any("Download laporan" in (lab or "") for lab in dls)
print("REPORT BG TEST PASSED")
