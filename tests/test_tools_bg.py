"""v8.5: slow robustness tools run as background jobs and survive a reconnect."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

TOK = f"tool{int(time.time())}"
def new():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
    at.query_params["s"] = TOK; at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at

def wait(ids):
    for jid in ids:
        while (J.status(jid) or {}).get("state") in ("antri", "jalan"):
            time.sleep(1)

at = new()
at.session_state["families"] = ["ma_cross", "donchian", "keltner"]; at.session_state["n_variants"] = 400; at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
wait([at.session_state["job_search"]]); at.run(); at.run()
b = next(b for b in at.button if b.label.startswith("▶️ Jalankan tes harga acak")); b.click(); at.run()
tj = dict(at.session_state["tool_jobs"]); print("tool job started", list(tj))
del at                                         # phone minimized
at2 = new()
print("tool job restored in new session", dict(at2.session_state["tool_jobs"]) == tj)
wait(tj.values()); at2.run()
sk = list(tj)[0].split("|", 1)[1]
t = at2.session_state["tests"][sk]
print("perm result after reconnect", "perm" in t, round(t["perm"]["p"], 3) if "perm" in t else None)
assert "perm" in t and not at2.exception
b = next(b for b in at2.button if b.label.startswith("▶️ Bandingkan exit")); b.click(); at2.run()
wait(at2.session_state["tool_jobs"].values()); at2.run()
t = at2.session_state["tests"][sk]
print("compare exits via job", "v7cmp" in t, len(t.get("v7cmp", [])), "rows")
assert "v7cmp" in t and not at2.exception, [e.message for e in at2.exception]
print("TOOLS BG TEST PASSED")
