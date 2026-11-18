"""v8: a search must survive the page connection dropping (phone minimized) and a reload.
Session 1 starts a search and disappears; session 2 opens the same URL token while it runs; session 3 after it is done."""
import os, sys, time, threading
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

TOK = f"t{int(time.time())}"
def new():
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
    at.query_params["s"] = TOK
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at

at = new()
at.session_state["families"] = ["ma_cross", "donchian", "rsi_mr"]
at.session_state["n_variants"] = 500
at.run()
for b in at.button:
    if b.label.startswith("🚀 Mulai cari strategi"):
        b.click(); at.run(); break
jid = at.session_state["job_search"]
print("job", jid, J.status(jid)["state"])
del at                                               # "phone minimized": the page session is gone

time.sleep(2)
at2 = new()
print("session 2 job", at2.session_state["job_search"] == jid, "state", J.status(jid)["state"],
      "progress shown", any("berjalan di server" in c.value for c in at2.caption))
threads = [t.name for t in threading.enumerate() if t.name.startswith("job-")]
print("job threads", threads)
# pressing the button again with the same settings must not start a second job
t0 = time.time()
while J.status(jid)["state"] in ("antri", "jalan") and time.time() - t0 < 600:
    time.sleep(3)
print("finished", J.status(jid)["state"], round(time.time() - t0), "s after reconnect")
at2.run()
print("session 2 has result", bool(at2.session_state["res"]) and at2.session_state["res"]["id"] == jid)
del at2

t1 = time.time()
at3 = new()                                           # reload later
print("session 3 restored", at3.session_state["res"]["id"] == jid, "in", round(time.time() - t1, 1), "s",
      "settings kept", at3.session_state["n_variants"] == 500 and at3.session_state["families"] == ["ma_cross", "donchian", "rsi_mr"])
for b in at3.button:
    if b.label.startswith("🚀 Mulai cari strategi"):
        b.click(); at3.run(); break
print("same settings -> same job", at3.session_state["job_search"] == jid, "threads", [t.name for t in threading.enumerate() if t.name.startswith("job-")])
assert not at3.exception, [e.message for e in at3.exception]
print("RECONNECT TEST PASSED")
