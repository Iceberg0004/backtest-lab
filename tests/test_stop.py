"""v8.1: Stop keeps partial results; best-so-far is shown while running; finished result opens the 'Hasil' tab."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
at.query_params["s"] = f"stop{int(time.time())}"
at.run()
at.session_state["n_variants"] = 20000
at.session_state["time_budget"] = 900
at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
jid = at.session_state["job_search"]
t0 = time.time()
while time.time() - t0 < 120 and not (J.status(jid) or {}).get("best"):
    time.sleep(2)
st_ = J.status(jid)
print("running", st_["state"], "| best so far rows", len(st_.get("best") or []), (st_.get("best") or [{}])[0])
at.run()
stop_btn = [b for b in at.button if b.label.startswith("⏹️ Stop")]
print("stop button shown", bool(stop_btn))
if stop_btn:
    stop_btn[0].click(); at.run()
else:
    J.request_stop(jid)
t1 = time.time()
while (J.status(jid) or {}).get("state") in ("antri", "jalan") and time.time() - t1 < 300:
    time.sleep(2)
print("finished after stop in", round(time.time() - t1), "s, state", J.status(jid)["state"])
for _ in range(3):
    at.run()
r = at.session_state["res"]
print("partial", r["meta"]["partial"], "| tested", r["meta"]["tested"], "| candidates", len(r["out"]), "| passed", sum(o["passed"] for o in r["out"]))
print("tab after finish", at.session_state["nav"])
print("summary shown", any("Hasil SEMENTARA" in w.value for w in at.warning), any(m.value.startswith("### 🏆 Hasil") for m in at.markdown))
assert not at.exception, [e.message for e in at.exception]
assert r["meta"]["partial"] and r["meta"]["tested"] < 20000
print("STOP TEST PASSED")
