"""v8.6: ML training runs as a background job (fork from the job thread) and its result shows up."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
import core.jobs as J
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=900)
at.query_params["s"] = f"ml{int(time.time())}"; at.run()
at.session_state["families"] = ["ma_cross", "donchian"]; at.session_state["n_variants"] = 300; at.run()
next(b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")).click(); at.run()
while (J.status(at.session_state["job_search"]) or {}).get("state") in ("antri", "jalan"):
    time.sleep(1)
at.run(); at.run()
at.selectbox(key="adv_tool").set_value("ml").run()
next(b for b in at.button if b.label.startswith("🤖 Latih & uji model")).click(); at.run()
ids = list(at.session_state["tool_jobs"].values()); print("ml job", ids)
t0 = time.time()
for jid in ids:
    while (J.status(jid) or {}).get("state") in ("antri", "jalan") and time.time() - t0 < 700:
        time.sleep(2)
print("state", [J.status(j)["state"] for j in ids], round(time.time() - t0), "s")
at.run()
store = at.session_state["ml"]
keys = [k for k in store if not isinstance(k, str)]
print("ml result present", bool(keys), "| exceptions", [e.message[:200] for e in at.exception])
assert keys and not at.exception
print("ML BG TEST PASSED")
