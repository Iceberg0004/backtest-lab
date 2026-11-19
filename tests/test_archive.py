"""v9: results kept beyond the 24 h job folder (SQLite) + save/restore as a .btlab file; tampered files are refused."""
import os, sys, time, tempfile, pickle, zipfile, io, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); os.chdir(ROOT)
T = tempfile.mkdtemp(prefix="bt_arch_")
os.environ["VELA_JOB_DIR"] = os.path.join(T, "jobs"); os.environ["BT_ARCHIVE"] = os.path.join(T, "a.sqlite")
import numpy as np, pandas as pd
import core.jobs as J
J.JOB_DIR = os.environ["VELA_JOB_DIR"]
from core import archive as A

res = dict(symbol="BTCUSDT", tf="1h", source="binance",
           df=pd.DataFrame({"x": np.arange(5.0)}, index=pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")),
           arr=np.arange(10), meta=dict(partial=False))
jid = J.job_id("search", {"t": time.time()})
J.start(jid, lambda prog: res, label="tes arsip", kind="search")
for _ in range(100):
    if J.status(jid)["state"] == "selesai":
        break
    time.sleep(0.05)
time.sleep(0.2)
assert A.has(jid), "not archived"
# simulate the 24 h clean-up
for ext in ("pkl", "json"):
    os.remove(os.path.join(J.JOB_DIR, f"{jid}.{ext}"))
assert J.status(jid)["state"] == "selesai"
r = J.result(jid)
assert r["df"].equals(res["df"]) and (r["arr"] == res["arr"]).all()
assert any(m["id"] == jid for m in J.history())
# file round trip into an empty archive
blob = A.export_bytes()
os.environ["BT_ARCHIVE"] = os.path.join(T, "b.sqlite")
assert A.count()[0] == 0
added, skipped, errs = A.import_bytes(blob)
assert (added, skipped, errs) == (1, 0, []), (added, skipped, errs)
assert A.import_bytes(blob)[1] == 1
# tampered file: code inside a result must be refused
class Evil:
    def __reduce__(self):
        return (os.system, ("echo PWNED > " + os.path.join(T, "pwned"),))
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w") as z:
    z.writestr("manifest.json", json.dumps(dict(format=A.MAGIC, version=1, items=[dict(id="search_evil1", kind="search", label="x")])))
    import zlib
    z.writestr("r/search_evil1.z", zlib.compress(pickle.dumps(Evil())))
os.environ["BT_ARCHIVE"] = os.path.join(T, "c.sqlite")
a2, _, e2 = A.import_bytes(buf.getvalue())
assert a2 == 0 and e2 == ["search_evil1"] and not os.path.exists(os.path.join(T, "pwned"))
try:
    A.import_bytes(b"not a zip"); raise SystemExit("accepted junk")
except ValueError:
    pass
# the app still renders the Riwayat page
from streamlit.testing.v1 import AppTest
at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
at.run()
assert not at.exception, [e.message for e in at.exception]
assert any("btlab" in (e.label or "") for e in at.expander), "save/restore box missing"
assert any("tes arsip" in m.value for m in at.markdown), "archived result not listed in Riwayat"
print("PASSED")
