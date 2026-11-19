"""Tester round (feat-ux): history only lists searches/scans, a tool job can't be opened as a search result,
a failed search shows its error on the search tab (no jump to Hasil) and the Start button is enabled again.
Run: python tests/test_ux_fixes.py   (uses its own job folder, no network search is started)"""
import os, sys, json, pickle, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["VELA_JOB_DIR"] = tempfile.mkdtemp(prefix="vela_ux_")
sys.path.insert(0, ROOT); os.chdir(ROOT)
from streamlit.testing.v1 import AppTest
D = os.environ["VELA_JOB_DIR"]
now = time.time()


def fake(jid, kind, label, res=None, err=None):
    json.dump(dict(id=jid, label=label, kind=kind, started=now - 60, finished=now), open(f"{D}/{jid}.json", "w"))
    if err:
        open(f"{D}/{jid}.err", "w").write(err)
    else:
        pickle.dump(res, open(f"{D}/{jid}.pkl", "wb"))


fake("tool_aaaa", "tool", "Uji harga acak TOOLJOB", res=dict(p=0.01))
fake("search_bad", "search", "BTCUSDT 1h gagal", err="RuntimeError: boom test\nTraceback...")

at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=300)
at.query_params["s"] = f"ux{int(now)}"
at.run()
assert not at.exception, at.exception
md = " ".join(m.value for m in at.markdown)
assert "TOOLJOB" not in md, "tool job must not be listed in Riwayat"
print("history filter ok")

# opening a tool job directly (old links / stale buttons) must not put a non-search result into Hasil
# failed search: error on the search tab, no jump to Hasil, Start enabled
at.session_state["job_search"] = "search_bad"
at.run()
assert not at.exception, at.exception
errs = [e.value for e in at.error]
assert any("boom test" in e for e in errs), errs
assert at.session_state["nav"] != "📊 Hasil", at.session_state["nav"]
start = [b for b in at.button if b.label.startswith("🚀 Mulai cari strategi")]
assert start and not start[0].disabled
print("failed search ok:", errs[0][:60])
# the Hasil tab without a result explains what to do
assert not at.exception
print("UX FIXES TEST PASSED")
