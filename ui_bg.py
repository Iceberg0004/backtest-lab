"""Run slow per-strategy tools (random-price tests, multi-asset, compare exits, ML ...) as server background jobs,
so minimizing the phone or a dropped connection does not lose the work. Results land in the strategy's test store
(ui_extra.tests_for) when done; job ids are kept in session state (and in the saved session) so they come back after a reload."""
from __future__ import annotations
import streamlit as st
import core.jobs as J

ss = st.session_state


def _jobs():
    return ss.setdefault("tool_jobs", {})


def launch(t, skey, name, label, fn, settings):
    """fn(prog) -> result; prog(frac, text). Same settings -> same job (no double work)."""
    jid = J.job_id("tool", dict(settings, tool=name, skey=skey))
    st_ = J.status(jid)
    if st_ is None or st_["state"] == "gagal":
        J.start(jid, fn, label=label, kind="tool")
    _jobs()[f"{name}|{skey}"] = jid
    t.pop(name, None)


def poll(t, skey, name, what):
    """Show progress while the job runs; put the result into t[name] when it is done. Returns True while running."""
    k = f"{name}|{skey}"
    jid = _jobs().get(k)
    if not jid:
        return False
    st_ = J.status(jid)
    if st_ is None:                       # server restarted / result expired
        _jobs().pop(k, None)
        return False
    if st_["state"] == "selesai":
        if name not in t:
            r = J.result(jid)
            if r is not None:
                t[name] = r
        return False
    if st_["state"] == "gagal":
        st.error(f"{what} gagal: " + (st_.get("error") or "").split("\n")[0])
        _jobs().pop(k, None)
        return False
    _panel(jid, what)
    return True


@st.fragment(run_every=2)
def _panel(jid, what):
    s_ = J.status(jid)
    if not s_ or s_["state"] not in ("antri", "jalan"):
        st.rerun()
        return
    st.progress(s_["frac"], text=s_["text"] or "Berjalan...")
    st.caption(f"⏳ {what} berjalan di server. Boleh minimize HP, hasilnya tetap muncul di sini.")
