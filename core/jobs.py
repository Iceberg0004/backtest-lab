"""v8: long jobs (strategy search, scan, robustness tests...) run in a background thread of the server, NOT in the page.
If the phone browser is minimized / the connection drops / the page is reloaded, the job keeps running and the result
is picked up again by its id (kept in the page URL). Results are saved on disk for 24 hours.
Only one heavy job runs at a time (the free server has ~1 GB RAM); others wait in a queue."""
from __future__ import annotations
import hashlib, json, os, pickle, threading, time, traceback, uuid

JOB_DIR = os.environ.get("VELA_JOB_DIR", "/tmp/vela_jobs")
TTL = float(os.environ.get("VELA_JOB_TTL_H", "24")) * 3600
MAX_MB = float(os.environ.get("VELA_JOB_MAX_MB", "300"))
# The registry lives in a tiny holder module (no __file__) so it survives app.py dropping and re-importing our own
# modules after a code update: a running job thread and the page must keep seeing the same dict / locks.
import sys as _sys, types as _types
_S = _sys.modules.get("_bt_jobs_state")
if _S is None:
    _S = _types.ModuleType("_bt_jobs_state")
    _S.RUN, _S.LOCK, _S.HEAVY = {}, threading.Lock(), threading.Semaphore(1)
    _sys.modules["_bt_jobs_state"] = _S
_RUN: dict = _S.RUN              # id -> live state of jobs started by THIS server process
_LOCK = _S.LOCK
_HEAVY = _S.HEAVY                # one heavy job at a time


def _dir():
    os.makedirs(JOB_DIR, exist_ok=True)
    return JOB_DIR


def _p(jid, ext):
    return os.path.join(_dir(), f"{jid}.{ext}")


def job_id(kind, settings):
    h = hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:12]
    return f"{kind}_{h}"


def seed_of(settings):
    """Same settings -> same random seed -> same result (so a finished job can safely be reused)."""
    return int(hashlib.sha1(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:8], 16) % 100000


def status(jid):
    """None | dict(state='antri'|'jalan'|'selesai'|'gagal', frac, text, error, started, label)."""
    if not jid:
        return None
    with _LOCK:
        s = _RUN.get(jid)
        if s is not None:
            return dict(s)
    if os.path.exists(_p(jid, "pkl")):
        return dict(state="selesai", frac=1.0, text="Selesai", **_meta(jid))
    if os.path.exists(_p(jid, "err")):
        return dict(state="gagal", frac=1.0, text="Gagal", error=open(_p(jid, "err")).read()[-600:], **_meta(jid))
    return None


def _trim():
    """Give memory freed by the job back to the OS (the free cloud has ~1 GB RAM)."""
    try:
        import gc, ctypes
        gc.collect()
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def request_stop(jid):
    """User pressed Stop: the job finishes early with what it has (partial result). True if the job was running."""
    with _LOCK:
        s = _RUN.get(jid)
        if s is None:
            return False
        s["stop"] = True
        s["text"] = "Dihentikan. Menyiapkan hasil sementara dari yang sudah ketemu..."
        return True


def should_stop(jid):
    with _LOCK:
        s = _RUN.get(jid)
        return bool(s and s.get("stop"))


def _meta(jid):
    try:
        m = json.load(open(_p(jid, "json")))
        return {k: m.get(k) for k in ("label", "kind", "started", "finished", "partial")}
    except Exception:
        return {}


def start(jid, fn, label="", kind="", heavy=True, extra_meta=None):
    """fn(progress) -> picklable result; progress(frac, text). Never starts twice: if the id is running, queued or done,
    nothing happens. Returns the status."""
    st_ = status(jid)
    if st_ and st_["state"] in ("antri", "jalan", "selesai"):
        return st_
    for ext in ("err",):
        try:
            os.remove(_p(jid, ext))
        except OSError:
            pass
    meta = dict(id=jid, label=label, kind=kind, started=time.time(), **(extra_meta or {}))
    with _LOCK:
        _RUN[jid] = dict(state="antri", frac=0.0, text="Menunggu giliran (ada pekerjaan lain yang sedang jalan)...",
                         started=meta["started"], label=label, kind=kind)
    json.dump(meta, open(_p(jid, "json"), "w"), default=str)

    def prog(frac, text="", best=None):
        with _LOCK:
            s = _RUN.get(jid)
            if s is not None:
                if frac is not None:
                    s["frac"] = float(max(0.0, min(1.0, frac)))
                if text is not None and not s.get("stop"):
                    s["text"] = str(text)
                if best is not None:
                    s["best"] = best

    def run():
        sem = _HEAVY if heavy else None
        if sem:
            sem.acquire()
        try:
            with _LOCK:
                _RUN[jid].update(state="jalan", text="Mulai...")
            res = fn(prog)
            tmp = _p(jid, f"tmp{uuid.uuid4().hex[:6]}")
            with open(tmp, "wb") as f:
                pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, _p(jid, "pkl"))
            meta["finished"] = time.time()
            try:
                meta["partial"] = bool(res.get("meta", {}).get("partial")) if isinstance(res, dict) else False
            except Exception:
                pass
            json.dump(meta, open(_p(jid, "json"), "w"), default=str)
            del res
        except Exception as e:
            open(_p(jid, "err"), "w").write(f"{type(e).__name__}: {e}\n" + traceback.format_exc()[-1500:])
        finally:
            with _LOCK:
                _RUN.pop(jid, None)
            _trim()
            if sem:
                sem.release()
            cleanup()

    threading.Thread(target=run, daemon=True, name=f"job-{jid}").start()
    return status(jid)


def result(jid):
    try:
        with open(_p(jid, "pkl"), "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def forget(jid):
    for ext in ("pkl", "json", "err"):
        try:
            os.remove(_p(jid, ext))
        except OSError:
            pass


def history(kind=None, limit=30):
    out = []
    try:
        names = [n for n in os.listdir(_dir()) if n.endswith(".json") and not n.startswith("sess_")]
    except OSError:
        return out
    for n in names:
        try:
            m = json.load(open(os.path.join(JOB_DIR, n)))
        except Exception:
            continue
        if kind and m.get("kind") != kind:
            continue
        if not os.path.exists(_p(m["id"], "pkl")):
            continue
        out.append(m)
    out.sort(key=lambda m: m.get("finished") or m.get("started") or 0, reverse=True)
    return out[:limit]


def cleanup():
    """Delete jobs older than TTL, then the oldest until the folder is under MAX_MB. Running jobs are never touched."""
    try:
        now = time.time()
        with _LOCK:
            live = set(_RUN)
        files = []
        for n in os.listdir(_dir()):
            f = os.path.join(JOB_DIR, n)
            jid = n.rsplit(".", 1)[0]
            if jid in live:
                continue
            try:
                stt = os.stat(f)
            except OSError:
                continue
            if now - stt.st_mtime > TTL:
                os.remove(f); continue
            files.append((stt.st_mtime, stt.st_size, f))
        tot = sum(s for _, s, _ in files)
        for _, s, f in sorted(files):
            if tot <= MAX_MB * 1e6:
                break
            if f.endswith(".pkl") or f.endswith(".json"):
                os.remove(f); tot -= s
    except Exception:
        pass


# ------------------------------------------------------------------ per-browser session state on disk (?s=<token> in the URL)
def sess_load(tok):
    try:
        return json.load(open(_p(f"sess_{tok}", "json")))
    except Exception:
        return {}


def sess_save(tok, data):
    try:
        tmp = _p(f"sess_{tok}", f"tmp{uuid.uuid4().hex[:6]}")
        json.dump(data, open(tmp, "w"), default=str)
        os.replace(tmp, _p(f"sess_{tok}", "json"))
    except Exception:
        pass


def new_token():
    return uuid.uuid4().hex[:10]
