"""Run a heavy, short job (ML training) in a forked child process.
The child shares the parent's data (copy-on-write, nothing is copied up front), imports scikit-learn and
trains there, sends back only the small result, then exits. So scikit-learn and the training memory never
stay in the long-running app process (the free cloud has ~1 GB RAM). Runs in-process only if fork is unavailable.
If the child is killed (out-of-memory killer) it is retried once in a fresh child, then IsolateError is raised;
the heavy job is never re-run inside the app process."""
from __future__ import annotations
import gc, os, time


def _call(fn, args, kwargs, progress):
    return fn(*args, progress=progress, **kwargs) if progress is not None else fn(*args, **kwargs)


class IsolateError(RuntimeError):
    """The forked child ended without a result (killed, e.g. by the out-of-memory killer, crashed, or timed out).
    Subclass of RuntimeError, so callers that already catch RuntimeError / Exception keep working."""


def _trim_parent():
    """Give freed memory back to the OS before a retry (glibc keeps freed heap pages otherwise)."""
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _once(fn, args, kwargs, progress, timeout):
    """One forked attempt. Returns ("ok", result) or ("lost", reason); re-raises the child's own exception."""
    import multiprocessing as mp
    import warnings
    ctx = mp.get_context("fork")
    rd, wr = ctx.Pipe(duplex=False)

    def child():
        try:
            prog = (lambda f, t: wr.send(("p", f, t))) if progress is not None else None
            wr.send(("ok", _call(fn, args, kwargs, prog)))
        except BaseException as e:  # report, the parent re-raises
            wr.send(("err", f"{type(e).__name__}: {e}"))
        finally:
            wr.close()

    # gc.freeze(): move every existing object to the permanent generation before forking, so the child's garbage
    # collector never walks (and writes to) the parent's objects. Those writes made the kernel copy the touched pages
    # into the child (copy-on-write), which is the transient RAM peak while a child runs. The parent unfreezes
    # right after the fork, so its own collection is unchanged.
    gc.collect()
    gc.freeze()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)   # "process is multi-threaded" notice from fork
            p = ctx.Process(target=child, daemon=True)
            p.start()
    finally:
        gc.unfreeze()
    wr.close()
    deadline = time.time() + timeout
    timed_out = True
    try:
        while time.time() < deadline:
            if rd.poll(0.2):
                try:
                    msg = rd.recv()
                except EOFError:
                    timed_out = False
                    break
                if msg[0] == "p":
                    if progress is not None:
                        progress(msg[1], msg[2])
                    continue
                if msg[0] == "ok":
                    return "ok", msg[1]
                raise RuntimeError(msg[1])
            if not p.is_alive() and not rd.poll(0):
                timed_out = False
                break
    finally:
        if p.is_alive():
            p.join(2)
        if p.is_alive():
            p.kill()
        p.join(1)
        rd.close()
    if timed_out:
        return "timeout", f"lebih dari {timeout} detik"
    code = p.exitcode
    if code is not None and code < 0:
        import signal
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return "killed", name
    return "lost", f"exit code {code}"


def run(fn, *args, progress=None, timeout=900, retries=None, **kwargs):
    """Run fn(*args, **kwargs) in a forked child and return its result (the child's exceptions are re-raised).
    If the child is killed (e.g. SIGKILL from the out-of-memory killer) the parent frees what it can and tries ONE
    more time in a fresh child; it never re-runs the heavy job inside the long-running app process (that would put
    the same RAM peak on the app itself and could take the whole app down on the 1 GB cloud). If the retry also
    fails, or the job times out, IsolateError (a RuntimeError) is raised with a short message for the page / job.
    Only when fork is not available (or VELA_NO_FORK=1) the job runs in-process, as before."""
    if os.environ.get("VELA_NO_FORK") == "1" or not hasattr(os, "fork"):
        return _call(fn, args, kwargs, progress)
    if retries is None:
        retries = int(os.environ.get("VELA_ISOLATE_RETRIES", "1"))
    why = ""
    for attempt in range(retries + 1):
        try:
            kind, val = _once(fn, args, kwargs, progress, timeout)
        except OSError as e:          # fork itself failed (often ENOMEM): do not try in-process either
            kind, val = "lost", f"fork gagal: {e}"
        if kind == "ok":
            return val
        why = val
        print(f"[isolate] child gave no result ({kind}: {val}), attempt {attempt + 1} of {retries + 1}", flush=True)
        if kind == "timeout":
            break                     # a second try would take just as long
        _trim_parent()
    if why.startswith("SIGKILL"):
        msg = ("Proses dihentikan sistem karena memori server penuh (SIGKILL). Coba lagi dengan data lebih pendek, "
               "variasi lebih sedikit, atau tunggu tugas berat lain selesai.")
    else:
        msg = f"Proses latar berhenti tanpa hasil ({why}). Coba lagi."
    raise IsolateError(msg)
