"""Run a heavy, short job (ML training) in a forked child process.
The child shares the parent's data (copy-on-write, nothing is copied up front), imports scikit-learn and
trains there, sends back only the small result, then exits. So scikit-learn and the training memory never
stay in the long-running app process (the free cloud has ~1 GB RAM). Falls back to running in-process if
fork is unavailable or the child dies without an answer."""
from __future__ import annotations
import os, time


def _call(fn, args, kwargs, progress):
    return fn(*args, progress=progress, **kwargs) if progress is not None else fn(*args, **kwargs)


def run(fn, *args, progress=None, timeout=900, **kwargs):
    if os.environ.get("VELA_NO_FORK") == "1" or not hasattr(os, "fork"):
        return _call(fn, args, kwargs, progress)
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)   # "process is multi-threaded" notice from fork
        p = ctx.Process(target=child, daemon=True)
        p.start()
    wr.close()
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            if rd.poll(0.2):
                try:
                    msg = rd.recv()
                except EOFError:
                    break
                if msg[0] == "p":
                    if progress is not None:
                        progress(msg[1], msg[2])
                    continue
                if msg[0] == "ok":
                    return msg[1]
                raise RuntimeError(msg[1])
            if not p.is_alive() and not rd.poll(0):
                break
    finally:
        if p.is_alive():
            p.join(2)
        if p.is_alive():
            p.kill()
        p.join(1)
        rd.close()
    print("[isolate] child gave no result, running in-process", flush=True)
    return _call(fn, args, kwargs, progress)
