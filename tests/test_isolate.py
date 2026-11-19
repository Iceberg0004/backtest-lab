"""core/isolate.py: results, progress, child errors, and what happens when the child is killed (OOM killer = SIGKILL).
Run: python tests/test_isolate.py"""
import os, sys, signal, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import isolate as ISO

failed = []
def check(name, cond, info=""):
    print(("OK  " if cond else "FAIL"), name, info, flush=True)
    if not cond:
        failed.append(name)

PARENT = os.getpid()

def add(a, b, progress=None):
    if progress:
        progress(0.5, "half")
    return a + b

seen = []
check("result comes back", ISO.run(add, 2, 3) == 5)
check("progress forwarded", ISO.run(add, 1, 1, progress=lambda f, t: seen.append((f, t))) == 2 and seen == [(0.5, "half")], seen)

def boom():
    raise ValueError("bad input")
try:
    ISO.run(boom); check("child exception re-raised", False)
except RuntimeError as e:
    check("child exception re-raised", "ValueError: bad input" in str(e) and not isinstance(e, ISO.IsolateError), str(e))

calls = tempfile.mktemp()
def die_always():
    open(calls, "a").write(f"{os.getpid()}\n")
    os.kill(os.getpid(), signal.SIGKILL)      # what the kernel OOM killer does
t0 = time.time()
try:
    ISO.run(die_always); check("killed child -> IsolateError", False)
except ISO.IsolateError as e:
    pids = open(calls).read().split()
    check("killed child -> IsolateError", "SIGKILL" in str(e), str(e)[:80])
    check("retried once in a fresh child, never in the app process", len(pids) == 2 and str(PARENT) not in pids, pids)
check("killed child is a RuntimeError for old callers", issubclass(ISO.IsolateError, RuntimeError))

flag = tempfile.mktemp()
def die_first_time():
    if not os.path.exists(flag):
        open(flag, "w").write("x")
        os.kill(os.getpid(), signal.SIGKILL)
    return os.getpid()
r = ISO.run(die_first_time)
check("retry succeeds after one kill", isinstance(r, int) and r != PARENT, r)

try:
    ISO.run(time.sleep, 30, timeout=2); check("timeout -> IsolateError, no in-process rerun", False)
except ISO.IsolateError as e:
    check("timeout -> IsolateError, no in-process rerun", time.time() - t0 < 60 and "detik" in str(e), str(e))

def exit_quiet():
    os._exit(3)
try:
    ISO.run(exit_quiet, retries=0); check("silent exit -> IsolateError", False)
except ISO.IsolateError as e:
    check("silent exit -> IsolateError", "exit code 3" in str(e), str(e))

os.environ["VELA_NO_FORK"] = "1"
check("VELA_NO_FORK runs in-process", ISO.run(os.getpid) == PARENT)
del os.environ["VELA_NO_FORK"]

print("ISOLATE TEST", "FAILED " + str(failed) if failed else "PASSED", flush=True)
sys.exit(1 if failed else 0)
