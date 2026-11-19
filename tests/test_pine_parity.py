"""Pine export parity: Python engine trades vs the exported indicator() and strategy() run by PineTS.
Includes TP1/TP2 partial take-profit legs (each leg = one closed trade in TradingView).
Needs node + `npm install` in tools/pinecheck. Usage: python tests/test_pine_parity.py"""
import os, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PC = os.path.join(ROOT, "tools", "pinecheck")
subprocess.run([sys.executable, os.path.join(PC, "make_case.py")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
out = subprocess.run(["node", "run_all.mjs"], cwd=PC, capture_output=True, text=True).stdout
print("\n".join(out.strip().splitlines()[-4:]))
assert "ALL PARITY CHECKS PASSED" in out, "parity failed"
print("PASSED")
