"""End-to-end test of the 'Grid · Martingale · DCA' tab with Streamlit AppTest.
Usage: python tests/e2e_multi.py binance BTCUSDT   (or dukascopy XAUUSD / EURUSD)"""
import os, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest

source, symbol = sys.argv[1], sys.argv[2]
t0 = time.time()
at = AppTest.from_file(os.path.join(os.getcwd(), "app.py"), default_timeout=900)
at.run()
at.selectbox(key="source").set_value(source).run()
at.selectbox(key="symbol").set_value(symbol).run()
at.selectbox(key="tf").set_value("1h").run()
at.toggle(key="mp_on").set_value(True).run()
assert not at.exception, [e.message for e in at.exception]


def click(key):
    at.button(key=key).click(); at.run()
    assert not at.exception, [e.message for e in at.exception]


def show(mode):
    s = at.session_state["mp_res"]
    if not s:
        print(mode, "NO RESULT"); return
    r = list(s.values())[0]; t = r["t"]
    def m(x): return f"net {x['net']*100:+.1f}% dd {x['maxdd']*100:.1f}% tr {x['trades']} wr {x['winrate']*100:.0f}% pf {x['pf']:.2f} expo {x['max_expo']:.1f} legs {x['max_legs']} under {x['under_days']:.0f}d blown {x['blown']}"
    print(f"[{mode}] IS  {m(t['is_m'])}")
    print(f"[{mode}] OOS {m(t['oos_m'])} | cost2x {t['cost2x']['net']*100:+.1f}% | jitter {t['jitter']}")
    if t.get("mc"):
        mc = t["mc"]; print(f"[{mode}] MC p_margin_call {mc['p_ruin']*100:.1f}% p_dd50 {mc['p_dd50']*100:.0f}% dd95 {mc['dd95']*100:.0f}% net5 {mc['net5']*100:.0f}% med {mc['net50']*100:.0f}%")
    print(f"[{mode}] GRADE {t['grade'][1]} {t['grade'][0] and round(t['grade'][0])} {t['grade'][2]}")
    if r.get("basket"): print(f"[{mode}] basket", r["basket"])


for mode in ["grid", "dca", "mart", "pyr", "hedge", "zone"]:
    t1 = time.time()
    at.selectbox(key="mp_mode").set_value(mode).run()
    at.select_slider(key="mp_npaths").set_value(100).run()
    click("mp_run")
    if mode in ("grid", "dca", "mart"):
        click("mp_mc")
    if mode == "grid":
        click("mp_basket")
    if mode == "dca":
        at.select_slider(key="mp_nv").set_value(300).run()
        click("mp_search")
        sr = at.session_state["mp_search_res"][3]
        r0 = sr["rows"][0]
        print(f"[dca search] tested {sr['tested']} ok_is {sr['ok_is']} best IS {r0['is_m']['net']*100:+.1f}% dd {r0['is_m']['maxdd']*100:.0f}% -> OOS {r0['oos_m']['net']*100:+.1f}% dd {r0['oos_m']['maxdd']*100:.0f}% blown {r0['oos_m']['blown']}")
        click("mp_apply")
        print("[dca] below: searched best settings re-run")
        click("mp_run"); click("mp_mc")
    show(mode)
    codes = [c.value for c in at.code]
    print(f"[{mode}] pine code blocks {len(codes)}", (codes[-1][:60].replace(chr(10), ' ') if codes else ""), round(time.time() - t1, 1), "s")
for mode in ["pairs", "rebal"]:
    at.selectbox(key="mp_mode").set_value(mode).run()
    click("mp_pair_run" if mode == "pairs" else "mp_reb_run")
    if mode == "pairs":
        r = at.session_state["mp_pair_res"][1]
        print("[pairs]", {k: round(v, 3) if isinstance(v, float) else v for k, v in r["fit"].items()}, r["pp"], "IS", {k: round(v, 3) for k, v in r["im"].items()}, "OOS", {k: round(v, 3) for k, v in r["om"].items()})
    else:
        _, _, r, d = at.session_state["mp_reb_res"]
        print("[rebal]", {k: round(v, 3) for k, v in r["m"].items()}, "bh", {k: round(v, 3) for k, v in r["m_bh"].items()}, "n", r["n_reb"],
              "| dca", round(d["dca_final"], 2), "lump", round(d["lump_final"], 2))
print("exceptions", at.exception)
print("total", round(time.time() - t0, 1), "s")
