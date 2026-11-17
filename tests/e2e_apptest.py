"""End-to-end UI test with Streamlit's AppTest (no browser needed).
Usage: python tests/e2e_apptest.py binance BTCUSDT   |   python tests/e2e_apptest.py dukascopy EURUSD"""
import os, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest

source, symbol = sys.argv[1], sys.argv[2]
carry = len(sys.argv) > 3 and sys.argv[3] == "carry"
t0 = time.time()
at = AppTest.from_file(os.path.join(os.getcwd(), "app.py"), default_timeout=900)
at.run()
assert not at.exception, at.exception
at.selectbox(key="source").set_value(source).run()
at.selectbox(key="symbol").set_value(symbol).run()
at.selectbox(key="tf").set_value("1h").run()
at.select_slider(key="n_variants").set_value(2000).run()
if carry:
    if source == "binance":
        at.checkbox(key="use_funding").check().run()
    else:
        at.checkbox(key="use_swap").check().run()
print("setup ok", round(time.time() - t0, 1), "s")


def click(label_start):
    for b in at.button:
        if b.label.startswith(label_start):
            b.click()
            at.run()
            assert not at.exception, [e.message for e in at.exception]
            return True
    raise AssertionError(f"button not found: {label_start} :: {[b.label for b in at.button]}")


click("🚀 Mulai cari strategi")
print("search done", round(time.time() - t0, 1), "s")
for m in at.success:
    print("SUCCESS:", m.value[:300])
for m in at.info:
    print("INFO:", m.value[:200])
res = at.session_state["res"]
print("tested", res["meta"]["tested"], "passed", sum(o["passed"] for o in res["out"]), "pbo", res["search_tests"]["pbo"] and round(res["search_tests"]["pbo"]["pbo"], 3))
# make sure a detail is shown: prefer a passing strategy
idx = next((i for i, o in enumerate(res["out"]) if o["passed"]), 0)
at.session_state["sel"] = idx
at.run()
assert not at.exception, [e.message for e in at.exception]
o = res["out"][idx]
print("selected", idx, o["fam"], o["params"], "passed", o["passed"])
for cb in at.checkbox:
    if cb.label.startswith("Ikutkan yang lolos unseen"):
        cb.check()
at.run()
for lab in ["▶️ Jalankan tes harga acak", "▶️ Jalankan tes geser parameter", "▶️ Jalankan tes pencarian acak",
            "▶️ Cek dengan candle 1 menit", "▶️ Jalankan walk-forward", "▶️ Jalankan tes multi-aset", "▶️ Buat portofolio",
            "➕ Tambah ke perbandingan", "▶️ Uji ulang semua di data terbaru"]:
    t1 = time.time()
    try:
        click(lab)
        print("clicked", lab, round(time.time() - t1, 1), "s")
    except AssertionError as e:
        print("SKIP", lab, str(e)[:200])
key = f"{res['id']}_{idx}"
t = at.session_state["tests"][key]
print("grade", t.get("grade"), "dsr", round(t["dsr"]["dsr"], 4), "N", t["dsr"]["N"], "oos_psr", t.get("oos_psr"))
if "perm" in t: print("perm p", t["perm"]["p"], "real", round(t["perm"]["real"], 4), "median fake", round(float(sorted(t["perm"]["perm"])[len(t["perm"]["perm"]) // 2]), 4))
if "perm_search" in t: print("perm_search p", t["perm_search"]["p"], "real", round(t["perm_search"]["real"], 3))
if "jitter" in t: print("jitter share profit", t["jitter"]["share_profit"])
if "stress" in t: print("stress", [(r["mult"], round(r["net"] * 100, 2), round(r["pf"], 2)) for r in t["stress"]])
if "intrabar" in t:
    i = t["intrabar"]; print("intrabar", i["n_amb"], i["n_checked"], i["n_tp_first"], round(i["before"]["net"] * 100, 2), "->", round(i["after"]["net"] * 100, 2), i.get("error"))
if "basket" in t: print("basket", [(r["symbol"], r["passed"], round(r["net"] * 100, 1), round(r["pf"], 2)) for r in t["basket"]], t.get("basket_err"))
w = res["search_tests"].get("wfo")
if w: print("wfo total", round(w["total"] * 100, 2), "dd", round(w["maxdd"] * 100, 2), "wfe", w["wfe"], "pos", w["pos_windows"], [r["chosen"] for r in w["rows"]])
p = res["search_tests"].get("port")
print("portfolio", p and (p["keep"], {k: round(v, 3) for k, v in p["stats_all"].items()}, {k: round(v, 3) for k, v in p["stats_oos"].items()}))
cr = at.session_state["compare_res"] if "compare_res" in at.session_state else {}
print("compare", {k: (round(v["full"]["net"] * 100, 2) if "full" in v else v.get("error")) for k, v in cr.items()})
# final grade after all tests: re-render
at.run()
t = at.session_state["tests"][key]
print("FINAL grade", t.get("grade"))
print("exceptions", at.exception)
print("total", round(time.time() - t0, 1), "s")
