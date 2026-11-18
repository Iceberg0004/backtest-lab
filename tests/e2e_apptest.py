"""End-to-end UI test with Streamlit's AppTest (no browser needed).
Usage: python tests/e2e_apptest.py binance BTCUSDT [carry] [vela] [evo] [combo] [scan] [deriv] [strict]
       python tests/e2e_apptest.py dukascopy EURUSD"""
import os, sys, time
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from streamlit.testing.v1 import AppTest
sys.path.insert(0, os.getcwd())
from core import engine as E

source, symbol = sys.argv[1], sys.argv[2]
flags = set(sys.argv[3:])
carry = "carry" in flags
t0 = time.time()
at = AppTest.from_file(os.path.join(os.getcwd(), "app.py"), default_timeout=900)
at.run()
assert not at.exception, at.exception
at.selectbox(key="source").set_value(source).run()
at.selectbox(key="symbol").set_value(symbol).run()
at.selectbox(key="tf").set_value("1h").run()
at.select_slider(key="n_variants").set_value(int(os.environ.get("E2E_VARIANTS", "2000"))).run()
if carry:
    if source == "binance":
        at.checkbox(key="use_funding").check().run()
    else:
        at.checkbox(key="use_swap").check().run()
if "vela" in flags:
    at.radio(key="mode").set_value("vela").run()
if "evo" in flags:
    at.radio(key="algo").set_value("evo").run()
if "combo" in flags:
    at.checkbox(key="combo").check().run()
if "deriv" in flags:
    at.checkbox(key="use_deriv").check().run()
if "strict" in flags:
    at.checkbox(key="strict_dsr").check().run()
if "pick" in flags:   # v7: search only with the new SL / TP types
    at.multiselect(key="v7_sl").set_value([3, 4, 5, 9, 10, 11, 13]).run()
    at.multiselect(key="v7_tp").set_value([1, 5, 7, 8]).run()
assert not at.exception, at.exception
print("setup ok", sorted(flags), round(time.time() - t0, 1), "s")


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
if res.get("deriv_info"):
    print("deriv", {k: (v.get("ok"), v.get("coverage")) for k, v in res["deriv_info"].items() if isinstance(v, dict)})
    print("deriv strategies listed", sum(E.uses_deriv(o["fam"], o["params"]) for o in res["out"]), "of", len(res["out"]),
          "passed with deriv", sum(o["passed"] and E.uses_deriv(o["fam"], o["params"]) for o in res["out"]))
print("tested", res["meta"]["tested"], "passed", sum(o["passed"] for o in res["out"]), "pbo", res["search_tests"]["pbo"] and round(res["search_tests"]["pbo"]["pbo"], 3))
# make sure a detail is shown: prefer a passing strategy
idx = next((i for i, o in enumerate(res["out"]) if o["passed"]), 0)
at.session_state["sel"] = idx
at.run()
assert not at.exception, [e.message for e in at.exception]
o = res["out"][idx]
print("selected", idx, o["fam"], o["params"], "passed", o["passed"])
if "pick" in flags:
    _bad = [(r["fam"], r["params"]["sl_mode"], r["params"].get("tp_mode")) for r in res["out"]
            if r["params"]["sl_mode"] not in (3, 4, 5, 9, 10, 11, 13) or r["params"].get("tp_mode") not in (1, 5, 7, 8)]
    print("pick check: results outside chosen types", len(_bad), _bad[:3])
for cb in at.checkbox:
    if cb.label.startswith("Ikutkan yang lolos unseen"):
        cb.check()
at.run()
for lab in ["▶️ Jalankan tes harga acak", "▶️ Jalankan tes geser parameter", "▶️ Jalankan tes pencarian acak",
            "▶️ Cek dengan candle 1 menit", "▶️ Jalankan walk-forward", "▶️ Jalankan tes multi-aset", "▶️ Buat portofolio", "🔗 Hitung kemiripan",
            "➕ Tambah ke perbandingan", "▶️ Uji ulang semua di data terbaru", "▶️ Bandingkan exit"]:
    t1 = time.time()
    try:
        click(lab)
        print("clicked", lab, round(time.time() - t1, 1), "s")
    except AssertionError as e:
        print("SKIP", lab, str(e)[:200])
        for m in list(at.error) + list(at.warning):
            print("  page says:", m.value[:300])
key = f"{res['id']}_{idx}"
try:
    _cmp = at.session_state[f"v7cmp_{key}"]
    print("exit compare", len(_cmp) - 1, "combos; profitable IS+unseen",
          sum(1 for r in _cmp[1:] if r["ok"] and r["is_m"]["net"] > 0 and r["oos_m"]["net"] > 0))
except Exception as e:
    print("exit compare missing", e)
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
sim = res["search_tests"].get("sim")
print("similarity", sim and (sim[2]["n"], [len(g) for g in sim[2]["groups"]][:10]))
cr = at.session_state["compare_res"] if "compare_res" in at.session_state else {}
print("compare", {k: (round(v["full"]["net"] * 100, 2) if "full" in v else v.get("error")) for k, v in cr.items()})
# ---- v4 features (all opt-in buttons)
import numpy as np
t1 = time.time()
at.toggle(key="ana_on").set_value(True).run()
assert not at.exception, [e.message for e in at.exception]
print("analytics shown", round(time.time() - t1, 1), "s", len(at.get("plotly_chart")), "charts")
click("🗺️ Buat peta"); print("sensitivity map ok")
at.toggle(key="ana_on").set_value(False).run()
at.toggle(key="sz_on").set_value(True).run()
at.selectbox(key="sz_mode").set_value("risk").run()
assert not at.exception, [e.message for e in at.exception]
click("▶️ Simulasikan challenge"); print("clicked challenge")
for mode in ("fixed", "vol", "kelly"):
    at.selectbox(key="sz_mode").set_value(mode).run()
    assert not at.exception, [e.message for e in at.exception]
print("sizing modes ok")
at.selectbox(key="sz_mode").set_value("risk").run()
at.toggle(key="sz_on").set_value(False).run()
for lab in ["🤖 Latih meta-label", "🌦️ Deteksi regime", "✨ Pilih otomatis", "Buat laporan HTML",
            "🤖 Latih & uji model", "📡 Cek sinyal sekarang"]:
    t1 = time.time()
    try:
        click(lab)
        print("clicked", lab, round(time.time() - t1, 1), "s")
    except AssertionError as e:
        print("SKIP", lab, str(e)[:300])
ss_ = at.session_state
if "meta" in ss_:
    for r in ss_["meta"].values():
        if "error" in r: print("meta", r["error"])
        else: print("meta", r["kind"], "thr", r["threshold"], "auc", r["auc_cv"] and round(r["auc_cv"], 3), "rule_oos", round(r["rule_oos"]["net"] * 100, 1), round(r["rule_oos"]["pf"], 2), r["rule_oos"]["trades"], "meta_oos", round(r["meta_oos"]["net"] * 100, 1), round(r["meta_oos"]["pf"], 2), r["meta_oos"]["trades"])
if "regime" in ss_:
    for k, (lab_, names, rows) in ss_["regime"].items(): print("regime", names, [round(r["Porsi unseen %"], 1) for r in rows])
if "pf_res" in ss_:
    r = ss_["pf_res"]; print("prop", r["mc"] and {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r["mc"].items()}, "hist", len(r["hist"]))
if "report" in ss_: print("report html bytes", len(ss_["report"][1]))
if "ml" in ss_:
    for r in ss_["ml"].values(): print("ml", r["kind"], "is", round(r["is_m"]["net"] * 100, 1), round(r["is_m"]["pf"], 2), "oos", round(r["oos_m"]["net"] * 100, 1), round(r["oos_m"]["pf"], 2), r["oos_m"]["trades"], "auc", {k: round(v, 3) for k, v in r["auc"].items()}, "cost2x", round(r["cost2x"]["net"] * 100, 1), round(r["seconds"], 1), "s")
if "lv_rows" in ss_:
    for r in ss_["lv_rows"]: print("live", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
# final grade after all tests: re-render
at.run()
t = at.session_state["tests"][key]
print("FINAL grade", t.get("grade"), "strict_pass", t.get("strict_pass"))
if "scan" in flags:
    at.multiselect(key="scan_tfs").set_value(["1h"]).run()
    at.select_slider(key="scan_var").set_value(500).run()
    at.select_slider(key="scan_time").set_value(120).run()
    t1 = time.time()
    click("🛰️ Mulai scan")
    rows = at.session_state["scan_rows"]
    print("scan", round(time.time() - t1, 1), "s", "matrix charts", len(at.get("plotly_chart")),
          "fam_grades", sum(len(r.get("fam_grades") or {}) for r in rows))
    for r in rows:
        print("  ", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("key", "best_idx")})
    keys = [r["key"] for r in rows if r.get("key")]
    if keys:
        click("Buka di 'Hasil & Detail'")
        print("opened", at.session_state["res"]["symbol"], at.session_state["res"]["tf"], "exceptions", at.exception)
print("exceptions", at.exception)
print("total", round(time.time() - t0, 1), "s")
