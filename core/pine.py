"""Export a found strategy to TradingView Pine Script v5 (strategy + indicator).
Family blocks come from core/families.py; filters and exits below mirror core/engine.py.
The indicator() version re-runs the app's own position simulation bar by bar (exact, checked with PineTS).
The strategy() version uses TradingView's order engine; see NOTES for where it can differ."""
from . import engine as E

R_ = repr


def _filter_block(p):
    """Entry confirmations, same order and maths as engine.filter_masks."""
    p = E.full_params(p)
    L = ["// ---- filter / konfirmasi entry", "fltL = true", "fltS = true"]
    t = p.get("trend", 0)
    if t:
        L += [f"flTr = ta.ema(close, {t})", "fltL := fltL and close > flTr", "fltS := fltS and close < flTr",
              f"plot(flTr, 'Trend EMA {t}', color=color.gray)"]
    rg = p.get("regime", 0)
    if rg:
        op = ">=" if rg == 1 else "<"
        L += ["// kondisi pasar (ADX 14): " + ("hanya saat trending" if rg == 1 else "hanya saat sideways"),
              "dUp = high - high[1]", "dDn = low[1] - low",
              "pDM = na(dUp) ? na : (dUp > dDn and dUp > 0 ? dUp : 0.0)",
              "mDM = na(dDn) ? na : (dDn > dUp and dDn > 0 ? dDn : 0.0)",
              "pDI = 100 * ta.rma(pDM, 14) / atr", "mDI = 100 * ta.rma(mDM, 14) / atr", "diSum = pDI + mDI",
              "adxV = 100 * ta.rma(math.abs(pDI - mDI) / (diSum == 0 ? 1 : diSum), 14)",
              f"regimeOk = adxV {op} {p['adx_thr']}", "fltL := fltL and regimeOk", "fltS := fltS and regimeOk"]
    if p.get("sess"):
        a, b, nm = E.SESS[p["sess"]]
        L += [f"// sesi {nm} {a:02d}:00-{b:02d}:00 UTC (jam buka candle)", "flHr = math.floor(time / 3600000) % 24",
              f"flSes = flHr >= {a} and flHr < {b}", "fltL := fltL and flSes", "fltS := fltS and flSes"]
    dw = p.get("dow", 0)
    if dw:
        cond = {1: "flDow < 5", 2: "flDow != 0", 3: "flDow != 4"}[dw]
        L += [f"// hari: {E.DOW[dw]} (UTC, 0 = Senin)", "flDow = (math.floor(time / 86400000) + 3) % 7",
              f"flDay = {cond}", "fltL := fltL and flDay", "fltS := fltS and flDay"]
    if p.get("htf_m"):
        P = p["htf_m"] * 60000; n = p["htf_len"]
        L += [f"// tren timeframe besar ({p['htf_m']} menit, EMA {n}) dihitung sendiri: hanya candle besar yang sudah TUTUP (tanpa lookahead)",
              "var float hE = na", "var float hS = 0.0", "var int hC = 0", "var float hLast = na",
              f"if bar_index > 0 and math.floor(time / {P}) != math.floor(time[1] / {P})",
              "    hX = close[1]", "    hLast := hX", "    hC += 1",
              f"    if hC <= {n}", "        hS += hX", f"        if hC == {n}", f"            hE := hS / {n}",
              "    else", f"        hE := 2.0 / ({n} + 1) * hX + (1 - 2.0 / ({n} + 1)) * hE",
              "htfUp = not na(hE) and hLast > hE", "htfDn = not na(hE) and hLast < hE",
              "fltL := fltL and htfUp", "fltS := fltS and htfDn"]
    if p.get("vol"):
        L += ["flVr = atr / ta.sma(atr, 100)", f"flVol = flVr {'>' if p['vol'] == 1 else '<'} 1",
              "fltL := fltL and flVol", "fltS := fltS and flVol"]
    if p.get("rsi_f"):
        L += ["flR = ta.rsi(close, 14)", "fltL := fltL and flR > 50", "fltS := fltS and flR < 50"]
    if p.get("volf"):
        L += ["flVm = ta.sma(volume, 20)", "flVok = volume > flVm", "fltL := fltL and flVok", "fltS := fltS and flVok"]
    if p.get("exit_sig", 0) == 1:
        L += ["// keluar hanya saat sinyal kebalikan (sebelum filter)", "sigLX = shortEntry", "sigSX = longEntry"]
    else:
        L += ["sigLX = longExit", "sigSX = shortExit"]
    L += ["sigLE = longEntry and fltL", "sigSE = shortEntry and fltS"]
    return "\n".join(L)


def _signals_block(fam, p):
    p = E.full_params(p)
    return "// ---- sinyal dasar\n" + "\n".join(E.FAMILIES[fam]["pine"](p)) + "\n" + _filter_block(p)


def _header_inputs(p, direction, lev=1.0, pip=1.0, tf_min=60, cost_pct=0.0):
    """Exit settings as constants (same numbers as the app) + the SL/TP level series from core/exits.py."""
    p = E.full_params(p)
    trc, trk = E.X.trail_args(p)
    tp2 = float(p["tp2_r"]) if p["tp1_r"] > 0 else 0.0
    L = [f"allowLong  = input.bool({str(direction in ('long', 'both')).lower()}, 'Izinkan Long')",
         f"allowShort = input.bool({str(direction in ('short', 'both')).lower()}, 'Izinkan Short')",
         "atr = ta.atr(14)",
         f"// Exit: {E.X.describe(p, lev)}",
         f"slNone = {str(int(p['sl_mode']) == 14).lower()}",
         f"tp1R = {R_(float(p['tp1_r']))}", f"tp1F = {R_(float(p['tp1_f']))}", f"tp2R = {R_(tp2)}", f"tp2F = {R_(float(p['tp2_f']))}",
         f"beR = {R_(float(p['be_r']))}", f"beOff = {R_(float(p['be_off']))}",
         f"trCode = {int(trc)}", f"trK = {R_(float(trk))}", f"trAct = {R_(float(p['trail_act']))}",
         f"maxBars = {int(p['max_bars'])}", f"tsBars = {int(p['ts_bars'])}", f"coolN = {int(p['cool'])}", f"maxDay = {int(p['max_day'])}",
         f"dLoss = {R_(float(p['dloss']) / 100.0)}", f"dProf = {R_(float(p['dprof']) / 100.0)}", f"reEnt = {int(p['reent'])}",
         f"cfC = {R_(float(cost_pct) / 100.0)}", f"levC = {R_(float(lev))}"]
    L += E.X.pine_lines(p, lev, pip, tf_min)
    return "\n".join(L)


SIM = r"""
// ==== simulasi posisi (sama persis dengan aplikasi) ====
f_rr(p, e, x) => p == 1 ? (x * (1 - cfC) - 0.0 - (e * (1 + cfC) + 0.0)) / e : ((e * (1 - cfC) - 0.0) - (x * (1 + cfC) + 0.0)) / e
f_tmax(s, lv, p) => na(lv) ? s : na(s) ? lv : p == 1 ? math.max(s, lv) : math.min(s, lv)
var int pos = 0
var float ep = na
var int ei = 0
var float stp = na
var float stp0 = na
var float tgt = na
var float tg1 = na
var float tg2 = na
var float bel = na
var float rV = 0.0
var float cRef = 0.0
var float qty = 1.0
var float realP = 0.0
var bool p1 = false
var bool p2 = false
var bool beD = false
var float hhI = na
var float llI = na
var int dKey = -1
var float dRet = 0.0
var int eKey = -1
var int eCnt = 0
var int lastLoss = -1000000000
var int blockDir = 0
var bool nxExit = false
var int nxDir = 0
var int nxXr = 0
var float nxR = na
var float nxC = na
var float nxTp = na
var bool nxOk = false
var int nxDay = 0
bool goLong = false
bool goShort = false
bool exitSig = false
bool exitStop = false
bool exitTp = false
bool exitTime = false
bool hitTp1 = false
bool hitTp2 = false
float tradePnl = na
int xDayI = int(xDay)

// A) open candle ini: order dari keputusan di close candle sebelumnya
if nxExit and pos != 0
    float r = (realP + qty * f_rr(pos, ep, open)) * levC
    tradePnl := r * 100
    exitSig := nxXr == 0
    exitTime := nxXr != 0
    if xDayI != dKey
        dKey := xDayI
        dRet := 0.0
    dRet := dRet + r
    if r < 0
        lastLoss := bar_index
    if reEnt == 1
        blockDir := pos
    pos := 0
if nxDir != 0 and nxOk and not na(nxR)
    pos := nxDir
    ep := open
    ei := bar_index
    qty := 1.0
    realP := 0.0
    p1 := false
    p2 := false
    beD := false
    hhI := na
    llI := na
    rV := nxR
    cRef := nxC
    stp0 := cRef - pos * rV
    stp := slNone ? na : stp0
    tgt := nxTp
    tg1 := tp1R > 0 ? cRef + pos * tp1R * rV : na
    tg2 := tp2R > 0 ? cRef + pos * tp2R * rV : na
    if not na(tg1) and not na(tgt) and (pos == 1 ? tg1 >= tgt : tg1 <= tgt)
        tg1 := na
    if na(tg1) or (not na(tg2) and not na(tgt) and (pos == 1 ? tg2 >= tgt : tg2 <= tgt))
        tg2 := na
    bel := beR > 0 ? cRef + pos * beR * rV : na
    if nxDay != eKey
        eKey := nxDay
        eCnt := 0
    eCnt := eCnt + 1
    goLong := pos == 1
    goShort := pos == -1

// B) di dalam candle: stop dulu (aman), lalu TP1, TP2, target akhir
if pos != 0
    bool hs = pos == 1 ? low <= stp : high >= stp
    bool h1 = not p1 and not na(tg1) and (pos == 1 ? high >= tg1 : low <= tg1)
    bool h2 = not p2 and not na(tg2) and (pos == 1 ? high >= tg2 : low <= tg2)
    bool ht = not na(tgt) and (pos == 1 ? high >= tgt : low <= tgt)
    float px = na
    if hs
        px := pos == 1 ? math.min(open, stp) : math.max(open, stp)
        exitStop := true
    else
        if h1
            float q1 = pos == 1 ? math.max(open, tg1) : math.min(open, tg1)
            realP := realP + tp1F * f_rr(pos, ep, q1)
            qty := 1.0 - tp1F
            p1 := true
            hitTp1 := true
        if h2 and p1
            float q2 = pos == 1 ? math.max(open, tg2) : math.min(open, tg2)
            realP := realP + tp2F * f_rr(pos, ep, q2)
            qty := 1.0 - tp1F - tp2F
            p2 := true
            hitTp2 := true
        if ht
            px := pos == 1 ? math.max(open, tgt) : math.min(open, tgt)
            exitTp := true
    if not na(px)
        float r2 = (realP + qty * f_rr(pos, ep, px)) * levC
        tradePnl := r2 * 100
        if xDayI != dKey
            dKey := xDayI
            dRet := 0.0
        dRet := dRet + r2
        if r2 < 0
            lastLoss := bar_index
        if reEnt == 1
            blockDir := pos
        pos := 0

// C) close candle: garis SL, trailing, breakeven
if pos != 0
    float fav = na
    if pos == 1
        hhI := na(hhI) ? high : math.max(hhI, high)
        fav := hhI - cRef
        if not na(xStL) and xStL < close
            stp := f_tmax(stp, xStL, 1)
    else
        llI := na(llI) ? low : math.min(llI, low)
        fav := cRef - llI
        if not na(xStS) and xStS > close
            stp := f_tmax(stp, xStS, -1)
    if trCode > 0 and (trAct <= 0 or fav >= trAct * rV)
        float lv = na
        if trCode == 1
            lv := na(atr) ? na : pos == 1 ? high - trK * atr : low + trK * atr
        else if trCode == 2
            lv := na(atr) ? na : pos == 1 ? hhI - trK * atr : llI + trK * atr
        else if trCode == 3
            lv := pos == 1 ? hhI * trK : llI * (2.0 - trK)
        else if trCode == 4
            float xa = pos == 1 ? xTrL : xTrS
            lv := not na(xa) and ((pos == 1 and xa < close) or (pos == -1 and xa > close)) ? xa : na
        else
            float st_ = trK * rV
            float k_ = st_ > 0 ? math.floor(fav / st_) : 0.0
            lv := k_ >= 1 ? stp0 + pos * k_ * st_ : na
        stp := f_tmax(stp, lv, pos)
    if beR > 0 and not beD and (pos == 1 ? high >= bel : low <= bel)
        stp := f_tmax(stp, ep + pos * beOff * rV, pos)
        beD := true

// D) keputusan untuk open candle berikutnya
if blockDir == 1 and sigSE
    blockDir := 0
else if blockDir == -1 and sigLE
    blockDir := 0
bool gate = not xNb
if coolN > 0 and bar_index - lastLoss < coolN
    gate := false
if maxDay > 0 and xDayI == eKey and eCnt >= maxDay
    gate := false
if dKey == xDayI
    if dLoss > 0 and dRet <= -dLoss
        gate := false
    if dProf > 0 and dRet >= dProf
        gate := false
bool tUp = maxBars > 0 and pos != 0 and (bar_index - ei + 1) >= maxBars
bool tSt = tsBars > 0 and pos != 0 and (bar_index - ei + 1) >= tsBars and pos * (close - ep) <= 0
int nd = 0
bool ne = false
int xr = 0
if pos == 1
    if allowShort and sigSE
        ne := true
        nd := -1
    else if sigLX
        ne := true
    else if tUp
        ne := true
        xr := 4
    else if tSt
        ne := true
        xr := 6
    else if xFx
        ne := true
        xr := 5
else if pos == -1
    if allowLong and sigLE
        ne := true
        nd := 1
    else if sigSX
        ne := true
    else if tUp
        ne := true
        xr := 4
    else if tSt
        ne := true
        xr := 6
    else if xFx
        ne := true
        xr := 5
else
    if allowLong and sigLE
        nd := 1
    else if allowShort and sigSE
        nd := -1
if nd != 0 and (not gate or nd == blockDir or xFx)
    nd := 0
nxExit := ne
nxDir := nd
nxXr := xr
nxR := nd == 1 ? xRL : nd == -1 ? xRS : na
nxC := close
nxTp := nd == 1 ? xTpL : nd == -1 ? xTpS : na
nxOk := not na(atr)
nxDay := xDayI
"""

ORDERS = r"""
// ==== order TradingView mengikuti simulasi di atas ====
if pos == 0 and nxDir == 0 and strategy.position_size != 0
    strategy.close_all(comment="Sinkron")
if nxExit and pos != 0 and nxDir == 0
    strategy.close_all(comment=nxXr == 0 ? "Exit" : nxXr == 4 ? "Waktu habis" : nxXr == 6 ? "Time stop" : "Tutup sesi")
if nxDir == 1 and nxOk and not na(nxR)
    strategy.entry("Long", strategy.long)
else if nxDir == -1 and nxOk and not na(nxR)
    strategy.entry("Short", strategy.short)
float oStop = na
float oTp = na
float oT1 = na
float oT2 = na
bool oP1 = false
bool oP2 = false
int oDir = 0
if nxDir != 0 and nxOk and not na(nxR)
    oDir := nxDir
    oStop := slNone ? na : nxC - nxDir * nxR
    oTp := nxTp
    oT1 := tp1R > 0 ? nxC + nxDir * tp1R * nxR : na
    oT2 := tp2R > 0 ? nxC + nxDir * tp2R * nxR : na
    if not na(oT1) and not na(oTp) and (nxDir == 1 ? oT1 >= oTp : oT1 <= oTp)
        oT1 := na
    if na(oT1) or (not na(oT2) and not na(oTp) and (nxDir == 1 ? oT2 >= oTp : oT2 <= oTp))
        oT2 := na
    oP1 := not na(oT1)
    oP2 := not na(oT2)
else if pos != 0 and not nxExit
    oDir := pos
    oStop := stp
    oTp := tgt
    oT1 := tg1
    oT2 := tg2
    oP1 := not na(tg1) and not p1
    oP2 := not na(tg2) and not p2
if oDir != 0
    string idE = oDir == 1 ? "Long" : "Short"
    if oP1
        strategy.exit(idE + "-TP1", idE, qty_percent=tp1F * 100, stop=oStop, limit=oT1)
    if oP2
        strategy.exit(idE + "-TP2", idE, qty_percent=tp2F * 100, stop=oStop, limit=oT2)
    if not na(oStop) or not na(oTp)
        strategy.exit(idE + "-SL/TP", idE, stop=oStop, limit=oTp)
"""


NOTES = ("// Catatan versi strategy(): order TradingView mengikuti simulasi yang sama dengan aplikasi (masuk/keluar di candle yang sama).\\n"
         "// Kalau dalam SATU candle stop loss dan take profit sama-sama tersentuh, TradingView menebak urutannya sendiri,\\n"
         "// sedangkan aplikasi menganggap stop loss duluan. TP sebagian memakai qty_percent. Karena itu harga keluar di strategy()\\n"
         "// bisa sedikit beda. Versi indicator() menghitung posisi sendiri dan sama persis dengan aplikasi.")


def _esc(t):
    return str(t).replace('"', "'")


def _table(rows, live_strategy):
    """On-chart stats table. App test results are fixed text; the strategy() version also shows
    TradingView's own live backtest numbers so the two can be compared."""
    n = len(rows) + (5 if live_strategy else 0) + 1
    L = ["// ==== TABLE START (tabel statistik di chart)",
         "showTable = input.bool(true, 'Tampilkan tabel statistik')",
         f"var table statsT = table.new(position.top_right, 2, {n}, bgcolor=color.new(color.black, 15), border_width=1)",
         "if showTable and barstate.islast",
         '    table.cell(statsT, 0, 0, "Backtest Lab", text_color=color.yellow, text_size=size.small)',
         '    table.cell(statsT, 1, 0, "hasil uji, bukan saran", text_color=color.gray, text_size=size.small)']
    r = 1
    for k, v in rows:
        L += [f'    table.cell(statsT, 0, {r}, "{_esc(k)}", text_color=color.white, text_size=size.small)',
              (f'    table.cell(statsT, 1, {r}, pos == 1 ? "LONG" : pos == -1 ? "SHORT" : "Tidak ada posisi", text_color=color.white, text_size=size.small)'
               if v == "__POS__" else f'    table.cell(statsT, 1, {r}, "{_esc(v)}", text_color=color.white, text_size=size.small)')]
        r += 1
    if live_strategy:
        live = [("TV: net profit", 'str.tostring(strategy.netprofit / strategy.initial_capital * 100, "#.##") + "%"'),
                ("TV: trades", "str.tostring(strategy.closedtrades)"),
                ("TV: win rate", 'strategy.closedtrades > 0 ? str.tostring(strategy.wintrades / strategy.closedtrades * 100, "#.#") + "%" : "-"'),
                ("TV: profit factor", 'strategy.grossloss > 0 ? str.tostring(strategy.grossprofit / strategy.grossloss, "#.##") : "-"'),
                ("TV: max DD", 'str.tostring(strategy.max_drawdown / strategy.initial_capital * 100, "#.##") + "% (modal awal)"')]
        for k, expr in live:
            L += [f'    table.cell(statsT, 0, {r}, "{k}", text_color=color.aqua, text_size=size.small)',
                  f'    table.cell(statsT, 1, {r}, {expr}, text_color=color.aqua, text_size=size.small)']
            r += 1
    L.append("// ==== TABLE END")
    return "\n".join(L)


class NoPine(ValueError):
    pass


PINE_DERIV_MSG = ("Strategi ini memakai data derivatif kripto (funding, open interest, long/short atau volume taker) yang "
                  "tidak tersedia di Pine Script TradingView, jadi tidak bisa diekspor ke Pine.")


BENCH_SYM = "BINANCE:BTCUSDT"


def _bench_line(bench_vals=None):
    if bench_vals is not None:      # parity tests only: the benchmark closes embedded as an array
        vals = ", ".join(repr(float(x)) for x in bench_vals)
        return f"var float[] BENCH = array.from({vals})\nbenchC = array.get(BENCH, bar_index)"
    return (f'benchSym = "{BENCH_SYM}"  // pembanding (ganti kalau perlu)\n'
            "benchC = request.security(benchSym, timeframe.period, close, lookahead=barmerge.lookahead_off)")


def to_pine(fam, p, direction, title, symbol, tf, cost_pct, leverage=1.0, stats_note="", table_rows=None, pip=None, tf_min=None,
            bench_vals=None):
    if E.uses_deriv(fam, p):
        raise NoPine(PINE_DERIV_MSG)
    p = E.full_params(p)
    name = E.FAMILIES[fam]["name"]
    desc = E.describe(fam, p, direction)
    pip = float(pip) if pip else 1.0
    tf_min = int(tf_min) if tf_min else int({"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}.get(tf, 60))
    head = f"""// {title}
// Dibuat oleh aplikasi backtest pribadi (non-komersial). Data asal: {symbol} {tf}.
// Logika: {desc}
// {stats_note}
// Catatan: sinyal dihitung saat candle tutup, order masuk di open candle berikutnya
// (sama seperti backtest di aplikasi). Pasang di chart {symbol} timeframe {tf}.
// Jam sesi / hari / tren timeframe besar dihitung dari waktu UTC candle (bukan zona waktu chart).
// Biaya (fee + slippage + spread) didekati sebagai komisi {cost_pct:.4f}% per sisi.
// Backtest bukan jaminan hasil di masa depan.
"""
    sig = _signals_block(fam, p)
    if "benchC" in sig:
        sig = _bench_line(bench_vals) + "\n" + sig
    hdr = _header_inputs(p, direction, leverage, pip, tf_min, cost_pct)
    notes = NOTES.replace("\\n", "\n")
    strat = head + notes + f"""
//@version=5
strategy("{name} - {symbol} {tf}", overlay=true, initial_capital=10000,
     default_qty_type=strategy.percent_of_equity, default_qty_value={100 * leverage:g},
     commission_type=strategy.commission.percent, commission_value={cost_pct:.4f},
     pyramiding=0, process_orders_on_close=false, margin_long={0 if leverage <= 1 else round(100 / leverage, 2):g}, margin_short={0 if leverage <= 1 else round(100 / leverage, 2):g})
{hdr}
{sig}
{SIM}
{ORDERS}
plot(pos != 0 ? stp : na, "Stop", color=color.red, style=plot.style_linebr)
plot(pos != 0 ? tgt : na, "Target", color=color.green, style=plot.style_linebr)
alertcondition(nxDir == 1, "Long", "{symbol} {tf}: sinyal LONG (masuk di open candle berikutnya)")
alertcondition(nxDir == -1, "Short", "{symbol} {tf}: sinyal SHORT (masuk di open candle berikutnya)")
alertcondition(ta.change(strategy.closedtrades) > 0, "Trade selesai", "{symbol} {tf}: posisi ditutup (sinyal/SL/TP) di {{{{close}}}}")
{_table(table_rows or [], True)}
"""
    indi = head + f"""//@version=5
indicator("{name} (indikator) - {symbol} {tf}", overlay=true, max_labels_count=500)
{hdr}
{sig}
{SIM}
plotshape(goLong, "Long", shape.labelup, location.belowbar, color.new(color.green, 0), text="Long", textcolor=color.white)
plotshape(goShort, "Short", shape.labeldown, location.abovebar, color.new(color.red, 0), text="Short", textcolor=color.white)
plotshape(hitTp1, "TP1", shape.diamond, location.abovebar, color.new(color.teal, 0), size=size.tiny)
plotshape(hitTp2, "TP2", shape.diamond, location.abovebar, color.new(color.aqua, 0), size=size.tiny)
if not na(tradePnl)
    label.new(bar_index, high, (exitStop ? "SL " : exitTp ? "TP " : exitTime ? "Waktu " : "Exit ") + str.tostring(tradePnl, "#.##") + "%",
         style=label.style_label_down, color=tradePnl >= 0 ? color.new(color.teal, 20) : color.new(color.maroon, 20),
         textcolor=color.white, size=size.small)
plot(pos != 0 ? stp : na, "Stop", color=color.red, style=plot.style_linebr)
plot(pos != 0 ? tgt : na, "Target", color=color.green, style=plot.style_linebr)

alertcondition(nxDir == 1, "Long", "{symbol} {tf}: sinyal LONG (masuk di open candle berikutnya)")
alertcondition(nxDir == -1, "Short", "{symbol} {tf}: sinyal SHORT (masuk di open candle berikutnya)")
alertcondition(exitSig or exitTime, "Exit", "{symbol} {tf}: posisi ditutup (sinyal / waktu / sesi)")
alertcondition(exitStop, "Kena stop loss", "{symbol} {tf}: posisi kena STOP LOSS di {{{{close}}}}")
alertcondition(exitTp, "Kena take profit", "{symbol} {tf}: posisi kena TAKE PROFIT di {{{{close}}}}")
alertcondition(hitTp1 or hitTp2, "TP sebagian kena", "{symbol} {tf}: sebagian posisi ambil untung")
alertcondition(goLong or goShort, "Posisi baru dibuka", "{symbol} {tf}: posisi baru dibuka di {{{{open}}}}")
{_table((table_rows or []) + [("Posisi sekarang", "__POS__")], False)}
"""
    return strat, indi
