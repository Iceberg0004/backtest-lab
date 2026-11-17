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


def _header_inputs(p, direction):
    p = E.full_params(p)
    pct = p["sl_mode"] == 1
    sl, tp = (p["sl_pct"], p["tp_pct"]) if pct else (p["sl_atr"], p["tp_atr"])
    u = "% harga" if pct else "x ATR"
    return f"""allowLong  = input.bool({str(direction in ('long', 'both')).lower()}, 'Izinkan Long')
allowShort = input.bool({str(direction in ('short', 'both')).lower()}, 'Izinkan Short')
slK = input.float({R_(float(sl))}, 'Stop loss ({u})')
tpK = input.float({R_(float(tp))}, 'Take profit ({u}, 0 = tidak ada)')
trK = input.float({R_(float(p['trail_atr']))}, 'Trailing stop (x ATR, 0 = tidak ada)')
trMode = input.int({int(p['trail_mode'])}, 'Jenis trailing (0 = dari high/low candle, 1 = chandelier)')
tp1R = input.float({R_(float(p['tp1_r']))}, 'Ambil untung 50% di ... R (0 = tidak)')
beR = input.float({R_(float(p['be_r']))}, 'Stop ke harga masuk setelah ... R (0 = tidak)')
maxBars = input.int({int(p['max_bars'])}, 'Keluar paksa setelah ... candle (0 = tidak)')
atr = ta.atr(14)
unitV = {"close * 0.01" if pct else "atr"}
tp1K = tp1R * slK
beK = beR * slK"""


NOTES = ("// Catatan versi strategy(): TradingView memakai mesin ordernya sendiri. Kalau dalam SATU candle stop loss\\n"
         "// dan take profit sama-sama tersentuh, TradingView menebak urutan dari open/high/low, sedangkan aplikasi\\n"
         "// menganggap stop loss duluan. Ambil untung 50% (TP1) memakai qty_percent=50. Karena itu angka strategy()\\n"
         "// bisa sedikit beda dari aplikasi. Versi indicator() menghitung posisi sendiri dan sama persis dengan aplikasi.")


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


def to_pine(fam, p, direction, title, symbol, tf, cost_pct, leverage=1.0, stats_note="", table_rows=None):
    if E.uses_deriv(fam, p):
        raise NoPine(PINE_DERIV_MSG)
    p = E.full_params(p)
    name = E.FAMILIES[fam]["name"]
    desc = E.describe(fam, p, direction)
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
    notes = NOTES.replace("\\n", "\n")
    strat = head + notes + f"""
//@version=5
strategy("{name} - {symbol} {tf}", overlay=true, initial_capital=10000,
     default_qty_type=strategy.percent_of_equity, default_qty_value={100 * leverage:g},
     commission_type=strategy.commission.percent, commission_value={cost_pct:.4f},
     pyramiding=0, process_orders_on_close=false, margin_long={0 if leverage <= 1 else round(100 / leverage, 2):g}, margin_short={0 if leverage <= 1 else round(100 / leverage, 2):g})
{_header_inputs(p, direction)}
{sig}

var float lStop = na
var float lTp = na
var float lTp1 = na
var float lBe = na
var float sStop = na
var float sTp = na
var float sTp1 = na
var float sBe = na
var float hhT = na
var float llT = na
var bool beDone = false
var float fullQty = 0.0
var bool p1Done = false

// 1) candle tutup: update posisi yang sedang berjalan (TP1 sudah kena?, chandelier, trailing, breakeven)
if strategy.position_size != 0
    if strategy.opentrades.entry_bar_index(0) == bar_index and not p1Done and fullQty == 0.0
        fullQty := math.abs(strategy.position_size)
    if fullQty > 0 and math.abs(strategy.position_size) < fullQty * 0.99
        p1Done := true
if strategy.position_size > 0
    hhT := na(hhT) ? high : math.max(hhT, high)
    if trK > 0 and not na(atr)
        lStop := math.max(lStop, (trMode == 1 ? hhT : high) - trK * atr)
    if beK > 0 and not beDone and high >= lBe
        lStop := math.max(lStop, strategy.position_avg_price)
        beDone := true
if strategy.position_size < 0
    llT := na(llT) ? low : math.min(llT, low)
    if trK > 0 and not na(atr)
        sStop := math.min(sStop, (trMode == 1 ? llT : low) + trK * atr)
    if beK > 0 and not beDone and low <= sBe
        sStop := math.min(sStop, strategy.position_avg_price)
        beDone := true

// 2) sinyal candle ini -> order di open candle berikutnya
goL = false
goS = false
if strategy.position_size > 0
    if allowShort and sigSE
        goS := true
    else if sigLX
        strategy.close("Long", comment="Exit")
    else if maxBars > 0 and bar_index - strategy.opentrades.entry_bar_index(0) + 1 >= maxBars
        strategy.close("Long", comment="Waktu habis")
else if strategy.position_size < 0
    if allowLong and sigLE
        goL := true
    else if sigSX
        strategy.close("Short", comment="Exit")
    else if maxBars > 0 and bar_index - strategy.opentrades.entry_bar_index(0) + 1 >= maxBars
        strategy.close("Short", comment="Waktu habis")
else
    if allowLong and sigLE
        goL := true
    else if allowShort and sigSE
        goS := true
if goL and not na(atr) and not na(unitV)
    strategy.entry("Long", strategy.long)
    lStop := close - slK * unitV
    lTp := tpK > 0 ? close + tpK * unitV : na
    lTp1 := tp1K > 0 ? close + tp1K * unitV : na
    lBe := beK > 0 ? close + beK * unitV : na
    hhT := na
    beDone := false
    p1Done := false
    fullQty := 0.0
else if goL and strategy.position_size < 0
    strategy.close("Short", comment="Exit")
if goS and not na(atr) and not na(unitV)
    strategy.entry("Short", strategy.short)
    sStop := close + slK * unitV
    sTp := tpK > 0 ? close - tpK * unitV : na
    sTp1 := tp1K > 0 ? close - tp1K * unitV : na
    sBe := beK > 0 ? close - beK * unitV : na
    llT := na
    beDone := false
    p1Done := false
    fullQty := 0.0
else if goS and strategy.position_size > 0
    strategy.close("Long", comment="Exit")

// 3) stop / target (aktif mulai candle berikutnya)
if tp1K > 0 and not p1Done
    strategy.exit("L-TP1", "Long", qty_percent=50, stop=lStop, limit=lTp1)
    strategy.exit("S-TP1", "Short", qty_percent=50, stop=sStop, limit=sTp1)
strategy.exit("L-SL/TP", "Long", stop=lStop, limit=lTp)
strategy.exit("S-SL/TP", "Short", stop=sStop, limit=sTp)

plotshape(allowLong and sigLE and strategy.position_size <= 0, "Sinyal Long", shape.labelup, location.belowbar, color.new(color.green, 0), text="Long", textcolor=color.white)
plotshape(allowShort and sigSE and strategy.position_size >= 0, "Sinyal Short", shape.labeldown, location.abovebar, color.new(color.red, 0), text="Short", textcolor=color.white)
alertcondition(allowLong and sigLE, "Long", "{symbol} {tf}: sinyal LONG")
alertcondition(allowShort and sigSE, "Short", "{symbol} {tf}: sinyal SHORT")
alertcondition(sigLX or sigSX, "Exit", "{symbol} {tf}: sinyal EXIT")
alertcondition(ta.change(strategy.closedtrades) > 0, "Trade selesai", "{symbol} {tf}: posisi ditutup (sinyal/SL/TP) di {{{{close}}}}")
{_table(table_rows or [], True)}
"""
    indi = head + f"""//@version=5
indicator("{name} (indikator) - {symbol} {tf}", overlay=true, max_labels_count=500)
{_header_inputs(p, direction)}
{sig}

// Simulasi posisi yang sama persis dengan aplikasi: sinyal candle sebelumnya dieksekusi di open,
// lalu cek stop / TP1 / target dengan high/low (stop dianggap kena lebih dulu), lalu trailing & breakeven di close.
var int pos = 0
var float ep = na
var float stp = na
var float tgt = na
var float tg1 = na
var float bel = na
var float hhI = na
var float llI = na
var int ei = 0
var float qty = 1.0
var float realP = 0.0
var bool p1 = false
var bool beD = false
bool goLong = false
bool goShort = false
bool exitSig = false
bool exitStop = false
bool exitTp = false
bool exitTime = false
bool hitTp1 = false
float tradePnl = na
float px = na

timeUp = maxBars > 0 and pos != 0 and (bar_index - 1 - ei + 1) >= maxBars
if pos == 1
    if allowShort and sigSE[1]
        tradePnl := (realP + qty * (open - ep) / ep) * 100
        goShort := true
        pos := 0
    else if sigLX[1]
        tradePnl := (realP + qty * (open - ep) / ep) * 100
        exitSig := true
        pos := 0
    else if timeUp
        tradePnl := (realP + qty * (open - ep) / ep) * 100
        exitTime := true
        pos := 0
else if pos == -1
    if allowLong and sigLE[1]
        tradePnl := (realP + qty * (ep - open) / ep) * 100
        goLong := true
        pos := 0
    else if sigSX[1]
        tradePnl := (realP + qty * (ep - open) / ep) * 100
        exitSig := true
        pos := 0
    else if timeUp
        tradePnl := (realP + qty * (ep - open) / ep) * 100
        exitTime := true
        pos := 0
else
    if allowLong and sigLE[1]
        goLong := true
    else if allowShort and sigSE[1]
        goShort := true
goLong := goLong and not na(atr[1]) and not na(unitV[1])
goShort := goShort and not na(atr[1]) and not na(unitV[1])

if goLong or goShort
    pos := goLong ? 1 : -1
    ep := open
    ei := bar_index
    qty := 1.0
    realP := 0.0
    p1 := false
    beD := false
    hhI := na
    llI := na
    stp := close[1] - pos * slK * unitV[1]
    tgt := tpK > 0 ? close[1] + pos * tpK * unitV[1] : na
    tg1 := tp1K > 0 ? close[1] + pos * tp1K * unitV[1] : na
    bel := beK > 0 ? close[1] + pos * beK * unitV[1] : na

if pos != 0
    hs = pos == 1 ? low <= stp : high >= stp
    h1 = not p1 and tp1K > 0 and (pos == 1 ? high >= tg1 : low <= tg1)
    ht = tpK > 0 and (pos == 1 ? high >= tgt : low <= tgt)
    if hs
        px := pos == 1 ? math.min(open, stp) : math.max(open, stp)
        tradePnl := (realP + qty * pos * (px - ep) / ep) * 100
        exitStop := true
        pos := 0
    else
        if h1
            p1px = pos == 1 ? math.max(open, tg1) : math.min(open, tg1)
            realP := realP + 0.5 * pos * (p1px - ep) / ep
            qty := 0.5
            p1 := true
            hitTp1 := true
        if ht
            px := pos == 1 ? math.max(open, tgt) : math.min(open, tgt)
            tradePnl := (realP + qty * pos * (px - ep) / ep) * 100
            exitTp := true
            pos := 0

if pos == 1
    hhI := na(hhI) ? high : math.max(hhI, high)
    if trK > 0 and not na(atr)
        stp := math.max(stp, (trMode == 1 ? hhI : high) - trK * atr)
    if beK > 0 and not beD and high >= bel
        stp := math.max(stp, ep)
        beD := true
if pos == -1
    llI := na(llI) ? low : math.min(llI, low)
    if trK > 0 and not na(atr)
        stp := math.min(stp, (trMode == 1 ? llI : low) + trK * atr)
    if beK > 0 and not beD and low <= bel
        stp := math.min(stp, ep)
        beD := true

plotshape(goLong, "Long", shape.labelup, location.belowbar, color.new(color.green, 0), text="Long", textcolor=color.white)
plotshape(goShort, "Short", shape.labeldown, location.abovebar, color.new(color.red, 0), text="Short", textcolor=color.white)
plotshape(hitTp1, "TP1 (50%)", shape.diamond, location.abovebar, color.new(color.teal, 0), size=size.tiny)
if not na(tradePnl)
    label.new(bar_index, high, (exitStop ? "SL " : exitTp ? "TP " : exitTime ? "Waktu " : "Exit ") + str.tostring(tradePnl, "#.##") + "%",
         style=label.style_label_down, color=tradePnl >= 0 ? color.new(color.teal, 20) : color.new(color.maroon, 20),
         textcolor=color.white, size=size.small)
plot(pos != 0 ? stp : na, "Stop", color=color.red, style=plot.style_linebr)
plot(pos != 0 and tpK > 0 ? tgt : na, "Target", color=color.green, style=plot.style_linebr)

alertcondition(allowLong and sigLE, "Long", "{symbol} {tf}: sinyal LONG (masuk di open candle berikutnya)")
alertcondition(allowShort and sigSE, "Short", "{symbol} {tf}: sinyal SHORT (masuk di open candle berikutnya)")
alertcondition(exitSig or exitTime, "Exit", "{symbol} {tf}: posisi ditutup (sinyal / waktu habis)")
alertcondition(exitStop, "Kena stop loss", "{symbol} {tf}: posisi kena STOP LOSS di {{{{close}}}}")
alertcondition(exitTp, "Kena take profit", "{symbol} {tf}: posisi kena TAKE PROFIT di {{{{close}}}}")
alertcondition(hitTp1, "TP1 kena", "{symbol} {tf}: 50% posisi ambil untung (TP1)")
alertcondition(goLong or goShort, "Posisi baru dibuka", "{symbol} {tf}: posisi baru dibuka di {{{{open}}}}")
{_table((table_rows or []) + [("Posisi sekarang", "__POS__")], False)}
"""
    return strat, indi
