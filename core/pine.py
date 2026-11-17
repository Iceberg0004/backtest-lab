"""Export a found strategy to TradingView Pine Script v5 (strategy + indicator).
Indicator definitions and the order/stop logic mirror core/engine.py."""
from . import engine as E


def _signals_block(fam, p):
    t = p.get("trend", 0)
    L = []
    if t:
        L += [f"trendMa = ta.ema(close, {t})", "trendL = close > trendMa", "trendS = close < trendMa",
              "plot(trendMa, 'Trend EMA', color=color.gray)"]
    else:
        L += ["trendL = true", "trendS = true"]
    if fam == "ma_cross":
        f = "ta.ema" if p["ma_type"] == "EMA" else "ta.sma"
        L += [f"fastMa = {f}(close, {p['fast']})", f"slowMa = {f}(close, {p['slow']})",
              "up = ta.crossover(fastMa, slowMa)", "dn = ta.crossunder(fastMa, slowMa)",
              "longEntry = up and trendL", "longExit = dn", "shortEntry = dn and trendS", "shortExit = up",
              f"plot(fastMa, '{p['ma_type']} {p['fast']}', color=color.orange)",
              f"plot(slowMa, '{p['ma_type']} {p['slow']}', color=color.blue)"]
    elif fam == "rsi_mr":
        L += [f"r = ta.rsi(close, {p['rsi_len']})",
              f"longEntry = ta.crossover(r, {p['lower']}) and trendL", f"longExit = ta.crossover(r, {p['exit_level']})",
              f"shortEntry = ta.crossunder(r, {p['upper']}) and trendS", f"shortExit = ta.crossunder(r, {100 - p['exit_level']})"]
    elif fam == "donchian":
        L += [f"hh = ta.highest(high, {p['entry_len']})[1]", f"ll = ta.lowest(low, {p['entry_len']})[1]",
              f"xh = ta.highest(high, {p['exit_len']})[1]", f"xl = ta.lowest(low, {p['exit_len']})[1]",
              "longEntry = close > hh and trendL", "longExit = close < xl",
              "shortEntry = close < ll and trendS", "shortExit = close > xh",
              "plot(hh, 'Donchian high', color=color.teal)", "plot(ll, 'Donchian low', color=color.teal)"]
    elif fam == "bb_squeeze":
        L += [f"basis = ta.sma(close, {p['bb_len']})", f"dev = {p['bb_mult']} * ta.stdev(close, {p['bb_len']})",
              "upper = basis + dev", "lower = basis - dev", "bbw = (upper - lower) / basis",
              f"thr = ta.sma(bbw, {p['sq_len']}) * {p['sq_factor']}", "sq = bbw < thr ? 1.0 : 0.0",
              "recent = nz(ta.highest(sq, 5)[1]) == 1",
              "longEntry = ta.crossover(close, upper) and recent", "longExit = ta.crossunder(close, basis)",
              "shortEntry = ta.crossunder(close, lower) and recent", "shortExit = ta.crossover(close, basis)",
              "plot(upper, 'BB upper', color=color.purple)", "plot(basis, 'BB basis', color=color.gray)",
              "plot(lower, 'BB lower', color=color.purple)"]
    elif fam == "macd":
        L += [f"macdLine = ta.ema(close, {p['fast']}) - ta.ema(close, {p['slow']})",
              f"sigLine = ta.ema(macdLine, {p['signal']})",
              "up = ta.crossover(macdLine, sigLine)", "dn = ta.crossunder(macdLine, sigLine)",
              "longEntry = up and trendL", "longExit = dn", "shortEntry = dn and trendS", "shortExit = up"]
    elif fam == "supertrend":
        L += [f"stAtr = ta.atr({p['st_len']})", f"stBu = hl2 - {p['st_mult']} * stAtr", f"stBd = hl2 + {p['st_mult']} * stAtr",
              "var float stUp = na", "var float stDn = na", "var int stDir = 0",
              "if not na(stAtr)",
              "    stUp := na(stUp) or close[1] <= stUp ? stBu : math.max(stBu, stUp)",
              "    stDn := na(stDn) or close[1] >= stDn ? stBd : math.min(stBd, stDn)",
              "    stDir := stDir == 0 ? 1 : stDir == 1 and close < stUp ? -1 : stDir == -1 and close > stDn ? 1 : stDir",
              "flipUp = stDir == 1 and stDir[1] == -1", "flipDn = stDir == -1 and stDir[1] == 1",
              "longEntry = flipUp and trendL", "longExit = flipDn", "shortEntry = flipDn and trendS", "shortExit = flipUp",
              "plot(stDir == 1 ? stUp : stDir == -1 ? stDn : na, 'Supertrend', color=stDir == 1 ? color.green : color.red, style=plot.style_linebr)"]
    elif fam == "keltner":
        L += [f"kcBasis = ta.ema(close, {p['kc_len']})", f"kcUp = kcBasis + {p['kc_mult']} * atr", f"kcLo = kcBasis - {p['kc_mult']} * atr",
              "longEntry = ta.crossover(close, kcUp) and trendL", "longExit = ta.crossunder(close, kcBasis)",
              "shortEntry = ta.crossunder(close, kcLo) and trendS", "shortExit = ta.crossover(close, kcBasis)",
              "plot(kcUp, 'KC upper', color=color.teal)", "plot(kcBasis, 'KC basis', color=color.gray)",
              "plot(kcLo, 'KC lower', color=color.teal)"]
    elif fam == "stochrsi":
        w = p["rsi_len"] + p["stoch_len"] + 3 + 1
        L += [f"srR = ta.rsi(close, {p['rsi_len']})", f"srH = ta.highest(srR, {p['stoch_len']})", f"srL = ta.lowest(srR, {p['stoch_len']})",
              "srRng = srH - srL", "srSt = srRng == 0 ? na : 100 * (srR - srL) / srRng", "srK0 = ta.sma(srSt, 3)",
              f"srK = bar_index < {w} ? na : srK0",
              f"longEntry = ta.crossover(srK, {p['k_lo']}) and trendL", f"longExit = ta.crossunder(srK, {p['k_hi']})",
              f"shortEntry = ta.crossunder(srK, {p['k_hi']}) and trendS", f"shortExit = ta.crossover(srK, {p['k_lo']})"]
    elif fam == "zscore":
        L += [f"zM = ta.sma(close, {p['z_len']})", f"zSd = ta.stdev(close, {p['z_len']})",
              f"zV = bar_index < {p['z_len']} or zSd == 0 ? na : (close - zM) / zSd",
              f"longEntry = ta.crossover(zV, -{p['z_entry']}) and trendL", f"longExit = ta.crossover(zV, {p['z_exit']})",
              f"shortEntry = ta.crossunder(zV, {p['z_entry']}) and trendS", f"shortExit = ta.crossunder(zV, -{p['z_exit']})",
              f"plot(zM, 'Z-score mean {p['z_len']}', color=color.gray)"]
    rg = p.get("regime", 0)
    if rg:
        op = ">=" if rg == 1 else "<"
        L += ["// filter kondisi pasar (ADX 14): " + ("hanya saat trending" if rg == 1 else "hanya saat sideways"),
              "dUp = high - high[1]", "dDn = low[1] - low",
              "pDM = na(dUp) ? na : (dUp > dDn and dUp > 0 ? dUp : 0.0)",
              "mDM = na(dDn) ? na : (dDn > dUp and dDn > 0 ? dDn : 0.0)",
              "pDI = 100 * ta.rma(pDM, 14) / atr", "mDI = 100 * ta.rma(mDM, 14) / atr", "diSum = pDI + mDI",
              "adxV = 100 * ta.rma(math.abs(pDI - mDI) / (diSum == 0 ? 1 : diSum), 14)",
              f"regimeOk = adxV {op} {p['adx_thr']}",
              "longEntry := longEntry and regimeOk", "shortEntry := shortEntry and regimeOk"]
    return "\n".join(L)


def _header_inputs(p, direction):
    return f"""allowLong  = input.bool({str(direction in ('long', 'both')).lower()}, 'Izinkan Long')
allowShort = input.bool({str(direction in ('short', 'both')).lower()}, 'Izinkan Short')
slK = input.float({p['sl_atr']}, 'Stop loss (x ATR)')
tpK = input.float({p['tp_atr']}, 'Take profit (x ATR, 0 = tidak ada)')
trK = input.float({p['trail_atr']}, 'Trailing stop (x ATR, 0 = tidak ada)')
atr = ta.atr(14)"""


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


def to_pine(fam, p, direction, title, symbol, tf, cost_pct, leverage=1.0, stats_note="", table_rows=None):
    name = E.FAMILIES[fam]["name"]
    desc = E.describe(fam, p, direction)
    head = f"""// {title}
// Dibuat oleh aplikasi backtest pribadi (non-komersial). Data asal: {symbol} {tf}.
// Logika: {desc}
// {stats_note}
// Catatan: sinyal dihitung saat candle tutup, order masuk di open candle berikutnya
// (sama seperti backtest di aplikasi). Pasang di chart {symbol} timeframe {tf}.
// Biaya (fee + slippage + spread) didekati sebagai komisi {cost_pct:.4f}% per sisi.
// Backtest bukan jaminan hasil di masa depan.
"""
    sig = _signals_block(fam, p)
    strat = head + f"""//@version=5
strategy("{name} - {symbol} {tf}", overlay=true, initial_capital=10000,
     default_qty_type=strategy.percent_of_equity, default_qty_value={100 * leverage:g},
     commission_type=strategy.commission.percent, commission_value={cost_pct:.4f},
     pyramiding=0, process_orders_on_close=false, margin_long={0 if leverage <= 1 else round(100 / leverage, 2):g}, margin_short={0 if leverage <= 1 else round(100 / leverage, 2):g})
{_header_inputs(p, direction)}
{sig}

var float lStop = na
var float lTp = na
var float sStop = na
var float sTp = na

if strategy.position_size > 0
    if allowShort and shortEntry
        strategy.entry("Short", strategy.short)
        sStop := close + slK * atr
        sTp := tpK > 0 ? close - tpK * atr : na
    else if longExit
        strategy.close("Long", comment="Exit")
else if strategy.position_size < 0
    if allowLong and longEntry
        strategy.entry("Long", strategy.long)
        lStop := close - slK * atr
        lTp := tpK > 0 ? close + tpK * atr : na
    else if shortExit
        strategy.close("Short", comment="Exit")
else
    if allowLong and longEntry
        strategy.entry("Long", strategy.long)
        lStop := close - slK * atr
        lTp := tpK > 0 ? close + tpK * atr : na
    else if allowShort and shortEntry
        strategy.entry("Short", strategy.short)
        sStop := close + slK * atr
        sTp := tpK > 0 ? close - tpK * atr : na

// trailing stop diperbarui saat candle tutup, berlaku mulai candle berikutnya
if strategy.position_size > 0 and trK > 0
    lStop := math.max(lStop, high - trK * atr)
if strategy.position_size < 0 and trK > 0
    sStop := math.min(sStop, low + trK * atr)

strategy.exit("L-SL/TP", "Long", stop=lStop, limit=lTp)
strategy.exit("S-SL/TP", "Short", stop=sStop, limit=sTp)

plotshape(allowLong and longEntry and strategy.position_size <= 0, "Sinyal Long", shape.labelup, location.belowbar, color.new(color.green, 0), text="Long", textcolor=color.white)
plotshape(allowShort and shortEntry and strategy.position_size >= 0, "Sinyal Short", shape.labeldown, location.abovebar, color.new(color.red, 0), text="Short", textcolor=color.white)
alertcondition(allowLong and longEntry, "Long", "{symbol} {tf}: sinyal LONG")
alertcondition(allowShort and shortEntry, "Short", "{symbol} {tf}: sinyal SHORT")
alertcondition(longExit or shortExit, "Exit", "{symbol} {tf}: sinyal EXIT")
alertcondition(ta.change(strategy.closedtrades) > 0, "Trade selesai", "{symbol} {tf}: posisi ditutup (sinyal/SL/TP) di {{{{close}}}}")
{_table(table_rows or [], True)}
"""
    indi = head + f"""//@version=5
indicator("{name} (indikator) - {symbol} {tf}", overlay=true)
{_header_inputs(p, direction)}
{sig}

// Simulasi posisi yang sama persis dengan versi strategy(): sinyal candle sebelumnya dieksekusi di open,
// lalu cek stop/target dengan high/low (stop dianggap kena lebih dulu), lalu trailing di close.
var int pos = 0
var float ep = na
var float stp = na
var float tgt = na
bool goLong = false
bool goShort = false
bool exitSig = false
bool exitStop = false
bool exitTp = false
float tradePnl = na

if pos == 1
    if allowShort and shortEntry[1]
        tradePnl := (open - ep) / ep * 100
        goShort := true
    else if longExit[1]
        tradePnl := (open - ep) / ep * 100
        exitSig := true
        pos := 0
else if pos == -1
    if allowLong and longEntry[1]
        tradePnl := (ep - open) / ep * 100
        goLong := true
    else if shortExit[1]
        tradePnl := (ep - open) / ep * 100
        exitSig := true
        pos := 0
else
    if allowLong and longEntry[1]
        goLong := true
    else if allowShort and shortEntry[1]
        goShort := true

if goLong
    pos := 1
    ep := open
    stp := close[1] - slK * atr[1]
    tgt := tpK > 0 ? close[1] + tpK * atr[1] : na
if goShort
    pos := -1
    ep := open
    stp := close[1] + slK * atr[1]
    tgt := tpK > 0 ? close[1] - tpK * atr[1] : na

if pos == 1
    if low <= stp
        tradePnl := (math.min(open, stp) - ep) / ep * 100
        exitStop := true
        pos := 0
    else if tpK > 0 and high >= tgt
        tradePnl := (math.max(open, tgt) - ep) / ep * 100
        exitTp := true
        pos := 0
else if pos == -1
    if high >= stp
        tradePnl := (ep - math.max(open, stp)) / ep * 100
        exitStop := true
        pos := 0
    else if tpK > 0 and low <= tgt
        tradePnl := (ep - math.min(open, tgt)) / ep * 100
        exitTp := true
        pos := 0

if pos == 1 and trK > 0
    stp := math.max(stp, high - trK * atr)
if pos == -1 and trK > 0
    stp := math.min(stp, low + trK * atr)

plotshape(goLong, "Long", shape.labelup, location.belowbar, color.new(color.green, 0), text="Long", textcolor=color.white)
plotshape(goShort, "Short", shape.labeldown, location.abovebar, color.new(color.red, 0), text="Short", textcolor=color.white)
if not na(tradePnl)
    label.new(bar_index, high, (exitStop ? "SL " : exitTp ? "TP " : "Exit ") + str.tostring(tradePnl, "#.##") + "%",
         style=label.style_label_down, color=tradePnl >= 0 ? color.new(color.teal, 20) : color.new(color.maroon, 20),
         textcolor=color.white, size=size.small)
plot(pos != 0 ? stp : na, "Stop", color=color.red, style=plot.style_linebr)
plot(pos != 0 and tpK > 0 ? tgt : na, "Target", color=color.green, style=plot.style_linebr)

alertcondition(allowLong and longEntry, "Long", "{symbol} {tf}: sinyal LONG (masuk di open candle berikutnya)")
alertcondition(allowShort and shortEntry, "Short", "{symbol} {tf}: sinyal SHORT (masuk di open candle berikutnya)")
alertcondition((pos[1] == 1 and longExit) or (pos[1] == -1 and shortExit), "Exit", "{symbol} {tf}: sinyal EXIT")
alertcondition(exitStop, "Kena stop loss", "{symbol} {tf}: posisi kena STOP LOSS di {{{{close}}}}")
alertcondition(exitTp, "Kena take profit", "{symbol} {tf}: posisi kena TAKE PROFIT di {{{{close}}}}")
alertcondition(goLong or goShort, "Posisi baru dibuka", "{symbol} {tf}: posisi baru dibuka di {{{{open}}}}")
{_table((table_rows or []) + [("Posisi sekarang", "__POS__")], False)}
"""
    return strat, indi
