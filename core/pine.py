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
    return "\n".join(L)


def _header_inputs(p, direction):
    return f"""allowLong  = input.bool({str(direction in ('long', 'both')).lower()}, 'Izinkan Long')
allowShort = input.bool({str(direction in ('short', 'both')).lower()}, 'Izinkan Short')
slK = input.float({p['sl_atr']}, 'Stop loss (x ATR)')
tpK = input.float({p['tp_atr']}, 'Take profit (x ATR, 0 = tidak ada)')
trK = input.float({p['trail_atr']}, 'Trailing stop (x ATR, 0 = tidak ada)')
atr = ta.atr(14)"""


def to_pine(fam, p, direction, title, symbol, tf, cost_pct, leverage=1.0, stats_note=""):
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
"""
    return strat, indi
