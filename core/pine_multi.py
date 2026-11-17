"""Pine v5 export for the multi-leg modes (where TradingView can represent them)."""
from __future__ import annotations
import re
from . import engine as E, position as PS
from .pine import to_pine as _rule_pine, NoPine

WHY_NOT = {
    "hedge": "Hedging tidak bisa diekspor: strategy() TradingView memakai satu posisi bersih (netting), jadi long dan short tidak bisa terbuka bersamaan.",
    "zone": "Zone recovery tidak bisa diekspor: butuh long dan short terbuka bersamaan (TradingView strategy() memakai netting).",
    "pyr": "Pyramiding belum diekspor: penambahan unit per ATR dengan stop bersama belum bisa dibuat sama persis di Pine. Gunakan strategi dasarnya dari tab Export.",
}
NOTE = {
    "dca": "strategy() dengan pyramiding: base order di open setelah sinyal, safety order sebagai limit order, take profit dari harga rata-rata. "
           "Belum dicek otomatis dengan PineTS seperti strategi biasa: safety order baru aktif mulai candle setelah base order terisi, "
           "jadi hasilnya bisa sedikit beda dari aplikasi.",
    "mart": "strategy() biasa + ukuran posisi martingale. Langkah naik setelah trade rugi (anti: setelah untung). TradingView menghitung "
            "ambil untung sebagian (TP1) sebagai trade terpisah, jadi urutan langkah bisa beda kalau TP1 dipakai. Margin call ikut aturan TradingView.",
    "grid": "Grid diekspor sebagai indikator: garis level grid dan alert saat harga turun/naik melewati level. Strategy() TradingView tidak bisa "
            "melacak tiap slot grid (posisinya netting), jadi angka backtest grid hanya dari aplikasi.",
}


def to_pine(mode, p, symbol, tf, last_price, cost_pct, lev=10.0):
    if mode == "dca":
        return _dca(p, symbol, tf, cost_pct, lev)
    if mode == "mart":
        return _mart(p, symbol, tf, cost_pct, lev)
    if mode == "grid":
        return _grid(p, symbol, tf, last_price)
    return None


def _head(title, symbol, tf, extra):
    return (f"// {title}\n// Dibuat oleh aplikasi backtest pribadi (non-komersial). Data asal: {symbol} {tf}.\n// {extra}\n"
            "// Backtest bukan jaminan hasil di masa depan. Strategi ini punya risiko ekor (bisa kena margin call).\n")


def _dca(p, symbol, tf, cost_pct, lev):
    long = p["ddir"] == "long"
    start = {"always": "true",
             "rsi": ("ta.crossover(ta.rsi(close, 14), 30)" if long else "ta.crossunder(ta.rsi(close, 14), 70)"),
             "trend": ("close > ta.ema(close, 200)" if long else "close < ta.ema(close, 200)")}[p["start"]]
    m = round(100 / lev, 2) if lev > 1 else 100
    return _head("DCA / safety order bot", symbol, tf, NOTE["dca"]) + f"""//@version=5
strategy("DCA bot - {symbol} {tf}", overlay=true, initial_capital=10000, pyramiding={int(p['n_so']) + 1},
     commission_type=strategy.commission.percent, commission_value={cost_pct:.4f}, process_orders_on_close=false,
     margin_long={m:g}, margin_short={m:g})
isLong = {str(long).lower()}
baseP = input.float({p['base']}, 'Base order (% modal awal)')
soP = input.float({p['so_size']}, 'Safety order pertama (% modal awal)')
nSO = input.int({int(p['n_so'])}, 'Jumlah safety order')
devP = input.float({p['dev']}, 'Jarak SO pertama (%)')
sScale = input.float({p['sscale']}, 'Step scale')
vScale = input.float({p['vscale']}, 'Volume scale')
tpP = input.float({p['tp']}, 'Take profit dari harga rata-rata (%)')
slP = input.float({p['sl']}, 'Stop loss dari harga rata-rata (%), 0 = tidak')
startCond = {start}
var float basePx = na
inPos = strategy.position_size != 0
if not inPos
    basePx := na
    strategy.cancel_all()
    if startCond
        strategy.entry("Base", isLong ? strategy.long : strategy.short, qty = 10000 * baseP / 100 / close)
else
    if na(basePx)
        basePx := strategy.position_avg_price
    nf = strategy.opentrades - 1
    cum = 0.0
    step = devP
    for k = 0 to nSO - 1
        cum += step
        lvl = isLong ? basePx * (1 - cum / 100) : basePx * (1 + cum / 100)
        if k >= nf
            strategy.entry("SO" + str.tostring(k), isLong ? strategy.long : strategy.short, qty = 10000 * soP / 100 * math.pow(vScale, k) / lvl, limit = lvl)
        step *= sScale
    avg = strategy.position_avg_price
    tpx = isLong ? avg * (1 + tpP / 100) : avg * (1 - tpP / 100)
    slx = slP > 0 ? (isLong ? avg * (1 - slP / 100) : avg * (1 + slP / 100)) : na
    strategy.exit("TP/SL", stop = slx, limit = tpx)
plot(strategy.position_size != 0 ? strategy.position_avg_price : na, 'Harga rata-rata', color=color.orange, style=plot.style_linebr)
"""


def _mart(p, symbol, tf, cost_pct, lev):
    fp = PS.fam_params(p["fam"], p["fparams"])
    try:
        strat, _ = _rule_pine(p["fam"], fp, p.get("mdir", "both"), f"{'Anti-martingale' if p['anti'] else 'Martingale'} · {E.FAMILIES[p['fam']]['name']}",
                              symbol, tf, cost_pct, leverage=lev)
    except NoPine:
        return None
    block = f"""// ---- ukuran posisi {'anti-martingale' if p['anti'] else 'martingale'}
mgBase = input.float({p['base']}, 'Ukuran awal (% modal awal)')
mgMult = input.float({p['mult']}, 'Pengali ukuran')
mgMax = input.int({int(p['max_steps'])}, 'Langkah maks (lalu reset)')
mgAnti = {str(bool(p['anti'])).lower()}
var int mgStep = 0
var int mgSeen = 0
if strategy.closedtrades > mgSeen
    mgWin = strategy.closedtrades.profit(strategy.closedtrades - 1) > 0
    if (not mgAnti and not mgWin) or (mgAnti and mgWin)
        mgStep := mgStep < mgMax ? mgStep + 1 : 0
    else
        mgStep := 0
    mgSeen := strategy.closedtrades
mgQty = 10000 * mgBase / 100 * math.pow(mgMult, mgStep) / close
"""
    strat = strat.replace("// ---- sinyal dasar", block + "// ---- sinyal dasar", 1)
    strat = strat.replace('strategy.entry("Long", strategy.long)', 'strategy.entry("Long", strategy.long, qty = mgQty)')
    strat = strat.replace('strategy.entry("Short", strategy.short)', 'strategy.entry("Short", strategy.short, qty = mgQty)')
    strat = strat.replace("// Backtest bukan jaminan", "// " + NOTE["mart"] + "\n// Backtest bukan jaminan", 1)
    return strat


def _grid(p, symbol, tf, x):
    lo, hi = x * (1 - p["lo_pct"] / 100), x * (1 + p["hi_pct"] / 100)
    return _head("Level grid", symbol, tf, NOTE["grid"] + f" Batas dihitung dari harga terakhir {x:g} saat export.") + f"""//@version=5
indicator("Grid levels - {symbol} {tf}", overlay=true, max_lines_count=500)
lo = input.float({lo:.6g}, 'Batas bawah')
hi = input.float({hi:.6g}, 'Batas atas')
n = input.int({int(p['grids'])}, 'Jumlah grid')
geo = input.bool({str(bool(p['geo'])).lower()}, 'Geometris (% sama)')
lvl(k) => geo ? lo * math.pow(hi / lo, k / n) : lo + (hi - lo) * k / n
var line[] ls = array.new_line()
if barstate.islast and array.size(ls) == 0
    for k = 0 to n
        array.push(ls, line.new(bar_index - 1, lvl(k), bar_index, lvl(k), extend=extend.both, color=color.new(color.gray, 40)))
bool gDn = false
bool gUp = false
for k = 0 to n
    L = lvl(k)
    if low <= L and close[1] > L
        gDn := true
    if high >= L and close[1] < L
        gUp := true
plotshape(gDn, 'Turun ke level grid', shape.triangleup, location.belowbar, color.green, size=size.tiny)
plotshape(gUp, 'Naik ke level grid', shape.triangledown, location.abovebar, color.red, size=size.tiny)
alertcondition(gDn, 'Grid: harga turun ke level', 'Harga menyentuh level grid dari atas (grid long: beli)')
alertcondition(gUp, 'Grid: harga naik ke level', 'Harga menyentuh level grid dari bawah (grid long: jual)')
"""
