// Runs the multi-leg Pine exports in PineTS and compares DCA deals with the Python engine.
import { PineTS } from 'pinets';
import fs from 'fs';
const c = JSON.parse(fs.readFileSync('multi_case.json'));
for (const k of ['dca', 'mart', 'grid']) {
  try {
    const r = await new PineTS(c.candles).run(c[k]);
    const ct = r.strategy ? r.strategy.closedtrades : null;
    const arr = ct ? (Array.isArray(ct) ? ct : (ct?.data || ct?._trades || Object.values(ct))) : [];
    if (k === 'dca') {
      // group closed trades (one per filled entry) into deals by exit bar
      const exits = [...new Set(arr.map(t => t.exit_bar_index ?? t.exitBarIndex ?? t.exit_bar))].sort((a, b) => a - b);
      const py = c.py_dca.exit_bars;
      const same = exits.filter(x => py.includes(x)).length;
      console.log(JSON.stringify({ mode: k, pine_fills_closed: arr.length, pine_deals: exits.length, py_deals: py.length, same_exit_bars: same,
                                   first_pine: exits.slice(0, 8), first_py: py.slice(0, 8) }));
    } else if (k === 'mart') {
      const q = arr.map(t => t.size ?? t.qty ?? t.quantity).slice(0, 12);
      console.log(JSON.stringify({ mode: k, closed: arr.length, first_sizes: q }));
    } else {
      console.log(JSON.stringify({ mode: k, ran: true, plots: Object.keys(r.plots || {}) }));
    }
  } catch (e) { console.log(JSON.stringify({ mode: k, error: String(e).slice(0, 400) })); }
}
