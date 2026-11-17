// Parity: Python engine trades vs the exported Pine indicator() and strategy(), both run by PineTS.
import { PineTS } from 'pinets';
import fs from 'fs';
const files = fs.readdirSync('.').filter(f => /^case\d+\.json$/.test(f)).sort((a, b) => parseInt(a.slice(4)) - parseInt(b.slice(4)));
let allOk = true;
for (const f of files) {
  const c = JSON.parse(fs.readFileSync(f));
  const out = { case: f, fam: c.fam, sym: c.sym, py_trades: c.py.ent.length };
  try {
    const ctx = await new PineTS(c.candles).run(c.indicator);
    const g = (n) => ctx.plots[n].data.map(x => x.value);
    const xL = g('xL'), xS = g('xS'), xX = g('xX');
    const pyEnt = new Map(c.py.ent.map((e, i) => [e, c.py.dir[i]]));
    let pine = 0, match = 0, mism = [];
    for (let i = 0; i < xL.length; i++) for (const [arr, d] of [[xL, 1], [xS, -1]]) if (arr[i] === 1) { pine++; if (pyEnt.get(i) === d) match++; else mism.push(i); }
    const pyEx = new Set(c.py.ex); let pex = 0, exm = 0;
    for (let i = 0; i < xX.length; i++) if (xX[i] === 1) { pex++; if (pyEx.has(i)) exm++; }
    Object.assign(out, { ind_entries: pine, ind_entries_match: match, ind_exits: pex, ind_exits_match: exm, first_mism: mism.slice(0, 3) });
    if (!(pine === c.py.ent.length && match === pine)) allOk = false;
  } catch (e) { out.ind_error = String(e).slice(0, 300); allOk = false; }
  try {
    const s = await new PineTS(c.candles).run(c.strategy);
    const ct = s.strategy.closedtrades;
    const arr = Array.isArray(ct) ? ct : (ct?.data || ct?._trades || Object.values(ct));
    let same = 0, prices = 0;
    for (let i = 0; i < Math.min(arr.length, c.py.ent.length); i++) {
      const t = arr[i];
      const eb = t.entry_bar_index ?? t.entryBarIndex ?? t.entry_bar, xb = t.exit_bar_index ?? t.exitBarIndex ?? t.exit_bar;
      const ep = t.entry_price ?? t.entryPrice, xp = t.exit_price ?? t.exitPrice;
      if (eb === c.py.ent[i] && xb === c.py.ex[i]) same++;
      if (Math.abs(ep - c.py.ep[i]) / c.py.ep[i] < 1e-9 && Math.abs(xp - c.py.xp[i]) / c.py.xp[i] < 1e-6) prices++;
    }
    // the last Python trade may be "closed at end of data" while Pine keeps it open
    const pyClosed = c.py.reason.filter(r => r !== 3).length;
    Object.assign(out, { strat_closed: arr.length, py_closed: pyClosed, strat_same_bars: same, strat_same_prices: prices });
    if (!(arr.length === pyClosed && same >= pyClosed && prices >= pyClosed)) allOk = false;
  } catch (e) { out.strat_error = String(e).slice(0, 300); allOk = false; }
  console.log(JSON.stringify(out));
}
console.log(allOk ? 'ALL PARITY CHECKS PASSED' : 'SOME PARITY CHECKS FAILED');
