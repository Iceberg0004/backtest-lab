// Parity: Python engine trades vs the exported Pine indicator() and strategy(), both run by PineTS.
import { PineTS } from 'pinets';
import fs from 'fs';
const files = fs.readdirSync('.').filter(f => /^case\d+\.json$/.test(f)).sort((a, b) => parseInt(a.slice(4)) - parseInt(b.slice(4)));
let allOk = true, stratFail = 0, totalAmb = 0;
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
    const exp = c.strat_expect;
    const recs = arr.map(t => [t.entry_bar_index ?? t.entryBarIndex ?? t.entry_bar, t.exit_bar_index ?? t.exitBarIndex ?? t.exit_bar,
                               t.entry_price ?? t.entryPrice, t.exit_price ?? t.exitPrice]);
    recs.sort((a, b) => (a[1] - b[1]) || (a[0] - b[0]));
    let same = 0, prices = 0, firstBad = null, ambDiff = 0;
    for (let i = 0; i < Math.min(recs.length, exp.length); i++) {
      const [eb, xb, ep, xp] = recs[i], e = exp[i];
      const okB = eb === e[0] && xb === e[1], okP = Math.abs(ep - e[2]) / e[2] < 1e-9 && Math.abs(xp - e[3]) / e[3] < 1e-6;
      if (okB) same++; if (okP) prices++; else if (okB && e[4] === 1) { prices++; ambDiff++; }
      if ((!okB || (!okP && e[4] !== 1)) && firstBad === null) firstBad = { i, pine: recs[i], py: e };
    }
    Object.assign(out, { strat_closed: recs.length, py_expect: exp.length, strat_same_bars: same, strat_same_prices: prices, strat_first_bad: firstBad, strat_amb_candle_diffs: ambDiff });
    totalAmb += ambDiff;
    out.strat_ok = recs.length === exp.length && same === exp.length && prices === exp.length;
    if (!out.strat_ok) stratFail++;
  } catch (e) { out.strat_error = String(e).slice(0, 300); stratFail++; }
  console.log(JSON.stringify(out));
}
console.log(allOk ? 'INDICATOR PARITY PASSED (all cases)' : 'INDICATOR PARITY FAILED');
console.log(stratFail === 0 ? 'STRATEGY PARITY PASSED (all cases)' : `STRATEGY PARITY: ${stratFail} of ${files.length} cases differ`);
console.log(`strategy(): ${totalAmb} trade(s) differ only because SL and TP were both touched in one candle (TradingView guesses the order, the app assumes SL first)`);
console.log(allOk && stratFail === 0 ? 'ALL PARITY CHECKS PASSED' : 'SOME PARITY CHECKS FAILED');
