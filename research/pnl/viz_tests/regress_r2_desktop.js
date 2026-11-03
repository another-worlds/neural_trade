// Desktop regression checks for the BTCUSDT replay viewer, round-2 findings 0, 1, 5, 6, 11, 13, 14, 15, 17.
//   0      trade table is empty with a "Loading <name>..." note while a picked strategy loads; refilled on a failed pick
//   1, 15  readout box keeps one height while the mouse sweeps the plot at 701-960 px (incl. DejaVu Sans), label "(HH:MM)"
//   5      a click 1 px inside a non-minute-aligned edge of a 20-40 min view pins a minute inside the axis; it survives a redraw
//   6, 13  a pinned month's 1-minute load does not overwrite the hover readout; the pin still gets the exact close
//   11     double-click with 0-6 px movement and triple clicks reset and keep the earlier pin; single click and drag-then-click pin
//   14     a failed strategy load message stays in #res through background renders; the next view change or pick clears it
//   17     after chips, zoom bar, drag zoom, double-click and strategy switches the hover readout follows the pointer and view;
//          with the pointer off the plot a redraw shows the pin or the hint
// Requires only playwright and fs (Node >= 18 for fetch).
// Env: PLOTLY (required) path of plotly-2.34.0.min.js, served for **/plotly.min.js
//      PAGE   (default test.html) page under BASE;  BASE (default http://localhost:8765/)
//      SITE   (optional) site directory for ground-truth files; without it they are fetched from BASE (s/*.json, m/*.txt, meta.json, price_hour.bin)
// Prints one JSON line per check, then a summary line. Exit 1 when any non-advisory check fails, 2 on a setup error.
// Advisory checks (marked "advisory": true) document an open defect found in round 3 and do not affect the exit code.
'use strict';
const { chromium } = require('playwright');
const fs = require('fs');

const PLOTLY = process.env.PLOTLY;
if (!PLOTLY || !fs.existsSync(PLOTLY)) { console.error('PLOTLY must point to plotly-2.34.0.min.js'); process.exit(2); }
const BASE = process.env.BASE || 'http://localhost:8765/';
const PAGE = process.env.PAGE || 'test.html';
const SITE = process.env.SITE || '';
const URL = BASE + PAGE;
const PL = fs.readFileSync(PLOTLY);
const MIN = 60000;

const results = [];
function record(name, findings, pass, measured, advisory) {
  const r = { check: name, findings, pass: !!pass, measured };
  if (advisory) r.advisory = true;                // documents a known open defect; not counted in the exit code
  results.push(r); console.log(JSON.stringify(r));
}
async function check(name, findings, fn, advisory) {
  try { const { pass, measured } = await fn(); record(name, findings, pass, measured, advisory); }
  catch (e) { record(name, findings, false, { error: String(e && e.message || e).slice(0, 300) }, advisory); }
}
const wait = ms => new Promise(r => setTimeout(r, ms));
const ready = (p, t = 60000) => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: t });

// ---------- ground truth (site files)
async function siteFile(rel) {
  if (SITE) return fs.readFileSync(SITE.replace(/\/$/, '') + '/' + rel);
  const r = await fetch(BASE + rel); if (!r.ok) throw new Error(rel + ' ' + r.status);
  return Buffer.from(await r.arrayBuffer());
}
const ab = b => b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
const b64 = s => ab(Buffer.from(s, 'base64'));
const T = {};
async function trades(id) {
  if (T[id]) return T[id];
  const d = JSON.parse((await siteFile('s/' + id.replace('&', 'and') + '.json')).toString());
  return (T[id] = { tin: Array.from(new Int32Array(b64(d.tr_tin)), v => v * MIN), tout: Array.from(new Int32Array(b64(d.tr_tout)), v => v * MIN) });
}
async function tradesInView(id, v0, v1) { const s = await trades(id); let n = 0; for (let i = 0; i < s.tin.length; i++) if (s.tin[i] < v1 && s.tout[i] >= v0) n++; return n; }
let META = null; const MONTH = {};
async function meta() { return META || (META = JSON.parse((await siteFile('meta.json')).toString())); }
async function minuteClose(ms) {      // month m index i is the bar whose close is stamped at (T0 + m.start_min + i + 1) min
  const M = await meta(), t = ms / MIN, T0 = M.price.t0_min;
  for (const m of M.price.months) {
    const i = t - T0 - m.start_min - 1;
    if (i >= 0 && i < m.n) { const a = MONTH[m.name] || (MONTH[m.name] = new Float32Array(b64((await siteFile('m/' + m.name + '.txt')).toString().trim()))); return a[i]; }
  }
  return null;
}
let HOUR = null;
async function hourClose(ms) {         // last hourly close stamped at or before ms
  const M = await meta(), T0 = M.price.t0_min;
  if (!HOUR) HOUR = new Float32Array(ab(await siteFile('price_hour.bin')));
  const k = Math.floor((ms / MIN - T0) / 60) - 1;
  return k >= 0 ? { stamp: new Date((T0 + (k + 1) * 60) * MIN).toISOString().slice(11, 16), v: HOUR[k] } : null;
}
const fmt1 = v => v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });

// ---------- page helpers
async function open(browser, vp, setup) {
  const ctx = await browser.newContext({ viewport: vp });
  // the page opens on the embedded strategy (T2-P) unless the user picked one; these checks were written for a T1-LIN start
  await ctx.addInitScript(() => { try { if (!localStorage.getItem('replay-pick')) localStorage.setItem('replay-pick', 'T1-LIN'); } catch (e) { /* no storage */ } });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  if (setup) await setup(page, ctx);
  await page.goto(URL); await ready(page); await page.waitForTimeout(600);
  return { ctx, page, errs };
}
const plotBox = p => p.evaluate(() => {
  const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect();
  return { l: r.left + s.l, t: r.top + s.t, w: s.w, h: s.h };
});
const axis = p => p.evaluate(() => document.getElementById('chart')._fullLayout.xaxis.range.map(v => Date.parse(String(v).replace(' ', 'T').slice(0, 23) + 'Z')));
const readout = p => p.$eval('#readout', e => e.innerText.replace(/\s+/g, ' ').trim());
const rtime = t => { const m = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC/.exec(t); return m ? m[1] : null; };
const iso = ms => new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
const toMs = s => Date.parse(s.replace(' ', 'T') + ':00Z');
const pins = p => p.evaluate(() => (document.getElementById('chart').layout.shapes || []).filter(s => s.line && s.line.width === 1.5).map(s => String(s.x0).slice(0, 16)));
const hoverLine = p => p.evaluate(() => { const h = document.getElementById('hoverline'); return { shown: h.style.display === 'block', left: parseFloat(h.style.left) }; });
const pick = (p, id) => p.click(`#seg-t1 button[data-key="${id}"], #seg-t2 button[data-key="${id}"], #seg-t0 button[data-key="${id}"]`);
const chip = (p, k) => p.click(`#seg-range button[data-key="${k}"]`);
const kbd = async (p, sel) => { await p.evaluate(s => document.querySelector(s).focus({ preventScroll: true }), sel); await p.keyboard.press('Enter'); };
// the minute the page reads at a client x (same rounding as the page), and the time span of one plot pixel
const atX = (p, cx) => p.evaluate(cx => {
  const g = document.getElementById('chart'), xa = g._fullLayout.xaxis, r = g.getBoundingClientRect();
  const px = Math.max(0, Math.min(xa._length, Math.round(cx) - r.left - xa._offset));
  return { ms: Math.round(xa.p2c(px) / 60000) * 60000, perPx: (xa.p2c(xa._length) - xa.p2c(0)) / xa._length };
}, cx);
const tbl = p => p.evaluate(() => ({ title: document.getElementById('tbl-title').textContent, note: document.getElementById('tbl-note').textContent,
  rows: document.querySelectorAll('#tbl tbody tr').length, sel: document.getElementById('strat-select').value, res: document.getElementById('res').textContent }));
// counts: the click counts Chromium assigns (1,2 = same spot; 1,1 = moved >= 3 px); offs: pointer offset [dx, dy] of each press
const pressDbl = async (page, x, y, counts, offs) => {
  for (let i = 0; i < counts.length; i++) {
    if (i) { await page.waitForTimeout(40); await page.mouse.move(x + offs[i][0], y + offs[i][1]); }
    await page.mouse.down({ clickCount: counts[i] }); await page.mouse.up({ clickCount: counts[i] });
  }
};

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();

  // ---- 1, 15: readout height while sweeping the plot at 701-960 px (parallel pages, no timing dependence)
  const labelsSeen = [];
  // [window width, strategy, font, advisory]; the last one runs with a classic 15 px scrollbar
  const sbBrowser = await chromium.launch({ ignoreDefaultArgs: ['--hide-scrollbars'] });
  const sweeps = [[720, 'T2-P', ''], [780, 'T2-P', ''], [820, 'tsmom_unscaled', ''], [860, 'tsmom_unscaled', 'DejaVu Sans'], [940, 'tsmom_unscaled', 'DejaVu Sans'], [961, 'tsmom_unscaled', ''],
    // advisory (open, older than round 2): at 701-708 px windows with a classic scrollbar row 2 (the trade line) also wraps
    // on long trades, so the box grows from 70 to 89 px and the chart jumps 18 px under the pointer
    [705, 'T2-P', '', true]];
  await Promise.all(sweeps.map(([w, id, font, advisory]) => check(`readout_height_constant_w${w}_${id}${font ? '_' + font.replace(' ', '') : ''}${advisory ? '_scrollbar' : ''}`, [1, 15], async () => {
    const { ctx, page, errs } = await open(advisory ? sbBrowser : browser, { width: w, height: 900 });
    if (font) await page.addStyleTag({ content: `:root{--font:"${font}" !important}` });
    await page.evaluate(id => { const s = document.getElementById('strat-select'); s.value = id; s.dispatchEvent(new Event('change')); }, id);
    await ready(page); await page.waitForTimeout(300);
    const r = await page.evaluate(async () => {
      const gd = document.getElementById('chart'), ro = document.getElementById('readout'), z = gd._fullLayout._size, rect = gd.getBoundingClientRect();
      const raf = () => new Promise(res => requestAnimationFrame(() => requestAnimationFrame(res)));
      const hs = {}, tops = new Set(), labels = new Set(); let wraps = 0, n = 0;
      for (let px = 1; px < z.w; px += 4) {
        gd.dispatchEvent(new MouseEvent('mousemove', { clientX: rect.left + z.l + px, clientY: rect.top + z.t + z.h * 0.3, bubbles: true }));
        await raf(); n++;
        const h = Math.round(ro.getBoundingClientRect().height); hs[h] = (hs[h] || 0) + 1; tops.add(Math.round(gd.getBoundingClientRect().top));
        const row = ro.children[0]; if (row && row.getBoundingClientRect().height > 25) wraps++;
        const lab = row && row.children[1] ? row.children[1].textContent.replace(/[\d,.]+\s*$/, '').trim() : ''; if (lab) labels.add(lab.replace(/\d\d:\d\d/, 'HH:MM'));
      }
      return { n, heights: hs, chartTops: [...tops], row1Wrapped: wraps, labels: [...labels], minH: getComputedStyle(ro).minHeight, layoutWidth: document.documentElement.clientWidth };
    });
    const pass = Object.keys(r.heights).length === 1 && r.chartTops.length === 1 && !errs.length;
    if (!advisory) labelsSeen.push(...r.labels);
    await ctx.close();
    return { pass, measured: Object.assign({ width: w, strategy: id, font: font || 'default (system-ui)' }, r, { errs }) };
  }, advisory)));
  await sbBrowser.close();
  await check('readout_hourly_label_is_short', [15], async () => {   // '(HH:MM)', no ' close' (finding 15's fix)
    const l = [...new Set(labelsSeen)];
    return { pass: l.includes('Price (HH:MM)') && !l.some(x => /close/.test(x)), measured: { priceLabelsSeen: l } };
  });

  // ---- 0 (+14): trade table while a picked strategy loads, after it loads, and after a failed pick
  {
    let delay = 0;
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      await p.route('**/s/T2-NET.json', async r => { if (delay) await wait(delay); r.continue().catch(() => {}); });
      await p.route('**/s/T2-LIN.json', async r => { await wait(800); r.fulfill({ status: 404, body: 'missing' }).catch(() => {}); });
    });
    const during = [];
    delay = 2500;
    await pick(page, 'T2-NET'); await page.waitForTimeout(250); during.push(Object.assign({ at: 'after pick' }, await tbl(page)));
    await chip(page, '1m'); await page.waitForTimeout(500); during.push(Object.assign({ at: '1 month chip' }, await tbl(page)));
    await page.setViewportSize({ width: 1300, height: 1000 }); await page.waitForTimeout(450); during.push(Object.assign({ at: 'window resize' }, await tbl(page)));
    await check('table_empty_while_picked_strategy_loads', [0], async () => ({
      pass: during.every(s => s.rows === 0 && s.title === 'Trades in view' && s.note === 'Loading T2-NET band net...' && s.sel === 'T2-NET'), measured: { during } }));
    await ready(page); await page.waitForTimeout(500);
    let ax = await axis(page); const d1 = await tbl(page), truth1 = await tradesInView('T2-NET', ax[0], ax[1]);
    await chip(page, 'all'); await page.waitForTimeout(800);
    ax = await axis(page); const d2 = await tbl(page), truth2 = await tradesInView('T2-NET', ax[0], ax[1]);
    await check('table_lists_picked_strategy_after_load', [0], async () => ({
      pass: d1.title === 'Trades in view: ' + truth1.toLocaleString('en-US') && d2.title === 'Trades in view: ' + truth2.toLocaleString('en-US') && d2.rows === Math.min(300, truth2),
      measured: { oneMonthView: { title: d1.title, truth: truth1 }, allView: { title: d2.title, rows: d2.rows, truth: truth2 } } }));
    await pick(page, 'T2-LIN'); await page.waitForTimeout(300); const f1 = await tbl(page);
    await page.waitForTimeout(1200); const f2 = await tbl(page);
    await check('table_refilled_after_failed_pick', [0, 14], async () => ({
      pass: f1.rows === 0 && f1.note === 'Loading T2-LIN ridge...' && f2.sel === 'T2-NET' && f2.title === 'Trades in view: ' + truth2.toLocaleString('en-US') && f2.rows === Math.min(300, truth2) &&
        f2.res === 'Could not load T2-LIN: T2-LIN (404)' && !errs.length,
      measured: { duringFailingLoad: f1, afterFailure: f2, truthAll: truth2, errs } }));
    await ctx.close();
  }

  // ---- 14: failed pick message vs a background 1-minute render, a B&H redraw, then a view change and a pick
  {
    let mDelay = 0;
    const months = [];
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      await p.route('**/s/T1-NET.json', async r => { await wait(150); r.fulfill({ status: 404, body: 'missing' }).catch(() => {}); });
      await p.route('**/m/*.txt', async r => { if (mDelay) await wait(mDelay); r.continue().catch(() => {}); });
      p.on('requestfinished', q => { if (/\/m\/.*\.txt$/.test(q.url())) months.push(q.url().split('/').pop()); });
    });
    await chip(page, '1m'); await page.waitForTimeout(1200);
    mDelay = 900; months.length = 0;
    await chip(page, '1d'); await page.waitForTimeout(100);
    await pick(page, 'T1-NET'); await page.waitForTimeout(1700);
    const r1 = await page.textContent('#res'), minuteRendered = await page.evaluate(() => document.getElementById('chart').data[0].x.length);
    await page.click('#bh-toggle'); await page.waitForTimeout(600); const r2 = await page.textContent('#res');
    await page.click('#bh-toggle'); await page.waitForTimeout(600);
    const MSG = 'Could not load T1-NET: T1-NET (404)';
    await check('failed_pick_message_survives_background_renders', [14], async () => ({
      pass: months.length > 0 && r1 === MSG && r2 === MSG,
      measured: { monthsLoadedDuring: months, priceTracePoints: minuteRendered, resAfter1minRender: r1, resAfterBHToggle: r2 } }));
    await page.click('#z-in'); await page.waitForTimeout(1500); const r3 = await page.textContent('#res');
    await pick(page, 'T1-NET'); await page.waitForTimeout(700); const r4 = await page.textContent('#res');
    await pick(page, 'T1-NT'); await ready(page); await page.waitForTimeout(500); const r5 = await page.textContent('#res');
    await check('failed_pick_message_clears_on_view_change_or_pick', [14], async () => ({
      pass: /^Price: /.test(r3) && r4 === MSG && /^Price: /.test(r5) && !errs.length,
      measured: { afterZoomIn: r3, afterSecondFailedPick: r4, afterGoodPick: r5, errs } }));
    await ctx.close();
  }

  // ---- 5: clicks 1 px inside the edges of non-minute-aligned 20-40 min views
  await check('edge_click_pins_inside_axis_and_survives_redraw', [5], async () => {
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 });
    await chip(page, '1d'); await page.waitForTimeout(1200);
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const views = [];
    for (let i = 0; i < 6; i++) {
      const span = (20 + rnd() * 20) * MIN, a = Date.parse('2023-06-01T00:00:00Z') + Math.floor(rnd() * 500 * 86400) * 1000 + Math.floor(rnd() * 60000);
      views.push([a, a + span]);
    }
    views.push([Date.parse('2026-10-31T00:00:00Z') - 31.3 * MIN, Date.parse('2026-10-31T00:00:00Z')]);   // data end: its last minute must be pinnable
    const bad = [], seen = [];
    for (const v of views) {
      await page.evaluate(v => Plotly.relayout(document.getElementById('chart'), { 'xaxis.range': v.map(x => new Date(x).toISOString().slice(0, 23).replace('T', ' ')) }), v);
      await page.waitForTimeout(700);
      const ax = await axis(page), b = await plotBox(page), y = b.t + b.h * 0.3;
      for (const side of ['L', 'R']) {
        const x = side === 'L' ? b.l + 1 : b.l + b.w - 1;
        await page.mouse.move(x, y); await page.waitForTimeout(80);
        const hov = rtime(await readout(page));
        await page.mouse.click(x, y); await page.waitForTimeout(350);
        const p = await pins(page), pin = p[0] ? toMs(p[0]) : null;
        const edge = side === 'L' ? Math.ceil(ax[0] / MIN) * MIN : Math.floor(ax[1] / MIN) * MIN;
        const rec = { view: [new Date(ax[0]).toISOString().slice(0, 19), new Date(ax[1]).toISOString().slice(0, 19)], side, hover: hov, pin: p[0] || null, edgeMinute: iso(edge) };
        seen.push(rec);
        if (pin !== edge || !hov || toMs(hov) < ax[0] || toMs(hov) > ax[1]) bad.push(rec);
        await page.mouse.move(b.l + b.w / 2, b.t - 60); await page.waitForTimeout(320);   // off the plot, past Plotly's double-click window
        if (side === 'R') {
          await page.click('#bh-toggle'); await page.waitForTimeout(600);
          const after = await pins(page);
          if (JSON.stringify(after) !== JSON.stringify(p)) bad.push({ view: rec.view, droppedOnRedraw: p, after });
        }
      }
    }
    await ctx.close();
    return { pass: bad.length === 0 && !errs.length, measured: { clicks: seen.length, bad: bad.slice(0, 6), dataEnd: seen.slice(-2), errs } };
  });

  // ---- 6, 13: the pin's 1-minute month load vs the hover readout (month files delayed 1.2 s)
  {
    let fail = false;
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      await p.route('**/m/*.txt', async r => { await wait(1200); if (fail) r.fulfill({ status: 404, body: 'missing' }).catch(() => {}); else r.continue().catch(() => {}); });
    });
    const b = await plotBox(page), y = b.t + b.h * 0.3;
    await page.mouse.move(b.l + b.w * 0.3, y); await page.waitForTimeout(150);
    await page.mouse.click(b.l + b.w * 0.3, y); await page.waitForTimeout(80);
    const pinned = await readout(page), pinT = rtime(pinned);
    await page.mouse.move(b.l + b.w * 0.7, y, { steps: 5 }); await page.waitForTimeout(250);
    const hov = await readout(page), want = await atX(page, b.l + b.w * 0.7), near = t => !!t && Math.abs(toMs(t) - want.ms) <= 1.5 * want.perPx + MIN;
    await page.waitForTimeout(2200);
    const after = await readout(page), hl = await hoverLine(page);
    await check('pin_month_load_keeps_hover_readout', [6, 13], async () => ({
      pass: near(rtime(hov)) && rtime(after) === rtime(hov) && hl.shown && rtime(after) !== pinT,
      measured: { pinnedReadout: pinned.slice(0, 60), hoverBeforeLoad: hov.slice(0, 60), afterLoad: after.slice(0, 60), timeUnderPointer: iso(want.ms), hoverLineShown: hl.shown } }));
    // click and rest: the pin's readout gets the exact 1-minute close
    await page.mouse.move(b.l + b.w * 0.45, y); await page.waitForTimeout(150);
    await page.mouse.click(b.l + b.w * 0.45, y); await page.waitForTimeout(80);
    const p2 = await readout(page);
    await page.waitForTimeout(2000);
    const p3 = await readout(page), t3 = rtime(p3), truth = t3 ? await minuteClose(toMs(t3)) : null;
    await check('pin_gets_exact_minute_close_when_resting', [6, 13], async () => ({
      pass: truth != null && p3.includes('Price ' + fmt1(truth) + ' ') && rtime(p2) === t3,
      measured: { atClick: p2.slice(0, 50), afterLoad: p3.slice(0, 50), truthClose: truth } }));
    // the pin's month fails: the last hourly close, labelled with its time
    fail = true;
    await page.mouse.move(b.l + b.w * 0.15, y); await page.waitForTimeout(150);
    await page.mouse.click(b.l + b.w * 0.15, y); await page.waitForTimeout(80);
    const q1 = await readout(page);
    await page.mouse.move(b.l + b.w / 2, b.t - 60); await page.waitForTimeout(1800);
    const q2 = await readout(page), tq = rtime(q2), hc = tq ? await hourClose(toMs(tq)) : null;
    await check('pin_month_failure_shows_labelled_hourly_close', [6, 13], async () => ({
      pass: hc != null && (q2.includes('Price (' + hc.stamp + ') ' + fmt1(hc.v) + ' ') || q2.includes('Price (' + hc.stamp + ' close) ' + fmt1(hc.v) + ' ')) && !errs.length,
      measured: { atClick: q1.slice(0, 50), afterFailure: q2.slice(0, 60), truthHourly: hc, errs } }));
    await ctx.close();
  }

  // ---- 11: double and triple clicks (click counts as Chromium assigns them), single click, drag then click
  {
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 });
    const O = [0, 0];
    const cases = [['dblclick_0px', [1, 2], [O, O]], ['dblclick_3px', [1, 1], [O, [3, 0]]], ['dblclick_6px', [1, 1], [O, [6, 0]]], ['dblclick_3px_vertical', [1, 1], [O, [0, 3]]],
      ['tripleclick_0px', [1, 2, 3], [O, O, O]], ['tripleclick_3px_then_0px', [1, 1, 2], [O, [3, 0], [3, 0]]],
      // advisory (new, open): the 3rd press moved >= 3 px and came > 300 ms after the 1st (Plotly starts a new click train):
      // its click passes both guards and pins at the reset axis. Reproduced with real X input at 50 ms holds / 90 ms gaps.
      ['tripleclick_3px_each', [1, 1, 1], [O, [3, 0], [6, 0]], true]];
    for (const [name, counts, offs, advisory] of cases) {
      await check(name + '_resets_and_keeps_pin', [11], async () => {
        await chip(page, '1m'); await page.waitForTimeout(1200);
        const b = await plotBox(page), y = b.t + b.h * 0.3;
        await page.mouse.click(b.l + b.w * 0.3, y); await page.waitForTimeout(450);
        const A = await pins(page), ax0 = await axis(page);
        await page.mouse.move(b.l + b.w * 0.6, y); await page.waitForTimeout(100);
        const t1 = Date.now();
        await pressDbl(page, b.l + b.w * 0.6, y, counts, offs);
        const ms = Date.now() - t1;
        await page.waitForTimeout(1500);
        const ax1 = await axis(page), P = await pins(page), t = await tbl(page);
        await page.mouse.move(b.l + b.w / 2, b.t - 60); await page.waitForTimeout(350);
        return { pass: A.length === 1 && Math.round((ax1[1] - ax1[0]) / 864e5) === 2129 && JSON.stringify(P) === JSON.stringify(A) && t.title === 'Trades in view: 416',
          measured: { clickCounts: counts, offsets: offs, firstToLastPressMs: ms, pinBefore: A, viewDaysBefore: +((ax0[1] - ax0[0]) / 864e5).toFixed(2), viewDaysAfter: +((ax1[1] - ax1[0]) / 864e5).toFixed(2), pinsAfter: P, table: t.title } };
      }, advisory);
    }
    await check('single_click_still_pins', [11], async () => {
      await chip(page, '1m'); await page.waitForTimeout(1200);
      const b = await plotBox(page), y = b.t + b.h * 0.3, x = b.l + b.w * 0.45;
      await page.mouse.move(x, y); await page.waitForTimeout(150);
      const hov = rtime(await readout(page));
      await page.mouse.click(x, y); await page.waitForTimeout(450);
      const P = await pins(page), ax = await axis(page);
      await page.mouse.move(b.l + b.w / 2, b.t - 60); await page.waitForTimeout(350);
      return { pass: P.length === 1 && P[0] === hov && Math.round((ax[1] - ax[0]) / 864e5) === 30, measured: { hover: hov, pins: P, viewDays: +((ax[1] - ax[0]) / 864e5).toFixed(2) } };
    });
    await check('drag_zoom_then_click_pins', [11], async () => {
      const b = await plotBox(page), y = b.t + b.h * 0.3;
      await page.mouse.move(b.l + b.w * 0.3, y); await page.mouse.down(); await page.mouse.move(b.l + b.w * 0.5, y, { steps: 6 }); await page.mouse.up();
      await page.waitForTimeout(400);
      const x = b.l + b.w * 0.4; await page.mouse.move(x, y); await page.waitForTimeout(150);
      const hov = rtime(await readout(page));
      await page.mouse.click(x, y); await page.waitForTimeout(800);
      const P = await pins(page), ax = await axis(page), days = (ax[1] - ax[0]) / 864e5;
      return { pass: days > 5 && days < 7 && P.length === 1 && P[0] === hov && toMs(P[0]) >= ax[0] && toMs(P[0]) <= ax[1] && !errs.length,
        measured: { viewDays: +days.toFixed(2), hover: hov, pins: P, errs } };
    });
    await ctx.close();
  }

  // ---- 17: the hover readout after redraws (pointer resting on the plot; keyboard-driven changes), and off the plot
  await check('hover_readout_follows_pointer_and_view_after_redraws', [17], async () => {
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1400 });   // tall: chart and zoom bar on screen, no scrolling
    let b = await plotBox(page); const y = b.t + b.h * 0.3, x = b.l + b.w * 0.55;
    await page.mouse.move(x, y, { steps: 3 }); await page.waitForTimeout(250);
    const steps = [];
    const snap = async (tag, cx, strat) => {
      const t = await readout(page), ax = await axis(page), hl = await hoverLine(page), tm = rtime(t);
      const w = cx != null ? await atX(page, cx) : null;
      const ok = cx != null ? !!tm && Math.abs(toMs(tm) - w.ms) <= 1.5 * w.perPx + MIN && toMs(tm) >= ax[0] && toMs(tm) <= ax[1] && hl.shown && (!strat || t.includes(strat + ' PnL'))
        : null;
      const s = { tag, readout: t.slice(0, 48), underPointer: w ? iso(w.ms) : null, view: ax.map(iso), hoverLine: hl.shown, ok };
      steps.push(s); return s;
    };
    await kbd(page, '#seg-range button[data-key="seal"]'); await page.waitForTimeout(1300); await snap('keyboard Sealed-year chip', x, 'T1-LIN');
    await kbd(page, '#z-in'); await page.waitForTimeout(1300); await snap('keyboard zoom-in button', x, 'T1-LIN');
    await kbd(page, '#p-left'); await page.waitForTimeout(1300); await snap('keyboard pan-left button', x, 'T1-LIN');
    await kbd(page, 'button[data-key="T2-P"]'); await ready(page); await page.waitForTimeout(600); await snap('keyboard T2-P', x, 'T2-P');
    await kbd(page, '#z-reset'); await page.waitForTimeout(1300);
    b = await plotBox(page);
    const xa = b.l + b.w * 0.7, xb = b.l + b.w * 0.3;
    await page.mouse.move(xa, y, { steps: 2 }); await page.mouse.down(); await page.mouse.move(xb, y, { steps: 10 }); await page.mouse.up(); await page.waitForTimeout(1500);
    await snap('right-to-left drag zoom', xb, 'T2-P');
    await page.mouse.move(b.l + b.w * 0.5, y); await page.waitForTimeout(150);
    await pressDbl(page, b.l + b.w * 0.5, y, [1, 2], [[0, 0], [0, 0]]); await page.waitForTimeout(1500);
    await snap('double-click (no pin)', b.l + b.w * 0.5, 'T2-P');
    await chip(page, 'dev'); await page.waitForTimeout(1200);       // the mouse moves to the chip: pointer off the plot
    const off1 = await readout(page);
    await pick(page, 'T1-NT'); await ready(page); await page.waitForTimeout(600);
    const off2 = await readout(page);
    await page.mouse.click(b.l + b.w * 0.4, y); await page.waitForTimeout(500);
    const pinned = await pins(page);
    await chip(page, 'all'); await page.waitForTimeout(1200);
    const off3 = await readout(page);
    await ctx.close();
    const HINT = 'Point at the chart to read the values at that time.';
    const pass = steps.every(s => s.ok) && off1 === HINT && off2 === HINT && pinned.length === 1 && rtime(off3) === pinned[0] && off3.includes('T1-NT PnL') && !errs.length;
    return { pass, measured: { onPlot: steps, offPlotAfterChip: off1.slice(0, 40), offPlotAfterPick: off2.slice(0, 40), pin: pinned, offPlotPinnedAfterChip: off3.slice(0, 48), errs } };
  });

  await browser.close();
  const counted = results.filter(r => !r.advisory), failed = counted.filter(r => !r.pass), adv = results.filter(r => r.advisory);
  console.log(JSON.stringify({ summary: { page: PAGE, total: counted.length, passed: counted.length - failed.length, failed: failed.length, failedChecks: failed.map(r => r.check),
    advisory: adv.map(r => ({ check: r.check, pass: r.pass })), seconds: Math.round((Date.now() - t0) / 1000) } }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
