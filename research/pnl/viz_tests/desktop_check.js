const { chromium } = require('playwright'); const fs = require('fs');
const PL = fs.readFileSync(process.env.PLOTLY);
const ready = p => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 60000 });
const range = p => p.evaluate(() => document.getElementById('chart')._fullLayout.xaxis.range.map(s => Date.parse(s.replace(' ', 'T').slice(0, 23) + 'Z')));
async function open(browser, opts, delays = {}) {
  const ctx = await browser.newContext(opts); const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.route('https://cdnjs.cloudflare.com/**', r => r.fulfill({ body: PL, contentType: 'application/javascript' }));
  for (const [pat, ms] of Object.entries(delays)) await page.route('**/' + pat, async r => { await new Promise(z => setTimeout(z, ms)); r.continue(); });
  const t0 = Date.now(); await page.goto('http://localhost:8765/test.html'); await ready(page); const tLoad = Date.now() - t0; await page.waitForTimeout(500);
  return { ctx, page, errs, tLoad };
}
(async () => {
  const browser = await chromium.launch(); const out = {};
  // 1. Moscow time zone, first trade: ticks, readout, marker tooltip
  let { ctx, page, errs, tLoad } = await open(browser, { viewport: { width: 1360, height: 1000 }, timezoneId: 'Europe/Moscow' });
  out.tLoad = tLoad;
  out.coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
  out.fidelity = await page.evaluate(() => {
    // the drawn price line keeps the true highest and lowest hourly close in view; the PnL line ends at the true last value
    const g = document.getElementById('chart'), pr = g.data[0], b = atob(JSON.parse(document.getElementById('price-b64').textContent));
    const u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); const hy = new Float32Array(u.buffer);
    let lo = Infinity, hi = -Infinity; for (const v of hy) if (v === v) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    const ys = pr.y.filter(v => v != null);
    const f = JSON.parse(document.getElementById('first-json').textContent), pb = atob(f.pnl_L), pu = new Uint8Array(pb.length); for (let i = 0; i < pb.length; i++) pu[i] = pb.charCodeAt(i);
    const L = new Float32Array(pu.buffer), lastTrue = (Math.exp(L[L.length - 1]) - 1) * 100, pn = g.data[6];
    return { pricePts: pr.x.length, hourPts: hy.length, maxOk: Math.max(...ys) === hi, minOk: Math.min(...ys) === lo, pnlLastTrue: +lastTrue.toFixed(4), pnlLastDrawn: +pn.y[pn.y.length - 1].toFixed(4) };
  });
  await page.click('#tbl tbody tr:first-child button'); await page.waitForTimeout(2500);
  out.mosTicks = (await page.$$eval('.xtick text', ns => ns.map(n => n.textContent))).slice(0, 5).join(' | ');
  out.firstRow = await page.$eval('#tbl tbody tr:first-child', r => r.innerText.replace(/\s+/g, ' '));
  const box = await page.$eval('#chart', el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  // marker of the first trade: its pixel position from Plotly's own axis maps
  const mk = await page.evaluate(() => {
    const g = document.getElementById('chart'), fl = g._fullLayout, xa = fl.xaxis, ya = fl.yaxis, r = g.getBoundingClientRect();
    const tr = g.data.find(t => (t.name === 'Long entry' || t.name === 'Short entry') && t.x.length);
    return { x: r.left + xa._offset + xa.d2p(tr.x[0]), y: r.top + ya._offset + ya.d2p(tr.y[0]), name: tr.name, when: tr.x[0] };
  });
  await page.mouse.move(mk.x, mk.y); await page.waitForTimeout(700);
  out.markerTip = (await page.$$eval('.hoverlayer text', ns => ns.map(n => n.textContent).join(' / '))).slice(0, 200);
  out.markerAt = mk.when;
  out.readoutAtMarker = await page.$eval('#readout', e => e.innerText.replace(/\s+/g, ' '));
  out.hoverline = await page.$eval('#hoverline', e => getComputedStyle(e).display);
  // readout height must not change while the pointer moves
  const hs = new Set();
  for (let k = 0; k < 12; k++) { await page.mouse.move(box.x + box.w * (0.1 + 0.07 * k), box.y + box.h * 0.5); await page.waitForTimeout(80); hs.add(await page.$eval('#readout', e => Math.round(e.getBoundingClientRect().height))); }
  out.readoutHeights = [...hs];
  await page.screenshot({ path: 'd2_moscow_trade.png' });
  out.mosErrs = errs; await ctx.close();

  // 2. drag zoom, double-click reset, zoom bar, click pins the cursor
  ({ ctx, page, errs } = await open(browser, { viewport: { width: 1360, height: 1000 } }));
  const b2 = await page.$eval('#chart', el => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height }; });
  const r0 = await range(page);
  await page.mouse.move(b2.x + b2.w * 0.4, b2.y + b2.h * 0.3); await page.mouse.down();
  await page.mouse.move(b2.x + b2.w * 0.5, b2.y + b2.h * 0.3, { steps: 8 }); await page.mouse.up(); await page.waitForTimeout(1500);
  const r1 = await range(page);
  out.dragZoomRatio = +(((r1[1] - r1[0]) / (r0[1] - r0[0])).toFixed(3));
  await page.mouse.dblclick(b2.x + b2.w * 0.5, b2.y + b2.h * 0.3); await page.waitForTimeout(1500);
  const r2 = await range(page); out.dblclickFull = Math.abs((r2[1] - r2[0]) - (r0[1] - r0[0])) < 60000;
  await page.mouse.click(b2.x + b2.w * 0.3, b2.y + b2.h * 0.5); await page.waitForTimeout(600);
  out.pinned = await page.evaluate(() => document.getElementById('chart').layout.shapes.filter(s => s.line && s.line.width === 1.5).length);
  await page.click('#seg-range button[data-key="1w"]'); await page.waitForTimeout(1500);
  const r3 = await range(page); out.weekAroundCursorDays = +((r3[1] - r3[0]) / 864e5).toFixed(2);
  await page.click('#z-in'); await page.waitForTimeout(1500); const r4 = await range(page); out.zoomInRatio = +(((r4[1] - r4[0]) / (r3[1] - r3[0])).toFixed(3));
  out.res = await page.textContent('#res');
  // 3. race: slow T1-NET file, pick T1-NET then T1-LIN
  await page.unroute('**/s/T1-NET.json').catch(() => {});
  await ctx.close();
  ({ ctx, page, errs } = await open(browser, { viewport: { width: 1360, height: 1000 } }, { 's/T1-NET.json': 1500 }));
  await page.click('#seg-t1 button[data-key="T1-NET"]'); await page.waitForTimeout(100); await page.click('#seg-t1 button[data-key="T1-LIN"]');
  await page.waitForTimeout(3000);
  out.race = (await page.textContent('#leg-strat')) + ' | ' + (await page.textContent('#tbl-title')) + ' | sel=' + (await page.$eval('#strat-select', s => s.value));
  // 4. every strategy (timed), dark screenshot
  out.switchMs = {};
  for (const k of ['T1-NET', 'T1-FADEV', 'T1-NT', 'T2-P', 'T2-NET', 'T2-LIN', 'B&H', 'tsmom_unscaled', 'T1-LIN']) {
    const t = Date.now(); await page.click(`#seg-t0 button[data-key="${k}"], #seg-t1 button[data-key="${k}"], #seg-t2 button[data-key="${k}"]`); await ready(page); out.switchMs[k] = Date.now() - t;
  }
  await page.screenshot({ path: 'd2_light_lin.png' });
  out.raceErrs = errs; await ctx.close();
  ({ ctx, page, errs } = await open(browser, { viewport: { width: 1360, height: 1000 }, colorScheme: 'dark' }));
  await page.click('#seg-t2 button[data-key="T2-P"]'); await ready(page); await page.click('#seg-range button[data-key="seal"]'); await page.waitForTimeout(1500);
  await page.screenshot({ path: 'd2_dark_t2p_seal.png' });
  out.darkErrs = errs; await ctx.close();
  // 5. narrow desktop window (mouse, phone layout) and a mid width
  for (const w of [650, 900]) {
    ({ ctx, page, errs } = await open(browser, { viewport: { width: w, height: 900 } }));
    out['w' + w] = await page.evaluate(() => ({ overflowX: document.documentElement.scrollWidth - innerWidth, selectShown: getComputedStyle(document.querySelector('.only-narrow')).display !== 'none', chips: getComputedStyle(document.querySelector('.only-wide')).display !== 'none' }));
    await page.screenshot({ path: `d2_w${w}.png` }); out['w' + w].errs = errs; await ctx.close();
  }
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})();
