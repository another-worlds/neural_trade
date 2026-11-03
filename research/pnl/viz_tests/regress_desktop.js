// Desktop regression checks for the BTCUSDT replay viewer (earlier findings 0, 20, 6, 22, 5, 23, 24, 25).
// Requires only playwright and fs. Env: PLOTLY (path of plotly-2.34.0.min.js), PAGE (default test.html, served at
// http://localhost:8765/). Prints a JSON summary of every check and exits 1 when any check fails.
'use strict';
const { chromium } = require('playwright');
const fs = require('fs');

const PLOTLY = process.env.PLOTLY;
const PAGE = process.env.PAGE || 'test.html';
const BASE = 'http://localhost:8765/';
const URL = BASE + PAGE;
const PL = fs.readFileSync(PLOTLY);
const FULL_RES = 'Price: hourly · 2129 days in view', FULL_TBL = 'Trades in view: 416';   // T1-LIN, whole range

const results = [];
function record(name, findings, pass, measured) { results.push({ name, findings, pass: !!pass, measured }); }
async function check(name, findings, fn) {
  try { const { pass, measured } = await fn(); record(name, findings, pass, measured); }
  catch (e) { record(name, findings, false, { error: String(e && e.message || e).slice(0, 300) }); }
}

const wait = ms => new Promise(r => setTimeout(r, ms));
const ready = (p, t = 60000) => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: t });
async function open(browser, vp, setup) {
  const ctx = await browser.newContext({ viewport: vp });
  // the page opens on the embedded strategy (T2-P) unless the user picked one; these checks were written for a T1-LIN start
  await ctx.addInitScript(() => { try { if (!localStorage.getItem('replay-pick')) localStorage.setItem('replay-pick', 'T1-LIN'); } catch (e) { /* no storage */ } });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  if (setup) await setup(page, ctx);
  await page.goto(URL); await ready(page); await page.waitForTimeout(800);
  return { ctx, page, errs };
}
const plotBox = p => p.evaluate(() => {
  const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect();
  return { l: r.left + s.l, t: r.top + s.t, w: s.w, h: s.h };
});
// app state as the user sees it
const state = p => p.evaluate(() => {
  const g = document.getElementById('chart'), fl = g._fullLayout, px = g.data[0].x.filter(v => v != null);
  const ar = fl.xaxis.range.map(v => Date.parse(String(v).replace(' ', 'T').slice(0, 23) + 'Z'));
  return {
    axisDays: +((ar[1] - ar[0]) / 864e5).toFixed(1), res: document.getElementById('res').textContent,
    tbl: document.getElementById('tbl-title').textContent,
    priceFirst: String(px[0]).slice(0, 16), priceLast: String(px[px.length - 1]).slice(0, 16), pricePts: px.length,
    y1: fl.yaxis.range.map(v => +v.toFixed(1)), y2: fl.yaxis2.range.map(v => +v.toFixed(3)),
    pins: (g.layout.shapes || []).filter(s => s.line && s.line.width === 1.5).map(s => String(s.x0).slice(0, 16))
  };
});
const readoutText = p => p.$eval('#readout', e => e.innerText.replace(/\s+/g, ' ').trim());
function parseReadout(t) {
  const tm = /(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) UTC/.exec(t), pn = /(\S+) PnL ([+−-]?)([\d,]+\.\d)%/.exec(t);
  return { time: tm ? tm[1] : null, label: pn ? pn[1] : null, value: pn ? (pn[2] === '' || pn[2] === '+' ? 1 : -1) * +pn[3].replace(/,/g, '') : null };
}
async function dragZoom(page, b, f0, f1) {
  await page.mouse.move(b.l + b.w * f0, b.t + b.h * 0.3); await page.mouse.down();
  await page.mouse.move(b.l + b.w * f1, b.t + b.h * 0.3, { steps: 8 }); await page.mouse.up();
  await page.waitForTimeout(1500);
}
async function dblclick(page, x, y, gap) {   // native-like: click count 1, then click count 2
  await page.mouse.move(x, y);
  await page.mouse.down({ clickCount: 1 }); await page.mouse.up({ clickCount: 1 });
  if (gap) await page.waitForTimeout(gap);
  await page.mouse.down({ clickCount: 2 }); await page.mouse.up({ clickCount: 2 });
}
const settleFull = (page, t) => page.waitForFunction(r => document.getElementById('res').textContent === r, FULL_RES, { timeout: t }).catch(() => {});
function inSync(s, ref) {   // view after a double-click equals the Reset-button view
  return s.res === FULL_RES && s.tbl === FULL_TBL && s.priceFirst <= '2021-01-01 02:00' && s.priceLast >= '2026-10-30 00:00' &&
    JSON.stringify(s.y1) === JSON.stringify(ref.y1) && JSON.stringify(s.y2) === JSON.stringify(ref.y2);
}
// ground truth PnL (%) from the strategy file: pnl_t minutes since epoch, pnl_L log equity; linear between points
function pnlTruth(doc) {
  const buf = s => { const x = Buffer.from(s, 'base64'); return x.buffer.slice(x.byteOffset, x.byteOffset + x.length); };
  const t = new Int32Array(buf(doc.pnl_t)), L = new Float32Array(buf(doc.pnl_L));
  return ms => {
    let lo = 0, hi = t.length; while (lo < hi) { const m = (lo + hi) >> 1; if (t[m] * 60000 <= ms) lo = m + 1; else hi = m; }
    if (lo === 0) return null;
    const y = i => (Math.exp(L[i]) - 1) * 100;
    if (lo >= t.length) return y(lo - 1);
    const x0 = t[lo - 1] * 60000, x1 = t[lo] * 60000; return y(lo - 1) + (y(lo) - y(lo - 1)) * (ms - x0) / (x1 - x0);
  };
}

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();

  // ---- 1. drag-zoom then double-click: data, y ranges, table and #res follow the reset; no cursor is pinned
  {
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 });
    let after = null, zoomed = null;
    await check('dblclick_after_dragzoom_resyncs_view', [0, 20], async () => {
      const b = await plotBox(page);
      await dragZoom(page, b, 0.3, 0.45); zoomed = await state(page);
      await dblclick(page, b.l + b.w * 0.5, b.t + b.h * 0.3, 0);
      await settleFull(page, 5000); await page.waitForTimeout(600);
      after = await state(page);
      await page.click('#z-reset'); await page.waitForTimeout(1500);
      const ref = await state(page);
      return { pass: zoomed.res !== FULL_RES && inSync(after, ref) && !errs.length,
        measured: { zoomed: { res: zoomed.res, tbl: zoomed.tbl }, afterDblclick: { axisDays: after.axisDays, res: after.res, tbl: after.tbl, price: after.priceFirst + '..' + after.priceLast, y1: after.y1, y2: after.y2 },
          resetButton: { res: ref.res, tbl: ref.tbl, y1: ref.y1, y2: ref.y2 }, errs } };
    });
    await check('dblclick_does_not_pin_cursor', [6, 22], async () => ({ pass: after && after.pins.length === 0, measured: { pinsAfterDblclick: after && after.pins } }));
    await ctx.close();
  }

  // ---- 2. 1-minute view ("May 2021 crash"), CPU 4x, 60 ms between the clicks
  {
    const { ctx, page, errs } = await open(browser, { width: 1024, height: 768 });
    await check('dblclick_from_1min_view_cpu4_resyncs_view', [20, 0], async () => {
      const cdp = await ctx.newCDPSession(page); await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
      await page.click('#seg-range button[data-key="crash"]');
      await page.waitForFunction(() => /1-min/.test(document.getElementById('res').textContent), null, { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(800);
      const pre = await state(page), b = await plotBox(page);
      await dblclick(page, b.l + b.w * 0.5, b.t + b.h * 0.3, 60);
      await settleFull(page, 8000); await page.waitForTimeout(1200);
      const after = await state(page);
      await page.click('#z-reset'); await page.waitForTimeout(3000);
      const ref = await state(page);
      return { pass: /1-min/.test(pre.res) && inSync(after, ref) && after.pins.length === 0 && !errs.length,
        measured: { before: pre.res, afterDblclick: { res: after.res, tbl: after.tbl, price: after.priceFirst + '..' + after.priceLast, y1: after.y1, pins: after.pins }, resetButton: { y1: ref.y1, y2: ref.y2 }, errs } };
    });
    await ctx.close();
  }

  // ---- 3. double-click on the full view adds no cursor; a pin made before a double-click survives it
  {
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 });
    await check('dblclick_on_full_view_adds_no_cursor', [6, 22], async () => {
      const b = await plotBox(page);
      await dblclick(page, b.l + b.w * 0.25, b.t + b.h * 0.3, 0); await page.waitForTimeout(1500);
      const s = await state(page);
      return { pass: s.pins.length === 0 && s.res === FULL_RES, measured: { pins: s.pins, res: s.res } };
    });
    await page.reload(); await ready(page); await page.waitForTimeout(800);
    await check('dblclick_keeps_earlier_pin_and_resets', [6, 22], async () => {
      const b = await plotBox(page);
      await dragZoom(page, b, 0.3, 0.6);
      await page.mouse.click(b.l + b.w * 0.4, b.t + b.h * 0.3); await page.waitForTimeout(800);
      const pre = await state(page);
      await dblclick(page, b.l + b.w * 0.75, b.t + b.h * 0.3, 0);
      await settleFull(page, 5000); await page.waitForTimeout(600);
      const after = await state(page);
      await page.mouse.move(b.l + b.w * 0.5, b.t - 120); await page.waitForTimeout(400);   // leave the chart: the readout shows the pin
      const ro = parseReadout(await readoutText(page));
      return { pass: pre.pins.length === 1 && JSON.stringify(after.pins) === JSON.stringify(pre.pins) && after.res === FULL_RES && after.tbl === FULL_TBL && ro.time === pre.pins[0] && !errs.length,
        measured: { pinBefore: pre.pins, pinsAfter: after.pins, res: after.res, tbl: after.tbl, readoutTime: ro.time, errs } };
    });
    await ctx.close();
  }

  // ---- 4. slow strategy switch: the readout names the strategy whose numbers it shows
  {
    let truth = null;
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      await p.route('**/s/T2-LIN.json', async r => { await wait(4500); r.continue().catch(() => {}); });
    });
    await check('slow_switch_readout_label_matches_values', [5, 23], async () => {
      const docLin = await (await ctx.request.get(BASE + 's/T1-LIN.json')).json(), docP = await (await ctx.request.get(BASE + 's/T2-LIN.json')).json();
      truth = { 'T1-LIN': pnlTruth(docLin), 'T2-LIN': pnlTruth(docP) };
      const b = await plotBox(page);
      await page.mouse.click(b.l + b.w * 0.6, b.t + b.h * 0.3); await page.waitForTimeout(600);
      await page.mouse.move(b.l + b.w * 0.5, b.t - 120); await page.waitForTimeout(300);
      const pinned = parseReadout(await readoutText(page));
      await page.click('#seg-t2 button[data-key="T2-LIN"]'); await page.waitForTimeout(300);
      await page.click('#seg-range button[data-key="1m"]'); await page.waitForTimeout(900);   // a render while the file loads
      const stillLoading = await page.$eval('#loading', e => !e.hidden);
      const during = parseReadout(await readoutText(page));
      await ready(page, 20000); await page.waitForTimeout(800);
      const after = parseReadout(await readoutText(page));
      const ms = Date.parse(pinned.time.replace(' ', 'T') + ':00Z');
      const tv = { 'T1-LIN': +truth['T1-LIN'](ms).toFixed(2), 'T2-LIN': +truth['T2-LIN'](ms).toFixed(2) };
      const ok = r => r.label != null && tv[r.label] != null && Math.abs(r.value - tv[r.label]) <= 0.06;
      return { pass: stillLoading && during.time === pinned.time && ok(during) && after.label === 'T2-LIN' && ok(after) && !errs.length,
        measured: { pinnedAt: pinned.time, truthPct: tv, duringLoad: { overlayShown: stillLoading, label: during.label, value: during.value }, afterLoad: { label: after.label, value: after.value }, errs } };
    });
    await ctx.close();
  }

  // ---- 5. failed strategy load: controls, stats, legend return to the strategy on screen; retry works
  {
    let fail = true;
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      await p.route('**/s/T2-LIN.json', r => fail ? r.abort() : r.continue());
    });
    await check('failed_load_returns_controls_to_strategy_on_screen', [5], async () => {
      const ui = () => page.evaluate(() => ({
        select: document.getElementById('strat-select').value,
        pressed: [...document.querySelectorAll('#seg-t1 button, #seg-t2 button, #seg-t0 button')].filter(b => b.getAttribute('aria-pressed') === 'true').map(b => b.dataset.key),
        legend: document.getElementById('leg-strat').textContent, stats: document.getElementById('stats').innerText.replace(/\s+/g, ' '),
        overlayHidden: document.getElementById('loading').hidden, res: document.getElementById('res').textContent,
        tbl: document.getElementById('tbl-title').textContent, saved: (() => { try { return localStorage.getItem('replay-pick'); } catch (e) { return null; } })()
      }));
      const before = await ui(), b = await plotBox(page);
      await page.mouse.click(b.l + b.w * 0.6, b.t + b.h * 0.3); await page.waitForTimeout(600);
      await page.click('#seg-t2 button[data-key="T2-LIN"]'); await page.waitForTimeout(1200);
      const after = await ui();
      await page.click('#bh-toggle'); await page.waitForTimeout(900);              // any render after the failure
      await page.mouse.move(b.l + b.w * 0.5, b.t - 120); await page.waitForTimeout(300);
      const ro = parseReadout(await readoutText(page));
      await page.click('#bh-toggle'); await page.waitForTimeout(600);
      return { pass: after.select === 'T1-LIN' && JSON.stringify(after.pressed) === '["T1-LIN"]' && after.legend === before.legend && after.stats === before.stats &&
          after.overlayHidden && /Could not load T2-LIN/.test(after.res) && after.saved !== 'T2-LIN' && after.tbl === before.tbl && ro.label === 'T1-LIN',
        measured: { after: { select: after.select, pressed: after.pressed, legend: after.legend, statsUnchanged: after.stats === before.stats, overlayHidden: after.overlayHidden, res: after.res, saved: after.saved, tbl: after.tbl }, readoutLabelAfterRender: ro.label, errs } };
    });
    await check('failed_load_then_repick_loads', [5], async () => {
      fail = false;
      await page.click('#seg-t2 button[data-key="T2-LIN"]'); await ready(page, 20000); await page.waitForTimeout(800);
      const s = await page.evaluate(() => ({ select: document.getElementById('strat-select').value, legend: document.getElementById('leg-strat').textContent, tbl: document.getElementById('tbl-title').textContent }));
      return { pass: s.select === 'T2-LIN' && /^T2-LIN/.test(s.legend) && !errs.length, measured: s };
    });
    await ctx.close();
  }

  // ---- 6. re-picking a strategy whose file is still downloading: one download, one decode, one render
  {
    const reqs = [];
    const { ctx, page, errs } = await open(browser, { width: 1360, height: 1000 }, async p => {
      p.on('request', r => { if (/\/s\/T1-FADEV\.json/.test(r.url())) reqs.push(Date.now()); });
      await p.route('**/s/T1-FADEV.json', async r => { await wait(2000); r.continue().catch(() => {}); });
      await p.addInitScript(() => {
        window.__ev = [];
        const oj = Response.prototype.json;
        Response.prototype.json = function () { const u = this.url; return oj.call(this).then(d => { if (/T1-FADEV/.test(u)) window.__ev.push('json'); return d; }); };
      });
    });
    await check('repick_during_download_single_fetch_and_render', [24], async () => {
      await page.evaluate(() => { const o = Plotly.react; Plotly.react = function () { window.__ev.push('react'); return o.apply(this, arguments); }; });
      reqs.length = 0;
      const pick = k => page.click(`#seg-t1 button[data-key="${k}"]`);
      await pick('T1-FADEV'); await page.waitForTimeout(300);
      await pick('T1-NT'); await page.waitForTimeout(300);
      await pick('T1-FADEV');
      await ready(page, 20000); await page.waitForTimeout(1500);
      const ev = await page.evaluate(() => window.__ev.slice());
      const firstJson = ev.indexOf('json'), jsonN = ev.filter(e => e === 'json').length;
      const reactsAfter = firstJson < 0 ? null : ev.slice(firstJson).filter(e => e === 'react').length;
      const burstReqs = reqs.length;
      await pick('T1-NT'); await ready(page, 20000); await page.waitForTimeout(400);
      await pick('T1-FADEV'); await ready(page, 20000); await page.waitForTimeout(400);
      const fin = await page.evaluate(() => ({ pressed: [...document.querySelectorAll('#seg-t1 button[aria-pressed="true"]')].map(b => b.dataset.key), legend: document.getElementById('leg-strat').textContent, overlayHidden: document.getElementById('loading').hidden }));
      return { pass: burstReqs === 1 && jsonN === 1 && reactsAfter === 1 && reqs.length === 1 && JSON.stringify(fin.pressed) === '["T1-FADEV"]' && fin.overlayHidden && !errs.length,
        measured: { fadevRequestsDuringBurst: burstReqs, fadevJsonParses: jsonN, rendersAfterFadevArrived: reactsAfter, fadevRequestsAfterLaterRepick: reqs.length, events: ev.join(','), final: fin, errs } };
    });
    await ctx.close();
  }

  // ---- 7. short desktop windows: readout top to position-panel bottom fits the window; Plotly follows the height
  await check('readout_and_position_panel_fit_window', [25], async () => {
    const rows = []; let ok = true;
    for (const vp of [{ width: 1024, height: 768 }, { width: 1366, height: 657 }, { width: 1440, height: 560 }, { width: 1920, height: 1000 }]) {
      const { ctx, page } = await open(browser, vp);
      const g = await page.evaluate(() => {
        const gd = document.getElementById('chart'), fl = gd._fullLayout, sy = scrollY, ro = document.getElementById('readout').getBoundingClientRect(), cr = gd.getBoundingClientRect();
        return { coarse: matchMedia('(pointer: coarse)').matches, innerH: innerHeight, chartH: Math.round(cr.height), plotlyH: fl.height,
          span: Math.round(cr.top + sy + fl.yaxis3._offset + fl.yaxis3._length - (ro.top + sy)) };
      });
      const fits = g.span <= g.innerH && !g.coarse && Math.abs(g.chartH - g.plotlyH) <= 1 && (vp.height < 1000 || g.chartH === 760);
      ok = ok && fits; rows.push({ vp: vp.width + 'x' + vp.height, ...g, fits });
      await ctx.close();
    }
    const { ctx, page } = await open(browser, { width: 1366, height: 768 });
    await page.setViewportSize({ width: 1366, height: 600 }); await page.waitForTimeout(1500);
    const rz = await page.evaluate(() => { const gd = document.getElementById('chart'); return { div: Math.round(gd.getBoundingClientRect().height), plotly: gd._fullLayout.height }; });
    await ctx.close();
    ok = ok && Math.abs(rz.div - rz.plotly) <= 1 && rz.div < 760;
    return { pass: ok, measured: { windows: rows, heightOnlyResize768to600: rz } };
  });

  await browser.close();
  const failed = results.filter(r => !r.pass).length;
  console.log(JSON.stringify({ page: PAGE, seconds: Math.round((Date.now() - t0) / 1000), passed: results.length - failed, failed, checks: results }, null, 1));
  process.exit(failed ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
