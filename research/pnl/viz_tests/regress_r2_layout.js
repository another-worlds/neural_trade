#!/usr/bin/env node
// Regression checks for the round-2 phone-layout cluster (round-2 findings 7, 8, 9, 10, 18).
//   F7  in-plot "Price (USDT)" title clear of trade marks and the price line on portrait phones, including the worst
//       pans found in round 2 (an entry that is the view maximum sits under the title); lower titles stay in their gaps
//   F10 short landscape charts (568x320, 658x320): lower titles inside their gaps, no data under any title (charts with a
//       plot area under 200 px have no price title by design, so 2 titles are expected there); and no
//       overlapping y tick labels (or titles on tick labels) on short landscape charts (658x320, 740x360, 844x390)
//   F8/F18 the chart card (readout + chart + zoom bar + hint) fits one screen on iPhone 13, 13 mini, 15, Galaxy S9+,
//       360x740 and Pixel 7, at load and after a pin; zoom buttons and hint stay on screen after switching strategy
//   F9  landscape phones: with the zoom bar on screen the stuck readout covers (almost) none of the price panel,
//       its height is constant over taps, and "+" stays fully on screen
// Needs only playwright + fs. env: PLOTLY = path of plotly-2.34.0.min.js (required); PAGE = page under BASE
// (default test.html); BASE (default http://localhost:8765/). Prints one JSON line per check and a summary line;
// exits 1 if any counted check fails. Checks marked "info": true document a known residual and do not set the exit
// code (STRICT=1 counts them).
const { chromium, devices } = require('playwright'); const fs = require('fs');
if (!process.env.PLOTLY) { console.error('set PLOTLY to the path of plotly-2.34.0.min.js'); process.exit(2); }
const PL = fs.readFileSync(process.env.PLOTLY);
const PAGE = process.env.PAGE || 'test.html', BASE = process.env.BASE || 'http://localhost:8765/', URL = BASE + PAGE, STRICT = process.env.STRICT === '1';
const UA = devices['Pixel 7'].userAgent;
const custom = (w, h, dsf = 3) => ({ viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: dsf, isMobile: true, hasTouch: true, userAgent: UA });
const strip = d => { const o = { ...d }; delete o.defaultBrowserType; return o; };
const DEV = {
  'iPhone 13': strip(devices['iPhone 13']), 'iPhone 13 Mini': strip(devices['iPhone 13 Mini']), 'iPhone 15': strip(devices['iPhone 15']),
  'Galaxy S9+': strip(devices['Galaxy S9+']), 'iPhone SE': strip(devices['iPhone SE']), 'Pixel 7': strip(devices['Pixel 7']), '360x740': custom(360, 740),
  '568x320': strip(devices['iPhone SE landscape']), '658x320': strip(devices['Galaxy S9+ landscape']), '640x360': strip(devices['Moto G4 landscape']),
  '802x293': strip(devices['Pixel 5 landscape']), '740x360': custom(740, 360), '844x390': custom(844, 390),
};
const results = [];
const check = (name, pass, measured, info = false) => { const r = { name, pass: !!pass, ...(info ? { info: true } : {}), measured }; results.push(r); console.log(JSON.stringify(r)); };

async function open(browser, dn) {
  const ctx = await browser.newContext({ ...DEV[dn], colorScheme: 'light' });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  await page.goto(URL);
  await ready(page); await idle(page);
  return { ctx, page, errs };
}
const ready = p => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
async function idle(page, ms = 200) {
  let prev = null;
  for (let k = 0; k < 80; k++) {
    await page.waitForTimeout(ms);
    const cur = await page.evaluate(() => { const g = document.getElementById('chart'); return JSON.stringify([g._fullLayout && g._fullLayout.xaxis.range, document.getElementById('res').textContent, g.data && g.data.map(t => t.x.length)]); });
    if (cur === prev && !/Loading/.test(cur)) return; prev = cur;
  }
}
const pick = async (page, id) => {   // the select is hidden above 700 px, so drive it directly
  const changed = await page.evaluate(id => { const s = document.getElementById('strat-select'); if (s.value === id) return false; s.value = id; s.dispatchEvent(new Event('change')); return true; }, id);
  if (changed) { await ready(page); await idle(page); }
};
const chip = async (page, key) => { await page.evaluate(k => document.querySelector('#seg-range button[data-key="' + k + '"]').click(), key); await idle(page); };
const panTo = async (page, t, spanD) => {   // the view a sideways swipe ends in: an entry at t sits ~25 px into the plot
  await page.evaluate(([t, S]) => { const g = document.getElementById('chart'), f = 25 / g._fullLayout.xaxis._length, iso = ms => new Date(ms).toISOString().replace('T', ' ').replace('Z', '');
    Plotly.relayout(g, { 'xaxis.range': [iso(t - f * S), iso(t + (1 - f) * S)] }); }, [t, spanD * 864e5]);
  await idle(page);
};
// views: [strategy, chip] or [strategy, 'pan', t(ms), span days]; the pans are round 2's worst cases (cand.json)
const PANS = [['T1-NET', 'pan', 1738953600000, 1], ['T2-LIN', 'pan', 1663783200000, 1], ['T1-NET', 'pan', 1748473500000, 30]];
const setViewOf = async (page, v) => { await pick(page, v[0]); if (v[1] === 'pan') await panTo(page, v[2], v[3]); else await chip(page, v[1]); };

// Geometry of the in-plot titles against drawn data (Plotly symbol shapes, marks clipped to their panel), title boxes
// against the panel above, and y tick label / title collisions. Hold lines are left out: a trade that starts off view
// draws its thin hold line in from the plot edge, and that cannot be avoided.
const MEASURE = () => {
  const g = document.getElementById('chart'), fl = g._fullLayout, gr = g.getBoundingClientRect(), xa = fl.xaxis, axes = { y: fl.yaxis, y2: fl.yaxis2, y3: fl.yaxis3 };
  const R = e => { const b = e.getBoundingClientRect(); return { l: b.left - gr.left, r: b.right - gr.left, t: b.top - gr.top, b: b.bottom - gr.top, s: e.textContent }; };
  const titles = {}; [...g.querySelectorAll('.annotation')].forEach(a => { if (/^(Price \(USDT\)|Net PnL \(%\)|Position \(x\))$/.test(a.textContent)) titles[a.textContent] = R(a.querySelector('.bg') || a); });
  const cover = {};
  for (const t in titles) {
    const box = titles[t]; let worst = { ov: -99 };
    g._fullData.forEach(tr => {
      if (tr.visible !== true || !tr.x || !axes[tr.yaxis] || tr.name === 'Hold') return;
      const ya = axes[tr.yaxis], P = [ya._offset, ya._offset + ya._length], X = [xa._offset, xa._offset + xa._length];
      const px = i => xa._offset + xa.d2p(tr.x[i]), py = i => ya._offset + ya.d2p(tr.y[i]);
      if (/markers/.test(tr.mode || '') && tr.marker) {
        const r = +tr.marker.size / 2, lw = (tr.marker.line && +tr.marker.line.width) || 0, sym = tr.marker.symbol || 'circle';
        for (let i = 0; i < tr.x.length; i++) {
          if (tr.x[i] == null || tr.y[i] == null) continue;
          const x = px(i), y = py(i); let top, bot, hw;
          if (/triangle-up/.test(sym)) { top = y - r; bot = y + r / 2; hw = 2 * r / Math.sqrt(3); } else if (/triangle-down/.test(sym)) { top = y - r / 2; bot = y + r; hw = 2 * r / Math.sqrt(3); } else { top = y - r; bot = y + r; hw = r; }
          top = Math.max(P[0], top - lw / 2); bot = Math.min(P[1], bot + lw / 2);
          const l = Math.max(X[0], x - hw - lw / 2), rr = Math.min(X[1], x + hw + lw / 2);
          if (bot <= top || rr <= l || rr < box.l || l > box.r) continue;
          const ov = Math.min(box.b, bot) - Math.max(box.t, top); if (ov > worst.ov) worst = { ov: +ov.toFixed(2), what: tr.name + ' ' + tr.x[i] };
        }
      }
      if (/lines/.test(tr.mode || '') && tr.line) {
        const w = +tr.line.width || 2;
        for (let i = 0; i + 1 < tr.x.length; i++) {
          if (tr.x[i] == null || tr.y[i] == null || tr.x[i + 1] == null || tr.y[i + 1] == null) continue;
          const x0 = px(i), y0 = py(i), x1 = px(i + 1), y1 = py(i + 1);
          if (Math.max(x0, x1) < box.l || Math.min(x0, x1) > box.r) continue;
          const n = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0) * 2));
          for (let k = 0; k <= n; k++) {
            const x = x0 + (x1 - x0) * k / n, y = y0 + (y1 - y0) * k / n;
            if (x < box.l || x > box.r || x < X[0] || x > X[1]) continue;
            const top = Math.max(P[0], y - w / 2), bot = Math.min(P[1], y + w / 2); if (bot <= top) continue;
            const ov = Math.min(box.b, bot) - Math.max(box.t, top); if (ov > worst.ov) worst = { ov: +ov.toFixed(2), what: tr.name + ' line ' + tr.x[i] };
          }
        }
      }
    });
    cover[t] = worst;
  }
  const ov = (a, b) => Math.min(a.r, b.r) - Math.max(a.l, b.l) > 0.5 && Math.min(a.b, b.b) - Math.max(a.t, b.t) > 0.5;
  const ticks = [...g.querySelectorAll('.ytick text, .y2tick text, .y3tick text')].map(R).filter(x => x.r > x.l);
  const tt = [...g.querySelectorAll('.g-ytitle text, .g-y2title text, .g-y3title text')].map(R).concat(Object.values(titles));
  const clash = [];
  for (let i = 0; i < ticks.length; i++) for (let j = i + 1; j < ticks.length; j++) if (ov(ticks[i], ticks[j])) clash.push(ticks[i].s + '|' + ticks[j].s);
  tt.forEach(t => ticks.forEach(k => { if (ov(t, k)) clash.push('title ' + t.s + '|' + k.s); }));
  for (let i = 0; i < tt.length; i++) for (let j = i + 1; j < tt.length; j++) if (ov(tt[i], tt[j])) clash.push('titles ' + tt[i].s + '|' + tt[j].s);
  const P = k => [axes[k]._offset, axes[k]._offset + axes[k]._length], gapOver = {};
  if (titles['Net PnL (%)']) gapOver.pnl = +(P('y')[1] - titles['Net PnL (%)'].t).toFixed(1);       // > 0: the box reaches into the price panel
  if (titles['Position (x)']) gapOver.pos = +(P('y2')[1] - titles['Position (x)'].t).toFixed(1);    // > 0: the box reaches into the PnL panel
  return { coarse: matchMedia('(pointer: coarse)').matches, chartH: g.clientHeight, titles: Object.keys(titles).length, cover, gapOver, clash: clash.slice(0, 6), nClash: clash.length };
};

async function titleChecks(browser, dn, views, label, opts) {
  const { ctx, page, errs } = await open(browser, dn);
  const rows = [];
  for (const v of views) { await setViewOf(page, v); rows.push({ view: v[0] + ' ' + (v[1] === 'pan' ? 'pan' + v[3] + 'd@' + new Date(v[2]).toISOString().slice(0, 16) : v[1]), ...(await page.evaluate(MEASURE)) }); }
  await ctx.close();
  const worst = rows.map(r => ({ view: r.view, chartH: r.chartH, maxCover: Math.max(-99, ...Object.values(r.cover).map(c => c.ov)), what: (Object.entries(r.cover).sort((a, b) => b[1].ov - a[1].ov)[0] || [])[0], gapOver: r.gapOver, nClash: r.nClash, clash: r.clash }));
  if (opts.titles) {
    const ok = rows.every(r => r.coarse && r.titles === (r.chartH - 54 < 200 ? 2 : 3) && Object.values(r.cover).every(c => c.ov <= 0) && (r.gapOver.pnl == null || r.gapOver.pnl <= 0.5) && (r.gapOver.pos == null || r.gapOver.pos <= 0.5));
    check(`${label} ${dn}: in-plot panel titles cover no trade mark or data line, lower titles stay inside their gaps`, ok, { errs: errs.length, views: worst.map(w => ({ view: w.view, maxCoverPx: w.maxCover, gapOver: w.gapOver })) });
  }
  if (opts.ticks) check(`${label} ${dn}: no overlapping y tick labels, no title on a tick label`, rows.every(r => r.nClash === 0),
    { chartH: rows[0].chartH, views: worst.map(w => ({ view: w.view, nClash: w.nClash, clash: w.clash })) });
}

async function cardChecks(browser, dn) {
  const { ctx, page, errs } = await open(browser, dn);
  const M = async () => {
    await page.evaluate(() => window.scrollTo(0, document.getElementById('card').getBoundingClientRect().top + scrollY)); await page.waitForTimeout(150);
    return page.evaluate(() => {
      const probe = document.createElement('div'); probe.style.cssText = 'position:absolute;top:0;height:100svh;width:1px'; document.body.appendChild(probe);
      const fold = probe.getBoundingClientRect().height; probe.remove();
      const hint = document.getElementById('hint'), rg = document.createRange(); rg.selectNodeContents(hint);
      const lines = []; for (const r of rg.getClientRects()) if (!lines.some(l => Math.abs(l[0] - r.top) < 3)) lines.push([Math.round(r.top), Math.round(r.bottom)]);
      const btn = Math.max(...['z-out', 'z-in', 'p-left', 'p-right', 'z-reset'].map(id => document.getElementById(id).getBoundingClientRect().bottom));
      const c = document.getElementById('card').getBoundingClientRect();
      return { fold, coarse: matchMedia('(pointer: coarse)').matches, cardTop: Math.round(c.top), cardBottom: +c.bottom.toFixed(1), chartH: Math.round(document.getElementById('chart').getBoundingClientRect().height), buttonsBottom: +btn.toFixed(1), hintLines: lines };
    });
  };
  const fits = m => m.coarse && m.cardBottom <= m.fold + 0.5 && m.buttonsBottom <= m.fold && m.hintLines.every(l => l[1] <= m.fold + 0.5);
  const s = {}; s.load = await M();
  await pick(page, 'T2-NET'); await chip(page, 'seal'); await M();
  const pt = await page.evaluate(() => { const g = document.getElementById('chart'), fl = g._fullLayout, r = g.getBoundingClientRect(), a = fl.yaxis; return { x: r.left + fl._size.l + fl._size.w * 0.4, y: r.top + a._offset + a._length / 2 }; });
  await page.touchscreen.tap(pt.x, pt.y); await page.waitForTimeout(900);
  s.pinned = await M(); s.pinned.readout = await page.evaluate(() => /UTC/.test(document.getElementById('readout').innerText));
  await pick(page, 'tsmom_unscaled'); s.tsmom = await M();
  await ctx.close();
  check(`F8/F18 ${dn}: chart card fits one screen (card top at the screen top) at load and after a pin`, fits(s.load) && fits(s.pinned) && s.pinned.readout, { errs: errs.length, load: s.load, pinned: s.pinned });
  check(`F8/F18 ${dn}: zoom buttons and the hint's first line on screen after switching to tsmom_unscaled`, s.tsmom.buttonsBottom <= s.tsmom.fold && s.tsmom.hintLines.length && s.tsmom.hintLines[0][1] <= s.tsmom.fold, { tsmom: s.tsmom });
  if (!fits(s.tsmom)) check(`F8/F18 ${dn}: whole card fits after switching to tsmom_unscaled`, false, { tsmom: s.tsmom }, true);
}

async function landChecks(browser, dn) {
  const { ctx, page, errs } = await open(browser, dn);
  await pick(page, 'tsmom_unscaled'); await chip(page, 'all');
  const M = () => page.evaluate(() => {
    const ro = document.getElementById('readout'), r = ro.getBoundingClientRect(), g = document.getElementById('chart'), fl = g._fullLayout, gr = g.getBoundingClientRect(), a = fl.yaxis;
    const pt = gr.top + a._offset, pb = pt + a._length, zin = document.getElementById('z-in').getBoundingClientRect();
    const mid = (Math.max(pt, r.top) + Math.min(pb, r.bottom)) / 2, el = document.elementFromPoint(gr.left + fl._size.l + fl._size.w / 2, mid);
    return { ih: innerHeight, sticky: getComputedStyle(ro).position === 'sticky', readout: [Math.round(r.top), Math.round(r.bottom)], readoutH: +r.height.toFixed(1), chartH: Math.round(gr.height),
      pricePanel: [Math.round(pt), Math.round(pb)], priceCoveredPx: +Math.max(0, Math.min(pb, r.bottom) - Math.max(pt, r.top)).toFixed(1), coveredBy: pb > r.top && pt < r.bottom && el ? (el.closest('#readout') ? 'readout' : 'chart') : null,
      plus: [Math.round(zin.top), Math.round(zin.bottom)], pinned: /UTC/.test(ro.innerText),
      sealedLabelHiddenPx: (() => { const a = [...g.querySelectorAll('.annotation')].find(x => x.textContent === 'Sealed year'); if (!a) return null; const t = a.querySelector('text').getBoundingClientRect(); return +Math.max(0, Math.min(r.bottom, t.bottom) - Math.max(r.top, t.top)).toFixed(1); })() };
  });
  const toZoombar = async () => { await page.evaluate(() => { const zb = document.querySelector('.zoombar').getBoundingClientRect(); window.scrollBy(0, zb.bottom - innerHeight); }); await page.waitForTimeout(200); };
  await toZoombar(); const c0 = await M();
  const heights = [c0.readoutH], plusB = [];
  for (const f of [0.1, 0.3, 0.5, 0.7, 0.9]) {
    const pt = await page.evaluate(f => { const g = document.getElementById('chart'), fl = g._fullLayout, r = g.getBoundingClientRect(), a = fl.yaxis3; return { x: r.left + fl._size.l + fl._size.w * f, y: r.top + a._offset + a._length / 2 }; }, f);
    await page.touchscreen.tap(pt.x, pt.y); await page.waitForTimeout(500);
    const m = await M(); heights.push(m.readoutH); plusB.push(m.plus[1]);
  }
  const c1 = await M();
  const z = await page.evaluate(() => { const r = document.getElementById('z-in').getBoundingClientRect(); return { x: r.left + r.width / 2, y: r.top + r.height / 2 }; });
  await page.touchscreen.tap(z.x, z.y); await idle(page); await toZoombar();
  const c2 = await M(); heights.push(c2.readoutH); plusB.push(c2.plus[1]);
  await ctx.close();
  const uniq = [...new Set(heights)];
  check(`F9 ${dn}: zoom bar on screen -> the stuck readout covers <= 2 px of the price panel (hint, after 5 pins, after "+")`,
    [c0, c1, c2].every(m => m.sticky && m.priceCoveredPx <= 2) && c1.pinned, { errs: errs.length, beforeTaps: c0, afterPins: c1, afterPlus: c2 });
  check(`F9 ${dn}: readout height constant over taps and "+" stays fully on screen`, uniq.length === 1 && plusB.every(b => b <= c0.ih), { readoutHeights: uniq, plusBottoms: [...new Set(plusB)], ih: c0.ih });
  check(`N2 ${dn}: zoom bar on screen -> the "Sealed year" label is not under the stuck readout`, c0.sealedLabelHiddenPx === 0, { sealedLabelHiddenPx: c0.sealedLabelHiddenPx, readout: c0.readout, ih: c0.ih }, true);
}

async function priceRoomInfo(browser, dn) {   // new issue N1: how much height the price line gets, and whether a tick label sits on the data
  const { ctx, page } = await open(browser, dn);
  const out = {};
  for (const ck of ['all', 'crash']) {
    await chip(page, ck);
    out[ck] = await page.evaluate(() => { const g = document.getElementById('chart'), fl = g._fullLayout, ya = fl.yaxis, xa = fl.xaxis; let lo = Infinity, hi = -Infinity;
      g._fullData.forEach(t => { if (t.yaxis !== 'y' || !t.x) return; for (let i = 0; i < t.x.length; i++) { if (t.y[i] == null) continue; const x = xa.d2p(t.x[i]); if (x < 0 || x > xa._length) continue; lo = Math.min(lo, t.y[i]); hi = Math.max(hi, t.y[i]); } });
      const ticks = [...g.querySelectorAll('.ytick text')].map(t => +t.textContent.replace(/[^0-9.]/g, ''));
      return { chartH: g.clientHeight, pricePanelPx: +ya._length.toFixed(1), priceDataPx: +(ya.d2p(lo) - ya.d2p(hi)).toFixed(1), dataRange: [Math.round(lo), Math.round(hi)], ticks, ticksOnData: ticks.filter(v => v >= lo && v <= hi).length }; });
  }
  await ctx.close();
  check(`N1 ${dn}: the price line gets >= 40 px of height and a price tick label lies within the data (All, crash)`, Object.values(out).every(o => o.priceDataPx >= 40 && o.ticksOnData >= 1), out, true);
}

async function pool(tasks, n) { const q = tasks.slice(); await Promise.all(Array.from({ length: n }, async () => { while (q.length) await q.shift()(); })); }

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();
  const portraitViews = [['T1-NET', 'seal'], ['T1-FADEV', 'seal'], ...PANS];
  const landViews = [['T1-FADEV', 'seal'], ['T2-NET', '1m'], ['T2-NET', 'crash'], ['T2-NET', '1w'], ['T1-NET', 'seal'], ...PANS.slice(0, 1)];
  const tickViews = [['T1-LIN', 'all'], ['T1-LIN', 'seal'], ['T2-NET', '1w'], ['T1-NET', 'crash']];
  const tasks = [
    () => titleChecks(browser, 'iPhone 13', portraitViews, 'F7', { titles: true }),
    () => titleChecks(browser, 'iPhone SE', portraitViews, 'F7', { titles: true }),
    () => titleChecks(browser, '360x740', portraitViews, 'F7', { titles: true }),
    () => titleChecks(browser, '568x320', landViews, 'F10', { titles: true, ticks: true }),
    () => titleChecks(browser, '658x320', tickViews, 'F10', { titles: true, ticks: true }),
    () => titleChecks(browser, '740x360', tickViews, 'F10', { ticks: true }),
    () => titleChecks(browser, '844x390', tickViews, 'F10', { ticks: true }),
    ...['iPhone 13', 'iPhone 13 Mini', 'iPhone 15', 'Galaxy S9+', '360x740', 'Pixel 7'].map(dn => () => cardChecks(browser, dn)),
    ...['568x320', '658x320', '640x360', '802x293', '844x390'].map(dn => () => landChecks(browser, dn)),
    ...['568x320', '802x293'].map(dn => () => priceRoomInfo(browser, dn)),
  ];
  await pool(tasks, +(process.env.JOBS || 3));
  await browser.close();
  const counted = results.filter(r => STRICT || !r.info), failed = counted.filter(r => !r.pass);
  console.log(JSON.stringify({ summary: { page: PAGE, checks: counted.length, passed: counted.length - failed.length, failed: failed.map(r => r.name), info: results.filter(r => r.info).map(r => ({ name: r.name, pass: r.pass })), seconds: Math.round((Date.now() - t0) / 1000) } }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('ERROR', e); process.exit(2); });
