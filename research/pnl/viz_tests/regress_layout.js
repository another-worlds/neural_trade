#!/usr/bin/env node
// Regression checks for the phone-layout cluster (earlier findings 9, 10, 14, 15, 16, 17, 19).
// Needs only playwright + fs. env PLOTLY = path of plotly-2.34.0.min.js, PAGE = page served at http://localhost:8765/
// (default test.html). Prints one JSON object per check and a summary; exits 1 if any counted check fails.
// Checks marked "info": true document a known follow-up problem and do not set the exit code (STRICT=1 counts them).
const { chromium, devices } = require('playwright'); const fs = require('fs');
const PLOTLY = process.env.PLOTLY;
const PAGE = process.env.PAGE || 'test.html', URL = 'http://localhost:8765/' + PAGE, STRICT = process.env.STRICT === '1';
const PL = fs.readFileSync(PLOTLY);
const UA = devices['Pixel 5'].userAgent;
const dev = (w, h, dsf = 2, ua = UA) => ({ viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: dsf, isMobile: true, hasTouch: true, userAgent: ua });
const results = [];
const check = (name, pass, measured, info = false) => { const r = { name, pass: !!pass, ...(info ? { info: true } : {}), measured }; results.push(r); console.log(JSON.stringify(r)); };

async function open(browser, d) {
  const ctx = await browser.newContext({ ...d, colorScheme: 'light' });
  const page = await ctx.newPage();
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  await page.goto(URL);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await idle(page);
  return { ctx, page };
}
async function idle(page, ms = 250) {
  let prev = null;
  for (let k = 0; k < 60; k++) {
    await page.waitForTimeout(ms);
    const cur = await page.evaluate(() => { const g = document.getElementById('chart'); return JSON.stringify([g._fullLayout && g._fullLayout.xaxis.range, document.getElementById('res').textContent, g.data && g.data.map(t => t.x.length)]); });
    if (cur === prev && !/Loading/.test(cur)) return; prev = cur;
  }
}
const ready = p => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
const chip = async (page, key) => { await page.evaluate(k => document.querySelector('#seg-range button[data-key="' + k + '"]').click(), key); await idle(page); };
const pick = async (page, id) => { await page.selectOption('#strat-select', id); await ready(page); await idle(page, 120); };
const panels = page => page.evaluate(() => {   // page-y of the three plot panels
  const g = document.getElementById('chart'), fl = g._fullLayout, top = g.getBoundingClientRect().top + scrollY;
  const p = k => [top + fl[k]._offset, top + fl[k]._offset + fl[k]._length];
  return { price: p('yaxis'), pnl: p('yaxis2'), pos: p('yaxis3') };
});
const tapPanel = async (page, yPage, frac = 0.5) => {
  const pt = await page.evaluate(([y, f]) => { const g = document.getElementById('chart'), fl = g._fullLayout, r = g.getBoundingClientRect(); return { x: r.left + fl._size.l + fl._size.w * f, y: y - scrollY }; }, [yPage, frac]);
  await page.touchscreen.tap(pt.x, pt.y);
};

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();

  // ---- 9: landscape phones: chart fits, readout on screen and painted on top after a tap in each panel
  for (const [nm, d] of [['844x390', dev(844, 390, 3, devices['iPhone 13'].userAgent)], ['640x360', dev(640, 360)]]) {
    const { ctx, page } = await open(browser, d);
    const g = await page.evaluate(() => ({ ih: innerHeight, coarse: matchMedia('(pointer: coarse)').matches, chartH: Math.round(document.getElementById('chart').getBoundingClientRect().height), roH: Math.round(document.getElementById('readout').getBoundingClientRect().height) }));
    const P = await panels(page), taps = {};
    for (const k of ['price', 'pnl', 'pos']) {
      await page.evaluate(y => window.scrollTo(0, y), Math.max(0, P[k][1] - g.ih + 2)); await page.waitForTimeout(200);   // panel bottom at the screen bottom
      await tapPanel(page, P[k][1] - Math.min(15, (P[k][1] - P[k][0]) / 2)); await page.waitForTimeout(500);
      taps[k] = await page.evaluate(() => {
        const ro = document.getElementById('readout'), r = ro.getBoundingClientRect(); let cov = 0, n = 0;
        for (let y = Math.max(0, Math.ceil(r.top) + 1); y < Math.min(innerHeight, r.bottom - 1); y += 3) for (let x = r.left + 4; x < r.right - 4; x += 20) { n++; const h = document.elementFromPoint(x, y); if (!h || !ro.contains(h)) cov++; }
        return { top: Math.round(r.top), bottom: Math.round(r.bottom), coveredFrac: n ? +(cov / n).toFixed(2) : 1, pinned: /UTC/.test(ro.innerText) };
      });
      taps[k].ok = taps[k].top >= 0 && taps[k].bottom <= g.ih && taps[k].coveredFrac === 0 && taps[k].pinned;
    }
    check(`F9 landscape ${nm}: chart+readout fit the screen, readout on screen and on top after a tap in each panel`,
      g.coarse && g.chartH + g.roH <= g.ih && Object.values(taps).every(t => t.ok), { innerH: g.ih, chartH: g.chartH, readoutH: g.roH, taps });
    await ctx.close();
  }

  // ---- follow-ups found in the re-check (info only): short landscape phones
  {
    const { ctx, page } = await open(browser, devices['Galaxy S9+ landscape']);   // 658x320
    await page.evaluate(() => { const zb = document.querySelector('.zoombar').getBoundingClientRect(); window.scrollTo(0, zb.bottom + scrollY - innerHeight); }); await page.waitForTimeout(200);
    const m = await page.evaluate(() => { const ro = document.getElementById('readout').getBoundingClientRect(), g = document.getElementById('chart'), a = g._fullLayout.yaxis, t = g.getBoundingClientRect().top + a._offset;
      return { ih: innerHeight, readout: [Math.round(ro.top), Math.round(ro.bottom)], pricePanel: [Math.round(t), Math.round(t + a._length)], priceHiddenFrac: +(Math.max(0, Math.min(t + a._length, ro.bottom) - Math.max(t, ro.top)) / a._length).toFixed(2) }; });
    check('N3 658x320 landscape: with the zoom bar on screen the stuck readout hides <= 10% of the price panel', m.priceHiddenFrac <= 0.1, m, true);
    await ctx.close();
  }
  {
    const { ctx, page } = await open(browser, devices['iPhone SE landscape']);    // 568x320: narrow Plotly layout on a 260 px chart
    const m = await page.evaluate(() => { const g = document.getElementById('chart'), fl = g._fullLayout, gr = g.getBoundingClientRect(), t = {};
      [...g.querySelectorAll('.annotation')].forEach(a => { if (/^(Net PnL \(%\)|Position \(x\))$/.test(a.textContent)) { const b = (a.querySelector('.bg') || a).getBoundingClientRect(); t[a.textContent] = [+(b.top - gr.top).toFixed(1), +(b.bottom - gr.top).toFixed(1)]; } });
      const e = k => +(fl[k]._offset + fl[k]._length).toFixed(1);
      return { titles: t, priceBottom: e('yaxis'), pnlBottom: e('yaxis2') }; });
    const ok = m.titles['Net PnL (%)'] && m.titles['Net PnL (%)'][0] >= m.priceBottom - 0.5 && m.titles['Position (x)'] && m.titles['Position (x)'][0] >= m.pnlBottom - 0.5;
    check('N2 568x320 landscape: PnL/Position titles do not overlap the panel above', ok, m, true);
    await ctx.close();
  }

  // ---- 360x740: 10 (tick labels), 15 (titles), 16 (All chip), 17 (select font), 19 (card fits)
  const ctl = {};
  for (const [nm, d] of [['360x740', dev(360, 740)], ['Pixel 7', devices['Pixel 7']]]) {
    const { ctx, page } = await open(browser, d);
    // 19: card from top to hint fits one screen when scrolled to the card top (hint state and pinned state)
    const fit = async () => { await page.evaluate(() => window.scrollTo(0, document.getElementById('card').getBoundingClientRect().top + scrollY)); await page.waitForTimeout(150);
      return page.evaluate(() => { const c = document.getElementById('card').getBoundingClientRect(), h = document.getElementById('hint').getBoundingClientRect(); return { ih: innerHeight, cardTop: Math.round(c.top), cardBottom: Math.round(c.bottom), hintBottom: Math.round(h.bottom) }; }); };
    const f1 = await fit();
    await pick(page, 'T2-NET'); await chip(page, 'seal');
    const P = await panels(page); await page.evaluate(y => window.scrollTo(0, y - 200), P.price[0]); await page.waitForTimeout(150);
    await tapPanel(page, (P.price[0] + P.price[1]) / 2, 0.3); await page.waitForTimeout(600);
    await pick(page, 'tsmom_unscaled');
    const f2 = await fit();
    check(`F19 ${nm}: chart card (top to zoom-bar hint) fits one screen when scrolled to the card top`,
      [f1, f2].every(f => f.cardBottom <= f.ih && f.hintBottom <= f.ih), { atLoad: f1, pinnedTsmom: f2 });
    // 10: two-line date tick labels inside the svg
    const ticks = {};
    for (const k of ['crash', '1d']) {
      await chip(page, k);
      ticks[k] = await page.evaluate(() => { const g = document.getElementById('chart'), svg = g.querySelector('svg.main-svg'), sr = svg.getBoundingClientRect();
        const vis = [...g.querySelectorAll('.xtick text')].filter(t => getComputedStyle(t).opacity !== '0');
        return { svgH: Math.round(sr.height), lowestLabelBottom: Math.round(Math.max(...vis.map(t => t.getBoundingClientRect().bottom - sr.top))), marginB: g._fullLayout.margin.b, twoLine: vis.filter(t => t.querySelectorAll('tspan.line').length > 1).map(t => t.textContent) }; });
    }
    check(`F10 ${nm}: two-line x tick labels not clipped at the bottom`, Object.values(ticks).every(t => t.lowestLabelBottom <= t.svgH && t.twoLine.length), ticks);
    // 15: PnL/Position titles sit in the gaps above their panels; no drawn point under any title
    const cases = [['T2-NET', '1m'], ['T2-LIN', 'dev'], ['T1-NT', '1w'], ['T1-FADEV', '1m'], ['T2-LIN', 'all']], tres = [];
    for (const [sid, ck] of cases) {
      await pick(page, sid); await chip(page, ck);
      tres.push({ sid, ck, ...(await page.evaluate(() => {
        const g = document.getElementById('chart'), fl = g._fullLayout, gr = g.getBoundingClientRect(), xa = fl.xaxis, axes = { y: fl.yaxis, y2: fl.yaxis2, y3: fl.yaxis3 };
        const dom = k => [axes[k]._offset, axes[k]._offset + axes[k]._length], out = { hits: {}, inGap: {} };
        [...g.querySelectorAll('.annotation')].forEach(a => {
          const t = a.textContent; if (!/^(Price \(USDT\)|Net PnL \(%\)|Position \(x\))$/.test(t)) return;
          const b = (a.querySelector('.bg') || a).getBoundingClientRect(), box = { x0: b.left - gr.left, x1: b.right - gr.left, y0: b.top - gr.top, y1: b.bottom - gr.top };
          let n = 0;
          g._fullData.forEach(tr => { if (!tr.visible || !tr.x) return; const ya = axes[tr.yaxis];
            for (let i = 0; i < tr.x.length; i++) { if (tr.x[i] == null || tr.y[i] == null) continue;
              const px = xa._offset + xa.d2p(tr.x[i]), py = ya._offset + ya.d2p(tr.y[i]);
              if (px < xa._offset || px > xa._offset + xa._length || py < ya._offset - 0.5 || py > ya._offset + ya._length + 0.5) continue;
              if (px >= box.x0 && px <= box.x1 && py >= box.y0 && py <= box.y1) n++; } });
          out.hits[t] = n;
          if (t === 'Net PnL (%)') out.inGap[t] = [+box.y0.toFixed(1), +box.y1.toFixed(1), 'gap', +dom('y')[1].toFixed(1), +dom('y2')[0].toFixed(1)];
          if (t === 'Position (x)') out.inGap[t] = [+box.y0.toFixed(1), +box.y1.toFixed(1), 'gap', +dom('y2')[1].toFixed(1), +dom('y3')[0].toFixed(1)];
        });
        return out;
      })) });
    }
    const gapOk = r => Object.values(r.inGap).length === 2 && Object.values(r.inGap).every(v => v[0] >= v[3] - 0.5 && v[1] <= v[4] + 0.5);
    check(`F15 ${nm}: panel titles hide no data (PnL and Position titles in the gaps)`, tres.every(r => gapOk(r) && Object.values(r.hits).every(n => n === 0)), tres);
    ctl[nm] = await page.evaluate(() => { const a = document.querySelector('#seg-range button[data-key=all]').getBoundingClientRect(), s = document.getElementById('strat-select');
      return { coarse: matchMedia('(pointer: coarse)').matches, allChip: [+a.width.toFixed(1), +a.height.toFixed(1)], selectFont: parseFloat(getComputedStyle(s).fontSize), selectShown: s.getClientRects().length > 0 }; });
    await ctx.close();
  }
  check('F16: All range chip >= 40 x 40 px on touch phones', Object.values(ctl).every(c => c.coarse && c.allChip[0] >= 40 && c.allChip[1] >= 40), ctl);
  check('F17: strategy select font >= 16 px (no iOS focus zoom)', Object.values(ctl).every(c => c.selectShown && c.selectFont >= 16), ctl);

  // ---- 14: readout keeps one height across the hint, taps and strategy switches (ResizeObserver logs every height)
  for (const [nm, d] of [['320', dev(320, 568, 2, devices['iPhone SE'].userAgent)], ['360', dev(360, 740)], ['390', dev(390, 664, 3, devices['iPhone 13'].userAgent)], ['412', dev(412, 839, 2.625)]]) {
    const { ctx, page } = await open(browser, d);
    for (const id of ['T2-NET', 'tsmom_unscaled', 'B&H', 'T2-P']) await pick(page, id);   // preload
    await pick(page, 'T2-NET'); await chip(page, 'seal');
    await page.evaluate(() => { const ro = document.getElementById('readout'); window.__H = [Math.round(ro.getBoundingClientRect().height * 10) / 10];
      new ResizeObserver(() => window.__H.push(Math.round(ro.getBoundingClientRect().height * 10) / 10)).observe(ro);
      window.__T = new Set(); new MutationObserver(() => { if (/UTC/.test(ro.innerText)) window.__T.add(ro.innerText.replace(/\s+/g, ' ')); }).observe(ro, { childList: true, subtree: true, characterData: true }); });
    const P = await panels(page); await page.evaluate(y => window.scrollTo(0, y - 200), P.price[0]); await page.waitForTimeout(150);
    for (const f of [0.12, 0.5, 0.85]) {
      await pick(page, 'T2-NET');
      await tapPanel(page, (P.price[0] + P.price[1]) / 2, f); await page.waitForTimeout(500);
      for (const id of ['tsmom_unscaled', 'B&H', 'T2-P', 'T2-NET']) await pick(page, id);
    }
    // B&H selected while tapping a long trade (the widest trade line)
    await pick(page, 'B&H'); await tapPanel(page, (P.price[0] + P.price[1]) / 2, 0.6); await page.waitForTimeout(500); await pick(page, 'tsmom_unscaled');
    const H = await page.evaluate(() => window.__H);
    const uniq = [...new Set(H)];
    const T = await page.evaluate(() => [...window.__T]), maxLen = Math.max(0, ...T.map(t => t.length));
    check(`F14 ${nm}px: readout height constant across hint, taps and strategy switches`, uniq.length === 1 && T.length >= 12,
      { heights: uniq, resizeSamples: H.length, pinnedReadouts: T.length, longest: T.find(t => t.length === maxLen) });
    if (nm === '360') {
      // follow-up (introduced with the "Price (HH:MM)" readout label): tap with tsmom_unscaled selected in the All view,
      // before the tapped month is loaded -> the box is one line taller until the month file arrives
      await chip(page, 'all'); await pick(page, 'tsmom_unscaled');
      await page.evaluate(() => { window.__H = [Math.round(document.getElementById('readout').getBoundingClientRect().height * 10) / 10]; });
      const P2 = await panels(page);
      for (const f of [0.1, 0.45, 0.8]) { await tapPanel(page, (P2.price[0] + P2.price[1]) / 2, f); await page.waitForTimeout(500); }
      const H2 = await page.evaluate(() => window.__H);
      check('N1 360px: readout height constant on taps before the tapped month is loaded (All view, tsmom_unscaled)', new Set(H2).size === 1, { heights: [...new Set(H2)] }, true);
    }
    await ctx.close();
  }

  await browser.close();
  const counted = results.filter(r => STRICT || !r.info), failed = counted.filter(r => !r.pass);
  console.log(JSON.stringify({ summary: { page: PAGE, checks: counted.length, passed: counted.length - failed.length, failed: failed.map(r => r.name), info: results.filter(r => r.info).map(r => ({ name: r.name, pass: r.pass })), seconds: Math.round((Date.now() - t0) / 1000) } }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('ERROR', e); process.exit(2); });
