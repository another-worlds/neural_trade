// Layout regression for the round-3 'layout' cluster of the BTCUSDT backtest viewer (findings 2, 3, 4, 7).
//   F2  short landscape charts (plot area < 200 px): no in-plot "Price (USDT)" title and no 24 px headroom, so the
//       price line gets most of its panel and at least one price label sits on the data; taller compact charts keep both.
//   F3  701-960 px landscape phones: with the whole zoom bar on screen, the stuck readout does not hide the
//       'Sealed year' label (chart height budgets the 3-line readout: 100svh - 122 px).
//   F4  compact y axes: at least 2 price and 2 PnL labels, PnL labels keep the % suffix, no overlapping or duplicate
//       labels, position [-1, 1] or [-1, 0, 1]; desktop keeps Plotly's own ticks.
//   F7  short coarse-pointer windows: the zoom-bar hint sits beside the buttons only with >= 12em, else on its own row,
//       with no horizontal overflow; landscape phones and normal portrait phones keep their layout.
// Final review round:
//   F8  a short zoom on a flat PnL (B&H off, or the B&H strategy) shows distinct PnL labels, compact and desktop.
//   F9  a height-only resize (portrait split-screen) redraws: the layout equals a fresh load at the new size; a window
//       resize that leaves the chart box unchanged does not redraw.
// Usage: PLOTLY=/path/plotly-2.34.0.min.js [PAGE=test.html] [BASE=http://localhost:8765/] [ROBOTO=/path/roboto.woff2] node regress_r3_layout.js
// Prints one JSON line per check and a summary line; exits 1 on any failed counted check. Info checks never fail the run.
const { chromium, devices } = require('playwright');
const fs = require('fs');

if (!process.env.PLOTLY) { console.error('PLOTLY (path to plotly-2.34.0.min.js) is required'); process.exit(2); }
const PL = fs.readFileSync(process.env.PLOTLY);
const PAGE = process.env.PAGE || 'test.html';
const BASE = (process.env.BASE || 'http://localhost:8765/').replace(/\/?$/, '/');
const ROBOTO = process.env.ROBOTO ? fs.readFileSync(process.env.ROBOTO) : null;
const UA = devices['Pixel 7'].userAgent;
const mk = (w, h) => ({ viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: UA });

const results = [];
function check(id, finding, ok, data, info = false) {
  const r = { id, finding, pass: !!ok, ...(info ? { info: true } : {}), ...data };
  results.push(r); console.log(JSON.stringify(r));
}

async function open(browser, ctxOpts, { font = false } = {}) {
  const ctx = await browser.newContext({ colorScheme: 'light', ...ctxOpts });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  if (font && ROBOTO) {
    await page.route('**/__roboto.woff2', r => r.fulfill({ body: ROBOTO, contentType: 'font/woff2' }));
    await page.route('**/' + PAGE, async r => {
      const resp = await r.fetch(); let body = await resp.text();
      body = body.replace('</style>', '@font-face{font-family:"RobotoT";src:url(/__roboto.woff2) format("woff2");font-weight:100 900}\n:root{--font:"RobotoT", sans-serif !important}\nbody,button,select{font-family:"RobotoT",sans-serif !important}\n</style>');
      r.fulfill({ response: resp, body, headers: { ...resp.headers(), 'content-type': 'text/html' } });
    });
  }
  await page.goto(BASE + PAGE);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await page.evaluate(() => document.fonts.ready);
  return { ctx, page, errs };
}
// waits until the chart stops changing (x range, resolution note, trace lengths) and no load is pending
async function idle(page) {
  let prev = null;
  for (let k = 0; k < 60; k++) {
    await page.waitForTimeout(200);
    const cur = await page.evaluate(() => { const g = document.getElementById('chart'); return JSON.stringify([g._fullLayout && g._fullLayout.xaxis.range, document.getElementById('res').textContent, g.data && g.data.map(t => t.x.length), g._fullLayout && g._fullLayout.yaxis.range]); });
    if (cur === prev && !/Loading/.test(cur)) return;
    prev = cur;
  }
}
const chip = async (page, key) => { await page.evaluate(k => document.querySelector('#seg-range button[data-key="' + k + '"]').click(), key); await idle(page); };
const strat = async (page, id) => {
  await page.evaluate(i => { const s = document.getElementById('strat-select'); s.value = i; s.dispatchEvent(new Event('change')); }, id);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 }); await idle(page);
};
const relayout = async (page, t0, t1) => {
  await page.evaluate(([a, b]) => { const g = document.getElementById('chart'), iso = ms => new Date(ms).toISOString().replace('T', ' ').replace('Z', ''); Plotly.relayout(g, { 'xaxis.range': [iso(a), iso(b)] }); }, [t0, t1]);
  if (t1 - t0 < 4 * 864e5) await page.waitForFunction(() => /1-min/.test(document.getElementById('res').textContent), null, { timeout: 15000 }).catch(() => {});
  await idle(page);
};

// y axes of the compact chart: labels, their boxes, data extents in view, in-plot titles
const AXES = () => {
  const g = document.getElementById('chart'), fl = g._fullLayout, xa = fl.xaxis, gr = g.getBoundingClientRect();
  const lab = id => [...g.querySelectorAll('.' + id + 'tick text')].map(t => { const b = t.getBoundingClientRect(); return { s: t.textContent, t: b.top - gr.top, b: b.bottom - gr.top, l: b.left - gr.left, r: b.right - gr.left }; });
  const ext = ya => { let lo = Infinity, hi = -Infinity; g._fullData.forEach(t => { if (t.yaxis !== ya || !t.x) return; for (let i = 0; i < t.x.length; i++) { if (t.y[i] == null) continue; const x = xa.d2p(t.x[i]); if (x < 0 || x > xa._length) continue; lo = Math.min(lo, t.y[i]); hi = Math.max(hi, t.y[i]); } }); return [lo, hi]; };
  const titles = [...g.querySelectorAll('.annotation')].map(a => a.textContent).filter(s => /^(Price \(USDT\)|Net PnL \(%\)|Position \(x\))$/.test(s));
  const P = ext('y'), ya = fl.yaxis;
  return { ch: g.clientHeight, plotH: g.clientHeight - 54, compact: fl.margin.l === 40,
    y: lab('y'), y2: lab('y2'), y3: lab('y3'), titles,
    priceP: ya._length, priceData: P, priceTopPx: ya.d2p(P[1]), priceDataPx: ya.d2p(P[0]) - ya.d2p(P[1]) };
};
const num = s => +s.replace(/−/g, '-').replace(/[^0-9.\-]/g, '');
function labelProblems(a) {
  const p = [];
  for (const k of ['y', 'y2', 'y3']) {
    const L = a[k].slice().sort((u, v) => u.t - v.t), txt = L.map(x => x.s);
    if (new Set(txt).size !== txt.length) p.push(k + ' duplicate ' + txt.join(' '));
    for (let i = 1; i < L.length; i++) if (L[i].t < L[i - 1].b - 0.5) p.push(k + ' overlap ' + L[i - 1].s + '|' + L[i].s);
  }
  const all = [].concat(a.y, a.y2, a.y3);   // across panels
  for (let i = 0; i < all.length; i++) for (let j = i + 1; j < all.length; j++) {
    const u = all[i], v = all[j];
    if (Math.min(u.r, v.r) - Math.max(u.l, v.l) > 0.5 && Math.min(u.b, v.b) - Math.max(u.t, v.t) > 0.5 && !(a.y.includes(u) && a.y.includes(v)) && !(a.y2.includes(u) && a.y2.includes(v)) && !(a.y3.includes(u) && a.y3.includes(v))) p.push('cross-panel overlap ' + u.s + '|' + v.s);
  }
  return p;
}
const brief = a => ({ ch: a.ch, plotH: a.plotH, price: a.y.map(x => x.s), pnl: a.y2.map(x => x.s), pos: a.y3.map(x => x.s), titles: a.titles,
  priceP: +a.priceP.toFixed(1), priceDataPx: +a.priceDataPx.toFixed(1), priceTopPx: +a.priceTopPx.toFixed(1) });

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();

  // ---------- F2 + F4: compact charts, landscape and short portrait
  const COMPACT = [
    ['iPhone SE landscape 568x320', devices['iPhone SE landscape']],
    ['Pixel 5 landscape 802x293', devices['Pixel 5 landscape']],
    ['iPhone 13 landscape 750x342', devices['iPhone 13 landscape']],
    ['844x390', mk(844, 390)],
    ['375x548', mk(375, 548)],
  ];
  for (const [dn, opts] of COMPACT) {
    const { ctx, page, errs } = await open(browser, opts);
    const coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
    const views = [];
    for (const key of ['all', 'seal', 'crash', '1d']) { await chip(page, key); views.push([key, await page.evaluate(AXES)]); }
    if (/750x342/.test(dn)) {   // the round-2 worst pan: T1-NET, 1 day, an entry that is the view maximum
      await strat(page, 'T1-NET'); const t = 1738953600000, S = 864e5, f = 25 / 670;
      await relayout(page, t - f * S, t + (1 - f) * S); views.push(['T1-NET pan 1d @2025-02-07T18:40', await page.evaluate(AXES)]);
    }
    for (const [key, a] of views) {
      const v = dn + ' ' + key;
      // F4: a readable scale on every compact panel
      const probs = labelProblems(a);
      const suf = a.y2.every(x => /%$/.test(x.s));
      const pos = JSON.stringify(a.y3.map(x => num(x.s)));
      check('F4-ticks ' + v, 4, coarse && a.compact && a.y.length >= 2 && a.y2.length >= 2 && suf && !probs.length && (pos === '[-1,1]' || pos === '[-1,0,1]'),
        { need: '>=2 price and >=2 PnL labels, PnL labels end in %, position [-1,1] or [-1,0,1], no overlapping or duplicate labels', problems: probs, ...brief(a) });
      // F2: price title and headroom only on charts whose plot area is >= 200 px
      const P = a.priceData, onData = a.y.filter(x => { const q = num(x.s); return q >= P[0] && q <= P[1]; }).length;
      if (a.plotH < 200) {
        check('F2-short ' + v, 2, !a.titles.includes('Price (USDT)') && a.titles.includes('Net PnL (%)') && a.titles.includes('Position (x)') && a.priceDataPx >= 0.75 * a.priceP && onData >= 1,
          { need: 'plot < 200 px: no Price title, PnL and Position titles kept, price data >= 75% of its panel, >= 1 price label on the data', onData, ...brief(a) });
      } else {
        check('F2-tall ' + v, 2, a.titles.includes('Price (USDT)') && a.priceTopPx >= 23.5 && onData >= 1,
          { need: 'plot >= 200 px: Price title kept with >= 24 px headroom above the data, >= 1 price label on the data', onData, ...brief(a) });
      }
    }
    if (errs.length) check('page errors ' + dn, 0, false, { errs });
    await ctx.close();
  }

  // ---------- F4 info: desktop keeps Plotly's own ticks
  {
    const { ctx, page } = await open(browser, { viewport: { width: 1280, height: 800 } });
    await strat(page, 'T1-LIN');   // the thresholds below were set on T1-LIN (the page now opens on T2-P)
    await chip(page, 'all'); const a = await page.evaluate(AXES);
    check('F4-desktop 1280x800 all', 4, !a.compact && a.y.length >= 5 && a.y2.length >= 4 && !labelProblems(a).length, { need: 'wide layout unchanged: >= 5 price, >= 4 PnL labels', ...brief(a) });
    await ctx.close();
  }

  // ---------- F8 (final round): a short zoom on a flat PnL keeps distinct PnL labels (the PnL range is >= 0.1 pp,
  //            so the tick step is never finer than the 2-decimal labels); B&H off where the PnL panel shows only the strategy
  const FLAT = [   // [name, context, strategy, B&H on, view start (UTC ms), minutes]
    ['568x320 B&H 20 min @2024-02-08T02:20', devices['iPhone SE landscape'], 'B&H', true, Date.UTC(2024, 1, 8, 2, 20), 20],
    ['750x342 T1-NET no B&H 20 min @2021-03-29T07:07', devices['iPhone 13 landscape'], 'T1-NET', false, Date.UTC(2021, 2, 29, 7, 7), 20],
    ['390x844 T1-LIN no B&H 60 min @2026-05-15T06:27', mk(390, 844), 'T1-LIN', false, Date.UTC(2026, 4, 15, 6, 27), 60],
    ['1280x800 T1-FADEV no B&H 20 min @2025-05-27T13:03', { viewport: { width: 1280, height: 800 } }, 'T1-FADEV', false, Date.UTC(2025, 4, 27, 13, 3), 20],
  ];
  for (const [dn, opts, sid, bhOn, t, min] of FLAT) {
    const { ctx, page, errs } = await open(browser, opts);
    await strat(page, sid);
    if (!bhOn) { await page.evaluate(() => { const b = document.getElementById('bh-toggle'); if (b.checked) b.click(); }); await idle(page); }
    await relayout(page, t, t + min * 6e4); const a = await page.evaluate(AXES);
    const txt = a.y2.map(x => x.s), probs = labelProblems(a);
    check('F8-flat-pnl ' + dn, 8, txt.length >= 2 && new Set(txt).size === txt.length && txt.every(s => /%$/.test(s)) && !probs.length && !errs.length,
      { need: '>= 2 distinct PnL labels ending in %, no overlapping or duplicate labels', problems: probs, errs, ...brief(a) });
    await ctx.close();
  }

  // ---------- F9 (final round): a height-only resize (portrait split-screen) redraws the compact layout: titles, gaps
  //            and ticks equal a fresh load at the new size
  const LAYOUT = () => {
    const g = document.getElementById('chart'), fl = g._fullLayout, d = k => fl[k].domain.map(x => +x.toFixed(3));
    const t = id => [...g.querySelectorAll('.' + id + 'tick text')].map(e => e.textContent).join('|');
    return { chartH: g.clientHeight, dom: [d('yaxis'), d('yaxis2'), d('yaxis3')], y: t('y'), y2: t('y2'), y3: t('y3'),
      titles: [...g.querySelectorAll('.annotation-text')].map(a => a.textContent).filter(s => s !== 'Sealed year').join('|') };
  };
  for (const [w, h0, h1, key] of [[393, 727, 330, 'all'], [412, 420, 320, 'seal'], [393, 330, 727, 'seal']]) {
    const A = await open(browser, mk(w, h0)); await chip(A.page, key);
    await A.page.setViewportSize({ width: w, height: h1 }); await A.page.waitForTimeout(1500); await idle(A.page);
    const s1 = await A.page.evaluate(LAYOUT), p1 = labelProblems(await A.page.evaluate(AXES));
    const B = await open(browser, mk(w, h1)); await chip(B.page, key);
    const s2 = await B.page.evaluate(LAYOUT);
    const diff = Object.keys(s1).filter(k => JSON.stringify(s1[k]) !== JSON.stringify(s2[k]));
    check(`F9-height-resize ${w}x${h0}->${h1} ${key}`, 9, !diff.length && !p1.length && !A.errs.length,
      { need: 'after a height-only resize the layout equals a fresh load, no overlapping labels', differs: diff, problems: p1, resized: s1, fresh: s2 });
    await A.ctx.close(); await B.ctx.close();
  }
  {   // a window resize that leaves the chart box unchanged (chart capped at 760 px) does not redraw
    const { ctx, page } = await open(browser, { viewport: { width: 1280, height: 1000 } });
    await chip(page, 'all');
    const h0 = await page.evaluate(() => { window.__n = 0; const o = Plotly.react; Plotly.react = function () { window.__n++; return o.apply(this, arguments); }; return document.getElementById('chart').clientHeight; });
    await page.setViewportSize({ width: 1280, height: 950 }); await page.waitForTimeout(800);
    const m = await page.evaluate(() => ({ chartH: document.getElementById('chart').clientHeight, reacts: window.__n }));
    check('F9-no-extra-render 1280x1000->950', 9, m.chartH === h0 && m.reacts === 0, { need: 'chart box unchanged, 0 Plotly.react calls', before: h0, ...m });
    await ctx.close();
  }

  // ---------- F3: 701-960 px landscape phones, sealed view, scrolled so the zoom bar's bottom is at the screen bottom
  const F3DEV = ['iPhone 12 Mini landscape', 'iPhone 13 landscape', 'Pixel 7 landscape', 'iPhone 16 Pro Max landscape'].filter(n => devices[n]);
  const F3M = () => {
    const R = e => e.getBoundingClientRect(), ro = R(document.getElementById('readout'));
    const a = [...document.querySelectorAll('#chart .annotation')].find(x => x.textContent === 'Sealed year');
    const t = R(a.querySelector('text')), g = R(document.getElementById('chart')), zb = R(document.querySelector('.zoombar'));
    return { vw: innerWidth, vh: innerHeight, sticky: getComputedStyle(document.getElementById('readout')).position, roH: +ro.height.toFixed(1), chartH: Math.round(g.height),
      zbBottom: +zb.bottom.toFixed(1), labelHidden: +Math.max(0, Math.min(ro.bottom, t.bottom) - t.top).toFixed(1), chartCovered: +Math.max(0, ro.bottom - g.top).toFixed(1) };
  };
  const scrollC = page => page.evaluate(() => { const z = document.querySelector('.zoombar').getBoundingClientRect(); window.scrollBy(0, z.bottom - innerHeight); });
  for (const dn of F3DEV) {
    const { ctx, page } = await open(browser, devices[dn]);
    await chip(page, 'seal'); await scrollC(page); await page.waitForTimeout(250);
    const m = await page.evaluate(F3M);
    check('F3-sealed ' + dn, 3, m.sticky === 'sticky' && m.vw > 700 && m.vw <= 960 && m.labelHidden <= 0.5 && m.chartCovered <= 1 && Math.abs(m.zbBottom - m.vh) < 1.5,
      { need: "zoom bar bottom at the screen bottom: 'Sealed year' label hidden <= 0.5 px, chart covered <= 1 px", ...m });
    if (/iPhone 1[23]/.test(dn)) {   // a pinned readout (longest strategy name, B&H on) keeps its 3-line height
      await strat(page, 'tsmom_unscaled'); await chip(page, 'seal'); await page.evaluate(() => window.scrollTo(0, 0));
      const box = await page.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(); return { x: r.left + s.l + s.w * 0.6, y: r.top + s.t + s.h * 0.3 }; });
      await page.evaluate(([x, y]) => window.scrollBy(0, y - innerHeight / 2), [box.x, box.y]); await page.waitForTimeout(200);
      const b2 = await page.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(); return { x: r.left + s.l + s.w * 0.6, y: r.top + s.t + s.h * 0.45 }; });
      await page.touchscreen.tap(b2.x, b2.y); await page.waitForTimeout(800);
      await scrollC(page); await page.waitForTimeout(250);
      const m2 = await page.evaluate(F3M), ro = await page.$eval('#readout', e => e.innerText.replace(/\s+/g, ' ').trim());
      check('F3-sealed-pinned ' + dn, 3, /UTC/.test(ro) && m2.labelHidden <= 0.5 && m2.chartCovered <= 1, { need: 'pinned readout: label hidden <= 0.5 px', readout: ro.slice(0, 120), ...m2 });
    }
    await ctx.close();
  }

  // ---------- F7: zoom-bar hint on coarse-pointer windows
  const HINT = [   // [w, h, expect]: 'row' = own full-width row, 'beside' = next to the buttons with >= 12em
    [320, 460, 'row'], [360, 480, 'row'], [412, 430, 'row'], [370, 340, 'row'], [340, 320, 'row'],
    [568, 320, 'beside'], [667, 375, 'beside'], [360, 740, 'row'], [412, 915, 'row'],
  ];
  for (const [w, h, expect] of HINT) {
    const { ctx, page } = await open(browser, mk(w, h), { font: true });
    const m = await page.evaluate(() => {
      const H = document.getElementById('hint'), hr = H.getBoundingClientRect(), zb = document.querySelector('.zoombar'), z = zb.getBoundingClientRect();
      const rs = document.getElementById('z-reset').getBoundingClientRect(), cs = getComputedStyle(zb), fs = parseFloat(getComputedStyle(H).fontSize);
      const inner = z.width - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight), lh = parseFloat(getComputedStyle(H).lineHeight) || fs * 1.45;
      return { coarse: matchMedia('(pointer: coarse)').matches, iw: innerWidth, sw: document.documentElement.scrollWidth, hintW: +hr.width.toFixed(1), hintH: Math.round(hr.height),
        lines: Math.round(hr.height / lh), em12: 12 * fs, innerW: +inner.toFixed(1), sameRow: Math.abs(hr.top - rs.top) < 25 && hr.left > rs.right,
        ownRow: hr.top >= rs.bottom - 1, hintOverflow: H.scrollWidth - H.clientWidth, zoombarH: Math.round(z.height), text: H.textContent.slice(0, 14) };
    });
    const noOverflow = m.sw <= m.iw && m.hintOverflow <= 1;
    const placed = expect === 'row' ? m.ownRow && m.hintW >= m.innerW - 1 && m.lines <= 2 : m.sameRow && m.hintW >= m.em12 - 0.5 && m.lines <= 3;
    check(`F7-hint ${w}x${h}`, 7, m.coarse && /Swipe/.test(m.text) && noOverflow && placed,
      { need: expect === 'row' ? 'hint on its own full-width row, <= 2 lines, no horizontal overflow' : 'hint beside the buttons with >= 12em, <= 3 lines, no overflow', font: ROBOTO ? 'roboto' : 'default', ...m });
    await ctx.close();
  }

  await browser.close();
  const counted = results.filter(r => !r.info), failed = counted.filter(r => !r.pass);
  const byF = {}; for (const r of counted) { const k = 'F' + r.finding; byF[k] = byF[k] || { pass: 0, fail: 0 }; byF[k][r.pass ? 'pass' : 'fail']++; }
  console.log(JSON.stringify({ summary: true, page: PAGE, font: ROBOTO ? 'roboto' : 'default', checks: counted.length, passed: counted.length - failed.length, failed: failed.length, byFinding: byF,
    failedIds: failed.map(r => r.id), info: results.filter(r => r.info).map(r => r.id + (r.pass ? ' ok' : ' FAIL')), seconds: Math.round((Date.now() - t0) / 1000) }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('FATAL', e); process.exit(1); });
