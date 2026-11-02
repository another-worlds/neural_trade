// Safe-area regression for the BTCUSDT replay viewer: iPhones with a notch or home indicator, emulated with CDP
// Emulation.setSafeAreaInsetsOverride (Chromium 136 or later). The published page uses viewport-fit=cover, so the
// side insets in landscape (47-59 px) narrow the page, and the root gets top and bottom padding from the insets.
// Each case runs with and without its insets. Pass: no horizontal overflow, the same readout height, the stuck
// landscape readout covers <= 1 px of the chart and hides <= 0.5 px of the 'Sealed year' label with the zoom bar's
// bottom at the screen bottom, a tap pins a time (longest strategy name), and no page errors.
// Usage: PLOTLY=/path/plotly-2.34.0.min.js [PAGE=test.html] [ROBOTO=/path/roboto.woff2] node regress_safe_area.js
// Prints one JSON line per case and a summary line; exits 1 on any failed check.
const { chromium, devices } = require('playwright'); const fs = require('fs');
const PL = fs.readFileSync(process.env.PLOTLY); const PAGE = process.env.PAGE || 'test.html';
const RB = process.env.ROBOTO ? fs.readFileSync(process.env.ROBOTO) : null;
const mk = (w, h) => ({ viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: 3, isMobile: true, hasTouch: true, userAgent: devices['iPhone 13'].userAgent });
const CASES = [
  ['iPhone 13 landscape', devices['iPhone 13 landscape'], { top: 0, left: 47, right: 47, bottom: 21 }],
  ['iPhone 15 Pro Max landscape 932x430', mk(932, 430), { top: 0, left: 59, right: 59, bottom: 21 }],
  ['iPhone 12 mini landscape 812x375', mk(812, 375), { top: 0, left: 50, right: 50, bottom: 21 }],
  ['iPhone 13 portrait (bar collapsed)', devices['iPhone 13'], { top: 47, left: 0, right: 0, bottom: 34 }],
  ['iPhone SE 3 portrait (no notch)', mk(375, 667), { top: 0, left: 0, right: 0, bottom: 0 }],
];
const M = () => {
  const R = e => e.getBoundingClientRect(), ro = R(document.getElementById('readout')), g = R(document.getElementById('chart')), zb = R(document.querySelector('.zoombar'));
  const a = [...document.querySelectorAll('#chart .annotation')].find(x => x.textContent === 'Sealed year'), t = a ? R(a.querySelector('text')) : null;
  const btn = [...document.querySelectorAll('.zoombar button')].map(R);
  return { iw: innerWidth, ih: innerHeight, sw: document.documentElement.scrollWidth, padL: getComputedStyle(document.body).paddingLeft, rootPadB: getComputedStyle(document.documentElement).paddingBottom,
    roH: +ro.height.toFixed(1), roTop: +ro.top.toFixed(1), chartTop: Math.round(g.top), chartH: Math.round(g.height), chartW: Math.round(g.width), zbBottom: +zb.bottom.toFixed(1), zbH: Math.round(zb.height),
    labelHidden: t ? +Math.max(0, Math.min(ro.bottom, t.bottom) - t.top).toFixed(1) : null, chartCovered: +Math.max(0, ro.bottom - g.top).toFixed(1),
    btnMinW: Math.round(Math.min(...btn.map(b => b.width))), btnInBottomInset: null, roLines: Math.round(ro.height / 20.3) };
};
(async () => {
  const t0 = Date.now(), results = [], noIns = {};
  const b = await chromium.launch();
  try { const c0 = await b.newContext(); const s0 = await c0.newCDPSession(await c0.newPage()); await s0.send('Emulation.setSafeAreaInsetsOverride', { insets: { top: 0, left: 0, right: 0, bottom: 0 } }); await c0.close(); }
  catch (e) { console.error('this Chromium has no Emulation.setSafeAreaInsetsOverride: ' + e.message); process.exit(2); }
  for (const [name, opts, ins] of CASES) for (const withIns of [false, true]) {
    if (!withIns && !(ins.left || ins.top || ins.bottom)) continue;
    const ctx = await b.newContext({ ...opts, colorScheme: 'light' }); const p = await ctx.newPage(); const errs = [];
    p.on('pageerror', e => errs.push(e.message));
    await p.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
    if (RB) { await p.route('**/__rb.woff2', r => r.fulfill({ body: RB, contentType: 'font/woff2' }));
      await p.route('**/' + PAGE, async r => { const resp = await r.fetch(); let s = await resp.text(); s = s.replace('</style>\n', '</style>\n').replace(/<\/style>(?![\s\S]*<\/style>)/, '@font-face{font-family:"RobotoT";src:url(/__rb.woff2) format("woff2");font-weight:100 900}:root{--font:"RobotoT",sans-serif !important}body,button,select{font-family:"RobotoT",sans-serif !important}</style>'); r.fulfill({ response: resp, body: s, headers: { ...resp.headers(), 'content-type': 'text/html' } }); }); }
    const s = await ctx.newCDPSession(p);
    if (withIns) await s.send('Emulation.setSafeAreaInsetsOverride', { insets: ins });
    await p.goto('http://localhost:8765/' + PAGE);
    await p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 }); await p.evaluate(() => document.fonts.ready);
    await p.evaluate(() => document.querySelector('#seg-range button[data-key="seal"]').click()); await p.waitForTimeout(1500);
    const top = await p.evaluate(M);
    await p.evaluate(() => { const z = document.querySelector('.zoombar').getBoundingClientRect(); window.scrollBy(0, z.bottom - innerHeight); }); await p.waitForTimeout(300);
    const scrolled = await p.evaluate(M);
    // pinned readout on the longest strategy name
    await p.evaluate(() => { const s = document.getElementById('strat-select'); s.value = 'tsmom_unscaled'; s.dispatchEvent(new Event('change')); });
    await p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 }); await p.waitForTimeout(800);
    await p.evaluate(() => document.querySelector('#seg-range button[data-key="seal"]').click()); await p.waitForTimeout(1200);
    await p.evaluate(() => window.scrollTo(0, 0)); await p.waitForTimeout(200);
    const c = await p.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(); return { y: r.top + s.t + s.h * 0.4 }; });
    await p.evaluate(y => window.scrollBy(0, y - innerHeight / 2), c.y); await p.waitForTimeout(250);
    const c2 = await p.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(); return { x: r.left + s.l + s.w * 0.6, y: r.top + s.t + s.h * 0.45 }; });
    await p.touchscreen.tap(c2.x, c2.y); await p.waitForTimeout(800);
    await p.evaluate(() => { const z = document.querySelector('.zoombar').getBoundingClientRect(); window.scrollBy(0, z.bottom - innerHeight); }); await p.waitForTimeout(300);
    const pinned = await p.evaluate(M), ro = await p.$eval('#readout', e => e.innerText.replace(/\s+/g, ' ').trim().slice(0, 90));
        const land = top.iw > top.ih, base = withIns ? noIns[name] : null;
    const ok = top.sw <= top.iw && scrolled.sw <= scrolled.iw && /UTC/.test(ro) && /tsmom_unscaled/.test(ro) && !errs.length &&
      (!land || (scrolled.chartCovered <= 1 && pinned.chartCovered <= 1 && scrolled.labelHidden <= 0.5 && pinned.labelHidden <= 0.5)) &&
      (!base || (Math.abs(base.roH - top.roH) < 0.5 && Math.abs(base.pinRoH - pinned.roH) < 0.5));
    if (!withIns) noIns[name] = { roH: top.roH, pinRoH: pinned.roH };
    const r = { name, insets: withIns ? ins : 'none', pass: ok, top, scrolled, pinned, ro, errs }; results.push(r); console.log(JSON.stringify(r));
    await ctx.close();
  }
  await b.close();
  const failed = results.filter(r => !r.pass).map(r => r.name + (r.insets === 'none' ? ' (no insets)' : ' (insets)'));
  console.log(JSON.stringify({ summary: true, page: PAGE, checks: results.length, passed: results.length - failed.length, failed, seconds: Math.round((Date.now() - t0) / 1000) }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
