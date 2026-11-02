#!/usr/bin/env node
// Regression checks for the touch-gesture findings 1, 4, 8, 12 and 18 of the BTCUSDT replay viewer.
// Needs only playwright and fs. Env: PLOTLY = path of plotly 2.34.0 (default below), PAGE = page under
// http://localhost:8765/ (default test.html). Prints one JSON summary and exits 1 if any check fails.
//  1  finger-count changes restart the gesture from the view on screen (no jump back, no lost pan or zoom)
//  4  a render that lands while a finger is held does not snap the view back (1-minute load, theme, resize)
//  8  pen and mouse pan and tap on a coarse-pointer device; finger taps on the axis labels do not pin
// 12  pinching past the 20-minute minimum (and + at the minimum) does not shift the view sideways
// 18  a pinch with the fingers one above the other zooms (distance is Euclidean)
const { chromium, devices } = require('playwright');
const fs = require('fs');
const PLOTLY = process.env.PLOTLY;
const PAGE = process.env.PAGE || 'test.html';
const URL = 'http://localhost:8765/' + PAGE;
const PL = fs.readFileSync(PLOTLY);
const MIN = 60000, DAY = 864e5;
const toMs = s => Date.parse(String(s).replace(' ', 'T').slice(0, 23) + 'Z');
const iso = ms => new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
const fmt = r => iso(r[0]) + '..' + iso(r[1]);
const r3 = x => Math.round(x * 1000) / 1000;
const checks = [];
function check(name, pass, measured) { checks.push({ name, pass: !!pass, measured }); }

const PHONE = devices['Pixel 7'];
const SMALL = { viewport: { width: 360, height: 740 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: devices['Pixel 5'].userAgent };
const IPHONE = { viewport: { width: 390, height: 844 }, deviceScaleFactor: 3, isMobile: true, hasTouch: true, userAgent: devices['iPhone 13'].userAgent };

async function open(browser, dev, delays = {}) {
  const ctx = await browser.newContext({ ...dev });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  if (Object.keys(delays).length) await page.route('**/m/*.txt', async r => {
    const ms = delays[r.request().url().split('/').pop()] || 0; if (ms) await new Promise(z => setTimeout(z, ms)); r.continue().catch(() => {});
  });
  await page.goto(URL);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await page.waitForTimeout(400);
  const cdp = await ctx.newCDPSession(page);
  const T = {
    ctx, page, errs, cdp,
    tp: (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts }),
    mouse: (type, x, y, pt, buttons) => cdp.send('Input.dispatchMouseEvent', { type, x, y, button: 'left', buttons: buttons != null ? buttons : (type === 'mouseReleased' ? 0 : 1), clickCount: 1, pointerType: pt }),
    rng: async () => (await page.evaluate(() => document.getElementById('chart')._fullLayout.xaxis.range)).map(toMs),
    geo: () => page.evaluate(() => { const gd = document.getElementById('chart'), r = gd.getBoundingClientRect(), s = gd._fullLayout._size, xa = gd._fullLayout.xaxis;
      return { left: r.left, top: r.top, W: r.width, l: s.l, t: s.t, w: s.w, h: s.h, x0: r.left + xa._offset, len: xa._length }; }),
    toChart: async (off = 40) => { await page.evaluate(o => window.scrollTo(0, document.getElementById('chart').getBoundingClientRect().top + scrollY - o), off); await page.waitForTimeout(250); },
    btn: async (sel, wait = 600) => { await page.evaluate(s => document.querySelector(s).click(), sel); await page.waitForTimeout(wait); },
    cursor: () => page.evaluate(() => (document.getElementById('chart').layout.shapes || []).filter(s => s.line && s.line.width === 1.5).map(s => s.x0)),
    sampler: () => page.evaluate(() => {
      if (window.__samp) return; window.__samp = 1; window.__log = []; const gd = document.getElementById('chart');
      const P = s => Date.parse(String(s).replace(' ', 'T').slice(0, 23) + 'Z');
      (function f() { const r = gd._fullLayout.xaxis.range; window.__log.push([performance.now(), P(r[0]), P(r[1])]); requestAnimationFrame(f); })();
    }),
    now: () => page.evaluate(() => performance.now()),
    frames: t => page.evaluate(a => window.__log.filter(r => r[0] >= a), t),
  };
  return T;
}
const moves = async (T, n, f) => { for (let k = 1; k <= n; k++) { await T.tp('touchMove', f(k)); await T.page.waitForTimeout(16); } await T.page.waitForTimeout(120); };

// ---------------------------------------------------------------- finding 1
async function finding1(browser) {
  const T = await open(browser, PHONE); const { tp, rng } = T;
  const reset = async () => { await T.btn('#z-reset', 500); await T.btn('#seg-range button[data-key="1m"]', 900); await T.toChart(); };
  await reset();
  const g = await T.geo(), cx = g.x0 + g.len / 2, y = g.top + g.t + g.h * 0.3;
  const pinch = async () => {   // 60 -> 180 px around cx: 3x zoom in
    await tp('touchStart', [{ x: cx - 30, y, id: 0 }]); await tp('touchStart', [{ x: cx - 30, y, id: 0 }, { x: cx + 30, y, id: 1 }]);
    await moves(T, 10, k => [{ x: cx - 30 - 6 * k, y, id: 0 }, { x: cx + 30 + 6 * k, y, id: 1 }]);
  };
  // (a) pan +150 px, add a finger, spread 3 px, release: the pan must survive
  {
    await tp('touchStart', [{ x: cx - 100, y, id: 0 }]);
    await moves(T, 10, k => [{ x: cx - 100 + 15 * k, y, id: 0 }]);
    const rp = await rng();
    await tp('touchStart', [{ x: cx + 50, y, id: 0 }, { x: cx + 150, y, id: 1 }]); await T.page.waitForTimeout(50);
    await moves(T, 3, k => [{ x: cx + 50, y, id: 0 }, { x: cx + 150 + k, y, id: 1 }]);
    const rs = await rng();
    await tp('touchEnd', []); await T.page.waitForTimeout(1200);
    const rf = await rng(), span = rp[1] - rp[0];
    const shift = ((rf[0] + rf[1]) / 2 - (rp[0] + rp[1]) / 2) / span;
    check('F1a pan then add a finger keeps the pan', Math.abs(shift) < 0.05 && rf[0] === rs[0] && rf[1] === rs[1],
      { panned: fmt(rp), afterSpread: fmt(rs), committed: fmt(rf), centreShiftFractionOfSpan: r3(shift), limit: 0.05 });
  }
  // (b) pinch 3x, lift one finger, place a new one, spread 3 px: the zoom must survive
  {
    await reset(); await pinch(); const rz = await rng();
    await tp('touchEnd', [{ x: cx + 90, y, id: 1 }]); await T.page.waitForTimeout(80);
    await tp('touchStart', [{ x: cx - 90, y, id: 0 }, { x: cx + 90, y, id: 2 }]); await T.page.waitForTimeout(50);
    await moves(T, 3, k => [{ x: cx - 90, y, id: 0 }, { x: cx + 90 + k, y, id: 2 }]);
    await tp('touchEnd', []); await T.page.waitForTimeout(1200);
    const rf = await rng(), ratio = (rf[1] - rf[0]) / (rz[1] - rz[0]);
    check('F1b pinch, lift a finger, new finger keeps the zoom', Math.abs(ratio - 1) < 0.05,
      { pinched: fmt(rz), pinchedDays: r3((rz[1] - rz[0]) / DAY), committed: fmt(rf), committedDays: r3((rf[1] - rf[0]) / DAY), spanRatio: r3(ratio) });
  }
  // (c) pinch 3x, lift one finger, move the other +150 px: it must pan 150 px
  {
    await reset(); await pinch(); const rz = await rng();
    await tp('touchEnd', [{ x: cx + 90, y, id: 1 }]); await T.page.waitForTimeout(80);
    await moves(T, 10, k => [{ x: cx - 90 + 15 * k, y, id: 0 }]);
    await tp('touchEnd', []); await T.page.waitForTimeout(1200);
    const rf = await rng(), panPx = (rz[0] - rf[0]) / (rz[1] - rz[0]) * g.len;
    check('F1c pinch then pan with the remaining finger', Math.abs(panPx - 150) < 8 && Math.abs((rf[1] - rf[0]) / (rz[1] - rz[0]) - 1) < 0.01,
      { pinched: fmt(rz), committed: fmt(rf), panPx: r3(panPx), expectedPx: 150 });
  }
  // (d) three fingers, lift one, move the two left 3 px apart: no jump in zoom
  {
    await reset(); await pinch(); const rz = await rng();
    await tp('touchStart', [{ x: cx - 90, y, id: 0 }, { x: cx + 90, y, id: 1 }, { x: cx + 120, y: y + 60, id: 2 }]); await T.page.waitForTimeout(50);
    await tp('touchEnd', [{ x: cx - 90, y, id: 0 }]); await T.page.waitForTimeout(80);
    await moves(T, 3, k => [{ x: cx + 90, y, id: 1 }, { x: cx + 120 + k, y: y + 60, id: 2 }]);
    await tp('touchEnd', []); await T.page.waitForTimeout(1200);
    const rf = await rng(), ratio = (rf[1] - rf[0]) / (rz[1] - rz[0]);
    check('F1d three fingers to two keeps the zoom', Math.abs(ratio - 1) < 0.05, { pinched: fmt(rz), committed: fmt(rf), spanRatio: r3(ratio) });
  }
  check('F1 no page errors', T.errs.length === 0, { errors: T.errs });
  await T.ctx.close();
}

// ---------------------------------------------------------------- finding 4
async function finding4(browser) {
  const T = await open(browser, PHONE, { '2021-05.txt': 2500 }); const { tp, rng, page } = T;
  await T.sampler();
  await T.btn('#seg-range button[data-key="crash"]', 300); await T.toChart();
  const held = async (label, trigger, wait) => {
    const v0 = await rng(), g = await T.geo(), x = g.x0 + g.len / 2, y = g.top + g.t + g.h * 0.3;
    await tp('touchStart', [{ x: x - 100, y, id: 0 }]);
    await moves(T, 10, k => [{ x: x - 100 + 15 * k, y, id: 0 }]);
    const v1 = await rng(), t = await T.now();
    await trigger(); await page.waitForTimeout(wait);
    const v2 = await rng(), res = await page.textContent('#res'), fr = await T.frames(t);
    await tp('touchEnd', []); await page.waitForTimeout(1500);
    const v3 = await rng();
    const snap = fr.filter(r => r[1] === v0[0] && r[2] === v0[1]).length;
    check('F4 ' + label + ': held view is not snapped back', snap === 0 && v2[0] === v1[0] && v2[1] === v1[1] && v3[0] === v1[0],
      { beforeGesture: fmt(v0), panHeld: fmt(v1), afterRenderStillHeld: fmt(v2), resWhileHeld: res, framesShowingPreGestureView: snap, framesSampled: fr.length,
        afterRelease: fmt(v3), resAfterRelease: await page.textContent('#res') });
  };
  await held('1-minute month lands mid-pan', async () => {}, 2800);
  await held('theme change mid-pan', () => page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark')), 800);
  await held('resize mid-pan', () => page.setViewportSize({ width: 360, height: 800 }), 900);
  check('F4 no page errors', T.errs.length === 0, { errors: T.errs });
  await T.ctx.close();
}

// ---------------------------------------------------------------- finding 8
async function finding8(browser) {
  const T = await open(browser, PHONE); const { page, tp, mouse, rng } = T;
  const coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
  await T.btn('#z-in', 500); await T.btn('#z-in', 800); await T.toChart();
  const g = await T.geo(), cx = g.x0 + g.len / 2, cy = g.top + g.t + g.h * 0.2;
  const pxMoved = (a, b) => r3((b[0] - a[0]) / (a[1] - a[0]) * g.len);
  const settle = () => page.waitForTimeout(700);
  // pen tap and pen drag
  await mouse('mousePressed', cx, cy, 'pen'); await mouse('mouseReleased', cx, cy, 'pen'); await settle();
  const cPen = await T.cursor();
  check('F8 pen tap pins the cursor', coarse && cPen.length === 1, { coarse, cursor: cPen });
  let a = await rng();
  await mouse('mousePressed', cx, cy, 'pen'); for (let i = 1; i <= 10; i++) { await mouse('mouseMoved', cx - 15 * i, cy, 'pen'); await page.waitForTimeout(16); }
  await mouse('mouseReleased', cx - 150, cy, 'pen'); await settle();
  let b = await rng();
  check('F8 pen drag pans', Math.abs(pxMoved(a, b) - 150) < 8, { before: fmt(a), after: fmt(b), movedPx: pxMoved(a, b), expectedPx: 150 });
  // mouse click and mouse drag
  const c0 = await T.cursor();
  await mouse('mousePressed', cx + 40, cy, 'mouse'); await mouse('mouseReleased', cx + 40, cy, 'mouse'); await settle();
  const cMouse = await T.cursor();
  check('F8 mouse click pins the cursor', cMouse.length === 1 && cMouse[0] !== c0[0], { before: c0, after: cMouse });
  a = await rng();
  await mouse('mousePressed', cx, cy, 'mouse'); for (let i = 1; i <= 10; i++) { await mouse('mouseMoved', cx + 15 * i, cy, 'mouse', 1); await page.waitForTimeout(16); }
  await mouse('mouseReleased', cx + 150, cy, 'mouse'); await settle();
  b = await rng();
  check('F8 mouse drag pans', Math.abs(pxMoved(a, b) + 150) < 8, { before: fmt(a), after: fmt(b), movedPx: pxMoved(a, b), expectedPx: -150 });
  // a mouse drag released outside the chart must not leave the gesture stuck: a finger tap afterwards still pins
  await mouse('mousePressed', cx, cy + 30, 'mouse'); for (let i = 1; i <= 8; i++) { await mouse('mouseMoved', cx - 20 * i, cy + 30 - 14 * i, 'mouse', 1); await page.waitForTimeout(16); }
  await mouse('mouseReleased', cx - 160, cy + 30 - 112, 'mouse'); await settle();
  await tp('touchStart', [{ x: cx - 60, y: cy, id: 1 }]); await page.waitForTimeout(40); await tp('touchEnd', []); await settle();
  const cTap = await T.cursor();
  // finger taps on the y labels, right margin and x labels must not move it
  await tp('touchStart', [{ x: g.left + 15, y: cy, id: 2 }]); await page.waitForTimeout(40); await tp('touchEnd', []); await settle();
  const cY = await T.cursor();
  await tp('touchStart', [{ x: g.left + g.l + g.w + Math.max(2, (g.W - g.l - g.w) / 2), y: cy, id: 3 }]); await page.waitForTimeout(40); await tp('touchEnd', []); await settle();
  const cR = await T.cursor();
  await tp('touchStart', [{ x: cx + 30, y: g.top + g.t + g.h + 12, id: 4 }]); await page.waitForTimeout(40); await tp('touchEnd', []); await settle();
  const cB = await T.cursor();
  check('F8 finger tap after a mouse drag that ends outside the chart pins', cTap.length === 1 && cTap[0] !== cMouse[0], { cursor: cTap });
  check('F8 finger taps on the axis labels do not pin', cTap.length === 1 && cY[0] === cTap[0] && cR[0] === cTap[0] && cB[0] === cTap[0],
    { afterPlotTap: cTap, afterYLabelTap: cY, afterRightMarginTap: cR, afterXLabelTap: cB });
  check('F8 no page errors', T.errs.length === 0, { errors: T.errs });
  await T.ctx.close();
}

// ---------------------------------------------------------------- finding 12
async function finding12(browser) {
  const T = await open(browser, SMALL); const { page, tp, rng } = T;
  await T.btn('#seg-range button[data-key="crash"]', 1500);
  for (let i = 0; i < 12; i++) { const r = await rng(); if (r[1] - r[0] < 100 * MIN) break; await T.btn('#z-in', 500); }
  await T.toChart(60);
  const pinch = async (label, d0, d1, frac) => {
    const g = await T.geo(), r0 = await rng(), cx = g.x0 + g.len * frac, y = g.top + g.t + g.h * 0.3;
    const at = k => [{ x: cx - (d0 + (d1 - d0) * k / 12) / 2, y, id: 0 }, { x: cx + (d0 + (d1 - d0) * k / 12) / 2, y, id: 1 }];
    await tp('touchStart', at(0)); for (let k = 1; k <= 12; k++) { await tp('touchMove', at(k)); await page.waitForTimeout(16); }
    await tp('touchEnd', []); await page.waitForTimeout(1500);
    const r1 = await rng(), g1 = await T.geo();
    const tb = r0[0] + (cx - g.x0) / g.len * (r0[1] - r0[0]), ta = r1[0] + (cx - g1.x0) / g1.len * (r1[1] - r1[0]);
    const drift = (ta - tb) / MIN;
    check('F12 ' + label, Math.abs((r1[1] - r1[0]) / MIN - 20) < 0.01 && Math.abs(drift) < 0.3,
      { spanBeforeMin: r3((r0[1] - r0[0]) / MIN), requestedMin: r3((r0[1] - r0[0]) / MIN * d0 / d1), spanAfterMin: r3((r1[1] - r1[0]) / MIN),
        centreDriftMin: r3(drift), driftPx: r3(drift * MIN / (r1[1] - r1[0]) * g1.len), before: fmt(r0), after: fmt(r1) });
  };
  await pinch('pinch past the minimum at the plot centre does not drift', 40, 300, 0.5);
  await T.btn('#z-out'); await T.btn('#z-out');
  await pinch('pinch past the minimum at 75% of the plot does not drift', 30, 200, 0.75);
  // + from 40 to 20 min keeps the centre; + at 20 min does not move the view
  await T.btn('#z-out', 800);
  const a = await rng(); await T.btn('#z-in', 800); const b = await rng(); await T.btn('#z-in', 800); const c = await rng();
  const ca = (a[0] + a[1]) / 2;
  check('F12 + at and near the minimum keeps the centre', Math.abs((ca - b[0]) / (b[1] - b[0]) - 0.5) < 0.01 && c[0] === b[0] && c[1] === b[1],
    { start: fmt(a), afterPlus: fmt(b), anchorFracAfterPlus: r3((ca - b[0]) / (b[1] - b[0])), afterSecondPlus: fmt(c), spanMin: [(a[1] - a[0]) / MIN, (b[1] - b[0]) / MIN, (c[1] - c[0]) / MIN] });
  check('F12 no page errors', T.errs.length === 0, { errors: T.errs });
  await T.ctx.close();
}

// ---------------------------------------------------------------- finding 18
async function finding18(browser) {
  const T = await open(browser, IPHONE); const { page, tp, rng } = T;
  const G = [
    ['vertical spread (dy 60->260)', [0, -30], [0, 30], [0, -130], [0, 130]],
    ['vertical close (dy 260->60)', [0, -130], [0, 130], [0, -30], [0, 30]],
    ['vertical, one finger fixed (dy 60->260)', [0, -30], [0, 30], [0, -30], [0, 230]],
    ['near-vertical start to diagonal', [-2, -50], [2, 50], [-75, -75], [75, 75]],
    ['horizontal spread (control, dx 60->180)', [-30, 0], [30, 0], [-90, 0], [90, 0]],
  ];
  for (const [name, a0, b0, a1, b1] of G) {
    await T.btn('#seg-range button[data-key="1m"]', 900); await T.toChart();
    const g = await T.geo(), cx = g.x0 + g.len / 2, cy = g.top + g.t + g.h * 0.45;
    const r0 = await rng(), s0 = await page.evaluate(() => scrollY);
    const lerp = k => [[a0, a1], [b0, b1]].map(([s, e], i) => ({ x: cx + s[0] + (e[0] - s[0]) * k / 12, y: cy + s[1] + (e[1] - s[1]) * k / 12, id: i }));
    await tp('touchStart', lerp(0)); for (let k = 1; k <= 12; k++) { await tp('touchMove', lerp(k)); await page.waitForTimeout(16); }
    await tp('touchEnd', []); await page.waitForTimeout(1000);
    const r1 = await rng(), s1 = await page.evaluate(() => scrollY);
    const d = (p, q) => Math.max(20, Math.hypot(p[0] - q[0], p[1] - q[1])), want = d(a0, b0) / d(a1, b1), got = (r1[1] - r1[0]) / (r0[1] - r0[0]);
    check('F18 ' + name, Math.abs(got / want - 1) < 0.05 && s1 === s0, { spanRatio: r3(got), expected: r3(want), scrolledPx: s1 - s0 });
  }
  check('F18 no page errors', T.errs.length === 0, { errors: T.errs });
  await T.ctx.close();
}

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();
  for (const f of [finding1, finding4, finding8, finding12, finding18]) {
    try { await f(browser); } catch (e) { check(f.name + ' ran', false, { error: String(e && e.stack || e).slice(0, 400) }); }
  }
  await browser.close();
  const failed = checks.filter(c => !c.pass).length;
  console.log(JSON.stringify({ page: PAGE, seconds: Math.round((Date.now() - t0) / 1000), passed: checks.length - failed, failed, checks }, null, 1));
  process.exit(failed ? 1 : 0);
})();
