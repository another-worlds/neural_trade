// Touch + data regression checks for the BTCUSDT replay viewer, round-2 findings 2, 3, 4, 12, 16 (cluster touch_data).
//   2, 4, 12  a pinned cursor survives a held pan or pinch that moves it off screen for a moment while a render lands
//             (1-minute month load, theme change, resize); a pin that is still off screen at release is cleared
//   3         a render that lands during a held pan or pinch keeps the 1-minute price when 1-minute bars cover the screen
//             (no hourly frame while held); a stale window left over from a load in flight never leaves a blank strip
//   16        the readout keeps one height on taps (360/390/412 px portrait, 844x390 landscape, Roboto and default font) and on
//             desktop hover at 720/800 px; a pinned time whose month is loading shows "Price …" and then the exact close
//             (checked against the data files); a failed month shows "Price (HH:MM)" with the last hourly close and a
//             later redraw retries and shows the exact close; hover over an unloaded month says "Price (HH:MM)" (no "close")
// Requires only playwright and fs (Node >= 18 for fetch).
// Env: PLOTLY (required) path of plotly-2.34.0.min.js, served for **/plotly.min.js
//      PAGE   (default test.html) page under BASE;  BASE (default http://localhost:8765/)
//      SITE   (optional) site directory for ground truth (meta.json, price_hour.bin, m/*.txt); default: fetched from BASE
//      ROBOTO (optional) Roboto woff2 used for the phone readout-height checks (real phones use Roboto/SF, headless Chromium
//             here maps system-ui to the wider Inter); default: roboto-latin-var.woff2 next to this script; without the
//             file those checks run in the default font and say so in "font"
// Prints one JSON line per check, then a summary line. Exit 1 when any check fails, 2 on a setup error.
'use strict';
const { chromium, devices } = require('playwright');
const fs = require('fs');

const PLOTLY = process.env.PLOTLY;
if (!PLOTLY || !fs.existsSync(PLOTLY)) { console.error('PLOTLY must point to plotly-2.34.0.min.js'); process.exit(2); }
const BASE = process.env.BASE || 'http://localhost:8765/';
const PAGE = process.env.PAGE || 'test.html';
const SITE = process.env.SITE || '';
const ROBOTO = process.env.ROBOTO || __dirname + '/roboto-latin-var.woff2';
const RB = fs.existsSync(ROBOTO) ? fs.readFileSync(ROBOTO) : null;
const PL = fs.readFileSync(PLOTLY);
const MIN = 60000, DAY = 864e5;
const PIXEL7 = devices['Pixel 7'];
const phone = (w, h) => ({ viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: 3, isMobile: true, hasTouch: true, userAgent: PIXEL7.userAgent });

const results = [];
function record(name, findings, pass, measured) { const r = { check: name, findings, pass: !!pass, measured }; results.push(r); console.log(JSON.stringify(r)); }
async function check(name, findings, fn) {
  try { const { pass, measured } = await fn(); record(name, findings, pass, measured); }
  catch (e) { record(name, findings, false, { error: String(e && e.message || e).slice(0, 300) }); }
}
const wait = ms => new Promise(r => setTimeout(r, ms));
const iso = ms => ms == null ? null : new Date(ms).toISOString().slice(0, 16).replace('T', ' ');

// ---------------- ground truth (data files, not the page)
async function file(rel) {
  if (SITE) return fs.readFileSync(SITE.replace(/\/$/, '') + '/' + rel);
  const r = await fetch(BASE + rel); if (!r.ok) throw new Error(rel + ' ' + r.status); return Buffer.from(await r.arrayBuffer());
}
const f32 = b => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
let META, HOUR; const MONTHS = {};
async function truthMin(ms) {          // the 1-minute close stamped at ms: month m index i closes at (T0 + m.start_min + i + 1) min
  const g = Math.floor(ms / MIN) - META.price.t0_min - 1, m = META.price.months.find(q => g >= q.start_min && g < q.start_min + q.n);
  if (!MONTHS[m.name]) MONTHS[m.name] = f32(Buffer.from((await file('m/' + m.name + '.txt')).toString().trim(), 'base64'));
  return MONTHS[m.name][g - m.start_min];
}
function truthHour(ms) {               // the last hourly close at or before ms (hour k closes at T0 + (k + 1) * 60 min)
  const k = Math.floor((ms / MIN - META.price.t0_min) / 60) - 1;
  return { v: HOUR[k], at: new Date((META.price.t0_min + (k + 1) * 60) * MIN).toISOString().slice(11, 16), onHour: (META.price.t0_min + (k + 1) * 60) * MIN === ms };
}
const near = (a, b) => a != null && b != null && Math.abs(a - b) < 0.051;   // shown with 1 decimal

// ---------------- page harness
// opts: ctx (context options), font ('roboto' | ''), gate (RegExp of month files to hold until open()), fail (RegExp to abort),
//       cpu (CDP CPU throttling rate, as on a phone)
async function open(browser, opts = {}) {
  const ctx = await browser.newContext(opts.ctx || { ...PIXEL7 });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  const font = opts.font === 'roboto' && RB ? 'roboto' : 'default';
  if (font === 'roboto') {
    await page.route('**/__roboto.woff2', r => r.fulfill({ body: RB, contentType: 'font/woff2' }));
    await page.route(u => u.href === BASE + PAGE, async r => {
      const resp = await r.fetch(); let b = await resp.text();
      b = b.replace('</style>', '@font-face{font-family:"RobotoT";src:url(/__roboto.woff2) format("woff2");font-weight:100 900}\n:root{--font:"RobotoT", sans-serif !important}\n</style>');
      r.fulfill({ response: resp, body: b, headers: { ...resp.headers(), 'content-type': 'text/html' } });
    });
  }
  const net = { gate: opts.gate || null, fail: opts.fail || null, held: [], log: [] };
  net.open = () => { net.gate = null; net.held.splice(0).forEach(f => f()); };
  await page.route(/\/m\/[^/]+\.txt$/, async r => {
    const n = r.request().url().split('/').pop(); net.log.push(n);
    if (net.fail && net.fail.test(n)) return r.abort();
    if (net.gate && net.gate.test(n)) await new Promise(z => net.held.push(z));
    r.continue().catch(() => {});
  });
  await page.goto(BASE + PAGE);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => { window.__reacts = 0; const r = Plotly.react; Plotly.react = function () { window.__reacts++; return r.apply(this, arguments); }; });
  await wait(300);
  const cdp = await ctx.newCDPSession(page);
  if (opts.cpu) await cdp.send('Emulation.setCPUThrottlingRate', { rate: opts.cpu });
  const tp = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts });
  const touches = (type, xs, y) => tp(type, xs.map((x, i) => ({ x, y, id: i + 1 })));
  const st = () => page.evaluate(() => {
    const P = s => Date.parse(String(s).replace(' ', 'T').slice(0, 23) + 'Z');
    const gd = document.getElementById('chart'), fl = gd._fullLayout, r = fl.xaxis.range.map(P), x = gd.data[0].x;
    let n = 0, step = Infinity, prev = null;
    for (let i = 0; i < x.length; i++) { const t = P(x[i]); if (t >= r[0] && t <= r[1]) { n++; if (prev != null && t > prev) step = Math.min(step, t - prev); prev = t; } }
    const ro = document.getElementById('readout');
    return { range: r, pins: (gd.layout.shapes || []).filter(s => s.line && s.line.width === 1.5).map(s => P(s.x0)), n, stepMin: step / 60000,
      first: x.length ? P(x[0]) : null, last: x.length ? P(x[x.length - 1]) : null, res: document.getElementById('res').textContent,
      readout: ro.innerText.replace(/\s+/g, ' ').trim(), roH: +ro.getBoundingClientRect().height.toFixed(1),
      chartTop: +gd.getBoundingClientRect().top.toFixed(1), reacts: window.__reacts, coarse: matchMedia('(pointer: coarse)').matches };
  });
  const box = () => page.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(), xa = g._fullLayout.xaxis;
    return { l: r.left + xa._offset, t: r.top + s.t, w: xa._length, h: s.h }; });
  const xOf = ms => page.evaluate(ms => { const g = document.getElementById('chart'), xa = g._fullLayout.xaxis, r = g.getBoundingClientRect(); return r.left + xa._offset + xa.c2p(ms); }, ms);
  const toChart = async (off = 40) => { await page.evaluate(o => window.scrollTo(0, document.getElementById('chart').getBoundingClientRect().top + scrollY - o), off); await wait(250); return box(); };
  const chip = async key => { await page.evaluate(k => document.querySelector('#seg-range button[data-key="' + k + '"]').click(), key); };
  const tap = async (x, y) => { await tp('touchStart', [{ x, y, id: 0 }]); await wait(40); await tp('touchEnd', []); };
  const until = (fn, arg, t = 15000) => page.waitForFunction(fn, arg, { timeout: t });
  const close = () => ctx.close();
  return { ctx, page, errs, cdp, tp, touches, st, box, xOf, toChart, chip, tap, until, net, font, close };
}
// readout parsing: "<time> UTC Price[ (HH:MM[ close])] <value|…|n/a> ..."
function parseReadout(t) {
  const tm = /(\d{4}-\d\d-\d\d \d\d:\d\d) UTC/.exec(t), p = /Price(?: \(([^)]*)\))? ([\d,]+\.\d|[\d,]+|…|n\/a)/.exec(t);
  return { time: tm ? Date.parse(tm[1].replace(' ', 'T') + ':00Z') : null, label: p ? p[1] || '' : null, text: p ? p[2] : null, value: p && /\d/.test(p[2]) ? parseFloat(p[2].replace(/,/g, '')) : null };
}
const isHint = t => /^(Tap|Point at) the chart/.test(t);

// ---------------- F2 / F4 / F12: a pin survives a held gesture with a render landing; released off screen it is cleared
async function pinPanLoad(browser) {          // F2: May 2021 crash, 2021-05 held; pan the pin off, May lands, pan back, lift (CPU 4x)
  const o = await open(browser, { gate: /^2021-05\.txt$/, cpu: 4 });
  try {
    await o.chip('crash'); await wait(300); const b = await o.toChart(); const y = b.t + b.h * 0.3;
    await o.tap(b.l + b.w * 0.3, y); await wait(400);
    const s0 = await o.st(); const pin = s0.pins[0];
    const x0 = b.l + b.w * 0.8;
    await o.tp('touchStart', [{ x: x0, y, id: 1 }]);
    for (let k = 1; k <= 12; k++) { await o.tp('touchMove', [{ x: x0 - 20 * k, y, id: 1 }]); await wait(20); }
    await wait(200); const sOff = await o.st();
    o.net.open(); await o.until(r => window.__reacts > r, sOff.reacts); await wait(500); const sLand = await o.st();
    for (let k = 11; k >= 0; k--) { await o.tp('touchMove', [{ x: x0 - 20 * k, y, id: 1 }]); await wait(20); }
    await wait(150); await o.tp('touchEnd', []); await wait(1500);
    const s2 = await o.st(), rd = parseReadout(s2.readout), tv = await truthMin(pin);
    const pass = pin != null && sOff.range[0] > pin && sLand.pins[0] === pin && s2.pins[0] === pin && rd.time === pin && near(rd.value, tv) && rd.label === '';
    return { pass, measured: { pin: iso(pin), heldOff: iso(sOff.range[0]) + '..' + iso(sOff.range[1]), pinsAfterLandWhileHeld: sLand.pins.map(iso), readoutWhileHeld: sLand.readout.slice(0, 60),
      final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pinsFinal: s2.pins.map(iso), readoutFinal: s2.readout.slice(0, 60), truth: +tv.toFixed(1), errs: o.errs } };
  } finally { await o.close(); }
}
async function pinPinchLoad(browser) {        // F2: same with a held pinch (zoom in about 85% of the width, then back out)
  const o = await open(browser, { gate: /^2021-05\.txt$/ });
  try {
    await o.chip('crash'); await wait(300); const b = await o.toChart(); const y = b.t + b.h * 0.3;
    await o.tap(b.l + b.w * 0.3, y); await wait(400);
    const s0 = await o.st(); const pin = s0.pins[0];
    const cx = b.l + b.w * 0.85, d0 = 60;
    await o.touches('touchStart', [cx - d0, cx + d0], y);
    for (let k = 1; k <= 10; k++) { const d = d0 + 15 * k; await o.touches('touchMove', [cx - d, cx + d], y); await wait(20); }
    await wait(200); const sOff = await o.st();
    o.net.open(); await o.until(r => window.__reacts > r, sOff.reacts); await wait(500); const sLand = await o.st();
    for (let k = 9; k >= 0; k--) { const d = d0 + 15 * k; await o.touches('touchMove', [cx - d, cx + d], y); await wait(20); }
    await wait(150); await o.tp('touchEnd', [{ x: cx + d0, y, id: 2 }]); await wait(30); await o.tp('touchEnd', []); await wait(1500);
    const s2 = await o.st(), rd = parseReadout(s2.readout);
    const pass = pin != null && sOff.range[0] > pin && sLand.pins[0] === pin && s2.pins[0] === pin && rd.time === pin;
    return { pass, measured: { pin: iso(pin), heldOff: iso(sOff.range[0]) + '..' + iso(sOff.range[1]), pinsAfterLandWhileHeld: sLand.pins.map(iso),
      final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pinsFinal: s2.pins.map(iso), readoutFinal: s2.readout.slice(0, 60), errs: o.errs } };
  } finally { await o.close(); }
}
async function pinTwoSwipes(browser) {        // F4: 1-day view 2023-06-28/29, pin near the right edge, July held; 2nd swipe holds the pin off
  const o = await open(browser, { gate: /^2023-07\.txt$/ });
  try {
    await o.page.evaluate(() => Plotly.relayout(document.getElementById('chart'), { 'xaxis.range': ['2023-06-28 12:00', '2023-06-29 12:00'] }));
    await o.until(() => /1-min/.test(document.getElementById('res').textContent)); await wait(300);
    const b = await o.toChart(); const y = b.t + b.h * 0.35, X = f => b.l + b.w * f;
    await o.tap(X(0.9), y); await wait(500); const pin = (await o.st()).pins[0];
    await o.tp('touchStart', [{ x: X(0.75), y, id: 1 }]);
    for (let k = 1; k <= 12; k++) { await o.tp('touchMove', [{ x: X(0.75 - 0.05 * k), y, id: 1 }]); await wait(16); }
    await o.tp('touchEnd', []); await wait(300);
    const s1 = await o.st();
    await o.tp('touchStart', [{ x: X(0.8), y, id: 1 }]);
    for (let k = 1; k <= 10; k++) { await o.tp('touchMove', [{ x: X(0.8 - 0.05 * k), y, id: 1 }]); await wait(16); }
    await wait(150); const sOff = await o.st();
    o.net.open();
    for (let j = 0; j < 12; j++) { await o.tp('touchMove', [{ x: X(0.3) + (j % 2 ? 2 : -2), y, id: 1 }]); await wait(100); }   // a finger never holds still
    const sLand = await o.st();
    for (let k = 1; k <= 10; k++) { await o.tp('touchMove', [{ x: X(0.3 + 0.05 * k), y, id: 1 }]); await wait(16); }
    await o.tp('touchEnd', []); await wait(1500);
    const s2 = await o.st(), rd = parseReadout(s2.readout), tv = await truthMin(pin);
    const pass = pin != null && s1.pins[0] === pin && (pin < sOff.range[0] || pin > sOff.range[1]) && sLand.reacts > sOff.reacts && sLand.pins[0] === pin &&
      s2.pins[0] === pin && pin >= s2.range[0] && pin <= s2.range[1] && rd.time === pin && near(rd.value, tv);
    return { pass, measured: { pin: iso(pin), heldOff: iso(sOff.range[0]) + '..' + iso(sOff.range[1]), rendersWhileHeld: sLand.reacts - sOff.reacts, pinsWhileHeld: sLand.pins.map(iso),
      final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pinsFinal: s2.pins.map(iso), readoutFinal: s2.readout.slice(0, 60), truth: +tv.toFixed(1), errs: o.errs } };
  } finally { await o.close(); }
}
async function pinDayChip(browser) {          // F12: pin, '1 day' chip (its months held), pan 0.7 width, months land while held, back, lift
  const o = await open(browser);
  try {
    let b = await o.toChart(60); await o.tap(b.l + b.w * 0.5, b.t + b.h * 0.3); await wait(1500);
    const pin = (await o.st()).pins[0];
    o.net.gate = /./;
    await o.chip('1d'); await wait(150); b = await o.toChart(60);
    const cx = b.l + b.w * 0.5, y = b.t + b.h * 0.3;
    await o.tp('touchStart', [{ x: cx, y, id: 0 }]);
    for (let i = 1; i <= 10; i++) { await o.tp('touchMove', [{ x: cx - b.w * 0.07 * i, y, id: 0 }]); await wait(30); }
    await wait(300); const sOff = await o.st();
    o.net.open(); await o.until(r => window.__reacts > r, sOff.reacts); await wait(600); const sLand = await o.st();
    for (let i = 9; i >= 0; i--) { await o.tp('touchMove', [{ x: cx - b.w * 0.07 * i, y, id: 0 }]); await wait(30); }
    await o.tp('touchEnd', []); await wait(2000);
    const s2 = await o.st();
    const pass = pin != null && pin < sOff.range[0] && sLand.pins[0] === pin && s2.pins[0] === pin && parseReadout(s2.readout).time === pin;
    return { pass, measured: { pin: iso(pin), heldOff: iso(sOff.range[0]) + '..' + iso(sOff.range[1]), pinsWhileHeld: sLand.pins.map(iso), readoutWhileHeld: sLand.readout.slice(0, 50),
      final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pinsFinal: s2.pins.map(iso), errs: o.errs } };
  } finally { await o.close(); }
}
async function pinThemeResize(browser) {      // F2/F12: theme change and a resize land while the pin is held off screen
  const o = await open(browser);
  try {
    await o.chip('crash'); await o.until(() => /1-min/.test(document.getElementById('res').textContent));
    const b = await o.toChart(); const y = b.t + b.h * 0.3;
    await o.tap(b.l + b.w * 0.3, y); await wait(400); const pin = (await o.st()).pins[0];
    const x0 = b.l + b.w * 0.8;
    await o.tp('touchStart', [{ x: x0, y, id: 1 }]);
    for (let k = 1; k <= 12; k++) { await o.tp('touchMove', [{ x: x0 - 20 * k, y, id: 1 }]); await wait(20); }
    await wait(200); const sOff = await o.st();
    await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    await o.until(r => window.__reacts > r, sOff.reacts); await wait(300); const sTheme = await o.st();
    await o.page.setViewportSize({ width: 400, height: 915 }); await o.until(r => window.__reacts > r, sTheme.reacts); await wait(300); const sResize = await o.st();
    for (let k = 11; k >= 0; k--) { await o.tp('touchMove', [{ x: x0 - 20 * k, y, id: 1 }]); await wait(20); }
    await wait(150); await o.tp('touchEnd', []); await wait(1200);
    const s2 = await o.st();
    const pass = pin != null && pin < sOff.range[0] && sTheme.pins[0] === pin && sResize.pins[0] === pin && s2.pins[0] === pin && parseReadout(s2.readout).time === pin;
    return { pass, measured: { pin: iso(pin), pinsAfterTheme: sTheme.pins.map(iso), pinsAfterResize: sResize.pins.map(iso), readoutAfterResize: sResize.readout.slice(0, 50),
      final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pinsFinal: s2.pins.map(iso), errs: o.errs } };
  } finally { await o.close(); }
}
async function pinReleaseOff(browser) {       // finding 13 kept: a pin still off screen at release is cleared (render mid-hold; render racing the lift)
  const out = {};
  for (const mode of ['theme_mid_hold', 'load_races_lift']) {
    const o = await open(browser, mode === 'load_races_lift' ? { gate: /^2021-05\.txt$/ } : {});
    try {
      await o.chip('crash'); if (mode === 'theme_mid_hold') await o.until(() => /1-min/.test(document.getElementById('res').textContent)); else await wait(300);
      const b = await o.toChart(); const y = b.t + b.h * 0.3;
      await o.tap(b.l + b.w * 0.3, y); await wait(400); const pin = (await o.st()).pins[0];
      const x0 = b.l + b.w * 0.8;
      await o.tp('touchStart', [{ x: x0, y, id: 1 }]);
      for (let k = 1; k <= 12; k++) { await o.tp('touchMove', [{ x: x0 - 20 * k, y, id: 1 }]); await wait(20); }
      await wait(200);
      if (mode === 'theme_mid_hold') { await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark')); await wait(600); }
      else { o.net.open(); await wait(15); }
      await o.tp('touchEnd', []); await wait(2000);
      const s2 = await o.st();
      out[mode] = { pin: iso(pin), final: iso(s2.range[0]) + '..' + iso(s2.range[1]), pins: s2.pins.map(iso), readout: s2.readout.slice(0, 50), ok: pin != null && pin < s2.range[0] && s2.pins.length === 0 && isHint(s2.readout) };
    } finally { await o.close(); }
  }
  return { pass: Object.values(out).every(v => v.ok), measured: out };
}

// ---------------- F3: 1-minute price kept while held when it covers the screen; no blank strip
async function samplerOn(o) {
  await o.page.evaluate(() => {
    const gd = document.getElementById('chart'), P = s => Date.parse(String(s).replace(' ', 'T').slice(0, 23) + 'Z');
    window.__f = []; window.__on = true;
    (function f() { if (!window.__on) return;
      const r = gd._fullLayout.xaxis.range.map(P), x = gd.data[0].x; let step = Infinity, prev = null;
      for (let i = 0; i < x.length; i++) { const t = P(x[i]); if (t >= r[0] && t <= r[1]) { if (prev != null && t > prev) step = Math.min(step, t - prev); prev = t; } }
      window.__f.push([step / 60000, P(x[0]), P(x[x.length - 1]), r[0], r[1]]); requestAnimationFrame(f); })();
  });
}
const samplerOff = o => o.page.evaluate(() => { window.__on = false; const f = window.__f; window.__f = []; return f; });
async function resHeldPan(browser) {          // F3: crash view (3 d, 1-min loaded); held pan 20/150 px; theme, resize, or the month landing
  const out = {};
  for (const [trig, px] of [['theme', 20], ['theme', 150], ['resize', 20], ['load', 20]]) {
    const o = await open(browser, trig === 'load' ? { gate: /^2021-05\.txt$/ } : {});
    try {
      await o.chip('crash'); if (trig !== 'load') await o.until(() => /1-min/.test(document.getElementById('res').textContent)); else await wait(300);
      const b = await o.toChart(); const cx = b.l + b.w / 2, y = b.t + b.h * 0.3;
      await o.tp('touchStart', [{ x: cx - 100, y, id: 0 }]);
      for (let k = 1; k <= 10; k++) { await o.tp('touchMove', [{ x: cx - 100 + px * k / 10, y, id: 0 }]); await wait(16); }
      await wait(150); const s1 = await o.st();
      await samplerOn(o);
      if (trig === 'theme') await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
      if (trig === 'resize') await o.page.setViewportSize({ width: 400, height: 915 });
      if (trig === 'load') o.net.open();
      await o.until(r => window.__reacts > r, s1.reacts); await wait(600);
      const s2 = await o.st(), fr = await samplerOff(o);
      await o.tp('touchEnd', []); await wait(1200); const s3 = await o.st();
      const hourlyFramesHeld = trig === 'load' ? fr.filter((z, i) => i > 0 && z[0] >= 60 && fr.slice(0, i).some(q => q[0] < 60)).length : fr.filter(z => z[0] >= 60).length;
      out[trig + '_pan' + px] = { heldView: iso(s2.range[0]) + '..' + iso(s2.range[1]), heldPts: s2.n, heldStepMin: s2.stepMin, heldRes: s2.res, frames: fr.length, hourlyFramesHeld,
        afterRelease: s3.stepMin, ok: s2.stepMin === 1 && s2.n > 900 && /1-min/.test(s2.res) && hourlyFramesHeld === 0 && s3.stepMin === 1 && (trig === 'load' || s1.stepMin === 1) };
    } finally { await o.close(); }
  }
  return { pass: Object.values(out).every(v => v.ok), measured: out };
}
async function resHeldPinch(browser) {        // F3: 1-day view, pinch out 200 -> 120 px held (1.7 d, inside the 1-min window), theme flips
  const o = await open(browser);
  try {
    await o.page.evaluate(() => Plotly.relayout(document.getElementById('chart'), { 'xaxis.range': ['2023-12-01 00:00', '2023-12-02 00:00'] }));
    await o.until(() => /1-min/.test(document.getElementById('res').textContent)); await wait(300);
    const b = await o.toChart(); const cx = b.l + b.w / 2, y = b.t + b.h * 0.3;
    await o.touches('touchStart', [cx - 100, cx + 100], y);
    for (let k = 1; k <= 8; k++) { const d = 100 - 40 * k / 8; await o.touches('touchMove', [cx - d, cx + d], y); await wait(20); }
    await wait(200); const s1 = await o.st();
    await samplerOn(o);
    await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    await o.until(r => window.__reacts > r, s1.reacts); await wait(600);
    const s2 = await o.st(), fr = await samplerOff(o);
    await o.tp('touchEnd', [{ x: cx + 60, y, id: 2 }]); await wait(30); await o.tp('touchEnd', []); await wait(1200);
    const covered = s2.first <= s2.range[0] + MIN && s2.last >= s2.range[1] - MIN;
    const ok = s1.stepMin === 1 && s2.stepMin === 1 && covered && fr.filter(z => z[0] >= 60).length === 0;
    return { pass: ok, measured: { heldView: iso(s2.range[0]) + '..' + iso(s2.range[1]), heldStepMin: s2.stepMin, heldPts: s2.n, drawn: iso(s2.first) + '..' + iso(s2.last), hourlyFrames: fr.filter(z => z[0] >= 60).length, errs: o.errs } };
  } finally { await o.close(); }
}
async function noBlankStale(browser) {        // F3 guard: a window left from the previous view while June loads must not cut the line short
  const o = await open(browser);
  try {
    await o.chip('crash'); await o.until(() => /1-min/.test(document.getElementById('res').textContent));
    for (let k = 0; k < 5; k++) { await o.page.click('#p-right'); await wait(450); }
    await o.until(() => /1-min/.test(document.getElementById('res').textContent));
    o.net.gate = /^2021-06\.txt$/;
    await o.page.click('#p-right'); await wait(600);
    const b = await o.toChart(); const sx = b.l + b.w * 0.9, y = b.t + b.h * 0.3;
    await o.tp('touchStart', [{ x: sx, y, id: 1 }]);
    for (let k = 1; k <= 10; k++) { await o.tp('touchMove', [{ x: sx - 2 * k, y, id: 1 }]); await wait(20); }
    await wait(200); const s1 = await o.st();
    await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    await o.until(r => window.__reacts > r, s1.reacts); await wait(400);
    await samplerOn(o);
    for (let k = 1; k <= 20; k++) { await o.tp('touchMove', [{ x: sx - 20 - 14 * k, y, id: 1 }]); await wait(20); }
    await wait(300);
    const s2 = await o.st(), fr = await samplerOff(o);
    await o.tp('touchEnd', []); o.net.open(); await wait(800);
    const gap = z => Math.max(0, z[1] - z[3], z[4] - z[2]) / 36e5;   // hours of screen with no price drawn
    const tol = z => z[0] >= 60 ? 1 : 1 / 60;                         // one bar of the drawn resolution
    const worst = fr.reduce((m, z) => Math.max(m, gap(z) - tol(z)), 0);
    return { pass: worst <= 0 && fr.length > 5, measured: { heldView: iso(s2.range[0]) + '..' + iso(s2.range[1]), drawn: iso(s2.first) + '..' + iso(s2.last), stepMin: s2.stepMin, frames: fr.length, worstBlankHours: +worst.toFixed(2), errs: o.errs } };
  } finally { await o.close(); }
}

// ---------------- F16: readout states and heights on taps; desktop hover
async function phoneStates(browser, dev, font, sid) {
  const [W, H] = dev.split('x').map(Number);
  const o = await open(browser, { ctx: phone(W, H), font });
  try {
    if (sid !== 'T1-LIN') { await o.page.evaluate(id => { const s = document.getElementById('strat-select'); s.value = id; s.dispatchEvent(new Event('change')); }, sid); await o.until(() => document.getElementById('loading').hidden === true, null, 60000); }
    await o.chip('all'); await wait(700);
    await o.page.evaluate(() => window.scrollTo(0, document.getElementById('card').getBoundingClientRect().top + scrollY)); await wait(300);
    const b = await o.box(), hint = await o.st();
    const pinOf = async () => (await o.st()).pins[0];
    const R = hint.range, times = [0.12, 0.3, 0.48, 0.66, 0.84].map((f, i) => Math.floor((R[0] + (R[1] - R[0]) * f) / 36e5) * 36e5 + (37 + i) * MIN);
    const rows = []; let ok = true;
    // (a) month held: "Price …", then the exact close
    o.net.gate = /./;
    for (const t of times.slice(0, 3)) {
      await o.tap(await o.xOf(t), b.t + 30); await wait(250);
      const a = await o.st(), pin = await pinOf(), ra = parseReadout(a.readout);
      o.net.open(); await o.until(() => !/…/.test(document.getElementById('readout').innerText), null, 8000).catch(() => {}); await wait(150); o.net.gate = /./;
      const e = await o.st(), re = parseReadout(e.readout), tv = await truthMin(pin);
      const r = { pin: iso(pin), loading: ra.label ? 'Price (' + ra.label + ') ' + ra.text : 'Price ' + ra.text, exact: re.text, truth: +tv.toFixed(1), h: [hint.roH, a.roH, e.roH], top: [hint.chartTop, a.chartTop, e.chartTop] };
      r.ok = ra.time === pin && ra.text === '…' && ra.label === '' && re.time === pin && re.label === '' && near(re.value, tv) && a.roH === hint.roH && e.roH === hint.roH && a.chartTop === hint.chartTop && e.chartTop === hint.chartTop;
      ok = ok && r.ok; rows.push(r);
    }
    o.net.open();
    // (b) month fails: "Price (HH:MM)" + last hourly close; then the network is back and a redraw shows the exact close
    o.net.fail = /./;
    let lastPin = null;
    for (const t of times.slice(3)) {
      await o.tap(await o.xOf(t), b.t + 30); await wait(400);
      const a = await o.st(), pin = await pinOf(), ra = parseReadout(a.readout), th = truthHour(pin);
      const r = { pin: iso(pin), failed: (ra.label ? 'Price (' + ra.label + ') ' : 'Price ') + ra.text, truthHourly: th.at + ' ' + th.v.toFixed(1), h: a.roH, top: a.chartTop };
      // the height is judged in Roboto only: in Inter (headless default) "Price (HH:MM) 73,078.9" can wrap at 360 px, real phone fonts do not
      r.ok = ra.time === pin && (th.onHour ? ra.label === '' : ra.label === th.at) && near(ra.value, th.v) && (o.font !== 'roboto' || a.roH === hint.roH && a.chartTop === hint.chartTop);
      ok = ok && r.ok; rows.push(r); lastPin = pin;
    }
    o.net.fail = null;
    const s0 = await o.st();
    await o.page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
    await o.until(r => window.__reacts > r, s0.reacts); await o.until(() => !/Price \(/.test(document.getElementById('readout').innerText), null, 8000).catch(() => {}); await wait(200);
    const s1 = await o.st(), r1 = parseReadout(s1.readout), tv = await truthMin(lastPin);
    const retry = { pin: iso(lastPin), shown: (r1.label ? 'Price (' + r1.label + ') ' : 'Price ') + r1.text, truth: +tv.toFixed(1), ok: r1.time === lastPin && r1.label === '' && near(r1.value, tv) && s1.roH === hint.roH };
    ok = ok && retry.ok;
    return { pass: ok && hint.coarse, measured: { dev, font: o.font, sid, coarse: hint.coarse, hintH: hint.roH, rows, retry, errs: o.errs } };
  } finally { await o.close(); }
}
async function phoneInterLoading(browser) {   // F16 in the default font: the loading state of a pin never adds a line (360 px, the widest strategy)
  const o = await open(browser, { ctx: phone(360, 740) });
  try {
    await o.page.evaluate(() => { const s = document.getElementById('strat-select'); s.value = 'tsmom_unscaled'; s.dispatchEvent(new Event('change')); }); await o.until(() => document.getElementById('loading').hidden === true, null, 60000);
    await o.chip('seal'); await wait(700);
    await o.page.evaluate(() => window.scrollTo(0, document.getElementById('card').getBoundingClientRect().top + scrollY)); await wait(300);
    const b = await o.box(), hint = await o.st(); const hs = new Set([hint.roH]), tops = new Set([hint.chartTop]); let n = 0;
    o.net.gate = /./;
    for (let d = 2; d < 365; d += 30) {
      const x = await o.xOf(Date.UTC(2025, 10, 1) + d * DAY + 37 * MIN); if (x < b.l + 2 || x > b.l + b.w - 2) continue;
      await o.tap(x, b.t + 30); await wait(200); const a = await o.st();
      o.net.open(); await wait(400); o.net.gate = /./; const e = await o.st();
      hs.add(a.roH); hs.add(e.roH); tops.add(a.chartTop); tops.add(e.chartTop); n++;
    }
    o.net.open();
    return { pass: hs.size === 1 && tops.size === 1 && n >= 10, measured: { font: 'default', taps: n, readoutHeights: [...hs], chartTops: [...tops], errs: o.errs } };
  } finally { await o.close(); }
}
async function desktopHover(browser, W, sid) {   // F16 desktop 720/800 px: hover over unloaded months keeps the box height; label without "close"
  const o = await open(browser, { ctx: { viewport: { width: W, height: 900 } } });
  try {
    if (sid !== 'T1-LIN') { await o.page.evaluate(id => { const s = document.getElementById('strat-select'); s.value = id; s.dispatchEvent(new Event('change')); }, sid); await o.until(() => document.getElementById('loading').hidden === true, null, 60000); }
    await o.chip('seal'); await wait(700);
    await o.page.evaluate(() => window.scrollTo(0, document.getElementById('card').getBoundingClientRect().top + scrollY)); await wait(300);
    await o.page.mouse.move(5, 5); await wait(100);
    const idle = await o.st(), b = await o.box(); const tops = new Set([idle.chartTop]), hs = new Set([idle.roH]), labels = new Set();
    const hover = async (from, days, step) => { for (let d = 0; d <= days; d += step) {
      await o.page.mouse.move(await o.xOf(Date.parse(from) + d * DAY), b.t + 40); await wait(60);
      const s = await o.st(); tops.add(s.chartTop); hs.add(s.roH); const p = parseReadout(s.readout); if (p.label) labels.add(p.label); } };
    await hover('2025-12-10T12:00:00Z', 6, 3);
    o.net.gate = /^2026-01\.txt$/;
    const tj = Date.parse('2026-01-15T12:07:00Z');
    await o.page.mouse.click(await o.xOf(tj), b.t + 40); await wait(150);
    const sl = await o.st(), pl = parseReadout(sl.readout); tops.add(sl.chartTop); hs.add(sl.roH);
    o.net.open(); await o.until(() => !/…/.test(document.getElementById('readout').innerText), null, 8000).catch(() => {}); await wait(200);
    const se = await o.st(), pe = parseReadout(se.readout), pin = se.pins[0], tv = pin != null ? await truthMin(pin) : null;
    await hover('2025-12-10T12:00:00Z', 60, 3);
    const lab = [...labels];
    const pass = lab.length > 0 && lab.every(l => /^\d\d:\d\d$/.test(l)) && tops.size === 1 && hs.size === 1 && pl.text === '…' && pe.label === '' && near(pe.value, tv);
    return { pass, measured: { W, sid, coarse: idle.coarse, chartTops: [...tops], readoutHeights: [...hs], hoverLabels: lab.slice(0, 4), pinLoading: 'Price ' + (pl.label ? '(' + pl.label + ') ' : '') + pl.text,
      pinExact: pe.text, truth: tv != null ? +tv.toFixed(1) : null, errs: o.errs } };
  } finally { await o.close(); }
}

// ---------------- run
(async () => {
  try {
    META = JSON.parse((await file('meta.json')).toString()); HOUR = f32(await file('price_hour.bin'));
  } catch (e) { console.error('ground truth unavailable: ' + e.message); process.exit(2); }
  const t0 = Date.now();
  const browser = await chromium.launch();
  const jobs = [
    () => check('f2_pin_kept_held_pan_1min_load', [2, 12], () => pinPanLoad(browser)),
    () => check('f2_pin_kept_held_pinch_1min_load', [2, 12], () => pinPinchLoad(browser)),
    () => check('f4_pin_kept_second_swipe_month_lands', [4], () => pinTwoSwipes(browser)),
    () => check('f12_pin_kept_1day_chip_pan_months_land', [12], () => pinDayChip(browser)),
    () => check('f12_pin_kept_theme_and_resize_while_held', [2, 12], () => pinThemeResize(browser)),
    () => check('f13_pin_released_off_screen_is_cleared', [2, 4, 12], () => pinReleaseOff(browser)),
    () => check('f3_1min_kept_held_pan_theme_resize_load', [3], () => resHeldPan(browser)),
    () => check('f3_1min_kept_held_pinch_theme', [3], () => resHeldPinch(browser)),
    () => check('f3_no_blank_strip_stale_window', [3], () => noBlankStale(browser)),
    ...['360x740', '390x844', '412x915', '844x390'].map(d => () => check('f16_pin_states_heights_' + d + '_roboto', [16], () => phoneStates(browser, d, 'roboto', 'tsmom_unscaled'))),
    () => check('f16_pin_loading_height_360_default_font', [16], () => phoneInterLoading(browser)),
    () => check('f16_desktop_hover_720', [16], () => desktopHover(browser, 720, 'T1-LIN')),
    () => check('f16_desktop_hover_800', [16], () => desktopHover(browser, 800, 'tsmom_unscaled')),
  ];
  const CONC = +(process.env.CONC || 3); let next = 0;
  await Promise.all(Array.from({ length: CONC }, async () => { while (next < jobs.length) { const j = jobs[next++]; await j(); } }));
  await browser.close();
  const failed = results.filter(r => !r.pass);
  console.log(JSON.stringify({ summary: { page: PAGE, checks: results.length, passed: results.length - failed.length, failed: failed.map(r => r.check), seconds: Math.round((Date.now() - t0) / 1000) } }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
