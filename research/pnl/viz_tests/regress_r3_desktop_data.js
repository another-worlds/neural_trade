// Regression checks for the BTCUSDT replay viewer, round-3 findings 0, 1, 5, 6 (cluster desktop_data).
//   0  a triple click (0-6 px of movement between presses, 40-140 ms holds and gaps, click counts as Chromium assigns
//      them: it restarts the count after >= 3 px) resets the view and keeps the earlier pin; a single click pins; a
//      deliberate click 600+ ms after a double-click pins under the pointer
//   1  desktop windows 701-720 px with a classic 15 px scrollbar: the readout keeps one height (4 lines) while the
//      mouse sweeps the plot; 721-960 px keep the 3-line height; 701-720 px landscape phones (coarse) keep 70.4 px
//   5  a pinned time whose 1-minute month request stalls shows "Price …" for about 2 s, then the last hourly close
//      labelled "Price (HH:MM)", then the exact close when the file arrives (checked against the data files): desktop
//      click with the mouse off the plot and phone tap, a 2nd pin in the same stalled month before and after the 2 s
//      deadline, a fast request (no hourly flash), and a failed request (hourly close, then exact after a retry)
//   6  a render that lands during a drag zoom (1-minute month arriving, B&H toggle) leaves the hover line hidden for the
//      rest of the drag (the readout shows the pin, else the hint) and the hover is redrawn at the pointer after mouseup
// Click trains use CDP Input.dispatchMouseEvent with explicit clickCount; no X server is needed.
// Not covered: a triple click on a CPU 3-4x slower than the test machine (the 500 ms window now runs from the press's
// input timestamp). CDP events carry late timestamps under CPU throttling, so that case needs real OS input (Xvfb + XTest).
// Requires only playwright and fs (Node >= 18 for fetch).
// Env: PLOTLY (required) path of plotly-2.34.0.min.js, served for **/plotly.min.js
//      PAGE   (default test.html) page under BASE;  BASE (default http://localhost:8765/)
//      SITE   (optional) site directory for ground truth (meta.json, price_hour.bin, m/*.txt); default: fetched from BASE
//      ROBOTO (optional) Roboto woff2; when set, the phone stall check runs in Roboto (headless Chromium here maps
//             system-ui to the wider Inter); the checks here do not depend on font widths otherwise
// Prints one JSON line per check, then a summary line. Exit 1 when any check fails, 2 on a setup error.
'use strict';
const { chromium, devices } = require('playwright');
const fs = require('fs');

const PLOTLY = process.env.PLOTLY;
if (!PLOTLY || !fs.existsSync(PLOTLY)) { console.error('PLOTLY must point to plotly-2.34.0.min.js'); process.exit(2); }
const BASE = (process.env.BASE || 'http://localhost:8765/').replace(/\/?$/, '/');
const PAGE = process.env.PAGE || 'test.html';
const SITE = process.env.SITE || '';
const RB = process.env.ROBOTO && fs.existsSync(process.env.ROBOTO) ? fs.readFileSync(process.env.ROBOTO) : null;
const PL = fs.readFileSync(PLOTLY);
const MIN = 60000, DAY = 864e5;
const PIXEL7 = devices['Pixel 7'];

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
function monthOfMs(ms) { const g = Math.floor(ms / MIN) - META.price.t0_min - 1; return META.price.months.find(q => g >= q.start_min && g < q.start_min + q.n); }
async function truthMin(ms) {          // the 1-minute close stamped at ms: month m index i closes at (T0 + m.start_min + i + 1) min
  const g = Math.floor(ms / MIN) - META.price.t0_min - 1, m = monthOfMs(ms);
  if (!MONTHS[m.name]) MONTHS[m.name] = f32(Buffer.from((await file('m/' + m.name + '.txt')).toString().trim(), 'base64'));
  return MONTHS[m.name][g - m.start_min];
}
function truthHour(ms) {               // the last hourly close at or before ms (hour k closes at T0 + (k + 1) * 60 min)
  const k = Math.floor((ms / MIN - META.price.t0_min) / 60) - 1;
  return { v: HOUR[k], at: new Date((META.price.t0_min + (k + 1) * 60) * MIN).toISOString().slice(11, 16), onHour: (META.price.t0_min + (k + 1) * 60) * MIN === ms };
}
const near = (a, b) => a != null && b != null && Math.abs(a - b) < 0.051;   // shown with 1 decimal
// readout parsing: "<time> UTC Price[ (HH:MM)] <value|…|n/a> ..."
function parseReadout(t) {
  const tm = /(\d{4}-\d\d-\d\d \d\d:\d\d) UTC/.exec(t), p = /Price(?: \(([^)]*)\))? ([\d,]+\.\d|[\d,]+|…|n\/a)/.exec(t);
  return { time: tm ? Date.parse(tm[1].replace(' ', 'T') + ':00Z') : null, label: p ? p[1] || '' : null, text: p ? p[2] : null, value: p && /\d/.test(p[2]) ? parseFloat(p[2].replace(/,/g, '')) : null };
}
const isHint = t => /^(Tap|Point at) the chart/.test(t);
const stateOf = async (ro, pin) => {   // classify a readout for a pinned time: pending | hourly | exact | other
  const r = parseReadout(ro); if (r.time !== pin) return { kind: 'other', text: ro.slice(0, 50) };
  if (r.text === '…' && r.label === '') return { kind: 'pending' };
  const th = truthHour(pin), tv = await truthMin(pin);
  if (r.label === '' && near(r.value, tv)) return { kind: 'exact', v: r.value };
  if ((th.onHour ? r.label === '' : r.label === th.at) && near(r.value, th.v)) return { kind: 'hourly', label: r.label, v: r.value };
  return { kind: 'other', text: ro.slice(0, 50) };
};

// ---------------- page harness
// opts: ctx (context options), browser; m/*.txt can be held (net.gate), aborted (net.fail) or delayed (net.delay ms)
async function open(browser, opts = {}) {
  const ctx = await browser.newContext(opts.ctx || { viewport: { width: 1280, height: 800 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  let font = 'default';
  if (opts.roboto && RB) {
    font = 'roboto';
    await page.route('**/__roboto.woff2', r => r.fulfill({ body: RB, contentType: 'font/woff2' }));
    await page.route(u => u.href === BASE + PAGE, async r => {
      const resp = await r.fetch(); let b = await resp.text();
      b = b.replace('</style>', '@font-face{font-family:"RobotoT";src:url(/__roboto.woff2) format("woff2");font-weight:100 900}\n:root{--font:"RobotoT", sans-serif !important}\n</style>');
      r.fulfill({ response: resp, body: b, headers: { ...resp.headers(), 'content-type': 'text/html' } });
    });
  }
  const net = { gate: opts.gate || null, fail: opts.fail || null, delay: 0, held: [], log: [] };
  net.open = () => { net.gate = null; net.held.splice(0).forEach(f => f()); };
  await page.route(/\/m\/[^/]+\.txt$/, async r => {
    const n = r.request().url().split('/').pop(); net.log.push(n);
    if (net.fail && net.fail.test(n)) return r.abort().catch(() => {});
    if (net.gate && net.gate.test(n)) await new Promise(z => net.held.push(z));
    if (net.delay) await wait(net.delay);
    r.continue().catch(() => {});
  });
  await page.addInitScript(() => {
    window.__ev = [];
    for (const ty of ['mousedown', 'click']) window.addEventListener(ty, e => { if (window.__rec) window.__ev.push([ty, Math.round(performance.now()), e.detail]); }, true);
  });
  await page.goto(BASE + PAGE);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await page.evaluate(() => document.fonts.ready);
  await page.evaluate(() => {
    const gd = document.getElementById('chart'); window.__renders = 0;
    gd.on('plotly_afterplot', () => window.__renders++);
    gd.on('plotly_doubleclick', () => { if (window.__rec) window.__ev.push(['plotlyDbl', Math.round(performance.now()), 0]); });
  });
  await wait(300);
  const cdp = await ctx.newCDPSession(page);
  const st = () => page.evaluate(() => {
    const P = s => Date.parse(String(s).replace(' ', 'T').slice(0, 23) + 'Z');
    const gd = document.getElementById('chart'), fl = gd._fullLayout, ro = document.getElementById('readout'), hl = document.getElementById('hoverline');
    return { range: fl.xaxis.range.map(P), pins: (gd.layout.shapes || []).filter(s => s.line && s.line.width === 1.5).map(s => P(s.x0)),
      readout: ro.innerText.replace(/\s+/g, ' ').trim(), roH: +ro.getBoundingClientRect().height.toFixed(1), chartTop: +gd.getBoundingClientRect().top.toFixed(1),
      hl: getComputedStyle(hl).display === 'block' ? +(gd.getBoundingClientRect().left + parseFloat(hl.style.left)).toFixed(1) : null, renders: window.__renders };
  });
  const box = () => page.evaluate(() => { const g = document.getElementById('chart'), s = g._fullLayout._size, r = g.getBoundingClientRect(), xa = g._fullLayout.xaxis;
    return { l: r.left + xa._offset, t: r.top + s.t, w: xa._length, h: s.h }; });
  const xOf = ms => page.evaluate(ms => { const g = document.getElementById('chart'), xa = g._fullLayout.xaxis, r = g.getBoundingClientRect(); return r.left + xa._offset + xa.c2p(ms); }, ms);
  const msAt = x => page.evaluate(x => { const g = document.getElementById('chart'), xa = g._fullLayout.xaxis, r = g.getBoundingClientRect(); return Math.round(xa.p2c(x - r.left - xa._offset) / 60000) * 60000; }, x);
  const chip = async key => { await page.evaluate(k => document.querySelector('#seg-range button[data-key="' + k + '"]').click(), key); };
  const until = (fn, arg, t = 15000) => page.waitForFunction(fn, arg, { timeout: t });
  const tp = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts });
  const tap = async (x, y) => { await tp('touchStart', [{ x, y, id: 0 }]); await wait(40); await tp('touchEnd', []); };
  const mouse = (type, x, y, extra = {}) => cdp.send('Input.dispatchMouseEvent', { type, x, y, button: type === 'mouseMoved' ? 'none' : 'left', ...extra });
  const click1 = async (x, y) => { x = Math.round(x); y = Math.round(y); await mouse('mouseMoved', x, y); await wait(60); await mouse('mousePressed', x, y, { buttons: 1, clickCount: 1 }); await wait(50); await mouse('mouseReleased', x, y, { buttons: 0, clickCount: 1 }); };
  const close = () => ctx.close();
  return { ctx, page, errs, cdp, st, box, xOf, msAt, chip, until, tap, mouse, click1, net, font, close };
}

// ---------------- F0: click trains (CDP clickCount, explicit holds and gaps)
async function f0(browser) {
  // a 1100 px window keeps the reset render short: the round-3 guard measures 500 ms at handler time, so a busy main thread
  // (slow or loaded machine) can push the 3rd click past it; thirdClickAfterDblMs shows the margin
  const o = await open(browser, { ctx: { viewport: { width: 1100, height: 900 }, deviceScaleFactor: 1 } });
  try {
    const FULLD = async () => { const s = await o.st(); return +((s.range[1] - s.range[0]) / DAY).toFixed(2); };
    const setup = async () => {          // 1-month view, pin A by a single click at 30%
      await o.mouse('mouseMoved', 5, 5); await o.chip('1m'); await wait(1000);
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), x = Math.round(b.l + b.w * 0.3), want = await o.msAt(x);
      await o.click1(x, y); await wait(700);
      const s = await o.st();
      return { b, y, A: s.pins, want, days: +((s.range[1] - s.range[0]) / DAY).toFixed(2) };
    };
    // a train: presses at x0 + cumulative dx, clickCount per press as Chromium assigns it (it restarts after >= 3 px)
    const train = async (x0, y, hold, gaps, dxs, counts) => {
      await o.page.evaluate(() => { window.__ev = []; window.__rec = true; });
      let x = Math.round(x0); y = Math.round(y);
      await o.mouse('mouseMoved', x, y); await wait(120);
      for (let i = 0; i < counts.length; i++) {
        if (i) { await wait(gaps[i - 1]); x += dxs[i - 1]; await o.mouse('mouseMoved', x, y); }
        await o.mouse('mousePressed', x, y, { buttons: 1, clickCount: counts[i] }); await wait(hold);
        await o.mouse('mouseReleased', x, y, { buttons: 0, clickCount: counts[i] });
      }
      await wait(1100);
      const ev = await o.page.evaluate(() => { window.__rec = false; return window.__ev; });
      const downs = ev.filter(e => e[0] === 'mousedown').map(e => e[1]), dbl = ev.filter(e => e[0] === 'plotlyDbl').map(e => e[1]);
      return { x, ev: ev.map(e => e[0] + '@' + (e[1] - (downs[0] || 0)) + (e[0] === 'plotlyDbl' ? '' : ' d=' + e[2])).join(' '), lastPressAfterFirst: downs.length ? downs[downs.length - 1] - downs[0] : null, plotlyDbl: dbl.length,
        lastClickAfterDbl: dbl.length ? ev.filter(e => e[0] === 'click').pop()[1] - dbl[0] : null };
    };
    const cnt = (dxs) => { const c = [1]; for (const d of dxs) c.push(d >= 3 ? 1 : c[c.length - 1] + 1); return c; };
    // [hold, gap1, gap2, dx12, dx23]
    const cases = [[40, 60, 60, 3, 3], [50, 90, 90, 3, 3], [60, 110, 110, 3, 3], [70, 140, 140, 3, 3], [50, 90, 90, 0, 3],
      [50, 90, 90, 0, 0], [50, 90, 90, 2, 2], [50, 90, 90, 6, 6], [70, 140, 140, 0, 6], [50, 90, 250, 4, 4]];
    for (const [hold, g1, g2, d12, d23] of cases) {
      const counts = cnt([d12, d23]);
      await check(`tripleclick_h${hold}_g${g1}-${g2}_dx${d12}-${d23}_resets_keeps_pin`, [0], async () => {
        const s0 = await setup(), x0 = s0.b.l + s0.b.w * 0.6;
        const t = await train(x0, s0.y, hold, [g1, g2], [d12, d23], counts);
        const s1 = await o.st(), days = +((s1.range[1] - s1.range[0]) / DAY).toFixed(2);
        return { pass: s0.A.length === 1 && Math.round(s0.days) === 30 && days > 2000 && JSON.stringify(s1.pins) === JSON.stringify(s0.A) && !o.errs.length,
          measured: { counts, pinA: s0.A.map(iso), viewDaysAfter: days, pinsAfter: s1.pins.map(iso), thirdPressAfterFirstMs: t.lastPressAfterFirst, thirdClickAfterDblMs: t.lastClickAfterDbl, plotlyDoubleClicks: t.plotlyDbl, events: t.ev, errs: o.errs } };
      });
    }
    await check('single_click_pins_under_pointer', [0], async () => {
      const s0 = await setup();
      return { pass: s0.A.length === 1 && s0.A[0] === s0.want && Math.round(s0.days) === 30, measured: { pin: s0.A.map(iso), expected: iso(s0.want), viewDays: s0.days } };
    });
    for (const [after, dx] of [[650, 3], [700, 0]]) {
      await check(`click_${after}ms_after_doubleclick_dx${dx}_pins`, [0], async () => {
        const s0 = await setup(), x0 = s0.b.l + s0.b.w * 0.6;
        const t = await train(x0, s0.y, 50, [90, after], [dx, dx], dx >= 3 ? [1, 1, 1] : [1, 2, 1]);
        const s1 = await o.st(), days = +((s1.range[1] - s1.range[0]) / DAY).toFixed(2), want = await o.msAt(t.x);
        return { pass: days > 2000 && s1.pins.length === 1 && s1.pins[0] === want && s1.pins[0] !== s0.A[0],
          measured: { pinA: s0.A.map(iso), viewDaysAfter: days, pinsAfter: s1.pins.map(iso), expected: iso(want), thirdPressAfterFirstMs: t.lastPressAfterFirst, thirdClickAfterDblMs: t.lastClickAfterDbl, events: t.ev } };
      });
    }
  } finally { await o.close(); }
}

// ---------------- F1: readout height at 701-720 px with a classic scrollbar; 721-960 px; landscape phones 701-720 px
async function f1(sbBrowser, browser) {
  const o = await open(sbBrowser, { ctx: { viewport: { width: 701, height: 900 }, deviceScaleFactor: 1 } });
  try {
    await o.page.evaluate(() => { const s = document.getElementById('strat-select'); s.value = 'T2-P'; s.dispatchEvent(new Event('change')); });
    await o.until(() => document.getElementById('loading').hidden === true, null, 60000); await wait(300);
    for (const [w, step, lines] of [[701, 2, 4], [705, 2, 4], [709, 2, 4], [716, 4, 4], [720, 4, 4], [721, 6, 3], [800, 8, 3], [960, 10, 3]]) {
      await check(`readout_height_constant_w${w}_classic_scrollbar`, [1], async () => {
        await o.page.setViewportSize({ width: w, height: 900 }); await wait(900);
        await o.page.evaluate(() => window.scrollTo(0, window.scrollY + document.getElementById('readout').getBoundingClientRect().top - 60)); await wait(200);
        const head = await o.page.evaluate(() => ({ sb: innerWidth - document.documentElement.clientWidth, minH: getComputedStyle(document.getElementById('readout')).minHeight }));
        const b = await o.box(), y = b.t + b.h * 0.3, hs = {}, tops = new Set(); let jumps = 0, prev = null;
        for (let px = 1; px < b.w; px += step) {
          await o.page.mouse.move(b.l + px, y);
          const m = await o.page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(() => r({
            h: Math.round(document.getElementById('readout').getBoundingClientRect().height), top: Math.round(document.getElementById('chart').getBoundingClientRect().top) })))));
          hs[m.h] = (hs[m.h] || 0) + 1; tops.add(m.top); if (prev != null && prev !== m.h) jumps++; prev = m.h;
        }
        await o.page.mouse.move(2, 2);
        const H = Object.keys(hs).map(Number), want = lines === 4 ? [88, 90] : [69, 71];
        return { pass: head.sb >= 12 && H.length === 1 && tops.size === 1 && H[0] >= want[0] && H[0] <= want[1],
          measured: { scrollbarPx: head.sb, layoutWidth: w - head.sb, minHeight: head.minH, heights: hs, jumps, chartTops: [...tops], expectedLines: lines } };
      });
    }
  } finally { await o.close(); }
  for (const [w, h] of [[701, 360], [712, 360], [720, 400]]) {
    await check(`landscape_phone_${w}x${h}_readout_unchanged`, [1], async () => {
      const p = await open(browser, { ctx: { viewport: { width: w, height: h }, screen: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: PIXEL7.userAgent } });
      try {
        const b0 = await p.page.evaluate(() => { const g = document.getElementById('chart'); g.scrollIntoView({ block: 'center' }); return 0; }); await wait(250);
        const b = await p.box(); await p.tap(b.l + b.w * 0.4, b.t + b.h * 0.3); await wait(500);
        const m = await p.page.evaluate(() => {
          const ro = document.getElementById('readout'), zb = document.querySelector('.zoombar').getBoundingClientRect(), h0 = ro.getBoundingClientRect().height;
          window.scrollTo(0, window.scrollY + zb.bottom - innerHeight);
          const gd = document.getElementById('chart'), plotTop = gd.getBoundingClientRect().top + gd._fullLayout._size.t;
          return { coarse: matchMedia('(pointer: coarse)').matches, roH: +h0.toFixed(1), minH: getComputedStyle(ro).minHeight, coverPx: Math.max(0, Math.round(ro.getBoundingClientRect().bottom - plotTop)), readout: ro.innerText.replace(/\s+/g, ' ').slice(0, 40) };
        });
        return { pass: b0 === 0 && m.coarse && Math.abs(m.roH - 70.4) < 0.6 && m.coverPx === 0, measured: m };
      } finally { await p.close(); }
    });
  }
}

// ---------------- F5: stalled, slow, fast and failed 1-minute month requests for a pinned time
// poll the readout for a pinned time; returns the transitions [t, kind]
async function watch(o, pin, ms, t0, stopKind) {
  const tr = []; let last = null;
  while (Date.now() - t0 < ms) {
    const s = await stateOf((await o.st()).readout, pin);
    if (s.kind !== last) { tr.push([+((Date.now() - t0) / 1000).toFixed(2), s.kind, s.label != null ? s.label : s.text || '']); last = s.kind; }
    if (stopKind && s.kind === stopKind) break;
    await wait(80);
  }
  return tr;
}
const kinds = tr => tr.map(z => z[1]).join('>');
const tAt = (tr, kind) => { const z = tr.find(q => q[1] === kind); return z ? z[0] : null; };
async function pinTime(o) { const s = await o.st(); return s.pins[0]; }
// an integer x for a target time whose pinned minute is not on the hour (on the hour the hourly and exact closes coincide)
async function pinX(o, t) { let x = Math.round(await o.xOf(t)); if ((await o.msAt(x)) % 3600000 === 0) x += 1; return x; }

async function f5desk(browser) {
  // (a) stall, then the file arrives: desktop click, mouse off the plot
  await check('stalled_month_desktop_pending_then_hourly_then_exact', [5], async () => {
    const o = await open(browser, { gate: /./ });
    try {
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), x = await pinX(o, Date.UTC(2024, 6, 20, 20, 55));
      await o.page.mouse.move(x, y); await wait(100); await o.page.mouse.click(x, y); const t0 = Date.now();
      await o.page.mouse.move(b.l + b.w / 2, 5); const pin = await pinTime(o);
      const tr = await watch(o, pin, 3300, t0);
      o.net.open(); const tr2 = await watch(o, pin, 7000, t0, 'exact');
      const all = tr.concat(tr2.filter((z, i) => i || z[1] !== tr[tr.length - 1][1])), th = truthHour(pin), tv = await truthMin(pin), tH = tAt(all, 'hourly');
      return { pass: kinds(all) === 'pending>hourly>exact' && tH >= 1.7 && tH <= 3.2 && !o.errs.length,
        measured: { pin: iso(pin), transitions: all, truthHourly: th.at + ' ' + th.v.toFixed(1), truthExact: +tv.toFixed(1), requests: o.net.log, errs: o.errs } };
    } finally { await o.close(); }
  });
  // (b) a 2nd pin in the same stalled month before the deadline, (c) a 3rd one after it; then the file arrives
  await check('stalled_month_desktop_repins_same_month', [5], async () => {
    const o = await open(browser, { gate: /./ });
    try {
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), xs = [Date.UTC(2024, 6, 8, 10, 13), Date.UTC(2024, 6, 15, 7, 41), Date.UTC(2024, 6, 24, 3, 29)];
      const px = []; for (const t of xs) px.push(await pinX(o, t));
      await o.page.mouse.move(px[0], y); await o.page.mouse.click(px[0], y); const t0 = Date.now(); await o.page.mouse.move(b.l + b.w / 2, 5);
      const p1 = await pinTime(o); await wait(1000);
      await o.page.mouse.move(px[1], y); await o.page.mouse.click(px[1], y); const t1 = Date.now(); await o.page.mouse.move(b.l + b.w / 2, 5);
      const p2 = await pinTime(o);
      const tr2 = await watch(o, p2, 3000 + (t1 - t0) - 1000, t1);   // until ~3 s after the 1st pin
      await o.page.mouse.move(px[2], y); await o.page.mouse.click(px[2], y); const t2 = Date.now(); await o.page.mouse.move(b.l + b.w / 2, 5);
      const p3 = await pinTime(o);
      const tr3 = await watch(o, p3, 600, t2);
      o.net.open(); const tr3b = await watch(o, p3, 6000, t2, 'exact');
      const sameMonth = monthOfMs(p1).name === monthOfMs(p2).name && monthOfMs(p2).name === monthOfMs(p3).name;
      const hourly2 = tAt(tr2, 'hourly') != null ? tAt(tr2, 'hourly') + (t1 - t0) / 1000 : null;
      return { pass: sameMonth && kinds(tr2) === 'pending>hourly' && hourly2 <= 3.0 && kinds(tr3) === 'hourly' && tr3[0][0] <= 0.4 && tr3b[tr3b.length - 1][1] === 'exact' && !o.errs.length,
        measured: { pins: [p1, p2, p3].map(iso), month: monthOfMs(p1).name, pin2: tr2, pin2HourlyAfterFirstPinS: hourly2, pin3BeforeArrival: tr3, pin3AfterRelease: tr3b, errs: o.errs } };
    } finally { await o.close(); }
  });
  // (d) a fast request (300 ms) shows the exact close and never flashes the hourly close at the 2 s deadline
  await check('fast_month_desktop_no_hourly_flash', [5], async () => {
    const o = await open(browser, {});
    try {
      o.net.delay = 300;
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), x = await pinX(o, Date.UTC(2023, 2, 14, 13, 17));
      await o.page.mouse.move(x, y); await o.page.mouse.click(x, y); const t0 = Date.now(); await o.page.mouse.move(b.l + b.w / 2, 5);
      const pin = await pinTime(o); const tr = await watch(o, pin, 3000, t0);
      return { pass: kinds(tr) === 'pending>exact' || kinds(tr) === 'exact', measured: { pin: iso(pin), transitions: tr } };
    } finally { await o.close(); }
  });
  // (e) a failed request: the hourly close at once, no return to "…" at the deadline, exact after a retry
  await check('failed_month_desktop_hourly_then_exact_on_retry', [5], async () => {
    const o = await open(browser, { fail: /./ });
    try {
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), x = await pinX(o, Date.UTC(2022, 10, 9, 18, 47));
      await o.page.mouse.move(x, y); await o.page.mouse.click(x, y); const t0 = Date.now(); await o.page.mouse.move(b.l + b.w / 2, 5);
      const pin = await pinTime(o); const tr = await watch(o, pin, 2800, t0);
      o.net.fail = null; await o.chip('all'); const t1 = Date.now(); const tr2 = await watch(o, pin, 5000, t1, 'exact');
      return { pass: kinds(tr).replace(/^pending>/, '') === 'hourly' && tAt(tr, 'hourly') <= 0.6 && tr2[tr2.length - 1][1] === 'exact' && !o.errs.length,
        measured: { pin: iso(pin), whileFailing: tr, afterRetry: tr2, errs: o.errs } };
    } finally { await o.close(); }
  });
}
async function f5phone(browser) {
  const ctx = { ...PIXEL7 };
  await check('stalled_month_phone_tap_pending_then_hourly_then_exact', [5], async () => {
    const o = await open(browser, { ctx, gate: /./, roboto: true });
    try {
      await o.page.evaluate(() => document.getElementById('chart').scrollIntoView({ block: 'center' })); await wait(300);
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), h0 = (await o.st()).roH, x = await pinX(o, Date.UTC(2024, 6, 20, 20, 55));
      await o.tap(x, y); const t0 = Date.now(); const pin = await pinTime(o);
      const tr = await watch(o, pin, 3300, t0); const sH = await o.st();
      o.net.open(); const tr2 = await watch(o, pin, 7000, t0, 'exact'); const sE = await o.st();
      const all = tr.concat(tr2.filter((z, i) => i || z[1] !== tr[tr.length - 1][1])), tH = tAt(all, 'hourly');
      // the box height is judged only in Roboto (Inter can wrap "Price (HH:MM) 67,044.3" where real phone fonts do not)
      const hOk = o.font !== 'roboto' || (sH.roH === h0 && sE.roH === h0);
      return { pass: kinds(all) === 'pending>hourly>exact' && tH >= 1.7 && tH <= 3.2 && hOk && !o.errs.length,
        measured: { font: o.font, pin: iso(pin), transitions: all, readoutH: [h0, sH.roH, sE.roH], truthHourly: truthHour(pin).at + ' ' + truthHour(pin).v.toFixed(1), truthExact: +(await truthMin(pin)).toFixed(1), errs: o.errs } };
    } finally { await o.close(); }
  });
  await check('failed_month_phone_tap_hourly_then_exact_on_retry', [5], async () => {
    const o = await open(browser, { ctx, fail: /./ });
    try {
      await o.page.evaluate(() => document.getElementById('chart').scrollIntoView({ block: 'center' })); await wait(300);
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3), x = await pinX(o, Date.UTC(2022, 10, 9, 18, 47));
      await o.tap(x, y); const t0 = Date.now(); const pin = await pinTime(o);
      const tr = await watch(o, pin, 2800, t0);
      o.net.fail = null; await o.chip('all'); const tr2 = await watch(o, pin, 5000, Date.now(), 'exact');
      return { pass: kinds(tr).replace(/^pending>/, '') === 'hourly' && tAt(tr, 'hourly') <= 0.6 && tr2[tr2.length - 1][1] === 'exact' && !o.errs.length,
        measured: { pin: iso(pin), whileFailing: tr, afterRetry: tr2, errs: o.errs } };
    } finally { await o.close(); }
  });
}

// ---------------- F6: a render that lands during a drag zoom
async function dragRun(o, trigger, pinFirst) {
  const b = await o.box(), y = Math.round(b.t + b.h * 0.3), X = f => Math.round(b.l + b.w * f);
  let pin = null;
  if (pinFirst) { await o.page.mouse.move(X(0.5), y); await o.page.mouse.click(X(0.5), y); await wait(500); pin = await pinTime(o); }
  await o.page.mouse.move(X(0.2), y); await wait(150); await o.page.mouse.down();
  const rows = []; let landedAt = null;
  for (let k = 1; k <= 10; k++) {
    await o.page.mouse.move(X(0.2 + 0.05 * k), y, { steps: 2 }); await wait(120);
    if (k === 3) {
      const r0 = (await o.st()).renders; await trigger();
      await o.until(r => window.__renders > r, r0, 6000).catch(() => {}); await wait(150);
      if ((await o.st()).renders > r0) landedAt = k;
    }
    const s = await o.st();
    rows.push({ step: k, ptr: Math.round(X(0.2 + 0.05 * k)), hl: s.hl == null ? '-' : Math.round(s.hl), renders: s.renders, ro: s.readout.slice(0, 22) });
  }
  const ptrX = X(0.7);
  await o.page.mouse.up(); await wait(2000);
  const a = await o.st(), want = await o.msAt(ptrX), ra = parseReadout(a.readout);
  const after = rows.filter(r => landedAt != null && r.step >= landedAt);
  const roOk = after.every(r => pin != null ? r.ro.startsWith(iso(pin)) : isHint(r.ro));
  return { pass: landedAt != null && after.every(r => r.hl === '-') && roOk && a.hl != null && Math.abs(a.hl - ptrX) <= 2 && ra.time === want && !o.errs.length,
    measured: { pin: iso(pin), renderLandedAtStep: landedAt, steps: rows, afterMouseup: { hl: a.hl == null ? null : Math.round(a.hl), ptr: Math.round(ptrX), readout: a.readout.slice(0, 22), expected: iso(want) }, errs: o.errs } };
}
async function f6(browser) {
  await check('drag_month_render_mid_drag_hides_hover_line', [6], async () => {
    const o = await open(browser, { gate: /./ });                // the 1-week view's 1-minute months are held until mid-drag
    try {
      await o.chip('1w'); await wait(1000);
      const b = await o.box(), y = Math.round(b.t + b.h * 0.3);   // zoom to about 2 days: the view needs 1-minute months (held)
      await o.page.mouse.move(Math.round(b.l + b.w * 0.3), y); await o.page.mouse.down(); await o.page.mouse.move(Math.round(b.l + b.w * 0.6), y, { steps: 6 }); await o.page.mouse.up();
      await wait(500);
      const r = await dragRun(o, async () => { o.net.open(); }, false);
      r.measured.requests = o.net.log; return r;
    } finally { await o.close(); }
  });
  await check('drag_bh_toggle_mid_drag_hides_hover_line_no_pin', [6], async () => {
    const o = await open(browser, {});
    try { return await dragRun(o, () => o.page.evaluate(() => { const c = document.getElementById('bh-toggle'); c.checked = !c.checked; c.dispatchEvent(new Event('change')); }), false); }
    finally { await o.close(); }
  });
  await check('drag_bh_toggle_mid_drag_hides_hover_line_with_pin', [6], async () => {
    const o = await open(browser, {});
    try { return await dragRun(o, () => o.page.evaluate(() => { const c = document.getElementById('bh-toggle'); c.checked = !c.checked; c.dispatchEvent(new Event('change')); }), true); }
    finally { await o.close(); }
  });
}

(async () => {
  const t0 = Date.now();
  try {
    META = JSON.parse((await file('meta.json')).toString()); HOUR = f32(await file('price_hour.bin'));
  } catch (e) { console.error('ground truth unavailable: ' + e.message); process.exit(2); }
  const browser = await chromium.launch();
  const sbBrowser = await chromium.launch({ ignoreDefaultArgs: ['--hide-scrollbars'] });
  try {
    await f0(browser);                                     // timing-sensitive: alone
    await f5desk(browser);
    await f5phone(browser);
    await Promise.all([f1(sbBrowser, browser), f6(browser)]);   // neither depends on timing (F6 lands its render by a trigger)
  } finally { await browser.close(); await sbBrowser.close(); }
  const failed = results.filter(r => !r.pass);
  console.log(JSON.stringify({ summary: true, page: PAGE, checks: results.length, passed: results.length - failed.length, failed: failed.map(r => r.check), seconds: Math.round((Date.now() - t0) / 1000) }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
