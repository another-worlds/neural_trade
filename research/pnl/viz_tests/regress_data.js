// Regression checks for the 'data' cluster of the BTCUSDT replay viewer (earlier findings 2, 3, 7, 11, 13, 21).
// Usage: node data.js            (PAGE=test.html by default, served at http://localhost:8765/)
//        PAGE=test_old.html node data.js     (the build before the fixes: should fail)
// Env: PLOTLY = path of plotly-2.34.0.min.js (the CDN is routed to it), PAGE = page under http://localhost:8765/.
// Ground truth is decoded from site/m/*.txt and site/price_hour.bin: month m's index i is the 1-minute bar whose close
// is stamped at (T0 + m.start_min + i + 1) minutes; hourly close k is stamped at (T0 + (k + 1) * 60) minutes.
// Prints one JSON object per check and a summary; exits 1 if any gating check fails. Advisory checks (new issues found
// in the re-check) are reported but never change the exit code.
const { chromium, devices } = require('playwright'); const fs = require('fs');
const PL = fs.readFileSync(process.env.PLOTLY);
const PAGE = process.env.PAGE || 'test.html', URL = 'http://localhost:8765/' + PAGE;
const SITE = process.env.SITE;   // the built site directory (ground truth)
const meta = JSON.parse(fs.readFileSync(SITE + '/meta.json'));
const MIN = 60000, T0 = meta.price.t0_min;
const f32 = b => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
const hour = f32(fs.readFileSync(SITE + '/price_hour.bin'));
const mcache = {};
const monthData = n => mcache[n] || (mcache[n] = f32(Buffer.from(fs.readFileSync(SITE + '/m/' + n + '.txt', 'utf8').trim(), 'base64')));
function minuteClose(ms) { const t = ms / MIN; for (const m of meta.price.months) { const i = t - T0 - m.start_min - 1; if (i >= 0 && i < m.n) return monthData(m.name)[i]; } return undefined; }
function hourAtOrBefore(ms) { const k = Math.floor((ms / MIN - T0) / 60) - 1; return k >= 0 ? { stamp: (T0 + (k + 1) * 60) * MIN, v: hour[k] } : null; }
const D = s => Date.parse(s.replace(' ', 'T') + (s.length <= 16 ? ':00Z' : 'Z'));
const iso = ms => new Date(ms).toISOString().slice(0, 16).replace('T', ' ');
const r1 = v => (v === undefined || v === null || v !== v) ? null : Math.round(v * 10) / 10;
function parseRO(txt) {
  const m = /(\d{4}-\d\d-\d\d \d\d:\d\d) UTC.*?Price(?: \((\d\d:\d\d)(?: close)?\))? ([\d,.]+|n\/a|\u2026)/s.exec(txt);
  return m ? { t: m[1], ms: D(m[1]), lab: m[2] || null, pending: m[3] === '\u2026', price: m[3] === 'n/a' || m[3] === '\u2026' ? null : +m[3].replace(/,/g, '') } : null;
}
// The readout rule: plain "Price" = the exact 1-minute close at the readout time (n/a when the bar is missing);
// "Price (HH:MM)" = the hourly close stamped at HH:MM, the last hour stamp at or before the readout time;
// "Price \u2026" = a pinned time whose 1-minute month is still loading (valid only for a pin, never for a hover).
function judge(ro) {
  if (!ro) return { ok: false, why: 'no readout' };
  if (ro.pending) return { t: ro.t, label: 'Price', shown: '\u2026', kind: 'pending', ok: !ro.lab };
  const exact = r1(minuteClose(ro.ms)), eq = (a, b) => (a === null && b === null) || (a !== null && b !== null && Math.abs(a - b) < 0.051);
  if (!ro.lab) return { t: ro.t, label: 'Price', shown: ro.price, truth1m: exact, kind: 'exact', ok: eq(ro.price, exact) };
  const h = hourAtOrBefore(ro.ms), hs = h ? iso(h.stamp).slice(11) : null, hv = h ? r1(h.v) : null;
  return { t: ro.t, label: 'Price (' + ro.lab + ')', shown: ro.price, truthHourly: hv, hourStamp: hs, truth1m: exact, kind: 'hourly',
    ok: !!h && ro.lab === hs && h.stamp !== ro.ms && eq(ro.price, hv) };
}

async function openPage(browser, opts, cfg = {}) {
  const ctx = await browser.newContext(opts); const page = await ctx.newPage(); const errs = [], fetched = [];
  page.on('pageerror', e => errs.push(e.message));
  await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
  await page.route('**/m/*.txt', async r => {
    const n = r.request().url().split('/').pop(); fetched.push(n);
    const d = typeof cfg.delay === 'function' ? cfg.delay(n) : cfg.delay;
    if (d) await new Promise(z => setTimeout(z, d));
    if (cfg.fail && cfg.fail(n)) return r.fulfill({ status: 404, body: 'missing' });
    r.continue();
  });
  if (cfg.cpu) { const cdp = await ctx.newCDPSession(page); await cdp.send('Emulation.setCPUThrottlingRate', { rate: cfg.cpu }); }
  await page.goto(URL);
  await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
  await page.waitForTimeout(300);
  return { ctx, page, errs, fetched };
}
const res = p => p.textContent('#res');
const readout = p => p.$eval('#readout', e => e.innerText.replace(/\s+/g, ' '));
const view = p => p.evaluate(() => document.getElementById('chart')._fullLayout.xaxis.range.map(s => Date.parse(s.replace(' ', 'T').slice(0, 23) + 'Z')));
const cursorShape = p => p.evaluate(() => { const s = (document.getElementById('chart').layout.shapes || []).filter(s => s.line && s.line.width === 1.5)[0]; return s ? s.x0.slice(0, 16) : null; });
const plotBox = p => p.evaluate(() => { const g = document.getElementById('chart'), r = g.getBoundingClientRect(), s = g._fullLayout._size; return { l: r.left + s.l, t: r.top + s.t, w: s.w, h: s.h }; });
const xOf = (p, ms) => p.evaluate(ms => { const g = document.getElementById('chart'), xa = g._fullLayout.xaxis, r = g.getBoundingClientRect(); return r.left + xa._offset + xa.c2p(ms); }, ms);
const U = ms => new Date(ms).toISOString().slice(0, 23).replace('T', ' ');
async function setRange(p, a, b, wait = 700) { await p.evaluate(([a, b]) => Plotly.relayout(document.getElementById('chart'), { 'xaxis.range': [a, b] }), [a, b]); await p.waitForTimeout(wait); }
const waitRes = (p, re, timeout = 30000) => p.waitForFunction(r => new RegExp(r).test(document.getElementById('res').textContent), re, { timeout });
const chip = (p, k) => p.click('#seg-range button[data-key="' + k + '"]');
const priceStep = p => p.evaluate(() => { const x = document.getElementById('chart').data[0].x, ms = s => Date.parse(s.replace(' ', 'T').slice(0, 23) + 'Z'); let m = Infinity; for (let i = 1; i < Math.min(x.length, 400); i++) m = Math.min(m, ms(x[i]) - ms(x[i - 1])); return { n: x.length, minStepMin: m / 60000 }; });
const HINT_T = 'Tap the chart to read the values at that time.', HINT_D = 'Point at the chart to read the values at that time.';
const DESK = { viewport: { width: 1360, height: 1000 } }, PHONE = { ...devices['Pixel 7'] };

const results = [];
function report(name, finding, pass, measured, gating = true) {
  const o = { name, finding, pass: !!pass, gating, measured }; results.push(o); console.log(JSON.stringify(o));
}
async function guard(name, finding, fn, gating = true) {
  try { await fn(); } catch (e) { report(name, finding, false, { error: String(e.message || e).slice(0, 300) }, gating); }
}

(async () => {
  const t0 = Date.now();
  const browser = await chromium.launch();

  // 1. Finding 2 (+7): a stale 1-minute load for an earlier view must not replace the window of a covered view.
  await guard('stale_load_dropped_3day_view_stays_1min', 2, async () => {
    const { ctx, page, errs } = await openPage(browser, DESK, { delay: n => (n === '2021-06.txt' ? 2500 : 0) });
    await chip(page, 'crash'); await waitRes(page, 'Price: 1-min');
    for (let k = 0; k < 5; k++) { await page.click('#p-right'); await waitRes(page, 'Price: 1-min'); await page.waitForTimeout(100); }
    const june = page.waitForEvent('requestfinished', { predicate: r => r.url().endsWith('2021-06.txt'), timeout: 20000 });
    await page.click('#p-right'); await page.waitForTimeout(150);
    const during = await res(page);
    await page.click('#p-left'); await page.waitForTimeout(300);
    const back = await res(page);
    await june; await page.waitForTimeout(1000);
    const v = await view(page), st = await priceStep(page), fin = await res(page);
    await page.click('#bh-toggle'); await page.waitForTimeout(700);
    const afterRerender = await res(page);
    report('stale_load_dropped_3day_view_stays_1min', 2,
      fin === 'Price: 1-min · 3 days in view' && afterRerender === fin && st.minStepMin === 1 && iso(v[0]) === '2021-05-25 12:00' && iso(v[1]) === '2021-05-28 12:00' && !errs.length,
      { view: v.map(iso), statusWhileJuneLoads: during, statusAfterBack: back, statusAfterJuneArrived: fin, afterRerender, pricePoints: st.n, minPriceStepMin: st.minStepMin, errs });
    await ctx.close();
  });

  // 2. Findings 3, 21: hover readout at a 30-day view (no month loaded) follows the readout rule at every position.
  await guard('hover_readout_wide_view_exact_or_labelled', '3,21', async () => {
    const { ctx, page, errs, fetched } = await openPage(browser, DESK);
    await setRange(page, '2021-03-20', '2021-04-19', 1200);
    const b = await plotBox(page), rows = [];
    for (let k = 0; k < 10; k++) {
      await page.mouse.move(b.l + b.w * (0.04 + 0.1 * k), b.t + b.h * 0.15); await page.waitForTimeout(120);
      rows.push(judge(parseRO(await readout(page))));
    }
    const bad = rows.filter(r => !r.ok || r.kind === 'pending');   // a hover never shows the pending state
    report('hover_readout_wide_view_exact_or_labelled', '3,21', bad.length === 0 && rows.length === 10 && !errs.length,
      { status: await res(page), samples: rows.length, violations: bad.length, labelledHourly: rows.filter(r => r.kind === 'hourly').length,
        maxAbsGapIfUnlabelled: Math.max(0, ...bad.map(r => Math.abs((r.shown || 0) - (r.truth1m || 0)))), rows, monthsFetched: fetched, errs });
    await ctx.close();
  });

  // 3. Findings 3, 11, 21: a pin at a 5-day view loads the month; the readout then shows the exact 1-minute close.
  //    Month edges, a leap day, the seal start, the missing day 2026-10-28 (n/a) and the last bar.
  await guard('pin_loads_exact_1min_close_month_edges_and_gap', '3,11,21', async () => {
    const cfg = { delay: 700 };
    const { ctx, page, errs, fetched } = await openPage(browser, DESK, cfg);
    const SPAN = 5 * 86400000;
    const aim = async (ms) => {   // place the target on an integer client x in a 5-day view
      let a = ms - SPAN / 2;
      for (let k = 0; k < 3; k++) {
        await setRange(page, U(a), U(a + SPAN), 650);
        const v = await view(page), x = await xOf(page, ms), bx = await plotBox(page), xi = Math.round(x);
        if (Math.abs(x - xi) < 0.02 || k === 2) return { x: Math.min(xi, Math.floor(bx.l + bx.w)), y: bx.t + bx.h * 0.3 };
        a = v[0] + (x - xi) * (v[1] - v[0]) / bx.w;
      }
    };
    const targets = ['2021-01-31 23:59', '2021-02-01 00:01', '2021-12-04 05:28', '2024-02-29 23:59', '2024-03-01 00:01', '2025-11-01 00:01',
      '2026-10-28 00:30', '2026-10-28 13:37', '2026-10-29 00:00', '2026-10-29 00:01', '2026-10-31 00:00'];
    const rows = [];
    for (const t of targets) {
      const p = await aim(D(t));
      await page.mouse.move(p.x, p.y); await page.waitForTimeout(100);
      const hover = judge(parseRO(await readout(page)));
      await page.mouse.click(p.x, p.y); await page.waitForTimeout(60); await page.mouse.move(5, 5);
      const now = judge(parseRO(await readout(page)));
      await page.waitForTimeout(1300);
      const loaded = judge(parseRO(await readout(page)));
      rows.push({ target: t, onTarget: loaded.t === t, hover, pinnedBeforeLoad: now, pinnedAfterLoad: loaded });
    }
    const exactAfter = rows.filter(r => r.pinnedAfterLoad.ok && r.pinnedAfterLoad.kind === 'exact').length;
    const ruleOk = rows.every(r => r.hover.ok && r.hover.kind !== 'pending' && r.pinnedBeforeLoad.ok);
    const gap = rows.filter(r => /^2026-10-28 (00:30|13:3)/.test(r.pinnedAfterLoad.t || ''));
    report('pin_loads_exact_1min_close_month_edges_and_gap', '3,11,21',
      exactAfter === rows.length && ruleOk && gap.length === 2 && gap.every(r => r.pinnedAfterLoad.shown === null) && rows.filter(r => r.onTarget).length >= rows.length - 2 && !errs.length,
      { targets: rows.length, onTarget: rows.filter(r => r.onTarget).length, exactAfterLoad: exactAfter, ruleHeldBeforeLoad: ruleOk,
        labelledBeforeLoad: rows.filter(r => r.pinnedBeforeLoad.kind === 'hourly').length, pendingBeforeLoad: rows.filter(r => r.pinnedBeforeLoad.kind === 'pending').length,
        rows: rows.map(r => ({ target: r.target, hover: [r.hover.label, r.hover.shown], beforeLoad: [r.pinnedBeforeLoad.label, r.pinnedBeforeLoad.shown], afterLoad: [r.pinnedAfterLoad.t, r.pinnedAfterLoad.label, r.pinnedAfterLoad.shown], truth1m: r.pinnedAfterLoad.truth1m, ok: r.hover.ok && r.pinnedBeforeLoad.ok && r.pinnedAfterLoad.ok })),
        monthsFetched: fetched, errs });
    await ctx.close();
  });

  // 4. Finding 11: phone tap at the All view: pending (or a labelled hourly close) while the month loads, then the exact close.
  //    Then finding 13 on the phone: chip, swipe and pinch away from a pin clear it; a small swipe keeps it.
  await guard('phone_tap_readout_and_offscreen_pin', '11,13', async () => {
    const { ctx, page, errs } = await openPage(browser, PHONE, { delay: 1000, cpu: 4 });
    const cdp = await ctx.newCDPSession(page);
    const coarse = await page.evaluate(() => matchMedia('(pointer: coarse)').matches);
    await page.evaluate(() => document.getElementById('card').scrollIntoView({ block: 'start' })); await page.waitForTimeout(300);
    const tap = async fr => { const b = await plotBox(page); await page.touchscreen.tap(b.l + b.w * fr, b.t + b.h * 0.3); };
    const tp = (type, pts) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: pts.map((q, i) => ({ x: q[0], y: q[1], id: i })) });
    const swipe = async (f0, f1) => { const b = await plotBox(page), y = b.t + b.h * 0.3, x0 = b.l + b.w * f0, x1 = b.l + b.w * f1;
      await tp('touchStart', [[x0, y]]); for (let k = 1; k <= 8; k++) { await tp('touchMove', [[x0 + (x1 - x0) * k / 8, y]]); await page.waitForTimeout(30); } await tp('touchEnd', []); };
    const pinch = async (fc, f) => { const b = await plotBox(page), y = b.t + b.h * 0.3, c = b.l + b.w * fc;
      await tp('touchStart', [[c - 30, y], [c + 30, y]]); for (let k = 1; k <= 8; k++) { const d = 30 + 30 * (f - 1) * k / 8; await tp('touchMove', [[c - d, y], [c + d, y]]); await page.waitForTimeout(30); } await tp('touchEnd', []); };
    const taps = [];
    for (const fr of [0.31, 0.77]) {
      await tap(fr); await page.waitForTimeout(200); const now = judge(parseRO(await readout(page)));
      await page.waitForTimeout(2000); const loaded = judge(parseRO(await readout(page)));
      taps.push({ fr, beforeLoad: now, afterLoad: loaded });
    }
    const tapOk = taps.every(r => r.beforeLoad.ok && r.afterLoad.ok && r.afterLoad.kind === 'exact');
    const st = async () => ({ readout: (await readout(page)).slice(0, 64), view: (await view(page)).map(iso).join('..'), cursorShape: await cursorShape(page) });
    await chip(page, '1d'); await waitRes(page, 'Price: 1-min'); await page.waitForTimeout(300);
    const pin = {};
    await tap(0.5); await page.waitForTimeout(400); pin.pinned = await st();
    await chip(page, 'crash'); await waitRes(page, 'Price: 1-min'); await page.waitForTimeout(500); pin.afterCrashChip = await st();
    await chip(page, '1d'); await waitRes(page, 'Price: 1-min'); await page.waitForTimeout(300);
    await tap(0.5); await page.waitForTimeout(400); pin.pinned2 = await st();
    await swipe(0.6, 0.4); await page.waitForTimeout(1200); pin.afterSmallSwipe = await st();
    await swipe(0.85, 0.15); await page.waitForTimeout(1200); pin.afterLongSwipe = await st();
    await waitRes(page, 'Price: 1-min').catch(() => {}); await page.waitForTimeout(300);
    await tap(0.15); await page.waitForTimeout(400); pin.pinned3 = await st();
    await pinch(0.85, 4); await page.waitForTimeout(1500); pin.afterPinch = await st();
    const cleared = s => s.readout === HINT_T && !s.cursorShape;
    const pinOk = !cleared(pin.pinned) && cleared(pin.afterCrashChip) && pin.afterSmallSwipe.readout.slice(0, 20) === pin.pinned2.readout.slice(0, 20) && !!pin.afterSmallSwipe.cursorShape && cleared(pin.afterLongSwipe) && cleared(pin.afterPinch);
    report('phone_tap_readout_labelled_then_exact', 11, coarse && tapOk && !errs.length, { coarse, taps, errs });
    report('phone_offscreen_pin_cleared', 13, coarse && pinOk && !errs.length, { coarse, ...pin, errs });
    await ctx.close();
  });

  // 5. Finding 7: the status line shows "Loading 1-minute bars..." while the current request is in flight and never sticks.
  await guard('status_loading_shown_and_never_sticks', 7, async () => {
    const m = {};
    { const { ctx, page } = await openPage(browser, DESK, { delay: 1500 });
      await chip(page, '1d'); await page.waitForTimeout(200); m.slow_at200 = await res(page);
      await page.waitForTimeout(800); m.slow_at1000 = await res(page);
      await waitRes(page, 'Price: 1-min', 15000).catch(() => {}); m.slow_final = await res(page);
      // a 1-day view in uncached months, then zoom out past 4 days while that load is in flight: no "Loading" left behind
      await setRange(page, '2022-06-10 00:00', '2022-06-11 00:00', 300); m.far_loading = await res(page);
      await chip(page, '1m'); await page.waitForTimeout(300); m.zoomout_mid = await res(page);
      await page.waitForTimeout(2200); m.zoomout_final = await res(page);
      await ctx.close(); }
    { const { ctx, page } = await openPage(browser, DESK, { delay: 300, fail: n => true });
      await chip(page, '1d'); await page.waitForTimeout(100); m.fail_mid = await res(page);
      await page.waitForTimeout(1200); m.fail_final = await res(page);
      await ctx.close(); }
    const L = 'Loading 1-minute bars...';
    report('status_loading_shown_and_never_sticks', 7,
      m.slow_at200 === L && m.slow_at1000 === L && m.slow_final === 'Price: 1-min · 24.0 hours in view' && m.far_loading === L &&
      m.zoomout_mid === 'Price: hourly · 30 days in view' && m.zoomout_final === 'Price: hourly · 30 days in view' &&
      m.fail_mid === L && /^1-minute bars unavailable/.test(m.fail_final), m);
  });

  // 6. Finding 13 on desktop: pan buttons, drag zoom and a range chip clear an off-screen pin; zoom buttons keep it.
  await guard('desktop_offscreen_pin_cleared', 13, async () => {
    const { ctx, page, errs } = await openPage(browser, DESK);
    const st = async () => ({ readout: (await readout(page)).slice(0, 64), view: (await view(page)).map(iso).join('..'), cursorShape: await cursorShape(page) });
    const click = async fr => { const b = await plotBox(page); await page.mouse.click(b.l + b.w * fr, b.t + b.h * 0.3); await page.waitForTimeout(250); await page.mouse.move(5, 5); await page.waitForTimeout(150); };
    await chip(page, '1d'); await waitRes(page, 'Price: 1-min'); await page.waitForTimeout(300);
    const s = {};
    await click(0.7); s.pinned = await st();
    await page.click('#p-right'); await page.waitForTimeout(600); s.afterPan1 = await st();
    await page.click('#p-right'); await page.waitForTimeout(600); s.afterPan2 = await st();
    await page.waitForTimeout(600); await click(0.3); s.pinnedB = await st();
    const b = await plotBox(page); await page.mouse.move(b.l + b.w * 0.5, b.t + b.h * 0.3); await page.mouse.down();
    await page.mouse.move(b.l + b.w * 0.9, b.t + b.h * 0.3, { steps: 10 }); await page.mouse.up(); await page.mouse.move(5, 5);
    await page.waitForTimeout(1000); s.afterDragZoom = await st();
    await click(0.8); s.pinnedC = await st();
    await page.click('#z-in'); await page.waitForTimeout(600); await page.click('#z-out'); await page.waitForTimeout(600); await page.click('#z-out'); await page.waitForTimeout(600); s.afterZoomButtons = await st();
    await chip(page, 'crash'); await page.waitForTimeout(1200); s.afterCrashChip = await st();
    const cleared = q => q.readout === HINT_D && !q.cursorShape, same = (a, c) => a.readout.slice(0, 20) === c.readout.slice(0, 20) && !!c.cursorShape;
    report('desktop_offscreen_pin_cleared', 13,
      same(s.pinned, s.afterPan1) && cleared(s.afterPan2) && cleared(s.afterDragZoom) && same(s.pinnedC, s.afterZoomButtons) && cleared(s.afterCrashChip) && !errs.length, { ...s, errs });
    await ctx.close();
  });

  // ---------------- advisory (new issues seen in the re-check; reported, never gating)
  if (process.env.ADVISORY !== '0') {
    await guard('advisory_pin_kept_when_load_lands_mid_swipe', 'new', async () => {
      const { ctx, page, errs } = await openPage(browser, PHONE, { delay: n => (n === '2024-01.txt' ? 2000 : 0) });
      const cdp = await ctx.newCDPSession(page);
      await page.evaluate(() => document.getElementById('card').scrollIntoView({ block: 'start' })); await page.waitForTimeout(300);
      await setRange(page, '2023-12-29 12:00', '2023-12-30 12:00', 1200); await waitRes(page, 'Price: 1-min');
      const b = await plotBox(page), y = b.t + b.h * 0.3, X = f => b.l + b.w * f;
      const tp = (type, x) => cdp.send('Input.dispatchTouchEvent', { type, touchPoints: x == null ? [] : [{ x, y, id: 0 }] });
      await page.touchscreen.tap(X(0.9), y); await page.waitForTimeout(500); const pinned = parseRO(await readout(page));
      await tp('touchStart', X(0.7)); for (let k = 1; k <= 6; k++) { await tp('touchMove', X(0.7 - 0.1 * k)); await page.waitForTimeout(30); } await tp('touchEnd');
      await page.waitForTimeout(250);
      await tp('touchStart', X(0.8)); for (let k = 1; k <= 5; k++) { await tp('touchMove', X(0.8 - 0.1 * k)); await page.waitForTimeout(30); }
      await page.waitForTimeout(2500);
      for (let k = 1; k <= 5; k++) { await tp('touchMove', X(0.3 + 0.1 * k)); await page.waitForTimeout(30); } await tp('touchEnd');
      await page.waitForTimeout(1200);
      const v = await view(page), ro = await readout(page);
      report('advisory_pin_kept_when_load_lands_mid_swipe', 'new', !!pinned && pinned.ms >= v[0] && pinned.ms <= v[1] && ro.startsWith(pinned.t),
        { pinned: pinned && pinned.t, finalView: v.map(iso), pinnedInFinalView: !!pinned && pinned.ms >= v[0] && pinned.ms <= v[1], finalReadout: ro.slice(0, 48), errs }, false);
      await ctx.close();
    }, false);
    await guard('advisory_edge_pin_survives_strategy_switch', 'new', async () => {
      const { ctx, page, errs } = await openPage(browser, DESK);
      await setRange(page, '2023-12-01 12:00:40', '2023-12-01 12:40:40', 1200); await waitRes(page, 'Price: 1-min');
      const b = await plotBox(page), x = Math.floor(b.l + b.w) - 3, y = b.t + b.h * 0.3;
      await page.mouse.click(x, y); await page.waitForTimeout(300); await page.mouse.move(5, 5); await page.waitForTimeout(200);
      const pinned = parseRO(await readout(page)), v = await view(page);
      await page.click('#seg-t1 button[data-key="T1-NET"]');
      await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 30000 }); await page.waitForTimeout(800);
      const ro = await readout(page);
      report('advisory_edge_pin_survives_strategy_switch', 'new', !!pinned && ro.startsWith(pinned.t),
        { view: v.map(q => new Date(q).toISOString().slice(11, 19)), clickedPxFromRightEdge: Math.round(b.l + b.w - x), pinned: pinned && pinned.t, pinnedPastViewEnd: !!pinned && pinned.ms > v[1], afterSwitch: ro.slice(0, 48), errs }, false);
      await ctx.close();
    }, false);
    await guard('advisory_hover_not_overwritten_by_pin_load', 'new', async () => {
      const { ctx, page, errs } = await openPage(browser, DESK, { delay: 1500 });
      const b = await plotBox(page), y = b.t + b.h * 0.3;
      await page.mouse.move(b.l + b.w * 0.3, y); await page.waitForTimeout(120); await page.mouse.click(b.l + b.w * 0.3, y); await page.waitForTimeout(100);
      const pinned = parseRO(await readout(page));
      await page.mouse.move(b.l + b.w * 0.7, y, { steps: 5 }); await page.waitForTimeout(300);
      const hover = parseRO(await readout(page)); await page.waitForTimeout(2500);
      const after = parseRO(await readout(page)), hl = await page.evaluate(() => document.getElementById('hoverline').style.display);
      report('advisory_hover_not_overwritten_by_pin_load', 'new', !!after && !!hover && after.t === hover.t,
        { pinned: pinned && pinned.t, hoverReadout: hover && hover.t, readoutAfterPinMonthLoaded: after && after.t, hoverLineShown: hl, errs }, false);
      await ctx.close();
    }, false);
  }

  await browser.close();
  const gating = results.filter(r => r.gating), failed = gating.filter(r => !r.pass);
  const summary = { page: PAGE, seconds: Math.round((Date.now() - t0) / 1000), checks: gating.length, passed: gating.length - failed.length,
    failed: failed.map(r => r.name), advisory: results.filter(r => !r.gating).map(r => ({ name: r.name, pass: r.pass })) };
  console.log(JSON.stringify({ summary }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
