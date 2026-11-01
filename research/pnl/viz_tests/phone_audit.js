const { chromium, devices } = require('playwright'); const fs = require('fs');
const PL = fs.readFileSync(process.env.PLOTLY);
async function setup(browser, dev, scheme = 'light') {
  const ctx = await browser.newContext({ ...dev, colorScheme: scheme });
  const page = await ctx.newPage(); const errs = [];
  page.on('pageerror', e => errs.push(e.message)); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.route('https://cdnjs.cloudflare.com/**', r => r.fulfill({ body: PL, contentType: 'application/javascript' }));
  const cdp = await ctx.newCDPSession(page); await cdp.send('Emulation.setCPUThrottlingRate', { rate: 4 });
  return { ctx, page, cdp, errs };
}
const ready = p => p.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
const range = p => p.evaluate(() => document.getElementById('chart')._fullLayout.xaxis.range.map(s => Date.parse(s.replace(' ', 'T').slice(0, 23) + 'Z')));
async function drag(cdp, pts0, pts1, steps = 10) {
  const lerp = (k) => pts0.map((p, i) => ({ x: p.x + (pts1[i].x - p.x) * k / steps, y: p.y + (pts1[i].y - p.y) * k / steps, id: i }));
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: lerp(0) });
  for (let k = 1; k <= steps; k++) { await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: lerp(k) }); }
  await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
}
(async () => {
  const browser = await chromium.launch();
  const which = (process.env.DEVS || 'small,pixel7,iphone13').split(',');
  const DEV = { small: { viewport: { width: 360, height: 740 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: devices['Pixel 5'].userAgent }, pixel7: devices['Pixel 7'], iphone13: devices['iPhone 13'] };
  for (const name of which) {
    const dev = DEV[name];
    const { ctx, page, cdp, errs } = await setup(browser, dev, process.env.SCHEME || 'light');
    const t0 = Date.now(); await page.goto('http://localhost:8765/test.html'); await ready(page); const tLoad = Date.now() - t0;
    await page.waitForTimeout(400);
    const m = await page.evaluate(() => {
      const r = id => document.getElementById(id).getBoundingClientRect();
      const vis = el => el.offsetParent && getComputedStyle(el).display !== 'none';
      const btns = [...document.querySelectorAll('button, select, summary, .toggle')].filter(vis).map(b => ({ h: b.getBoundingClientRect().height, t: (b.id || b.textContent).slice(0, 14) }));
      const minB = btns.reduce((a, b) => (b.h < a.h ? b : a), { h: 1e9 });
      return { coarse: matchMedia('(pointer: coarse)').matches, chartTop: Math.round(r('chart').top + scrollY), chartBottom: Math.round(r('chart').bottom + scrollY), chartH: Math.round(r('chart').height),
        cardTop: Math.round(r('card').top + scrollY), innerH: innerHeight, docH: document.documentElement.scrollHeight,
        overflowX: document.documentElement.scrollWidth - innerWidth, minTarget: minB, tableW: document.getElementById('tbl').scrollWidth,
        scrollBoxW: document.querySelector('.scroll').clientWidth, rangeRowW: document.querySelector('#seg-range').scrollWidth + '/' + document.querySelector('#seg-range').clientWidth,
        nPts: document.getElementById('chart').data.map(t => t.x.length) };
    });
    await page.screenshot({ path: `m2_${name}_top.png` });
    const legendH = await page.$eval('#legend', el => Math.round(el.getBoundingClientRect().height));
    m.legendH = legendH;
    // vertical swipe over the chart scrolls the page and leaves the range alone
    await page.evaluate(() => window.scrollTo(0, document.getElementById('chart').getBoundingClientRect().top + scrollY - 60));
    await page.waitForTimeout(300);
    const cb = await page.$eval('#chart', el => { const r = el.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width }; });
    const r0 = await range(page), sc0 = await page.evaluate(() => scrollY);
    await drag(cdp, [{ x: cb.x, y: cb.y + 120 }], [{ x: cb.x + 4, y: cb.y - 120 }]); await page.waitForTimeout(800);
    const sc1 = await page.evaluate(() => scrollY), r1 = await range(page);
    const vswipe = { scrolled: sc1 - sc0, rangeSame: r0[0] === r1[0] && r0[1] === r1[1] };
    await page.evaluate(y => window.scrollTo(0, y), sc0); await page.waitForTimeout(300);
    // switch to the heaviest strategy with the dropdown
    const t1 = Date.now(); await page.selectOption('#strat-select', 'T1-NET'); await ready(page); const tSwitch = Date.now() - t1;
    // zoom to the crash with the range chip (1-minute bars)
    const t2 = Date.now(); await page.click('#seg-range button[data-key="crash"]');
    await page.waitForFunction(() => /1-min/.test(document.getElementById('res').textContent), null, { timeout: 60000 }); const tZoom = Date.now() - t2;
    await page.waitForTimeout(300);
    const cb2 = await page.$eval('#chart', el => { const r = el.getBoundingClientRect(); return { x: r.x + r.width / 2, y: r.y + r.height * 0.3, w: r.width }; });
    // horizontal swipe pans: the span stays, the range moves back in time, the page does not scroll
    const ra = await range(page), sa = await page.evaluate(() => scrollY);
    const t3 = Date.now();
    await drag(cdp, [{ x: cb2.x - 80, y: cb2.y }], [{ x: cb2.x + 80, y: cb2.y + 6 }]);
    await page.waitForFunction(() => /1-min|hourly/.test(document.getElementById('res').textContent)); await page.waitForTimeout(100);
    const tPan = Date.now() - t3;
    const rb = await range(page), sb = await page.evaluate(() => scrollY);
    const pan = { shiftMin: Math.round((rb[0] - ra[0]) / 60000), spanSame: Math.abs((rb[1] - rb[0]) - (ra[1] - ra[0])) < 1000, scrolled: sb - sa, tPan };
    // pinch out zooms in around the centre
    await drag(cdp, [{ x: cb2.x - 30, y: cb2.y }, { x: cb2.x + 30, y: cb2.y }], [{ x: cb2.x - 90, y: cb2.y }, { x: cb2.x + 90, y: cb2.y }]);
    await page.waitForTimeout(1500);
    const rc = await range(page);
    const pinch = { spanRatio: +(((rc[1] - rc[0]) / (rb[1] - rb[0])).toFixed(3)) };
    // tap reads values and pins a cursor
    await page.touchscreen.tap(cb2.x, cb2.y); await page.waitForTimeout(800);
    const tap = await page.evaluate(() => ({ readout: document.getElementById('readout').innerText.replace(/\s+/g, ' '), cursorShapes: document.getElementById('chart').layout.shapes.filter(s => s.line && s.line.width === 1.5).length }));
    // zoom bar buttons
    const rz0 = await range(page); await page.click('#z-out'); await page.waitForTimeout(1200); const rz1 = await range(page);
    const zbar = { outRatio: +(((rz1[1] - rz1[0]) / (rz0[1] - rz0[0])).toFixed(3)) };
    await page.click('#z-reset'); await page.waitForTimeout(1500); const rr = await range(page);
    zbar.resetDays = Math.round((rr[1] - rr[0]) / 864e5);
    await page.screenshot({ path: `m2_${name}_after.png` });
    await page.screenshot({ path: `m2_${name}_full.png`, fullPage: true });
    console.log(name, JSON.stringify({ tLoad, tSwitch, tZoom, ...m, vswipe, pan, pinch, tap, zbar, errs }));
    await ctx.close();
  }
  await browser.close();
})();
