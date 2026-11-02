// WebGL gap regression for the BTCUSDT replay viewer. A null in a WebGL (scattergl) line reaches the GPU as NaN; many
// phone GPUs draw it at the plot's bottom-left corner, so a line trace with gaps shows a fan of lines from the corner to
// every point (seen on a real Android phone; software GL in headless Chromium hides it). The hold lines (entry -> exit)
// are therefore one SVG path shape. Checks, for every strategy and several views on a phone and a desktop window:
//   A  no scattergl trace holds a null or NaN x or y value
//   B  no trace named Hold, and no SVG line trace drawn as more than 50 separate paths (one element per gap)
//   C  the hold shape: type path, layer between, yref y; every subpath is one entry -> exit line (one M, one L); no
//      drawn vertex at the price panel's bottom-left corner
//   D  close views of busy strategies draw hold lines (the shape has at least one subpath)
//   E  no NaN reaches a WebGL buffer (bufferData / bufferSubData are wrapped and count NaN values in float arrays);
//      with NaN read as 0, the build before this fix drew the fan in headless Chromium too
// Usage: PLOTLY=/path/plotly-2.34.0.min.js [PAGE=test.html] [BASE=http://localhost:8765/] node regress_gl_gaps.js
// Prints one JSON line per check and a summary line; exits 1 on any failure.
const { chromium, devices } = require('playwright');
const fs = require('fs');
if (!process.env.PLOTLY) { console.error('PLOTLY (path to plotly-2.34.0.min.js) is required'); process.exit(2); }
const PL = fs.readFileSync(process.env.PLOTLY);
const PAGE = process.env.PAGE || 'test.html';
const BASE = (process.env.BASE || 'http://localhost:8765/').replace(/\/?$/, '/');
const STRATS = ['T1-LIN', 'T1-NET', 'T1-FADEV', 'T1-NT', 'T2-LIN', 'T2-NET', 'T2-P', 'tsmom_unscaled', 'B&H'];
const results = [];
const check = (id, ok, data) => { const r = { id, pass: !!ok, ...data }; results.push(r); console.log(JSON.stringify(r)); };

async function idle(page) {
  let prev = null;
  for (let k = 0; k < 60; k++) {
    await page.waitForTimeout(200);
    const cur = await page.evaluate(() => { const g = document.getElementById('chart'); return JSON.stringify([g._fullLayout && g._fullLayout.xaxis.range, document.getElementById('res').textContent, g.data && g.data.map(t => t.x.length)]); });
    if (cur === prev && !/Loading/.test(cur)) return;
    prev = cur;
  }
}
const SHIM = () => {   // counts NaN values sent to WebGL buffers
  let hits = 0; window.__nan = () => hits; window.__nanReset = () => { hits = 0; };
  const count = d => { if (d && (d instanceof Float32Array || d instanceof Float64Array)) for (let i = 0; i < d.length; i++) if (d[i] !== d[i]) hits++; };
  for (const P of [window.WebGLRenderingContext, window.WebGL2RenderingContext].filter(Boolean).map(c => c.prototype)) {
    const bd = P.bufferData, bs = P.bufferSubData;
    P.bufferData = function (t, d, ...r) { count(d); return bd.call(this, t, d, ...r); };
    P.bufferSubData = function (t, o, d, ...r) { count(d); return bs.call(this, t, o, d, ...r); };
  }
};
const PROBE = () => {
  const g = document.getElementById('chart'), fl = g._fullLayout, bad = [], many = [];
  g.data.forEach((t, ti) => {
    if (t.type === 'scattergl') {
      let n = 0;
      for (let i = 0; i < (t.x || []).length; i++) {
        const x = t.x[i], y = t.y[i];
        if (x == null || (typeof x === 'number' && !isFinite(x)) || y == null || !isFinite(+y)) n++;
      }
      if (n) bad.push(t.name + ': ' + n + ' gap values in ' + t.x.length);
    }
  });
  g.querySelectorAll('.scatterlayer .trace').forEach(tr => { const n = tr.querySelectorAll('.js-line').length; if (n > 50) many.push(n); });
  const hs = (fl.shapes || []).filter(s => s.type === 'path' && s.yref === 'y');
  const h = hs[0], d = h ? h.path : '';
  const subs = d ? d.split('M').slice(1) : [];
  const malformed = subs.filter(x => (x.match(/L/g) || []).length !== 1).length;
  const ya = fl.yaxis, xa = fl.xaxis, corner = [];
  const hi = h ? (fl.shapes || []).indexOf(h) : -1, hel = hi >= 0 ? [...g.querySelectorAll('.shapelayer path[data-index="' + hi + '"]')] : [];
  if (h && !hel.length) corner.push('hold shape not drawn');
  hel.forEach(el => {
    const dd = el.getAttribute('d') || '';
    (dd.match(/-?[\d.]+,-?[\d.]+/g) || []).forEach(p => { const [x, y] = p.split(',').map(Number); if (Math.abs(x - xa._offset) < 1.5 && Math.abs(y - (ya._offset + ya._length)) < 1.5) corner.push(p); });
  });
  return { gl: g.data.filter(t => t.type === 'scattergl').map(t => t.name), bad, holdTrace: g.data.some(t => t.name === 'Hold'), many,
    shape: h ? { n: hs.length, layer: h.layer, subpaths: subs.length, malformed } : null, corner: corner.length, nanToGL: window.__nan(), markers: g.data.filter(t => /entry/.test(t.name)).reduce((q, t) => q + t.x.length, 0) };
};

(async () => {
  const t0 = Date.now(), browser = await chromium.launch();
  for (const [dn, opts] of [['Pixel 7', devices['Pixel 7']], ['desktop 1280x800', { viewport: { width: 1280, height: 800 } }]]) {
    const ctx = await browser.newContext(opts), page = await ctx.newPage(), errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.addInitScript(SHIM);
    await page.route('**/plotly.min.js', r => r.fulfill({ body: PL, contentType: 'text/javascript' }));
    await page.goto(BASE + PAGE);
    await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 });
    for (const sid of STRATS) {
      await page.evaluate(i => { const s = document.getElementById('strat-select'); s.value = i; s.dispatchEvent(new Event('change')); }, sid);
      await page.waitForFunction(() => document.getElementById('loading').hidden === true, null, { timeout: 120000 }); await idle(page);
      const views = ['all', 'seal', 'crash', '1d'];
      for (const key of views) {
        await page.evaluate(() => window.__nanReset());
        await page.evaluate(k => { const b = document.querySelector('#seg-range button[data-key="' + k + '"]'); if (b) b.click(); }, key); await idle(page);
        const p = await page.evaluate(PROBE);
        const shapeOk = !p.shape || (p.shape.n === 1 && p.shape.layer === 'between' && p.shape.malformed === 0);
        const drawn = !(key === 'crash' && ['T1-NET', 'T1-FADEV', 'T1-NT', 'T2-LIN'].includes(sid)) || (p.shape && p.shape.subpaths > 0);
        check(`${dn} ${sid} ${key}`, !p.bad.length && !p.holdTrace && !p.many.length && shapeOk && p.corner === 0 && drawn && p.nanToGL === 0 && !errs.length,
          { need: 'A no null/NaN in WebGL traces; B no Hold trace, no SVG trace with > 50 paths; C one hold path shape, layer between, one M and one L per subpath, no corner vertex; D holds drawn in the crash view of busy strategies; E no NaN sent to WebGL buffers', ...p, errs: errs.slice() });
      }
    }
    await ctx.close();
  }
  await browser.close();
  const failed = results.filter(r => !r.pass).map(r => r.id);
  console.log(JSON.stringify({ summary: true, page: PAGE, checks: results.length, passed: results.length - failed.length, failed, seconds: Math.round((Date.now() - t0) / 1000) }));
  process.exit(failed.length ? 1 : 0);
})().catch(e => { console.error('FATAL', e); process.exit(1); });
