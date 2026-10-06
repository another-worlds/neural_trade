"""Live dashboard of the tactical session: reads runs/tactical (screen results, logreg reference, git, GPU)
and writes runs/tactical/dashboard.html (self-contained data, plotly from CDN, auto-reloads every 20 s).
    python runs/tactical/dashboard.py            # write once
    python runs/tactical/dashboard.py --loop 20  # rewrite every 20 s
"""
import glob, json, math, os, statistics as st, subprocess, sys, time, collections, datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "dashboard.html")
PLANNED = 100  # trials per spec (5 slices x 20 seeds)
SPECS = {  # name -> (label, one-line meaning)
    "hc_baseline": ("Baseline", "today's defaults, no change"),
    "hc_r1_skip_only": ("skip_only", "direction = linear skip over lags only (joint logreg)"),
    "hc_r1_shrink1": ("shrink 1.0", "L2 penalty on the deep direction logit"),
    "hc_r1_drop05": ("dropout 0.5", "dropout on the deep direction path input"),
}


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20, cwd=REPO).stdout
    except Exception:
        return ""


def load(name):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "screens", name, "results*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            a = r.get("direction_auc") or {}
            ok = all(a.get(h) and a[h].get("auc") is not None for h in ("h0", "h1", "h2"))
            rows.append({
                "slice": r["data_end"][:10], "seed": r["seed"],
                "auc": st.mean(a[h]["auc"] for h in ("h0", "h1", "h2")) if ok else None,
                "h": [a[h]["auc"] if a.get(h) else None for h in ("h0", "h1", "h2")],
                "wall": r.get("wall_s"), "passed": r.get("passed"),
                "val_loss": (r.get("health") or {}).get("final_val_loss"),
            })
    return rows


def main():
    data = {n: load(n) for n in SPECS}
    for n in list(data):
        if not data[n] and n != "hc_baseline":
            data[n] = []
    logreg = {}
    p = os.path.join(ROOT, "hc_logreg.json")
    if os.path.exists(p):
        logreg = {k[:10]: v["mean"] for k, v in json.load(open(p)).items()}
    base = {(r["slice"], r["seed"]): r["auc"] for r in data["hc_baseline"] if r["auc"] is not None}
    comp = {}
    for n, rows in data.items():
        if n == "hc_baseline":
            continue
        per = collections.defaultdict(list)
        for r in rows:
            k = (r["slice"], r["seed"])
            if r["auc"] is not None and k in base:
                per[r["slice"]].append(r["auc"] - base[k])
        sl = {s: {"mean": st.mean(v), "n": len(v), "se": (st.stdev(v) / math.sqrt(len(v))) if len(v) > 1 else None}
              for s, v in per.items()}
        out = {"slices": sl}
        if len(sl) > 1:
            ms = [v["mean"] for v in sl.values()]
            m, se = st.mean(ms), st.stdev(ms) / math.sqrt(len(ms))
            t = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57}.get(len(ms), 2.0)
            out.update(mean=m, lo=m - t * se, hi=m + t * se, n_slices=len(ms))
        comp[n] = out
    procs = [l for l in sh('wmic process where "name=\'python.exe\'" get CommandLine').splitlines() if "cli screen" in l]
    running = collections.Counter()
    for l in procs:
        for n in SPECS:
            if f"/{n}.yaml" in l or f" configs/tactical/{n}.yaml" in l:
                running[n] += 1
    gpu = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader").strip()
    log = sh("git log --pretty=format:%h|%ad|%s --date=format:%H:%M -12").splitlines()
    state = {
        "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "specs": {n: {"label": SPECS[n][0], "desc": SPECS[n][1], "rows": data[n], "running": running.get(n, 0)} for n in SPECS},
        "logreg": logreg, "comp": comp, "gpu": gpu, "log": log, "planned": PLANNED,
        "shards_running": len(procs),
    }
    html = TEMPLATE.replace("__STATE__", json.dumps(state))
    tmp = OUT + ".tmp"
    open(tmp, "w", encoding="utf-8").write(html)
    os.replace(tmp, OUT)


TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Tactical session</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202a;--mut:#667;--line:#e1e5ea;--b:#3987e5;--o:#d95926;--g:#199e70;--p:#8a5cc2;--bad:#c0392b;--ok:#199e70}
@media(prefers-color-scheme:dark){:root{--bg:#10151b;--card:#18202a;--ink:#e6ebf0;--mut:#93a0ad;--line:#2a3541}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
.w{max-width:1200px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--mut);font-size:12px}.g{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));margin-top:12px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.full{grid-column:1/-1}
.kpi{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.k{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k b{font-size:22px;display:block}.k span{color:var(--mut);font-size:12px}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--b)}
.row{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center;margin:6px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.pos{color:var(--ok)}.neg{color:var(--bad)}code{background:var(--line);padding:0 4px;border-radius:3px}
</style></head><body><div class="w">
<h1>Тактическая сессия: прорыв в расчёте сети <span class="tag" id="live"></span></h1>
<div class="sub" id="sub"></div>
<div class="kpi" id="kpi"></div>
<div class="g">
<div class="c full"><h2>Прогресс раундов</h2><div id="prog"></div></div>
<div class="c full"><h2>Direction AUC (среднее h0–h2) по срезам: базовая линия, кандидаты, logreg_lags</h2><div id="strip" style="height:380px"></div>
<div class="sub">Каждая точка — один прогон (seed). Ромб — logreg_lags на том же блоке. 0,5 — монетка.</div></div>
<div class="c"><h2>Парная разность с базовой линией (по срезам)</h2><div id="diff" style="height:320px"></div>
<div class="sub">Критерий (задан до раунда): 95% интервал по срезам целиком выше 0. Единица вывода — срез, не прогон.</div></div>
<div class="c"><h2>Распределение AUC (все прогоны)</h2><div id="viol" style="height:320px"></div></div>
<div class="c"><h2>AUC по горизонтам h0 / h1 / h2</h2><div id="hor" style="height:300px"></div></div>
<div class="c"><h2>Время прогона, с (по порядку записи)</h2><div id="wall" style="height:300px"></div>
<div class="sub">Лимит владельца: 2 минуты = 120 с (D-063).</div></div>
<div class="c"><h2>Гипотезы</h2><table id="hyp"></table></div>
<div class="c"><h2>Последние коммиты nt-tactical</h2><table id="git"></table></div>
</div></div>
<script>
const S=__STATE__;const COL={hc_baseline:'#8a97a6',hc_r1_skip_only:'#3987e5',hc_r1_shrink1:'#d95926',hc_r1_drop05:'#199e70'};
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n);
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length,sd=a=>{if(a.length<2)return null;const m=mean(a);return Math.sqrt(a.reduce((x,y)=>x+(y-m)**2,0)/(a.length-1))};
const names=Object.keys(S.specs),ink=getComputedStyle(document.body).color,grid=css('--line');
const lay=(o)=>Object.assign({margin:{l:48,r:12,t:8,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:ink,size:12},xaxis:{gridcolor:grid},yaxis:{gridcolor:grid},legend:{orientation:'h',y:-0.2}},o||{});
const auc=n=>S.specs[n].rows.filter(r=>r.auc!=null).map(r=>r.auc);
document.getElementById('live').textContent=S.shards_running?('идёт: '+S.shards_running+' шард'):'простой';
document.getElementById('live').style.color=S.shards_running?'var(--ok)':'var(--mut)';
document.getElementById('sub').textContent='Обновлено '+S.now+' · страница перезагружается каждые 20 с · GPU: '+(S.gpu||'н/д');
// KPI
const b=auc('hc_baseline'),lr=Object.values(S.logreg);
const done=names.reduce((a,n)=>a+S.specs[n].rows.length,0),plan=names.length*S.planned;
const best=Object.entries(S.comp).filter(([n,c])=>c.mean!=null).sort((a,b)=>b[1].mean-a[1].mean)[0];
const k=(v,l)=>`<div class="k"><b>${v}</b><span>${l}</span></div>`;
document.getElementById('kpi').innerHTML=
 k(b.length?mean(b).toFixed(3):'—','базовая линия, средний AUC ('+b.length+' прогонов)')+
 k(lr.length?mean(lr).toFixed(3):'—','logreg_lags, средний по срезам')+
 k(best?((best[1].mean>=0?'+':'')+best[1].mean.toFixed(3)):'—',best?('лучший кандидат: '+S.specs[best[0]].label+(best[1].lo>0?' (значимо)':' (в пределах шума)')):'лучший кандидат: пока нет данных')+
 k(done+' / '+plan,'прогонов раунда готово');
// progress
document.getElementById('prog').innerHTML=names.map(n=>{const r=S.specs[n].rows.length,p=Math.min(100,100*r/S.planned);
 const st=r>=S.planned?'готово':(S.specs[n].running?'идёт ('+S.specs[n].running+' шард)':(r?'частично':'ожидает'));
 return `<div class="row"><div><b>${S.specs[n].label}</b></div><div class="bar"><i style="width:${p}%;background:${COL[n]}"></i></div><div class="sub">${r}/${S.planned} · ${st}</div></div>`}).join('');
// strip by slice
const slices=[...new Set(S.specs.hc_baseline.rows.map(r=>r.slice))].sort();
const tr=[];names.forEach((n,i)=>{const rows=S.specs[n].rows.filter(r=>r.auc!=null);if(!rows.length)return;
 tr.push({type:'box',name:S.specs[n].label,x:rows.map(r=>r.slice),y:rows.map(r=>r.auc),boxpoints:'all',jitter:.6,pointpos:0,marker:{color:COL[n],size:4,opacity:.55},line:{color:COL[n],width:1},fillcolor:'rgba(0,0,0,0)',boxmean:true})});
tr.push({type:'scatter',mode:'markers',name:'logreg_lags',x:slices,y:slices.map(s=>S.logreg[s]),marker:{symbol:'diamond',size:13,color:'#16202a',line:{color:'#fff',width:1.5}}});
Plotly.newPlot('strip',tr,lay({boxmode:'group',yaxis:{title:'AUC',gridcolor:grid,zeroline:false},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:.5,y1:.5,line:{color:'#c0392b',dash:'dash',width:1}}]}),{displayModeBar:false,responsive:true});
// paired diff
const dt=[];Object.entries(S.comp).forEach(([n,c])=>{const ss=Object.keys(c.slices).sort();if(!ss.length)return;
 dt.push({type:'bar',name:S.specs[n].label,x:ss,y:ss.map(s=>c.slices[s].mean),error_y:{type:'data',array:ss.map(s=>c.slices[s].se?1.96*c.slices[s].se:0),visible:true},marker:{color:COL[n]}})});
if(dt.length)Plotly.newPlot('diff',dt,lay({barmode:'group',yaxis:{title:'AUC кандидата − базы',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'}}),{displayModeBar:false,responsive:true});
else document.getElementById('diff').innerHTML='<div class="sub" style="padding:60px 0;text-align:center">Кандидаты ещё не набрали данных</div>';
// violin
Plotly.newPlot('viol',names.filter(n=>auc(n).length).map(n=>({type:'violin',name:S.specs[n].label,y:auc(n),box:{visible:true},meanline:{visible:true},line:{color:COL[n]},fillcolor:COL[n]+'44',points:false})),lay({yaxis:{title:'AUC',gridcolor:grid},showlegend:false}),{displayModeBar:false,responsive:true});
// horizons
const hz=names.filter(n=>S.specs[n].rows.length).map(n=>{const R=S.specs[n].rows.filter(r=>r.h.every(x=>x!=null));return {type:'bar',name:S.specs[n].label,x:['h0','h1','h2'],y:[0,1,2].map(i=>R.length?mean(R.map(r=>r.h[i])):null),marker:{color:COL[n]}}});
Plotly.newPlot('hor',hz,lay({barmode:'group',yaxis:{title:'средний AUC',range:[.4,.62],gridcolor:grid}}),{displayModeBar:false,responsive:true});
// wall
Plotly.newPlot('wall',names.filter(n=>S.specs[n].rows.length).map(n=>({type:'scatter',mode:'markers',name:S.specs[n].label,y:S.specs[n].rows.map(r=>r.wall),marker:{color:COL[n],size:4,opacity:.6}})).concat([{type:'scatter',mode:'lines',name:'лимит 120 с',x:[0,S.planned],y:[120,120],line:{color:'#c0392b',dash:'dash'}}]),lay({yaxis:{title:'с',gridcolor:grid,rangemode:'tozero'}}),{displayModeBar:false,responsive:true});
// hypotheses table
document.getElementById('hyp').innerHTML='<tr><th>Раунд</th><th>Что меняет</th><th>Δ AUC</th><th>95% CI</th></tr>'+Object.entries(S.specs).filter(([n])=>n!=='hc_baseline').map(([n,s])=>{const c=S.comp[n]||{};const f=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(3);
 const cls=c.lo>0?'pos':(c.hi<0?'neg':'');return `<tr><td><b>${s.label}</b></td><td>${s.desc}</td><td class="${cls}">${f(c.mean)}</td><td>${c.lo==null?'—':'['+f(c.lo)+'; '+f(c.hi)+']'}</td></tr>`}).join('');
document.getElementById('git').innerHTML=S.log.map(l=>{const[h,t,...m]=l.split('|');return `<tr><td><code>${h}</code></td><td class="sub">${t}</td><td>${m.join('|')}</td></tr>`}).join('');
</script></body></html>"""

if __name__ == "__main__":
    if "--loop" in sys.argv:
        step = int(sys.argv[sys.argv.index("--loop") + 1])
        while True:
            try:
                main()
            except Exception as e:
                print("dashboard error:", e, flush=True)
            time.sleep(step)
    else:
        main()
