"""Live dashboard of the tactical session (design v2: 40 climb slices x 3 seeds per variant).
Reads runs/tactical/screens/hc2_<variant>_c*/results*.jsonl (and round 1: hc_*), watchdog.out, git, GPU;
writes runs/tactical/dashboard.html (data inlined, plotly from CDN, reloads every 20 s).
    python runs/tactical/dashboard.py            # write once
    python runs/tactical/dashboard.py --loop 20  # rewrite every 20 s
"""
import collections, datetime, glob, json, math, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "dashboard.html")
sys.path.insert(0, ROOT)
from hc2_compare import T  # t-quantiles, the same rule as the verdict script

PLANNED = 120  # trials per variant: 40 climb slices x 3 seeds
# name -> (label, Config change, round)
VARIANTS = collections.OrderedDict([
    ("base", ("Базовая v2", "дефолты, без изменений", "база")),
    ("skiponly", ("skip_only", "DIRECTION_HEAD_MODE=skip_only: направление только из линейного пропуска", "раунд 2")),
    ("ep3", ("3 эпохи", "EPOCHS=3 вместо 8", "раунд 2")),
    ("lr3e4", ("LR 3e-4", "LR=0.0003 вместо 1e-3", "раунд 2")),
    ("dir5", ("вес direction ×5", "LAMBDA_DIR=5.0", "раунд 2")),
    ("look20", ("окно 20", "LOOKBACK=20 вместо 60", "раунд 2")),
    ("nophys", ("без физики", "все физические члены = 0", "раунд 2")),
    ("calval", ("калибровка: value", "run.calibrate + CALIB_MODE=value", "раунд 3: балансировка лоссов")),
    ("calgrad", ("калибровка: gradient", "run.calibrate + CALIB_MODE=gradient (GradNorm-стиль, NT-101)", "раунд 3: балансировка лоссов")),
])
R1 = [("skip_only", -0.012, -0.101, 0.076), ("shrink 1.0 (83/100)", -0.010, -0.037, 0.017), ("dropout 0.5", -0.015, -0.029, -0.0001)]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20, cwd=REPO).stdout
    except Exception:
        return ""


def load(v):
    rows = []
    for f in sorted(glob.glob(os.path.join(ROOT, "screens", f"hc2_{v}_c*", "results*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            a = r.get("direction_auc") or {}
            ok = all(a.get(h) and a[h].get("auc") is not None for h in ("h0", "h1", "h2"))
            rows.append({"slice": r["data_end"][:10], "seed": r["seed"],
                         "auc": st.mean(a[h]["auc"] for h in ("h0", "h1", "h2")) if ok else None,
                         "h": [a[h]["auc"] if a.get(h) else None for h in ("h0", "h1", "h2")],
                         "wall": r.get("wall_s"), "train": (r.get("timings") or {}).get("train_s")})
    return rows


def main():
    data = {v: load(v) for v in VARIANTS}
    base = collections.defaultdict(dict)
    for r in data["base"]:
        if r["auc"] is not None:
            base[r["slice"]][r["seed"]] = r["auc"]
    comp = {}
    for v, rows in data.items():
        if v == "base":
            continue
        per = collections.defaultdict(list)
        for r in rows:
            b = base.get(r["slice"], {}).get(r["seed"])
            if r["auc"] is not None and b is not None:
                per[r["slice"]].append(r["auc"] - b)
        sl = {s: st.mean(x) for s, x in per.items()}
        c = {"slices": sl, "pairs": sum(len(x) for x in per.values())}
        if len(sl) >= 3:
            ms = list(sl.values()); m = st.mean(ms); se = st.stdev(ms) / math.sqrt(len(ms)); t = T.get(len(ms) - 1, 2.0)
            lo, hi = m - t * se, m + t * se
            c.update(mean=m, lo=lo, hi=hi, n=len(ms), up=sum(x > 0 for x in ms), down=sum(x < 0 for x in ms),
                     verdict=("принять к подтверждению" if lo > 0 and m >= 0.01 else ("хуже базы" if hi < 0 else "эффекта не видно")),
                     final=len(ms) >= 40)
        comp[v] = c
    procs = [l for l in sh('wmic process where "name=\'python.exe\'" get CommandLine').splitlines() if "cli screen" in l]
    running = collections.Counter()
    for l in procs:
        for v in VARIANTS:
            if f"hc2_{v}_c" in l:
                running[v] += 1
    walls = [r["wall"] for v in data for r in data[v] if r["wall"]]
    kills = [l.strip() for l in (open(os.path.join(ROOT, "watchdog.out"), encoding="utf-8", errors="ignore").read().splitlines()
                                 if os.path.exists(os.path.join(ROOT, "watchdog.out")) else []) if "kill" in l]
    done_total = sum(len(r) for r in data.values())
    rate = st.mean(walls[-60:]) / 3 if walls else None  # seconds per trial with 3 shards
    eta_min = round((len(VARIANTS) * PLANNED - done_total) * rate / 60) if rate else None
    state = {
        "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "planned": PLANNED,
        "variants": {v: {"label": VARIANTS[v][0], "desc": VARIANTS[v][1], "round": VARIANTS[v][2], "rows": data[v],
                         "running": running.get(v, 0)} for v in VARIANTS},
        "comp": comp, "r1": R1, "gpu": sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").strip(),
        "log": sh("git log --pretty=format:%h|%ad|%s --date=format:%H:%M -10").splitlines(),
        "shards": len(procs), "over_limit": sum(w > 120 for w in walls), "n_walls": len(walls), "kills": kills[-5:],
        "n_kills": len(kills), "eta_min": eta_min, "done_total": done_total, "logreg_r1": 0.554,
    }
    tmp = OUT + ".tmp"
    open(tmp, "w", encoding="utf-8").write(TEMPLATE.replace("__STATE__", json.dumps(state)))
    os.replace(tmp, OUT)


TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Tactical session</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202a;--mut:#667;--line:#e1e5ea;--ok:#199e70;--bad:#c0392b;--warn:#b7791f}
@media(prefers-color-scheme:dark){:root{--bg:#10151b;--card:#18202a;--ink:#e6ebf0;--mut:#93a0ad;--line:#2a3541}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
.w{max-width:1280px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:0 0 8px}
.sub{color:var(--mut);font-size:12px}.g{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));margin-top:12px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.full{grid-column:1/-1}
.kpi{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}.k{flex:1;min-width:150px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.k b{font-size:22px;display:block}.k span{color:var(--mut);font-size:12px}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%}
.row{display:grid;grid-template-columns:170px 1fr 150px;gap:8px;align-items:center;margin:5px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.pos{color:var(--ok);font-weight:600}.neg{color:var(--bad);font-weight:600}.warn{color:var(--warn)}code{background:var(--line);padding:0 4px;border-radius:3px}
.goal{border-left:4px solid #3987e5;padding:6px 10px;background:var(--card);border-radius:6px;margin-top:10px;font-size:13px}
</style></head><body><div class="w">
<h1>Тактическая сессия: прорыв в расчёте сети <span class="tag" id="live"></span></h1>
<div class="sub" id="sub"></div>
<div class="goal"><b>Цель:</b> выйти из «стратегической ловушки» (нет направленного навыка, AUC≈0,5) оптимизацией, поиском тактического прорыва в расчёте сети. Риск-менеджмент вынесен в отдельную независимую ветку. <b>Метрика:</b> среднее direction AUC по h0–h2 на валидационном блоке (screen, ≤2 мин на прогон). <b>Дизайн (зафиксирован до прогонов):</b> 40 срезов × 3 seed'а = 120 прогонов на вариант, единица вывода — срез, 10 финальных срезов не трогаются. <b>Правило:</b> 95% интервал по срезам целиком выше 0 и средняя разность ≥ +0,01, затем одна проверка на финальных срезах.</div>
<div class="kpi" id="kpi"></div>
<div class="g">
<div class="c full"><h2>Очередь вариантов</h2><div id="prog"></div></div>
<div class="c full"><h2>Эффект каждого варианта против базы (лес): разность AUC и 95% интервал по срезам</h2><div id="forest" style="height:360px"></div>
<div class="sub">Зелёная зона правее +0,01 — порог «принять к подтверждению». Интервал, пересекающий 0, значит «эффекта не видно». Показаны варианты, набравшие ≥3 срезов.</div></div>
<div class="c full"><h2>Карта срезов: разность AUC вариант − база по датам</h2><div id="heat" style="height:340px"></div>
<div class="sub">Строки — варианты, столбцы — срезы истории (слева старые). Красное — хуже базы, синее — лучше. Закономерность по эпохам говорит о режимной зависимости, а не о навыке.</div></div>
<div class="c full"><h2>База: AUC по срезам истории (точки — seed'ы, линия — среднее)</h2><div id="basets" style="height:300px"></div>
<div class="sub">Красная линия — 0,5 (монетка). Разброс среднего по эпохам и есть тот шум, который не позволял сравнивать на 5 срезах.</div></div>
<div class="c"><h2>Распределение AUC по прогонам</h2><div id="viol" style="height:340px"></div></div>
<div class="c"><h2>AUC по горизонтам h0 / h1 / h2</h2><div id="hor" style="height:340px"></div></div>
<div class="c"><h2>Время прогона, с</h2><div id="wall" style="height:300px"></div><div class="sub">Лимит владельца 120 с. Превысили: <span id="ovl"></span>.</div></div>
<div class="c"><h2>Сводка по гипотезам</h2><table id="hyp"></table></div>
<div class="c"><h2>Раунд 1 (5 срезов, закрыт): без эффекта</h2><table id="r1"></table><div class="sub">Метрика та же, но 5 срезов дают ошибку ±0,03: потому дизайн v2 использует 40.</div></div>
<div class="c"><h2>Здоровье прогона</h2><div id="health"></div></div>
<div class="c full"><h2>Последние коммиты nt-tactical</h2><table id="git"></table></div>
</div></div>
<script>
const S=__STATE__,V=Object.keys(S.variants);
const PAL=['#8a97a6','#3987e5','#d95926','#199e70','#8a5cc2','#c9a227','#e0457b','#2aa3b8','#6d8b2f'];
const COL=Object.fromEntries(V.map((v,i)=>[v,PAL[i%PAL.length]]));
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n);
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length,ink=getComputedStyle(document.body).color,grid=css('--line');
const lay=o=>Object.assign({margin:{l:52,r:12,t:8,b:44},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:ink,size:12},xaxis:{gridcolor:grid},yaxis:{gridcolor:grid},legend:{orientation:'h',y:-0.22}},o||{});
const cfg={displayModeBar:false,responsive:true},R=v=>S.variants[v].rows,A=v=>R(v).filter(r=>r.auc!=null).map(r=>r.auc),L=v=>S.variants[v].label;
const f3=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(3);
document.getElementById('live').textContent=S.shards?('идёт: '+S.shards+' шард'):'простой';document.getElementById('live').style.color=S.shards?'var(--ok)':'var(--mut)';
document.getElementById('sub').textContent='Обновлено '+S.now+' · перезагрузка каждые 20 с · GPU: '+(S.gpu||'н/д');
// KPI
const b=A('base'),cs=Object.entries(S.comp).filter(([v,c])=>c.mean!=null),best=cs.sort((x,y)=>y[1].mean-x[1].mean)[0];
const k=(v,l)=>`<div class="k"><b>${v}</b><span>${l}</span></div>`;
document.getElementById('kpi').innerHTML=
 k(b.length?mean(b).toFixed(3):'—','база v2, средний AUC ('+b.length+' прогонов)')+
 k(S.done_total+' / '+(V.length*S.planned),'прогонов всего'+(S.eta_min!=null?' · ещё ~'+Math.floor(S.eta_min/60)+' ч '+(S.eta_min%60)+' мин (оценка)':''))+
 k(best?f3(best[1].mean):'—',best?('лучший: '+L(best[0])+' — '+best[1].verdict):'лучший вариант: данных ещё нет')+
 k(S.over_limit+' / '+S.n_walls,'прогонов дольше 120 с')+k(S.n_kills,'зависших шардов перезапущено');
// progress
document.getElementById('prog').innerHTML=V.map(v=>{const n=R(v).length,p=Math.min(100,100*n/S.planned),run=S.variants[v].running;
 const st=n>=S.planned?'готово':(run?'идёт ('+run+' шард)':(n?'частично':'в очереди'));
 return `<div class="row"><div><b>${L(v)}</b><div class="sub">${S.variants[v].round}</div></div><div><div class="bar"><i style="width:${p}%;background:${COL[v]}"></i></div><div class="sub">${S.variants[v].desc}</div></div><div class="sub">${n}/${S.planned} · ${st}</div></div>`}).join('');
// forest
const fv=cs.map(x=>x[0]).sort((x,y)=>S.comp[y].mean-S.comp[x].mean);
if(fv.length)Plotly.newPlot('forest',[{type:'scatter',mode:'markers',y:fv.map(L),x:fv.map(v=>S.comp[v].mean),marker:{size:11,color:fv.map(v=>COL[v])},
 error_x:{type:'data',symmetric:false,array:fv.map(v=>S.comp[v].hi-S.comp[v].mean),arrayminus:fv.map(v=>S.comp[v].mean-S.comp[v].lo),thickness:2,width:6},
 text:fv.map(v=>S.comp[v].n+' срезов, '+S.comp[v].verdict),hovertemplate:'%{y}: %{x:>+.3f}<br>%{text}<extra></extra>'}],
 lay({xaxis:{title:'Δ AUC (вариант − база)',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'},yaxis:{autorange:'reversed'},showlegend:false,
 shapes:[{type:'rect',xref:'x',yref:'paper',x0:0.01,x1:0.2,y0:0,y1:1,fillcolor:'rgba(25,158,112,.10)',line:{width:0}}]}),cfg);
else document.getElementById('forest').innerHTML='<div class="sub" style="padding:90px 0;text-align:center">Варианты ещё не набрали данных (нужна база и ≥3 среза у варианта)</div>';
// heatmap
const sl=[...new Set(R('base').map(r=>r.slice))].sort(),hv=Object.keys(S.comp).filter(v=>Object.keys(S.comp[v].slices).length);
if(hv.length&&sl.length)Plotly.newPlot('heat',[{type:'heatmap',x:sl,y:hv.map(L),z:hv.map(v=>sl.map(s=>S.comp[v].slices[s]??null)),zmid:0,colorscale:'RdBu',reversescale:false,
 colorbar:{title:'Δ AUC'},hovertemplate:'%{y}<br>%{x}: %{z:>+.3f}<extra></extra>'}],lay({yaxis:{autorange:'reversed'},margin:{l:150,r:12,t:8,b:70},xaxis:{tickangle:-45}}),cfg);
else document.getElementById('heat').innerHTML='<div class="sub" style="padding:90px 0;text-align:center">Появится, когда у вариантов будут срезы</div>';
// baseline over time
const bm={};R('base').filter(r=>r.auc!=null).forEach(r=>(bm[r.slice]=bm[r.slice]||[]).push(r.auc));const bs=Object.keys(bm).sort();
Plotly.newPlot('basets',[{type:'scatter',mode:'markers',name:'seed',x:R('base').filter(r=>r.auc!=null).map(r=>r.slice),y:A('base'),marker:{color:'#8a97a6',size:5,opacity:.5}},
 {type:'scatter',mode:'lines+markers',name:'среднее по срезу',x:bs,y:bs.map(s=>mean(bm[s])),line:{color:'#3987e5'}}],
 lay({yaxis:{title:'AUC',gridcolor:grid},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:.5,y1:.5,line:{color:'#c0392b',dash:'dash',width:1}}]}),cfg);
// violin, horizons, wall
const vv=V.filter(v=>A(v).length);
Plotly.newPlot('viol',vv.map(v=>({type:'violin',name:L(v),y:A(v),box:{visible:true},meanline:{visible:true},line:{color:COL[v]},fillcolor:COL[v]+'44',points:false})),lay({yaxis:{title:'AUC',gridcolor:grid},showlegend:false}),cfg);
Plotly.newPlot('hor',vv.map(v=>{const Rr=R(v).filter(r=>r.h.every(x=>x!=null));return {type:'bar',name:L(v),x:['h0','h1','h2'],y:[0,1,2].map(i=>Rr.length?mean(Rr.map(r=>r.h[i])):null),marker:{color:COL[v]}}}),lay({barmode:'group',yaxis:{title:'средний AUC',range:[.44,.58],gridcolor:grid}}),cfg);
Plotly.newPlot('wall',vv.map(v=>({type:'box',name:L(v),y:R(v).map(r=>r.wall).filter(x=>x),marker:{color:COL[v]},boxpoints:false})).concat([{type:'scatter',mode:'lines',name:'лимит 120 с',x:vv.map(L),y:vv.map(()=>120),line:{color:'#c0392b',dash:'dash'}}]),lay({yaxis:{title:'с',gridcolor:grid,rangemode:'tozero'},showlegend:false}),cfg);
document.getElementById('ovl').textContent=S.over_limit+' из '+S.n_walls+' прогонов';
// summary table
document.getElementById('hyp').innerHTML='<tr><th>Вариант</th><th>Что меняет</th><th>срезов</th><th>Δ AUC</th><th>95% CI</th><th>вердикт</th></tr>'+V.filter(v=>v!=='base').map(v=>{const c=S.comp[v]||{};
 const cls=c.verdict==='принять к подтверждению'?'pos':(c.verdict==='хуже базы'?'neg':'');
 return `<tr><td><b>${L(v)}</b></td><td>${S.variants[v].desc}</td><td>${c.n||Object.keys((c.slices)||{}).length||0}${c.n&&!c.final?'<span class="warn"> (неполно)</span>':''}</td><td class="${cls}">${f3(c.mean)}</td><td>${c.lo==null?'—':'['+f3(c.lo)+'; '+f3(c.hi)+']'}</td><td class="${cls}">${c.verdict||'ждёт данных'}</td></tr>`}).join('');
document.getElementById('r1').innerHTML='<tr><th>Вариант</th><th>Δ AUC</th><th>95% CI</th></tr>'+S.r1.map(r=>`<tr><td>${r[0]}</td><td>${f3(r[1])}</td><td>[${f3(r[2])}; ${f3(r[3])}]</td></tr>`).join('')+`<tr><td class="sub">logreg_lags (5 срезов)</td><td colspan="2" class="sub">${S.logreg_r1} — в пределах шума базы 0,522</td></tr>`;
document.getElementById('health').innerHTML=`<table><tr><td>Прогонов дольше лимита 2 мин</td><td><b>${S.over_limit}</b> из ${S.n_walls}</td></tr><tr><td>Зависших шардов убито watchdog'ом</td><td><b>${S.n_kills}</b></td></tr></table>`+(S.kills.length?'<div class="sub" style="margin-top:6px">'+S.kills.map(x=>'<div>'+x+'</div>').join('')+'</div>':'<div class="sub" style="margin-top:6px">Зависаний пока нет.</div>');
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
