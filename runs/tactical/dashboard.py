"""Live dashboard of the tactical session (v3: the current task first, then the finished rounds).
Reads runs/tactical (screen results incl. head_metrics, the candidate SPEC's runs, round archives, watchdog, git, GPU)
and writes runs/tactical/dashboard.html (data inlined, plotly from CDN, reloads every 20 s).
    python runs/tactical/dashboard.py            # write once
    python runs/tactical/dashboard.py --loop 20  # rewrite every 20 s
"""
import collections, datetime, glob, json, math, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(ROOT))
OUT = os.path.join(ROOT, "dashboard.html")
sys.path.insert(0, ROOT)
from hc2_compare import T

H = ("h0", "h1", "h2")
CAND = collections.OrderedDict([("cand_c1_skip", ("C1 skip_only", 10)), ("cand_c1_base", ("C1 база", 10)),
                                ("cand_c2_skip", ("C2 skip_only, 7 дней", 12)), ("cand_c2_base", ("C2 дефолт, 7 дней", 12))])
V2 = collections.OrderedDict([("skiponly", "skip_only"), ("ep3", "3 эпохи"), ("lr3e4", "LR 3e-4"), ("dir5", "direction ×5"),
                              ("look20", "окно 20"), ("nophys", "без физики"), ("calval", "калибровка value"),
                              ("calgrad", "калибровка gradient")])
HM = [("delta", "corr", "Цена: корреляция"), ("delta", "skill_vs_zero", "Цена: skill vs 0"),
      ("direction", "auc", "Направление: AUC"), ("direction", "hit_rate", "Направление: hit rate"),
      ("direction", "brier", "Направление: Brier ↓"), ("variance", "crpss", "Уверенность: CRPSS"),
      ("variance", "coverage90", "Уверенность: покрытие 90%"), ("variance", "corr_var_err2_spearman", "Уверенность: Spearman var~err²")]


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20, cwd=REPO).stdout
    except Exception:
        return ""


def rows_of(pattern):
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "screens", pattern, "results*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            try:
                r = json.loads(l)
            except Exception:
                continue
            a = r.get("direction_auc") or {}
            ok = all(a.get(h) and a[h].get("auc") is not None for h in H)
            out.append({"slice": r["data_end"][:16], "seed": r["seed"], "wall": r.get("wall_s"),
                        "auc": st.mean(a[h]["auc"] for h in H) if ok else None,
                        "h": [a[h]["auc"] if a.get(h) else None for h in H], "hm": r.get("head_metrics")})
    return out


def tci(xs):
    if len(xs) < 2:
        return None
    m = st.mean(xs); se = st.stdev(xs) / math.sqrt(len(xs)); t = T.get(len(xs) - 1, 2.0)
    return {"mean": m, "lo": m - t * se, "hi": m + t * se, "n": len(xs)}


def hm_mean(rows, grp, key):
    v = [r["hm"][h][grp].get(key) for r in rows if r.get("hm") for h in H
         if r["hm"].get(h) and r["hm"][h].get(grp) and r["hm"][h][grp].get(key) is not None]
    return st.mean(v) if v else None


def main():
    cand = {n: rows_of(n) for n in CAND}
    # ---- C1
    s1 = {r["seed"]: r for r in cand["cand_c1_skip"]}; b1 = {r["seed"]: r for r in cand["cand_c1_base"]}
    h1 = [r["h"][1] for r in s1.values() if r["h"][1] is not None]
    d1 = [s1[k]["auc"] - b1[k]["auc"] for k in s1 if k in b1 and s1[k]["auc"] is not None and b1[k]["auc"] is not None]
    m1 = [r["auc"] for r in s1.values() if r["auc"] is not None]
    c1 = {"a": {"val": st.mean(h1) if h1 else None, "thr": 0.75, "pass": (st.mean(h1) >= 0.75) if h1 else None},
          "b": {"ci": tci(d1), "pass": (tci(d1)["lo"] > 0) if tci(d1) else None},
          "c": {"val": st.mean(m1) if m1 else None, "thr": 0.756, "pass": (st.mean(m1) >= 0.756) if m1 else None},
          "done": len(d1) >= 10}
    # ---- C2
    s2 = {(r["slice"], r["seed"]): r for r in cand["cand_c2_skip"]}; b2 = {(r["slice"], r["seed"]): r for r in cand["cand_c2_base"]}
    per = collections.defaultdict(list)
    for k in s2:
        if k in b2 and s2[k]["auc"] is not None and b2[k]["auc"] is not None:
            per[k[0]].append(s2[k]["auc"] - b2[k]["auc"])
    d2 = {s: st.mean(v) for s, v in per.items()}
    g_crpss = (hm_mean(cand["cand_c2_skip"], "variance", "crpss") or 0) - (hm_mean(cand["cand_c2_base"], "variance", "crpss") or 0) \
        if cand["cand_c2_skip"] and cand["cand_c2_base"] else None
    g_cov = (hm_mean(cand["cand_c2_skip"], "variance", "coverage90") or 0) - (hm_mean(cand["cand_c2_base"], "variance", "coverage90") or 0) \
        if cand["cand_c2_skip"] and cand["cand_c2_base"] else None
    ci2 = tci(list(d2.values()))
    c2 = {"a": {"ci": ci2, "pass": (ci2["lo"] > 0) if ci2 else None, "slices": d2},
          "g": {"crpss": g_crpss, "cov": g_cov, "pass": (g_crpss >= -0.05 and g_cov >= -0.05) if g_crpss is not None else None},
          "done": sum(len(v) for v in per.values()) >= 12}
    # ---- 9-head table for the candidate runs
    heads = {n: {f"{g}.{k}": hm_mean(cand[n], g, k) for g, k, _ in HM} for n in CAND}
    perh = {n: {h: {f"{g}.{k}": (st.mean([r["hm"][h][g][k] for r in cand[n] if r.get("hm") and r["hm"].get(h) and r["hm"][h].get(g)
                                            and r["hm"][h][g].get(k) is not None]) if any(r.get("hm") for r in cand[n]) else None)
                    for g, k, _ in HM} for h in H} for n in CAND}
    # ---- round 2/3 archive (40 slices x 3 seeds)
    def load2(v):
        d = {}
        for r in rows_of(f"hc2_{v}_c*"):
            if r["auc"] is not None:
                d[(r["slice"], r["seed"])] = r["auc"]
        return d
    base2 = load2("base"); v2 = {}
    for v in V2:
        d = load2(v); g = collections.defaultdict(list)
        for k in d:
            if k in base2:
                g[k[0]].append(d[k] - base2[k])
        c = tci([st.mean(x) for x in g.values()])
        if c:
            c["verdict"] = "принять" if c["lo"] > 0 and c["mean"] >= 0.01 else ("хуже базы" if c["hi"] < 0 else "эффекта нет")
            c["slices"] = {s: st.mean(x) for s, x in g.items()}
        v2[v] = c
    hc3 = {}
    for n in ["default"] + [f"cand{i:02d}" for i in range(1, 11)]:
        rr = [r["auc"] for r in rows_of(f"hc3_{n}") if r["auc"] is not None]
        if rr:
            hc3[n] = {"mean": st.mean(rr), "n": len(rr)}
    procs = [l for l in sh('wmic process where "name=\'python.exe\'" get CommandLine').splitlines() if "cli screen" in l]
    running = collections.Counter()
    for l in procs:
        for n in CAND:
            if f"{n}.yaml" in l:
                running[n] += 1
    walls = [r["wall"] for n in CAND for r in cand[n] if r["wall"]]
    state = {
        "now": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader").strip(),
        "shards": len(procs), "cand": {n: {"label": CAND[n][0], "planned": CAND[n][1], "done": len(cand[n]), "running": running.get(n, 0),
                                           "walls": [r["wall"] for r in cand[n] if r["wall"]]} for n in CAND},
        "c1": c1, "c2": c2, "heads": heads, "perh": perh, "hm": [[g + "." + k, lab] for g, k, lab in HM],
        "c1rows": {"skip": [[r["seed"], r["h"]] for r in sorted(s1.values(), key=lambda r: r["seed"])],
                   "base": [[r["seed"], r["h"]] for r in sorted(b1.values(), key=lambda r: r["seed"])]},
        "v2": v2, "v2lab": V2, "base2": st.mean(base2.values()) if base2 else None, "hc3": hc3,
        "log": sh("git log --pretty=format:%h|%ad|%s --date=format:%m-%d %H:%M -10").splitlines(),
        "over": sum(w > 120 for w in walls), "nw": len(walls),
    }
    tmp = OUT + ".tmp"
    open(tmp, "w", encoding="utf-8").write(TEMPLATE.replace("__STATE__", json.dumps(state)))
    os.replace(tmp, OUT)


TEMPLATE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="20"><title>Tactical session</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.27.0/plotly.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16202a;--mut:#667;--line:#e1e5ea;--ok:#199e70;--bad:#c0392b;--warn:#b7791f;--acc:#3987e5}
@media(prefers-color-scheme:dark){:root{--bg:#10151b;--card:#18202a;--ink:#e6ebf0;--mut:#93a0ad;--line:#2a3541}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,sans-serif}
.w{max-width:1280px;margin:0 auto;padding:16px}h1{font-size:20px;margin:0 0 2px}h2{font-size:15px;margin:0 0 8px}h3{font-size:13px;margin:10px 0 4px}
.sec{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:22px 0 4px}
.sub{color:var(--mut);font-size:12px}.g{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));margin-top:8px}
.c{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px}.full{grid-column:1/-1}
.bar{height:10px;background:var(--line);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--acc)}
.row{display:grid;grid-template-columns:190px 1fr 160px;gap:8px;align-items:center;margin:5px 0}
table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
.tag{display:inline-block;padding:1px 8px;border-radius:9px;font-size:11px;border:1px solid var(--line)}
.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.wait{color:var(--mut)}code{background:var(--line);padding:0 4px;border-radius:3px}
.goal{border-left:4px solid var(--acc);padding:6px 10px;background:var(--card);border-radius:6px;margin-top:10px;font-size:13px}
.done li{margin:3px 0}
</style></head><body><div class="w">
<h1>Тактическая сессия: прорыв в расчёте сети <span class="tag" id="live"></span></h1>
<div class="sub" id="sub"></div>
<div class="goal"><b>Цель:</b> выйти из «ловушки» (нет навыка направления, AUC≈0,5) поиском тактического прорыва в расчёте сети; риск-менеджмент — отдельная ветка. <b>Правила владельца:</b> всегда все 9 выходов (цена, направление, уверенность × 3 горизонта); стандарт отсева 6 срезов × 2 seed'а (ошибка ≤0,05, ложные победы ≤10%); screen-прогон ≤2 мин, 7-дневные — по разрешению.</div>

<div class="sec">Сейчас: проверка кандидата 0,805</div>
<div class="g">
<div class="c full"><h2>Кандидат: <code>skip_only</code>, срез 2022-04-19, seed 2, голова h1 = 0,805</h2>
<div class="sub">Блок проверки — суббота 16.04.2022 (Пасха, штиль). Простое правило «против последних 10 минут» даёт там 0,756. На 40 срезах skip_only в среднем +0,008 (эффекта нет). Требования зафиксированы до прогона: <code>runs/tactical/cand_0805/SPEC.md</code>.</div>
<div id="prog" style="margin-top:8px"></div></div>
<div class="c"><h2>C1: повторяется ли на том же блоке (10 новых seed'ов)</h2><table id="c1"></table><div id="c1plot" style="height:240px"></div></div>
<div class="c"><h2>C2: переносится ли на новые блоки, 7 дней (6 срезов × 2 seed'а)</h2><table id="c2"></table><div id="c2plot" style="height:240px"></div></div>
<div class="c full"><h2>Все 9 выходов: средние по прогонам кандидата</h2><table id="heads"></table>
<div class="sub">Цена: корреляция прогноза с фактом и выигрыш против «цена не изменится». Направление: AUC, доля угаданных, Brier (меньше — лучше). Уверенность: CRPSS против постоянной дисперсии (&gt;0 — лучше), покрытие 90%-интервала (идеал 0,90), связь предсказанной дисперсии с ошибкой.</div>
<h3>По горизонтам</h3><div id="perh" style="height:330px"></div></div>
</div>

<div class="sec">Завершено</div>
<div class="g">
<div class="c full"><h2>Раунды 2–3: 8 вариантов против базы, 40 срезов × 3 seed'а (только направление)</h2><div id="forest" style="height:330px"></div>
<div class="sub">Ни один вариант не прошёл правило (интервал выше 0 и эффект ≥ +0,01). Хуже базы: LR 3e-4, окно 20. Измерялось только направление: головы цены и уверенности тогда не логировались.</div></div>
<div class="c"><h2>Итоги по задачам</h2><ul class="done" id="done"></ul></div>
<div class="c"><h2>7 дней, топ кандидатов старой кампании (остановлено)</h2><table id="hc3"></table><div class="sub">Остановлено по решению владельца после 3 полных конфигураций: прироста против дефолта нет.</div></div>
<div class="c"><h2>Здоровье и лимиты</h2><table id="health"></table></div>
<div class="c"><h2>Коммиты nt-tactical</h2><table id="git"></table></div>
</div></div>
<script>
const S=__STATE__,ink=getComputedStyle(document.body).color,grid=getComputedStyle(document.documentElement).getPropertyValue('--line');
const lay=o=>Object.assign({margin:{l:52,r:12,t:8,b:40},paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',font:{color:ink,size:12},xaxis:{gridcolor:grid},yaxis:{gridcolor:grid},legend:{orientation:'h',y:-0.25}},o||{});
const cfg={displayModeBar:false,responsive:true},f3=v=>v==null?'—':(v>=0?'+':'')+v.toFixed(3),p3=v=>v==null?'—':v.toFixed(3);
const mark=p=>p==null?'<span class="wait">ждёт данных</span>':(p?'<span class="ok">✔ пройдено</span>':'<span class="bad">✘ не пройдено</span>');
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
document.getElementById('live').textContent=S.shards?('идёт: '+S.shards+' процесса'):'простой';document.getElementById('live').style.color=S.shards?'var(--ok)':'var(--mut)';
document.getElementById('sub').textContent='Обновлено '+S.now+' · перезагрузка каждые 20 с · GPU: '+(S.gpu||'н/д');
// progress
document.getElementById('prog').innerHTML=Object.entries(S.cand).map(([n,c])=>{const p=Math.min(100,100*c.done/c.planned);
 const w=c.walls.length?Math.round(mean(c.walls))+' с/прогон':'';const st=c.done>=c.planned?'готово':(c.running?'идёт':(c.done?'частично':'в очереди'));
 return `<div class="row"><div><b>${c.label}</b></div><div class="bar"><i style="width:${p}%"></i></div><div class="sub">${c.done}/${c.planned} · ${st} ${w}</div></div>`}).join('');
// C1
const c1=S.c1;document.getElementById('c1').innerHTML=
 `<tr><th>Требование</th><th>Факт</th><th></th></tr>
 <tr><td>(a) h1 AUC в среднем ≥ 0,75</td><td>${p3(c1.a.val)}</td><td>${mark(c1.a.pass)}</td></tr>
 <tr><td>(b) лучше базы, интервал по seed'ам &gt; 0</td><td>${c1.b.ci?f3(c1.b.ci.mean)+' ['+f3(c1.b.ci.lo)+'; '+f3(c1.b.ci.hi)+']':'—'}</td><td>${mark(c1.b.pass)}</td></tr>
 <tr><td>(c) среднее h0–h2 ≥ 0,756 (правило «против 10 мин»)</td><td>${p3(c1.c.val)}</td><td>${mark(c1.c.pass)}</td></tr>
 <tr><td colspan="3" class="sub">${c1.done?'Все 10 seed\'ов готовы: вердикт окончательный.':'Неполные данные: вердикт предварительный.'}</td></tr>`;
const tr1=[];[['skip','skip_only','#3987e5'],['base','база','#8a97a6']].forEach(([k,l,c])=>{const R=S.c1rows[k];if(R.length)tr1.push({type:'scatter',mode:'markers',name:l+' h1',x:R.map(r=>r[0]),y:R.map(r=>r[1][1]),marker:{color:c,size:9}})});
Plotly.newPlot('c1plot',tr1,lay({xaxis:{title:'seed',gridcolor:grid,dtick:1},yaxis:{title:'h1 AUC',gridcolor:grid},shapes:[{type:'line',xref:'paper',x0:0,x1:1,y0:.75,y1:.75,line:{color:'#199e70',dash:'dot'}},{type:'line',xref:'paper',x0:0,x1:1,y0:.5,y1:.5,line:{color:'#c0392b',dash:'dash',width:1}}]}),cfg);
// C2
const c2=S.c2;document.getElementById('c2').innerHTML=
 `<tr><th>Требование</th><th>Факт</th><th></th></tr>
 <tr><td>(a) разность AUC, интервал по срезам &gt; 0</td><td>${c2.a.ci?f3(c2.a.ci.mean)+' ['+f3(c2.a.ci.lo)+'; '+f3(c2.a.ci.hi)+'], срезов '+c2.a.ci.n:'—'}</td><td>${mark(c2.a.pass)}</td></tr>
 <tr><td>(b) уверенность: CRPSS и покрытие не хуже −0,05</td><td>CRPSS ${f3(c2.g.crpss)} · покрытие ${f3(c2.g.cov)}</td><td>${mark(c2.g.pass)}</td></tr>
 <tr><td colspan="3" class="sub">${c2.done?'Все 12 пар готовы: вердикт окончательный.':'Неполные данные: вердикт предварительный.'}</td></tr>`;
const ss=Object.keys(c2.a.slices||{}).sort();
if(ss.length)Plotly.newPlot('c2plot',[{type:'bar',x:ss,y:ss.map(s=>c2.a.slices[s]),marker:{color:ss.map(s=>c2.a.slices[s]>=0?'#3987e5':'#d95926')}}],lay({yaxis:{title:'Δ AUC',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'},showlegend:false}),cfg);
else document.getElementById('c2plot').innerHTML='<div class="sub" style="padding:80px 0;text-align:center">7-дневные прогоны идут (~8 мин каждый)</div>';
// heads table
const names=Object.keys(S.cand);
document.getElementById('heads').innerHTML='<tr><th>Метрика (среднее h0–h2)</th>'+names.map(n=>`<th>${S.cand[n].label}</th>`).join('')+'</tr>'+
 S.hm.map(([k,l])=>'<tr><td>'+l+'</td>'+names.map(n=>`<td>${p3(S.heads[n][k])}</td>`).join('')+'</tr>').join('');
const hk=['delta.corr','direction.auc','variance.crpss','variance.coverage90'],hl={'delta.corr':'Цена corr','direction.auc':'Напр. AUC','variance.crpss':'Уверенность CRPSS','variance.coverage90':'Покрытие 90%'};
const col={cand_c1_skip:'#3987e5',cand_c1_base:'#8a97a6',cand_c2_skip:'#199e70',cand_c2_base:'#c9a227'};
const tp=[];names.forEach(n=>['h0','h1','h2'].forEach((h,i)=>{}));
names.forEach(n=>{if(!S.cand[n].done)return;tp.push({type:'bar',name:S.cand[n].label,x:hk.flatMap(k=>['h0','h1','h2'].map(h=>hl[k]+' '+h)),y:hk.flatMap(k=>['h0','h1','h2'].map(h=>S.perh[n][h][k])),marker:{color:col[n]}})});
if(tp.length)Plotly.newPlot('perh',tp,lay({barmode:'group',xaxis:{tickangle:-35},margin:{l:52,r:12,t:8,b:110}}),cfg);
// forest v2
const fv=Object.keys(S.v2).filter(v=>S.v2[v]).sort((a,b)=>S.v2[b].mean-S.v2[a].mean);
Plotly.newPlot('forest',[{type:'scatter',mode:'markers',y:fv.map(v=>S.v2lab[v]+' ('+S.v2[v].verdict+')'),x:fv.map(v=>S.v2[v].mean),marker:{size:11,color:fv.map(v=>S.v2[v].verdict==='хуже базы'?'#c0392b':(S.v2[v].verdict==='принять'?'#199e70':'#3987e5'))},
 error_x:{type:'data',symmetric:false,array:fv.map(v=>S.v2[v].hi-S.v2[v].mean),arrayminus:fv.map(v=>S.v2[v].mean-S.v2[v].lo),thickness:2,width:6}}],
 lay({xaxis:{title:'Δ AUC против базы ('+(S.base2?S.base2.toFixed(3):'—')+')',gridcolor:grid,zeroline:true,zerolinecolor:'#c0392b'},yaxis:{autorange:'reversed'},showlegend:false,margin:{l:230,r:12,t:8,b:40},
 shapes:[{type:'rect',xref:'x',yref:'paper',x0:0.01,x1:0.1,y0:0,y1:1,fillcolor:'rgba(25,158,112,.10)',line:{width:0}}]}),cfg);
document.getElementById('done').innerHTML=[
 '<b>Раунд 1</b> (5 срезов): ограничения нелинейного пути направления — без эффекта.',
 '<b>Раунды 2–3</b> (40 срезов): 7 Config-вариантов и балансировка лоссов — без эффекта, 2 хуже.',
 '<b>Шум измерения:</b> одна голова на 6-часовом блоке ±0,10; стандарт отсева 6×2 принят владельцем.',
 '<b>Срез 2022-04-19:</b> Пасха 16.04.2022, штиль; тривиальное правило «против 10 мин» 0,756 — режим, не навык.',
 '<b>9 голов:</b> измерение добавлено (b2e9884), ответы сети сохраняются в preds/*.npz.'].map(x=>'<li>'+x+'</li>').join('');
document.getElementById('hc3').innerHTML='<tr><th>Конфиг</th><th>срезов</th><th>AUC</th></tr>'+Object.entries(S.hc3).map(([n,v])=>`<tr><td>${n}</td><td>${v.n}</td><td>${p3(v.mean)}</td></tr>`).join('');
document.getElementById('health').innerHTML=`<tr><td>Screen-прогоны кандидата дольше 120 с</td><td>${S.over} из ${S.nw}</td></tr><tr><td>7-дневные прогоны</td><td>разрешены владельцем (≈8 мин)</td></tr>`;
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
