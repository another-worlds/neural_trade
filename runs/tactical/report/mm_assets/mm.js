// ---------------------------------------------------------------- multimodal report: components (the helper prelude of positive_assets/pos.js is prepended by the builder)
const ST={works:P.good,near:P.gold,nogain:P.neutral,wip:P.violet,newdata:P.salmon},SYM={works:'▲',near:'◐',nogain:'–',wip:'…',newdata:'◆'};
const fA=v=>fmt(v,4,true),fB=v=>fmt(v*100,2,true)+'%';
const T$=(s,a,t,p)=>el('text',Object.assign({'font-size':12,fill:P.ink2},a),p||s,t);
const COL={view:P.violet,c1:P.neutral,c2a:P.tan,c2b:P.pink,look2:P.gold};

// horizontal grouped bars. rows:[{label,sym,color,bars:[{v,color,tick}]}]; o:{W,L,R,top,rowH,barH,min,max,fmt,ref:[{v,label}]}
function hbars(svg,rows,o){
  svg.innerHTML='';const W=o.W,L=o.L,R=o.R||90,top=o.top||46,rh=o.rowH,bh=o.barH;o.min=o.min-(o.max-o.min)*(o.min<0?.1:0);const x=v=>L+(W-L-R)*(v-o.min)/(o.max-o.min),H=top+rows.length*rh+14;
  svg.setAttribute('viewBox',`0 0 ${W} ${H}`);
  (o.ref||[]).forEach(r=>{el('line',{x1:x(r.v),x2:x(r.v),y1:top-10,y2:H-10,stroke:P.axis,'stroke-dasharray':'4 3'},svg);T$(0,{x:x(r.v),y:top-16,'text-anchor':'middle','font-size':11,fill:P.muted},r.label,svg)});
  el('line',{x1:x(0),x2:x(0),y1:top-10,y2:H-10,stroke:P.mut||P.muted},svg);
  const items=[];
  rows.forEach((r,i)=>{const y0=top+i*rh;
    if(i%2===0&&o.band)el('rect',{x:0,y:y0-3,width:W,height:rh,fill:'#ffffff',opacity:.025},svg);
    T$(0,{x:L-12,y:y0+rh/2+3,'text-anchor':'end','font-size':o.fs||13,fill:P.ink},r.label,svg);
    if(r.sym)T$(0,{x:12,y:y0+rh/2+4,'font-size':13,fill:ST[r.sym]||P.muted},SYM[r.sym]+(r.symTxt?' '+r.symTxt:''),svg);
    r.bars.forEach((b,j)=>{const yy=y0+(rh-bh*r.bars.length-(r.bars.length-1)*3)/2+j*(bh+3),x0=x(0),x1=x(b.v);
      const rc=el('rect',{x:Math.min(x0,x1),y:yy,width:0,height:bh,rx:3,fill:b.color},svg),tx=el('text',{x:x1+(b.v>=0?6:-6),y:yy+bh-1,'text-anchor':b.v>=0?'start':'end','font-size':11,fill:P.ink2,opacity:0},svg,(o.fmt||fA)(b.v));
      let tk=null;if(b.tick!=null)tk=el('line',{x1:x(b.tick),x2:x(b.tick),y1:yy-2,y2:yy+bh+2,stroke:P.ink,'stroke-width':2,opacity:0},svg);
      items.push({rc,tx,tk,x0,x1})})});
  run(1100,p=>{const q=easeOut(p);items.forEach(it=>{it.rc.setAttribute('x',Math.min(it.x0,it.x0+(it.x1-it.x0)*q));it.rc.setAttribute('width',Math.abs((it.x1-it.x0)*q));it.tx.setAttribute('opacity',p>.7?1:0);if(it.tk)it.tk.setAttribute('opacity',p>.5?1:0)})},120);
}

// vertical grouped bars in a panel of a given svg. c:{x,y,w,h,title,groups:[{label,note,bars:[{v,lo,hi,color,name}]}],ref:{v,label},fmt}
function panel(svg,c){
  const vals=[0,c.ref?c.ref.v*1.25:0];c.groups.forEach(g=>g.bars.forEach(b=>{vals.push(b.v);if(b.lo!=null){vals.push(b.lo,b.hi)}}));
  let lo=Math.min(...vals),hi=Math.max(...vals);const pd=(hi-lo)*.12;hi+=pd;if(lo<0)lo-=pd;
  const T0=c.y+34,B0=c.y+c.h-30,y=v=>T0+(B0-T0)*(1-(v-lo)/(hi-lo)),gw=c.w/c.groups.length,items=[];
  T$(0,{x:c.x,y:c.y+14,'font-size':12.5,fill:P.ink,'font-weight':600},c.title,svg);
  el('line',{x1:c.x,x2:c.x+c.w,y1:y(0),y2:y(0),stroke:P.muted,'stroke-width':1},svg);
  if(c.ref){el('line',{x1:c.x,x2:c.x+c.w,y1:y(c.ref.v),y2:y(c.ref.v),stroke:P.good,'stroke-dasharray':'4 3','stroke-width':1.2,opacity:.85},svg);T$(0,{x:c.x+c.w,y:y(c.ref.v)-3,'text-anchor':'end','font-size':9.5,fill:P.good},c.ref.label,svg)}
  c.groups.forEach((g,i)=>{const gx=c.x+i*gw,n=g.bars.length,bw=Math.min(30,(gw-18)/n-4),start=gx+(gw-n*(bw+4)+4)/2;
    g.bars.forEach((b,j)=>{const bx=start+j*(bw+4),y0=y(0),y1=y(b.v);
      const rc=el('rect',{x:bx,y:y0,width:bw,height:0,rx:2,fill:b.color},svg);let wh=null,tt=null;
      if(b.lo!=null){wh=el('line',{x1:bx+bw/2,x2:bx+bw/2,y1:y(b.lo),y2:y(b.hi),stroke:P.ink,'stroke-width':1.4,opacity:0},svg)}
      tt=el('text',{x:bx+bw/2,y:b.v>=0?y1-(b.hi!=null?Math.abs(y(b.hi)-y1)+3:4):y1+12,'text-anchor':'middle','font-size':9.5,fill:P.ink2,opacity:0},svg,(c.fmt||fA)(b.v));
      items.push({rc,wh,tt,y0,y1})});
    T$(0,{x:gx+gw/2,y:c.y+c.h-12,'text-anchor':'middle','font-size':11,fill:P.muted},g.label,svg);
    if(g.note)T$(0,{x:gx+gw/2,y:c.y+c.h-1,'text-anchor':'middle','font-size':10,fill:g.noteColor||P.muted},g.note,svg)});
  return p=>{const q=easeOut(p);items.forEach(it=>{const yy=it.y0+(it.y1-it.y0)*q;it.rc.setAttribute('y',Math.min(it.y0,yy));it.rc.setAttribute('height',Math.abs(yy-it.y0));
    if(it.wh)it.wh.setAttribute('opacity',p>.8?1:0);it.tt.setAttribute('opacity',p>.8?1:0)})};
}

function ring(svg,cx,cy,r,frac,color,big,small,ref){
  el('circle',{cx,cy,r,fill:'none',stroke:P.grid,'stroke-width':14},svg);
  const mx=Math.max(frac,0),cir=2*Math.PI*r,arc=el('circle',{cx,cy,r,fill:'none',stroke:color,'stroke-width':14,'stroke-linecap':'round',transform:`rotate(-90 ${cx} ${cy})`,'stroke-dasharray':`0 ${cir}`},svg);
  if(ref!=null){const a=(ref*Math.min(1,1)*2*Math.PI)-Math.PI/2;el('line',{x1:cx+(r-12)*Math.cos(a),y1:cy+(r-12)*Math.sin(a),x2:cx+(r+12)*Math.cos(a),y2:cy+(r+12)*Math.sin(a),stroke:P.ink,'stroke-width':2},svg)}
  const bt=T$(0,{x:cx,y:cy+8,'text-anchor':'middle','font-size':30,fill:P.ink,'font-weight':650},big,svg);T$(0,{x:cx,y:cy+r+34,'text-anchor':'middle','font-size':12.5,fill:P.ink2},small,svg);
  return p=>{arc.setAttribute('stroke-dasharray',`${cir*Math.min(mx,1)*easeOut(p)} ${cir}`)};
}

// ---------------------------------------------------------------- hero bars
function heroBars(){
  const svg=document.getElementById('heroBars'),rows=D.hero.map(h=>({label:h.name,sym:h.st,bars:[{v:h.look1,color:P.violet,tick:h.c2a1},{v:h.look2,color:P.gold,tick:h.c2a2}]}));
  const all=D.hero.flatMap(h=>[h.look1,h.look2,h.c2a1,h.c2a2,0.005]);
  hbars(svg,rows,{W:1200,L:300,R:110,top:46,rowH:36,barH:12,min:Math.min(0,...all)*1.2,max:Math.max(...all)*1.1,band:1,ref:[{v:0.005,label:'порог +0,005'}]})}

// ---------------------------------------------------------------- the six views: a card chart per candidate, switchable variants
function cardCharts(){
  D.cands.forEach(c=>{
    const sA=document.getElementById('vch_'+c.id+'A'),sB=document.getElementById('vch_'+c.id+'B');if(!sA)return;
    const draw=i=>{const r=c.runs[i];
      [[sA,'A'],[sB,'B']].forEach(([svg,t])=>{svg.innerHTML='';const f=t==='A'?(v=>v):(v=>v*100),fm=t==='A'?fA:fB2;
        const groups=['look1','look2'].map((lk,k)=>{const d=r[lk][t];return {label:'взгляд '+(k+1),note:d.pass?'✓ прошёл':'',noteColor:P.good,bars:[
          {v:f(d.view),lo:f(d.lo),hi:f(d.hi),color:COL.view},{v:f(d.c1),color:COL.c1},{v:f(d.c2a),color:COL.c2a}]}});
        const step=panel(svg,{x:6,y:0,w:318,h:250,title:t==='A'?'A: прирост к V3 (скилл)':'B: Брайер к H3, %',groups,ref:t==='A'?{v:0.005,label:'порог +0,005'}:{v:1,label:'порог +1%'},fmt:t==='A'?fA:(v=>fmt(v,2,true)+'%')});
        run(900,step)})};
    function fB2(v){return fmt(v,2,true)+'%'}
    let started=false;once(sA,()=>{started=true;draw(0)},.3);
    const seg=document.querySelector('.seg[data-cand="'+c.id+'"]');
    if(seg)seg.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{seg.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');draw(+b.dataset.i)}));
  })}

// ---------------------------------------------------------------- concept map
function conceptMap(){
  const svg=document.getElementById('mapSvg'),card=document.getElementById('mapCard');svg.innerHTML='';
  const cx=550,cy=285,rx=410,ry=205,N=D.nodes,nodes=[];
  const defs=el('defs',{},svg);
  const c0=el('g',{},svg);
  // edges first
  const edges=[];
  N.forEach((n,i)=>{const a=-Math.PI/2+i*2*Math.PI/N.length,x=cx+rx*Math.cos(a),y=cy+ry*Math.sin(a);n.x=x;n.y=y;
    const mx=(x+cx)/2+(y-cy)*.12,my=(y+cy)/2-(x-cx)*.12,col=ST[n.status],act=n.status==='works'||n.status==='near';
    const path=el('path',{d:`M${x},${y}Q${mx},${my} ${cx},${cy}`,fill:'none',stroke:col,'stroke-width':act?3:1.6,'stroke-dasharray':act?'none':'5 5',opacity:act?.9:.55},c0);
    edges.push({path,act,col,n})});
  const centre=el('g',{class:'geo'},svg);
  el('ellipse',{cx,cy,rx:112,ry:50,fill:'#1f1f1d',stroke:P.ink,'stroke-width':2},centre);
  T$(0,{x:cx,y:cy-6,'text-anchor':'middle','font-size':15,fill:P.ink,'font-weight':650},'Рабочий стек',centre);T$(0,{x:cx,y:cy+14,'text-anchor':'middle','font-size':12.5,fill:P.ink2},'HAR · V3 · H3',centre);
  const gs=N.map((n,i)=>{const g=el('g',{class:'geo',style:'cursor:pointer',tabindex:0,role:'button','aria-label':n.label},svg);
    el('rect',{x:n.x-92,y:n.y-28,width:184,height:56,rx:10,fill:'#1d1d1b',stroke:ST[n.status],'stroke-width':2},g);
    T$(0,{x:n.x-80,y:n.y-6,'font-size':13,fill:P.ink,'font-weight':600},n.label,g);T$(0,{x:n.x-80,y:n.y+14,'font-size':11,fill:P.muted},n.sub,g);
    T$(0,{x:n.x+78,y:n.y-6,'text-anchor':'end','font-size':14,fill:ST[n.status]},SYM[n.status],g);
    const pick=()=>{gs.forEach(q=>q.g.querySelector('rect').setAttribute('stroke-width',2));g.querySelector('rect').setAttribute('stroke-width',4);
      card.innerHTML=`<div class="tagrow"><span class="chip st-${n.status}">${SYM[n.status]} ${D.stat[n.status]}</span></div><h3>${n.label}</h3><div class="kind">${n.sub}</div><p>${n.card}</p><p class="nx"><b>Дальше.</b> ${n.nxt}</p>`+(n.link?`<p class="note"><a href="#${n.link}" style="color:var(--gold)">перейти к разделу →</a></p>`:'')};
    g.addEventListener('click',pick);g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();pick()}});
    return {g,pick,n}});
  const pulses=edges.filter(e=>e.act).map(e=>({e,dot:el('circle',{r:4.5,fill:e.col,opacity:0},svg),len:e.path.getTotalLength()}));
  edges.forEach(e=>drawable(e.path));
  const start=()=>{centre.classList.add('in');
    run(900,p=>edges.forEach(e=>setDraw(e.path,p)));
    gs.forEach((q,i)=>setTimeout(()=>q.g.classList.add('in'),REDUCED?0:(150+i*110)/SPEED));
    if(!REDUCED){const vis=visibleFlag(svg);let t0=performance.now();(function f(now){if(vis.v)pulses.forEach((p,i)=>{const u=(((now-t0)/3200+i*.23)%1);const pt=p.e.path.getPointAtLength(p.len*(1-u));p.dot.setAttribute('cx',pt.x);p.dot.setAttribute('cy',pt.y);p.dot.setAttribute('opacity',(.9*Math.sin(Math.PI*u)).toFixed(2))});requestAnimationFrame(f)})(t0)}};
  if(REDUCED){edges.forEach(e=>setDraw(e.path,1));centre.classList.add('in');gs.forEach(q=>q.g.classList.add('in'))}else once(svg,start,.25);
  (gs.find(q=>q.n.id==='gbm')||gs[0]).pick();
  document.querySelector('.replay[data-for="map"]').addEventListener('click',()=>{gs.forEach(q=>q.g.classList.remove('in'));centre.classList.remove('in');edges.forEach(e=>{e.path.style.strokeDashoffset='1'});setTimeout(start,50)});
}

// ---------------------------------------------------------------- section 3: the storyline
function storyline(){
  const S=D.story,svg=document.getElementById('stepSvg'),note=document.getElementById('stepNote'),btns=[...document.querySelectorAll('#flow .step')];let cur=0,timer=null,auto=true;
  const fm=(v,t)=>t==='A'?fA(v):fmt(v*100,2,true)+'%',val=(r,t,k)=>t==='A'?r[k]:r[k]*100;
  function pan(svg,x,r,t,title,c2b){const f=t==='A'?(v=>v):(v=>v*100);
    const groups=['look1','look2'].map((lk,k)=>{const d=r[lk][t],bars=[{v:f(d.view),lo:f(d.lo),hi:f(d.hi),color:COL.view},{v:f(d.c1),color:COL.c1},{v:f(d.c2a),color:COL.c2a}];if(c2b&&d.c2b!=null)bars.push({v:f(d.c2b),color:COL.c2b});
      return {label:'взгляд '+(k+1),note:d.pass?'✓ прошёл':'',noteColor:P.good,bars}});
    return panel(svg,{x,y:0,w:480,h:320,title,groups,ref:t==='A'?{v:0.005,label:'порог +0,005'}:{v:1,label:'порог +1%'},fmt:t==='A'?fA:(v=>fmt(v,2,true)+'%')})}
  function show(i){cur=i;btns.forEach((b,k)=>b.classList.toggle('on',k===i));svg.innerHTML='';const fs=[];
    if(i===0){const r=S.m6.M6;fs.push(pan(svg,20,r,'A','M6: аналоги, k = 50 — цель A (фиолет — вид, серый — C1, тан — C2a, розовый — C2b)',true),pan(svg,570,r,'B','M6: цель B',false));
      note.innerHTML=`<b>Подсказка.</b> Среднее остатков HAR у 50 прошлых окон-аналогов: A ${fA(r.look1.A.view)} / ${fA(r.look2.A.view)}, выше C1 и C2a; второй контроль M6 (${fA(r.look1.A.c2b)} / ${fA(r.look2.A.c2b)}) выше, поэтому предрегистрация не пройдена. Шаг M6b (остатки HAR у соседей) проходит цель A на обоих взглядах: ${fA(S.m6.M6b_k50.look1.A.view)} / ${fA(S.m6.M6b_k50.look2.A.view)}. Вывод: в режимных ключах есть нелинейная зависимость.`}
    else if(i===1){const r=S.m7;fs.push(pan(svg,20,r,'A','M7: GBM на 12 режимных ключах — цель A (розовый — ядро Надарая–Ватсона, C2b)',true),pan(svg,570,r,'B','M7: цель B',false));
      note.innerHTML=`<b>Проходит обе цели на обоих взглядах.</b> A ${fA(r.look1.A.view)} / ${fA(r.look2.A.view)}, B ${fB(r.look1.B.view)} / ${fB(r.look2.B.view)}; выше C1, C2a и ядерного контроля. 12 ключей: перцентили волатильности, час и выходной, |растяжение|, расстояния до S/R, лог-RV последних часов.`}
    else if(i===2){const a=S.abl.A,b=S.abl.B,g=(L,k)=>L.find(x=>x.g===k).share;
      const rA=ring(svg,170,140,82,g(a,'hourwk'),P.gold,Math.round(g(a,'hourwk')*100)+'%','час × выходной: доля выигрыша A');
      const rB=ring(svg,470,140,82,g(b,'hourwk'),P.pink,Math.round(g(b,'hourwk')*100)+'%','час × выходной: доля выигрыша B');
      const rC=ring(svg,770,140,82,g(a,'stretch'),P.violet,Math.round(g(a,'stretch')*100)+'%','|растяжение|: доля выигрыша A');fs.push(rA,rB,rC);
      note.innerHTML=`<b>Кто несёт.</b> Без «час × выходной» цель B теряет ${Math.round(g(b,'hourwk')*100)}% выигрыша, цель A — ${Math.round(g(a,'hourwk')*100)}%; без |растяжения| цель A теряет ${Math.round(g(a,'stretch')*100)}%. Остальные группы (перцентиль волатильности, лог-RV, S/R) не несут. Дешёвая линейная замена (восемь простых членов) вернула по A ${S.m8.R_lin_A.toFixed(2)}, по B ${S.m8.R_lin_B.toFixed(2)}: нужна форма суточного профиля.`}
    else{const h=S.rec.find(r=>r.key==='M9h6_prof_str'),h1=S.rec.find(r=>r.key==='M7c_gbm_resid');
      fs.push(ring(svg,230,140,82,Math.min(h.RA,1),P.gold,h.RA.toFixed(2),'M9h6: доля выигрыша M7 по A (метка = 1,0)',1),ring(svg,570,140,82,h.RB,P.pink,h.RB.toFixed(2),'M9h6: доля выигрыша M7 по B (метка = 1,0)',1),ring(svg,910,140,82,h1.RB,P.violet,h1.RB.toFixed(2),'M7c: доля M7 по B (по A '+h1.RA.toFixed(2)+')',1));
      note.innerHTML=`<b>Компактный модуль.</b> Профиль «час × выходной» с гармониками плюс |растяжение|, всего 29 столбцов в ridge и логистике, возвращает ${h.RA.toFixed(2)} выигрыша M7 по A и ${h.RB.toFixed(2)} по B. Это кандидат после факта: заранее зарегистрированные виды M9 не прошли оба взгляда (мало данных в первые месяцы Binance); судья — проверка вперёд.`}
    run(1000,p=>fs.forEach(f=>f(p)))}
  btns.forEach((b,k)=>b.addEventListener('click',()=>{auto=false;clearTimeout(timer);show(k)}));
  document.querySelector('.replay[data-for="step"]').addEventListener('click',()=>{auto=true;show(cur)});
  const loop=()=>{timer=setTimeout(()=>{if(!auto)return;show((cur+1)%4);loop()},6500/SPEED)};
  show(0);if(!REDUCED)once(svg,()=>{show(0);loop()},.35);
}

// ---------------------------------------------------------------- ablation bars
function ablation(){
  const svg=document.getElementById('ablSvg'),seg=document.querySelectorAll('#ablSeg button');
  function draw(t){const rows=D.story.abl[t].map(g=>({label:g.name,sym:g.carries?'works':null,symTxt:'',bars:[{v:g.share,color:g.carries?P.gold:P.neutral}]}));
    hbars(svg,rows,{W:540,L:230,R:60,top:44,rowH:50,barH:22,min:-.2,max:1,fmt:v=>fmt(v*100,0,false)+'%',fs:12,ref:[{v:.25,label:'25%'}]});
    D.story.abl[t].forEach((g,i)=>{el('text',{x:236,y:44+i*50+49,'font-size':10,fill:P.muted},svg,`потеря ${t==='A'?fA(g.drop):fB(g.drop)} [${t==='A'?fA(g.lo):fB(g.lo)}; ${t==='A'?fA(g.hi):fB(g.hi)}]${g.carries?' · несёт':''}`)})}
  let first=true;once(svg,()=>draw('A'),.3);seg.forEach(b=>b.addEventListener('click',()=>{seg.forEach(x=>x.classList.remove('on'));b.classList.add('on');draw(b.dataset.t)}))}

// ---------------------------------------------------------------- polar clock
function clock(){
  const svg=document.getElementById('clockSvg'),read=document.getElementById('clockRead'),seg=document.querySelectorAll('#clkSeg button'),Pf=D.story.prof,cx=280,cy=280;let tgt='B',hand=null,hover=null,drawn=null;
  const cfg=t=>t==='B'?{lo:0,hi:.9,rings:[.2,.4,.6,.8],f:v=>v.toFixed(2).replace('.',','),title:'P(касания)'}:{lo:-.5,hi:.7,rings:[-.4,-.2,0,.2,.4,.6],f:v=>fmt(v,3,true),title:'изменение log RV'};
  const ang=h=>h/24*2*Math.PI-Math.PI/2;
  function build(){svg.innerHTML='';const c=cfg(tgt),R0=36,R1=232,rr=v=>R0+(R1-R0)*(v-c.lo)/(c.hi-c.lo);
    c.rings.forEach(v=>{el('circle',{cx,cy,r:rr(v),fill:'none',stroke:v===0&&tgt==='A'?P.muted:P.grid,'stroke-dasharray':v===0&&tgt==='A'?'4 3':'none'},svg);T$(0,{x:cx+3,y:cy-rr(v)-2,'font-size':9.5,fill:P.muted},c.f(v),svg)});
    for(let h=0;h<24;h++){const a=ang(h);el('line',{x1:cx+R0*Math.cos(a),y1:cy+R0*Math.sin(a),x2:cx+R1*Math.cos(a),y2:cy+R1*Math.sin(a),stroke:P.grid,'stroke-width':h%3===0?1.2:.5},svg);
      if(h%3===0)T$(0,{x:cx+(R1+16)*Math.cos(a),y:cy+(R1+16)*Math.sin(a)+4,'text-anchor':'middle','font-size':12,fill:P.ink2},String(h).padStart(2,'0'),svg)}
    [13,14].forEach(h=>{const a=ang(h);el('line',{x1:cx+R0*Math.cos(a),y1:cy+R0*Math.sin(a),x2:cx+(R1+6)*Math.cos(a),y2:cy+(R1+6)*Math.sin(a),stroke:P.ink,'stroke-dasharray':'2 3','stroke-width':1.2,opacity:.8},svg)});
    T$(0,{x:cx+(R1+4)*Math.cos(ang(13.5)),y:cy+(R1+4)*Math.sin(ang(13.5))-18,'text-anchor':'middle','font-size':10.5,fill:P.muted},'открытие США',svg);
    const key=tgt==='B'?['B_view','B_base','B_real']:['A_view','A_base','A_real'],ws=[['weekday',P.gold],['weekend',P.pink]],layers=[];
    ws.forEach(([w,col])=>{
      const base=el('path',{fill:'none',stroke:col,'stroke-width':1.2,'stroke-dasharray':'4 3',opacity:.6},svg),poly=el('path',{fill:col,'fill-opacity':.14,stroke:col,'stroke-width':2.6,'stroke-linejoin':'round'},svg),
        dots=Pf.hours.map(h=>el('circle',{r:3.2,fill:col,stroke:P.paper,'stroke-width':1,opacity:0},svg));
      layers.push({w,col,base,poly,dots,view:Pf[key[0]][w],bs:Pf[key[1]][w],real:Pf[key[2]][w]})});
    const hd=el('line',{x1:cx,y1:cy,x2:cx,y2:cy-R1,stroke:P.ink,'stroke-width':1.6,opacity:0},svg);hand=hd;
    const pt=(v,h,p)=>{const r=R0+(rr(v)-R0)*p,a=ang(h);return [cx+r*Math.cos(a),cy+r*Math.sin(a)]},closed=(vals,p)=>'M'+vals.map((v,h)=>pt(v,h,p).map(q=>q.toFixed(1)).join(',')).join('L')+'Z';
    drawn=p=>{layers.forEach(L=>{L.poly.setAttribute('d',closed(L.view,p));L.base.setAttribute('d',closed(L.bs,p));L.dots.forEach((d,h)=>{const q=pt(L.real[h],h,p);d.setAttribute('cx',q[0]);d.setAttribute('cy',q[1]);d.setAttribute('opacity',p>.85?1:0)})})};
    layers.hd=hd;svg._layers=layers;readout(14.5);run(1400,drawn);drawn(REDUCED?1:0)}
  function readout(hf){const h=Math.floor(hf)%24,c=cfg(tgt),L=svg._layers;if(!L)return;
    const k=tgt==='B'?'':'',g=w=>L.find(x=>x.w===w);
    read.innerHTML=`<b>${String(h).padStart(2,'0')}:00 UTC</b> · будни: подогнано <b>${c.f(g('weekday').view[h])}</b>, факт ${c.f(g('weekday').real[h])}, база без модуля ${c.f(g('weekday').bs[h])} (n = ${Pf.n_by_hour.weekday[h]}) · выходные: подогнано <b>${c.f(g('weekend').view[h])}</b>, факт ${c.f(g('weekend').real[h])}, база ${c.f(g('weekend').bs[h])} (n = ${Pf.n_by_hour.weekend[h]})`;
    if(hand){const a=ang(hf);hand.setAttribute('x2',cx+232*Math.cos(a));hand.setAttribute('y2',cy+232*Math.sin(a));hand.setAttribute('opacity',.85)}}
  svg.addEventListener('mousemove',e=>{const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*560-cx,y=(e.clientY-r.top)/r.height*560-cy;hover=((Math.atan2(y,x)+Math.PI/2)/(2*Math.PI)*24+24)%24;readout(hover)});
  svg.addEventListener('mouseleave',()=>{hover=null});
  seg.forEach(b=>b.addEventListener('click',()=>{seg.forEach(x=>x.classList.remove('on'));b.classList.add('on');tgt=b.dataset.t;build()}));
  once(svg,()=>{build();if(!REDUCED){const vis=visibleFlag(svg);let hf=14.5,last=performance.now();(function f(now){const dt=(now-last)/1000;last=now;if(vis.v&&hover==null){hf=(hf+dt*24/30*SPEED)%24;readout(hf)}requestAnimationFrame(f)})(last)}},.3);
}

// ---------------------------------------------------------------- recovery
function recovery(){
  const svg=document.getElementById('recSvg');
  once(svg,()=>{const rows=D.story.rec.map(r=>({label:r.name,sym:null,bars:[{v:r.RA,color:P.gold},{v:r.RB,color:P.pink}]}));
    hbars(svg,rows,{W:1100,L:290,R:80,top:46,rowH:40,barH:14,min:-.3,max:1.8,fmt:v=>fmt(v,2,false),band:1,ref:[{v:.4,label:'частично'},{v:.75,label:'восстанавливает'},{v:1,label:'= M7'}]});
    D.story.rec.forEach((r,i)=>{const y=46+i*40+18;el('text',{x:1100-78,y,'font-size':10.5,fill:P.muted},svg,(r.passA?'A ✓ ':'')+(r.passB?'B ✓':''))});
    el('text',{x:1100-78,y:34,'font-size':10.5,fill:P.muted},svg,'оба взгляда');
    el('rect',{x:300,y:8,width:10,height:10,fill:P.gold},svg);T$(0,{x:316,y:17,'font-size':11,fill:P.ink2},'по цели A',svg);el('rect',{x:400,y:8,width:10,height:10,fill:P.pink},svg);T$(0,{x:416,y:17,'font-size':11,fill:P.ink2},'по цели B',svg)},.3)}

// ---------------------------------------------------------------- controls
function controls(){
  const svg=document.getElementById('ctlSvg'),seg=document.querySelectorAll('#ctlSeg button');
  function draw(t){const rows=[];D.control.forEach(c=>['look1','look2'].forEach((lk,k)=>{const d=c[lk][t],f=t==='A'?1:100;rows.push({label:k===0?c.name:'',sub:'',bars:[{v:d.view*f,color:P.violet},{v:d.c2a*f,color:P.tan}]})}));
    const all=rows.flatMap(r=>r.bars.map(b=>b.v)).concat(t==='A'?[0.005]:[1]);
    hbars(svg,rows,{W:1100,L:220,R:100,top:40,rowH:22,barH:8,min:Math.min(0,...all)*1.25,max:Math.max(...all)*1.15,fmt:t==='A'?fA:(v=>fmt(v,2,true)+'%'),fs:12,ref:[{v:t==='A'?0.005:1,label:t==='A'?'порог +0,005':'порог +1%'}]})}
  once(svg,()=>draw('A'),.25);seg.forEach(b=>b.addEventListener('click',()=>{seg.forEach(x=>x.classList.remove('on'));b.classList.add('on');draw(b.dataset.t)}));
  document.querySelector('.replay[data-for="ctl"]').addEventListener('click',()=>draw(document.querySelector('#ctlSeg .on').dataset.t))}

// ---------------------------------------------------------------- timeline (the positive report's, with the sprint's NEW rows)
function timeline(){
  const tl=document.getElementById('tl'),pc=s=>{const [Y,M,Dd]=s.split('-').map(Number),mi=(Y-2026)*12+(M-1)-9,dim=new Date(Y,M,0).getDate();return (mi+(Dd-1)/dim)/5*100};
  D.timeline.forEach(r=>{const row=document.createElement('div');row.className='row'+(r.group==='new'?' new':'');
    row.innerHTML=`<div class="lbl"><b>${r.id}</b>${r.group==='new'?'<span class="newchip">NEW</span>':''}<span class="sha" title="${r.src}">sha256 ${r.sha}…</span><span>${r.what} · ${r.crit}</span></div><div class="track"><div class="bar" style="left:${pc(r.a)}%;--w:${pc(r.b)-pc(r.a)}%"></div><div class="dia" style="left:${pc(r.b)}%"></div></div>`;tl.appendChild(row)});
  const ax=document.createElement('div');ax.className='axis';ax.innerHTML='<div></div><div class="months"><span>окт 2026</span><span>ноя</span><span>дек</span><span>янв 2027</span><span>фев</span></div>';tl.appendChild(ax);
  const today=document.createElement('div');today.className='today';today.innerHTML='<span>сегодня</span>';tl.appendChild(today);
  function place(){const tr=tl.querySelector('.track'),r=tr.getBoundingClientRect(),b=tl.getBoundingClientRect();today.style.left=(r.left-b.left+r.width*pc('2026-10-17')/100)+'px'}place();addEventListener('resize',place);
  const bars=[...tl.querySelectorAll('.bar')];
  if(REDUCED){bars.forEach(b=>b.style.width=getComputedStyle(b).getPropertyValue('--w'));tl.classList.add('go');return}
  once(tl,()=>{bars.forEach((b,i)=>setTimeout(()=>{b.style.transition='width 1.1s cubic-bezier(.2,.8,.2,1)';b.style.width=b.style.getPropertyValue('--w')},i*110/SPEED));setTimeout(()=>tl.classList.add('go'),(bars.length*110+900)/SPEED);place()},.2)}

counters();heroBars();cardCharts();conceptMap();storyline();ablation();clock();recovery();controls();timeline();
document.querySelector('.replay[data-for="hero"]').addEventListener('click',heroBars);
