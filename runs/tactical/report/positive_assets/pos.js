(()=>{
const D=JSON.parse(document.getElementById('data').textContent),P=D.pal,NS='http://www.w3.org/2000/svg';
const qs=new URLSearchParams(location.search);
const REDUCED=matchMedia('(prefers-reduced-motion: reduce)').matches||qs.has('static');
if(qs.has('static'))document.body.classList.add('static');
const SPEED=parseFloat(qs.get('speed')||'1');
function el(tag,a,parent,txt){const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);if(txt!=null)e.textContent=txt;if(parent)parent.appendChild(e);return e}
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const ease=t=>t<.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2, easeOut=t=>1-Math.pow(1-t,3);
const fmt=(v,d,sign)=>{const s=Math.abs(v).toFixed(d).replace('.',',');return (v<0?'−':(sign?'+':''))+s};
const mix=(a,b,t)=>a+(b-a)*t;
function hex(c){return [1,3,5].map(i=>parseInt(c.slice(i,i+2),16))}
function lerpColor(stops,t){ // stops: [[pos,'#rrggbb'],...]
  if(t<=stops[0][0])return stops[0][1];if(t>=stops[stops.length-1][0])return stops[stops.length-1][1];
  for(let i=1;i<stops.length;i++)if(t<=stops[i][0]){const u=(t-stops[i-1][0])/(stops[i][0]-stops[i-1][0]),a=hex(stops[i-1][1]),b=hex(stops[i][1]);
    return 'rgb('+a.map((x,k)=>Math.round(mix(x,b[k],u))).join(',')+')'}}
// a timed run: step(p) with p in 0..1; resolves when done. REDUCED: jumps to the end.
function run(dur,step,delay){return new Promise(res=>{
  if(REDUCED){step(1);res();return}
  setTimeout(()=>{const t0=performance.now(),d=dur/SPEED;(function f(n){const p=clamp((n-t0)/d,0,1);step(p);p<1?requestAnimationFrame(f):res()})(t0)},(delay||0)/SPEED)})}
function once(node,fn,th){
  if(REDUCED||!('IntersectionObserver' in window)){fn();return}
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){io.disconnect();fn()}})},{threshold:th==null?.3:th});io.observe(node)}
function visibleFlag(node){const o={v:false};if(!('IntersectionObserver' in window)){o.v=true;return o}
  new IntersectionObserver(es=>es.forEach(e=>o.v=e.isIntersecting),{threshold:.05}).observe(node);return o}
function pathLine(pts){return 'M'+pts.map(p=>p[0].toFixed(1)+','+p[1].toFixed(1)).join('L')}
function drawable(path){path.setAttribute('pathLength','1');path.style.strokeDasharray='1';path.style.strokeDashoffset='1'}
function setDraw(path,p){path.style.strokeDashoffset=String(1-p)}

// ---------------------------------------------------------------- nav highlight
(function(){const links=[...document.querySelectorAll('nav a')],secs=links.map(a=>document.querySelector(a.getAttribute('href')));
  function f(){let cur=0;secs.forEach((s,i)=>{if(s&&s.getBoundingClientRect().top<160)cur=i});links.forEach((a,i)=>a.classList.toggle('on',i===cur))}
  addEventListener('scroll',f,{passive:true});f()})();

// ---------------------------------------------------------------- counters
function counters(){
  document.querySelectorAll('.cnt').forEach(c=>{
    const to=parseFloat(c.dataset.to),dec=+c.dataset.dec,sign=!!c.dataset.sign,suf=c.dataset.suf||'',final=fmt(to,dec,sign)+suf;
    c.textContent=final;if(REDUCED)return;
    c.textContent=fmt(0,dec,false)+suf;
    once(c,()=>{run(1500,p=>{c.textContent=(p>=1?final:fmt(to*easeOut(p),dec,sign)+suf)},Math.random()*250)},.4)})}

// ---------------------------------------------------------------- sparklines (decay, T01, O11)
function spark(id,vals,labels,color,ymin,ymax,fmtLabel){
  const s=document.getElementById(id);if(!s)return;const W=320,H=+s.getAttribute('viewBox').split(' ')[3],L=14,R=14,T=8,B=labels?18:10;
  const n=vals.length,x=i=>L+(W-L-R)*i/(n-1),lo=ymin==null?Math.min(...vals):ymin,hi=ymax==null?Math.max(...vals):ymax,y=v=>T+(H-T-B)*(1-(v-lo)/(hi-lo));
  const pts=vals.map((v,i)=>[x(i),y(v)]);
  if(lo<=0&&hi>=0)el('line',{x1:L,x2:W-R,y1:y(0),y2:y(0),stroke:P.axis,'stroke-dasharray':'3 3'},s);
  const area=el('path',{d:pathLine(pts)+'L'+x(n-1)+','+(H-B)+'L'+x(0)+','+(H-B)+'Z',fill:color,'fill-opacity':0},s);
  const ln=el('path',{d:pathLine(pts),fill:'none',stroke:color,'stroke-width':2.2,'stroke-linejoin':'round'},s);drawable(ln);
  const dots=pts.map((p,i)=>el('circle',{cx:p[0],cy:p[1],r:3,fill:P.paper,stroke:color,'stroke-width':1.6,opacity:0},s));
  if(labels)labels.forEach((t,i)=>{if(t!=null)el('text',{x:x(i),y:H-3,'text-anchor':i===0?'start':(i===n-1?'end':'middle'),'font-size':9.5,fill:P.muted},s,t)});
  once(s,()=>run(1400,p=>{setDraw(ln,p);area.setAttribute('fill-opacity',(.16*p).toFixed(3));dots.forEach((d,i)=>d.setAttribute('opacity',p>i/(n-1)?1:0))}),.4)}

// ---------------------------------------------------------------- HERO
function hero(){
  const h=D.hero,svg=document.getElementById('heroSvg');svg.innerHTML='';
  const W=1200,H=470,L=66,R=16,Tp=20,B=40,n=h.c.length,N=h.nodes;
  let lo=Math.min(...h.c,...N.map(q=>q.e40)),hi=Math.max(...h.c,...N.map(q=>q.e40));const pad=(hi-lo)*.06;lo-=pad;hi+=pad;
  const x=i=>L+(W-L-R)*i/(n-1),y=v=>Tp+(H-Tp-B)*(1-(v-lo)/(hi-lo));
  for(let k=0;k<=5;k++){const v=lo+(hi-lo)*k/5;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},svg);el('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':11,fill:P.muted},svg,Math.round(v).toLocaleString('ru-RU'))}
  for(let hr=0;hr<=24;hr+=3){const xx=x(Math.min(hr*60,n-1));el('line',{x1:xx,x2:xx,y1:Tp,y2:H-B,stroke:P.grid,'stroke-dasharray':'2 4'},svg);el('text',{x:xx,y:H-B+18,'text-anchor':'middle','font-size':11,fill:P.muted},svg,('0'+hr).slice(-2)+':00')}
  el('text',{x:L,y:H-6,'font-size':11,fill:P.muted},svg,'UTC, 2025-11-15 · цена в USDT');
  const priceP=el('path',{d:pathLine(h.c.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.ink,'stroke-width':1.5,'stroke-linejoin':'round'},svg);
  const e20=el('path',{d:pathLine(h.e20.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.gold,'stroke-width':1.3,opacity:.9},svg);
  const e60=el('path',{d:pathLine(h.e60.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.pink,'stroke-width':1.3,opacity:.9},svg);
  const q20=el('path',{d:pathLine(N.map(q=>[x(q.m),y(q.e20)])),fill:'none',stroke:P.violet,'stroke-width':3.2,'stroke-linecap':'round','stroke-linejoin':'round'},svg);
  const q40=el('path',{d:pathLine(N.map(q=>[x(q.m),y(q.e40)])),fill:'none',stroke:P.tan,'stroke-width':3.2,'stroke-linecap':'round','stroke-linejoin':'round'},svg);
  [priceP,e20,e60,q20,q40].forEach(drawable);
  const head=el('circle',{cx:x(0),cy:y(h.c[0]),r:5,fill:P.ink,opacity:0},svg);
  // geometry marks at the hourly 15-minute nodes
  const maxd2=Math.max(...N.map(q=>Math.abs(q.d2))),dxbar=(W-L-R)*15/(n-1),marks=[];
  N.forEach((q,j)=>{ if(j<6||j%4!==3)return;
    const cx=x(q.m),cy=y(q.e20),ux=3*dxbar,uy=y(q.e20)-y(q.e20-q.d3),ul=Math.hypot(ux,uy),tx=ux/ul,ty=uy/ul;
    const g=el('g',{class:'geo'},svg);
    const col=q.d3>=0?P.up:P.down;
    el('line',{x1:cx-tx*22,y1:cy-ty*22,x2:cx+tx*22,y2:cy+ty*22,stroke:col,'stroke-width':3.4,'stroke-linecap':'round'},g);
    const ax=cx+tx*22,ay=cy+ty*22,nx=-ty,ny=tx;
    el('polygon',{points:`${ax+tx*10},${ay+ty*10} ${ax+nx*6},${ay+ny*6} ${ax-nx*6},${ay-ny*6}`,fill:col},g);
    const b=clamp(q.d2/maxd2,-1,1)*30;
    el('path',{d:`M${cx-tx*40},${cy-ty*40}Q${cx},${cy+b*2} ${cx+tx*40},${cy+ty*40}`,fill:'none',stroke:q.d2>=0?P.pink:P.tan,'stroke-width':2,class:'arcpulse',style:'animation-delay:'+(j%7*.35)+'s'},g);
    el('circle',{cx,cy,r:3,fill:P.ink},g);marks.push(g)});
  if(REDUCED){[priceP,e20,e60,q20,q40].forEach(p=>setDraw(p,1));marks.forEach(g=>g.classList.add('in'));return}
  [priceP,e20,e60,q20,q40].forEach(p=>setDraw(p,0));marks.forEach(g=>g.classList.remove('in'));
  head.setAttribute('opacity',1);
  run(3800,p=>{const q=ease(p);setDraw(priceP,q);const i=Math.min(n-1,Math.floor(q*(n-1)));head.setAttribute('cx',x(i));head.setAttribute('cy',y(h.c[i]));if(p>=1)head.setAttribute('opacity',0)},200);
  run(3600,p=>setDraw(e20,ease(p)),700);run(3600,p=>setDraw(e60,ease(p)),900);
  run(2000,p=>setDraw(q20,ease(p)),3900);run(2000,p=>setDraw(q40,ease(p)),4300);
  marks.forEach((g,i)=>setTimeout(()=>g.classList.add('in'),(6100+i*110)/SPEED));
}

// ---------------------------------------------------------------- VOLATILITY CLOCK
function clock(){
  const svg=document.getElementById('clockSvg'),C=D.clock,cx=320,cy=320,R0=118,LS=150,N=96;
  const cols=[[.7,'#4b4a46'],[.95,P.violet],[1.2,P.gold],[1.55,P.pink]];
  function rel(prof){return C[prof].slot96}
  for(let s=0;s<N;s++){const a=(s/N)*2*Math.PI-Math.PI/2,major=s%4===0;
    el('line',{x1:cx+Math.cos(a)*(R0-5),y1:cy+Math.sin(a)*(R0-5),x2:cx+Math.cos(a)*(R0-(major?15:9)),y2:cy+Math.sin(a)*(R0-(major?15:9)),stroke:major?P.ink2:P.muted,'stroke-width':major?1.6:1},svg)}
  for(let hr=0;hr<24;hr+=3){const a=(hr/24)*2*Math.PI-Math.PI/2;el('text',{x:cx+Math.cos(a)*(R0-32),y:cy+Math.sin(a)*(R0-32)+4,'text-anchor':'middle','font-size':12,fill:P.ink2},svg,('0'+hr).slice(-2))}
  el('circle',{cx,cy,r:R0-2,fill:'none',stroke:P.axis},svg);
  const bars=[];for(let s=0;s<N;s++){const a=((s+.5)/N)*2*Math.PI-Math.PI/2;
    bars.push(el('line',{x1:cx+Math.cos(a)*R0,y1:cy+Math.sin(a)*R0,x2:cx+Math.cos(a)*R0,y2:cy+Math.sin(a)*R0,'stroke-width':5.2,'stroke-linecap':'round'},svg))}
  // pulse sector UTC 14:00-15:00 (slots 56..59)
  const aS=(14/24)*2*Math.PI-Math.PI/2,aE=(15/24)*2*Math.PI-Math.PI/2,pulse=[];
  for(let k=0;k<3;k++){const r=R0+LS+16;pulse.push(el('path',{d:`M${cx+Math.cos(aS)*r},${cy+Math.sin(aS)*r}A${r},${r} 0 0 1 ${cx+Math.cos(aE)*r},${cy+Math.sin(aE)*r}`,fill:'none',stroke:P.pink,'stroke-width':7,'stroke-linecap':'round',opacity:0},svg))}
  const lblA=(14.5/24)*2*Math.PI-Math.PI/2,lbl=el('text',{x:cx+Math.cos(lblA)*(R0+LS+40)-30,y:cy+Math.sin(lblA)*(R0+LS+40)+10,'text-anchor':'middle','font-size':12,fill:P.pink,opacity:0},svg,'UTC 14 · открытие США');
  const hand=el('line',{x1:cx,y1:cy,x2:cx,y2:cy-R0-LS-6,stroke:P.ink,'stroke-width':1.4,opacity:.8},svg);
  el('circle',{cx,cy,r:4,fill:P.ink},svg);
  const tTxt=el('text',{x:cx,y:cy-14,'text-anchor':'middle','font-size':30,fill:P.ink,'font-weight':600},svg,'00:00');
  const vTxt=el('text',{x:cx,y:cy+14,'text-anchor':'middle','font-size':14,fill:P.ink2},svg,'');
  const pTxt=el('text',{x:cx,y:cy+34,'text-anchor':'middle','font-size':11,fill:P.muted},svg,'слот · к среднему дня');
  let cur=C.before.slot96.slice(),prof='before',vis=visibleFlag(svg),ang=0,hover=null,last=performance.now(),morph=null;
  function paint(vals){vals.forEach((v,s)=>{const a=((s+.5)/N)*2*Math.PI-Math.PI/2,len=Math.max(4,(v-.55)*LS),c=cols;
    bars[s].setAttribute('x2',cx+Math.cos(a)*(R0+len));bars[s].setAttribute('y2',cy+Math.sin(a)*(R0+len));bars[s].setAttribute('stroke',lerpColor(c,v))})}
  paint(cur);
  function stats(){const k=C[prof],a=k.slot96,h=k.hour,mn=Math.min(...a),mx=Math.max(...a),hn=Math.min(...h),hx=Math.max(...h);
    const name={before:'до 2021-11',after:'после 2021-11',binance:'Binance 2025-10..2026-09'}[prof];
    document.getElementById('clockStats').innerHTML=`<b>${name}</b> · дней ${k.n_days}<br>по часам: от ${fmt(hn,3)} до ${fmt(hx,3)} · пик в UTC ${h.indexOf(hx)}:00 (×${fmt(hx,3)})<br>по 15-минутным слотам: от ${fmt(mn,3)} до ${fmt(mx,3)} · пик в ${('0'+Math.floor(a.indexOf(mx)/4)).slice(-2)}:${('0'+(a.indexOf(mx)%4)*15).slice(-2)} UTC`}
  stats();
  function setProf(p){ if(p===prof&&!morph)return; const from=cur.slice(),to=C[p].slot96;prof=p;stats();
    document.querySelectorAll('#clockBtns button').forEach(b=>b.classList.toggle('on',b.dataset.prof===p));
    const showP=(p!=='before');
    if(REDUCED){cur=to.slice();paint(cur);pulse.forEach(q=>q.setAttribute('opacity',showP?.9:0));lbl.setAttribute('opacity',showP?1:0);return}
    morph={t0:performance.now(),from,to,dur:1800/SPEED};lbl.setAttribute('opacity',showP?1:0)}
  document.querySelectorAll('#clockBtns button').forEach(b=>b.onclick=()=>setProf(b.dataset.prof));
  svg.addEventListener('pointermove',e=>{const r=svg.getBoundingClientRect(),dx=(e.clientX-r.left)/r.width*640-cx,dy=(e.clientY-r.top)/r.height*640-cy;hover=Math.atan2(dy,dx)+Math.PI/2;if(hover<0)hover+=2*Math.PI});
  svg.addEventListener('pointerleave',()=>hover=null);
  function readout(a){const sl=Math.floor((a/(2*Math.PI))*96)%96,hh=Math.floor(sl/4),mm=(sl%4)*15;tTxt.textContent=('0'+hh).slice(-2)+':'+('0'+mm).slice(-2);vTxt.textContent='×'+fmt(cur[sl],3);}
  function frame(now){const dt=Math.min(.1,(now-last)/1000);last=now;
    if(vis.v||REDUCED){ if(hover==null)ang=(ang+dt*(2*Math.PI/30)*SPEED)%(2*Math.PI);const a=hover!=null?hover:ang;
      if(morph){const p=clamp((now-morph.t0)/morph.dur,0,1),q=ease(p);cur=morph.from.map((v,i)=>mix(v,morph.to[i],q));paint(cur);if(p>=1)morph=null}
      hand.setAttribute('transform',`rotate(${a*180/Math.PI} ${cx} ${cy})`);readout(a);
      bars.forEach((b,s)=>{const sa=((s+.5)/N)*2*Math.PI;let d=Math.abs(sa-a);d=Math.min(d,2*Math.PI-d);b.setAttribute('opacity',(.55+.45*Math.exp(-d*d/.05)).toFixed(2))});
      if(prof!=='before'){const ph=(now/1000*SPEED)%2.4/2.4;pulse.forEach((q,k)=>{const f=(ph+k/3)%1;q.setAttribute('opacity',(.9*(1-f)).toFixed(2));q.setAttribute('transform',`translate(${cx} ${cy}) scale(${1+f*.07}) translate(${-cx} ${-cy})`)})}
      else pulse.forEach(q=>q.setAttribute('opacity',0));
    }
    requestAnimationFrame(frame)}
  if(REDUCED){setProf('after');readout(14.5/24*2*Math.PI);hand.setAttribute('transform',`rotate(${14.5/24*360} ${cx} ${cy})`);return}
  requestAnimationFrame(frame);
  once(svg,()=>{setTimeout(()=>setProf('after'),1200/SPEED)},.5);
}

// ---------------------------------------------------------------- GEOMETRY panels
function geomPanels(){
  const box=document.getElementById('geomPanels');
  const names={'g15.ema20.slope3':'наклон EMA 20 за 3 бара (15 мин)','g15.ema20.pos':'положение закрытия относительно EMA 20','g15.ema40.slope3':'наклон EMA 40 за 3 бара (15 мин)'};
  D.geom.forEach((g,gi)=>{
    const d=document.createElement('div');d.className='gp';
    d.innerHTML=`<h4>${g.feature}: ${names[g.feature]}</h4><div class="v">коэффициент ${fmt(g.coef,3,true)} · значение признака ${fmt(g.value,2,true)} · окно с ${g.anchor.slice(0,16)}</div>`;
    const svg=el('svg',{viewBox:'0 0 380 230'},d);box.appendChild(d);
    const W=380,H=230,L=8,R=44,Tp=14,B=14,n=g.c.length,all=[...g.c,...g.e20,...g.e40];let lo=Math.min(...all),hi=Math.max(...all);const pd=(hi-lo)*.1;lo-=pd;hi+=pd;
    const x=i=>L+(W-L-R)*i/(n-1),y=v=>Tp+(H-Tp-B)*(1-(v-lo)/(hi-lo));
    for(let k=0;k<4;k++)el('line',{x1:L,x2:W-R,y1:Tp+(H-Tp-B)*k/3,y2:Tp+(H-Tp-B)*k/3,stroke:P.grid},svg);
    const pc=el('path',{d:pathLine(g.c.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.ink,'stroke-width':1.6},svg),
      p20=el('path',{d:pathLine(g.e20.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.violet,'stroke-width':2.6},svg),
      p40=el('path',{d:pathLine(g.e40.map((v,i)=>[x(i),y(v)])),fill:'none',stroke:P.tan,'stroke-width':2.6},svg);
    [pc,p20,p40].forEach(drawable);
    // last-bar geometry: slope arrow over 3 bars, position bracket, curvature arc
    const j=n-1,x1=x(j),yE=y(g.e20[j]),yE3=y(g.e20[j-3]),x0=x(j-3),yC=y(g.c[j]),yE40=y(g.e40[j]),yE403=y(g.e40[j-3]);
    const grp=el('g',{class:'geo'},svg);
    const useE40=g.feature.indexOf('ema40')>=0,ya=useE40?yE40:yE,ya3=useE40?yE403:yE3,col=(ya3-ya)>=0?P.up:P.down;
    el('line',{x1:x0,y1:ya3,x2:x1,y2:ya,stroke:col,'stroke-width':4,'stroke-linecap':'round'},grp);
    const ang=Math.atan2(ya-ya3,x1-x0);el('polygon',{points:`${x1+Math.cos(ang)*12},${ya+Math.sin(ang)*12} ${x1+Math.cos(ang+2.5)*9},${ya+Math.sin(ang+2.5)*9} ${x1+Math.cos(ang-2.5)*9},${ya+Math.sin(ang-2.5)*9}`,fill:col},grp);
    if(g.feature.indexOf('pos')>=0){el('line',{x1:x1+14,x2:x1+14,y1:yC,y2:yE,stroke:P.gold,'stroke-width':3},grp);el('line',{x1:x1+9,x2:x1+19,y1:yC,y2:yC,stroke:P.gold,'stroke-width':2},grp);el('line',{x1:x1+9,x2:x1+19,y1:yE,y2:yE,stroke:P.gold,'stroke-width':2},grp)}
    const d2=(g.e20[j]-g.e20[j-3])-(g.e20[j-3]-g.e20[j-6]),mid=y(g.e20[j-3]);
    el('path',{d:`M${x(j-6)},${y(g.e20[j-6])}Q${x(j-3)},${mid+(d2>=0?14:-14)} ${x1},${yE}`,fill:'none',stroke:d2>=0?P.pink:P.tan,'stroke-width':2,'stroke-dasharray':'4 3',class:'arcpulse'},grp);
    el('text',{x:W-R+4,y:yE+4,'font-size':10.5,fill:P.violet},svg,'EMA 20');el('text',{x:W-R+4,y:yE40+(yE40>yE?14:-6),'font-size':10.5,fill:P.tan},svg,'EMA 40');
    if(REDUCED){[pc,p20,p40].forEach(p=>setDraw(p,1));grp.classList.add('in')}
    else once(d,()=>{run(1800,p=>setDraw(pc,ease(p)),gi*250);run(1800,p=>setDraw(p20,ease(p)),500+gi*250);run(1800,p=>setDraw(p40,ease(p)),800+gi*250);setTimeout(()=>grp.classList.add('in'),(2700+gi*250)/SPEED)},.4)});
}
function bars(id,months,vals,color,ymax,tickEvery){
  const s=document.getElementById(id);if(!s)return;const W=560,H=220,L=40,R=10,Tp=12,B=34,n=vals.length,bw=(W-L-R)/n,y=v=>Tp+(H-Tp-B)*(1-v/ymax);
  for(let k=0;k<=4;k++){const v=ymax*k/4;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},s);el('text',{x:L-5,y:y(v)+4,'text-anchor':'end','font-size':10,fill:P.muted},s,fmt(v,4))}
  const rs=vals.map((v,i)=>el('rect',{x:L+i*bw+1.2,y:y(0),width:Math.max(2,bw-2.4),height:0,rx:2,fill:color},s));
  months.forEach((m,i)=>{if(i%tickEvery===0)el('text',{x:L+i*bw+bw/2,y:H-B+14,'text-anchor':'middle','font-size':9.5,fill:P.muted},s,m.slice(2))});
  const mean=vals.reduce((a,b)=>a+b,0)/n,ml=el('line',{x1:L,x2:W-R,y1:y(mean),y2:y(mean),stroke:P.gold,'stroke-dasharray':'5 4','stroke-width':1.6,opacity:0},s);
  const mt=el('text',{x:W-R,y:y(mean)-5,'text-anchor':'end','font-size':11,fill:P.gold,opacity:0},s,'среднее '+fmt(mean,4,true));
  once(s,()=>run(1600,p=>{rs.forEach((r,i)=>{const q=easeOut(clamp(p*1.6-i/n*.6,0,1)),hh=(y(0)-y(vals[i]))*q;r.setAttribute('y',y(0)-hh);r.setAttribute('height',hh)});if(p>.9){ml.setAttribute('opacity',1);mt.setAttribute('opacity',1)}}),.4)}

function brierBars(){
  const s=document.getElementById('brierBars');if(!s)return;const B_=D.brier,vals=B_.v,W=560,H=250,L=40,R=10,Tp=14,Bm=44,n=vals.length,bw=(W-L-R)/n,ymax=25,y=v=>Tp+(H-Tp-Bm)*(1-v/ymax);
  for(let k=0;k<=5;k++){const v=ymax*k/5;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},s);el('text',{x:L-5,y:y(v)+4,'text-anchor':'end','font-size':10,fill:P.muted},s,fmt(v,0)+'%')}
  const rs=vals.map((v,i)=>v==null?null:el('rect',{x:L+i*bw+1.4,y:y(0),width:bw-2.8,height:0,rx:2,fill:P.violet},s));
  B_.months.forEach((m,i)=>{if(i%4===0)el('text',{x:L+i*bw+bw/2,y:H-Bm+14,'text-anchor':'middle','font-size':9.5,fill:P.muted},s,m.slice(2))});
  const gi=vals.indexOf(null);if(gi>=0)el('text',{x:L+gi*bw+bw/2,y:y(3),'text-anchor':'middle','font-size':9,fill:P.muted},s,'нет');
  el('text',{x:L+6,y:H-6,'font-size':11,fill:P.ink2},s,'Look 1 · Bitstamp');el('text',{x:L+(gi+1)*bw,y:H-6,'font-size':11,fill:P.ink2},s,'Look 2 · Binance');
  const lo=Math.min(...vals.filter(v=>v!=null)),hi=Math.max(...vals.filter(v=>v!=null));
  el('text',{x:W-R,y:Tp+12,'text-anchor':'end','font-size':11,fill:P.gold},s,'от '+fmt(lo,1,true)+'% до '+fmt(hi,1,true)+'%');
  once(s,()=>run(1700,p=>{rs.forEach((r,i)=>{if(!r)return;const q=easeOut(clamp(p*1.6-i/n*.6,0,1)),hh=(y(0)-y(vals[i]))*q;r.setAttribute('y',y(0)-hh);r.setAttribute('height',hh)})}),.4)}

// ---------------------------------------------------------------- BARRIER animation
function barrier(){
  const b=D.barrier,svg=document.getElementById('barSvg');svg.innerHTML='';
  const W=1200,H=480,L=70,R=18,Tp=30,B=46,n=b.c.length,box=b.boxes;
  let lo=Math.min(...b.c,...box.map(q=>q.lo)),hi=Math.max(...b.c,...box.map(q=>q.hi));const pd=(hi-lo)*.06;lo-=pd;hi+=pd;
  const x=m=>L+(W-L-R)*m/(n-1),y=v=>Tp+(H-Tp-B)*(1-(v-lo)/(hi-lo));
  for(let k=0;k<=5;k++){const v=lo+(hi-lo)*k/5;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},svg);el('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':11,fill:P.muted},svg,Math.round(v).toLocaleString('ru-RU'))}
  for(let hr=0;hr<=24;hr+=3){const xx=x(hr*60);el('line',{x1:xx,x2:xx,y1:Tp,y2:H-B,stroke:P.grid,'stroke-dasharray':'2 4'},svg);el('text',{x:xx,y:H-B+18,'text-anchor':'middle','font-size':11,fill:P.muted},svg,('0'+(hr%24)).slice(-2)+':00')}
  el('text',{x:L,y:H-8,'font-size':11,fill:P.muted},svg,'UTC, 2025-11-15 · P(касание) модели H3 заполняет коробку снизу вверх');
  const groups=box.map((q,k)=>{
    const g=el('g',{opacity:0},svg),x0=x(q.m),x1=x(q.m+60),yt=y(q.hi),yb=y(q.lo);
    el('rect',{x:x0,y:yt,width:x1-x0,height:yb-yt,fill:P.surface,'fill-opacity':.35,stroke:P.neutral,'stroke-width':1.2,'stroke-dasharray':'4 3',class:'bx'},g);
    const fill=el('rect',{x:x0,y:yb,width:x1-x0,height:0,fill:P.violet,'fill-opacity':.55},g);
    el('text',{x:(x0+x1)/2,y:yt-5,'text-anchor':'middle','font-size':10.5,fill:P.ink2},g,fmt(q.p,2));
    const mk=q.hit?el('path',{d:`M${x(q.m+q.first)},${q.side>0?yt:yb}m0,-7l7,7l-7,7l-7,-7z`,fill:P.gold,stroke:P.paper,'stroke-width':1.5,opacity:0,transform:'',class:''},g):null;
    return {g,fill,mk,yb,full:yb-yt,q,shown:false,t0:0,done:false}});
  const priceP=el('path',{d:pathLine(b.c.map((v,m)=>[x(m),y(v)])),fill:'none',stroke:P.ink,'stroke-width':1.5,'stroke-linejoin':'round'},svg);drawable(priceP);
  const head=el('circle',{r:4.5,fill:P.ink,cx:x(0),cy:y(b.c[0])},svg);
  const tally=document.getElementById('barStats');
  function stat(k,done){tally.innerHTML=`Окон: ${box.length} · касаний границы: <b>${k}</b>`+(done?` · средний P(касание) в окнах с касанием <b>${fmt(b.mean_p_hit,3)}</b>, в окнах без касания <b>${fmt(b.mean_p_miss,3)}</b>`:'')}
  stat(0,false);
  function finish(){setDraw(priceP,1);head.setAttribute('opacity',0);groups.forEach(o=>{o.g.setAttribute('opacity',1);o.fill.setAttribute('height',o.full*o.q.p);o.fill.setAttribute('y',o.yb-o.full*o.q.p);
      if(o.mk)o.mk.setAttribute('opacity',1);else o.g.setAttribute('opacity',.55);if(o.q.hit)o.g.querySelector('.bx').setAttribute('stroke',P.gold)});stat(b.n_hit,true)}
  if(REDUCED){finish();return}
  setDraw(priceP,0);let hits=0;const DUR=14000/SPEED,t0=performance.now()+300/SPEED;
  (function f(now){const p=clamp((now-t0)/DUR,0,1),cur=p*(n-1);setDraw(priceP,p);const mi=Math.min(n-1,Math.floor(cur));head.setAttribute('cx',x(mi));head.setAttribute('cy',y(b.c[mi]));
    groups.forEach(o=>{const q=o.q;
      if(!o.shown&&cur>=q.m){o.shown=true;o.t0=now;o.g.setAttribute('opacity',1)}
      if(o.shown){const u=clamp((now-o.t0)/(900/SPEED),0,1),h=o.full*q.p*easeOut(u);o.fill.setAttribute('height',h);o.fill.setAttribute('y',o.yb-h)}
      if(o.shown&&!o.done&&q.hit&&cur>=q.m+q.first){o.done=true;hits++;stat(hits,false);o.g.querySelector('.bx').setAttribute('stroke',P.gold);o.g.querySelector('.bx').setAttribute('stroke-width',2.4);o.g.querySelector('.bx').setAttribute('stroke-dasharray','');o.g.classList.add('flash');
        o.mk.setAttribute('opacity',1)}
      if(o.shown&&!o.done&&!q.hit&&cur>=q.m+60){o.done=true;o.g.setAttribute('opacity',.55)}
    });
    if(p<1)requestAnimationFrame(f);else{head.setAttribute('opacity',0);stat(b.n_hit,true)}})(performance.now());
}

// ---------------------------------------------------------------- HAR bands
let harDay=0;
function har(){
  const days=D.har.slice().sort((a,b)=>b.sd-a.sd),btn=document.getElementById('harBtns');
  if(!btn.children.length){days.forEach((d,i)=>{const b=document.createElement('button');b.textContent=d.label+' · '+d.date;b.className=i===harDay?'on':'';b.onclick=()=>{harDay=i;[...btn.children].forEach((c,k)=>c.classList.toggle('on',k===i));har()};btn.appendChild(b)})}
  const d=days[harDay],svg=document.getElementById('harSvg');svg.innerHTML='';
  const W=1200,Hh=430,L=70,R=18,Tp=20,B=40,n=1500;
  const all=[...d.c,...d.win.flatMap(w=>[w.hlo,w.hhi,w.tlo,w.thi])];let lo=Math.min(...all),hi=Math.max(...all);const pd=(hi-lo)*.04;lo-=pd;hi+=pd;
  const x=m=>L+(W-L-R)*m/(n-1),y=v=>Tp+(Hh-Tp-B)*(1-(v-lo)/(hi-lo));
  for(let k=0;k<=5;k++){const v=lo+(hi-lo)*k/5;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},svg);el('text',{x:L-8,y:y(v)+4,'text-anchor':'end','font-size':11,fill:P.muted},svg,Math.round(v).toLocaleString('ru-RU'))}
  for(let hr=0;hr<=24;hr+=3){const xx=x(hr*60);el('line',{x1:xx,x2:xx,y1:Tp,y2:Hh-B,stroke:P.grid,'stroke-dasharray':'2 4'},svg);el('text',{x:xx,y:Hh-B+18,'text-anchor':'middle','font-size':11,fill:P.muted},svg,('0'+(hr%24)).slice(-2)+':00')}
  const wins=d.win.map((w,i)=>{const g=el('g',{opacity:0},svg),x0=x(w.m),x1=x(w.m+60);
    el('rect',{x:x0,y:y(w.thi),width:x1-x0,height:y(w.tlo)-y(w.thi),fill:'none',stroke:P.neutral,'stroke-dasharray':'4 3','stroke-width':1.2},g);
    const r=el('rect',{x:x0,y:y(w.c0),width:x1-x0,height:0,fill:P.violet,'fill-opacity':.24,stroke:P.violet,'stroke-width':1.5,class:'breathe',style:'animation-delay:'+(i%6*.5)+'s'},g);
    const dot=el('circle',{cx:x1,cy:y(w.end),r:3.6,fill:P.gold,stroke:P.paper,opacity:0},g);return {g,r,dot,w,t0:null,dh:0}});
  const priceP=el('path',{d:pathLine(d.c.map((v,i)=>[x(d.m[i]),y(v)])),fill:'none',stroke:P.ink,'stroke-width':1.4},svg);drawable(priceP);
  const head=el('circle',{r:4,fill:P.ink},svg);
  const st=document.getElementById('harStats');
  st.innerHTML=`<b>${d.label}</b> · 1-минутный σ дня ${fmt(d.sd,2)} б.п. · средняя ширина 90%-интервала на 60 минут: <b>HAR ${fmt(d.w_har,1)} б.п.</b> · скользящая RV ${fmt(d.w_trail,1)} б.п. · окон ${d.n}`;
  function setWin(o,u){o.g.setAttribute('opacity',1);const w=o.w,top=mix(y(w.c0),y(w.hhi),easeOut(u)),bot=mix(y(w.c0),y(w.hlo),easeOut(u));o.r.setAttribute('y',top);o.r.setAttribute('height',Math.max(0,bot-top))}
  if(REDUCED){setDraw(priceP,1);head.setAttribute('opacity',0);wins.forEach(o=>{setWin(o,1);o.dot.setAttribute('opacity',1)});return}
  setDraw(priceP,0);const DUR=9000/SPEED,t0=performance.now()+300/SPEED;
  (function f(now){const p=clamp((now-t0)/DUR,0,1),cur=p*(n-1);setDraw(priceP,p);const i=Math.min(d.c.length-1,Math.floor(p*(d.c.length-1)));head.setAttribute('cx',x(d.m[i]));head.setAttribute('cy',y(d.c[i]));
    wins.forEach(o=>{if(o.t0==null&&cur>=o.w.m)o.t0=now;if(o.t0!=null){setWin(o,clamp((now-o.t0)/(900/SPEED),0,1));if(cur>=o.w.m+60)o.dot.setAttribute('opacity',1)}});
    if(p<1)requestAnimationFrame(f);else head.setAttribute('opacity',0)})(performance.now());
}

// ---------------------------------------------------------------- the fade schematic (a scheme, not data)
let fadeTimer=null;
function fade(){
  const svg=document.getElementById('fadeSvg');svg.innerHTML='';if(fadeTimer){clearTimeout(fadeTimer);fadeTimer=null}
  const W=560,H=330,L=24,R=104,Tp=34,B=26,n=200,vis=visibleFlag(svg);
  const x=i=>L+(W-L-R)*i/(n-1),yv=v=>Tp+(H-Tp-B)*(0.78-v/5.4);
  const base=i=>0.35*Math.sin(i/200*Math.PI*1.2),
    stretch=i=>{const t=i/n;return t<.15?0:t<.5?3.4*Math.sin((t-.15)/.35*Math.PI/2)**1.3:3.4*Math.exp(-((t-.5)/.14))},
    price=i=>base(i)+stretch(i)+.13*Math.sin(i*.9)+.08*Math.sin(i*2.1);
  el('path',{d:'M'+[...Array(n)].map((_,i)=>x(i)+','+yv(base(i)+1.1)).join('L')+'L'+[...Array(n)].map((_,i)=>x(n-1-i)+','+yv(base(n-1-i)-1.1)).join('L')+'Z',fill:P.violet,'fill-opacity':.14,stroke:'none'},svg);
  el('path',{d:pathLine([...Array(n)].map((_,i)=>[x(i),yv(base(i))])),fill:'none',stroke:P.violet,'stroke-width':2.4},svg);
  el('text',{x:L+4,y:yv(base(0)+1.1)-6,'font-size':11,fill:P.violet},svg,'полоса EMA 20');
  const seg1=el('path',{d:'',fill:'none',stroke:P.ink,'stroke-width':2},svg),seg2=el('path',{d:'',fill:'none',stroke:P.pink,'stroke-width':3.4},svg);
  const head=el('circle',{r:5,fill:P.ink},svg);
  const gx=W-R+40,gh=H-Tp-B-6;el('rect',{x:gx,y:Tp,width:22,height:gh,rx:6,fill:'none',stroke:P.axis},svg);
  const gf=el('rect',{x:gx+2,y:Tp+gh-2,width:18,height:0,rx:4,fill:P.pink},svg),gmax=el('line',{x1:gx-4,x2:gx+26,y1:Tp+gh,y2:Tp+gh,stroke:P.gold,'stroke-width':2,opacity:0},svg);
  el('text',{x:gx+11,y:Tp-12,'text-anchor':'middle','font-size':11,fill:P.ink2},svg,'растяжение');
  const arrow=el('g',{opacity:0},svg);el('path',{d:'M0,0L0,40',stroke:P.gold,'stroke-width':4,'stroke-linecap':'round'},arrow);el('polygon',{points:'0,52 -9,36 9,36',fill:P.gold},arrow);
  el('text',{x:0,y:-8,'text-anchor':'middle','font-size':12,fill:P.gold,'font-weight':600},arrow,'откат');
  const stl=el('text',{x:x(70),y:yv(4.2),'text-anchor':'middle','font-size':12,fill:P.pink,opacity:0},svg,'растянуто');
  let gtop=0;
  function upto(k){const pts=[];for(let i=0;i<=k;i++)pts.push([x(i),yv(price(i))]);
    seg1.setAttribute('d',pathLine(pts));let ov='',open=false;pts.forEach((p,i)=>{if(stretch(i)>1.2){ov+=(open?'L':'M')+p[0].toFixed(1)+','+p[1].toFixed(1);open=true}else open=false});seg2.setAttribute('d',ov||'M0,0');
    head.setAttribute('cx',pts[k][0]);head.setAttribute('cy',pts[k][1]);const z=clamp(stretch(k)/3.4,0,1)*(gh-4);gtop=Math.max(gtop,z);
    gf.setAttribute('y',Tp+gh-2-z);gf.setAttribute('height',Math.max(0,z));gmax.setAttribute('y1',Tp+gh-2-gtop);gmax.setAttribute('y2',Tp+gh-2-gtop);gmax.setAttribute('opacity',gtop>4?1:0);
    stl.setAttribute('opacity',stretch(k)>2.2&&k<n*.5?1:0);
    const rev=k>n*.5&&stretch(k)>.8;arrow.setAttribute('opacity',rev?1:0);if(rev)arrow.setAttribute('transform',`translate(${x(k)+18},${yv(price(k))-30})`)}
  if(REDUCED){upto(n-1);return}
  function cycle(){gtop=0;upto(0);run(6500,p=>upto(Math.floor(ease(p)*(n-1))),200).then(()=>{fadeTimer=setTimeout(()=>{if(vis.v)cycle();else{fadeTimer=setTimeout(cycle,1500)}},2600/SPEED)})}
  cycle();
}

// ---------------------------------------------------------------- deciles
let decEra='2017-19';
function deciles(){
  const eras=Object.keys(D.dec),btn=document.getElementById('decBtns');
  if(!btn.children.length)eras.forEach(e=>{const b=document.createElement('button');b.textContent=e;b.className=e===decEra?'on':'';b.onclick=()=>{decEra=e;[...btn.children].forEach(c=>c.classList.toggle('on',c.textContent===e));paint(true)};btn.appendChild(b)});
  const svg=document.getElementById('decSvg');svg.innerHTML='';const W=560,H=300,L=46,R=12,Tp=22,B=40,lo=.48,hi=.62,y=v=>Tp+(H-Tp-B)*(1-(v-lo)/(hi-lo)),bw=(W-L-R)/10;
  for(let k=0;k<=7;k++){const v=lo+(hi-lo)*k/7;el('line',{x1:L,x2:W-R,y1:y(v),y2:y(v),stroke:P.grid},svg);el('text',{x:L-6,y:y(v)+4,'text-anchor':'end','font-size':10.5,fill:P.muted},svg,fmt(v,3))}
  el('line',{x1:L,x2:W-R,y1:y(.5),y2:y(.5),stroke:P.neutral,'stroke-dasharray':'5 4','stroke-width':1.4},svg);el('text',{x:L+4,y:y(.5)+14,'text-anchor':'start','font-size':10.5,fill:P.neutral},svg,'монетка 0,500');
  const rs=[],ts=[];for(let i=0;i<10;i++){rs.push(el('rect',{x:L+i*bw+4,y:y(.5),width:bw-8,height:0,rx:3,fill:P.gold},svg));ts.push(el('text',{x:L+i*bw+bw/2,y:y(.5)-6,'text-anchor':'middle','font-size':10.5,fill:P.ink,opacity:0},svg,''));
    el('text',{x:L+i*bw+bw/2,y:H-B+16,'text-anchor':'middle','font-size':11,fill:P.muted},svg,String(i+1))}
  el('text',{x:(L+W-R)/2,y:H-6,'text-anchor':'middle','font-size':11,fill:P.muted},svg,'дециль |bb20| (10 — самая растянутая цена)');
  window.__decPaint=paint;
  function paint(anim){const vals=D.dec[decEra];
    const set=p=>vals.forEach((v,i)=>{const q=easeOut(clamp(p*1.5-i/10*.5,0,1)),yy=y(mix(.5,v,q));const top=Math.min(yy,y(.5)),hh=Math.abs(yy-y(.5));rs[i].setAttribute('y',top);rs[i].setAttribute('height',hh);rs[i].setAttribute('fill',v>=.5?P.gold:P.muted);
      ts[i].setAttribute('y',top-5);ts[i].textContent=(i===0||i===9||p>=1&&false)?fmt(v,3):'';ts[i].setAttribute('opacity',(i===0||i===9)&&p>.85?1:0)});
    if(!anim||REDUCED){set(1);return}run(1100,set)}
  if(REDUCED)paint(false);else once(svg,()=>paint(true),.4);
}

// ---------------------------------------------------------------- timeline
function timeline(){
  const tl=document.getElementById('tl'),t0=Date.UTC(2026,9,1),t1=Date.UTC(2027,2,1),pc=s=>{const [Y,M,Dd]=s.split('-').map(Number),mi=(Y-2026)*12+(M-1)-9,dim=new Date(Y,M,0).getDate();return (mi+(Dd-1)/dim)/5*100};
  D.timeline.forEach(r=>{const row=document.createElement('div');row.className='row';
    row.innerHTML=`<div class="lbl"><b>${r.id}</b><span class="sha" title="${r.src}">sha256 ${r.sha}…</span><span>${r.what} · ${r.crit}</span></div><div class="track"><div class="bar" style="left:${pc(r.a)}%;--w:${pc(r.b)-pc(r.a)}%"></div><div class="dia" style="left:${pc(r.b)}%"></div></div>`;tl.appendChild(row)});
  const ax=document.createElement('div');ax.className='axis';ax.innerHTML='<div></div><div class="months"><span>окт 2026</span><span>ноя</span><span>дек</span><span>янв 2027</span><span>фев</span></div>';tl.appendChild(ax);
  const track=tl.querySelector('.track'),today=document.createElement('div');today.className='today';today.innerHTML='<span>сегодня</span>';
  // the "today" line spans the track column: place it using the first track's offset
  function placeToday(){const tr=tl.querySelector('.track'),r=tr.getBoundingClientRect(),b=tl.getBoundingClientRect();today.style.left=(r.left-b.left+r.width*pc('2026-10-15')/100)+'px'}
  tl.appendChild(today);placeToday();addEventListener('resize',placeToday);
  const bars=[...tl.querySelectorAll('.bar')];
  if(REDUCED){bars.forEach(b=>b.style.width=getComputedStyle(b).getPropertyValue('--w'));tl.classList.add('go');return}
  once(tl,()=>{bars.forEach((b,i)=>setTimeout(()=>{b.style.transition='width 1.1s cubic-bezier(.2,.8,.2,1)';b.style.width=b.style.getPropertyValue('--w')},i*140/SPEED));setTimeout(()=>tl.classList.add('go'),(bars.length*140+900)/SPEED);placeToday()},.25);
}

// ---------------------------------------------------------------- go
document.querySelectorAll('[data-src]').forEach(e=>e.title='Источник: '+e.dataset.src);
counters();
spark('decaySpark',Object.values(D.decay),Object.keys(D.decay),P.violet,.2,.45);
spark('t01Spark',D.t01.v,D.t01.months.map((m,i)=>i%3===0?m.slice(2):null),P.gold,0,.2);
spark('o11Spark',D.o11a.diff,D.o11a.months.map((m,i)=>i%6===0?m.slice(2):null),P.pink,0,.09);
hero();document.querySelectorAll('.replay').forEach(b=>b.onclick=()=>({hero,barrier,har,fade})[b.dataset.for]&&({hero,barrier,har,fade})[b.dataset.for]());
once(document.getElementById('barSvg'),()=>{barrier()},.35);
once(document.getElementById('harSvg'),()=>{har()},.35);
once(document.getElementById('fadeSvg'),()=>{fade()},.4);
clock();geomPanels();brierBars();bars('o5Bars',D.o5[15].months,D.o5[15].diff,P.violet,.03,3);deciles();timeline();
})();
