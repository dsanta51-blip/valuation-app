"""
Portfólio dashboard (pages/2_Watchlist.py): tabuľka pozícií, dvojvrstvový donut sektor/akcia, KPI karty a pridanie akcie.
Ceny zadávaš ručne; dáta sa ukladajú v prehliadači (localStorage). Nové knižnice netreba.
"""
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Portfólio", page_icon="📈", layout="wide")

HTML = r"""<!DOCTYPE html>
<html lang="sk">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Portfólio akcií</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#f4f5f8;--card:#fff;--text:#111827;--mut:#6b7280;--line:#e5e7eb;--acc:#4f46e5;--pos:#0f9d58;--neg:#d93025;color-scheme:light;box-sizing:border-box;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#0b0f17;--card:#131a28;--text:#e6e9f0;--mut:#8b93a7;--line:#232c40;--acc:#8b92ff;--pos:#34d399;--neg:#f87171;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#0b0f17;--card:#131a28;--text:#e6e9f0;--mut:#8b93a7;--line:#232c40;--acc:#8b92ff;--pos:#34d399;--neg:#f87171;color-scheme:dark}
html{scroll-padding-top:env(safe-area-inset-top,0px)}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
body::before{content:"";position:fixed;inset:0 0 auto;height:env(safe-area-inset-top,0px);background:var(--bg);z-index:10}
header{position:sticky;top:env(safe-area-inset-top,0px);z-index:9;background:var(--bg);display:flex;justify-content:space-between;align-items:center;gap:12px;padding:14px 20px;border-bottom:1px solid var(--line)}
h1{font-size:18px;margin:0}
h2{font-size:14px;margin:0 0 12px;color:var(--mut);font-weight:600}
main{max-width:1200px;margin:0 auto;padding:20px;display:grid;gap:16px}
.btn{background:var(--acc);color:#fff;border:0;border-radius:10px;padding:9px 16px;font:600 14px inherit;cursor:pointer}
.btn.g{background:transparent;color:var(--mut);border:1px solid var(--line)}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}
.kpi small{color:var(--mut);display:block}
.kpi b{font-size:26px;display:block;margin:4px 0 2px;letter-spacing:-.5px}
.pos{color:var(--pos)}.neg{color:var(--neg)}
.donut{display:grid;grid-template-columns:minmax(0,420px) 1fr;gap:20px;align-items:center}
.dw{position:relative}
svg{width:100%;height:auto;display:block}
.seg{stroke:var(--card);stroke-width:2;transition:opacity .15s}
svg.h .seg:not(.on){opacity:.25}
#tip{position:absolute;pointer-events:none;background:var(--text);color:var(--bg);padding:8px 10px;border-radius:8px;font-size:12px;opacity:0;transition:opacity .1s;white-space:nowrap;z-index:5}
.lg div{display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--line)}
.lg i{width:10px;height:10px;border-radius:50%}.lg span{margin-left:auto;color:var(--mut)}
.tw{overflow-x:auto}
table{width:100%;border-collapse:collapse;min-width:900px}
th,td{padding:10px 8px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--mut);font-weight:500;font-size:12px}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
td small{display:block;color:var(--mut)}
dialog{border:1px solid var(--line);border-radius:16px;background:var(--card);color:var(--text);padding:20px;width:min(480px,92vw)}
dialog::backdrop{background:rgba(0,0,0,.5)}
form{display:grid;grid-template-columns:1fr 1fr;gap:10px}
label{font-size:12px;color:var(--mut);display:grid;gap:4px}
label.w,.act{grid-column:1/-1}
input{background:var(--bg);color:var(--text);border:1px solid var(--line);border-radius:8px;padding:8px;font:inherit;width:100%}
.act{display:flex;gap:8px;justify-content:flex-end}
@media (max-width:820px){.donut{grid-template-columns:1fr}main{padding:12px}}
@media (max-width:640px){table,tbody,tr,td{display:block;min-width:0}thead{display:none}tr{padding:10px 0;border-bottom:1px solid var(--line)}td{display:flex;justify-content:space-between;border:0;padding:3px 0}td::before{content:attr(data-l);color:var(--mut)}td small{display:none}}
</style>
</head>
<body>
<header><h1>📈 Portfólio akcií</h1><div><button class="btn g" id="clr">Vymazať ukážkové dáta</button> <button class="btn" id="add">+ Pridať akciu</button></div></header>
<main>
<section class="kpis">
<div class="card kpi"><small>Celková hodnota portfólia</small><b id="k1">–</b><small id="k1s"></small></div>
<div class="card kpi"><small>Celkový zisk / strata</small><b id="k2">–</b><small id="k2s"></small></div>
<div class="card kpi"><small>Najvýnosnejší sektor</small><b id="k3">–</b><small id="k3s"></small></div>
</section>
<section class="card"><h2>Rozloženie investovaného kapitálu: vnútorný kruh sektory, vonkajší akcie</h2>
<div class="donut"><div class="dw"><svg id="ch" viewBox="0 0 400 400" role="img" aria-label="Dvojvrstvový kruhový graf portfólia"></svg><div id="tip"></div></div><div class="lg" id="lg"></div></div></section>
<section class="card"><h2>Pozície</h2><div class="tw"><table><thead><tr><th>Ticker</th><th>Firma a sektor</th><th>Aktuálna cena</th><th>Férová cena</th><th>Priemer 5 rokov</th><th>Nákupná cena</th><th>Predajná cena</th><th>Zisk / strata</th><th>Yield on Cost</th></tr></thead><tbody id="tb"></tbody></table></div>
<p style="color:var(--mut);font-size:12px;margin:12px 0 0">Ceny zadávaš ručne a ukladajú sa len v tomto prehliadači. Ukážkové čísla nie sú živé kotácie.</p></section>
</main>
<dialog id="dlg"><form id="f" method="dialog"><h2 class="w" style="grid-column:1/-1">Nová akcia</h2>
<label>Ticker<input name="t" required maxlength="12"></label><label>Názov firmy<input name="n"></label>
<label class="w">Sektor<input name="s" list="sl" required><datalist id="sl"><option value="Technológie"><option value="Zdravotníctvo"><option value="Spotrebný tovar"><option value="Financie"><option value="Energetika"><option value="Priemysel"></datalist></label>
<label>Počet kusov<input name="q" type="number" step="any" min="0" required></label><label>Nákupná cena ($)<input name="b" type="number" step="any" min="0" required></label>
<label>Aktuálna cena ($)<input name="c" type="number" step="any" min="0" required></label><label>Férová cena ($)<input name="f" type="number" step="any" min="0"></label>
<label>Priemerná cena 5 r. ($)<input name="a" type="number" step="any" min="0"></label><label>Predajná cena ($, ak predané)<input name="x" type="number" step="any" min="0"></label>
<label class="w">Ročná dividenda na akciu ($)<input name="d" type="number" step="any" min="0"></label>
<div class="act"><button class="btn g" type="button" id="cx">Zrušiť</button><button class="btn" type="submit">Uložiť</button></div></form></dialog>
<script>
const NB='\u00A0',nf=new Intl.NumberFormat('sk-SK',{minimumFractionDigits:2,maximumFractionDigits:2});
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const money=v=>Number.isFinite(v)?nf.format(v)+NB+'$':'–',pct=(v,s)=>Number.isFinite(v)?(s&&v>0?'+':'')+nf.format(v)+NB+'%':'–';
const DEMO=[
{t:'AAPL',n:'Apple Inc.',s:'Technológie',q:20,b:150,c:225,f:210,a:165,x:null,d:1},
{t:'MSFT',n:'Microsoft Corp.',s:'Technológie',q:10,b:300,c:420,f:400,a:330,x:null,d:3.3},
{t:'GOOGL',n:'Alphabet Inc.',s:'Technológie',q:12,b:120,c:170,f:185,a:135,x:null,d:.8},
{t:'JNJ',n:'Johnson & Johnson',s:'Zdravotníctvo',q:25,b:150,c:160,f:175,a:160,x:null,d:4.9},
{t:'ABBV',n:'AbbVie Inc.',s:'Zdravotníctvo',q:12,b:140,c:190,f:180,a:150,x:null,d:6.4},
{t:'PG',n:'Procter & Gamble',s:'Spotrebný tovar',q:18,b:140,c:165,f:160,a:150,x:null,d:4},
{t:'KO',n:'Coca-Cola Co.',s:'Spotrebný tovar',q:40,b:58,c:62,f:65,a:59,x:null,d:1.9},
{t:'NKE',n:'Nike Inc.',s:'Spotrebný tovar',q:15,b:130,c:80,f:100,a:110,x:95,d:1.5}];
let S=null;try{S=JSON.parse(localStorage.getItem('pf1'))}catch(e){}
if(!Array.isArray(S))S=DEMO.slice();
const save=()=>{try{localStorage.setItem('pf1',JSON.stringify(S))}catch(e){}};
const HUE=[226,160,28,330,270,190,50,0];
function arc(r0,r1,a0,a1){if(a1-a0>6.283)a1=a0+6.283;const P=(r,a)=>(200+r*Math.cos(a)).toFixed(2)+','+(200+r*Math.sin(a)).toFixed(2),l=a1-a0>Math.PI?1:0;
return`M${P(r1,a0)}A${r1} ${r1} 0 ${l} 1 ${P(r1,a1)}L${P(r0,a1)}A${r0} ${r0} 0 ${l} 0 ${P(r0,a0)}Z`}
function render(){
const L=S.map(p=>{const sold=p.x>0,inv=p.q*p.b,val=p.q*(sold?p.x:p.c);return{...p,sold,inv,val,pl:(val/inv-1)*100,yoc:p.d>0?p.d/p.b*100:NaN}}).filter(p=>p.inv>0);
const held=L.filter(p=>!p.sold),tot=held.reduce((a,p)=>a+p.q*p.c,0),inv=L.reduce((a,p)=>a+p.inv,0),val=L.reduce((a,p)=>a+p.val,0);
const sc=[...new Set(L.map(p=>p.s))],cls=v=>v>=0?'pos':'neg';
$('#k1').textContent=money(tot);$('#k1s').textContent=held.length+' držaných pozícií';
const pl=inv?(val/inv-1)*100:NaN;$('#k2').textContent=pct(pl,1);$('#k2').className=cls(pl);$('#k2s').textContent=money(val-inv)+' (vrátane predaných)';
let best=null;sc.forEach(s=>{const g=L.filter(p=>p.s===s),i=g.reduce((a,p)=>a+p.inv,0),v=g.reduce((a,p)=>a+p.val,0),r=(v/i-1)*100;if(!best||r>best.r)best={s,r}});
$('#k3').textContent=best?best.s:'–';$('#k3s').textContent=best?pct(best.r,1):'';
const hi=held.reduce((a,p)=>a+p.inv,0),secs=sc.map(s=>({s,items:held.filter(p=>p.s===s).sort((a,b)=>b.inv-a.inv),h:HUE[sc.indexOf(s)%8]})).filter(g=>g.items.length);
secs.forEach(g=>g.amt=g.items.reduce((a,p)=>a+p.inv,0));secs.sort((a,b)=>b.amt-a.amt);
let a=-Math.PI/2,h='',lg='';
secs.forEach(g=>{const a0=a,P=g.amt/hi;
g.items.forEach((p,i)=>{const sp=p.inv/hi*6.2832,l=52+14*i/Math.max(g.items.length-1,1);
h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(p.t+' · '+p.n)}" data-a="${p.inv}" data-p="${p.inv/hi*100}" fill="hsl(${g.h} 70% ${l}%)" d="${arc(122,172,a,a+sp)}"/>`;a+=sp});
h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(g.s)}" data-a="${g.amt}" data-p="${P*100}" fill="hsl(${g.h} 70% 45%)" d="${arc(76,116,a0,a)}"/>`;
lg+=`<div><i style="background:hsl(${g.h} 70% 45%)"></i>${esc(g.s)}<span>${money(g.amt)} · ${pct(P*100)}</span></div>`});
const ch=$('#ch');ch.innerHTML=h+`<text x="200" y="192" text-anchor="middle" fill="var(--mut)" font-size="13" id="c1">Investované</text><text x="200" y="218" text-anchor="middle" fill="var(--text)" font-size="20" font-weight="700" id="c2">${money(hi)}</text>`;
$('#lg').innerHTML=lg||'<div>Žiadne držané pozície</div>';
$('#tb').innerHTML=L.map(p=>{const d=(c,f)=>Number.isFinite(f)&&f>0?`<small class="${cls(c-f)==='pos'?'neg':'pos'}">${pct((c/f-1)*100,1)}</small>`:'';
return`<tr><td data-l="Ticker"><b>${esc(p.t)}</b></td><td data-l="Firma"><span>${esc(p.n)}</span><small>${esc(p.s)}</small></td><td data-l="Aktuálna cena">${p.sold?'–':money(p.c)}</td><td data-l="Férová cena">${money(p.f)}${p.sold?'':d(p.c,p.f)}</td><td data-l="Priemer 5 rokov">${money(p.a)}${p.sold?'':d(p.c,p.a)}</td><td data-l="Nákupná cena">${money(p.b)}</td><td data-l="Predajná cena">${p.sold?money(p.x):'N/A'}</td><td data-l="Zisk / strata" class="${cls(p.pl)}"><b>${p.pl>=0?'▲':'▼'} ${pct(p.pl,1)}</b></td><td data-l="Yield on Cost">${pct(p.yoc)}</td></tr>`}).join('')||'<tr><td>Portfólio je prázdne. Pridaj prvú akciu.</td></tr>';
const tip=$('#tip'),w=ch.parentNode,c1=$('#c1'),c2=$('#c2'),b0=money(hi);
ch.onmouseover=ch.onmousemove=e=>{const s=e.target.closest('.seg');if(!s)return;ch.classList.add('h');ch.querySelectorAll('.seg').forEach(x=>x.classList.toggle('on',x===s||x.dataset.g===s.dataset.g&&(s.dataset.t===s.dataset.g||x.dataset.t===x.dataset.g)));
const r=w.getBoundingClientRect();tip.innerHTML=`<b>${esc(s.dataset.t)}</b><br>Investovaná suma: ${money(+s.dataset.a)}<br>Podiel z portfólia: ${pct(+s.dataset.p)}`;tip.style.left=Math.min(e.clientX-r.left+14,r.width-190)+'px';tip.style.top=e.clientY-r.top+14+'px';tip.style.opacity=1;c1.textContent=s.dataset.t.split(' · ')[0];c2.textContent=money(+s.dataset.a)};
ch.onmouseleave=()=>{ch.classList.remove('h');tip.style.opacity=0;c1.textContent='Investované';c2.textContent=b0}}
render();
const dlg=$('#dlg'),f=$('#f');
$('#add').onclick=()=>{f.reset();dlg.showModal()};$('#cx').onclick=()=>dlg.close();
f.onsubmit=e=>{e.preventDefault();const v=Object.fromEntries(new FormData(f)),n=k=>v[k]===''?null:+v[k];
S.push({t:v.t.trim().toUpperCase(),n:v.n.trim()||v.t.trim().toUpperCase(),s:v.s.trim(),q:n('q'),b:n('b'),c:n('c'),f:n('f'),a:n('a'),x:n('x'),d:n('d')||0});save();dlg.close();render()};
$('#clr').onclick=()=>{S=[];save();render()};
</script>
</body>
</html>
"""

components.html(HTML, height=1500, scrolling=True)
