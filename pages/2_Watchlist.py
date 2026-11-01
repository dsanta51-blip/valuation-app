"""
Portfólio dashboard (pages/2_Watchlist.py)
- Nákupy zadávaš v sekcii „Správa nákupov“ (viac nákupov jednej akcie na rôznych cenách).
- Aktuálne ceny, 5-ročný priemer a dividenda sa sťahujú automaticky z Yahoo Finance (cez yfinance).
- Zisk a strata sa počítajú v EUR vrátane zmeny kurzu EUR/USD (ako v XTB).
- Dáta sa ukladajú do portfolio_lots.csv v aplikácii (po reštarte sa môžu stratiť): používaj zálohu (CSV).
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Portfólio", page_icon="📈", layout="wide")

FILE = "portfolio_lots.csv"
COLS = ["Ticker", "Názov", "Sektor", "Dátum nákupu", "Kusy", "Cena nákupu", "Vložené €", "Férová cena", "Predajná cena"]
NUM = ["Kusy", "Cena nákupu", "Vložené €", "Férová cena", "Predajná cena"]
ALIAS = {"BRKB": "BRK-B", "BRKA": "BRK-A"}
REFRESH = {"Vypnuté": 0, "1 min": 60, "5 min": 300, "15 min": 900}


def sf(x, d=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else d
    except Exception:
        return d


def ysym(t):
    t = str(t).strip().upper()
    t = t[:-3] if t.endswith(".US") else t
    return ALIAS.get(t, t)


def normalize(df):
    df = pd.DataFrame() if df is None else df.copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS]
    for c in ("Ticker", "Názov", "Sektor"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["Ticker"] = df["Ticker"].str.upper()
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Dátum nákupu"] = pd.to_datetime(df["Dátum nákupu"], errors="coerce")
    return df[~((df["Ticker"] == "") & df[NUM].isna().all(axis=1))].reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes(symbols: tuple) -> dict:
    def one(s):
        out = dict(price=np.nan, cur="", prev=np.nan, exch="")
        try:
            fi = yf.Ticker(s).fast_info
            out["price"], out["cur"] = sf(fi["last_price"]), str(fi["currency"] or "")
            for k, key in (("prev", "previous_close"), ("exch", "exchange")):
                try:
                    out[k] = sf(fi[key]) if k == "prev" else str(fi[key] or "")
                except Exception:
                    pass
        except Exception:
            pass
        return s, out

    with ThreadPoolExecutor(max_workers=5) as ex:
        return dict(ex.map(one, symbols))


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_hist(symbols: tuple) -> dict:
    def one(s):
        out = dict(avg5=np.nan, last=np.nan, dps=np.nan, h5=[], m12=[])
        try:
            h = yf.Ticker(s).history(period="5y", auto_adjust=False, actions=True)
            c = h["Close"].dropna()
            if len(c):
                out.update(avg5=float(c.mean()), last=float(c.iloc[-1]))
                w = c.copy()
                w.index = w.index.tz_localize(None) if getattr(w.index, "tz", None) is not None else w.index
                out["h5"] = [[d.strftime("%Y-%m-%d"), float(v)] for d, v in w.resample("W").last().dropna().items()]
                out["m12"] = [float(v) for v in w.iloc[-252:].iloc[::5].tolist()]
            if "Dividends" in h:
                idx = h.index.tz_localize(None) if getattr(h.index, "tz", None) is not None else h.index
                out["dps"] = float(h["Dividends"][idx >= pd.Timestamp.now() - pd.Timedelta(days=365)].sum())
        except Exception:
            pass
        return s, out

    with ThreadPoolExecutor(max_workers=5) as ex:
        return dict(ex.map(one, symbols))


@st.cache_data(ttl=300, show_spinner=False)
def get_rates(ccys: tuple) -> dict:
    """Koľko jednotiek meny je 1 EUR (EUR → 1)."""
    out = {"EUR": 1.0}
    for c in ccys:
        if c and c != "EUR":
            try:
                out[c] = sf(yf.Ticker(f"EUR{c}=X").fast_info["last_price"])
            except Exception:
                out[c] = np.nan
    return out


def build(lots, quotes, hist, rates):
    pos = {}
    for _, r in lots.iterrows():
        q, b = r["Kusy"], r["Cena nákupu"]
        if not r["Ticker"] or not (q > 0 and b > 0):
            continue
        ys = ysym(r["Ticker"])
        qu, hs = quotes.get(ys, {}), hist.get(ys, {})
        pen = qu.get("cur") == "GBp"
        sc, ccy = (0.01 if pen else 1.0), ("GBP" if pen else (qu.get("cur") or "USD"))
        rate = rates.get(ccy, np.nan)
        price = sf(qu.get("price")) * sc
        live = np.isfinite(price)
        if not live:
            price = sf(hs.get("last")) * sc
        p = pos.setdefault(r["Ticker"], dict(t=r["Ticker"], n="", s="", fair=np.nan, cur=price, live=live,
                                              avg5=sf(hs.get("avg5")) * sc, dps=sf(hs.get("dps")) * sc, lots=[],
                                              prev=sf(qu.get("prev")) * sc, exch=qu.get("exch", ""), ccy=ccy, rate=rate,
                                              h5=[[d, v * sc] for d, v in hs.get("h5", [])],
                                              m12=[v * sc for v in hs.get("m12", [])]))
        p["n"] = r["Názov"] or p["n"] or r["Ticker"]
        p["s"] = r["Sektor"] or p["s"] or "Nezaradené"
        if np.isfinite(r["Férová cena"]) and r["Férová cena"] > 0:
            p["fair"] = r["Férová cena"]
        est = not (np.isfinite(r["Vložené €"]) and r["Vložené €"] > 0)
        inv = q * b / rate if est and np.isfinite(rate) else r["Vložené €"]
        sold = np.isfinite(r["Predajná cena"]) and r["Predajná cena"] > 0
        val = q * (r["Predajná cena"] if sold else price) / rate if np.isfinite(rate) else np.nan
        p["lots"].append(dict(d=r["Dátum nákupu"].strftime("%Y-%m-%d") if pd.notna(r["Dátum nákupu"]) else None,
                              q=q, b=b, inv=inv, est=est, sold=sold, x=r["Predajná cena"] if sold else np.nan,
                              val=val, pl=(val / inv - 1) * 100 if inv and np.isfinite(val) else np.nan,
                              plEur=val - inv if np.isfinite(val) else np.nan))
    out = []
    for p in pos.values():
        L = p["lots"]
        held = [l for l in L if not l["sold"]]
        base = held or L
        q = sum(l["q"] for l in base)
        inv, val = sum(l["inv"] for l in base), sum(l["val"] for l in base)
        sold = [l for l in L if l["sold"]]
        avgB = sum(l["q"] * l["b"] for l in base) / q
        out.append(dict(
            t=p["t"], n=p["n"], s=p["s"], fair=p["fair"], cur=p["cur"], live=p["live"], avg5=p["avg5"], q=q, avgB=avgB,
            sell=(sum(l["q"] * l["x"] for l in sold) / sum(l["q"] for l in sold)) if sold else np.nan,
            inv=inv if held else 0.0, held=bool(held), val=val if held else 0.0,
            invAll=sum(l["inv"] for l in L), valAll=sum(l["val"] for l in L),
            pl=(val / inv - 1) * 100 if inv and np.isfinite(val) else np.nan, plEur=val - inv,
            yoc=p["dps"] / avgB * 100 if np.isfinite(p["dps"]) and p["dps"] > 0 else np.nan, lots=L,
            exch=p["exch"], ccy=p["ccy"], h5=p["h5"], m12=p["m12"],
            dayEur=q * (p["cur"] - p["prev"]) / p["rate"] if held and np.isfinite(p["prev"]) and np.isfinite(p["rate"]) else np.nan,
            dayPct=(p["cur"] / p["prev"] - 1) * 100 if np.isfinite(p["prev"]) and p["prev"] > 0 else np.nan,
            valLoc=q * p["cur"] if held else 0.0, invLoc=sum(l["q"] * l["b"] for l in base) if held else 0.0))
    return out


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


TEMPLATE = r"""<!DOCTYPE html>
<html lang="sk"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{--bg:#0e1424;--pn:#151c30;--pn2:#1b243b;--tx:#eef1f8;--mu:#8e98b3;--ln:#27314c;--ac:#f97316;--pos:#34d399;--neg:#f87171}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.4 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}small{display:block;color:var(--mu);font-size:11px}
.app{display:grid;grid-template-columns:250px 1fr;gap:12px;padding:12px;max-width:1500px;margin:0 auto}
.side{background:var(--pn);border:1px solid var(--ln);border-radius:14px;padding:16px;display:grid;gap:14px;align-content:start}
.side h3{margin:0;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ac)}
.m small{font-size:12px}.m b{font-size:22px;display:block;line-height:1.2}.m.s b{font-size:17px}
.main{display:grid;gap:12px;min-width:0}.card{background:var(--pn);border:1px solid var(--ln);border-radius:14px;padding:14px;min-width:0}
.card h2{margin:0 0 10px;font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--ac);font-weight:600;display:flex;justify-content:space-between;gap:8px;align-items:center}
select{background:var(--pn2);color:var(--tx);border:1px solid var(--ln);border-radius:8px;padding:4px 8px;font:inherit}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:1180px}
th,td{padding:8px 7px;text-align:right;border-bottom:1px solid var(--ln);white-space:nowrap}th{color:var(--mu);font-weight:500;font-size:11px}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}.tk{cursor:pointer;font-weight:700;color:var(--ac)}.tk::before{content:"▸ ";color:var(--mu)}.o .tk::before{content:"▾ "}
tr.lot{display:none}tr.lot.s{display:table-row}tr.lot td{color:var(--mu);font-size:12px;background:var(--pn2)}
.vb{background:linear-gradient(90deg,rgba(249,115,22,.35) var(--w),transparent var(--w))}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.charts .wide{grid-column:1/-1}
.dw{position:relative;max-width:380px;margin:0 auto}svg{width:100%;height:auto;display:block}
.seg{stroke:var(--pn);stroke-width:2;transition:opacity .15s}svg.h .seg:not(.on){opacity:.25}
#tip{position:absolute;pointer-events:none;background:#fff;color:#111;padding:7px 9px;border-radius:8px;font-size:12px;opacity:0;white-space:nowrap;z-index:5}
.br{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center;margin:8px 0}.br i{display:block;height:14px;border-radius:4px;background:var(--ac)}
.lg div{display:flex;gap:8px;align-items:center;padding:3px 0}.lg i{width:9px;height:9px;border-radius:50%}.lg span{margin-left:auto;color:var(--mu)}
@media(max-width:900px){.app{grid-template-columns:1fr}.charts{grid-template-columns:1fr}.side{grid-template-columns:1fr 1fr}.side h3{grid-column:1/-1}}
</style></head><body>
<div class="app">
<aside class="side"><h3>Portfólio</h3>
<div class="m"><small>Hodnota portfólia</small><b id="a1">–</b></div><div class="m"><small>Celková investícia</small><b id="a2">–</b></div>
<div class="m"><small>Nerealizovaný zisk / strata</small><b id="a3">–</b><small id="a3p"></small></div><div class="m"><small>Dnešná zmena</small><b id="a4">–</b><small id="a4p"></small></div>
<h3>Akcie v USD (bez kurzu)</h3>
<div class="m s"><small>Hodnota</small><b id="b1">–</b></div><div class="m s"><small>Zisk / strata</small><b id="b2">–</b><small id="b2p"></small></div><div class="m s"><small>Kurz EUR/USD</small><b id="fx">–</b><small id="ts"></small></div>
</aside>
<div class="main">
<div class="card"><h2>Pozície <span style="text-transform:none;letter-spacing:0;color:var(--mu)">klikni na ticker pre jednotlivé nákupy</span></h2><div class="tw"><table><thead><tr><th>Ticker</th><th>Firma · sektor · burza</th><th>Počet</th><th>Vložené</th><th>Nákup. cena (priem.)</th><th>Aktuálna cena</th><th>Férová cena</th><th>Priemer 5 r.</th><th>Predajná</th><th>Hodnota</th><th>Dnes</th><th>Od začiatku</th><th>Yield on Cost</th><th>12 mesiacov</th></tr></thead><tbody id="tb"></tbody></table></div></div>
<div class="charts">
<div class="card"><h2>Štruktúra portfólia <select id="dm"><option value="2">Sektory a akcie</option><option value="a">Akcie</option><option value="o">Odvetvia</option></select></h2><div class="dw"><svg id="ch" viewBox="0 0 400 400"></svg><div id="tip"></div></div><div class="lg" id="lg"></div></div>
<div class="card"><h2>Tržná hodnota podľa odvetví</h2><div id="bars"></div></div>
<div class="card wide"><h2>Vývoj ceny za 5 rokov so značkami nákupov <select id="st"></select></h2><svg id="ln" viewBox="0 0 800 280" preserveAspectRatio="none" style="height:260px"></svg><small id="lnn"></small></div>
</div></div></div>
<script>
const D=__DATA__,P=D.pos,NB='\u00A0',F=Number.isFinite,H=P.filter(p=>p.held);
const nf=new Intl.NumberFormat('sk-SK',{minimumFractionDigits:2,maximumFractionDigits:2}),nq=new Intl.NumberFormat('sk-SK',{maximumFractionDigits:4});
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const usd=v=>F(v)?nf.format(v)+NB+'$':'–',eur=v=>F(v)?nf.format(v)+NB+'€':'–',pct=(v,s)=>F(v)?(s&&v>0?'+':'')+nf.format(v)+NB+'%':'–',cl=v=>F(v)?(v>=0?'pos':'neg'):'',ar=v=>F(v)?(v>=0?'▲ ':'▼ '):'';
const sum=(a,f)=>a.reduce((x,p)=>x+(F(f(p))?f(p):0),0),HUE=[24,200,150,330,270,50,0,100];
let O={};try{O=JSON.parse(localStorage.getItem('pf4o'))||{}}catch(e){}
// boční panel
const iA=sum(P,p=>p.invAll),vA=sum(P,p=>p.valAll),dT=sum(H,p=>p.dayEur),vH=sum(H,p=>p.val);
$('#a1').textContent=eur(vH);$('#a2').textContent=eur(sum(H,p=>p.inv));
$('#a3').textContent=eur(vA-iA);$('#a3').className=cl(vA-iA);$('#a3p').textContent=ar(vA-iA)+pct(iA?(vA/iA-1)*100:NaN,1);
$('#a4').textContent=eur(dT);$('#a4').className=cl(dT);$('#a4p').textContent=ar(dT)+pct(vH-dT?dT/(vH-dT)*100:NaN,1);
const U=H.filter(p=>p.ccy==='USD'),uv=sum(U,p=>p.valLoc),ui=sum(U,p=>p.invLoc);
$('#b1').textContent=usd(uv);$('#b2').textContent=usd(uv-ui);$('#b2').className=cl(uv-ui);$('#b2p').textContent=ar(uv-ui)+pct(ui?(uv/ui-1)*100:NaN,1);
$('#fx').textContent=D.fx?nf.format(D.fx):'–';$('#ts').textContent='ceny z '+D.ts;
// tabuľka
const spark=a=>{if(!a||a.length<2)return'';const mn=Math.min(...a),mx=Math.max(...a),r=mx-mn||1,pts=a.map((v,i)=>(i/(a.length-1)*70).toFixed(1)+','+(22-(v-mn)/r*20).toFixed(1)).join(' ');return`<svg viewBox="0 0 70 24" width="70" height="24"><polyline fill="none" stroke="${a[a.length-1]>=a[0]?'#34d399':'#f87171'}" stroke-width="1.5" points="${pts}"/></svg>`};
const vmax=Math.max(1,...H.map(p=>p.val)),tb=$('#tb'),d=(c,f)=>F(f)&&f>0&&F(c)?`<small class="${c<f?'pos':'neg'}">${pct((c/f-1)*100,1)}</small>`:'';
tb.innerHTML=P.map((p,i)=>{let r=`<tr data-i="${i}" class="${O[p.t]?'o':''}"><td class="tk">${esc(p.t)}</td><td>${esc(p.n)}<small>${esc(p.s)}${p.exch?' · '+esc(p.exch):''}</small></td><td>${nq.format(p.q)}</td><td>${eur(p.inv)}</td><td>${usd(p.avgB)}</td><td>${usd(p.cur)}${p.live?'':'<small>posl. zatvorenie</small>'}</td><td>${usd(p.fair)}${d(p.cur,p.fair)}</td><td>${usd(p.avg5)}${d(p.cur,p.avg5)}</td><td>${F(p.sell)?usd(p.sell):'N/A'}</td><td class="vb" style="--w:${(p.val/vmax*100).toFixed(0)}%">${eur(p.val)}</td><td class="${cl(p.dayEur)}">${ar(p.dayEur)}${eur(p.dayEur)}<small>${pct(p.dayPct,1)}</small></td><td class="${cl(p.pl)}"><b>${ar(p.pl)}${pct(p.pl,1)}</b><small>${eur(p.plEur)}</small></td><td>${pct(p.yoc)}</td><td>${spark(p.m12)}</td></tr>`;
p.lots.forEach((l,k)=>{r+=`<tr class="lot${O[p.t]?' s':''}" data-p="${i}"><td></td><td>nákup ${k+1}${l.d?' · '+esc(l.d):''}${l.sold?' (predané)':''}</td><td>${nq.format(l.q)}</td><td>${eur(l.inv)}${l.est?'<small>odhad</small>':''}</td><td>${usd(l.b)}</td><td></td><td></td><td></td><td>${l.sold?usd(l.x):''}</td><td>${eur(l.val)}</td><td></td><td class="${cl(l.pl)}">${ar(l.pl)}${pct(l.pl,1)}<small>${eur(l.plEur)}</small></td><td></td><td></td></tr>`});return r}).join('')||'<tr><td colspan="14" style="text-align:center;color:var(--mu)">Portfólio je prázdne. Pridaj nákupy v sekcii „Správa nákupov“ nad dashboardom.</td></tr>';
tb.onclick=e=>{const c=e.target.closest('.tk');if(!c)return;const tr=c.parentNode,p=P[+tr.dataset.i],on=!tr.classList.contains('o');tr.classList.toggle('o',on);tb.querySelectorAll('tr.lot[data-p="'+tr.dataset.i+'"]').forEach(x=>x.classList.toggle('s',on));O[p.t]=on;try{localStorage.setItem('pf4o',JSON.stringify(O))}catch(e){}};
// koláč
const sc=[...new Set(P.map(p=>p.s))],col=(s,l)=>`hsl(${HUE[sc.indexOf(s)%8]} 75% ${l}%)`;
function arc(r0,r1,a0,a1){if(a1-a0>6.283)a1=a0+6.283;const Q=(r,a)=>(200+r*Math.cos(a)).toFixed(2)+','+(200+r*Math.sin(a)).toFixed(2),l=a1-a0>Math.PI?1:0;return`M${Q(r1,a0)}A${r1} ${r1} 0 ${l} 1 ${Q(r1,a1)}L${Q(r0,a1)}A${r0} ${r0} 0 ${l} 0 ${Q(r0,a0)}Z`}
const ch=$('#ch'),tip=$('#tip'),tot=sum(H,p=>p.inv),secs=sc.map(s=>({s,items:H.filter(p=>p.s===s).sort((a,b)=>b.inv-a.inv)})).filter(g=>g.items.length);secs.forEach(g=>g.amt=sum(g.items,p=>p.inv));secs.sort((a,b)=>b.amt-a.amt);
function donut(){const m=$('#dm').value;let a=-Math.PI/2,h='',lg='';
if(m==='a'){H.slice().sort((x,y)=>y.inv-x.inv).forEach(p=>{const sp=p.inv/tot*6.2832;h+=`<path class="seg" data-g="${esc(p.t)}" data-t="${esc(p.t+' · '+p.n)}" data-a="${p.inv}" data-p="${p.inv/tot*100}" fill="${col(p.s,55)}" d="${arc(76,172,a,a+sp)}"/>`;lg+=`<div><i style="background:${col(p.s,55)}"></i>${esc(p.t)}<span>${eur(p.inv)} · ${pct(p.inv/tot*100)}</span></div>`;a+=sp})}
else secs.forEach(g=>{const a0=a;
if(m==='2')g.items.forEach((p,i)=>{const sp=p.inv/tot*6.2832;h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(p.t+' · '+p.n)}" data-a="${p.inv}" data-p="${p.inv/tot*100}" fill="${col(g.s,42+16*i/Math.max(g.items.length-1,1))}" d="${arc(122,172,a,a+sp)}"/>`;a+=sp});else a+=g.amt/tot*6.2832;
h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(g.s)}" data-a="${g.amt}" data-p="${g.amt/tot*100}" fill="${col(g.s,45)}" d="${m==='2'?arc(76,116,a0,a):arc(76,172,a0,a)}"/>`;lg+=`<div><i style="background:${col(g.s,45)}"></i>${esc(g.s)}<span>${eur(g.amt)} · ${pct(g.amt/tot*100)}</span></div>`});
ch.innerHTML=h+`<text x="200" y="192" text-anchor="middle" fill="#8e98b3" font-size="13" id="c1">Investované</text><text x="200" y="218" text-anchor="middle" fill="#eef1f8" font-size="20" font-weight="700" id="c2">${eur(tot)}</text>`;$('#lg').innerHTML=lg||'<div>Žiadne držané pozície</div>'}
ch.onmouseover=ch.onmousemove=e=>{const s=e.target.closest('.seg');if(!s)return;ch.classList.add('h');ch.querySelectorAll('.seg').forEach(x=>x.classList.toggle('on',x===s||x.dataset.g===s.dataset.g));const r=ch.parentNode.getBoundingClientRect();
tip.innerHTML=`<b>${esc(s.dataset.t)}</b><br>Investovaná suma: ${eur(+s.dataset.a)}<br>Podiel z portfólia: ${pct(+s.dataset.p)}`;tip.style.left=Math.min(e.clientX-r.left+14,r.width-200)+'px';tip.style.top=e.clientY-r.top+14+'px';tip.style.opacity=1;$('#c1').textContent=s.dataset.t.split(' · ')[0];$('#c2').textContent=eur(+s.dataset.a)};
ch.onmouseleave=()=>{ch.classList.remove('h');tip.style.opacity=0;$('#c1').textContent='Investované';$('#c2').textContent=eur(tot)};
$('#dm').onchange=donut;donut();
// pruhy
const bv=sc.map(s=>({s,v:sum(H.filter(p=>p.s===s),p=>p.val)})).filter(b=>b.v>0).sort((a,b)=>b.v-a.v),bm=Math.max(1,...bv.map(b=>b.v));
$('#bars').innerHTML=bv.map(b=>`<div class="br"><span>${esc(b.s)}</span><i style="width:${(b.v/bm*100).toFixed(0)}%;background:${col(b.s,50)}"></i><span style="text-align:right">${eur(b.v)}</span></div>`).join('')||'<small>Žiadne držané pozície</small>';
// čiarový graf
const st=$('#st');st.innerHTML=P.map((p,i)=>`<option value="${i}">${esc(p.t)}</option>`).join('');
function line(){const p=P[+st.value],sv=$('#ln'),a=(p&&p.h5)||[];if(a.length<2){sv.innerHTML='';$('#lnn').textContent=p?'História cien nie je dostupná.':'';return}
const t=a.map(x=>Date.parse(x[0])),y=a.map(x=>x[1]),t0=t[0],t1=t[t.length-1],mn=Math.min(...y),mx=Math.max(...y),X=v=>(v-t0)/(t1-t0)*780+10,Y=v=>260-(v-mn)/((mx-mn)||1)*240;
let s=`<polyline fill="none" stroke="#f97316" stroke-width="1.6" vector-effect="non-scaling-stroke" points="${a.map((x,i)=>X(t[i]).toFixed(1)+','+Y(y[i]).toFixed(1)).join(' ')}"/>`;
p.lots.forEach(l=>{if(!l.d)return;const tx=Date.parse(l.d);if(!(tx>=t0&&tx<=t1))return;s+=`<line x1="${X(tx)}" x2="${X(tx)}" y1="10" y2="262" stroke="#34d399" stroke-dasharray="4 3" vector-effect="non-scaling-stroke"><title>Nákup ${esc(l.d)}: ${nq.format(l.q)} ks @ ${usd(l.b)}</title></line>`});
sv.innerHTML=s;const nd=p.lots.filter(l=>!l.d).length;$('#lnn').textContent='Rozsah '+usd(mn)+' – '+usd(mx)+' · '+a[0][0]+' až '+a[a.length-1][0]+' · zelené čiary = dátumy nákupov'+(nd?' ('+nd+' nákupov bez dátumu)':'')}
st.onchange=line;line();
</script></body></html>
"""


def dashboard(lots):
    if lots.empty:
        data = dict(pos=[], fx=None, ts=datetime.now().strftime("%H:%M"))
    else:
        syms = tuple(sorted({ysym(t) for t in lots["Ticker"] if t}))
        quotes, hist = get_quotes(syms), get_hist(syms)
        ccys = tuple(sorted({("GBP" if q.get("cur") == "GBp" else (q.get("cur") or "USD")) for q in quotes.values()}))
        rates = get_rates(ccys + ("USD",))
        data = dict(pos=build(lots, quotes, hist, rates), fx=rates.get("USD"), ts=datetime.now().strftime("%H:%M:%S"))
    blob = json.dumps(clean(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    components.html(TEMPLATE.replace("__DATA__", blob), height=1750, scrolling=True)


with st.sidebar:
    st.header("⚙️ Nastavenia")
    refresh = REFRESH[st.selectbox("Automatické obnovovanie cien", list(REFRESH), index=2, key="pf_refresh")]
    if st.button("🔄 Obnoviť ceny teraz", key="pf_reload"):
        st.cache_data.clear()
        st.rerun()

if "pf_base" not in st.session_state:
    try:
        st.session_state["pf_base"] = normalize(pd.read_csv(FILE)) if os.path.exists(FILE) else normalize(None)
    except Exception:
        st.session_state["pf_base"] = normalize(None)
    st.session_state["pf_ver"] = 0

up = st.file_uploader("Nahrať zálohu (CSV)", type="csv", key="pf_up")
if up is not None and st.session_state.get("pf_sig") != (up.name, up.size):
    try:
        st.session_state["pf_base"] = normalize(pd.read_csv(up))
        st.session_state["pf_ver"] += 1
        st.session_state["pf_sig"] = (up.name, up.size)
    except Exception as e:
        st.error(f"Zálohu sa nepodarilo načítať: {e}")

with st.expander("➕ Správa nákupov (pridať / upraviť / odstrániť)", expanded=st.session_state["pf_base"].empty):
    st.caption("Jeden riadok = jeden nákup. Ticker môžeš zadať aj v tvare XTB (napr. BRKB.US). Viac nákupov jednej akcie "
               "= viac riadkov s rovnakým tickerom. „Vložené €“ je suma v eurách (v XTB hodnota mínus zisk). "
               "„Dátum nákupu“ sa použije na značky v grafe ceny. Pre predaný nákup vyplň „Predajná cena“. Riadok odstrániš označením a klávesou Delete.")
    edited = st.data_editor(
        st.session_state["pf_base"], num_rows="dynamic", hide_index=True, use_container_width=True,
        key=f"pf_editor_{st.session_state['pf_ver']}",
        column_config={
            "Dátum nákupu": st.column_config.DateColumn("Dátum nákupu"),
            "Kusy": st.column_config.NumberColumn(format="%.4f", min_value=0.0),
            "Cena nákupu": st.column_config.NumberColumn("Cena nákupu ($)", format="%.2f", min_value=0.0),
            "Vložené €": st.column_config.NumberColumn("Vložené (€)", format="%.2f", min_value=0.0),
            "Férová cena": st.column_config.NumberColumn("Férová cena ($)", format="%.2f", min_value=0.0),
            "Predajná cena": st.column_config.NumberColumn("Predajná cena ($)", format="%.2f", min_value=0.0),
        })
lots = normalize(edited)
try:
    lots.to_csv(FILE, index=False)
except Exception:
    pass
st.download_button("⬇️ Stiahnuť zálohu (CSV)", lots.to_csv(index=False).encode("utf-8-sig"), file_name="portfolio_lots.csv",
                   mime="text/csv", key="pf_dl")
st.caption("⚠️ Súbor v aplikácii sa môže po reštarte stratiť: zálohu si pravidelne sťahuj a po reštarte nahraj.")

frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if refresh and frag is not None:
    try:
        frag(run_every=refresh)(dashboard)(lots)
    except TypeError:
        dashboard(lots)
else:
    dashboard(lots)
