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
COLS = ["Ticker", "Názov", "Sektor", "Kusy", "Cena nákupu", "Vložené €", "Férová cena", "Predajná cena"]
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
    return df[~((df["Ticker"] == "") & df[NUM].isna().all(axis=1))].reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def get_quotes(symbols: tuple) -> dict:
    def one(s):
        out = dict(price=np.nan, cur="")
        try:
            fi = yf.Ticker(s).fast_info
            out["price"], out["cur"] = sf(fi["last_price"]), str(fi["currency"] or "")
        except Exception:
            pass
        return s, out

    with ThreadPoolExecutor(max_workers=5) as ex:
        return dict(ex.map(one, symbols))


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_hist(symbols: tuple) -> dict:
    def one(s):
        out = dict(avg5=np.nan, last=np.nan, dps=np.nan)
        try:
            h = yf.Ticker(s).history(period="5y", auto_adjust=False, actions=True)
            c = h["Close"].dropna()
            if len(c):
                out.update(avg5=float(c.mean()), last=float(c.iloc[-1]))
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
                                              avg5=sf(hs.get("avg5")) * sc, dps=sf(hs.get("dps")) * sc, lots=[]))
        p["n"] = r["Názov"] or p["n"] or r["Ticker"]
        p["s"] = r["Sektor"] or p["s"] or "Nezaradené"
        if np.isfinite(r["Férová cena"]) and r["Férová cena"] > 0:
            p["fair"] = r["Férová cena"]
        est = not (np.isfinite(r["Vložené €"]) and r["Vložené €"] > 0)
        inv = q * b / rate if est and np.isfinite(rate) else r["Vložené €"]
        sold = np.isfinite(r["Predajná cena"]) and r["Predajná cena"] > 0
        val = q * (r["Predajná cena"] if sold else price) / rate if np.isfinite(rate) else np.nan
        p["lots"].append(dict(q=q, b=b, inv=inv, est=est, sold=sold, x=r["Predajná cena"] if sold else np.nan,
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
            yoc=p["dps"] / avgB * 100 if np.isfinite(p["dps"]) and p["dps"] > 0 else np.nan, lots=L))
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
.ic{background:none;border:1px solid var(--line);color:var(--mut);border-radius:6px;padding:2px 8px;margin-left:4px;cursor:pointer;font:inherit}.ic.x{color:var(--neg);border-color:var(--neg)}
.tk{cursor:pointer;font-weight:700}.tk::before{content:"▸ ";color:var(--mut)}.o .tk::before{content:"▾ "}tr.lot td{color:var(--mut);font-size:12px;background:var(--bg)}tr.lot{display:none}tr.lot.s{display:table-row}
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
@media (max-width:640px){table,tbody,tr,td{display:block;min-width:0}tr.lot.s{display:block}thead{display:none}tr{padding:10px 0;border-bottom:1px solid var(--line)}td{display:flex;justify-content:space-between;border:0;padding:3px 0}td::before{content:attr(data-l);color:var(--mut)}td small{display:none}}
</style>
</head>
<body>
<header><h1>📈 Portfólio akcií</h1><div><span id="ts" style="color:var(--mut);font-size:12px"></span></div></header>
<main>
<section class="kpis">
<div class="card kpi"><small>Celková hodnota portfólia</small><b id="k1">–</b><small id="k1s"></small></div>
<div class="card kpi"><small>Celkový zisk / strata</small><b id="k2">–</b><small id="k2s"></small></div>
<div class="card kpi"><small>Najvýnosnejší sektor</small><b id="k3">–</b><small id="k3s"></small></div>
</section>
<section class="card"><h2>Rozloženie investovaného kapitálu: vnútorný kruh sektory, vonkajší akcie</h2>
<div class="donut"><div class="dw"><svg id="ch" viewBox="0 0 400 400" role="img" aria-label="Dvojvrstvový kruhový graf portfólia"></svg><div id="tip"></div></div><div class="lg" id="lg"></div></div></section>
<section class="card"><h2>Pozície</h2><div class="tw"><table><thead><tr><th>Ticker</th><th>Firma a sektor</th><th>Aktuálna cena</th><th>Férová cena</th><th>Priemer 5 rokov</th><th>Nákupná cena (priem.)</th><th>Predajná cena</th><th>Vložené</th><th>Zisk / strata</th><th>Yield on Cost</th></tr></thead><tbody id="tb"></tbody></table></div>
<p style="color:var(--mut);font-size:12px;margin:12px 0 0">Ceny sa sťahujú automaticky z Yahoo Finance (môžu meškať). Zisk a strata sú v EUR vrátane zmeny kurzu, ako v XTB. Klikni na ticker pre jednotlivé nákupy.</p></section>
</main>
<script>
const D=__DATA__,NB='\u00A0',nf=new Intl.NumberFormat('sk-SK',{minimumFractionDigits:2,maximumFractionDigits:2}),nq=new Intl.NumberFormat('sk-SK',{maximumFractionDigits:4});
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const F=v=>Number.isFinite(v),usd=v=>F(v)?nf.format(v)+NB+'$':'–',eur=v=>F(v)?nf.format(v)+NB+'€':'–',pct=(v,s)=>F(v)?(s&&v>0?'+':'')+nf.format(v)+NB+'%':'–',cls=v=>v>=0?'pos':'neg';
let O={};try{O=JSON.parse(localStorage.getItem('pf3o'))||{}}catch(e){}
const P=D.pos,H=P.filter(p=>p.held),HUE=[226,160,28,330,270,190,50,0];
const sum=(a,f)=>a.reduce((x,p)=>x+(f(p)||0),0);
function arc(r0,r1,a0,a1){if(a1-a0>6.283)a1=a0+6.283;const Q=(r,a)=>(200+r*Math.cos(a)).toFixed(2)+','+(200+r*Math.sin(a)).toFixed(2),l=a1-a0>Math.PI?1:0;return`M${Q(r1,a0)}A${r1} ${r1} 0 ${l} 1 ${Q(r1,a1)}L${Q(r0,a1)}A${r0} ${r0} 0 ${l} 0 ${Q(r0,a0)}Z`}
$('#ts').textContent='Ceny z '+D.ts+' · EUR/USD '+(D.fx?nf.format(D.fx):'–');
$('#k1').textContent=eur(sum(H,p=>p.val));$('#k1s').textContent=H.length+' držaných pozícií';
const iA=sum(P,p=>p.invAll),vA=sum(P,p=>p.valAll),pl=iA?(vA/iA-1)*100:NaN;
$('#k2').textContent=pct(pl,1);$('#k2').className=F(pl)?cls(pl):'';$('#k2s').textContent=eur(vA-iA)+' (v EUR, vrátane kurzu)';
const sc=[...new Set(P.map(p=>p.s))];let best=null;
sc.forEach(s=>{const g=P.filter(p=>p.s===s),r=(sum(g,p=>p.valAll)/sum(g,p=>p.invAll)-1)*100;if(F(r)&&(!best||r>best.r))best={s,r}});
$('#k3').textContent=best?best.s:'–';$('#k3s').textContent=best?pct(best.r,1):'';
const hi=sum(H,p=>p.inv),secs=sc.map((s,i)=>({s,h:HUE[i%8],items:H.filter(p=>p.s===s).sort((a,b)=>b.inv-a.inv)})).filter(g=>g.items.length);
secs.forEach(g=>g.amt=sum(g.items,p=>p.inv));secs.sort((a,b)=>b.amt-a.amt);
let a=-Math.PI/2,h='',lg='';
secs.forEach(g=>{const a0=a;
g.items.forEach((p,i)=>{const sp=p.inv/hi*6.2832,l=52+14*i/Math.max(g.items.length-1,1);h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(p.t+' · '+p.n)}" data-a="${p.inv}" data-p="${p.inv/hi*100}" fill="hsl(${g.h} 70% ${l}%)" d="${arc(122,172,a,a+sp)}"/>`;a+=sp});
h+=`<path class="seg" data-g="${esc(g.s)}" data-t="${esc(g.s)}" data-a="${g.amt}" data-p="${g.amt/hi*100}" fill="hsl(${g.h} 70% 45%)" d="${arc(76,116,a0,a)}"/>`;
lg+=`<div><i style="background:hsl(${g.h} 70% 45%)"></i>${esc(g.s)}<span>${eur(g.amt)} · ${pct(g.amt/hi*100)}</span></div>`});
const ch=$('#ch');ch.innerHTML=h+`<text x="200" y="192" text-anchor="middle" fill="var(--mut)" font-size="13" id="c1">Investované</text><text x="200" y="218" text-anchor="middle" fill="var(--text)" font-size="20" font-weight="700" id="c2">${eur(hi)}</text>`;
$('#lg').innerHTML=lg||'<div>Žiadne držané pozície</div>';
const tb=$('#tb');
tb.innerHTML=P.map((p,i)=>{const d=(c,f)=>F(f)&&f>0&&F(c)?`<small class="${c<f?'pos':'neg'}">${pct((c/f-1)*100,1)}</small>`:'';
let r=`<tr data-i="${i}" class="${O[p.t]?'o':''}"><td data-l="Ticker" class="tk">${esc(p.t)}</td><td data-l="Firma"><span>${esc(p.n)}</span><small>${esc(p.s)}</small></td><td data-l="Aktuálna cena">${usd(p.cur)}${p.live?'':'<small>posl. zatvorenie</small>'}</td><td data-l="Férová cena">${usd(p.fair)}${d(p.cur,p.fair)}</td><td data-l="Priemer 5 rokov">${usd(p.avg5)}${d(p.cur,p.avg5)}</td><td data-l="Nákupná cena">${usd(p.avgB)}</td><td data-l="Predajná cena">${F(p.sell)?usd(p.sell):'N/A'}</td><td data-l="Vložené">${eur(p.inv)}<small>${nq.format(p.q)} ks</small></td><td data-l="Zisk / strata" class="${F(p.pl)?cls(p.pl):''}"><b>${F(p.pl)?(p.pl>=0?'▲ ':'▼ '):''}${pct(p.pl,1)}</b><small>${eur(p.plEur)}</small></td><td data-l="Yield on Cost">${pct(p.yoc)}</td></tr>`;
p.lots.forEach((l,k)=>{r+=`<tr class="lot${O[p.t]?' s':''}" data-p="${i}"><td></td><td>nákup ${k+1}${l.sold?' (predané)':''}</td><td></td><td></td><td></td><td>${usd(l.b)}</td><td>${l.sold?usd(l.x):''}</td><td>${eur(l.inv)}<small>${nq.format(l.q)} ks${l.est?' · odhad':''}</small></td><td class="${F(l.pl)?cls(l.pl):''}">${pct(l.pl,1)}<small>${eur(l.plEur)}</small></td><td></td></tr>`});
return r}).join('')||'<tr><td colspan="10" style="text-align:center;color:var(--mut)">Portfólio je prázdne. Pridaj nákupy v sekcii „Správa nákupov“ nad dashboardom.</td></tr>';
tb.onclick=e=>{const c=e.target.closest('.tk');if(!c)return;const tr=c.parentNode,p=P[+tr.dataset.i],on=!tr.classList.contains('o');tr.classList.toggle('o',on);
tb.querySelectorAll('tr.lot[data-p="'+tr.dataset.i+'"]').forEach(x=>x.classList.toggle('s',on));O[p.t]=on;try{localStorage.setItem('pf3o',JSON.stringify(O))}catch(e){}};
const tip=$('#tip'),w=ch.parentNode,c1=$('#c1'),c2=$('#c2'),b0=eur(hi);
ch.onmouseover=ch.onmousemove=e=>{const s=e.target.closest('.seg');if(!s)return;ch.classList.add('h');ch.querySelectorAll('.seg').forEach(x=>x.classList.toggle('on',x===s||x.dataset.g===s.dataset.g&&(s.dataset.t===s.dataset.g||x.dataset.t===x.dataset.g)));
const r=w.getBoundingClientRect();tip.innerHTML=`<b>${esc(s.dataset.t)}</b><br>Investovaná suma: ${eur(+s.dataset.a)}<br>Podiel z portfólia: ${pct(+s.dataset.p)}`;tip.style.left=Math.min(e.clientX-r.left+14,r.width-190)+'px';tip.style.top=e.clientY-r.top+14+'px';tip.style.opacity=1;c1.textContent=s.dataset.t.split(' · ')[0];c2.textContent=eur(+s.dataset.a)};
ch.onmouseleave=()=>{ch.classList.remove('h');tip.style.opacity=0;c1.textContent='Investované';c2.textContent=b0};
</script>
</body>
</html>
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
    components.html(TEMPLATE.replace("__DATA__", blob), height=1500, scrolling=True)


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
               "Pre predaný nákup vyplň „Predajná cena“. Riadok odstrániš označením a klávesou Delete.")
    edited = st.data_editor(
        st.session_state["pf_base"], num_rows="dynamic", hide_index=True, use_container_width=True,
        key=f"pf_editor_{st.session_state['pf_ver']}",
        column_config={
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
