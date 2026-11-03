"""
Portfólio dashboard (pages/2_Watchlist.py)
- Nákupy zadávaš v sekcii „Správa nákupov“ (viac nákupov jednej akcie na rôznych cenách).
- Aktuálne ceny, 5-ročný priemer a dividenda sa sťahujú automaticky z Yahoo Finance (cez yfinance).
- Zisk a strata sa počítajú v EUR vrátane zmeny kurzu EUR/USD (ako v XTB).
- Dáta sa ukladajú do portfolio_lots.csv v aplikácii (po reštarte sa môžu stratiť): používaj zálohu (CSV).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

st.set_page_config(page_title="Portfólio", page_icon="📈", layout="wide")

FILE = "portfolio_lots.csv"
COLS = ["Ticker", "Názov", "Sektor", "Krajina", "Typ", "Dátum nákupu", "Kusy", "Cena nákupu", "Vložené €", "Férová cena", "Predajná cena"]
NUM = ["Kusy", "Cena nákupu", "Vložené €", "Férová cena", "Predajná cena"]
ASSET_TYPES = ["Akcia", "ETF", "Krypto", "Iné"]
WCOLS = ["Ticker", "Férová cena", "MoS %", "Poznámka"]
WFILE, SFILE = "portfolio_watch.csv", "portfolio_settings.json"
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
    for c in ("Ticker", "Názov", "Sektor", "Krajina", "Typ"):
        df[c] = df[c].fillna("").astype(str).str.strip()
    df["Ticker"] = df["Ticker"].str.upper()
    df["Typ"] = df["Typ"].where(df["Typ"].isin(ASSET_TYPES), "Akcia")
    for c in NUM:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Dátum nákupu"] = pd.to_datetime(df["Dátum nákupu"], errors="coerce")
    return df[~((df["Ticker"] == "") & df[NUM].isna().all(axis=1))].reset_index(drop=True)


@st.cache_resource
def _store():
    return {}


def _cached(store, key, ttl, fn, ok):
    """Cache len pre ÚSPEŠNÉ výsledky: výpadok Yahoo sa nezapamätá a nabudúce sa skúsi znova."""
    hit = store.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    val = fn()
    if ok(val):
        store[key] = (time.time(), val)
    return val


def _quote(s):
    out = dict(price=np.nan, cur="", prev=np.nan, exch="")
    t = yf.Ticker(s)
    for attempt in range(2):
        try:
            fi = t.fast_info
            out["price"], out["cur"] = sf(fi["last_price"]), str(fi["currency"] or "")
            for k, key in (("prev", "previous_close"), ("exch", "exchange")):
                try:
                    out[k] = sf(fi[key]) if k == "prev" else str(fi[key] or "")
                except Exception:
                    pass
        except Exception:
            pass
        if np.isfinite(out["price"]):
            break
        time.sleep(0.4)
    if not np.isfinite(out["price"]):          # záloha: posledné zatvorenie z krátkej histórie
        try:
            c = t.history(period="5d")["Close"].dropna()
            if len(c):
                out["price"] = float(c.iloc[-1])
                out["prev"] = float(c.iloc[-2]) if len(c) > 1 else np.nan
                out["fallback"] = True
        except Exception:
            pass
    return out


def _hist(s):
    out = dict(avg5=np.nan, last=np.nan, dps=np.nan, h5=[], m12=[], divs=[])
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
            mask = idx >= pd.Timestamp.now() - pd.Timedelta(days=365)
            out["dps"] = float(h["Dividends"][mask].sum())
            out["divs"] = [[d.strftime("%Y-%m-%d"), float(v)] for d, v in zip(idx[mask], h["Dividends"][mask]) if v > 0]
    except Exception:
        pass
    return out


def get_quotes(symbols: tuple) -> dict:
    store = _store()
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(ex.map(lambda s: (s, _cached(store, ("q", s), 60, lambda: _quote(s),
                                                 lambda v: np.isfinite(v["price"]))), symbols))


def get_hist(symbols: tuple) -> dict:
    store = _store()
    with ThreadPoolExecutor(max_workers=4) as ex:
        return dict(ex.map(lambda s: (s, _cached(store, ("h", s), 3 * 3600, lambda: _hist(s),
                                                 lambda v: np.isfinite(v["avg5"]))), symbols))


def get_rates(ccys: tuple) -> dict:
    """Koľko jednotiek meny je 1 EUR (EUR → 1)."""
    store, out = _store(), {"EUR": 1.0}
    for c in ccys:
        if c and c != "EUR":
            def one(c=c):
                try:
                    return sf(yf.Ticker(f"EUR{c}=X").fast_info["last_price"])
                except Exception:
                    return np.nan
            out[c] = _cached(store, ("r", c), 300, one, np.isfinite)
    return out


def tax(dd):
    """Časový test 1 rok od dátumu nákupu (orientačne)."""
    if pd.isna(dd):
        return dict(free=False, freeOn=None, left=None)
    free_on, today = dd + pd.DateOffset(years=1), pd.Timestamp.now().normalize()
    return dict(free=bool(today > free_on), freeOn=free_on.strftime("%d.%m.%Y"), left=int((free_on - today).days))


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
                                              prev=sf(qu.get("prev")) * sc, exch=qu.get("exch", ""), ccy=ccy, rate=rate, fb=bool(qu.get("fallback")),
                                              typ=r["Typ"], country=r["Krajina"], divs=[[d, a * sc] for d, a in hs.get("divs", [])],
                                              h5=[[d, v * sc] for d, v in hs.get("h5", [])],
                                              m12=[v * sc for v in hs.get("m12", [])]))
        p["n"] = r["Názov"] or p["n"] or r["Ticker"]
        p["s"] = r["Sektor"] or p["s"] or "Nezaradené"
        p["country"] = r["Krajina"] or p["country"]
        if np.isfinite(r["Férová cena"]) and r["Férová cena"] > 0:
            p["fair"] = r["Férová cena"]
        est = not (np.isfinite(r["Vložené €"]) and r["Vložené €"] > 0)
        inv = q * b / rate if est and np.isfinite(rate) else r["Vložené €"]
        sold = np.isfinite(r["Predajná cena"]) and r["Predajná cena"] > 0
        val = q * (r["Predajná cena"] if sold else price) / rate if np.isfinite(rate) else np.nan
        p["lots"].append(dict(**tax(r["Dátum nákupu"]), d=r["Dátum nákupu"].strftime("%Y-%m-%d") if pd.notna(r["Dátum nákupu"]) else None,
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
        dm = [0.0] * 12
        if held and np.isfinite(p["rate"]):
            for d_, a_ in p["divs"]:
                dm[int(d_[5:7]) - 1] += q * a_ / p["rate"]
        out.append(dict(
            divm=dm, typ=p["typ"], country=p["country"] or "Nezaradené",
            t=p["t"], n=p["n"], s=p["s"], fair=p["fair"], cur=p["cur"], live=p["live"], avg5=p["avg5"], q=q, avgB=avgB,
            sell=(sum(l["q"] * l["x"] for l in sold) / sum(l["q"] for l in sold)) if sold else np.nan,
            inv=inv if held else 0.0, held=bool(held), val=val if held else 0.0,
            invAll=sum(l["inv"] for l in L), valAll=sum(l["val"] for l in L),
            pl=(val / inv - 1) * 100 if inv and np.isfinite(val) else np.nan, plEur=val - inv,
            yoc=p["dps"] / avgB * 100 if np.isfinite(p["dps"]) and p["dps"] > 0 else np.nan, lots=L,
            exch=p["exch"], ccy=p["ccy"], h5=p["h5"], m12=p["m12"], rate=p["rate"],
            chk=(sum(l["q"] * l["b"] for l in base) / inv) / p["rate"] if inv and np.isfinite(p["rate"]) else np.nan,
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
:root{--bg:#fff;--pn:#f8f9fa;--tx:#0f172a;--mu:#64748b;--ln:#e5e7eb;--pos:#10b981;--neg:#ef4444;--bl:#2563eb}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:13px/1.45 Inter,system-ui,-apple-system,"Segoe UI",sans-serif;font-variant-numeric:tabular-nums}
.pos{color:var(--pos)}.neg{color:var(--neg)}small{display:block;color:var(--mu);font-size:11px}
.wrap{max-width:1500px;margin:0 auto;padding:14px;display:grid;gap:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px}
.card{background:var(--pn);border:1px solid var(--ln);border-radius:14px;padding:16px;min-width:0}
.kpi small{font-size:12px}.kpi b{display:block;font-size:28px;letter-spacing:-.5px;line-height:1.2;margin:3px 0}
.card h2{margin:0 0 10px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mu);font-weight:600;display:flex;justify-content:space-between;gap:8px;align-items:center}
select{background:#fff;color:var(--tx);border:1px solid var(--ln);border-radius:8px;padding:4px 8px;font:inherit}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;min-width:1250px}.t2{min-width:760px}
th,td{padding:9px 7px;text-align:right;border-bottom:1px solid var(--ln);white-space:nowrap}th{color:var(--mu);font-weight:500;font-size:11px}
th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}.tk{cursor:pointer;font-weight:700}.tk::before{content:"▸ ";color:var(--mu)}.o .tk::before{content:"▾ "}
tr.lot{display:none}tr.lot.s{display:table-row}tr.lot td{color:var(--mu);font-size:12px;background:#fff}
.vb{background:linear-gradient(90deg,rgba(37,99,235,.14) var(--w),transparent var(--w))}
.bd{display:inline-block;padding:3px 10px;border-radius:999px;font-weight:700;font-size:12px}.g{background:rgba(16,185,129,.15);color:#047857}.r{background:#eef0f3;color:var(--mu)}.t{background:rgba(37,99,235,.12);color:var(--bl)}
.alert{border-color:var(--pos);background:rgba(16,185,129,.08)}.warn{border-color:var(--neg);color:var(--neg);background:rgba(239,68,68,.06)}
.g2{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:14px}.wide{grid-column:1/-1}
.dw{position:relative;max-width:360px;margin:0 auto}svg{width:100%;height:auto;display:block}
.seg{stroke:#fff;stroke-width:2;transition:opacity .15s}svg.h .seg:not(.on){opacity:.25}
#tip{position:absolute;pointer-events:none;background:var(--tx);color:#fff;padding:7px 9px;border-radius:8px;font-size:12px;opacity:0;white-space:nowrap;z-index:5}
.br{display:grid;grid-template-columns:110px 1fr 90px;gap:8px;align-items:center;margin:7px 0}.br i{display:block;height:13px;border-radius:4px}
.lg div{display:flex;gap:8px;align-items:center;padding:3px 0}.lg i{width:9px;height:9px;border-radius:50%}.lg span{margin-left:auto;color:var(--mu)}
.mo{display:grid;grid-template-columns:repeat(6,1fr);gap:6px}.mo div{background:#fff;border:1px solid var(--ln);border-radius:8px;padding:6px;text-align:center}.mo b{display:block;font-size:12px}
</style></head><body><div class="wrap">
<div class="kpis">
<div class="card kpi"><small>Celková hodnota majetku</small><b id="a1">–</b><small id="a1s"></small></div>
<div class="card kpi"><small>Celkový zisk / strata</small><b id="a2">–</b><small id="a2s"></small></div>
<div class="card kpi"><small>Denná zmena</small><b id="a3">–</b><small id="a3s"></small></div>
<div class="card kpi"><small>Ročná návratnosť (XIRR)</small><b id="a4">–</b><small id="a4s"></small></div></div>
<div id="wn"></div>
<div class="card"><h2>Otvorené pozície <span style="text-transform:none;letter-spacing:0">klikni na ticker pre jednotlivé nákupy · <span id="ts"></span></span></h2><div class="tw"><table><thead><tr><th>Ticker</th><th>Firma · sektor · burza</th><th>Počet</th><th>Vložené</th><th>Nákup. cena (priem.)</th><th>Aktuálna cena</th><th>Hodnota</th><th>Váha</th><th>Dnes</th><th>Zisk / strata</th><th>Férová cena</th><th>Priemer 5 r.</th><th>Predajná</th><th>Yield on Cost</th><th>12 mesiacov</th></tr></thead><tbody id="tb"></tbody></table></div></div>
<div class="card"><h2>Čakáreň na férovú cenu <span style="text-transform:none;letter-spacing:0" id="wsum"></span></h2><div class="tw"><table class="t2"><thead><tr><th>Ticker</th><th>Poznámka</th><th>Aktuálna cena</th><th>Férová cena</th><th>Margin of Safety</th><th>Cieľová nákupná cena</th><th>Vzdialenosť od nákupu</th><th>Signál</th></tr></thead><tbody id="wb"></tbody></table></div></div>
<div class="g2">
<div class="card"><h2>Alokácia <select id="dm"><option value="t">Typ aktíva</option><option value="2">Sektory a akcie</option><option value="a">Akcie</option></select></h2><div class="dw"><svg id="ch" viewBox="0 0 400 400"></svg><div id="tip"></div></div><div class="lg" id="lg"></div></div>
<div class="card"><h2>Sektory</h2><div id="bs"></div><h2 style="margin-top:18px">Geografia</h2><div id="bg"></div></div>
<div class="card wide"><h2>Vývoj majetku vs. S&amp;P 500 (rovnaké vklady v rovnakých dňoch)<span id="cl" style="text-transform:none;letter-spacing:0"></span></h2><svg id="eq" viewBox="0 0 900 260" preserveAspectRatio="none" style="height:250px"></svg><small id="eqn"></small></div>
<div class="card"><h2>Dividendový kalendár <span id="dv" style="text-transform:none;letter-spacing:0"></span></h2><div class="mo" id="mo"></div><small>Odhad podľa dividend za posledných 12 mesiacov (mesiac ex-dátumu), v EUR.</small></div>
<div class="card"><h2>Daňový tracker (časový test 1 rok)</h2><div id="tx"></div><small>Orientačné: počíta čas od dátumu nákupu. Pravidlá si over, nie je to daňové poradenstvo.</small></div>
</div></div>
<script>
const D=__DATA__,P=D.pos,H=P.filter(p=>p.held),W=D.watch,NB='\u00A0',F=Number.isFinite;
const nf=new Intl.NumberFormat('sk-SK',{minimumFractionDigits:2,maximumFractionDigits:2}),nq=new Intl.NumberFormat('sk-SK',{maximumFractionDigits:4});
const $=s=>document.querySelector(s),esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const eur=v=>F(v)?nf.format(v)+NB+'€':'–',pct=(v,s)=>F(v)?(s&&v>0?'+':'')+nf.format(v)+NB+'%':'–',cl=v=>F(v)?(v>=0?'pos':'neg'):'',ar=v=>F(v)?(v>=0?'▲ ':'▼ '):'';
const CS={USD:'$',EUR:'€',GBP:'£',CHF:'CHF',CZK:'Kč'},px=(v,p)=>F(v)?nf.format(v)+NB+(CS[p.ccy]||p.ccy||'$'):'–';
const sum=(a,f)=>a.reduce((x,p)=>x+(F(f(p))?f(p):0),0),HUE=[221,160,25,330,270,190,45,0];
let O={};try{O=JSON.parse(localStorage.getItem('pf5o'))||{}}catch(e){}
const V=P.filter(p=>F(p.valAll)&&F(p.invAll)),HV=H.filter(p=>F(p.val)),miss=H.filter(p=>!F(p.val)),bad=H.filter(p=>F(p.chk)&&Math.abs(p.chk-1)>.08);
const iA=sum(V,p=>p.invAll),vA=sum(V,p=>p.valAll),dT=sum(HV,p=>p.dayEur),vH=sum(HV,p=>p.val),tot=vH+D.cash;
$('#a1').textContent=eur(tot);$('#a1s').textContent='akcie a ETF '+eur(vH)+' · hotovosť '+eur(D.cash);
$('#a2').textContent=eur(vA-iA);$('#a2').className=cl(vA-iA);$('#a2s').textContent=ar(vA-iA)+pct(iA?(vA/iA-1)*100:NaN,1)+' (v EUR vrátane kurzu)';
$('#a3').textContent=eur(dT);$('#a3').className=cl(dT);$('#a3s').textContent=ar(dT)+pct(vH-dT?dT/(vH-dT)*100:NaN,1)+' voči včerajšiemu zatvoreniu';
$('#a4').textContent=F(D.xirr)?pct(D.xirr,1):'–';$('#a4').className=cl(D.xirr);$('#a4s').textContent=F(D.xirr)?'ročne, vážená vkladmi (nie TWR)':'treba nákupy s dátumom staršie ako 6 mesiacov';
$('#ts').textContent='ceny z '+D.ts+' · EUR/USD '+(D.fx?nf.format(D.fx):'–');
const wn=[];W.filter(w=>w.sig).forEach(w=>wn.push(['alert','🟩 <b>NAKUPUJ: '+esc(w.t)+'</b> – cena '+nf.format(w.price)+' je na alebo pod cieľom '+nf.format(w.trig)]));
if(miss.length)wn.push(['warn','⚠️ Chýba cena pre '+miss.map(p=>esc(p.t)).join(', ')+': pozícia sa nezapočítala do hodnoty ani zisku. Skús „Obnoviť ceny teraz“.']);
bad.forEach(p=>wn.push(['warn','⚠️ Pri '+esc(p.t)+' nesedí „Vložené €“ s počtom kusov a nákupnou cenou (zodpovedalo by kurzu '+nf.format(p.chk*p.rate)+' namiesto dnešných '+nf.format(p.rate)+'). Zadaj sumu, ktorú si vložil, nie aktuálnu hodnotu.']));
$('#wn').innerHTML=wn.map(x=>`<div class="card ${x[0]}" style="margin-bottom:10px">${x[1]}</div>`).join('');
const spark=a=>{if(!a||a.length<2)return'';const mn=Math.min(...a),mx=Math.max(...a),r=mx-mn||1;return`<svg viewBox="0 0 70 24" width="70" height="24"><polyline fill="none" stroke="${a[a.length-1]>=a[0]?'#10b981':'#ef4444'}" stroke-width="1.5" points="${a.map((v,i)=>(i/(a.length-1)*70).toFixed(1)+','+(22-(v-mn)/r*20).toFixed(1)).join(' ')}"/></svg>`};
const vmax=Math.max(1,...H.map(p=>F(p.val)?p.val:0)),tb=$('#tb'),d=(c,f)=>F(f)&&f>0&&F(c)?`<small class="${c<f?'pos':'neg'}">${pct((c/f-1)*100,1)}</small>`:'';
tb.innerHTML=P.map((p,i)=>{let r=`<tr data-i="${i}" class="${O[p.t]?'o':''}"><td class="tk">${esc(p.t)}</td><td>${esc(p.n)}<small>${esc(p.s)}${p.exch?' · '+esc(p.exch):''}</small></td><td>${nq.format(p.q)}</td><td>${eur(p.inv)}</td><td>${px(p.avgB,p)}</td><td>${px(p.cur,p)}${p.live&&!p.fb?'':'<small>posl. zatvorenie</small>'}</td><td class="vb" style="--w:${(p.val/vmax*100||0).toFixed(0)}%">${eur(p.val)}</td><td>${F(p.val)&&tot?pct(p.val/tot*100):'–'}</td><td class="${cl(p.dayEur)}">${ar(p.dayEur)}${eur(p.dayEur)}<small>${pct(p.dayPct,1)}</small></td><td class="${cl(p.pl)}"><b>${ar(p.pl)}${pct(p.pl,1)}</b><small>${eur(p.plEur)}</small></td><td>${px(p.fair,p)}${d(p.cur,p.fair)}</td><td>${px(p.avg5,p)}${d(p.cur,p.avg5)}</td><td>${F(p.sell)?px(p.sell,p):'N/A'}</td><td>${pct(p.yoc)}</td><td>${spark(p.m12)}</td></tr>`;
p.lots.forEach((l,k)=>{r+=`<tr class="lot${O[p.t]?' s':''}" data-p="${i}"><td></td><td>nákup ${k+1}${l.d?' · '+esc(l.d):''}${l.sold?' (predané)':''}</td><td>${nq.format(l.q)}</td><td>${eur(l.inv)}${l.est?'<small>odhad</small>':''}</td><td>${px(l.b,p)}</td><td></td><td>${eur(l.val)}</td><td></td><td></td><td class="${cl(l.pl)}">${ar(l.pl)}${pct(l.pl,1)}<small>${eur(l.plEur)}</small></td><td></td><td></td><td>${l.sold?px(l.x,p):''}</td><td></td><td>${l.d&&!l.sold?(l.free?'<span class="bd g">po teste</span>':'<span class="bd t">do '+esc(l.freeOn)+'</span>'):''}</td></tr>`});return r}).join('')||'<tr><td colspan="15" style="text-align:center;color:var(--mu)">Portfólio je prázdne. Pridaj nákupy v sekcii „Správa nákupov“ nad dashboardom.</td></tr>';
tb.onclick=e=>{const c=e.target.closest('.tk');if(!c)return;const tr=c.parentNode,p=P[+tr.dataset.i],on=!tr.classList.contains('o');tr.classList.toggle('o',on);tb.querySelectorAll('tr.lot[data-p="'+tr.dataset.i+'"]').forEach(x=>x.classList.toggle('s',on));O[p.t]=on;try{localStorage.setItem('pf5o',JSON.stringify(O))}catch(e){}};
// watchlist
$('#wsum').textContent=W.length?W.filter(w=>w.sig).length+' z '+W.length+' na nákupnej cene':'';
$('#wb').innerHTML=W.map(w=>`<tr><td><b>${esc(w.t)}</b></td><td>${esc(w.note||'')}</td><td>${F(w.price)?nf.format(w.price):'–'}</td><td>${F(w.fair)?nf.format(w.fair):'–'}</td><td>${F(w.mos)?nf.format(w.mos)+NB+'%':'–'}</td><td>${F(w.trig)?nf.format(w.trig):'–'}</td><td class="${F(w.dist)?(w.dist<=0?'pos':''):''}">${F(w.dist)?pct(w.dist,1):'–'}</td><td>${!F(w.price)?'<span class="bd r">bez ceny</span>':w.sig?'<span class="bd g">🟩 NAKUPUJ</span>':'<span class="bd r">🟥 ČAKAJ</span>'}</td></tr>`).join('')||'<tr><td colspan="8" style="text-align:center;color:var(--mu)">Zoznam je prázdny. Pridaj akcie v sekcii „Čakáreň“ nad dashboardom.</td></tr>';
// alokácia
const sc=[...new Set(P.map(p=>p.s))],col=(s,l)=>`hsl(${HUE[sc.indexOf(s)%8]} 65% ${l}%)`,ch=$('#ch'),tip=$('#tip');
function arc(r0,r1,a0,a1){if(a1-a0>6.283)a1=a0+6.283;const Q=(r,a)=>(200+r*Math.cos(a)).toFixed(2)+','+(200+r*Math.sin(a)).toFixed(2),l=a1-a0>Math.PI?1:0;return`M${Q(r1,a0)}A${r1} ${r1} 0 ${l} 1 ${Q(r1,a1)}L${Q(r0,a1)}A${r0} ${r0} 0 ${l} 0 ${Q(r0,a0)}Z`}
const HH=HV.filter(p=>p.val>0),secs=sc.map(s=>({s,items:HH.filter(p=>p.s===s).sort((a,b)=>b.val-a.val)})).filter(g=>g.items.length);secs.forEach(g=>{g.v=sum(g.items,p=>p.val);g.i=sum(g.items,p=>p.inv)});secs.sort((a,b)=>b.v-a.v);
function seg(g,t,v,i,fill,dd){return`<path class="seg" data-g="${esc(g)}" data-t="${esc(t)}" data-v="${v}" data-i="${i}" fill="${fill}" d="${dd}"/>`}
function donut(){const m=$('#dm').value;let a=-Math.PI/2,h='',lg='',T=tot,items=[];
if(m==='t'){const g={};HH.forEach(p=>{const k=p.typ||'Akcia';g[k]=g[k]||{v:0,i:0};g[k].v+=p.val;g[k].i+=p.inv});if(D.cash>0)g['Hotovosť']={v:D.cash,i:D.cash};Object.keys(g).sort((x,y)=>g[y].v-g[x].v).forEach((k,n)=>{const sp=g[k].v/T*6.2832,f=`hsl(${HUE[n%8]} 65% 48%)`;h+=seg(k,k,g[k].v,g[k].i,f,arc(76,172,a,a+sp));lg+=`<div><i style="background:${f}"></i>${esc(k)}<span>${eur(g[k].v)} · ${pct(g[k].v/T*100)}</span></div>`;a+=sp})}
else if(m==='a'){HH.slice().sort((x,y)=>y.val-x.val).forEach(p=>{const sp=p.val/T*6.2832;h+=seg(p.t,p.t+' · '+p.n,p.val,p.inv,col(p.s,50),arc(76,172,a,a+sp));lg+=`<div><i style="background:${col(p.s,50)}"></i>${esc(p.t)}<span>${eur(p.val)} · ${pct(p.val/T*100)}</span></div>`;a+=sp})}
else{T=vH;secs.forEach(g=>{const a0=a;g.items.forEach((p,i)=>{const sp=p.val/T*6.2832;h+=seg(g.s,p.t+' · '+p.n,p.val,p.inv,col(g.s,40+18*i/Math.max(g.items.length-1,1)),arc(122,172,a,a+sp));a+=sp});h+=seg(g.s,g.s,g.v,g.i,col(g.s,45),arc(76,116,a0,a));lg+=`<div><i style="background:${col(g.s,45)}"></i>${esc(g.s)}<span>${eur(g.v)} · ${pct(g.v/T*100)}</span></div>`})}
ch.dataset.T=T;ch.innerHTML=h+`<text x="200" y="192" text-anchor="middle" fill="#64748b" font-size="13" id="c1">Hodnota</text><text x="200" y="218" text-anchor="middle" fill="#0f172a" font-size="20" font-weight="700" id="c2">${eur(T)}</text>`;$('#lg').innerHTML=lg||'<div>Žiadne pozície</div>'}
ch.onmouseover=ch.onmousemove=e=>{const s=e.target.closest('.seg');if(!s)return;ch.classList.add('h');ch.querySelectorAll('.seg').forEach(x=>x.classList.toggle('on',x===s||x.dataset.g===s.dataset.g));const r=ch.parentNode.getBoundingClientRect(),T=+ch.dataset.T;
tip.innerHTML=`<b>${esc(s.dataset.t)}</b><br>Investované: ${eur(+s.dataset.i)}<br>Aktuálna hodnota: ${eur(+s.dataset.v)}<br>Podiel z portfólia: ${pct(+s.dataset.v/T*100)}`;tip.style.left=Math.min(e.clientX-r.left+14,r.width-200)+'px';tip.style.top=e.clientY-r.top+14+'px';tip.style.opacity=1;$('#c1').textContent=s.dataset.t.split(' · ')[0];$('#c2').textContent=eur(+s.dataset.v)};
ch.onmouseleave=()=>{ch.classList.remove('h');tip.style.opacity=0;$('#c1').textContent='Hodnota';$('#c2').textContent=eur(+ch.dataset.T)};$('#dm').onchange=donut;donut();
function bars(id,key){const g={};HV.forEach(p=>{const k=p[key]||'Nezaradené';g[k]=(g[k]||0)+p.val});const e=Object.entries(g).sort((a,b)=>b[1]-a[1]),m=Math.max(1,...e.map(x=>x[1]));
$(id).innerHTML=e.map((x,n)=>`<div class="br"><span>${esc(x[0])}</span><i style="width:${(x[1]/m*100).toFixed(0)}%;background:hsl(${HUE[n%8]} 65% 50%)"></i><span style="text-align:right">${eur(x[1])}</span></div>`).join('')||'<small>Žiadne pozície</small>'}
bars('#bs','s');bars('#bg','country');
// equity curve
(function(){const c=D.curve,sv=$('#eq');if(!c||!c.d||c.d.length<2){sv.innerHTML='';$('#eqn').textContent='Pre graf vývoja doplň „Dátum nákupu“ pri nákupoch (dostupné ceny a kurz sa stiahnu automaticky).';return}
const n=c.d.length,all=c.p.concat(c.b),mn=Math.min(...all),mx=Math.max(...all),X=i=>10+i/(n-1)*880,Y=v=>250-(v-mn)/((mx-mn)||1)*235,L=a=>a.map((v,i)=>X(i).toFixed(1)+','+Y(v).toFixed(1)).join(' ');
sv.innerHTML=`<polyline fill="none" stroke="#94a3b8" stroke-width="1.6" vector-effect="non-scaling-stroke" points="${L(c.b)}"/><polyline fill="none" stroke="#2563eb" stroke-width="2" vector-effect="non-scaling-stroke" points="${L(c.p)}"/>`;
const p1=c.p[n-1],b1=c.b[n-1];$('#cl').innerHTML=`<span style="color:#2563eb">■ Moje portfólio ${eur(p1)}</span> &nbsp; <span style="color:#94a3b8">■ S&amp;P 500 ${eur(b1)}</span>`;$('#eqn').textContent=c.d[0]+' až '+c.d[n-1]+' · rozdiel oproti S&P 500: '+eur(p1-b1)+' (ceny akcií prepočítané kurzom EUR/USD)'})();
// dividendy
const M=['Jan','Feb','Mar','Apr','Máj','Jún','Júl','Aug','Sep','Okt','Nov','Dec'],dm=M.map((_,i)=>sum(HV,p=>p.divm&&p.divm[i])),da=dm.reduce((a,b)=>a+b,0);
$('#mo').innerHTML=dm.map((v,i)=>`<div style="${v>0?'background:rgba(16,185,129,.12);border-color:#10b981':''}"><small>${M[i]}</small><b>${v>0?eur(v):'–'}</b></div>`).join('');$('#dv').textContent='ročný pasívny príjem ≈ '+eur(da);
// daň
const L=[];H.forEach(p=>p.lots.forEach(l=>{if(l.d&&!l.sold)L.push({t:p.t,l})}));const fr=L.filter(x=>x.l.free),nx=L.filter(x=>!x.l.free).sort((a,b)=>a.l.left-b.l.left);
$('#tx').innerHTML=L.length?`<div><b>${fr.length}</b> z ${L.length} nákupov je po časovom teste (hodnota ${eur(sum(fr,x=>x.l.val))})</div>`+nx.slice(0,5).map(x=>`<div style="margin-top:6px"><span class="bd t">${esc(x.t)}</span> za ${x.l.left} dní (${esc(x.l.freeOn)}) · ${eur(x.l.val)}</div>`).join(''):'<small>Doplň „Dátum nákupu“ pri nákupoch.</small>';
</script></body></html>
"""


def make_curve(pos, hist):
    """Hodnota portfólia v čase vs. S&P 500 s rovnakými vkladmi v rovnaké dni (všetko v EUR, týždenne)."""
    spx, fxh = hist.get("^GSPC", {}).get("h5"), hist.get("EURUSD=X", {}).get("h5")
    lots = [(p, l) for p in pos for l in p["lots"] if l["d"] and not l["sold"] and p["h5"]]
    if not spx or not fxh or not lots:
        return {}
    S = pd.Series({pd.Timestamp(d): v for d, v in spx}).sort_index()
    X = pd.Series({pd.Timestamp(d): v for d, v in fxh}).sort_index()
    start = min(pd.Timestamp(l["d"]) for _, l in lots)
    wk = S.index[S.index >= start - pd.Timedelta(days=7)]
    if len(wk) < 3:
        return {}
    fx = X.reindex(wk, method="ffill").bfill()
    se = S.reindex(wk) / fx
    port, bench = np.zeros(len(wk)), np.zeros(len(wk))
    for p, l in lots:
        px = pd.Series({pd.Timestamp(d): v for d, v in p["h5"]}).sort_index().reindex(wk, method="ffill")
        rate = fx if p["ccy"] == "USD" else (1.0 if p["ccy"] == "EUR" else p["rate"])
        t0 = pd.Timestamp(l["d"])
        m = np.asarray(wk >= t0)
        port += np.where(m, np.asarray((l["q"] * px / rate).fillna(0.0)), 0.0)
        base = se.asof(t0)
        if np.isfinite(base) and base > 0 and np.isfinite(l["inv"]):
            bench += np.where(m, l["inv"] * np.asarray(se / base), 0.0)
    return dict(d=[d.strftime("%Y-%m-%d") for d in wk], p=port.tolist(), b=bench.tolist())


def xirr(pos):
    """Ročná návratnosť vážená vkladmi (money-weighted) z nákupov s dátumom; menej než 6 mesiacov = bez hodnoty."""
    L = [l for p in pos for l in p["lots"] if l["d"] and not l["sold"] and np.isfinite(l["val"])]
    if not L:
        return np.nan
    t0, now = min(pd.Timestamp(l["d"]) for l in L), pd.Timestamp.now().normalize()
    cur = sum(l["val"] for l in L)
    if (now - t0).days < 180 or cur <= 0:
        return np.nan

    def npv(r):
        return (sum(-l["inv"] / (1 + r) ** ((pd.Timestamp(l["d"]) - t0).days / 365) for l in L)
                + cur / (1 + r) ** ((now - t0).days / 365))

    lo, hi = -0.95, 10.0
    if npv(lo) * npv(hi) > 0:
        return np.nan
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if npv(lo) * npv(mid) <= 0 else (mid, hi)
    return mid * 100


def build_watch(w, quotes, dmos):
    out = []
    for _, r in w.iterrows():
        t = r["Ticker"]
        if not t:
            continue
        q = quotes.get(ysym(t), {})
        price = sf(q.get("price")) * (0.01 if q.get("cur") == "GBp" else 1.0)
        fair = r["Férová cena"] if np.isfinite(r["Férová cena"]) and r["Férová cena"] > 0 else np.nan
        mos = r["MoS %"] if np.isfinite(r["MoS %"]) and 0 <= r["MoS %"] < 100 else dmos
        trig = fair * (1 - mos / 100) if np.isfinite(fair) else np.nan
        out.append(dict(t=t, note=r["Poznámka"], price=price, fair=fair, mos=mos if np.isfinite(fair) else np.nan, trig=trig,
                        dist=(price / trig - 1) * 100 if np.isfinite(price) and np.isfinite(trig) else np.nan,
                        sig=bool(np.isfinite(price) and np.isfinite(trig) and price <= trig)))
    return out


def normalize_watch(df):
    df = pd.DataFrame() if df is None else df.copy()
    for c in WCOLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[WCOLS]
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip().str.upper()
    df["Poznámka"] = df["Poznámka"].fillna("").astype(str)
    for c in ("Férová cena", "MoS %"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[df["Ticker"] != ""].reset_index(drop=True)


def dashboard(lots, watch, cash, dmos):
    syms = tuple(sorted({ysym(t) for t in list(lots["Ticker"]) + list(watch["Ticker"]) if t}))
    if not syms:
        data = dict(pos=[], watch=[], cash=cash, fx=None, ts=datetime.now().strftime("%H:%M:%S"), curve={}, xirr=None)
    else:
        quotes, hist = get_quotes(syms), get_hist(syms + ("^GSPC", "EURUSD=X"))
        ccys = tuple(sorted({("GBP" if q.get("cur") == "GBp" else (q.get("cur") or "USD")) for q in quotes.values()}))
        rates = get_rates(ccys + ("USD",))
        pos = build(lots, quotes, hist, rates)
        data = dict(pos=pos, watch=build_watch(watch, quotes, dmos), cash=cash, fx=rates.get("USD"),
                    ts=datetime.now().strftime("%H:%M:%S"), curve=make_curve(pos, hist), xirr=xirr(pos))
        seen = st.session_state.setdefault("pf_toasted", set())
        now_sig = {w["t"] for w in data["watch"] if w["sig"]}
        for t in sorted(now_sig - seen):
            try:
                st.toast(f"🟩 {t} je na nákupnej cene", icon="🟢")
            except Exception:
                pass
        st.session_state["pf_toasted"] = now_sig
    blob = json.dumps(clean(data), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    components.html(TEMPLATE.replace("__DATA__", blob), height=2300, scrolling=True)


def load_settings():
    try:
        with open(SFILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


with st.sidebar:
    st.header("⚙️ Nastavenia")
    cfg = load_settings()
    cash = st.number_input("Hotovosť (€)", min_value=0.0, value=float(cfg.get("cash", 0.0)), step=50.0, key="pf_cash")
    dmos = st.number_input("Predvolený margin of safety (%)", min_value=0.0, max_value=90.0,
                           value=float(cfg.get("mos", 15.0)), step=1.0, key="pf_mos")
    refresh = REFRESH[st.selectbox("Automatické obnovovanie cien", list(REFRESH), index=2, key="pf_refresh")]
    if st.button("🔄 Obnoviť ceny teraz", key="pf_reload"):
        st.cache_data.clear()
        _store().clear()
        st.rerun()
try:
    with open(SFILE, "w", encoding="utf-8") as f:
        json.dump({"cash": cash, "mos": dmos}, f)
except Exception:
    pass


def read_csv_safe(path):
    try:
        return pd.read_csv(path) if os.path.exists(path) else None
    except Exception:
        return None


if "pf_base" not in st.session_state:
    st.session_state["pf_base"] = normalize(read_csv_safe(FILE))
    st.session_state["pw_base"] = normalize_watch(read_csv_safe(WFILE))
    st.session_state["pf_ver"] = 0

c1, c2 = st.columns(2)
up = c1.file_uploader("Nahrať zálohu nákupov (CSV)", type="csv", key="pf_up")
if up is not None and st.session_state.get("pf_sig") != (up.name, up.size):
    try:
        st.session_state["pf_base"] = normalize(pd.read_csv(up))
        st.session_state["pf_ver"] += 1
        st.session_state["pf_sig"] = (up.name, up.size)
    except Exception as e:
        st.error(f"Zálohu sa nepodarilo načítať: {e}")
upw = c2.file_uploader("Nahrať zálohu čakárne (CSV)", type="csv", key="pw_up")
if upw is not None and st.session_state.get("pw_sig") != (upw.name, upw.size):
    try:
        st.session_state["pw_base"] = normalize_watch(pd.read_csv(upw))
        st.session_state["pf_ver"] += 1
        st.session_state["pw_sig"] = (upw.name, upw.size)
    except Exception as e:
        st.error(f"Zálohu sa nepodarilo načítať: {e}")

with st.expander("➕ Správa nákupov (pridať / upraviť / odstrániť)", expanded=st.session_state["pf_base"].empty):
    st.caption("Jeden riadok = jeden nákup. Ticker môžeš zadať aj v tvare XTB (napr. BRKB.US). „Vložené €“ je suma v eurách, "
               "ktorú si skutočne vložil (v XTB hodnota mínus zisk), nie aktuálna hodnota. „Dátum nákupu“ sa použije v grafe "
               "vývoja, pri XIRR a pri daňovom trackeri. „Krajina“ je pre graf geografie. Pre predaný nákup vyplň „Predajná cena“.")
    edited = st.data_editor(
        st.session_state["pf_base"], num_rows="dynamic", hide_index=True, use_container_width=True,
        key=f"pf_editor_{st.session_state['pf_ver']}",
        column_config={
            "Typ": st.column_config.SelectboxColumn("Typ", options=ASSET_TYPES),
            "Dátum nákupu": st.column_config.DateColumn("Dátum nákupu"),
            "Kusy": st.column_config.NumberColumn(format="%.4f", min_value=0.0),
            "Cena nákupu": st.column_config.NumberColumn("Cena nákupu (v mene akcie)", format="%.2f", min_value=0.0),
            "Vložené €": st.column_config.NumberColumn("Vložené (€)", format="%.2f", min_value=0.0),
            "Férová cena": st.column_config.NumberColumn("Férová cena", format="%.2f", min_value=0.0),
            "Predajná cena": st.column_config.NumberColumn("Predajná cena", format="%.2f", min_value=0.0),
        })
with st.expander("💡 Čakáreň na férovú cenu (akcie, ktoré chceš kúpiť)", expanded=st.session_state["pw_base"].empty):
    st.caption("Cieľová nákupná cena = férová cena − margin of safety. Keď cena klesne na cieľ, zasvieti zelené „NAKUPUJ“ "
               "(upozornenie funguje, len keď je táto stránka otvorená).")
    wedited = st.data_editor(
        st.session_state["pw_base"], num_rows="dynamic", hide_index=True, use_container_width=True,
        key=f"pw_editor_{st.session_state['pf_ver']}",
        column_config={"Férová cena": st.column_config.NumberColumn("Férová cena", format="%.2f", min_value=0.0),
                       "MoS %": st.column_config.NumberColumn("Môj MoS % (prázdne = predvolený)", format="%.0f", min_value=0.0, max_value=95.0)})
lots, watch = normalize(edited), normalize_watch(wedited)
for df_, path in ((lots, FILE), (watch, WFILE)):
    try:
        df_.to_csv(path, index=False)
    except Exception:
        pass
d1, d2 = st.columns(2)
d1.download_button("⬇️ Záloha nákupov (CSV)", lots.to_csv(index=False).encode("utf-8-sig"), file_name="portfolio_lots.csv",
                   mime="text/csv", key="pf_dl")
d2.download_button("⬇️ Záloha čakárne (CSV)", watch.to_csv(index=False).encode("utf-8-sig"), file_name="portfolio_watch.csv",
                   mime="text/csv", key="pw_dl")
st.caption("⚠️ Súbory v aplikácii sa môžu po reštarte stratiť: zálohy si pravidelne sťahuj a po reštarte nahraj.")

frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if refresh and frag is not None:
    try:
        frag(run_every=refresh)(dashboard)(lots, watch, cash, dmos)
    except TypeError:
        dashboard(lots, watch, cash, dmos)
else:
    dashboard(lots, watch, cash, dmos)
