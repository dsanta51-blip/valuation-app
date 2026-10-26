"""
Screener – kvalitné veľké firmy, ktoré môžu byť podhodnotené

Umiestnenie v repozitári:  pages/1_Screener.py   (Streamlit ho sám pridá do menu vľavo)
Hlavný súbor valuation_app.py sa nemení. Nové knižnice netreba: streamlit, yfinance, pandas, numpy.

Čo robí: stiahne rýchle údaje z Yahoo pre zoznam tickerov, spočíta skóre KVALITY a skóre OCENENIA,
označí rizikové vlajky (value traps) a zoradí kandidátov. Je to SITO, nie verdikt:
finalistov treba prejsť v hlavnej aplikácii (DCF / DDM, krížová kontrola) a overiť na roic.ai / macrotrends.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Screener – kvalitné firmy", page_icon="🔎", layout="wide")

# Univerzum na skenovanie (veľké, známe firmy naprieč sektormi). Nie je to odporúčanie – len zoznam na prehľadanie.
DEFAULT_UNIVERSE = (
    "AAPL, MSFT, GOOGL, META, AVGO, ORCL, CSCO, ADBE, CRM, TXN, QCOM, IBM, ACN, INTU, "
    "PEP, KO, PG, WMT, COST, MDLZ, PM, MO, CL, KMB, GIS, HSY, "
    "JNJ, UNH, LLY, ABBV, MRK, PFE, TMO, ABT, MDT, AMGN, GILD, BMY, ISRG, SYK, "
    "JPM, BAC, V, MA, BRK-B, GS, MS, BLK, AXP, SPGI, MMC, CB, PGR, "
    "CAT, DE, HON, UNP, UPS, GE, LMT, RTX, MMM, ETN, ITW, WM, "
    "AMZN, HD, MCD, NKE, SBUX, LOW, TJX, BKNG, "
    "XOM, CVX, COP, EOG, SLB, NEE, DUK, SO, "
    "DIS, CMCSA, VZ, T, NFLX, LIN, SHW, APD, PLD, AMT"
)
MAX_TICKERS = 150
KEYS = ["shortName", "sector", "currentPrice", "regularMarketPrice", "marketCap", "forwardPE", "trailingPE",
        "enterpriseToEbitda", "returnOnEquity", "operatingMargins", "profitMargins", "totalDebt", "totalCash",
        "ebitda", "freeCashflow", "totalRevenue", "revenueGrowth", "earningsGrowth", "payoutRatio",
        "fiftyTwoWeekHigh", "sharesOutstanding", "bookValue", "priceToBook", "trailingPegRatio",
        "currency", "financialCurrency"]
EXCLUDED_SECTORS = ("Financial Services", "Real Estate")   # iný typ oceňovania (BVPS / dividendy, nie FCF a dlh/EBITDA)


def sf(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ----------------------------------------------------------------------------
# DÁTA
# ----------------------------------------------------------------------------
def fetch_one(sym: str) -> dict:
    try:
        info = yf.Ticker(sym).info or {}
    except Exception:
        info = {}
    out = {k: info.get(k) for k in KEYS}
    out["sym"] = sym
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_batch(symbols: tuple) -> list:
    """Stiahne údaje paralelne (6 vlákien). Výsledok sa 6 hodín cachuje, aby sa Yahoo zbytočne nezaťažovalo."""
    with ThreadPoolExecutor(max_workers=6) as ex:
        return list(ex.map(fetch_one, symbols))


def parse_symbols(text: str) -> list:
    seen, out = set(), []
    for tok in re.split(r"[,\s;]+", text.upper()):
        tok = tok.strip()
        if tok and re.fullmatch(r"[A-Z0-9.\-^=]{1,12}", tok) and tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


# ----------------------------------------------------------------------------
# VÝPOČTY
# ----------------------------------------------------------------------------
def dcf_per_share(fcf_ps: float, g: float, gt: float, r: float, fade: bool = True) -> float:
    """Zjednodušený FCFE-DCF na akciu: 5 rokov rastu g, 5 rokov lineárneho spomalenia na gt, potom Gordon."""
    if not np.isfinite(fcf_ps) or fcf_ps <= 0 or r <= gt:
        return np.nan
    rates = [g] * 5 + ([g + (gt - g) * i / 5 for i in range(1, 6)] if fade else [])
    f, pv = fcf_ps, 0.0
    for y, gr in enumerate(rates, start=1):
        f *= 1 + gr
        pv += f / (1 + r) ** y
    tv = f * (1 + gt) / (r - gt)
    return pv + tv / (1 + r) ** len(rates)


def analyze(rows: list, p: dict) -> pd.DataFrame:
    recs = []
    for r in rows:
        if not r:
            continue
        price = sf(r.get("currentPrice"))
        if np.isnan(price):
            price = sf(r.get("regularMarketPrice"))
        mcap = sf(r.get("marketCap"))
        bvps, pb = sf(r.get("bookValue")), sf(r.get("priceToBook"))
        neg_equity = (np.isfinite(bvps) and bvps <= 0) or (np.isfinite(pb) and pb < 0)
        roe = sf(r.get("returnOnEquity")) * 100
        if neg_equity:
            roe = np.nan                                     # ROE pri zápornom vlastnom imaní je zavádzajúce
        opm, pm = sf(r.get("operatingMargins")) * 100, sf(r.get("profitMargins")) * 100
        ebitda, debt, cash = sf(r.get("ebitda")), sf(r.get("totalDebt")), sf(r.get("totalCash"), 0.0)
        nde = (debt - cash) / ebitda if np.isfinite(debt) and np.isfinite(ebitda) and ebitda > 0 else np.nan
        fcf, rev = sf(r.get("freeCashflow")), sf(r.get("totalRevenue"))
        fcfm = fcf / rev * 100 if np.isfinite(fcf) and np.isfinite(rev) and rev > 0 else np.nan
        fcfy = fcf / mcap * 100 if np.isfinite(fcf) and np.isfinite(mcap) and mcap > 0 else np.nan
        revg, epsg = sf(r.get("revenueGrowth")) * 100, sf(r.get("earningsGrowth")) * 100
        payout = sf(r.get("payoutRatio")) * 100
        hi52 = sf(r.get("fiftyTwoWeekHigh"))
        dd = (price / hi52 - 1) * 100 if np.isfinite(price) and np.isfinite(hi52) and hi52 > 0 else np.nan
        # počet akcií = trhová kapitalizácia / cena (správne aj pri viacerých triedach akcií, napr. GOOGL, BRK-B)
        shares = mcap / price if np.isfinite(mcap) and np.isfinite(price) and price > 0 else sf(r.get("sharesOutstanding"))
        fv = dcf_per_share(fcf / shares if np.isfinite(shares) and shares > 0 else np.nan,
                           p["g"] / 100, p["gt"] / 100, p["r"] / 100, p["fade"])
        mos = (fv / price - 1) * 100 if np.isfinite(fv) and np.isfinite(price) and price > 0 else np.nan
        recs.append(dict(
            Ticker=r["sym"], Názov=r.get("shortName") or "", Sektor=r.get("sector") or "n/a",
            price=price, mcap=mcap / 1e9 if np.isfinite(mcap) else np.nan,
            fpe=sf(r.get("forwardPE")), ev_ebitda=sf(r.get("enterpriseToEbitda")),
            roe=roe, opm=opm, pm=pm, nde=nde, fcfm=fcfm, fcfy=fcfy, revg=revg, epsg=epsg,
            payout=payout, dd=dd, fv=fv, mos=mos, neg_equity=neg_equity, fcf=fcf, ebitda=ebitda,
            peg=sf(r.get("trailingPegRatio")),
        ))
    df = pd.DataFrame(recs)
    if df.empty:
        return df

    # vylúčenie sektorov, kde FCF a dlh/EBITDA nedávajú zmysel
    df["excluded"] = df["Sektor"].isin(EXCLUDED_SECTORS) if p["excl_fin"] else False

    # sektorové mediány (v rámci skenovaného univerza; menej než 3 firmy v sektore → medián všetkých)
    pool = df[~df["excluded"]]

    def med(col):
        allm = pool[col][pool[col] > 0].median()
        out = {}
        for sec, grp in pool.groupby("Sektor"):
            v = grp[col][grp[col] > 0]
            out[sec] = v.median() if len(v) >= 3 else allm
        return out, allm

    med_pe, all_pe = med("fpe")
    med_ev, all_ev = med("ev_ebitda")

    def rel(value, sector, table, fallback):
        m = table.get(sector, fallback)
        if not (np.isfinite(value) and value > 0 and np.isfinite(m) and m > 0):
            return np.nan
        return 1 - value / m                                  # kladné = lacnejšie než sektorový medián

    df["rel_pe"] = [rel(v, s, med_pe, all_pe) for v, s in zip(df["fpe"], df["Sektor"])]
    df["rel_ev"] = [rel(v, s, med_ev, all_ev) for v, s in zip(df["ev_ebitda"], df["Sektor"])]

    def ladder(x, steps):          # steps: [(prah, body), ...] zostupne; NaN = 0 bodov
        for th, pts in steps:
            if np.isfinite(x) and x >= th:
                return pts
        return 0

    def lower(x, steps):           # čím nižšie, tým lepšie
        for th, pts in steps:
            if np.isfinite(x) and x <= th:
                return pts
        return 0

    df["Q"] = [
        ladder(a, [(15, 20), (10, 10)]) + ladder(b, [(15, 20), (10, 10)]) + ladder(c, [(10, 10), (5, 5)]) +
        lower(d_, [(2, 20), (3, 10)]) + ladder(e, [(10, 15), (5, 8)]) + ladder(f, [(3, 10), (0, 5)]) +
        ladder(g, [(0, 5)])
        for a, b, c, d_, e, f, g in zip(df["roe"], df["opm"], df["pm"], df["nde"], df["fcfm"], df["revg"], df["epsg"])
    ]
    df["V"] = [
        ladder(a, [(6, 30), (4, 20), (3, 10)]) + ladder(b, [(0.2, 25), (0.1, 15), (0.0, 8)]) +
        ladder(c, [(0.2, 20), (0.1, 12), (0.0, 6)]) + ladder(d_, [(30, 25), (15, 15), (0, 8)])
        for a, b, c, d_ in zip(df["fcfy"], df["rel_pe"], df["rel_ev"], df["mos"])
    ]
    df["Skóre"] = (df["Q"] + df["V"]) / 2

    # rizikové vlajky (value traps)
    def flags(row):
        f = []
        if np.isfinite(row.epsg) and row.epsg < -5:
            f.append("zisky klesajú")
        if np.isfinite(row.revg) and row.revg < 0:
            f.append("tržby klesajú")
        if np.isfinite(row.nde) and row.nde > 3:
            f.append("vysoký dlh")
        if np.isfinite(row.ebitda) and row.ebitda <= 0:
            f.append("záporné EBITDA")
        if np.isfinite(row.fcf) and row.fcf <= 0:
            f.append("záporné FCF")
        if np.isfinite(row.payout) and row.payout > 100:
            f.append("payout > 100 %")
        if np.isfinite(row.dd) and row.dd < -40:
            f.append("pokles > 40 % od maxima")
        if row.neg_equity:
            f.append("záporné vlastné imanie")
        if (np.isfinite(row.fcfy) and row.fcfy > 15) or (np.isfinite(row.mos) and row.mos > 200):
            f.append("podozrivé dáta – over ručne")
        return f

    fl = [flags(r) for r in df.itertuples()]
    df["Vlajky"] = [", ".join(x) for x in fl]
    df["n_flags"] = [len(x) for x in fl]

    need = ["roe", "opm", "pm", "nde", "fcfm", "fcfy", "revg", "epsg", "fpe", "ev_ebitda"]
    df["Úplnosť dát %"] = df[need].notna().mean(axis=1) * 100

    # kandidáti a dôvody vyradenia
    out_why = []
    for _, r in df.iterrows():
        why = []
        if r["excluded"]:
            why.append("finančný sektor / REIT")
        if not np.isfinite(r["mcap"]) or r["mcap"] < p["min_mcap"]:
            why.append("malá kapitalizácia")
        if r["Úplnosť dát %"] < 70:
            why.append("neúplné dáta")
        if r["Q"] < p["min_q"]:
            why.append(f"kvalita < {p['min_q']}")
        if r["V"] < p["min_v"]:
            why.append(f"ocenenie < {p['min_v']}")
        if p["no_flags"] and r["n_flags"] > 0:
            why.append("rizikové vlajky")
        if not np.isfinite(r["mos"]) or r["mos"] < p.get("min_mos", 0.0):
            why.append(f"marža bezpečnosti < {p.get('min_mos', 0.0):.0f} %")
        if not np.isfinite(r["fcfy"]) or r["fcfy"] < p.get("min_fcfy", 0.0):
            why.append(f"FCF yield < {p.get('min_fcfy', 0.0):.1f} %")
        out_why.append(", ".join(why))
    df["Prečo nie"] = out_why
    df["Kandidát"] = df["Prečo nie"] == ""
    return df


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Ticker": df["Ticker"], "Názov": df["Názov"], "Sektor": df["Sektor"],
        "Cena": df["price"].round(2), "MCap (mld.)": df["mcap"].round(0),
        "Kvalita": df["Q"], "Ocenenie": df["V"], "Skóre": df["Skóre"].round(1),
        "Rýchly DCF": df["fv"].round(2), "MoS %": df["mos"].round(0),
        "FCF yield %": df["fcfy"].round(1), "Fwd P/E": df["fpe"].round(1), "EV/EBITDA": df["ev_ebitda"].round(1),
        "ROE %": df["roe"].round(0), "Op. marža %": df["opm"].round(1), "ND/EBITDA": df["nde"].round(1),
        "Od maxima %": df["dd"].round(0), "Vlajky": df["Vlajky"], "Dáta %": df["Úplnosť dát %"].round(0),
    })
    return out


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🔎 Screener – kvalitné veľké firmy, ktoré môžu byť podhodnotené")
st.caption("Sito na výber kandidátov, nie verdikt. Finalistov vždy prejdi v hlavnej aplikácii (DCF / DDM, krížová kontrola) "
           "a over kľúčové čísla na roic.ai / macrotrends.net.")

with st.expander("📋 Stratégia krok za krokom", expanded=False):
    st.markdown("""
**1. Univerzum:** len veľké firmy so stabilným biznisom (predvolene trhová kapitalizácia nad 50 mld. USD), bez bánk a REIT.
**2. Kvalita (skóre 0–100):** ROE, prevádzková a čistá marža, dlh/EBITDA, FCF marža, rast tržieb a ziskov.
**3. Ocenenie (skóre 0–100):** FCF yield, forward P/E a EV/EBITDA voči sektorovému mediánu, rýchly DCF (marža bezpečnosti).
**4. Rizikové vlajky:** klesajúce zisky či tržby, vysoký dlh, záporné FCF, payout nad 100 %, pokles nad 40 % od maxima.
**5. Kandidát:** kvalita aj ocenenie nad prahom, bez vlajok, s kladnou maržou bezpečnosti rýchleho DCF a FCF yieldom aspoň 5 %. Zoznam je len začiatok.
**6. Hĺbková kontrola (ručne):** hlavná aplikácia, 5–10-ročný priemer FCF a P/E, DCF + DDM, bear/base/bull, krížová kontrola.
**7. Dôvod, prečo je lacná:** napíš jednou vetou. Ak ju nevieš, nekupuj.
**8. Nákup po častiach:** 3 tranže, limit na sektor a na jednu pozíciu, v hlavnej aplikácii sleduj pásma BUY / HOLD / SELL.
""")

with st.sidebar:
    st.header("⚙️ Filtre")
    min_mcap = st.number_input("Min. trhová kapitalizácia (mld. USD)", min_value=0.0, value=50.0, step=10.0, key="s_mcap")
    min_q = st.number_input("Min. skóre kvality", min_value=0, max_value=100, value=60, step=5, key="s_q")
    min_v = st.number_input("Min. skóre ocenenia", min_value=0, max_value=100, value=40, step=5, key="s_v")
    excl_fin = st.checkbox("Vylúčiť finančný sektor a REIT", value=True, key="s_fin",
                           help="Banky, poisťovne a REIT sa oceňujú inak (BVPS, dividendy), FCF a dlh/EBITDA tu zavádzajú.")
    no_flags = st.checkbox("Vylúčiť firmy s rizikovými vlajkami", value=True, key="s_flags")
    min_mos = st.number_input("Min. marža bezpečnosti rýchleho DCF (%)", min_value=-100.0, max_value=100.0,
                              value=0.0, step=5.0, key="s_mos",
                              help="Firma musí byť podľa rýchleho DCF aspoň o toľko pod férovou cenou. 0 = len nie je drahá.")
    min_fcfy = st.number_input("Min. FCF yield (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.5, key="s_fcfy")
    st.divider()
    st.markdown("**Rýchly DCF (len na prvé sito)**")
    g = st.number_input("Rast FCF, roky 1–5 (%)", min_value=-5.0, max_value=15.0, value=4.0, step=0.5, key="s_g")
    gt = st.number_input("Terminálny rast (%, max 2.5)", min_value=0.0, max_value=2.5, value=2.0, step=0.25, key="s_gt")
    r_disc = st.number_input("Diskontná sadzba – cost of equity (%)", min_value=5.0, max_value=15.0, value=9.0,
                             step=0.5, key="s_r")
    fade = st.checkbox("Spomalenie rastu v rokoch 6–10", value=True, key="s_fade")

params = dict(min_mcap=min_mcap, min_q=min_q, min_v=min_v, excl_fin=excl_fin, no_flags=no_flags,
              min_mos=min_mos, min_fcfy=min_fcfy,
              g=g, gt=min(gt, 2.5), r=r_disc, fade=fade)

universe_text = st.text_area("Zoznam tickerov na skenovanie (čiarkou; môžeš upraviť alebo pridať vlastné)",
                             value=DEFAULT_UNIVERSE, height=130, key="universe")
symbols = parse_symbols(universe_text)
if len(symbols) > MAX_TICKERS:
    st.warning(f"Zoznam má {len(symbols)} tickerov, skenuje sa prvých {MAX_TICKERS}.")
    symbols = symbols[:MAX_TICKERS]
st.caption(f"Na skenovanie: {len(symbols)} tickerov. Sken trvá zvyčajne 1–2 minúty, výsledok sa 6 hodín pamätá.")

if st.button("🚀 Spustiť sken", key="scan"):
    if not symbols:
        st.error("Zoznam tickerov je prázdny.")
    else:
        with st.spinner("Sťahujem údaje z Yahoo Finance…"):
            st.session_state["scan_rows"] = fetch_batch(tuple(symbols))
            st.session_state["scan_time"] = datetime.now().strftime("%d.%m.%Y %H:%M")

rows = st.session_state.get("scan_rows")
if not rows:
    st.info("Klikni na „Spustiť sken“.")
    st.stop()

df = analyze(rows, params)
if df.empty:
    st.error("Z Yahoo sa nepodarilo načítať žiadne dáta. Skús to o chvíľu znova.")
    st.stop()

n_all = len(rows)
n_data = int((df["Úplnosť dát %"] >= 70).sum())
cand = df[df["Kandidát"]].sort_values("Skóre", ascending=False)
st.caption(f"Sken z {st.session_state.get('scan_time', '')} · načítané {len(df)} z {n_all} · "
           f"s dostatočnými dátami {n_data} · kandidátov {len(cand)}")
if len(df) < n_all or n_data < 0.7 * n_all:
    st.warning("Časť dát chýba (Yahoo môže dočasne blokovať požiadavky). Skús sken zopakovať neskôr.")

st.markdown("## ✅ Kandidáti")
if cand.empty:
    st.info("Pri týchto filtroch nie je žiadny kandidát. Zníž prahy v bočnom paneli alebo si pozri tabuľku nižšie "
            "a zisti, prečo firmy vypadli.")
else:
    st.dataframe(display_frame(cand.head(30)), use_container_width=True, hide_index=True)
    top_sector = cand["Sektor"].value_counts(normalize=True)
    if len(cand) >= 3 and top_sector.iloc[0] > 0.4:
        st.warning(f"Koncentrácia: {top_sector.iloc[0] * 100:.0f} % kandidátov je zo sektora „{top_sector.index[0]}“. "
                   "Pri výbere si stráž limit na sektor.")
    st.download_button("⬇️ Stiahnuť kandidátov (CSV)", display_frame(cand).to_csv(index=False).encode("utf-8-sig"),
                       file_name="kandidati.csv", mime="text/csv")
    st.caption("Ďalší krok: skopíruj ticker do hlavnej aplikácie a prejdi DCF / DDM, bear/base/bull a krížovú kontrolu.")

with st.expander("📊 Všetky skenované firmy a dôvod vyradenia", expanded=False):
    full = display_frame(df.sort_values("Skóre", ascending=False))
    full["Prečo nie"] = df.sort_values("Skóre", ascending=False)["Prečo nie"].values
    st.dataframe(full, use_container_width=True, hide_index=True)

with st.expander("ℹ️ Metodika a obmedzenia", expanded=False):
    st.markdown("""
**Kvalita (max 100):** ROE ≥ 15 % (20 b.), prevádzková marža ≥ 15 % (20), čistá marža ≥ 10 % (10), (čistý dlh)/EBITDA ≤ 2 (20),
FCF marža ≥ 10 % (15), rast tržieb ≥ 3 % (10), nezáporný rast zisku (5). Nižšie prahy dávajú čiastočné body.

**Ocenenie (max 100):** FCF yield ≥ 6 % (30), forward P/E lacnejšie o 20 % než sektorový medián (25),
EV/EBITDA lacnejšie o 20 % než sektorový medián (20), rýchly DCF s maržou bezpečnosti ≥ 30 % (25).

**Dodatočné filtre kandidáta:** marža bezpečnosti rýchleho DCF aspoň 0 % a FCF yield aspoň 5 % (nastaviteľné). Bránia tomu,
aby firma prešla len vďaka tomu, že je lacnejšia než sektorový medián, hoci podľa DCF drahá je.

**Čo to nevie:** sektorové mediány sú len v rámci skenovaného zoznamu; ROE môžu nafúknuť spätné odkupy a dlh;
rast ziskov z Yahoo je za posledný kvartál a je zašumený; rýchly DCF používa jednotný rast a FCF z Yahoo (TTM), počet akcií berie z trhovej kapitalizácie;
cyklické firmy môžu vyzerať lacno na vrchole zisku; dáta z Yahoo môžu chýbať alebo meškať.
Preto je výstup len zoznam na hlbšiu kontrolu, nie nákupný signál.
""")
st.caption("⚠️ Nie je to investičné odporúčanie. Údaje z Yahoo sú neoficiálne a môžu byť neúplné.")
