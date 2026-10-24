"""
Valuačná kalkulačka akcií – Graham & Buffett (Streamlit + yfinance)

Inštalácia:   pip install streamlit yfinance pandas numpy jinja2
Spustenie:    streamlit run valuation_app.py

Modely: DCF | DDM (Gordon) | Excess Returns / NAV | Normalized Multiples
Pásma:  BUY (<= FV*0.85) | HOLD (FV ± 15 %) | PARTIAL SELL (> FV*1.15) | FULL EXIT (> FV*1.35)
Dáta:   yfinance (cena + fundamenty). Fundamenty sú editovateľné (Manual Override Form),
        aby si ich mohol skontrolovať podľa roic.ai / macrotrends.net / stockanalysis.com.
"""

import html
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Valuačná kalkulačka – Graham & Buffett", page_icon="📊", layout="wide")

# ----------------------------------------------------------------------------
# KONŠTANTY / SANITY GUARDS
# ----------------------------------------------------------------------------
M = 1_000_000
TERMINAL_CAP = 2.5          # % – natvrdo maximum terminálneho rastu
STRESS_THRESHOLD = 15.0     # % – nad touto hodnotou rastu sa zobrazí konzervatívny scenár
STRESS_GROWTH = 8.0         # % – rast konzervatívneho scenára
BUY_LEVEL, HOLD_LEVEL, EXIT_LEVEL = 0.85, 1.15, 1.35

OCF_NAMES = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
             "Total Cash From Operating Activities"]
CAPEX_NAMES = ["Capital Expenditure", "Capital Expenditures"]
CASH_NAMES = ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]
CYCLICAL_SECTORS = ("Energy", "Basic Materials", "Consumer Cyclical")

MODEL_LABELS = {
    "dcf": "DCF (Discounted Cash Flow)",
    "ddm": "DDM (Gordon Growth)",
    "excess": "Excess Returns / NAV",
    "multiples": "Normalized Multiples",
}

ZONES = {
    "BUY": dict(
        key="BUY", label="NÁKUPNÁ ZÓNA – BUY", color="#1b8a3a", cell="#c8efd3",
        status="Podhodnotené. Bezpečnostný polštár aktívny.",
        addendum="Podhodnotené. Bezpečnostný polštár (Margin of Safety ≥ 15 %) je aktívny. "
                 "Vhodná zóna na nákup / postupné navyšovanie pozície podľa tvojich pravidiel alokácie.",
    ),
    "HOLD": dict(
        key="HOLD", label="FÉROVÁ ZÓNA – HOLD", color="#1e5fa8", cell="#cfe0f5",
        status="Zodpovedá reálnej hodnote. Držať pozíciu.",
        addendum="Zodpovedá reálnej hodnote. Držať pozíciu.",
    ),
    "PARTIAL": dict(
        key="PARTIAL", label="MIERNE NADHODNOTENÉ – PARTIAL SELL", color="#d97706", cell="#fde2b8",
        status="Akcia sa obchoduje s prémiou (+15 % až +35 % nad férovou cenou).",
        addendum="Akcia sa obchoduje s prémiou. Na zváženie: Pozastaviť nové nákupy. "
                 "Ak pozícia tvorí >15% portfólia, zváž odpredaj 15–25% pozície na uzamknutie zisku "
                 "a rebalansovanie kapitálu.",
    ),
    "EXIT": dict(
        key="EXIT", label="EXTRÉMNE NADHODNOTENÉ – FULL EXIT", color="#c62828", cell="#f7c6c6",
        status="Cena je viac než 35 % nad férovou hodnotou.",
        addendum="KRITICKÉ VAROVANIE: Cena výrazne predbieha fundamentálnu hodnotu. Riziko korekcie je vysoké. "
                 "Odporúča sa výrazný čiastočný odpredaj (30–50% pozície alebo vytiahnutie vloženého kapitálu) "
                 "prípadne kompletný výstup.",
    ),
}


def classify(price: float, fair: float) -> dict:
    if price <= fair * BUY_LEVEL:
        return ZONES["BUY"]
    if price <= fair * HOLD_LEVEL:
        return ZONES["HOLD"]
    if price <= fair * EXIT_LEVEL:
        return ZONES["PARTIAL"]
    return ZONES["EXIT"]


# ----------------------------------------------------------------------------
# POMOCNÉ FUNKCIE
# ----------------------------------------------------------------------------
def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _series(df, names):
    """Vráti prvý nájdený riadok z yfinance tabuľky ako časovo vzostupnú Series."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            s.index = pd.to_datetime(s.index)
            return s.sort_index()
    return None


def _last(s) -> float:
    if s is None:
        return np.nan
    s = s.dropna()
    return float(s.iloc[-1]) if len(s) else np.nan


def style_matrix(df: pd.DataFrame, price: float):
    """Zafarbí bunky citlivostnej tabuľky podľa zóny voči aktuálnej cene."""
    def color(v):
        if pd.isna(v) or v <= 0:
            return "background-color:#dddddd;color:#111111"
        ratio = price / v
        if ratio <= BUY_LEVEL:
            bg = ZONES["BUY"]["cell"]
        elif ratio <= HOLD_LEVEL:
            bg = ZONES["HOLD"]["cell"]
        elif ratio <= EXIT_LEVEL:
            bg = ZONES["PARTIAL"]["cell"]
        else:
            bg = ZONES["EXIT"]["cell"]
        return f"background-color:{bg};color:#111111"

    try:
        sty = df.style.map(color)
    except AttributeError:
        sty = df.style.applymap(color)
    return sty.format("{:,.2f}", na_rep="–")


# ----------------------------------------------------------------------------
# NAČÍTANIE DÁT (yfinance)
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Sťahujem dáta z Yahoo Finance…")
def load_ticker(symbol: str) -> dict:
    t = yf.Ticker(symbol)

    def attempt(fn, default=None):
        try:
            return fn()
        except Exception:
            return default

    info = attempt(lambda: t.info, {}) or {}
    price = safe_float(attempt(lambda: t.fast_info["last_price"]))
    hist = attempt(lambda: t.history(period="7y", auto_adjust=False), pd.DataFrame())
    if hist is None:
        hist = pd.DataFrame()
    if not hist.empty:
        if getattr(hist.index, "tz", None) is not None:
            hist.index = hist.index.tz_localize(None)
        if np.isnan(price):
            price = safe_float(hist["Close"].iloc[-1])
    if np.isnan(price):
        price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

    def frame(fn):
        r = attempt(fn, pd.DataFrame())
        return r if isinstance(r, pd.DataFrame) else pd.DataFrame()

    return dict(
        symbol=symbol, info=info, price=price,
        hist=hist[["Close"]] if not hist.empty else pd.DataFrame(),
        cf=frame(lambda: t.cashflow), qcf=frame(lambda: t.quarterly_cashflow),
        inc=frame(lambda: t.income_stmt), qinc=frame(lambda: t.quarterly_income_stmt),
        bs=frame(lambda: t.balance_sheet), qbs=frame(lambda: t.quarterly_balance_sheet),
        fetched=datetime.now().strftime("%d.%m.%Y %H:%M"),
    )


def hist_multiples(raw: dict) -> pd.DataFrame:
    """Ročné P/E, P/S, EV/EBITDA počítané z ceny ku koncu fiškálneho roka (yfinance dáva ~4 roky)."""
    inc, bs, hist = raw["inc"], raw["bs"], raw["hist"]
    eps = _series(inc, ["Diluted EPS"])
    rev = _series(inc, ["Total Revenue"])
    ebitda = _series(inc, ["EBITDA", "Normalized EBITDA"])
    sh = _series(inc, ["Diluted Average Shares"])
    debt = _series(bs, ["Total Debt"])
    cash = _series(bs, CASH_NAMES)
    if eps is None or hist is None or hist.empty:
        return pd.DataFrame()

    def g(s, dt):
        if s is None or dt not in s.index:
            return np.nan
        return safe_float(s.loc[dt])

    rows = []
    for dt in eps.index:
        px = hist["Close"].loc[:dt]
        if px.empty:
            continue
        p = float(px.iloc[-1])
        e, rv, eb, s_ = g(eps, dt), g(rev, dt), g(ebitda, dt), g(sh, dt)
        mcap = p * s_
        ev = mcap + g(debt, dt) - g(cash, dt)
        rows.append({
            "Rok": dt.year, "Cena (koniec FY)": p,
            "P/E": p / e if e > 0 else np.nan,
            "P/S": mcap / rv if rv > 0 else np.nan,
            "EV/EBITDA": ev / eb if eb > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("Rok") if rows else pd.DataFrame()


def derive(raw: dict) -> dict:
    """Z yfinance tabuliek odvodí všetky fundamentálne vstupy (v jednotkách meny, nie v mil.)."""
    info = raw["info"]
    d = {"info": info, "price": raw["price"], "symbol": raw["symbol"]}
    d["name"] = info.get("longName") or info.get("shortName") or raw["symbol"]
    d["sector"] = info.get("sector", "") or ""
    d["industry"] = info.get("industry", "") or ""
    d["currency"] = info.get("currency", "") or ""
    d["fin_currency"] = info.get("financialCurrency", d["currency"]) or d["currency"]

    # --- FCF = Operating Cash Flow - Capital Expenditures
    ocf = _series(raw["cf"], OCF_NAMES)
    cap = _series(raw["cf"], CAPEX_NAMES)
    fcf_row = _series(raw["cf"], ["Free Cash Flow"])
    if ocf is not None and cap is not None:
        fa = pd.DataFrame({"OCF": ocf, "CapEx": cap.abs()}).dropna()
        fa["FCF"] = fa["OCF"] - fa["CapEx"]
    elif fcf_row is not None:
        fa = pd.DataFrame({"FCF": fcf_row.dropna()})
        fa["OCF"], fa["CapEx"] = np.nan, np.nan
    else:
        fa = pd.DataFrame(columns=["OCF", "CapEx", "FCF"], dtype=float)
    fa = fa[["OCF", "CapEx", "FCF"]]
    d["fcf_annual"] = fa

    qocf = _series(raw["qcf"], OCF_NAMES)
    qcap = _series(raw["qcf"], CAPEX_NAMES)
    ttm = np.nan
    if qocf is not None and qcap is not None:
        q = pd.DataFrame({"o": qocf, "c": qcap.abs()}).dropna()
        if len(q) >= 4:
            ttm = float((q["o"] - q["c"]).tail(4).sum())
    if np.isnan(ttm) and len(fa):
        ttm = float(fa["FCF"].iloc[-1])

    opts = {}
    if not np.isnan(ttm):
        opts["TTM (alebo posledný rok)"] = ttm / M
    if len(fa) >= 1:
        opts["Priemer 3 roky"] = float(fa["FCF"].tail(3).mean()) / M
        opts["Priemer 5 rokov (dostupné)"] = float(fa["FCF"].tail(5).mean()) / M
    d["fcf_options"] = opts

    # --- Diluted shares
    shares, shares_src = np.nan, ""
    for key, label in (("qinc", "quarterly"), ("inc", "annual")):
        v = _last(_series(raw[key], ["Diluted Average Shares"]))
        if not np.isnan(v):
            shares, shares_src = v, f"Diluted Average Shares ({label} income statement)"
            break
    if np.isnan(shares):
        shares = safe_float(info.get("sharesOutstanding"))
        shares_src = "sharesOutstanding (Yahoo info – NIE je to diluted, skontroluj!)"
    d["shares"], d["shares_src"] = shares, shares_src

    # --- Dlh a cash
    debt = _last(_series(raw["qbs"], ["Total Debt"]))
    if np.isnan(debt):
        debt = _last(_series(raw["bs"], ["Total Debt"]))
    if np.isnan(debt):
        debt = safe_float(info.get("totalDebt"), 0.0)
    cash = _last(_series(raw["qbs"], CASH_NAMES))
    if np.isnan(cash):
        cash = _last(_series(raw["bs"], CASH_NAMES))
    if np.isnan(cash):
        cash = safe_float(info.get("totalCash"), 0.0)
    d["debt"], d["cash"] = debt, cash

    # --- Per-share údaje
    div = safe_float(info.get("dividendRate"))
    if np.isnan(div):
        div = safe_float(info.get("trailingAnnualDividendRate"), 0.0)
    d["div_rate"] = div

    bv = safe_float(info.get("bookValue"))
    if np.isnan(bv):
        eq = _last(_series(raw["bs"], ["Stockholders Equity", "Common Stock Equity"]))
        bv = eq / shares if not np.isnan(eq) and shares > 0 else np.nan
    d["bvps"] = bv
    roe = safe_float(info.get("returnOnEquity"))
    d["roe"] = roe * 100 if not np.isnan(roe) else np.nan

    eps = safe_float(info.get("trailingEps"))
    if np.isnan(eps):
        eps = _last(_series(raw["inc"], ["Diluted EPS"]))
    d["eps"] = eps
    rev = safe_float(info.get("totalRevenue"))
    if np.isnan(rev):
        rev = _last(_series(raw["inc"], ["Total Revenue"]))
    d["sps"] = rev / shares if not np.isnan(rev) and shares > 0 else np.nan
    eb = safe_float(info.get("ebitda"))
    if np.isnan(eb):
        eb = _last(_series(raw["inc"], ["EBITDA", "Normalized EBITDA"]))
    d["ebitda"] = eb

    # --- Historické násobky
    hm = hist_multiples(raw)
    d["hist_mult"] = hm

    def avg(col):
        if hm.empty or col not in hm:
            return np.nan
        s = hm[col][hm[col] > 0]
        return float(s.mean()) if len(s) else np.nan

    d["avg_pe"], d["avg_ps"], d["avg_ev"] = avg("P/E"), avg("P/S"), avg("EV/EBITDA")

    # --- ROIC (orientačne): NOPAT / Invested Capital
    ebit = _series(raw["inc"], ["EBIT", "Operating Income"])
    tax = _series(raw["inc"], ["Tax Rate For Calcs"])
    ic = _series(raw["bs"], ["Invested Capital"])
    roic = np.nan
    if ebit is not None and ic is not None:
        taxr = tax.reindex(ebit.index).fillna(0.21).clip(0, 0.4) if tax is not None else 0.21
        r_ = (ebit * (1 - taxr) / ic.reindex(ebit.index)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(r_):
            roic = float(r_.mean()) * 100
    d["roic"] = roic

    # --- Default growth (orientačný, investor ho má skontrolovať)
    gr = safe_float(info.get("earningsGrowth"))
    if np.isnan(gr):
        gr = safe_float(info.get("revenueGrowth"))
    d["default_growth"] = round(float(np.clip(gr * 100, 3, 12)), 1) if not np.isnan(gr) else 5.0
    return d


def fcf_negative_3y(d: dict) -> bool:
    last3 = d["fcf_annual"]["FCF"].tail(3).dropna()
    return len(last3) > 0 and bool((last3 < 0).all())


def recommend_model(d: dict):
    sector = d["sector"]
    if sector == "Financial Services":
        return "excess", "Finančný sektor → Excess Returns / NAV"
    if sector in ("Utilities", "Real Estate") and d["div_rate"] > 0:
        return "ddm", f"Sektor {sector} s dividendou → DDM"
    if fcf_negative_3y(d):
        return "multiples", "Záporný FCF 3 roky → DCF nevhodné, odporúčané Multiples"
    if sector in CYCLICAL_SECTORS:
        return "multiples", f"Cyklický sektor ({sector}) → Normalized Multiples"
    return "dcf", "Ziskový podnik s FCF → DCF"


# ----------------------------------------------------------------------------
# VALUAČNÉ JADRÁ (čisté funkcie)
# ----------------------------------------------------------------------------
def dcf_value(fcf, g1, gt, r, net_debt, shares, years=5):
    """Všetky sadzby ako desatinné čísla; fcf/net_debt/shares v rovnakých jednotkách (mil.)."""
    if r <= gt or shares <= 0:
        return None
    rows, pv_sum, f = [], 0.0, fcf
    for y in range(1, years + 1):
        f = f * (1 + g1)
        disc = (1 + r) ** y
        pv = f / disc
        rows.append((y, f, disc, pv))
        pv_sum += pv
    tv = f * (1 + gt) / (r - gt)
    pv_tv = tv / ((1 + r) ** years)
    ev = pv_sum + pv_tv
    eq = ev - net_debt
    return dict(rows=rows, pv_sum=pv_sum, tv=tv, pv_tv=pv_tv, ev=ev, equity=eq, per_share=eq / shares)


def ddm_value(d0, g, r):
    if r <= g:
        return None
    d1 = d0 * (1 + g)
    return dict(d1=d1, per_share=d1 / (r - g))


def excess_value(bv, roe, coe, g):
    if coe <= g:
        return None
    excess = bv * (roe - coe) / (coe - g)
    return dict(excess=excess, per_share=bv + excess)


def matrix(rows, cols, fn, row_fmt="{:.1f} %", col_fmt="{:.1f} %"):
    data = []
    for r_ in rows:
        data.append([fn(r_, c_) for c_ in cols])
    return pd.DataFrame(data, index=[row_fmt.format(x) for x in rows],
                        columns=[col_fmt.format(x) for x in cols], dtype=float)


# ----------------------------------------------------------------------------
# MODELY: vstupy (Manual Override Form) + výpočet
# ----------------------------------------------------------------------------
def num(col, label, value, key, min_value=None, max_value=None, step=1.0, help=None):
    value = float(value) if value is not None and np.isfinite(value) else 0.0
    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return col.number_input(label, value=value, min_value=min_value, max_value=max_value,
                            step=step, format="%.2f", key=key, help=help)


def run_dcf(tk: str, d: dict):
    st.markdown("#### ✍️ Vstupy – DCF (Manual Override Form)")
    if fcf_negative_3y(d):
        st.error("DCF Nerelevantné – Firma nemá stabilný cash flow.")
        st.caption("Posledné 3 ročné FCF sú záporné. Prepni model na Normalized Multiples.")
        return None
    opts = d["fcf_options"]
    if not opts:
        st.warning("Yahoo nevrátil cash-flow dáta – zadaj normalizovaný FCF manuálne (roic.ai / macrotrends).")
        opts = {"Manuálne": 0.0}
    basis = st.radio("Báza normalizovaného FCF", list(opts.keys()), horizontal=True, key=f"{tk}_basis")
    st.caption("Tip: 5-ročný priemer FCF z roic.ai / macrotrends.net je presnejší než Yahoo (ktorý dáva ~4 roky). "
               "Hodnotu môžeš prepísať v poli nižšie.")
    c1, c2, c3 = st.columns(3)
    fcf = num(c1, "Normalizovaný FCF (mil.)", opts[basis], f"{tk}_fcf_{basis}", step=10.0)
    g1 = num(c2, "Expected growth rate, roky 1–5 (%)", d["default_growth"], f"{tk}_g1",
             min_value=-20.0, max_value=60.0, step=0.5)
    gt_in = num(c3, f"Terminal growth (%, max {TERMINAL_CAP})", 2.0, f"{tk}_gt",
                min_value=0.0, max_value=TERMINAL_CAP, step=0.25)
    gt = min(gt_in, TERMINAL_CAP)
    c4, c5, c6, c7 = st.columns(4)
    r = num(c4, "Discount rate (%)", 10.0, f"{tk}_r", min_value=3.0, max_value=30.0, step=0.5)
    shares = num(c5, "Diluted shares outstanding (mil.)", d["shares"] / M if np.isfinite(d["shares"]) else 1.0,
                 f"{tk}_sh", min_value=0.001, step=1.0, help=d["shares_src"])
    debt = num(c6, "Total Debt (mil.)", d["debt"] / M, f"{tk}_debt", min_value=0.0, step=10.0)
    cash = num(c7, "Cash & ekvivalenty (mil.)", d["cash"] / M, f"{tk}_cash", min_value=0.0, step=10.0)
    net_debt = debt - cash
    st.caption(f"Net Debt = Total Debt − Cash = **{net_debt:,.1f} mil.** · Zdroj akcií: {d['shares_src']}")
    roic = num(st.columns(4)[0], "ROIC 5y priemer (%) – informatívne", d["roic"] if np.isfinite(d["roic"]) else 0.0,
               f"{tk}_roic", step=0.5, help="Do výpočtu nevstupuje. Skontroluj na roic.ai.")

    if fcf <= 0:
        st.error("DCF Nerelevantné – Firma nemá stabilný cash flow. (Normalizovaný FCF ≤ 0)")
        return None
    if r <= gt:
        st.error("Discount rate musí byť vyšší než terminal growth.")
        return None

    res = dcf_value(fcf, g1 / 100, gt / 100, r / 100, net_debt, shares)
    steps = pd.DataFrame(
        [(f"Rok {y}", f_, df_, pv) for (y, f_, df_, pv) in res["rows"]],
        columns=["Rok", "FCF (mil.)", "Diskontný faktor (1+r)^t", "Súčasná hodnota PV (mil.)"]).set_index("Rok")
    summary = [
        ("Normalizovaný FCF₀", f"{fcf:,.2f} mil."),
        ("Σ PV(FCF 1–5)", f"{res['pv_sum']:,.2f} mil."),
        ("Terminálna hodnota TV₅", f"{res['tv']:,.2f} mil."),
        ("PV(TV₅)", f"{res['pv_tv']:,.2f} mil."),
        ("Enterprise Value", f"{res['ev']:,.2f} mil."),
        ("− Net Debt", f"{net_debt:,.2f} mil."),
        ("Equity Value", f"{res['equity']:,.2f} mil."),
        ("÷ Diluted shares", f"{shares:,.3f} mil."),
        ("Férová cena / akcia", f"{res['per_share']:,.2f}"),
    ]
    formulas = [
        r"FCF = Operating\ Cash\ Flow - Capital\ Expenditures",
        r"FCF_t = FCF_0\,(1+g)^t \qquad PV_t = \dfrac{FCF_t}{(1+r)^t}\qquad t=1\ldots5",
        r"TV_5 = \dfrac{FCF_5\,(1+g_T)}{r-g_T}\qquad PV(TV)=\dfrac{TV_5}{(1+r)^5}",
        r"Fair\ Value = \dfrac{\sum PV_t + PV(TV) - Net\ Debt}{Diluted\ Shares}",
    ]
    rs = [r - 2, r - 1, r, r + 1, r + 2]
    gs = [g1 - 5, g1 - 2.5, g1, g1 + 2.5, g1 + 5]

    def cell(rr, gg):
        out = dcf_value(fcf, gg / 100, gt / 100, rr / 100, net_debt, shares)
        return out["per_share"] if out else np.nan

    stress = None
    if g1 > STRESS_THRESHOLD:
        s = dcf_value(fcf, STRESS_GROWTH / 100, gt / 100, r / 100, net_debt, shares)
        if s:
            stress = dict(fair=s["per_share"], g=STRESS_GROWTH, requested=g1)
    notes = [f"Terminal growth použitý vo výpočte: {gt:.2f} % (tvrdý strop {TERMINAL_CAP} %)."]
    return dict(fair=res["per_share"], formulas=formulas, tables=[("Projekcia FCF a diskontovanie", steps)],
                summary=summary, sens=(matrix(rs, gs, cell), "Discount rate ↓", "Growth rate (roky 1–5) →"),
                stress=stress, notes=notes)


def run_ddm(tk: str, d: dict):
    st.markdown("#### ✍️ Vstupy – DDM / Gordon Growth (Manual Override Form)")
    c1, c2, c3 = st.columns(3)
    d0 = num(c1, "Ročná dividenda na akciu D₀", d["div_rate"], f"{tk}_d0", min_value=0.0, step=0.1)
    g_in = num(c2, f"Dlhodobý rast dividendy g (%, max {TERMINAL_CAP})", 2.0, f"{tk}_ddm_g",
               min_value=0.0, max_value=TERMINAL_CAP, step=0.25)
    r = num(c3, "Požadovaná výnosnosť r (%)", 10.0, f"{tk}_ddm_r", min_value=3.0, max_value=30.0, step=0.5)
    g = min(g_in, TERMINAL_CAP)
    if d0 <= 0:
        st.error("DDM Nerelevantné – firma neplatí dividendu.")
        return None
    if r <= g:
        st.error("Požadovaná výnosnosť musí byť vyššia než rast dividendy.")
        return None
    res = ddm_value(d0, g / 100, r / 100)
    gs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    rs = [r - 2, r - 1, r, r + 1, r + 2]

    def cell(rr, gg):
        out = ddm_value(d0, gg / 100, rr / 100)
        return out["per_share"] if out else np.nan

    return dict(
        fair=res["per_share"],
        formulas=[r"D_1 = D_0\,(1+g)", r"Fair\ Value = \dfrac{D_1}{r-g}"],
        tables=[],
        summary=[("D₀", f"{d0:,.4f}"), ("g", f"{g:.2f} %"), ("r", f"{r:.2f} %"),
                 ("D₁ = D₀·(1+g)", f"{res['d1']:,.4f}"), ("r − g", f"{r - g:.2f} p.b."),
                 ("Férová cena / akcia", f"{res['per_share']:,.2f}")],
        sens=(matrix(rs, gs, cell), "Required return r ↓", "Dividend growth g →"),
        stress=None, notes=[f"Rast g je obmedzený na max {TERMINAL_CAP} % (perpetuita)."])


def run_excess(tk: str, d: dict):
    st.markdown("#### ✍️ Vstupy – Excess Returns / NAV (Manual Override Form)")
    c1, c2, c3, c4 = st.columns(4)
    bv = num(c1, "Book Value per Share", d["bvps"], f"{tk}_bv", min_value=0.0, step=0.5)
    roe = num(c2, "ROE (%, trvalo udržateľné)", d["roe"] if np.isfinite(d["roe"]) else 10.0, f"{tk}_roe",
              min_value=-50.0, max_value=100.0, step=0.5, help="Skontroluj 5-ročný priemer na macrotrends / roic.ai.")
    coe = num(c3, "Cost of Equity (%)", 10.0, f"{tk}_coe", min_value=3.0, max_value=30.0, step=0.5)
    g_in = num(c4, f"Rast g (%, max {TERMINAL_CAP})", 2.0, f"{tk}_ex_g", min_value=0.0,
               max_value=TERMINAL_CAP, step=0.25)
    g = min(g_in, TERMINAL_CAP)
    if bv <= 0:
        st.error("Excess Returns Nerelevantné – Book Value per Share ≤ 0.")
        return None
    if coe <= g:
        st.error("Cost of Equity musí byť vyšší než g.")
        return None
    if roe > 40:
        st.warning("ROE nad 40 % je pre banky nerealistické – skontroluj vstup.")
    res = excess_value(bv, roe / 100, coe / 100, g / 100)
    gs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    rs = [coe - 2, coe - 1, coe, coe + 1, coe + 2]

    def cell(rr, gg):
        out = excess_value(bv, roe / 100, rr / 100, gg / 100)
        return out["per_share"] if out else np.nan

    return dict(
        fair=res["per_share"],
        formulas=[r"Excess\ Return = BVPS\cdot\dfrac{ROE - CoE}{CoE - g}", r"Fair\ Value = BVPS + Excess\ Return"],
        tables=[],
        summary=[("BVPS", f"{bv:,.2f}"), ("ROE", f"{roe:.2f} %"), ("Cost of Equity", f"{coe:.2f} %"),
                 ("g", f"{g:.2f} %"), ("ROE − CoE", f"{roe - coe:.2f} p.b."),
                 ("Excess Return (hodnota nad BV)", f"{res['excess']:,.2f}"),
                 ("Férová cena / akcia", f"{res['per_share']:,.2f}")],
        sens=(matrix(rs, gs, cell), "Cost of Equity ↓", "Growth g →"),
        stress=None, notes=[])


def run_multiples(tk: str, d: dict):
    st.markdown("#### ✍️ Vstupy – Normalized Multiples (Manual Override Form)")
    st.caption("Priemerné násobky sú predvyplnené z Yahoo dát (cena ku koncu FY, ~4 roky). "
               "Pre presný 5-ročný priemer ich prepíš podľa stockanalysis.com / finviz / macrotrends.")
    c1, c2, c3, c4 = st.columns(4)
    shares = num(c1, "Diluted shares outstanding (mil.)", d["shares"] / M if np.isfinite(d["shares"]) else 1.0,
                 f"{tk}_m_sh", min_value=0.001, step=1.0, help=d["shares_src"])
    debt = num(c2, "Total Debt (mil.)", d["debt"] / M, f"{tk}_m_debt", min_value=0.0, step=10.0)
    cash = num(c3, "Cash & ekvivalenty (mil.)", d["cash"] / M, f"{tk}_m_cash", min_value=0.0, step=10.0)
    net_debt = debt - cash
    c4.metric("Net Debt (mil.)", f"{net_debt:,.1f}")

    def ok(v):
        return bool(np.isfinite(v) and v > 0)

    st.markdown("**Metódy (zaškrtni, ktoré sa majú zahrnúť do priemeru):**")
    a, b, c = st.columns(3)
    use_pe = a.checkbox("P/E × EPS", value=ok(d["avg_pe"]) and ok(d["eps"]), key=f"{tk}_use_pe")
    pe = num(a, "Normalizovaný P/E (5y priemer)", d["avg_pe"], f"{tk}_pe", min_value=0.0, step=0.5)
    eps = num(a, "Normalizované EPS (diluted)", d["eps"], f"{tk}_eps", step=0.1)
    use_ps = b.checkbox("P/S × Sales per share", value=ok(d["avg_ps"]) and ok(d["sps"]), key=f"{tk}_use_ps")
    ps = num(b, "Normalizovaný P/S (5y priemer)", d["avg_ps"], f"{tk}_ps", min_value=0.0, step=0.1)
    sps = num(b, "Sales per share", d["sps"], f"{tk}_sps", step=0.5)
    use_ev = c.checkbox("EV/EBITDA × EBITDA", value=ok(d["avg_ev"]) and ok(d["ebitda"]), key=f"{tk}_use_ev")
    ev_m = num(c, "Normalizovaný EV/EBITDA (5y priemer)", d["avg_ev"], f"{tk}_evm", min_value=0.0, step=0.5)
    ebitda = num(c, "EBITDA (mil.)", d["ebitda"] / M if np.isfinite(d["ebitda"]) else 0.0, f"{tk}_ebitda", step=10.0)

    def blend(fm=1.0, fb=1.0, detail=False):
        vals = {}
        if use_pe and pe > 0 and eps > 0:
            vals["P/E"] = pe * fm * eps * fb
        if use_ps and ps > 0 and sps > 0:
            vals["P/S"] = ps * fm * sps * fb
        if use_ev and ev_m > 0 and ebitda > 0:
            vals["EV/EBITDA"] = (ev_m * fm * ebitda * fb - net_debt) / shares
        if not vals:
            return (np.nan, vals) if detail else np.nan
        mean = float(np.mean(list(vals.values())))
        return (mean, vals) if detail else mean

    fair, vals = blend(detail=True)
    if not vals:
        st.error("Nie je zaškrtnutá žiadna použiteľná metóda (násobok aj základ musia byť > 0).")
        return None
    fm_list = [0.7, 0.85, 1.0, 1.15, 1.3]
    fb_list = [0.8, 0.9, 1.0, 1.1, 1.2]
    sens = matrix([x * 100 - 100 for x in fm_list], [x * 100 - 100 for x in fb_list],
                  lambda rr, cc: blend(1 + rr / 100, 1 + cc / 100), row_fmt="{:+.0f} %", col_fmt="{:+.0f} %")
    method_tbl = pd.DataFrame({"Férová cena / akcia": vals}).rename_axis("Metóda")
    return dict(
        fair=fair,
        formulas=[r"P/E:\ \ P_{fair}=PE_{norm}\cdot EPS_{diluted}",
                  r"P/S:\ \ P_{fair}=PS_{norm}\cdot \dfrac{Sales}{Diluted\ Shares}",
                  r"EV/EBITDA:\ \ P_{fair}=\dfrac{EVEBITDA_{norm}\cdot EBITDA - Net\ Debt}{Diluted\ Shares}",
                  r"Fair\ Value = \text{priemer zahrnutých metód}"],
        tables=[("Férová cena podľa jednotlivých metód", method_tbl)],
        summary=[(f"Metóda {k}", f"{v:,.2f}") for k, v in vals.items()] +
                [("Férová cena (priemer)", f"{fair:,.2f}")],
        sens=(sens, "Zmena násobkov ↓", "Zmena základu (EPS / Sales / EBITDA) →"),
        stress=None,
        notes=["Citlivosť: všetky násobky sa škálujú naraz (riadky) a všetky základy (EPS, Sales, EBITDA) naraz (stĺpce)."])


RUNNERS = {"dcf": run_dcf, "ddm": run_ddm, "excess": run_excess, "multiples": run_multiples}

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("📊 Valuačná kalkulačka – Graham & Buffett")
st.caption("Margin of Safety ±15 % · Diluted shares · FCF = Operating Cash Flow − CapEx")

with st.sidebar:
    st.header("⚙️ Nastavenia")
    ticker = st.text_input("Ticker (napr. AAPL, MSFT, SAP.DE)", value="AAPL").strip().upper()
    weight = st.number_input("Podiel pozície v portfóliu (%)", min_value=0.0, max_value=100.0,
                             value=5.0, step=0.5)
    if st.button("🔄 Obnoviť dáta z Yahoo"):
        st.cache_data.clear()
        st.rerun()
    st.divider()
    st.caption("Fundamenty z Yahoo sú len východiskový bod. Skontroluj ich na roic.ai, macrotrends.net "
               "alebo stockanalysis.com a prepíš vo formulári.")

if not ticker:
    st.info("Zadaj ticker v ľavom paneli.")
    st.stop()

raw = load_ticker(ticker)
price = raw["price"]
if np.isnan(price) or price <= 0:
    st.error(f"Pre ticker „{ticker}“ sa nepodarilo načítať cenu. Skontroluj symbol (pri nie-US akciách pridaj "
             "príponu burzy, napr. .DE, .L).")
    st.stop()

d = derive(raw)
tk = ticker

h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
h1.subheader(f"{d['name']} ({tk})")
h1.caption(f"{d['sector'] or 'n/a'} · {d['industry'] or 'n/a'} · dáta načítané {raw['fetched']}")
h2.metric("Živá cena", f"{price:,.2f} {d['currency']}")
h3.metric("Diluted shares (mil.)", f"{d['shares'] / M:,.2f}" if np.isfinite(d["shares"]) else "n/a")
h4.metric("FCF TTM (mil.)", f"{d['fcf_options'].get('TTM (alebo posledný rok)', float('nan')):,.0f}"
          if d["fcf_options"] else "n/a")

if d["fin_currency"] and d["currency"] and d["fin_currency"] != d["currency"]:
    st.warning(f"Cena je v {d['currency']}, ale finančné výkazy v {d['fin_currency']} (typické pri ADR). "
               "Pred výpočtom prepočítaj fundamenty na menu ceny, inak bude férová cena skreslená.")
if "NIE je to diluted" in d["shares_src"]:
    st.warning("Diluted Average Shares sa nepodarilo načítať – použitý je základný počet akcií. "
               "Prepíš ho podľa roic.ai / macrotrends.")

rec, reason = recommend_model(d)
keys = list(MODEL_LABELS.keys())
model = st.selectbox("Valuačný model", keys, index=keys.index(rec), format_func=lambda k: MODEL_LABELS[k])
st.caption(f"Odporúčanie aplikácie: **{MODEL_LABELS[rec]}** – {reason}")

with st.expander("🔎 Zdrojové dáta z Yahoo Finance (na kontrolu)", expanded=False):
    fa = d["fcf_annual"]
    if len(fa):
        show = (fa / M).round(1)
        show.index = [i.year for i in show.index]
        show.columns = ["Operating CF (mil.)", "CapEx (mil.)", "FCF (mil.)"]
        st.markdown("**Ročné FCF = Operating CF − CapEx**")
        st.dataframe(show, use_container_width=True)
    else:
        st.write("Ročné cash-flow dáta nie sú dostupné.")
    if not d["hist_mult"].empty:
        st.markdown("**Historické násobky (cena ku koncu fiškálneho roka)**")
        st.dataframe(d["hist_mult"].round(2), use_container_width=True)
    st.write({"Total Debt (mil.)": round(d["debt"] / M, 1), "Cash (mil.)": round(d["cash"] / M, 1),
              "Dividenda/akcia": d["div_rate"], "BVPS": d["bvps"], "ROE %": d["roe"],
              "ROIC % (odhad)": d["roic"], "Zdroj akcií": d["shares_src"]})

st.divider()
result = RUNNERS[model](tk, d)
if result is None:
    st.stop()

fair = result["fair"]
if not np.isfinite(fair) or fair <= 0:
    st.error("Model vyšiel so zápornou / neplatnou férovou cenou – model nie je pre túto firmu relevantný "
             "alebo sú vstupy chybné. Skús iný model alebo uprav vstupy.")
    st.stop()

# --- DASHBOARD -------------------------------------------------------------
st.divider()
st.markdown("## 🎯 Výsledkový dashboard")
z = classify(price, fair)
st.markdown(
    f"""<div style="background:{z['color']};padding:22px;border-radius:14px;color:white">
    <div style="font-size:14px;opacity:.85">{html.escape(d['name'])} ({html.escape(tk)}) · model: {MODEL_LABELS[model]}</div>
    <div style="font-size:30px;font-weight:700;margin:4px 0">{z['label']}</div>
    <div style="font-size:16px">{z['status']}</div></div>""",
    unsafe_allow_html=True)
st.write("")
m1, m2, m3, m4 = st.columns(4)
dev = (price / fair - 1) * 100
m1.metric("Trhová cena", f"{price:,.2f}")
m2.metric("Férová cena (Intrinsic Value)", f"{fair:,.2f}")
m3.metric("Cena vs. férová cena", f"{dev:+.1f} %")
m4.metric("Nákupná hranica (−15 %)", f"{fair * BUY_LEVEL:,.2f}")

ladder = pd.DataFrame([
    ["BUY", "cena ≤ FV × 0.85", f"≤ {fair * BUY_LEVEL:,.2f}", "BUY"],
    ["HOLD", "FV × 0.85 – FV × 1.15", f"{fair * BUY_LEVEL:,.2f} – {fair * HOLD_LEVEL:,.2f}", "HOLD"],
    ["PARTIAL SELL", "FV × 1.15 – FV × 1.35", f"{fair * HOLD_LEVEL:,.2f} – {fair * EXIT_LEVEL:,.2f}", "PARTIAL"],
    ["FULL EXIT", "cena > FV × 1.35", f"> {fair * EXIT_LEVEL:,.2f}", "EXIT"],
], columns=["Zóna", "Podmienka", "Cenové pásmo", "k"])
ladder["Aktuálne"] = np.where(ladder["k"] == z["key"], "◀ TU SME", "")
st.table(ladder.drop(columns="k").set_index("Zóna"))

st.markdown("### 📝 Dodatok k rebalansovaniu / odpredaju")
st.text_area("Odporúčanie", value=z["addendum"], height=130, disabled=True, label_visibility="collapsed",
             key=f"add_{tk}_{z['key']}_{model}")
if z["key"] == "PARTIAL":
    if weight > 15:
        st.warning(f"Tvoja pozícia tvorí {weight:.1f} % portfólia (>15 %) → podmienka je splnená, "
                   "zváž odpredaj 15–25 % pozície.")
    else:
        st.info(f"Tvoja pozícia tvorí {weight:.1f} % portfólia (≤15 %) → odpredaj nie je podmienený, "
                "stačí pozastaviť nové nákupy.")

if result["stress"]:
    s = result["stress"]
    zs = classify(price, s["fair"]) if s["fair"] > 0 else None
    st.warning(f"⚠️ High Growth Stress Test: zadaný rast {s['requested']:.1f} % presahuje {STRESS_THRESHOLD:.0f} %. "
               f"Konzervatívny scenár ({s['g']:.0f} % rast) dáva férovú cenu **{s['fair']:,.2f}**"
               + (f" → zóna **{zs['label']}**." if zs else " (neplatná hodnota)."))
    if zs:
        st.text_area("Dodatok – konzervatívny scenár", value=zs["addendum"], height=110, disabled=True,
                     key=f"add_stress_{tk}_{zs['key']}")
for n in result["notes"]:
    st.caption("ℹ️ " + n)

# --- SENSITIVITY MATRIX -----------------------------------------------------
st.divider()
st.markdown("## 🔥 Tabuľka citlivosti (Sensitivity Matrix) – férová cena / akcia")
sens_df, row_t, col_t = result["sens"]
st.caption(f"Riadky: {row_t} · Stĺpce: {col_t} · Farba bunky = zóna aktuálnej ceny "
           f"({price:,.2f}) voči danej férovej cene (zelená BUY, modrá HOLD, oranžová PARTIAL SELL, červená EXIT).")
st.dataframe(style_matrix(sens_df, price), use_container_width=True)

# --- ROZPAD VÝPOČTU ---------------------------------------------------------
st.divider()
st.markdown("## 🧮 Kompletný rozpad výpočtu")
st.markdown(f"**Model:** {MODEL_LABELS[model]}")
for f in result["formulas"]:
    st.latex(f)
for title, tbl in result["tables"]:
    st.markdown(f"**{title}**")
    st.dataframe(tbl.style.format("{:,.2f}"), use_container_width=True)
st.markdown("**Krok za krokom**")
st.table(pd.DataFrame(result["summary"], columns=["Krok", "Hodnota"]).set_index("Krok"))

st.markdown("**Margin of Safety – klasifikácia**")
st.latex(rf"\frac{{Cena}}{{FV}} = \frac{{{price:,.2f}}}{{{fair:,.2f}}} = {price / fair:.3f}"
         rf"\ \Rightarrow\ \text{{{z['label']}}}")
st.table(pd.DataFrame({
    "Prah": ["BUY: FV × 0.85", "HOLD horná hranica: FV × 1.15", "PARTIAL SELL horná hranica: FV × 1.35"],
    "Výpočet": [f"{fair:,.2f} × 0.85", f"{fair:,.2f} × 1.15", f"{fair:,.2f} × 1.35"],
    "Cena": [f"{fair * BUY_LEVEL:,.2f}", f"{fair * HOLD_LEVEL:,.2f}", f"{fair * EXIT_LEVEL:,.2f}"],
}).set_index("Prah"))

st.caption("⚠️ Nástroj slúži na vlastnú analýzu, nie je investičným odporúčaním. Dáta z yfinance sú neoficiálne "
           "a môžu byť neúplné – kľúčové vstupy (FCF 5y priemer, ROIC, diluted shares) over na roic.ai / macrotrends.net.")
