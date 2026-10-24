"""
Valuačná kalkulačka akcií – Graham & Buffett (Streamlit + yfinance)  ·  verzia 2

Inštalácia:   pip install streamlit yfinance pandas numpy jinja2
Spustenie:    streamlit run valuation_app.py

Modely: DCF | DDM (Gordon / dvojfázový) | Excess Returns / NAV | Normalized Multiples
Pásma:  BUY (<= FV*0.85) | HOLD (FV ± 15 %) | PARTIAL SELL (> FV*1.15) | FULL EXIT (> FV*1.35)

Novinky v2 (presnosť a bezpečnosť):
  1. Konzistentné zaobchádzanie s dlhom (FCFE / FCFF / pôvodné zadanie) + voliteľný odpočet SBC
  2. Kontrola kvality dát, automatický prepočet meny výkazov (ADR), oprava GBp
  3. Postupné spomalenie rastu (roky 6–10), kontrola reálnosti rastu, podiel terminálnej hodnoty
  4. Pomocník pre discount rate (CAPM, Blumeho beta, podlaha, WACC)
  5. Bear/Base/Bull pásmo, robustnosť zóny a reverzný DCF (čo implikuje trhová cena)
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
FADE_YEARS = 5              # roky 6–10: lineárne spomalenie z g na terminálny rast
DISC_FLOOR = 8.0            # % – podlaha pre cost of equity v pomocníkovi
TV_SHARE_WARN = 0.75        # podiel terminálnej hodnoty, nad ktorý varujeme
TTM_KEY = "TTM (alebo posledný rok)"

OCF_NAMES = ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities",
             "Total Cash From Operating Activities"]
CAPEX_NAMES = ["Capital Expenditure", "Capital Expenditures"]
SBC_NAMES = ["Stock Based Compensation"]
CASH_NAMES = ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]
CYCLICAL_SECTORS = ("Energy", "Basic Materials", "Consumer Cyclical")

MODEL_LABELS = {
    "dcf": "DCF (Discounted Cash Flow)",
    "ddm": "DDM (Gordon Growth)",
    "excess": "Excess Returns / NAV",
    "multiples": "Normalized Multiples",
}

MODE_FCFE = "FCFE – FCF po úrokoch, BEZ odpočtu Net Debt (diskontuj cost of equity)"
MODE_FCFF = "FCFF – FCF + čistý úrok po dani, odpočet Net Debt (diskontuj WACC)"
MODE_LEGACY = "Pôvodné zadanie – FCF a odpočet Net Debt (dlh sa započíta 2×, konzervatívne)"
MODES = [MODE_FCFE, MODE_FCFF, MODE_LEGACY]

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


def num(col, label, value, key, min_value=None, max_value=None, step=1.0, help=None, disabled=False):
    """number_input, ktorý nekoliduje so Session State (hodnotu z value dáva len pri prvom vykreslení)."""
    kwargs = dict(min_value=min_value, max_value=max_value, step=step, format="%.2f",
                  key=key, help=help, disabled=disabled)
    if key not in st.session_state:
        v = float(value) if value is not None and np.isfinite(value) else 0.0
        if min_value is not None:
            v = max(v, min_value)
        if max_value is not None:
            v = min(v, max_value)
        kwargs["value"] = v
    return col.number_input(label, **kwargs)


def matrix(rows, cols, fn, row_fmt="{:.1f} %", col_fmt="{:.1f} %"):
    data = [[fn(r_, c_) for c_ in cols] for r_ in rows]
    return pd.DataFrame(data, index=[row_fmt.format(x) for x in rows],
                        columns=[col_fmt.format(x) for x in cols], dtype=float)


def bisect(fn, target, lo, hi, increasing=True, iters=80):
    """Nájde x v <lo, hi> s fn(x) = target pre monotónnu fn. Vráti float alebo 'low'/'high' (mimo rozsahu)."""
    flo, fhi = fn(lo), fn(hi)
    if not (np.isfinite(flo) and np.isfinite(fhi)):
        return None
    if increasing:
        if target < flo:
            return "low"
        if target > fhi:
            return "high"
    else:
        if target > flo:
            return "low"
        if target < fhi:
            return "high"
    for _ in range(iters):
        mid = (lo + hi) / 2
        fm = fn(mid)
        if not np.isfinite(fm):
            return None
        if (fm < target) == increasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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

    divs = attempt(lambda: t.dividends, pd.Series(dtype=float))
    if divs is None:
        divs = pd.Series(dtype=float)
    if len(divs) and getattr(divs.index, "tz", None) is not None:
        divs.index = divs.index.tz_localize(None)

    # Londýnske akcie: Yahoo uvádza cenu v pencoch (GBp), výkazy v librách (GBP)
    gbp_fix = False
    if info.get("currency") == "GBp":
        gbp_fix = True
        if np.isfinite(price):
            price = price / 100
        if not hist.empty:
            hist = hist.copy()
            hist["Close"] = hist["Close"] / 100
        if len(divs):
            divs = divs / 100
        info = dict(info)
        info["currency"] = "GBP"

    return dict(
        symbol=symbol, info=info, price=price, divs=divs, gbp_fix=gbp_fix,
        hist=hist[["Close"]] if not hist.empty else pd.DataFrame(),
        cf=frame(lambda: t.cashflow), qcf=frame(lambda: t.quarterly_cashflow),
        inc=frame(lambda: t.income_stmt), qinc=frame(lambda: t.quarterly_income_stmt),
        bs=frame(lambda: t.balance_sheet), qbs=frame(lambda: t.quarterly_balance_sheet),
        fetched=datetime.now().strftime("%d.%m.%Y %H:%M"),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def get_10y_yield() -> float:
    """Výnos 10-ročného US dlhopisu (^TNX) v %, alebo NaN."""
    try:
        v = safe_float(yf.Ticker("^TNX").fast_info["last_price"])
        if np.isfinite(v) and 20 < v < 200:   # staršie kotácie Yahoo = výnos × 10
            v = v / 10
        if np.isfinite(v) and 0 < v < 20:
            return float(v)
    except Exception:
        pass
    return np.nan


@st.cache_data(ttl=3600, show_spinner=False)
def get_fx(fin: str, cur: str) -> float:
    """Kurz: koľko jednotiek meny `cur` je 1 jednotka meny `fin` (napr. EUR→USD = EURUSD=X)."""
    try:
        v = safe_float(yf.Ticker(f"{fin}{cur}=X").fast_info["last_price"])
        if np.isfinite(v) and v > 0:
            return float(v)
    except Exception:
        pass
    return np.nan


def hist_multiples(raw: dict, fx: float = 1.0) -> pd.DataFrame:
    """Ročné P/E, P/S, EV/EBITDA z ceny ku koncu fiškálneho roka (yfinance dáva ~4 roky)."""
    inc, bs, hist = raw["inc"], raw["bs"], raw["hist"]
    eps = _series(inc, ["Diluted EPS"])
    rev = _series(inc, ["Total Revenue"])
    ebitda = _series(inc, ["EBITDA", "Normalized EBITDA"])
    sh = _series(inc, ["Diluted Average Shares"])
    debt = _series(bs, ["Total Debt"])
    cash = _series(bs, CASH_NAMES)
    if eps is None or hist is None or hist.empty:
        return pd.DataFrame()

    def g(s, dt, mult=1.0):
        if s is None or dt not in s.index:
            return np.nan
        return safe_float(s.loc[dt]) * mult

    rows = []
    for dt in eps.index:
        px = hist["Close"].loc[:dt]
        if px.empty:
            continue
        p = float(px.iloc[-1])
        e, rv, eb = g(eps, dt, fx), g(rev, dt, fx), g(ebitda, dt, fx)
        s_ = g(sh, dt)
        mcap = p * s_
        ev = mcap + g(debt, dt, fx) - g(cash, dt, fx)
        rows.append({
            "Rok": dt.year, "Cena (koniec FY)": p,
            "P/E": p / e if e > 0 else np.nan,
            "P/S": mcap / rv if rv > 0 else np.nan,
            "EV/EBITDA": ev / eb if eb > 0 else np.nan,
        })
    return pd.DataFrame(rows).set_index("Rok") if rows else pd.DataFrame()


def derive(raw: dict, fx: float = 1.0) -> dict:
    """Z yfinance tabuliek odvodí všetky fundamentálne vstupy. Sumy výkazov sú prepočítané kurzom fx."""
    info = raw["info"]
    d = {"info": info, "price": raw["price"], "symbol": raw["symbol"], "fx": fx,
         "gbp_fix": raw.get("gbp_fix", False)}
    d["name"] = info.get("longName") or info.get("shortName") or raw["symbol"]
    d["sector"] = info.get("sector", "") or ""
    d["industry"] = info.get("industry", "") or ""
    d["currency"] = info.get("currency", "") or ""
    d["fin_currency"] = info.get("financialCurrency", d["currency"]) or d["currency"]
    d["beta"] = safe_float(info.get("beta"))
    d["mcap_info"] = safe_float(info.get("marketCap"))

    # --- FCF = Operating Cash Flow - Capital Expenditures (+ SBC pre voliteľný odpočet)
    ocf = _series(raw["cf"], OCF_NAMES)
    cap = _series(raw["cf"], CAPEX_NAMES)
    fcf_row = _series(raw["cf"], ["Free Cash Flow"])
    sbc = _series(raw["cf"], SBC_NAMES)
    if ocf is not None and cap is not None:
        fa = pd.DataFrame({"OCF": ocf, "CapEx": cap.abs()}).dropna()
        fa["FCF"] = fa["OCF"] - fa["CapEx"]
    elif fcf_row is not None:
        fa = pd.DataFrame({"FCF": fcf_row.dropna()})
        fa["OCF"], fa["CapEx"] = np.nan, np.nan
    else:
        fa = pd.DataFrame(columns=["OCF", "CapEx", "FCF"], dtype=float)
    fa = fa[["OCF", "CapEx", "FCF"]].astype(float)
    fa["SBC"] = sbc.reindex(fa.index).abs() if sbc is not None else np.nan
    fa = fa * fx
    fa["FCF_SBC"] = fa["FCF"] - fa["SBC"].fillna(0)
    d["sbc_available"] = bool(fa["SBC"].notna().any())
    d["fcf_annual"] = fa

    qocf = _series(raw["qcf"], OCF_NAMES)
    qcap = _series(raw["qcf"], CAPEX_NAMES)
    ttm, ttm_from_q = np.nan, False
    if qocf is not None and qcap is not None:
        q = pd.DataFrame({"o": qocf, "c": qcap.abs()}).dropna()
        if len(q) >= 4:
            ttm, ttm_from_q = float((q["o"] - q["c"]).tail(4).sum()) * fx, True
    if not np.isfinite(ttm) and len(fa):
        ttm = float(fa["FCF"].iloc[-1])
    ttm_sbc = np.nan
    if ttm_from_q:
        qs = _series(raw["qcf"], SBC_NAMES)
        if qs is not None and len(qs.dropna()) >= 4:
            ttm_sbc = float(qs.dropna().abs().tail(4).sum()) * fx
    if not np.isfinite(ttm_sbc):
        ttm_sbc = float(fa["SBC"].iloc[-1]) if len(fa) and np.isfinite(fa["SBC"].iloc[-1]) else 0.0

    def build_opts(col, ttm_value):
        o = {}
        if np.isfinite(ttm_value):
            o[TTM_KEY] = ttm_value / M
        if len(fa):
            o["Priemer 3 roky"] = float(fa[col].tail(3).mean()) / M
            o["Priemer 5 rokov (dostupné)"] = float(fa[col].tail(5).mean()) / M
        return o

    d["fcf_options"] = build_opts("FCF", ttm)
    d["fcf_options_sbc"] = build_opts("FCF_SBC", ttm - ttm_sbc if np.isfinite(ttm) else np.nan)

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
    d["debt"], d["cash"] = debt * fx, cash * fx

    # --- Čistý úrokový náklad (TTM) a daň – pre FCFF
    def ttm_or_last(key_q, key_a, names):
        s = _series(raw[key_q], names)
        if s is not None:
            v = s.dropna()
            if len(v) >= 4:
                return float(v.tail(4).sum())
        return _last(_series(raw[key_a], names))

    ie = ttm_or_last("qinc", "inc", ["Interest Expense"])
    ii = ttm_or_last("qinc", "inc", ["Interest Income"])
    d["gross_interest"] = abs(ie) * fx if np.isfinite(ie) else np.nan
    d["net_interest"] = (abs(ie) - (abs(ii) if np.isfinite(ii) else 0.0)) * fx if np.isfinite(ie) else np.nan
    tr = _last(_series(raw["inc"], ["Tax Rate For Calcs"]))
    d["tax_rate"] = float(np.clip(tr, 0, 0.4)) if np.isfinite(tr) else 0.21

    # --- Per-share údaje
    div = safe_float(info.get("dividendRate"))
    if np.isnan(div):
        div = safe_float(info.get("trailingAnnualDividendRate"), 0.0)
    d["div_rate"] = div

    bv = safe_float(info.get("bookValue")) * fx
    if np.isnan(bv):
        eq = _last(_series(raw["bs"], ["Stockholders Equity", "Common Stock Equity"])) * fx
        bv = eq / shares if not np.isnan(eq) and shares > 0 else np.nan
    d["bvps"] = bv
    roe = safe_float(info.get("returnOnEquity"))
    d["roe"] = roe * 100 if not np.isnan(roe) else np.nan

    eps = safe_float(info.get("trailingEps")) * fx
    if np.isnan(eps):
        eps = _last(_series(raw["inc"], ["Diluted EPS"])) * fx
    d["eps"] = eps
    rev = safe_float(info.get("totalRevenue"))
    if np.isnan(rev):
        rev = _last(_series(raw["inc"], ["Total Revenue"]))
    d["sps"] = rev * fx / shares if not np.isnan(rev) and shares > 0 else np.nan
    eb = safe_float(info.get("ebitda"))
    if np.isnan(eb):
        eb = _last(_series(raw["inc"], ["EBITDA", "Normalized EBITDA"]))
    d["ebitda"] = eb * fx

    # --- Historické násobky
    hm = hist_multiples(raw, fx)
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

    # --- Priemerný dividendový výnos (5 plných kalendárnych rokov) a FCF CAGR
    d["avg_div_yield"] = np.nan
    divs, hh = raw.get("divs"), raw["hist"]
    if divs is not None and len(divs) and hh is not None and not hh.empty:
        ys = []
        cy = datetime.now().year
        for y in range(cy - 5, cy):
            dsum = float(divs[divs.index.year == y].sum())
            px = hh["Close"][hh.index.year == y]
            if dsum > 0 and len(px):
                ys.append(dsum / float(px.mean()) * 100)
        if ys:
            d["avg_div_yield"] = float(np.mean(ys))
    d["fcf_cagr"], d["fcf_cagr_label"] = np.nan, ""
    fs = fa["FCF"].dropna()
    if len(fs) >= 2 and fs.iloc[0] > 0 and fs.iloc[-1] > 0:
        yrs = (fs.index[-1] - fs.index[0]).days / 365.25
        if yrs > 0:
            d["fcf_cagr"] = ((fs.iloc[-1] / fs.iloc[0]) ** (1 / yrs) - 1) * 100
            d["fcf_cagr_label"] = f"{fs.index[0].year}–{fs.index[-1].year}"

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


def data_quality(d: dict, price: float) -> list:
    """Zoznam (úroveň, správa) – kontrola, či sú načítané dáta použiteľné a konzistentné."""
    out = []
    fa = d["fcf_annual"]
    if fa.empty or fa["FCF"].dropna().empty:
        out.append(("warning", "Chýbajú ročné cash-flow dáta – FCF a ďalšie vstupy zadaj ručne podľa roic.ai / macrotrends."))
    else:
        if len(fa) < 3:
            out.append(("warning", f"Yahoo vrátil len {len(fa)} roky FCF – normalizácia je nespoľahlivá, over na roic.ai / macrotrends."))
        age = (pd.Timestamp.now() - fa.index[-1]).days / 30.4
        if age > 15:
            out.append(("warning", f"Posledný ročný výkaz je z {fa.index[-1].date()} (~{age:.0f} mes. starý) – dáta môžu byť zastarané."))
        m = float(fa["FCF"].mean())
        if len(fa) >= 3 and m > 0:
            cv = float(fa["FCF"].std() / m)
            if cv > 0.35:
                out.append(("warning", f"FCF kolíše (variačný koeficient {cv:.2f}) – nepoužívaj TTM, ale viacročný priemer."))
    shares, mcap = d["shares"], d["mcap_info"]
    if np.isfinite(shares) and np.isfinite(mcap) and price > 0:
        implied = mcap / price
        if implied > 0 and abs(shares / implied - 1) > 0.10:
            out.append(("warning", f"Diluted shares ({shares / M:,.0f} mil.) sa líšia od marketCap/cena ({implied / M:,.0f} mil.) "
                                   f"o {abs(shares / implied - 1) * 100:.0f} % – možné viaceré triedy akcií alebo ADR pomer. Skontroluj počet akcií."))
    if "NIE je to diluted" in d["shares_src"]:
        out.append(("warning", "Diluted Average Shares sa nepodarilo načítať – použitý je základný počet akcií."))
    if d["fin_currency"] and d["currency"] and d["fin_currency"] != d["currency"]:
        if d["fx"] != 1.0:
            out.append(("info", f"Výkazy sú v {d['fin_currency']}, cena v {d['currency']}: sumy z výkazov sú prepočítané kurzom {d['fx']:.4f}. "
                                f"Dividenda sa neprepočítava (Yahoo ju uvádza v mene obchodovania)."))
        else:
            out.append(("warning", f"Výkazy sú v {d['fin_currency']}, cena v {d['currency']}, ale kurz je 1,0 – férová cena bude skreslená. "
                                   "Zadaj kurz v bočnom paneli."))
    if d["currency"] in ("ILA", "ZAc"):
        out.append(("warning", f"Mena {d['currency']} je v podjednotkách (agorot / centy) – ceny a výkazy nemusia byť v rovnakej škále."))
    if d["gbp_fix"]:
        out.append(("info", "Cena a história boli prepočítané z pencí (GBp) na libry (GBP). Over výšku dividendy."))
    return out


# ----------------------------------------------------------------------------
# VALUAČNÉ JADRÁ (čisté funkcie; sadzby ako desatinné čísla)
# ----------------------------------------------------------------------------
def dcf_value(fcf, g1, gt, r, net_debt, shares, fade=True, years=5):
    """5 rokov rastu g1; ak fade, ďalších 5 rokov lineárne k terminálnemu rastu gt; potom Gordon."""
    if r <= gt or shares <= 0:
        return None
    rates = [g1] * years
    if fade:
        rates += [g1 + (gt - g1) * i / FADE_YEARS for i in range(1, FADE_YEARS + 1)]
    rows, pv_sum, f = [], 0.0, fcf
    for y, g in enumerate(rates, start=1):
        f = f * (1 + g)
        disc = (1 + r) ** y
        pv = f / disc
        rows.append((y, g, f, disc, pv))
        pv_sum += pv
    n = len(rates)
    tv = f * (1 + gt) / (r - gt)
    pv_tv = tv / ((1 + r) ** n)
    ev = pv_sum + pv_tv
    eq = ev - net_debt
    return dict(rows=rows, n=n, pv_sum=pv_sum, tv=tv, pv_tv=pv_tv, ev=ev, equity=eq,
                per_share=eq / shares, tv_share=pv_tv / ev if ev > 0 else np.nan)


def ddm_value(d0, g, r, g1=None, years=5):
    """Gordon (g1=None) alebo dvojfázový DDM (g1 počas `years` rokov, potom g)."""
    if r <= g:
        return None
    if g1 is None:
        d1 = d0 * (1 + g)
        return dict(d1=d1, rows=[], per_share=d1 / (r - g))
    rows, pv_sum, dd = [], 0.0, d0
    for y in range(1, years + 1):
        dd *= (1 + g1)
        disc = (1 + r) ** y
        rows.append((y, dd, disc, dd / disc))
        pv_sum += dd / disc
    tv = dd * (1 + g) / (r - g)
    pv_tv = tv / ((1 + r) ** years)
    return dict(rows=rows, pv_sum=pv_sum, tv=tv, pv_tv=pv_tv, per_share=pv_sum + pv_tv)


def excess_value(bv, roe, coe, g):
    if coe <= g:
        return None
    excess = bv * (roe - coe) / (coe - g)
    return dict(excess=excess, per_share=bv + excess)


# ----------------------------------------------------------------------------
# POMOCNÍK PRE DISCOUNT RATE (CAPM / WACC)
# ----------------------------------------------------------------------------
MODE_CODES = {MODE_FCFE: "fcfe", MODE_FCFF: "fcff", MODE_LEGACY: "legacy"}


def estimate_rates(d: dict, price: float, rf: float, beta: float, erp: float) -> dict:
    """CAPM (surová aj Blumeho beta), cost of equity s podlahou a odhad WACC. Všetko v %."""
    beta_adj = 0.67 * beta + 0.33
    capm_raw = rf + beta * erp
    capm_adj = rf + beta_adj * erp
    ke = max(DISC_FLOOR, capm_adj)
    wacc = np.nan
    mcap = price * d["shares"] if np.isfinite(d["shares"]) else np.nan
    if np.isfinite(mcap) and d["debt"] > 0 and np.isfinite(d["gross_interest"]):
        kd = d["gross_interest"] / d["debt"] * 100
        wacc = (mcap * ke + d["debt"] * kd * (1 - d["tax_rate"])) / (mcap + d["debt"])
    return dict(beta_adj=beta_adj, capm_raw=capm_raw, capm_adj=capm_adj, ke=ke, wacc=wacc)


def _quarter(x: float) -> float:
    return float(np.clip(round(x * 4) / 4, 3.0, 30.0))


def render_discount_helper(tk: str, d: dict, model: str, price: float):
    mode = st.session_state.get(f"{tk}_mode", MODE_FCFE)
    code = MODE_CODES.get(mode, "fcfe")
    target = {"dcf": f"{tk}_r_{code}", "ddm": f"{tk}_ddm_r", "excess": f"{tk}_coe"}.get(model)
    if target is None:
        return
    use_wacc = model == "dcf" and mode == MODE_FCFF

    def _set(key, val):
        st.session_state[key] = val

    with st.expander("🧭 Pomocník pre discount rate (CAPM / WACC)", expanded=False):
        tnx = get_10y_yield()
        c1, c2, c3 = st.columns(3)
        rf = num(c1, "Bezrizikový výnos – 10Y US (%)", tnx if np.isfinite(tnx) else 4.0, f"{tk}_rf",
                 min_value=0.0, max_value=15.0, step=0.05,
                 help="Ak sa výnos nepodarilo načítať, je tu len zástupná hodnota – zadaj aktuálny.")
        beta = num(c2, "Beta (Yahoo)", d["beta"] if np.isfinite(d["beta"]) else 1.0, f"{tk}_beta",
                   min_value=-1.0, max_value=3.0, step=0.05)
        erp = num(c3, "Rizikové zdvihnutie trhu (%)", 5.0, f"{tk}_erp", min_value=3.0, max_value=8.0, step=0.25)
        if not np.isfinite(tnx):
            st.caption("⚠️ Výnos 10Y dlhopisu sa nepodarilo načítať – zadaj ho ručne.")
        if not np.isfinite(d["beta"]):
            st.caption("⚠️ Beta nie je dostupná – použitá je 1,0.")
        est = estimate_rates(d, price, rf, beta, erp)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("CAPM (surová beta)", f"{est['capm_raw']:.2f} %")
        m2.metric("CAPM (Blumeho beta)", f"{est['capm_adj']:.2f} %",
                  help=f"Upravená beta = 0,67 × beta + 0,33 = {est['beta_adj']:.2f}")
        m3.metric(f"Cost of equity (min. {DISC_FLOOR:.0f} %)", f"{est['ke']:.2f} %")
        m4.metric("WACC (odhad)", f"{est['wacc']:.2f} %" if np.isfinite(est["wacc"]) else "n/a")
        st.caption("Podlaha 8 % bráni tomu, aby nízka beta (typicky defenzívne firmy) nafúkla férovú cenu. "
                   "Cost of equity patrí k FCFE, DDM a Excess Returns; WACC k FCFF (pozri režim DCF).")
        if use_wacc and np.isfinite(est["wacc"]):
            st.button(f"Použiť WACC = {_quarter(est['wacc']):.2f} %", on_click=_set,
                      args=(target, _quarter(est["wacc"])), key=f"{tk}_btn_wacc")
        else:
            st.button(f"Použiť cost of equity = {_quarter(est['ke']):.2f} %", on_click=_set,
                      args=(target, _quarter(est["ke"])), key=f"{tk}_btn_ke")


# ----------------------------------------------------------------------------
# MODELY: vstupy (Manual Override Form) + výpočet
# ----------------------------------------------------------------------------
def run_dcf(tk: str, d: dict, price: float):
    st.markdown("#### ✍️ Vstupy – DCF (Manual Override Form)")
    if fcf_negative_3y(d):
        st.error("DCF Nerelevantné – Firma nemá stabilný cash flow.")
        st.caption("Posledné 3 ročné FCF sú záporné. Prepni model na Normalized Multiples.")
        return None

    ca, cb = st.columns([3, 1])
    mode = ca.radio("Spôsob zohľadnenia dlhu (musí zodpovedať typu diskontnej sadzby)", MODES, key=f"{tk}_mode")
    use_sbc = cb.checkbox("Odpočítať SBC od FCF", value=False, key=f"{tk}_sbc", disabled=not d["sbc_available"],
                          help="Stock-based compensation je skutočný náklad, ktorý OCF pripočítava späť. "
                               "Odporúčané pri tech firmách.")
    use_sbc = use_sbc and d["sbc_available"]
    opts = d["fcf_options_sbc"] if use_sbc else d["fcf_options"]
    if not opts:
        st.warning("Yahoo nevrátil cash-flow dáta – zadaj normalizovaný FCF manuálne (roic.ai / macrotrends).")
        opts = {"Manuálne": 0.0}
    basis = st.radio("Báza normalizovaného FCF", list(opts.keys()), horizontal=True, key=f"{tk}_basis")
    st.caption("Tip: 5-ročný priemer FCF z roic.ai / macrotrends.net je presnejší než Yahoo (~4 roky). "
               "Hodnotu môžeš prepísať v poli nižšie.")
    c1, c2, c3 = st.columns(3)
    fcf = num(c1, "Normalizovaný FCF (mil.)", opts[basis], f"{tk}_fcf_{basis}_{int(use_sbc)}", step=10.0)
    g1 = num(c2, "Expected growth rate, roky 1–5 (%)", d["default_growth"], f"{tk}_g1",
             min_value=-20.0, max_value=60.0, step=0.5)
    gt_in = num(c3, f"Terminal growth (%, max {TERMINAL_CAP})", 2.0, f"{tk}_gt",
                min_value=0.0, max_value=TERMINAL_CAP, step=0.25)
    gt = min(gt_in, TERMINAL_CAP)
    c4, c5, c6, c7 = st.columns(4)
    code = MODE_CODES[mode]
    r_default = 10.0
    if mode == MODE_FCFF:
        _tnx = get_10y_yield()
        _est = estimate_rates(d, price, _tnx if np.isfinite(_tnx) else 4.0,
                              d["beta"] if np.isfinite(d["beta"]) else 1.0, 5.0)
        r_default = _quarter(_est["wacc"]) if np.isfinite(_est["wacc"]) else 8.0
    r = num(c4, "Discount rate (%)" + (" – WACC" if mode == MODE_FCFF else " – cost of equity"), r_default,
            f"{tk}_r_{code}", min_value=3.0, max_value=30.0, step=0.5,
            help="FCFF → WACC; FCFE a pôvodné zadanie → cost of equity. Pomocník je nad formulárom.")
    shares = num(c5, "Diluted shares outstanding (mil.)", d["shares"] / M if np.isfinite(d["shares"]) else 1.0,
                 f"{tk}_sh", min_value=0.001, step=1.0, help=d["shares_src"])
    debt = num(c6, "Total Debt (mil.)", d["debt"] / M, f"{tk}_debt", min_value=0.0, step=10.0)
    cash = num(c7, "Cash & ekvivalenty (mil.)", d["cash"] / M, f"{tk}_cash", min_value=0.0, step=10.0)
    net_debt = debt - cash
    fade = st.checkbox("Postupné spomalenie rastu v rokoch 6–10 (lineárne z rastu rokov 1–5 na terminálny rast)",
                       value=True, key=f"{tk}_fade",
                       help="Bez tohto prepínača rast z roka 5 na rok 6 skokovo klesne na terminálny rast.")

    int_at = 0.0
    if mode == MODE_FCFF:
        ni = d["net_interest"]
        default_int = ni * (1 - d["tax_rate"]) / M if np.isfinite(ni) else 0.0
        int_at = num(st.columns(4)[0], "Čistý úrok po dani (mil.) – pripočíta sa k FCF", default_int,
                     f"{tk}_int", step=10.0,
                     help=f"Čistý úrokový náklad × (1 − daň {d['tax_rate'] * 100:.0f} %). Prepíš podľa výkazov.")
        if not np.isfinite(ni):
            st.warning("Úrokový náklad sa nenašiel – pripočítaný úrok je 0, FCFF je preto podhodnotené.")
    nd_used = 0.0 if mode == MODE_FCFE else net_debt
    fcf_used = fcf + int_at

    if mode == MODE_FCFE:
        st.caption("FCFE: FCF už je po úrokoch, preto sa dlh neodpočítava. Použi **cost of equity** "
                   f"· Net Debt = {net_debt:,.1f} mil. (len informatívne)")
    elif mode == MODE_FCFF:
        st.caption(f"FCFF: FCF + úrok po dani = {fcf_used:,.1f} mil.; od hodnoty firmy sa odpočíta Net Debt "
                   f"{net_debt:,.1f} mil. Použi **WACC**.")
    else:
        st.caption(f"Pôvodné zadanie: odpočíta sa Net Debt {net_debt:,.1f} mil. od FCF, ktorý už je po úrokoch → "
                   "dlh sa započíta dvakrát (konzervatívny odhad).")
    st.caption(f"Zdroj akcií: {d['shares_src']}")
    roic_col = st.columns(4)[0]
    num(roic_col, "ROIC 5y priemer (%) – informatívne", d["roic"] if np.isfinite(d["roic"]) else 0.0,
        f"{tk}_roic", step=0.5, help="Do výpočtu nevstupuje. Skontroluj na roic.ai.")

    if fcf_used <= 0:
        st.error("DCF Nerelevantné – Firma nemá stabilný cash flow. (Použitý FCF ≤ 0)")
        return None
    if r <= gt:
        st.error("Discount rate musí byť vyšší než terminal growth.")
        return None

    def val(rr, gg, f=fcf_used):
        out = dcf_value(f, gg / 100, gt / 100, rr / 100, nd_used, shares, fade)
        return out["per_share"] if out else np.nan

    res = dcf_value(fcf_used, g1 / 100, gt / 100, r / 100, nd_used, shares, fade)
    steps = pd.DataFrame(
        [(f"Rok {y}", g * 100, f_, df_, pv) for (y, g, f_, df_, pv) in res["rows"]],
        columns=["Rok", "Rast FCF (%)", "FCF (mil.)", "Diskontný faktor (1+r)^t", "Súčasná hodnota PV (mil.)"]
    ).set_index("Rok")
    n = res["n"]
    summary = [
        ("Normalizovaný FCF", f"{fcf:,.2f} mil."),
    ]
    if mode == MODE_FCFF:
        summary.append(("+ Čistý úrok po dani", f"{int_at:,.2f} mil."))
    summary += [
        ("FCF₀ použitý v modeli", f"{fcf_used:,.2f} mil."),
        (f"Σ PV(FCF 1–{n})", f"{res['pv_sum']:,.2f} mil."),
        (f"Terminálna hodnota TV{n}", f"{res['tv']:,.2f} mil."),
        (f"PV(TV{n})", f"{res['pv_tv']:,.2f} mil. ({res['tv_share'] * 100:.0f} % hodnoty)"),
        ("Hodnota firmy / equity pred dlhom", f"{res['ev']:,.2f} mil."),
        ("− Net Debt (použitý)", f"{nd_used:,.2f} mil."),
        ("Equity Value", f"{res['equity']:,.2f} mil."),
        ("÷ Diluted shares", f"{shares:,.3f} mil."),
        ("Férová cena / akcia", f"{res['per_share']:,.2f}"),
    ]
    formulas = [
        r"FCF = Operating\ Cash\ Flow - Capital\ Expenditures" + (r" - SBC" if use_sbc else ""),
        r"FCF_0 = FCF_{norm}" + (r" + Interest_{net}\,(1-t)" if mode == MODE_FCFF else ""),
        r"FCF_t = FCF_{t-1}\,(1+g_t)\qquad PV_t = \dfrac{FCF_t}{(1+r)^t}",
        (r"g_t = g\ (t=1\ldots5),\qquad g_t = g + (g_T - g)\cdot\dfrac{t-5}{5}\ (t=6\ldots10)" if fade
         else r"g_t = g\ (t=1\ldots5)"),
        rf"TV_{{{n}}} = \dfrac{{FCF_{{{n}}}\,(1+g_T)}}{{r-g_T}}\qquad PV(TV)=\dfrac{{TV_{{{n}}}}}{{(1+r)^{{{n}}}}}",
        r"Fair\ Value = \dfrac{\sum PV_t + PV(TV) - Net\ Debt_{použitý}}{Diluted\ Shares}",
    ]
    rs = [r - 2, r - 1, r, r + 1, r + 2]
    gs = [g1 - 5, g1 - 2.5, g1, g1 + 2.5, g1 + 5]

    stress = None
    if g1 > STRESS_THRESHOLD:
        s = dcf_value(fcf_used, STRESS_GROWTH / 100, gt / 100, r / 100, nd_used, shares, fade)
        if s:
            stress = dict(fair=s["per_share"], g=STRESS_GROWTH, requested=g1)

    # Reverzný DCF: aký rast implikuje trhová cena?
    imp = bisect(lambda g: val(r, g * 100), price, -0.30, 0.80, increasing=True)
    imp = None if imp is None else imp
    implied = None
    if isinstance(imp, float):
        implied = (f"Reverzný DCF: pri cene {price:,.2f}, r = {r:.1f} % a terminálnom raste {gt:.1f} % trh diskontuje "
                   f"ročný rast FCF ≈ **{imp * 100:.1f} %** počas rokov 1–5. Tvoj odhad: {g1:.1f} %.")
    elif imp == "low":
        implied = "Reverzný DCF: aj pri raste −30 % ročne je férová cena nad trhovou – cena je výrazne pod modelom."
    elif imp == "high":
        implied = "Reverzný DCF: ani pri raste +80 % ročne model nedosiahne trhovú cenu – pri týchto vstupoch nie je obhájiteľná."

    warnings = []
    hist_g = d["fcf_cagr"]
    if np.isfinite(hist_g) and g1 > max(8.0, hist_g * 1.5):
        warnings.append(f"Zadaný rast {g1:.1f} % je výrazne nad historickým FCF CAGR {hist_g:+.1f} % ({d['fcf_cagr_label']}). "
                        "Over, že ho firma udrží 5 rokov.")
    if np.isfinite(res["tv_share"]) and res["tv_share"] > TV_SHARE_WARN:
        warnings.append(f"Terminálna hodnota tvorí {res['tv_share'] * 100:.0f} % férovej ceny – výsledok je silne "
                        "citlivý na discount rate a terminálny rast.")
    if r - gt < 3:
        warnings.append("Medzera r − g_T je menšia než 3 p.b. – férová cena je extrémne citlivá na vstupy.")
    if mode == MODE_FCFF and r > 12:
        warnings.append("Pri FCFF sa očakáva WACC (zvyčajne nižší než cost of equity) – skontroluj, či nepoužívaš príliš vysokú sadzbu.")

    notes = [f"Terminal growth použitý vo výpočte: {gt:.2f} % (tvrdý strop {TERMINAL_CAP} %).",
             {"FCFE": "Režim FCFE: dlh sa neodpočítava.", "FCFF": "Režim FCFF: úrok po dani pripočítaný, Net Debt odpočítaný.",
              "Pôvodné": "Režim pôvodného zadania: dlh je započítaný dvakrát."}[mode.split(" ")[0]]]
    lo, hi = val(r + 1, g1 - 2.5), val(r - 1, g1 + 2.5)
    return dict(fair=res["per_share"], formulas=formulas, tables=[("Projekcia FCF a diskontovanie", steps)],
                summary=summary, sens=(matrix(rs, gs, val), "Discount rate ↓", "Growth rate (roky 1–5) →"),
                stress=stress, notes=notes, range=(lo, hi), implied=implied, warnings=warnings)


def run_ddm(tk: str, d: dict, price: float):
    st.markdown("#### ✍️ Vstupy – DDM (Manual Override Form)")
    c1, c2, c3 = st.columns(3)
    d0 = num(c1, "Ročná dividenda na akciu D₀", d["div_rate"], f"{tk}_d0", min_value=0.0, step=0.1)
    g_in = num(c2, f"Dlhodobý rast dividendy g (%, max {TERMINAL_CAP})", 2.0, f"{tk}_ddm_g",
               min_value=0.0, max_value=TERMINAL_CAP, step=0.25)
    r = num(c3, "Požadovaná výnosnosť r (%)", 10.0, f"{tk}_ddm_r", min_value=3.0, max_value=30.0, step=0.5)
    g = min(g_in, TERMINAL_CAP)
    two = st.checkbox("Dvojfázový DDM (vyšší rast dividendy v rokoch 1–5, potom g)", value=False, key=f"{tk}_ddm_two")
    g1 = None
    if two:
        g1 = num(st.columns(3)[0], "Rast dividendy v rokoch 1–5 (%)", float(np.clip(d["default_growth"], 0, 15)),
                 f"{tk}_ddm_g1", min_value=-10.0, max_value=15.0, step=0.5) / 100
    if d0 <= 0:
        st.error("DDM Nerelevantné – firma neplatí dividendu.")
        return None
    if r <= g:
        st.error("Požadovaná výnosnosť musí byť vyššia než rast dividendy.")
        return None

    warnings = []
    shares_, ttm, eps = d["shares"], d["fcf_options"].get(TTM_KEY, np.nan), d["eps"]
    if np.isfinite(shares_) and np.isfinite(ttm) and ttm > 0:
        payout = d0 * shares_ / (ttm * M)
        if payout > 0.9:
            warnings.append(f"Dividendy tvoria {payout * 100:.0f} % FCF – dividenda je blízko hranice udržateľnosti, "
                            "priestor na rast je malý.")
    if np.isfinite(eps) and eps > 0 and d0 > eps:
        warnings.append("Dividenda je vyššia než EPS – dlhodobo neudržateľné.")
    if r - g < 3:
        warnings.append("Medzera r − g je menšia než 3 p.b. – férová cena je extrémne citlivá na vstupy.")
    if d["sector"] not in ("Utilities", "Real Estate", "Consumer Defensive", "Financial Services", "") and not two:
        warnings.append(f"DDM je najvhodnejší pre stabilných dividendových platiteľov; sektor {d['sector']} "
                        "môže byť lepšie oceniť cez DCF.")

    def val(rr, gg, gg1=g1):
        out = ddm_value(d0, gg / 100, rr / 100, gg1)
        return out["per_share"] if out else np.nan

    res = ddm_value(d0, g / 100, r / 100, g1)
    gs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    rs = [r - 2, r - 1, r, r + 1, r + 2]
    tables, summary = [], [("D₀", f"{d0:,.4f}"), ("g", f"{g:.2f} %"), ("r", f"{r:.2f} %")]
    if g1 is None:
        formulas = [r"D_1 = D_0\,(1+g)", r"Fair\ Value = \dfrac{D_1}{r-g}"]
        summary += [("D₁ = D₀·(1+g)", f"{res['d1']:,.4f}"), ("r − g", f"{r - g:.2f} p.b.")]
    else:
        formulas = [r"D_t = D_{t-1}(1+g_1)\ (t=1\ldots5)\qquad PV_t = \dfrac{D_t}{(1+r)^t}",
                    r"TV_5 = \dfrac{D_5\,(1+g)}{r-g}\qquad Fair\ Value = \sum PV_t + \dfrac{TV_5}{(1+r)^5}"]
        tables.append(("Dividendy rokov 1–5", pd.DataFrame(
            [(f"Rok {y}", dd, df_, pv) for (y, dd, df_, pv) in res["rows"]],
            columns=["Rok", "Dividenda", "Diskontný faktor", "PV"]).set_index("Rok")))
        summary += [("Σ PV(dividend 1–5)", f"{res['pv_sum']:,.4f}"), ("PV terminálnej hodnoty", f"{res['pv_tv']:,.4f}")]
    summary.append(("Férová cena / akcia", f"{res['per_share']:,.2f}"))

    # Reverzný DDM: aká požadovaná výnosnosť je v cene?
    imp = bisect(lambda x: ddm_value(d0, g / 100, x, g1)["per_share"] if x > g / 100 else np.nan,
                 price, g / 100 + 0.0005, 0.40, increasing=False)
    implied = None
    if isinstance(imp, float):
        implied = (f"Reverzný DDM: pri cene {price:,.2f} a raste g = {g:.1f} % je v cene zabudovaná požadovaná "
                   f"výnosnosť ≈ **{imp * 100:.1f} %**. Tvoja požadovaná výnosnosť: {r:.1f} %.")
    return dict(fair=res["per_share"], formulas=formulas, tables=tables, summary=summary,
                sens=(matrix(rs, gs, val), "Required return r ↓", "Dividend growth g →"),
                stress=None, notes=[f"Rast g je obmedzený na max {TERMINAL_CAP} % (perpetuita)."],
                range=(val(r + 1, max(g - 1, 0)), val(r - 1, min(g + 0.5, TERMINAL_CAP))),
                implied=implied, warnings=warnings)


def run_excess(tk: str, d: dict, price: float):
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
    warnings = []
    if roe > 40:
        warnings.append("ROE nad 40 % je pre banky nerealistické – skontroluj vstup.")
    if d["sector"] != "Financial Services":
        warnings.append("Excess Returns je určený pre banky a poisťovne; pre nefinančné firmy je BVPS často zavádzajúci.")
    res = excess_value(bv, roe / 100, coe / 100, g / 100)
    gs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    rs = [coe - 2, coe - 1, coe, coe + 1, coe + 2]

    def val(rr, gg):
        out = excess_value(bv, roe / 100, rr / 100, gg / 100)
        return out["per_share"] if out else np.nan

    implied = None
    if roe / 100 > g / 100:
        imp = bisect(lambda x: excess_value(bv, roe / 100, x, g / 100)["per_share"] if x > g / 100 else np.nan,
                     price, g / 100 + 0.0005, 0.40, increasing=False)
        if isinstance(imp, float):
            implied = (f"Reverzný model: pri cene {price:,.2f} je v cene zabudovaný cost of equity ≈ "
                       f"**{imp * 100:.1f} %** (tvoj: {coe:.1f} %).")
    return dict(
        fair=res["per_share"],
        formulas=[r"Excess\ Return = BVPS\cdot\dfrac{ROE - CoE}{CoE - g}", r"Fair\ Value = BVPS + Excess\ Return"],
        tables=[],
        summary=[("BVPS", f"{bv:,.2f}"), ("ROE", f"{roe:.2f} %"), ("Cost of Equity", f"{coe:.2f} %"),
                 ("g", f"{g:.2f} %"), ("ROE − CoE", f"{roe - coe:.2f} p.b."),
                 ("Excess Return (hodnota nad BV)", f"{res['excess']:,.2f}"),
                 ("Férová cena / akcia", f"{res['per_share']:,.2f}")],
        sens=(matrix(rs, gs, val), "Cost of Equity ↓", "Growth g →"),
        stress=None, notes=[], range=(val(coe + 1, max(g - 1, 0)), val(coe - 1, min(g + 0.5, TERMINAL_CAP))),
        implied=implied, warnings=warnings)


def run_multiples(tk: str, d: dict, price: float):
    st.markdown("#### ✍️ Vstupy – Normalized Multiples (Manual Override Form)")
    n_years = 0 if d["hist_mult"].empty else len(d["hist_mult"])
    st.caption(f"Priemerné násobky sú predvyplnené z Yahoo dát (cena ku koncu FY, {n_years} rokov). "
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
    warnings = []
    if n_years and n_years < 4:
        warnings.append(f"Priemerné násobky sú len z {n_years} rokov – prepíš ich 5-ročným priemerom z externého zdroja.")
    spread = max(vals.values()) / min(vals.values()) if min(vals.values()) > 0 else np.inf
    if spread > 1.5:
        warnings.append("Jednotlivé metódy sa výrazne rozchádzajú (viac než 50 %) – priemer môže byť zavádzajúci, "
                        "pozri tabuľku metód.")
    warnings.append("Násobky z minulosti predpokladajú, že trh bude firmu oceňovať podobne ako predtým. "
                    "Priemer môže obsahovať obdobie s inými úrokovými sadzbami a prémiou.")
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
        notes=["Citlivosť: všetky násobky sa škálujú naraz (riadky) a všetky základy (EPS, Sales, EBITDA) naraz (stĺpce)."],
        range=(blend(0.85, 0.9), blend(1.15, 1.1)), implied=None, warnings=warnings)


RUNNERS = {"dcf": run_dcf, "ddm": run_ddm, "excess": run_excess, "multiples": run_multiples}


# ----------------------------------------------------------------------------
# KRÍŽOVÁ KONTROLA (P/E, P/S, EV/EBITDA, FCF yield, dividenda, payout, dlh, rast, ROIC)
# ----------------------------------------------------------------------------
def build_cross_check(d: dict, price: float):
    """Vráti (DataFrame s metrikami, počty valuačných signálov {'G','Y','R'}). Počíta z Yahoo dát."""
    rows, counts = [], {"G": 0, "Y": 0, "R": 0}

    def add(metric, cur, ref, status, note="", sig=None):
        rows.append({"Metrika": metric, "Aktuálne": cur, "Referencia": ref,
                     "Hodnotenie": status, "Poznámka": note})
        if sig in counts:
            counts[sig] += 1

    shares = d["shares"]
    mcap = price * shares if np.isfinite(shares) and shares > 0 else np.nan
    hm = d["hist_mult"]

    def hist_range(col):
        if hm.empty or col not in hm:
            return ""
        s = hm[col][hm[col] > 0]
        return f"rozsah {s.min():.1f}–{s.max():.1f} z {len(s)} r." if len(s) else ""

    def vs_avg(label, cur, avg, col, fmt):
        if not (np.isfinite(cur) and np.isfinite(avg) and avg > 0):
            add(label, "n/a", "n/a", "⚪ nedostupné")
            return
        ratio = cur / avg
        if ratio <= 0.85:
            sig, txt = "G", "🟢 pod priemerom"
        elif ratio <= 1.15:
            sig, txt = "Y", "🟡 okolo priemeru"
        else:
            sig, txt = "R", "🔴 nad priemerom"
        add(label, fmt.format(cur), fmt.format(avg) + " (priemer)", f"{txt} ({(ratio - 1) * 100:+.0f} %)",
            hist_range(col), sig)

    eps, sps, ebitda = d["eps"], d["sps"], d["ebitda"]
    cur_pe = price / eps if np.isfinite(eps) and eps > 0 else np.nan
    cur_ps = price / sps if np.isfinite(sps) and sps > 0 else np.nan
    ev = mcap + d["debt"] - d["cash"] if np.isfinite(mcap) else np.nan
    cur_ev = ev / ebitda if np.isfinite(ev) and np.isfinite(ebitda) and ebitda > 0 else np.nan

    vs_avg("P/E (cena / EPS)", cur_pe, d["avg_pe"], "P/E", "{:.1f}×")
    vs_avg("P/S (cena / tržby na akciu)", cur_ps, d["avg_ps"], "P/S", "{:.2f}×")
    vs_avg("EV/EBITDA", cur_ev, d["avg_ev"], "EV/EBITDA", "{:.1f}×")

    ttm = d["fcf_options"].get(TTM_KEY, np.nan)
    if np.isfinite(ttm) and np.isfinite(mcap) and mcap > 0:
        fcf_y = ttm * M / mcap * 100
        tnx = get_10y_yield()
        if np.isfinite(tnx):
            spread = fcf_y - tnx
            sig, txt = ("G", "🟢 nad výnosom dlhopisu") if spread >= 0 else \
                       (("Y", "🟡 tesne pod dlhopisom") if spread >= -1.5 else ("R", "🔴 výrazne pod dlhopisom"))
            add("FCF yield (FCF / market cap)", f"{fcf_y:.2f} %", f"{tnx:.2f} % (10Y US dlhopis)",
                f"{txt} ({spread:+.2f} p.b.)", "Prah ±1,5 p.b. je orientačný.", sig)
        else:
            add("FCF yield (FCF / market cap)", f"{fcf_y:.2f} %", "n/a", "⚪ výnos dlhopisu nedostupný")

    div = d["div_rate"]
    if np.isfinite(div) and div > 0:
        cur_dy = div / price * 100
        avg_dy = d["avg_div_yield"]
        if np.isfinite(avg_dy) and avg_dy > 0:
            ratio = cur_dy / avg_dy
            sig, txt = ("G", "🟢 vyšší než priemer") if ratio >= 1.15 else \
                       (("Y", "🟡 okolo priemeru") if ratio >= 0.85 else ("R", "🔴 nižší než priemer"))
            add("Dividendový výnos", f"{cur_dy:.2f} %", f"{avg_dy:.2f} % (5-ročný priemer)",
                f"{txt} ({(ratio - 1) * 100:+.0f} %)", "Vyšší výnos pri rovnakej dividende = nižšia cena.", sig)
        else:
            add("Dividendový výnos", f"{cur_dy:.2f} %", "n/a", "⚪ história nedostupná")
        if np.isfinite(eps) and eps > 0:
            p = div / eps * 100
            st_ = "🟢 zdravé" if p <= 60 else ("🟡 vyššie" if p <= 80 else "🔴 vysoké")
            add("Payout ratio (dividenda / EPS)", f"{p:.0f} %", "≤ 60 % komfortné", st_, "Orientačné prahy.")
        if np.isfinite(ttm) and ttm > 0 and np.isfinite(shares):
            p = div * shares / (ttm * M) * 100
            st_ = "🟢 zdravé" if p <= 60 else ("🟡 vyššie" if p <= 80 else "🔴 vysoké")
            add("Payout ratio (dividendy / FCF)", f"{p:.0f} %", "≤ 60 % komfortné", st_,
                "Čím vyšší, tým menej priestoru na zvyšovanie dividendy.")

    if d["sector"] != "Financial Services" and np.isfinite(ebitda) and ebitda > 0:
        nde = (d["debt"] - d["cash"]) / ebitda
        st_ = "🟢 nízke" if nde <= 2 else ("🟡 stredné" if nde <= 3.5 else "🔴 vysoké")
        add("Net Debt / EBITDA", f"{nde:.2f}×", "≤ 2× nízke", st_, "Orientačné prahy.")

    if np.isfinite(d["fcf_cagr"]):
        c = d["fcf_cagr"]
        st_ = "🟢 rastie" if c >= 3 else ("🟡 stagnuje" if c >= 0 else "🔴 klesá")
        add(f"FCF CAGR ({d['fcf_cagr_label']})", f"{c:+.1f} % ročne", "≥ 3 % rastie", st_,
            "Počítané len z dostupných ročných dát Yahoo.")
    if np.isfinite(d["roic"]):
        r_ = d["roic"]
        st_ = "🟢 silný" if r_ >= 15 else ("🟡 priemerný" if r_ >= 8 else "🔴 slabý")
        add("ROIC (odhad, priemer)", f"{r_:.1f} %", "≥ 15 % silný", st_, "Orientačný odhad z Yahoo.")

    return pd.DataFrame(rows), counts


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

tk = ticker

# Kurz výkazov → mena ceny (ADR a pod.)
fx = 1.0
_fin = raw["info"].get("financialCurrency") or ""
_cur = raw["info"].get("currency") or ""
if _fin and _cur and _fin != _cur and _cur not in ("ILA", "ZAc"):
    _fx_auto = get_fx(_fin, _cur)
    with st.sidebar:
        st.divider()
        st.markdown(f"**Kurz {_fin} → {_cur}**")
        fx = st.number_input(f"1 {_fin} = ? {_cur}", min_value=0.0001,
                             value=float(_fx_auto) if np.isfinite(_fx_auto) else 1.0,
                             step=0.01, format="%.4f", key=f"{tk}_fx",
                             help="Sumy z výkazov (FCF, dlh, cash, EPS, EBITDA) sa vynásobia týmto kurzom.")

d = derive(raw, fx)

h1, h2, h3, h4 = st.columns([3, 1, 1, 1])
h1.subheader(f"{d['name']} ({tk})")
h1.caption(f"{d['sector'] or 'n/a'} · {d['industry'] or 'n/a'} · dáta načítané {raw['fetched']}")
h2.metric("Živá cena", f"{price:,.2f} {d['currency']}")
h3.metric("Diluted shares (mil.)", f"{d['shares'] / M:,.2f}" if np.isfinite(d["shares"]) else "n/a")
h4.metric("FCF TTM (mil.)", f"{d['fcf_options'][TTM_KEY]:,.0f}" if TTM_KEY in d["fcf_options"] else "n/a")

dq = data_quality(d, price)
n_warn = sum(1 for lvl, _ in dq if lvl == "warning")
with st.expander(f"🩺 Kvalita dát: {n_warn} upozornení" if n_warn else "🩺 Kvalita dát: bez upozornení",
                 expanded=n_warn > 0):
    if not dq:
        st.success("Pri automatických kontrolách sa nezistil žiadny problém.")
    for lvl, msg in dq:
        (st.warning if lvl == "warning" else st.info)(msg)

rec, reason = recommend_model(d)
keys = list(MODEL_LABELS.keys())
model = st.selectbox("Valuačný model", keys, index=keys.index(rec), format_func=lambda k: MODEL_LABELS[k])
st.caption(f"Odporúčanie aplikácie: **{MODEL_LABELS[rec]}** – {reason}")

with st.expander("🔎 Zdrojové dáta z Yahoo Finance (na kontrolu)", expanded=False):
    fa = d["fcf_annual"]
    if len(fa):
        show = (fa[["OCF", "CapEx", "FCF", "SBC"]] / M).round(1)
        show.index = [i.year for i in show.index]
        show.columns = ["Operating CF (mil.)", "CapEx (mil.)", "FCF (mil.)", "SBC (mil.)"]
        st.markdown("**Ročné FCF = Operating CF − CapEx**")
        st.dataframe(show, use_container_width=True)
    else:
        st.write("Ročné cash-flow dáta nie sú dostupné.")
    if not d["hist_mult"].empty:
        st.markdown("**Historické násobky (cena ku koncu fiškálneho roka)**")
        st.dataframe(d["hist_mult"].round(2), use_container_width=True)
    st.write({"Total Debt (mil.)": round(d["debt"] / M, 1), "Cash (mil.)": round(d["cash"] / M, 1),
              "Čistý úrok TTM (mil.)": None if not np.isfinite(d["net_interest"]) else round(d["net_interest"] / M, 1),
              "Dividenda/akcia": d["div_rate"], "BVPS": d["bvps"], "ROE %": d["roe"],
              "ROIC % (odhad)": d["roic"], "Beta": d["beta"], "Zdroj akcií": d["shares_src"]})

st.divider()
render_discount_helper(tk, d, model, price)
result = RUNNERS[model](tk, d, price)
if result is None:
    st.stop()

fair = result["fair"]
if not np.isfinite(fair) or fair <= 0:
    st.error("Model vyšiel so zápornou / neplatnou férovou cenou – model nie je pre túto firmu relevantný "
             "alebo sú vstupy chybné. Skús iný model alebo uprav vstupy.")
    st.stop()

for w in result["warnings"]:
    st.warning("⚠️ " + w)

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

# --- Bear / Base / Bull a robustnosť zóny
lo_hi = result.get("range")
if lo_hi and all(np.isfinite(x) and x > 0 for x in lo_hi):
    lo, hi = min(lo_hi), max(lo_hi)
    zl, zh = classify(price, lo), classify(price, hi)
    st.markdown("### 🧪 Bear / Base / Bull a robustnosť zóny")
    st.table(pd.DataFrame([
        ["Bear (opatrnejšie vstupy)", f"{lo:,.2f}", zl["label"]],
        ["Base (tvoje vstupy)", f"{fair:,.2f}", z["label"]],
        ["Bull (optimistickejšie vstupy)", f"{hi:,.2f}", zh["label"]],
    ], columns=["Scenár", "Férová cena", "Zóna"]).set_index("Scenár"))
    if zl["key"] == z["key"] == zh["key"]:
        st.success("Robustný signál: zóna je rovnaká vo všetkých troch scenároch.")
    else:
        st.info("Zóna závisí od vstupov: pri opatrnejších / optimistickejších predpokladoch sa mení. "
                "Rozhodnutie neopieraj o jediné číslo.")
if result.get("implied"):
    st.info("🔁 " + result["implied"])

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

# --- KRÍŽOVÁ KONTROLA ------------------------------------------------------
st.divider()
st.markdown("## 🔍 Krížová kontrola (P/E, FCF, dividenda, dlh)")
st.caption("Počíta sa z dát Yahoo (nie z tvojich override polí). Priemery sú len z dostupných ročných dát "
           "(~4 roky), prahy hodnotenia sú orientačné. Slúži na kontrolu, či zónu z modelu potvrdzujú aj "
           "iné metriky.")
cc_df, cc_cnt = build_cross_check(d, price)
if cc_df.empty:
    st.info("Nepodarilo sa spočítať žiadnu kontrolnú metriku.")
else:
    st.dataframe(cc_df.astype(str).set_index("Metrika"), use_container_width=True)
    g, y, rd = cc_cnt["G"], cc_cnt["Y"], cc_cnt["R"]
    total = g + y + rd
    if total:
        st.markdown(f"**Valuačné signály** (P/E, P/S, EV/EBITDA, FCF yield, dividendový výnos): "
                    f"🟢 {g} lacnejšie · 🟡 {y} neutrálne · 🔴 {rd} drahšie")
    if total >= 3:
        if z["key"] in ("PARTIAL", "EXIT") and g > rd:
            st.info("Model ukazuje nadhodnotenie, ale väčšina metrík je lacnejšia než vlastná história. "
                    "Rozpor často znamená, že model je pri tvojich vstupoch (najmä discount rate a rast) prísny, "
                    "alebo že historický priemer obsahuje obdobie s vyššou prémiou. Skontroluj oboje.")
        elif z["key"] in ("PARTIAL", "EXIT") and rd >= g:
            st.warning("Nadhodnotenie z modelu potvrdzuje aj väčšina kontrolných metrík.")
        elif z["key"] == "BUY" and rd > g:
            st.warning("Model hovorí BUY, ale väčšina metrík je drahšia než história. "
                       "Skontroluj, či nie sú vstupy modelu (rast, discount rate) príliš optimistické.")
        elif z["key"] == "BUY":
            st.success("Podhodnotenie z modelu potvrdzuje aj väčšina kontrolných metrík.")
        else:
            st.info("Zóna HOLD: kontrolné metriky ju nijako výrazne nespochybňujú ani nepotvrdzujú.")

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
