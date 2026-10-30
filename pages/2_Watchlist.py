"""
Watchlist – sledované a plánované nákupy

Umiestnenie v repozitári:  pages/2_Watchlist.py   (Streamlit ho sám pridá do menu vľavo)
Hlavný súbor valuation_app.py a Screener sa nemenia. Nové knižnice netreba.

Čo robí: do tabuľky zapíšeš ticker, férovú cenu (z hlavnej aplikácie), svoj margin of safety a/alebo vlastnú cieľovú
nákupnú cenu. Stránka načíta živé ceny z Yahoo a pri každej akcii ukáže stav:
🟢 NÁKUP (cena je na alebo pod nákupnou cenou), 🟡 BLÍZKO, ⚪ čakám.

DÔLEŽITÉ:
- Upozornenie funguje len, keď je táto stránka otvorená v prehliadači (aplikácia nevie poslať správu, keď je zatvorená).
- Dáta sa ukladajú do súboru watchlist.csv v aplikácii. Na Streamlit Cloud sa po reštarte môže stratiť,
  preto si pravidelne sťahuj zálohu (CSV) a po reštarte ju nahraj.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Watchlist", page_icon="🔔", layout="wide")

WL_FILE = "watchlist.csv"
COLS = ["Ticker", "Typ", "Férová cena", "Dátum FV", "MoS %", "Cieľová cena", "Moja nákupná cena", "Poznámka"]
NUM_COLS = ["Férová cena", "MoS %", "Cieľová cena", "Moja nákupná cena"]
TYPES = ["Nákup", "Sledujem"]
REFRESH = {"Vypnuté": 0, "1 min": 60, "5 min": 300, "15 min": 900}
TICKER_RE = re.compile(r"^[A-Z0-9.\-^=]{1,12}$")


def sf(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# ----------------------------------------------------------------------------
# ZOZNAM: načítanie, čistenie, ukladanie
# ----------------------------------------------------------------------------
def normalize(df) -> pd.DataFrame:
    df = pd.DataFrame() if df is None else df.copy()
    for c in COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[COLS]
    df["Ticker"] = df["Ticker"].fillna("").astype(str).str.strip().str.upper()
    df["Typ"] = df["Typ"].where(df["Typ"].isin(TYPES), "Nákup")
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Dátum FV"] = pd.to_datetime(df["Dátum FV"], errors="coerce")
    df["Poznámka"] = df["Poznámka"].fillna("").astype(str)
    empty = (df["Ticker"] == "") & df[NUM_COLS].isna().all(axis=1)
    return df[~empty].reset_index(drop=True)


def load_local():
    try:
        if os.path.exists(WL_FILE):
            return pd.read_csv(WL_FILE)
    except Exception:
        pass
    return None


def save_local(df: pd.DataFrame) -> bool:
    try:
        df.to_csv(WL_FILE, index=False)
        return True
    except Exception:
        return False


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df.copy()
    out["Dátum FV"] = out["Dátum FV"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False).encode("utf-8-sig")


# ----------------------------------------------------------------------------
# CENY (Yahoo): živá cena krátko v cache, história na priemer dlhšie
# ----------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def get_quotes(symbols: tuple) -> dict:
    def one(sym):
        out = dict(price=np.nan, prev=np.nan, cur="")
        try:
            fi = yf.Ticker(sym).fast_info
            out["price"] = sf(fi["last_price"])
            out["prev"] = sf(fi["previous_close"])
            out["cur"] = str(fi["currency"] or "")
        except Exception:
            pass
        return sym, out

    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, symbols))


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def get_history_stats(symbols: tuple) -> dict:
    def one(sym):
        out = dict(avg5=np.nan, last=np.nan)
        try:
            c = yf.Ticker(sym).history(period="5y", auto_adjust=False)["Close"].dropna()
            if len(c):
                out.update(avg5=float(c.mean()), last=float(c.iloc[-1]))
        except Exception:
            pass
        return sym, out

    with ThreadPoolExecutor(max_workers=6) as ex:
        return dict(ex.map(one, symbols))


# ----------------------------------------------------------------------------
# VÝPOČET STAVU
# ----------------------------------------------------------------------------
def status_for(price, buy, near_pct):
    if not np.isfinite(price):
        return 4, "⚠️ bez ceny"
    if not np.isfinite(buy):
        return 3, "ℹ️ doplň cieľ alebo FV"
    if price <= buy:
        return 0, "🟢 NÁKUP"
    if price <= buy * (1 + near_pct / 100):
        return 1, "🟡 BLÍZKO"
    return 2, "⚪ čakám"


def build_table(df: pd.DataFrame, quotes: dict, hist: dict, p: dict) -> pd.DataFrame:
    rows = []
    today = pd.Timestamp.now().normalize()
    for _, r in df.iterrows():
        sym = r["Ticker"]
        q, h = quotes.get(sym, {}), hist.get(sym, {})
        scale = 0.01 if q.get("cur") == "GBp" else 1.0           # Londýn: pence → libry
        price, note_src = sf(q.get("price")) * scale, ""
        if not np.isfinite(price):
            price, note_src = sf(h.get("last")) * scale, "posledné zatvorenie"
        prev = sf(q.get("prev")) * scale
        avg5 = sf(h.get("avg5")) * scale
        fv = r["Férová cena"] if np.isfinite(r["Férová cena"]) and r["Férová cena"] > 0 else np.nan
        mos = r["MoS %"] if np.isfinite(r["MoS %"]) and 0 <= r["MoS %"] < 100 else p["mos"]
        target = r["Cieľová cena"] if np.isfinite(r["Cieľová cena"]) and r["Cieľová cena"] > 0 else np.nan
        buy_fv = fv * (1 - mos / 100) if np.isfinite(fv) else np.nan
        buy = target if np.isfinite(target) else buy_fv
        src = "vlastná cieľová" if np.isfinite(target) else (f"FV × (1 − {mos:.0f} %)" if np.isfinite(buy_fv) else "")
        prio, label = status_for(price, buy, p["near"])
        if note_src:   # živá cena chýba: upozornenie na nákup nespúšťame zo starej ceny
            prio, label = 4, "⚠️ bez živej ceny"

        warn = []
        if note_src:
            warn.append("cena = posledné zatvorenie")
        if np.isfinite(fv):
            if pd.isna(r["Dátum FV"]):
                warn.append("⏰ FV bez dátumu")
            elif (today - r["Dátum FV"]).days > p["stale"]:
                warn.append(f"⏰ FV stará {(today - r['Dátum FV']).days} dní – prepočítaj")
            if np.isfinite(price) and price > fv:
                warn.append("cena nad férovou")
        mine = r["Moja nákupná cena"]
        rows.append({
            "_prio": prio, "_dist": (price / buy - 1) * 100 if np.isfinite(price) and np.isfinite(buy) else np.inf,
            "Ticker": sym, "Typ": r["Typ"], "Cena": price, "Mena": ("GBP" if q.get("cur") == "GBp" else q.get("cur", "")),
            "Zmena %": (price / prev - 1) * 100 if np.isfinite(price) and np.isfinite(prev) and prev > 0 else np.nan,
            "Priem. cena 5r": avg5,
            "Cena vs priem. %": (price / avg5 - 1) * 100 if np.isfinite(price) and np.isfinite(avg5) and avg5 > 0 else np.nan,
            "Férová cena": fv,
            "MoS dnes %": (1 - price / fv) * 100 if np.isfinite(price) and np.isfinite(fv) else np.nan,
            "Nákupná cena": buy, "Zdroj hranice": src,
            "Do hranice %": (price / buy - 1) * 100 if np.isfinite(price) and np.isfinite(buy) else np.nan,
            "Stav": label, "Upozornenia": ", ".join(warn),
            "P/L %": (price / mine - 1) * 100 if np.isfinite(price) and np.isfinite(mine) and mine > 0 else np.nan,
            "Poznámka": r["Poznámka"],
        })
    t = pd.DataFrame(rows)
    return t.sort_values(["_prio", "_dist"], kind="stable").reset_index(drop=True) if not t.empty else t


def style_table(t: pd.DataFrame):
    show = t.drop(columns=["_prio", "_dist"])

    def color(row):
        if "NÁKUP" in row["Stav"]:
            css = "background-color:#c8efd3;color:#111111;font-weight:600"
        elif "BLÍZKO" in row["Stav"]:
            css = "background-color:#fde2b8;color:#111111"
        else:
            css = ""
        return [css] * len(row)

    fmt = {"Cena": "{:,.2f}", "Priem. cena 5r": "{:,.2f}", "Férová cena": "{:,.2f}", "Nákupná cena": "{:,.2f}",
           "Zmena %": "{:+.1f}", "Cena vs priem. %": "{:+.1f}", "MoS dnes %": "{:+.1f}", "Do hranice %": "{:+.1f}",
           "P/L %": "{:+.1f}"}
    return show.style.apply(color, axis=1).format(fmt, na_rep="–")


def dashboard(df: pd.DataFrame, p: dict):
    symbols = tuple(sorted({s for s in df["Ticker"] if TICKER_RE.match(s)}))
    quotes, hist = get_quotes(symbols), get_history_stats(symbols)
    t = build_table(df, quotes, hist, p)
    if t.empty:
        return

    n_buy, n_near, n_wait = int((t["_prio"] == 0).sum()), int((t["_prio"] == 1).sum()), int((t["_prio"] == 2).sum())
    n_miss = int((t["_prio"] >= 3).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🟢 Na nákupnej cene", n_buy)
    c2.metric("🟡 Blízko", n_near)
    c3.metric("⚪ Čakám", n_wait)
    c4.metric("⚠️ Chýbajú údaje", n_miss)

    hit = t[t["_prio"] == 0]
    if not hit.empty:
        st.success("🔔 **NA NÁKUPNEJ CENE:** " + " · ".join(
            f"**{r['Ticker']}** {r['Cena']:,.2f} ≤ {r['Nákupná cena']:,.2f}" for _, r in hit.iterrows()))
    close = t[t["_prio"] == 1]
    if not close.empty:
        st.warning("🟡 **Blízko nákupnej ceny:** " + " · ".join(
            f"**{r['Ticker']}** {r['Cena']:,.2f} (do hranice {r['Do hranice %']:+.1f} %)" for _, r in close.iterrows()))

    # jednorazové oznámenie (toast) pre akcie, ktoré sa práve dostali na nákupnú cenu
    seen = st.session_state.get("wl_toasted", set())
    now_green = set(hit["Ticker"])
    for sym in sorted(now_green - seen):
        try:
            st.toast(f"🔔 {sym} je na nákupnej cene", icon="🟢")
        except Exception:
            pass
    st.session_state["wl_toasted"] = now_green

    st.dataframe(style_table(t), use_container_width=True, hide_index=True)
    used_close = [r.Ticker for r in t.itertuples() if "posledné zatvorenie" in r.Upozornenia]
    st.caption(f"Ceny z Yahoo Finance, aktualizované {datetime.now().strftime('%H:%M:%S')} (Yahoo môže meškať až 15 minút; "
               f"živá cena sa cachuje 60 s). Priemerná cena = jednoduchý priemer zatváracích cien za 5 rokov."
               + (f" Pri {', '.join(used_close)} chýba živá cena, použité je posledné zatvorenie." if used_close else ""))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title("🔔 Watchlist – sledované a plánované nákupy")
st.caption("Zapíš si akcie, férovú cenu a svoj margin of safety. Stránka ukáže, ktoré sú na nákupnej cene. "
           "Férovú cenu si prenes z hlavnej aplikácie (vo rovnakej mene ako cena akcie).")
st.warning("⚠️ **Upozornenie funguje len, keď je táto stránka otvorená.** Dáta sa ukladajú v aplikácii, kde sa po reštarte "
           "môžu stratiť. Pravidelne si stiahni zálohu (CSV) a po reštarte ju nahraj.")

with st.sidebar:
    st.header("⚙️ Nastavenia")
    default_mos = st.number_input("Predvolený margin of safety (%)", min_value=0.0, max_value=90.0, value=15.0, step=1.0,
                                  key="wl_mos", help="Použije sa, ak pri akcii nezadáš vlastný MoS.")
    near = st.number_input("„Blízko“ = do (%) nad nákupnou cenou", min_value=0.0, max_value=50.0, value=5.0, step=1.0,
                           key="wl_near")
    stale = st.number_input("Férová cena je stará po (dňoch)", min_value=7, max_value=720, value=90, step=7, key="wl_stale")
    refresh_label = st.selectbox("Automatické obnovovanie cien", list(REFRESH.keys()), index=2, key="wl_refresh")
    if st.button("🔄 Obnoviť ceny teraz", key="wl_reload"):
        st.cache_data.clear()
        st.rerun()

p = dict(mos=default_mos, near=near, stale=int(stale), refresh=REFRESH[refresh_label])

# počiatočný stav: súbor v aplikácii (ak existuje)
if "wl_base" not in st.session_state:
    st.session_state["wl_base"] = normalize(load_local())
    st.session_state["wl_ver"] = 0

up = st.file_uploader("Nahrať zálohu (CSV)", type="csv", key="wl_upload")
if up is not None:
    sig = (up.name, up.size)
    if st.session_state.get("wl_up_sig") != sig:
        try:
            loaded = normalize(pd.read_csv(up))
            st.session_state["wl_base"] = loaded
            st.session_state["wl_ver"] += 1
            st.session_state["wl_up_sig"] = sig
            st.success(f"Načítaných {len(loaded)} riadkov zo zálohy.")
        except Exception as e:
            st.error(f"Zálohu sa nepodarilo načítať: {e}")

st.markdown("### 📝 Zoznam")
st.caption("Klikni do tabuľky a pridaj riadok (dole). Nákupná cena = tvoja cieľová cena, ak ju zadáš, inak "
           "férová cena × (1 − MoS). Dátum FV ti pomôže odhaliť zastaranú férovú cenu.")
cfg = {
    "Ticker": st.column_config.TextColumn("Ticker", help="napr. AAPL, MSFT, SAP.DE", required=True),
    "Typ": st.column_config.SelectboxColumn("Typ", options=TYPES, required=True),
    "Férová cena": st.column_config.NumberColumn("Férová cena", min_value=0.0, format="%.2f"),
    "Dátum FV": st.column_config.DateColumn("Dátum odhadu FV"),
    "MoS %": st.column_config.NumberColumn("Môj MoS %", min_value=0.0, max_value=95.0, format="%.0f",
                                           help="Prázdne = predvolený MoS z bočného panela."),
    "Cieľová cena": st.column_config.NumberColumn("Cieľová nákupná cena", min_value=0.0, format="%.2f",
                                                  help="Ak vyplníš, použije sa namiesto FV × (1 − MoS)."),
    "Moja nákupná cena": st.column_config.NumberColumn("Moja priemerná nákupná cena", min_value=0.0, format="%.2f",
                                                       help="Len ak už akciu držíš; ukáže zisk alebo stratu v %."),
    "Poznámka": st.column_config.TextColumn("Poznámka / dôvod"),
}
edited = st.data_editor(st.session_state["wl_base"], num_rows="dynamic", hide_index=True, use_container_width=True,
                        column_config=cfg, key=f"wl_editor_{st.session_state['wl_ver']}")
df = normalize(edited)

bad = [s for s in df["Ticker"] if s and not TICKER_RE.match(s)]
if bad:
    st.error("Neplatný ticker: " + ", ".join(bad) + " (povolené sú písmená, čísla, bodka, pomlčka).")
dups = df["Ticker"][df["Ticker"].duplicated() & (df["Ticker"] != "")].unique().tolist()
if dups:
    st.info("Ticker sa v zozname opakuje: " + ", ".join(dups))

# ukladanie (do súboru v aplikácii) a záloha
sig_now = to_csv_bytes(df)
if st.session_state.get("wl_saved_sig") != sig_now:
    saved = save_local(df)
    st.session_state["wl_saved_sig"] = sig_now
    st.session_state["wl_saved_at"] = datetime.now().strftime("%H:%M:%S") if saved else "nepodarilo sa uložiť"
b1, b2 = st.columns([1, 3])
b1.download_button("⬇️ Stiahnuť zálohu (CSV)", sig_now, file_name="watchlist.csv", mime="text/csv", key="wl_download")
b2.caption(f"Posledné uloženie v aplikácii: {st.session_state.get('wl_saved_at', '–')}. "
           "Záloha v počítači je jediné trvalé uloženie.")

st.divider()
if df.empty or (df["Ticker"] == "").all():
    st.info("Zoznam je prázdny. Pridaj riadok do tabuľky vyššie (ticker a férová cena alebo cieľová cena).")
    st.stop()

work = df[df["Ticker"].apply(lambda s: bool(TICKER_RE.match(s)))]
frag = getattr(st, "fragment", None) or getattr(st, "experimental_fragment", None)
if p["refresh"] and frag is not None:
    try:
        frag(run_every=p["refresh"])(dashboard)(work, p)
    except TypeError:
        dashboard(work, p)
else:
    dashboard(work, p)

st.caption("⚠️ Nie je to investičné odporúčanie. Férová cena je tvoj vlastný odhad: pri zastaranom odhade alebo zmene "
           "firmy ju prepočítaj v hlavnej aplikácii.")

