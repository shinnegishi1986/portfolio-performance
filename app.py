import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import sqlite3
import json

st.set_page_config(page_title="Multi-Asset Portfolio Simulator", layout="wide")

DB_PATH = "portfolios.db"

# ------------------ SQLite helper functions ------------------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            title TEXT,
            tickers TEXT,
            weights TEXT,
            initial_capital REAL,
            start_date TEXT,
            end_date TEXT,
            interval TEXT,
            created_at TEXT
        )
        """
    )
    return conn

def save_portfolio(
    name: str,
    title: str,
    tickers: list,
    weights: np.ndarray,
    initial_capital: float,
    start_date: date,
    end_date: date,
    interval: str,
):
    if not name:
        return False, "Portfolio name is empty."

    conn = get_db()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO portfolios
            (name, title, tickers, weights, initial_capital, start_date, end_date, interval, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                title,
                json.dumps(tickers),
                json.dumps(list(map(float, weights))),
                float(initial_capital),
                start_date.isoformat(),
                end_date.isoformat(),
                interval,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    except Exception as e:
        conn.close()
        return False, f"Save failed: {e}"
    conn.close()
    return True, "Portfolio saved."

def load_portfolio_names():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM portfolios ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]

def load_portfolio_by_name(name: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT title, tickers, weights, initial_capital, start_date, end_date, interval
        FROM portfolios WHERE name = ?
        """,
        (name,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    title = row[0] or ""
    tickers = json.loads(row[1])
    weights = np.array(json.loads(row[2]), dtype=float)
    initial_capital = float(row[3])
    start_date = datetime.fromisoformat(row[4]).date()
    end_date = datetime.fromisoformat(row[5]).date()
    interval = row[6]
    return {
        "title": title,
        "tickers": tickers,
        "weights": weights,
        "initial_capital": initial_capital,
        "start_date": start_date,
        "end_date": end_date,
        "interval": interval,
    }

def generate_untitled_name():
    existing = load_portfolio_names()
    base = "Untitled portfolio "
    n = 1
    while f"{base}{n}" in existing:
        n += 1
    return f"{base}{n}"

# ------------------ Session init ------------------
if "loaded_portfolio" not in st.session_state:
    st.session_state["loaded_portfolio"] = None
if "editor_df" not in st.session_state:
    st.session_state["editor_df"] = pd.DataFrame(columns=["Ticker", "Weight(%)"])
if "new_ticker" not in st.session_state:
    st.session_state["new_ticker"] = ""
if "auto_run_after_load" not in st.session_state:
    st.session_state["auto_run_after_load"] = False

st.title("Multi-Asset Portfolio Simulator")

# ------------------ Sidebar: global settings ------------------
st.sidebar.header("Portfolio Settings")

loaded = st.session_state["loaded_portfolio"]

default_capital = loaded["initial_capital"] if loaded else 10000.0
initial_capital = st.sidebar.number_input(
    "Initial Investment Amount",
    min_value=1.0,
    value=float(default_capital),
    step=1000.0,
)

weight_mode = st.sidebar.radio(
    "Weight Mode",
    [
        "Use specified weights (normalized)",
        "Force equal weight (ignore specified weights)",
    ],
    index=0,
)

today = date.today()
default_from = today - timedelta(days=365)

if loaded:
    default_start = loaded["start_date"]
    default_end = loaded["end_date"]
else:
    default_start = default_from
    default_end = today

start_dt_input = st.sidebar.date_input("From Date", value=default_start)
end_dt_input = st.sidebar.date_input("To Date", value=default_end)

start_dt = start_dt_input
end_dt = end_dt_input

default_interval = loaded["interval"] if loaded else "1d"
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=["1d", "1wk", "1mo"].index(default_interval),
)

st.sidebar.markdown("---")
st.sidebar.subheader("Saved Portfolios")

existing_names = load_portfolio_names()
selected_saved = st.sidebar.selectbox(
    "Load saved portfolio",
    options=["(None)"] + existing_names,
    index=0,
)

if st.sidebar.button("Load Selected Portfolio") and selected_saved != "(None)":
    data = load_portfolio_by_name(selected_saved)
    if data:
        st.session_state["loaded_portfolio"] = data
        st.session_state["editor_df"] = pd.DataFrame(
            {
                "Ticker": data["tickers"],
                "Weight(%)": data["weights"] * 100.0,
            }
        )
        st.session_state["auto_run_after_load"] = True
        st.success(f"Loaded: {selected_saved}")
        st.rerun()

if st.sidebar.button("Create New Portfolio"):
    st.session_state["loaded_portfolio"] = None
    st.session_state["editor_df"] = pd.DataFrame(columns=["Ticker", "Weight(%)"])
    st.session_state["new_ticker"] = ""
    st.session_state["auto_run_after_load"] = False
    st.rerun()

# ------------------ Helper to clean yfinance data ------------------
def clean_yf_df(raw_df: pd.DataFrame) -> pd.Series:
    if raw_df is None or raw_df.empty:
        return pd.Series(dtype=float)
    df_ = raw_df.copy()
    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.get_level_values(0)
    
    col_map = {}
    for c in df_.columns:
        c_lower = str(c).lower()
        if "adj close" in c_lower: col_map[c] = "Adj Close"
        elif "close" in c_lower: col_map[c] = "Close"
    
    if col_map:
        df_.rename(columns=col_map, inplace=True)
    
    target_col = "Adj Close" if "Adj Close" in df_.columns else "Close"
    if target_col not in df_.columns:
        return pd.Series(dtype=float)
    
    return df_[target_col].dropna()

def add_ticker_from_input():
    t = st.session_state["new_ticker"].strip().upper()
    if not t: return
    df = st.session_state["editor_df"].copy()
    if t not in df["Ticker"].astype(str).tolist():
        df.loc[len(df)] = [t, 0.0]
        st.session_state["editor_df"] = df
    st.session_state["new_ticker"] = ""

# ------------------ UI Components ------------------
st.header("1. Edit Portfolio")
c1, c2 = st.columns([5, 1])
with c1:
    st.text_input("Add Ticker (e.g. AAPL)", key="new_ticker", on_change=add_ticker_from_input)
with c2:
    st.write("##")
    if st.button("Add"): add_ticker_from_input()

edited_df = st.data_editor(st.session_state["editor_df"], num_rows="dynamic", key="portfolio_editor")

portfolio_name_title = st.text_input("Portfolio Name", value=loaded["title"] if loaded else "")

run_col, save_col = st.columns([1, 1])
run_clicked = run_col.button("Run Simulation", use_container_width=True)
save_clicked = save_col.button("Save Portfolio", use_container_width=True)

# ------------------ Logic Helpers ------------------
def parse_from_editor(df: pd.DataFrame):
    df = df.dropna(subset=["Ticker"])
    tickers = df["Ticker"].astype(str).str.strip().tolist()
    tickers = [t for t in tickers if t != ""]
    raw_weights = df["Weight(%)"].tolist() if "Weight(%)" in df.columns else [0] * len(tickers)
    return tickers, raw_weights

def determine_weights(tickers, raw_weights, mode):
    n = len(tickers)
    if n == 0: return np.array([])
    if mode == "Force equal weight (ignore specified weights)":
        return np.ones(n) / n
    w = np.array([float(x) if x else 0.0 for x in raw_weights])
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n

def run_simulation(tickers, raw_weights):
    weights = determine_weights(tickers, raw_weights, weight_mode)
    price_df = pd.DataFrame()
    
    for t in tickers:
        data = yf.download(t, start=start_dt, end=end_dt, interval=interval, progress=False)
        series = clean_yf_df(data)
        if not series.empty:
            price_df[t] = series

    if price_df.empty:
        st.error("No data found.")
        return

    price_df.ffill(inplace=True)
    price_df.dropna(inplace=True)
    
    # Calculate returns for the Chart (Performance by Percent)
    # Formula: (Price / Start_Price - 1) * 100
    percent_df = (price_df / price_df.iloc[0] - 1) * 100

    # Portfolio Value Calculation
    alloc = initial_capital * weights
    shares = alloc / price_df.iloc[0].values
    portfolio_value = (price_df * shares).sum(axis=1)

    st.header("2. Results")
    m1, m2, m3 = st.columns(3)
    final_val = portfolio_value.iloc[-1]
    ret = (final_val / initial_capital - 1) * 100
    m1.metric("Initial", f"${initial_capital:,.2f}")
    m2.metric("Final", f"${final_val:,.2f}")
    m3.metric("Return", f"{ret:.2f}%")

    st.header("3. Performance Comparison (%)")
    fig_pct = go.Figure()
    for col in percent_df.columns:
        fig_pct.add_trace(go.Scatter(x=percent_df.index, y=percent_df[col], name=col, mode='lines'))
    fig_pct.update_layout(title="Asset Growth Comparison (Cumulative %)", yaxis_title="Return %", hovermode="x unified")
    st.plotly_chart(fig_pct, use_container_width=True)

    st.header("4. Portfolio Total Value ($)")
    fig_val = go.Figure()
    fig_val.add_trace(go.Scatter(x=portfolio_value.index, y=portfolio_value, name="Portfolio", fill='tozeroy'))
    fig_val.update_layout(yaxis_title="Total Value ($)")
    st.plotly_chart(fig_val, use_container_width=True)

# ------------------ Actions ------------------
if save_clicked:
    tks, rws = parse_from_editor(edited_df)
    if tks:
        name = resolve_name_and_title(portfolio_name_title)
        w = determine_weights(tks, rws, weight_mode)
        ok, msg = save_portfolio(name, name, tks, w, initial_capital, start_dt, end_dt, interval)
        st.success(msg) if ok else st.error(msg)

if run_clicked:
    tks, rws = parse_from_editor(edited_df)
    if tks: run_simulation(tks, rws)
elif st.session_state.get("auto_run_after_load", False):
    tks, rws = parse_from_editor(edited_df)
    if tks: run_simulation(tks, rws)
    st.session_state["auto_run_after_load"] = False

def resolve_name_and_title(raw):
    return raw.strip() if raw.strip() else generate_untitled_name()