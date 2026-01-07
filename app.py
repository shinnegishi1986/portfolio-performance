import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, date, timedelta
import sqlite3
import json

# ------------------ Streamlit config ------------------
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


def resolve_name_and_title(raw: str):
    raw = (raw or "").strip()
    return raw if raw else generate_untitled_name()


# ------------------ Session init ------------------
if "loaded_portfolio" not in st.session_state:
    st.session_state["loaded_portfolio"] = None
if "editor_df" not in st.session_state:
    st.session_state["editor_df"] = pd.DataFrame(columns=["Ticker", "Weight(%)"])
if "new_ticker" not in st.session_state:
    st.session_state["new_ticker"] = ""
if "auto_run_after_load" not in st.session_state:
    st.session_state["auto_run_after_load"] = False
if "portfolio_name" not in st.session_state:
    st.session_state["portfolio_name"] = ""

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
        # set portfolio name into session_state so input reflects it
        st.session_state["portfolio_name"] = data["title"] or ""
        st.session_state["auto_run_after_load"] = True
        st.success(f"Loaded: {selected_saved}")
        st.rerun()

if st.sidebar.button("Create New Portfolio"):
    st.session_state["loaded_portfolio"] = None
    st.session_state["editor_df"] = pd.DataFrame(columns=["Ticker", "Weight(%)"])
    st.session_state["new_ticker"] = ""
    st.session_state["auto_run_after_load"] = False
    # clear Portfolio Name
    st.session_state["portfolio_name"] = ""
    st.rerun()

# ------------------ Helper to clean yfinance data ------------------
def clean_yf_df(raw_df: pd.DataFrame) -> pd.Series:
    if raw_df is None or raw_df.empty:
        return pd.Series(dtype=float)

    df_ = raw_df.copy()

    # Handle MultiIndex: use first level (e.g. 'Adj Close', 'Close')
    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = df_.columns.get_level_values(0)

    # Normalize column names
    col_map = {}
    for c in df_.columns:
        c_lower = str(c).lower()
        if "adj close" in c_lower:
            col_map[c] = "Adj Close"
        elif "close" in c_lower:
            col_map[c] = "Close"

    if col_map:
        df_.rename(columns=col_map, inplace=True)

    target_col = "Adj Close" if "Adj Close" in df_.columns else "Close"
    if target_col not in df_.columns:
        return pd.Series(dtype=float)

    return df_[target_col].dropna()


def add_ticker_from_input():
    t = st.session_state["new_ticker"].strip().upper()
    if not t:
        return
    df = st.session_state["editor_df"].copy()
    if "Ticker" not in df.columns:
        df = pd.DataFrame(columns=["Ticker", "Weight(%)"])
    if t not in df["Ticker"].astype(str).tolist():
        df.loc[len(df)] = [t, 0.0]
        st.session_state["editor_df"] = df
    st.session_state["new_ticker"] = ""

# ------------------ UI Components ------------------
st.header("1. Edit Portfolio")

# Add ticker only via Enter
st.text_input("Add Ticker (e.g. AAPL)", key="new_ticker", on_change=add_ticker_from_input)

edited_df = st.data_editor(
    st.session_state["editor_df"],
    num_rows="dynamic",
    key="portfolio_editor"
)

# bind Portfolio Name to session_state["portfolio_name"] so it can be cleared
portfolio_name_title = st.text_input(
    "Portfolio Name",
    key="portfolio_name"
)

run_col, save_col = st.columns([1, 1])
run_clicked = run_col.button("Run Simulation", use_container_width=True)
save_clicked = save_col.button("Save Portfolio", use_container_width=True)

# ------------------ Logic Helpers ------------------
def parse_from_editor(df: pd.DataFrame):
    if "Ticker" not in df.columns:
        return [], []
    df = df.dropna(subset=["Ticker"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""]
    tickers = df["Ticker"].tolist()
    raw_weights = df["Weight(%)"].tolist() if "Weight(%)" in df.columns else [0] * len(tickers)
    return tickers, raw_weights


def determine_weights(tickers, raw_weights, mode):
    n = len(tickers)
    if n == 0:
        return np.array([])
    if mode == "Force equal weight (ignore specified weights)":
        return np.ones(n) / n
    w = np.array([float(x) if x not in [None, ""] else 0.0 for x in raw_weights], dtype=float)
    total = w.sum()
    if total <= 0:
        return np.ones(n) / n
    return w / total


def run_simulation(tickers, raw_weights):
    if not tickers:
        st.error("Please add at least one ticker.")
        return

    weights = determine_weights(tickers, raw_weights, weight_mode)

    price_df = pd.DataFrame()
    failed = []

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start_dt,
                end=end_dt,
                interval=interval,
                progress=False,
            )
            series = clean_yf_df(data)
            if not series.empty:
                price_df[t] = series
            else:
                failed.append(t)
        except Exception:
            failed.append(t)

    if price_df.empty:
        st.error("No data found for the selected tickers and date range.")
        return

    # Align and clean
    price_df.sort_index(inplace=True)
    price_df.ffill(inplace=True)
    price_df.dropna(how="any", inplace=True)

    # Re-align tickers & weights to columns that actually exist
    valid_cols = list(price_df.columns)
    mask = [t in valid_cols for t in tickers]
    tickers = [t for t, m in zip(tickers, mask) if m]
    weights = np.array([w for w, m in zip(weights, mask) if m], dtype=float)

    if len(tickers) == 0:
        st.error("All tickers failed to retrieve valid data.")
        return

    if weights.sum() <= 0:
        weights = np.ones(len(tickers)) / len(tickers)
    else:
        weights = weights / weights.sum()

    price_df = price_df[tickers]

    # ---------- Percent performance (cumulative %) ----------
    percent_df = (price_df / price_df.iloc[0] - 1.0) * 100.0

    # ---------- Portfolio Value Calculation ----------
    alloc = initial_capital * weights
    first_prices = price_df.iloc[0].values
    shares = alloc / first_prices
    portfolio_value = (price_df.values * shares).sum(axis=1)
    portfolio_series = pd.Series(portfolio_value, index=price_df.index, name="PortfolioValue")

    final_val = float(portfolio_series.iloc[-1])
    total_return_pct = (final_val / initial_capital - 1.0) * 100.0

    # ---------- Portfolio cumulative % series ----------
    portfolio_pct = (portfolio_series / portfolio_series.iloc[0] - 1.0) * 100.0

    # ---------- Results metrics ----------
    st.header("2. Portfolio Results")

    m1, m2, m3 = st.columns(3)
    m1.metric("Initial Value", f"${initial_capital:,.2f}")
    m2.metric("Final Value", f"${final_val:,.2f}")
    m3.metric("Total Return", f"{total_return_pct:.2f}%")

    # Allocation table
    alloc_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weights,
            "Allocated Capital": alloc,
            "Shares": shares,
        }
    )
    st.subheader("Effective Allocation")
    st.dataframe(
        alloc_df.style.format(
            {
                "Weight": "{:.2%}",
                "Allocated Capital": "{:,.2f}",
                "Shares": "{:.4f}",
            }
        )
    )

    if failed:
        st.warning(f"No data for: {', '.join(set(failed))} (ignored).")

    # ---------- Charts ----------
    st.header("3. Charts")

    # 3-1 Individual asset price performance
    st.subheader("Asset Price Performance")
    fig_prices = go.Figure()
    for col in price_df.columns:
        fig_prices.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[col],
                mode="lines",
                name=col,
            )
        )
    fig_prices.update_layout(
        title="Individual Asset Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend_title="Ticker",
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # 3-2 Asset cumulative % performance chart
    st.subheader("Asset Cumulative Performance (%)")
    fig_pct = go.Figure()
    for col in percent_df.columns:
        fig_pct.add_trace(
            go.Scatter(
                x=percent_df.index,
                y=percent_df[col],
                mode="lines",
                name=col,
            )
        )
    fig_pct.update_layout(
        title="Cumulative Percent Performance by Asset",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
        legend_title="Ticker",
    )
    st.plotly_chart(fig_pct, use_container_width=True)

    # 3-3 Portfolio cumulative % area chart
    st.subheader("Portfolio Cumulative Performance (%)")
    fig_port_pct = go.Figure()
    fig_port_pct.add_trace(
        go.Scatter(
            x=portfolio_pct.index,
            y=portfolio_pct.values,
            mode="lines",
            name="Portfolio",
            fill="tozeroy",
        )
    )
    fig_port_pct.update_layout(
        title="Portfolio Cumulative Return (%)",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_port_pct, use_container_width=True)

    # 3-4 Portfolio total value area chart
    st.subheader("Portfolio Total Value ($)")
    fig_val = go.Figure()
    fig_val.add_trace(
        go.Scatter(
            x=portfolio_series.index,
            y=portfolio_series.values,
            mode="lines",
            name="Portfolio Value",
            fill="tozeroy",
        )
    )
    fig_val.update_layout(
        title="Portfolio Total Value Over Time",
        xaxis_title="Date",
        yaxis_title="Total Value ($)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_val, use_container_width=True)


# ------------------ Actions ------------------
if save_clicked:
    tks, rws = parse_from_editor(edited_df)
    if tks:
        name = resolve_name_and_title(st.session_state["portfolio_name"])
        w = determine_weights(tks, rws, weight_mode)
        ok, msg = save_portfolio(name, name, tks, w, initial_capital, start_dt, end_dt, interval)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
    else:
        st.error("Cannot save: please add at least one ticker.")

if run_clicked:
    tks, rws = parse_from_editor(edited_df)
    if tks:
        run_simulation(tks, rws)
    else:
        st.error("Please add at least one ticker.")
elif st.session_state.get("auto_run_after_load", False):
    tks, rws = parse_from_editor(edited_df)
    if tks:
        run_simulation(tks, rws)
    st.session_state["auto_run_after_load"] = False
