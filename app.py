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

st.title("Multi-Asset Portfolio Simulator (Saved List Always Updated)")

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

start_dt_input = st.sidebar.date_input(
    "From Date (optional)",
    value=default_start if loaded else None,
)
end_dt_input = st.sidebar.date_input(
    "To Date (optional)",
    value=default_end if loaded else None,
)

start_dt = start_dt_input[0] if isinstance(start_dt_input, (list, tuple)) and start_dt_input else start_dt_input
end_dt = end_dt_input[0] if isinstance(end_dt_input, (list, tuple)) and end_dt_input else end_dt_input

if start_dt is None:
    start_dt = default_from
if end_dt is None:
    end_dt = today

default_interval = loaded["interval"] if loaded else "1d"
interval = st.sidebar.selectbox(
    "Interval",
    ["1d", "1wk", "1mo"],
    index=["1d", "1wk", "1mo"].index(default_interval),
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: Yahoo Finance via yfinance")

# ------------- Sidebar: Saved portfolio loader + New portfolio -------------
st.sidebar.subheader("Saved Portfolios")

# This always re-queries the DB so newly saved portfolios are listed
existing_names = load_portfolio_names()
selected_saved = st.sidebar.selectbox(
    "Load saved portfolio",
    options=["(None)"] + existing_names,
    index=0,
)

load_clicked = st.sidebar.button("Load Selected Portfolio")
new_clicked = st.sidebar.button("Create New Portfolio")

if load_clicked and selected_saved != "(None)":
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
        st.success(f"Loaded portfolio: {selected_saved}")
    else:
        st.error("Failed to load portfolio data.")

if new_clicked:
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
        cols = list(df_.columns)
        if any(c[0] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] for c in cols):
            tmp = {}
            for price, ticker in cols:
                if price in ["Open", "High", "Low", "Close", "Adj Close"]:
                    tmp[price] = df_[(price, ticker)]
            df_ = pd.DataFrame(tmp, index=df_.index)
        elif any(c[1] in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] for c in cols):
            tmp = {}
            for ticker, price in cols:
                if price in ["Open", "High", "Low", "Close", "Adj Close"]:
                    tmp[price] = df_[(ticker, price)]
            df_ = pd.DataFrame(tmp, index=df_.index)
        else:
            df_.columns = [" ".join([str(x) for x in c]).strip() for c in cols]

    if isinstance(df_.columns, pd.MultiIndex):
        df_.columns = [" ".join([str(x) for x in c]).strip() for c in df_.columns]

    col_map = {}
    for c in df_.columns:
        c_lower = c.lower()
        if " adj close" in c_lower or c_lower == "adj close":
            col_map[c] = "Adj Close"
        elif " close" in c_lower and "adj" not in c_lower:
            col_map[c] = "Close"
    if col_map:
        df_.rename(columns=col_map, inplace=True)

    if "Close" not in df_.columns:
        if "Adj Close" in df_.columns:
            df_["Close"] = df_["Adj Close"]
        else:
            return pd.Series(dtype=float)

    if not isinstance(df_.index, pd.DatetimeIndex):
        try:
            df_.index = pd.to_datetime(df_.index)
        except Exception:
            pass

    return df_["Close"].dropna()

# ------------------ Add ticker via Enter or button ------------------
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
    else:
        st.warning(f"{t} is already in the portfolio.")
    st.session_state["new_ticker"] = ""

# ------------------ Portfolio editor UI ------------------
st.header("1. Edit Portfolio (Tickers & Weights)")

c1, c2, c3 = st.columns([6, 1.2, 0.8])
with c1:
    st.text_input(
        "Add Ticker (e.g. AAPL) and press Enter",
        key="new_ticker",
        on_change=add_ticker_from_input,
    )
with c2:
    st.write("")
    if st.button("Add", key="add_ticker_btn"):
        add_ticker_from_input()
with c3:
    st.write("")

editor_df = st.session_state["editor_df"]

st.write(
    "Edit Weight(%) cells. You can force equal weight in the sidebar even if some weights are specified."
)

edited_df = st.data_editor(
    editor_df,
    num_rows="dynamic",
    key="portfolio_editor",
)

# ------------------ Portfolio meta: name & title are same ------------------
st.subheader("Portfolio Name / Title")

if loaded:
    default_name_title = loaded["title"] or ""
else:
    default_name_title = ""

portfolio_name_title = st.text_input(
    "Portfolio Name (also used as Title; if blank will become 'Untitled portfolio N')",
    value=default_name_title,
)

run_col, save_col = st.columns([2, 1])
run_clicked = run_col.button("Run Portfolio Simulation")
save_clicked = save_col.button("Save / Update Portfolio in SQLite")

# ------------------ helpers ------------------
def parse_from_editor(df: pd.DataFrame):
    df = df.copy()
    if "Ticker" not in df.columns:
        return [], []
    df = df.dropna(subset=["Ticker"])
    df["Ticker"] = df["Ticker"].astype(str).str.strip()
    df = df[df["Ticker"] != ""]
    tickers = df["Ticker"].tolist()
    raw_weights = df["Weight(%)"].tolist() if "Weight(%)" in df.columns else [None] * len(tickers)
    return tickers, raw_weights

def determine_weights(tickers, raw_weights, mode: str):
    n = len(tickers)
    if n == 0:
        return np.array([])

    if mode == "Force equal weight (ignore specified weights)":
        return np.ones(n, dtype=float) / n

    weights = np.zeros(n, dtype=float)
    if any(w is not None for w in raw_weights):
        tmp = np.array([0.0 if w is None else float(w) for w in raw_weights], dtype=float)
        total = tmp.sum()
        if total <= 0:
            weights[:] = 1.0 / n
        else:
            weights = tmp / total
    else:
        weights[:] = 1.0 / n
    return weights

def resolve_name_and_title(raw: str) -> str:
    name = (raw or "").strip()
    if not name:
        name = generate_untitled_name()
    return name

def maybe_save_portfolio(tickers, raw_weights, action_label: str):
    if len(tickers) == 0:
        return
    weights = determine_weights(tickers, raw_weights, weight_mode)
    name_title = resolve_name_and_title(portfolio_name_title)
    ok, msg = save_portfolio(
        name_title,
        name_title,
        tickers,
        weights,
        initial_capital,
        start_dt,
        end_dt,
        interval,
    )
    if ok:
        st.session_state["editor_df"] = edited_df
        st.success(f"{msg} (saved as: {name_title}) via {action_label}")
    else:
        st.error(msg)

def run_simulation(tickers, raw_weights):
    if len(tickers) == 0:
        st.error("Please input at least one valid ticker.")
        return

    weights = determine_weights(tickers, raw_weights, weight_mode)

    price_df = pd.DataFrame()
    failed_tickers = []

    for t in tickers:
        try:
            data = yf.download(
                t,
                start=start_dt,
                end=end_dt,
                interval=interval,
                auto_adjust=False,
                progress=False,
            )
            close_series = clean_yf_df(data)
            if close_series.empty:
                failed_tickers.append(t)
                continue
            price_df[t] = close_series
        except Exception:
            failed_tickers.append(t)

    if price_df.empty:
        st.error("No valid price data could be downloaded for the given tickers and date range.")
        return

    price_df.dropna(how="all", inplace=True)

    valid_mask = [t in price_df.columns for t in tickers]
    tickers = [t for t, ok in zip(tickers, valid_mask) if ok]
    weights = np.array([w for w, ok in zip(weights, valid_mask) if ok], dtype=float)

    if len(tickers) == 0:
        st.error("All tickers failed to download data.")
        return

    if weights.sum() <= 0:
        weights[:] = 1.0 / len(tickers)
    else:
        weights = weights / weights.sum()

    price_df = price_df[tickers]

    first_prices = price_df.iloc[0]
    alloc_capital = initial_capital * weights
    shares = alloc_capital / first_prices.values

    portfolio_values = (price_df.values * shares).sum(axis=1)
    portfolio_series = pd.Series(portfolio_values, index=price_df.index, name="PortfolioValue")

    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1.0
    pnl = portfolio_series.iloc[-1] - initial_capital

    st.header("2. Portfolio Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Investment", f"{initial_capital:,.2f}")
    col2.metric("Final Portfolio Value", f"{portfolio_series.iloc[-1]:,.2f}")
    col3.metric("Total Return", f"{total_return * 100:.2f}%")

    st.metric("Profit / Loss", f"{pnl:,.2f}")

    alloc_df = pd.DataFrame(
        {
            "Ticker": tickers,
            "Weight": weights,
            "Allocated Capital": alloc_capital,
            "Shares": shares,
        }
    )
    st.subheader("Portfolio Allocation (Effective)")
    st.dataframe(
        alloc_df.style.format(
            {
                "Weight": "{:.2%}",
                "Allocated Capital": "{:,.2f}",
                "Shares": "{:.4f}",
            }
        )
    )

    if failed_tickers:
        st.warning(f"No data for: {', '.join(failed_tickers)} (ignored).")

    st.header("3. Charts")

    fig_prices = go.Figure()
    for t in price_df.columns:
        fig_prices.add_trace(
            go.Scatter(
                x=price_df.index,
                y=price_df[t],
                mode="lines",
                name=t,
            )
        )
    fig_prices.update_layout(
        title="Individual Asset Prices",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Ticker",
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    fig_port = go.Figure()
    fig_port.add_trace(
        go.Scatter(
            x=portfolio_series.index,
            y=portfolio_series.values,
            mode="lines",
            name="Portfolio Value",
        )
    )
    fig_port.update_layout(
        title="Portfolio Total Value Over Time",
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
    )
    st.plotly_chart(fig_port, use_container_width=True)

# ------------------ Save button ------------------
if save_clicked:
    tks, rws = parse_from_editor(edited_df)
    if len(tks) == 0:
        st.error("Cannot save: please input at least one valid ticker.")
    else:
        maybe_save_portfolio(tks, rws, "Save button")

# ------------------ Run button (also auto-save) ------------------
if run_clicked:
    tickers, raw_weights = parse_from_editor(edited_df)
    if len(tickers) == 0:
        st.error("Please input at least one valid ticker.")
    else:
        maybe_save_portfolio(tickers, raw_weights, "Run Portfolio Simulation")
        run_simulation(tickers, raw_weights)
elif st.session_state.get("auto_run_after_load", False):
    tickers, raw_weights = parse_from_editor(edited_df)
    if len(tickers) > 0:
        run_simulation(tickers, raw_weights)
    st.session_state["auto_run_after_load"] = False
