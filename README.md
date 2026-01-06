# Multi-Asset Portfolio Simulator (Streamlit)

This app is a **multi-asset portfolio simulator** built with Streamlit.  
You can define and edit portfolios of multiple tickers, assign weights, save them into SQLite, and visualize the combined portfolio performance over a chosen date range.

## Features

- **Multiple tickers portfolio**
  - Input any number of tickers (Yahoo Finance symbols).
  - Weights can be specified or auto-equalized if omitted or invalid.
- **Date range control**
  - From / To date selectable.
  - If From is empty → defaults to 1 year before today.  
  - If To is empty → defaults to today.[1][2]
- **Portfolio simulation**
  - Initial capital setting.
  - Capital allocation by weights (or equal split).
  - Portfolio value time series.
  - Total return (%) and profit / loss in currency.
- **Interactive charts**
  - Individual asset price lines.
  - Combined portfolio value over time (Plotly in Streamlit).[3][4]
- **SQLite portfolio storage**
  - Save portfolio definition (name, title, tickers, weights, dates, interval, capital) to SQLite.[5][6]
  - Load saved portfolios from sidebar.
  - Loaded portfolio can be edited and re-saved (swappable).
- **Editable portfolio table**
  - Add/remove rows (tickers) dynamically via `st.data_editor`.
  - Directly edit ticker names and weight (%) in the table.[7][8]

***

## Tech Stack

- **Python 3.9+** (recommended)
- **Streamlit** for the web application UI.[9]
- **yfinance** for downloading historical price data from Yahoo Finance.[10]
- **pandas / numpy** for data manipulation and portfolio math.[11]
- **plotly** for interactive charts embedded in Streamlit.[4]
- **SQLite (sqlite3)** in the Python standard library for storing portfolios.[6][5]

***

## Installation

1. Clone or download the repository:

```bash
git clone https://github.com/your-username/portfolio-simulator.git
cd portfolio-simulator
```

2. (Optional but recommended) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# or
.venv\Scripts\activate       # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should contain:

```text
streamlit>=1.34.0
pandas>=2.0.0
numpy>=1.24.0
yfinance>=0.2.50
plotly>=5.20.0
```

These versions are compatible with building a Streamlit dashboard that uses yfinance and Plotly.[12][13][10]

***

## Running the App

From the project root, run:

```bash
streamlit run app.py
```

- Replace `app.py` with the actual filename where you saved the provided code.  
- When the command runs, a browser window will open (or give you a local URL) to access the app.[9]

***

## How to Use

### 1. Set global options (sidebar)

- **Initial Investment Amount**: Total capital to invest.  
- **Capital Allocation Mode**:
  - *Use Weights*: Use weights from the editor table (normalized to 1 or 100).
  - *Equal Allocation if No Weights*: If weights are missing/invalid, auto-assign equal weights.  
- **From / To Date (optional)**:
  - Leave blank to use default (last 1 year to today).  
- **Interval**: `1d`, `1wk`, or `1mo` price frequency.

### 2. Edit tickers & weights

- In the main page, use the editable table:
  - Add new rows to add tickers.
  - Delete rows to remove tickers.
  - Edit **Ticker** and **Weight(%)** cells directly.
- Weights are interpreted as **percentages** when the checkbox is on (default).

### 3. Portfolio meta & saving

- **Portfolio Name**: Database key (must be unique to avoid overwriting an existing one).  
- **Portfolio Title**: Free-form display title for the portfolio.  
- Click **“Save / Update Portfolio in SQLite”**:
  - Creates or overwrites a row identified by the given name.  
  - Stores: name, title, tickers, weights, capital, dates, interval.[14][5]

### 4. Loading & editing saved portfolios

- In the sidebar:
  - Choose a portfolio from **“Load saved portfolio”**.
  - Click **“Load Selected Portfolio”**.
- The editor table and meta fields will be filled with the saved settings.  
- You can then:
  - Add/remove tickers.
  - Change weights.
  - Adjust dates, capital, etc.
  - Re-save with the same or a new name.

### 5. Running a simulation

- Click **“Run Portfolio Simulation”**:
  - The app downloads prices for each ticker from Yahoo Finance for the chosen date range.[15][10]
  - Allocates initial capital by weights.
  - Computes and plots the portfolio value over time.  

You will see:

- **Metrics**:
  - Initial Investment  
  - Final Portfolio Value  
  - Total Return (%)  
  - Profit / Loss  
- **Allocation table** (effective after filtering failed tickers):
  - Ticker  
  - Weight  
  - Allocated Capital  
  - Shares held  
- **Charts**:
  - Individual asset price lines.
  - Portfolio total value line.

***

## Notes & Limitations

- Yahoo Finance symbols must be valid and may fail for delisted or illiquid assets.[16][10]
- Missing data for a ticker in the chosen period can cause that ticker to be dropped from the portfolio in the simulation.  
- This app does **not** handle intraday intervals (`1m`, `5m`, etc.) for very long date ranges due to yfinance limitations.[17][15]
- SQLite database file (`portfolios.db`) is created in the same directory as the app if it does not exist.
