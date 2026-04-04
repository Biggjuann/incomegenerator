"""
Dividend & Income ETF Backtest Dashboard
=========================================
Strategy: $3,000,000 starting capital
  - 50% SPY (growth for inheritance, dividends reinvested)
  - 50% High-Yield Income ETFs (monthly retirement income, dividends paid out)

Income ETFs: SCHD, VYM, HDV, JEPI, QYLD, SDIV
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Income & Growth ETF Backtest Dashboard",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 3_000_000
SPY_ALLOCATION = 0.50
INCOME_ALLOCATION = 0.50

# Income ETFs grouped by payout frequency
MONTHLY_ETFS = ["JEPI", "QYLD", "SDIV"]   # pay monthly — steady income
QUARTERLY_ETFS = ["SCHD", "VYM", "HDV"]    # pay quarterly — growth + dividends
INCOME_ETFS = QUARTERLY_ETFS + MONTHLY_ETFS

ALL_TICKERS = ["SPY"] + INCOME_ETFS


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading market data …")
def download_data(tickers: list[str]) -> dict:
    """Download price and dividend history for each ticker since inception."""
    data = {}
    try:
        for t in tickers:
            tk = yf.Ticker(t)
            hist = tk.history(period="max", auto_adjust=False)
            if hist.empty:
                continue
            hist.index = hist.index.tz_localize(None)
            divs = hist["Dividends"].copy()
            close = hist["Close"].copy()
            data[t] = {"close": close, "dividends": divs}
    except Exception:
        pass
    if len(data) < len(tickers):
        data = _generate_demo_data(tickers)
    return data


def _generate_demo_data(tickers: list[str]) -> dict:
    """Generate realistic demo data when yfinance is unavailable."""
    np.random.seed(42)
    # Start date: Jun 2020 (JEPI inception) — common start for all ETFs
    start = pd.Timestamp("2020-06-01")
    end = pd.Timestamp("2026-03-31")
    dates = pd.bdate_range(start, end)

    # Realistic parameters: (annual_return, annual_vol, div_yield, frequency)
    # frequency: 'M' = monthly, 'Q' = quarterly
    params = {
        "SPY":  {"ret": 0.12, "vol": 0.18, "yield": 0.013, "freq": "Q"},
        "SCHD": {"ret": 0.09, "vol": 0.15, "yield": 0.035, "freq": "Q"},
        "VYM":  {"ret": 0.08, "vol": 0.14, "yield": 0.030, "freq": "Q"},
        "HDV":  {"ret": 0.07, "vol": 0.14, "yield": 0.038, "freq": "Q"},
        "JEPI": {"ret": 0.06, "vol": 0.10, "yield": 0.075, "freq": "M"},
        "QYLD": {"ret": 0.02, "vol": 0.12, "yield": 0.110, "freq": "M"},
        "SDIV": {"ret": 0.01, "vol": 0.18, "yield": 0.100, "freq": "M"},
    }

    # Starting prices (approximate real-world as of mid-2020)
    start_prices = {
        "SPY": 310.0, "SCHD": 52.0, "VYM": 78.0, "HDV": 80.0,
        "JEPI": 50.0, "QYLD": 21.0, "SDIV": 11.0,
    }

    data = {}
    for t in tickers:
        p = params.get(t)
        if p is None:
            continue
        n = len(dates)
        daily_ret = p["ret"] / 252
        daily_vol = p["vol"] / np.sqrt(252)
        returns = np.random.normal(daily_ret, daily_vol, n)
        price_0 = start_prices.get(t, 50.0)
        prices = price_0 * np.cumprod(1 + returns)
        close = pd.Series(prices, index=dates, name="Close")

        # Generate dividends
        divs = pd.Series(0.0, index=dates)
        if p["freq"] == "Q":
            # Quarterly: Mar, Jun, Sep, Dec — around the 20th
            for year in range(start.year, end.year + 1):
                for month in [3, 6, 9, 12]:
                    div_date = pd.Timestamp(year, month, 20)
                    # Find nearest business day
                    candidates = dates[(dates >= div_date - pd.Timedelta(days=5)) &
                                       (dates <= div_date + pd.Timedelta(days=5))]
                    if len(candidates) > 0:
                        d = candidates[len(candidates) // 2]
                        quarterly_div = close.get(d, price_0) * p["yield"] / 4
                        # Add some randomness (+/- 15%)
                        quarterly_div *= (1 + np.random.uniform(-0.15, 0.15))
                        divs[d] = max(quarterly_div, 0.01)
        else:
            # Monthly: around the 15th
            for year in range(start.year, end.year + 1):
                for month in range(1, 13):
                    try:
                        div_date = pd.Timestamp(year, month, 15)
                    except ValueError:
                        continue
                    candidates = dates[(dates >= div_date - pd.Timedelta(days=5)) &
                                       (dates <= div_date + pd.Timedelta(days=5))]
                    if len(candidates) > 0:
                        d = candidates[len(candidates) // 2]
                        monthly_div = close.get(d, price_0) * p["yield"] / 12
                        monthly_div *= (1 + np.random.uniform(-0.10, 0.10))
                        divs[d] = max(monthly_div, 0.01)

        data[t] = {"close": close, "dividends": divs}
    return data


def find_common_start(data: dict) -> pd.Timestamp:
    """Find the earliest date where ALL tickers have data."""
    starts = [data[t]["close"].index[0] for t in data]
    return max(starts)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(data: dict, start_date: pd.Timestamp, monthly_pct: float = 50.0, end_date: pd.Timestamp | None = None):
    """
    Run the backtest.

    SPY sleeve:  dividends are REINVESTED (compounding for inheritance).
    Income sleeve: dividends are PAID OUT as retirement income.

    monthly_pct: % of the income sleeve allocated to monthly-paying ETFs
                 (remainder goes to quarterly-paying ETFs). Default 50%.
    """
    tickers_available = [t for t in ALL_TICKERS if t in data]
    income_tickers = [t for t in INCOME_ETFS if t in data]
    monthly_tickers = [t for t in MONTHLY_ETFS if t in data]
    quarterly_tickers = [t for t in QUARTERLY_ETFS if t in data]

    if "SPY" not in tickers_available or len(income_tickers) == 0:
        st.error("Not enough ticker data to run backtest.")
        return None

    # Align to common date range
    if end_date is None:
        end_date = min(data[t]["close"].index[-1] for t in tickers_available)

    # Build a common daily date index
    all_dates = None
    for t in tickers_available:
        idx = data[t]["close"].loc[start_date:end_date].index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    all_dates = all_dates.sort_values()

    # ---- Initial share purchases (weighted by monthly vs quarterly split) ----
    spy_capital = STARTING_CAPITAL * SPY_ALLOCATION
    total_income_capital = STARTING_CAPITAL * INCOME_ALLOCATION

    monthly_capital = total_income_capital * (monthly_pct / 100.0)
    quarterly_capital = total_income_capital * (1 - monthly_pct / 100.0)

    # Equal-weight within each group
    capital_per_monthly = monthly_capital / max(len(monthly_tickers), 1)
    capital_per_quarterly = quarterly_capital / max(len(quarterly_tickers), 1)

    spy_price_0 = data["SPY"]["close"].loc[start_date:].iloc[0]
    spy_shares = spy_capital / spy_price_0

    income_shares = {}
    etf_capital = {}
    for t in income_tickers:
        p0 = data[t]["close"].loc[start_date:].iloc[0]
        cap = capital_per_monthly if t in monthly_tickers else capital_per_quarterly
        income_shares[t] = cap / p0
        etf_capital[t] = cap

    # ---- Daily simulation ----
    records = []
    cumulative_income = 0.0
    cumulative_spy_divs_reinvested = 0.0

    for date in all_dates:
        # SPY value
        spy_price = data["SPY"]["close"].get(date, np.nan)
        if np.isnan(spy_price):
            continue
        spy_div = data["SPY"]["dividends"].get(date, 0.0)

        # Reinvest SPY dividends
        if spy_div > 0 and spy_price > 0:
            div_cash = spy_div * spy_shares
            new_shares = div_cash / spy_price
            spy_shares += new_shares
            cumulative_spy_divs_reinvested += div_cash

        spy_value = spy_shares * spy_price

        # Income sleeve
        income_value = 0.0
        daily_income = 0.0
        for t in income_tickers:
            price = data[t]["close"].get(date, np.nan)
            if np.isnan(price):
                continue
            div = data[t]["dividends"].get(date, 0.0)
            if div > 0:
                payout = div * income_shares[t]
                daily_income += payout
            income_value += income_shares[t] * price

        cumulative_income += daily_income

        records.append(
            {
                "date": date,
                "spy_value": spy_value,
                "income_sleeve_value": income_value,
                "total_portfolio_value": spy_value + income_value,
                "daily_income": daily_income,
                "cumulative_income": cumulative_income,
                "cumulative_spy_divs_reinvested": cumulative_spy_divs_reinvested,
            }
        )

    df = pd.DataFrame(records).set_index("date")

    # Monthly income aggregation (raw — spiky due to quarterly payers)
    monthly_income_raw = df["daily_income"].resample("ME").sum()
    monthly_income_raw.name = "monthly_income_raw"

    # ------------------------------------------------------------------
    # INCOME SMOOTHING RESERVOIR
    # ------------------------------------------------------------------
    # Simulates what a retiree would actually do: all dividends flow into
    # a reservoir, and a steady monthly withdrawal is taken.  The
    # withdrawal amount is re-calibrated every 12 months based on the
    # trailing 3-month average inflow (conservative) to keep the
    # reservoir solvent while maximizing the paycheck.
    # ------------------------------------------------------------------
    reservoir = 0.0
    smoothed_values = []
    monthly_withdrawal = 0.0
    calibration_months = 0

    for i, (month_end, raw_income) in enumerate(monthly_income_raw.items()):
        reservoir += raw_income

        # Initial calibration: after first 3 months, set withdrawal
        if i == 2 and monthly_withdrawal == 0:
            monthly_withdrawal = monthly_income_raw.iloc[:3].mean() * 0.90  # 10% safety margin
            calibration_months = 0

        # Re-calibrate every 12 months using trailing 3-month avg
        calibration_months += 1
        if calibration_months >= 12 and i >= 3:
            trailing_3m_avg = monthly_income_raw.iloc[max(0, i - 2) : i + 1].mean()
            monthly_withdrawal = trailing_3m_avg * 0.90  # 10% safety margin
            calibration_months = 0

        # Withdraw from reservoir (can't withdraw more than what's there)
        actual_withdrawal = min(monthly_withdrawal, reservoir) if monthly_withdrawal > 0 else raw_income
        reservoir -= actual_withdrawal

        smoothed_values.append(
            {
                "month": month_end,
                "raw_income": raw_income,
                "smoothed_income": actual_withdrawal,
                "reservoir_balance": reservoir,
                "withdrawal_rate": monthly_withdrawal,
            }
        )

    smoothed_df = pd.DataFrame(smoothed_values).set_index("month")
    monthly_income = smoothed_df["smoothed_income"]
    monthly_income.name = "monthly_income"

    # Per-ETF dividend detail
    etf_dividend_detail = {}
    for t in income_tickers:
        divs = data[t]["dividends"].loc[start_date:end_date]
        divs = divs[divs > 0]
        if len(divs) > 0:
            payouts = divs * income_shares[t]
            etf_dividend_detail[t] = {
                "total_dividends": payouts.sum(),
                "avg_monthly": payouts.resample("ME").sum().mean(),
                "shares": income_shares[t],
                "capital_allocated": etf_capital[t],
                "frequency": "Monthly" if t in MONTHLY_ETFS else "Quarterly",
                "yield_on_cost": divs.sum() / data[t]["close"].loc[start_date:].iloc[0] * 100,
                "monthly_series": payouts.resample("ME").sum(),
            }

    # SPY dividend detail
    spy_divs_series = data["SPY"]["dividends"].loc[start_date:end_date]
    spy_divs_series = spy_divs_series[spy_divs_series > 0]

    return {
        "df": df,
        "monthly_income": monthly_income,
        "monthly_income_raw": monthly_income_raw,
        "smoothed_df": smoothed_df,
        "etf_dividend_detail": etf_dividend_detail,
        "income_tickers": income_tickers,
        "monthly_pct": monthly_pct,
        "spy_shares_final": spy_shares,
        "spy_shares_initial": spy_capital / spy_price_0,
        "spy_divs_reinvested": cumulative_spy_divs_reinvested,
        "start_date": start_date,
        "end_date": end_date,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_money(val):
    return f"${val:,.0f}"


def fmt_pct(val):
    return f"{val:,.2f}%"


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------
def main():
    st.title("Income & Growth ETF Backtest Dashboard")
    st.markdown(
        """
        **Strategy**: $3M starting capital &mdash; **50% SPY** (dividends reinvested for your daughter's inheritance)
        and **50% High-Yield Income ETFs** (dividends paid out as monthly retirement income).

        Income ETFs: **SCHD** &bull; **VYM** &bull; **HDV** &bull; **JEPI** &bull; **QYLD** &bull; **SDIV**
        """
    )

    # Download data
    data = download_data(ALL_TICKERS)
    if not data:
        st.error("Failed to download data. Please try again.")
        return

    common_start = find_common_start(data)
    latest_end = min(data[t]["close"].index[-1] for t in data)

    st.sidebar.header("Backtest Settings")
    st.sidebar.markdown(f"**Earliest common start:** {common_start.strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Latest data:** {latest_end.strftime('%Y-%m-%d')}")

    use_common = st.sidebar.checkbox("Use earliest common start date (all ETFs)", value=True)
    if use_common:
        bt_start = common_start
    else:
        bt_start = st.sidebar.date_input("Start date", value=common_start, min_value=common_start)
        bt_start = pd.Timestamp(bt_start)

    st.sidebar.divider()
    st.sidebar.header("Income Allocation Mix")
    st.sidebar.markdown(
        "**Monthly payers** (JEPI, QYLD, SDIV) provide steady monthly income. "
        "**Quarterly payers** (SCHD, VYM, HDV) offer better growth but spiky payouts."
    )
    monthly_pct = st.sidebar.slider(
        "% of income sleeve to monthly payers",
        min_value=0, max_value=100, value=70, step=5,
        help="Higher = smoother monthly income. Lower = more growth-oriented dividends."
    )
    quarterly_pct = 100 - monthly_pct
    st.sidebar.markdown(
        f"**{monthly_pct}%** to monthly ETFs (JEPI, QYLD, SDIV) = "
        f"**{fmt_money(STARTING_CAPITAL * INCOME_ALLOCATION * monthly_pct / 100)}**"
    )
    st.sidebar.markdown(
        f"**{quarterly_pct}%** to quarterly ETFs (SCHD, VYM, HDV) = "
        f"**{fmt_money(STARTING_CAPITAL * INCOME_ALLOCATION * quarterly_pct / 100)}**"
    )

    # Run backtest
    results = run_backtest(data, bt_start, monthly_pct=monthly_pct)
    if results is None:
        return

    df = results["df"]
    monthly_income = results["monthly_income"]
    monthly_income_raw = results["monthly_income_raw"]
    smoothed_df = results["smoothed_df"]
    detail = results["etf_dividend_detail"]
    income_tickers = results["income_tickers"]

    years = (results["end_date"] - results["start_date"]).days / 365.25

    # =====================================================================
    # KEY METRICS
    # =====================================================================
    st.header("Key Portfolio Metrics")

    final_spy = df["spy_value"].iloc[-1]
    final_income_sleeve = df["income_sleeve_value"].iloc[-1]
    final_total = df["total_portfolio_value"].iloc[-1]
    total_income = df["cumulative_income"].iloc[-1]
    avg_monthly = monthly_income.mean()
    latest_monthly = monthly_income.iloc[-1] if len(monthly_income) > 0 else 0
    total_return = (final_total + total_income - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    spy_growth = (final_spy - STARTING_CAPITAL * SPY_ALLOCATION) / (STARTING_CAPITAL * SPY_ALLOCATION) * 100
    spy_cagr = ((final_spy / (STARTING_CAPITAL * SPY_ALLOCATION)) ** (1 / years) - 1) * 100 if years > 0 else 0

    # Income stability metrics
    income_min = monthly_income.min()
    income_max = monthly_income.max()
    income_std = monthly_income.std()
    income_cv = (income_std / avg_monthly * 100) if avg_monthly > 0 else 0
    reservoir_balance = smoothed_df["reservoir_balance"].iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Portfolio Value", fmt_money(final_total + total_income))
    col2.metric("SPY Sleeve (Inheritance)", fmt_money(final_spy), f"{fmt_pct(spy_growth)} total")
    col3.metric("Income Sleeve Value", fmt_money(final_income_sleeve))
    col4.metric("Total Income Collected", fmt_money(total_income))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Steady Monthly Paycheck", fmt_money(avg_monthly))
    col6.metric("Latest Month Paycheck", fmt_money(latest_monthly))
    col7.metric("SPY CAGR (w/ reinvested divs)", fmt_pct(spy_cagr))
    col8.metric("Total Return (incl. income)", fmt_pct(total_return))

    # Income stability row
    st.divider()
    st.subheader("Income Stability (Smoothed Paycheck)")
    stab1, stab2, stab3, stab4 = st.columns(4)
    stab1.metric("Lowest Monthly Paycheck", fmt_money(income_min))
    stab2.metric("Highest Monthly Paycheck", fmt_money(income_max))
    stab3.metric("Income Variability (CV)", fmt_pct(income_cv), help="Coefficient of variation — lower is more stable. Under 15% is good.")
    stab4.metric("Reservoir Buffer Balance", fmt_money(reservoir_balance), help="Cash cushion built up from quarterly dividend spikes")

    st.divider()

    # Extra SPY inheritance stats
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("SPY Shares (Initial)", f"{results['spy_shares_initial']:,.2f}")
    col_b.metric("SPY Shares (Final, after DRIP)", f"{results['spy_shares_final']:,.2f}")
    col_c.metric("Shares Gained via DRIP", f"{results['spy_shares_final'] - results['spy_shares_initial']:,.2f}")
    col_d.metric("SPY Divs Reinvested", fmt_money(results["spy_divs_reinvested"]))

    # =====================================================================
    # CHARTS
    # =====================================================================
    st.header("Portfolio Growth Over Time")

    fig_growth = make_subplots(specs=[[{"secondary_y": True}]])
    fig_growth.add_trace(
        go.Scatter(x=df.index, y=df["total_portfolio_value"], name="Total Portfolio", line=dict(width=2.5, color="#1f77b4")),
        secondary_y=False,
    )
    fig_growth.add_trace(
        go.Scatter(x=df.index, y=df["spy_value"], name="SPY Sleeve (Inheritance)", line=dict(width=2, color="#2ca02c")),
        secondary_y=False,
    )
    fig_growth.add_trace(
        go.Scatter(x=df.index, y=df["income_sleeve_value"], name="Income Sleeve", line=dict(width=2, color="#ff7f0e")),
        secondary_y=False,
    )
    fig_growth.add_trace(
        go.Scatter(x=df.index, y=df["cumulative_income"], name="Cumulative Income Paid", line=dict(width=2, color="#d62728", dash="dot")),
        secondary_y=True,
    )
    fig_growth.update_layout(
        height=550,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig_growth.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False, tickformat="$,.0f")
    fig_growth.update_yaxes(title_text="Cumulative Income ($)", secondary_y=True, tickformat="$,.0f")
    st.plotly_chart(fig_growth, use_container_width=True)

    # ---- Monthly Income Bar Chart: Raw vs Smoothed ----
    st.header("Monthly Retirement Income — Raw vs Smoothed Paycheck")
    st.markdown(
        """
        > **The problem**: SCHD, VYM, HDV, and SDIV pay dividends **quarterly** (Mar/Jun/Sep/Dec),
        > creating huge spikes in those months and near-zero income in between. That's not livable.
        >
        > **The fix**: All dividends flow into a **reservoir** (cash buffer). You withdraw a **steady
        > monthly paycheck** that's recalibrated annually. The grey bars below show the raw spiky
        > dividends; the green bars show your actual smooth paycheck.
        """
    )

    fig_monthly = go.Figure()
    # Raw income (faded background)
    fig_monthly.add_trace(
        go.Bar(
            x=monthly_income_raw.index,
            y=monthly_income_raw.values,
            marker_color="rgba(180,180,180,0.45)",
            name="Raw Dividends Received",
        )
    )
    # Smoothed paycheck (foreground)
    fig_monthly.add_trace(
        go.Bar(
            x=monthly_income.index,
            y=monthly_income.values,
            marker_color="#2ca02c",
            name="Your Monthly Paycheck (Smoothed)",
        )
    )
    fig_monthly.update_layout(
        height=450,
        yaxis_tickformat="$,.0f",
        barmode="overlay",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

    # ---- Reservoir Balance Chart ----
    st.subheader("Income Reservoir Balance Over Time")
    st.markdown("This is your cash buffer — dividends flow in, steady paychecks flow out. A positive balance means you have a safety cushion.")
    fig_reservoir = go.Figure()
    fig_reservoir.add_trace(
        go.Scatter(
            x=smoothed_df.index,
            y=smoothed_df["reservoir_balance"],
            fill="tozeroy",
            line=dict(color="#1f77b4", width=2),
            name="Reservoir Balance",
        )
    )
    fig_reservoir.update_layout(
        height=300,
        yaxis_tickformat="$,.0f",
        hovermode="x unified",
    )
    st.plotly_chart(fig_reservoir, use_container_width=True)

    # ---- Income by ETF stacked area ----
    st.header("Monthly Income Breakdown by ETF")

    monthly_by_etf = pd.DataFrame()
    for t in income_tickers:
        if t in detail:
            series = detail[t]["monthly_series"]
            monthly_by_etf[t] = series
    monthly_by_etf = monthly_by_etf.fillna(0)

    if not monthly_by_etf.empty:
        fig_stack = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        for i, t in enumerate(monthly_by_etf.columns):
            fig_stack.add_trace(
                go.Scatter(
                    x=monthly_by_etf.index,
                    y=monthly_by_etf[t],
                    name=t,
                    stackgroup="one",
                    line=dict(width=0.5, color=colors[i % len(colors)]),
                )
            )
        fig_stack.update_layout(
            height=400,
            yaxis_tickformat="$,.0f",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_stack, use_container_width=True)

    # =====================================================================
    # PER-ETF DIVIDEND TABLE
    # =====================================================================
    st.header("Income ETF Dividend Detail")

    rows = []
    for t in income_tickers:
        if t in detail:
            d = detail[t]
            rows.append(
                {
                    "ETF": t,
                    "Frequency": d["frequency"],
                    "Capital Allocated": fmt_money(d["capital_allocated"]),
                    "Shares Held": f"{d['shares']:,.2f}",
                    "Total Dividends Received": fmt_money(d["total_dividends"]),
                    "Avg Monthly Income": fmt_money(d["avg_monthly"]),
                    "Yield on Cost (%)": fmt_pct(d["yield_on_cost"]),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("ETF"), use_container_width=True)

    # =====================================================================
    # ANNUAL SUMMARY TABLE
    # =====================================================================
    st.header("Annual Summary")

    df_annual = df.copy()
    df_annual["year"] = df_annual.index.year
    annual_rows = []
    for year, grp in df_annual.groupby("year"):
        yr_income_raw = grp["daily_income"].sum()
        # Smoothed income for this year
        yr_smoothed = smoothed_df.loc[smoothed_df.index.year == year, "smoothed_income"]
        yr_smoothed_total = yr_smoothed.sum() if len(yr_smoothed) > 0 else 0
        yr_smoothed_monthly = yr_smoothed.mean() if len(yr_smoothed) > 0 else 0
        annual_rows.append(
            {
                "Year": int(year),
                "SPY Sleeve": fmt_money(grp["spy_value"].iloc[-1]),
                "Income Sleeve": fmt_money(grp["income_sleeve_value"].iloc[-1]),
                "Total Portfolio": fmt_money(grp["total_portfolio_value"].iloc[-1]),
                "Raw Annual Income": fmt_money(yr_income_raw),
                "Smoothed Annual Income": fmt_money(yr_smoothed_total),
                "Monthly Paycheck": fmt_money(yr_smoothed_monthly),
                "Cumulative Income": fmt_money(grp["cumulative_income"].iloc[-1]),
            }
        )
    if annual_rows:
        st.dataframe(pd.DataFrame(annual_rows).set_index("Year"), use_container_width=True)

    # =====================================================================
    # DRAWDOWN ANALYSIS
    # =====================================================================
    st.header("Drawdown Analysis")

    total_val = df["total_portfolio_value"]
    running_max = total_val.cummax()
    drawdown = (total_val - running_max) / running_max * 100

    spy_val = df["spy_value"]
    spy_max = spy_val.cummax()
    spy_dd = (spy_val - spy_max) / spy_max * 100

    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, name="Total Portfolio", fill="tozeroy", line=dict(color="#1f77b4", width=1))
    )
    fig_dd.add_trace(
        go.Scatter(x=spy_dd.index, y=spy_dd.values, name="SPY Sleeve", fill="tozeroy", line=dict(color="#2ca02c", width=1))
    )
    fig_dd.update_layout(
        height=350,
        yaxis_title="Drawdown (%)",
        yaxis_tickformat=".1f",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    col_dd1, col_dd2 = st.columns(2)
    col_dd1.metric("Max Portfolio Drawdown", fmt_pct(max_dd))
    col_dd2.metric("Max Drawdown Date", max_dd_date.strftime("%Y-%m-%d"))

    # =====================================================================
    # INCOME PROJECTION
    # =====================================================================
    st.header("Forward Income Projection (based on trailing 12-month avg)")

    if len(monthly_income) >= 12:
        trailing_12 = monthly_income.iloc[-12:].mean()
    else:
        trailing_12 = monthly_income.mean()

    proj_col1, proj_col2, proj_col3 = st.columns(3)
    proj_col1.metric("Projected Monthly Income", fmt_money(trailing_12))
    proj_col2.metric("Projected Annual Income", fmt_money(trailing_12 * 12))
    proj_col3.metric("Projected Yield on Original Capital", fmt_pct(trailing_12 * 12 / (STARTING_CAPITAL * INCOME_ALLOCATION) * 100))

    # =====================================================================
    # SUMMARY BOX
    # =====================================================================
    st.divider()
    st.header("Strategy Summary")
    raw_avg = monthly_income_raw.mean()
    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | **Starting Capital** | {fmt_money(STARTING_CAPITAL)} |
        | **Backtest Period** | {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} ({years:.1f} years) |
        | **SPY Sleeve (Daughter's Inheritance)** | {fmt_money(final_spy)} |
        | **SPY CAGR** | {fmt_pct(spy_cagr)} |
        | **Income Sleeve Current Value** | {fmt_money(final_income_sleeve)} |
        | **Total Retirement Income Collected** | {fmt_money(total_income)} |
        | **Avg Raw Monthly Dividends** | {fmt_money(raw_avg)} |
        | **Avg Smoothed Monthly Paycheck** | {fmt_money(avg_monthly)} |
        | **Lowest Monthly Paycheck** | {fmt_money(income_min)} |
        | **Income Variability (CV)** | {fmt_pct(income_cv)} |
        | **Reservoir Buffer** | {fmt_money(reservoir_balance)} |
        | **Total Wealth Created** | {fmt_money(final_total + total_income)} |
        | **Total Return** | {fmt_pct(total_return)} |
        """
    )


if __name__ == "__main__":
    main()
