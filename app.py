"""
Total Return + Income Sleeve Dashboard
========================================
3-Bucket Institutional Retirement Strategy on $3,500,000

Bucket 1 — Income Engine (50%):  JEPI, JEPQ, SPYI, QQQI  (dividends paid out)
Bucket 2 — Growth Engine (35%):  SPY, VTI                 (dividends reinvested)
Bucket 3 — Dividend Growth (15%): SCHD, DGRO              (dividends reinvested)
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Total Return + Income Sleeve Dashboard",
    page_icon="💰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STARTING_CAPITAL = 3_500_000
INCOME_PCT = 0.50
GROWTH_PCT = 0.35
DIV_GROWTH_PCT = 0.15

INCOME_ETFS = ["JEPI", "JEPQ", "SPYI", "QQQI"]
GROWTH_ETFS = ["SPY", "VTI"]
DIV_GROWTH_ETFS = ["SCHD", "DGRO"]
ALL_TICKERS = INCOME_ETFS + GROWTH_ETFS + DIV_GROWTH_ETFS


def fmt_money(val):
    return f"${val:,.0f}"


def fmt_pct(val):
    return f"{val:,.2f}%"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Downloading market data …")
def download_data(tickers):
    data = {}
    try:
        for t in tickers:
            tk = yf.Ticker(t)
            hist = tk.history(period="max", auto_adjust=False)
            if hist.empty:
                continue
            hist.index = hist.index.tz_localize(None)
            data[t] = {"close": hist["Close"].copy(), "dividends": hist["Dividends"].copy()}
    except Exception:
        pass
    if len(data) < len(tickers):
        data = _generate_demo_data(tickers)
    return data


def _generate_demo_data(tickers):
    np.random.seed(42)
    start = pd.Timestamp("2020-06-01")
    end = pd.Timestamp("2026-03-31")
    dates = pd.bdate_range(start, end)
    params = {
        "SPY":  {"ret": 0.12, "vol": 0.18, "yld": 0.013, "freq": "Q", "p0": 310},
        "VTI":  {"ret": 0.11, "vol": 0.17, "yld": 0.014, "freq": "Q", "p0": 155},
        "SCHD": {"ret": 0.09, "vol": 0.15, "yld": 0.035, "freq": "Q", "p0": 52},
        "DGRO": {"ret": 0.10, "vol": 0.15, "yld": 0.022, "freq": "Q", "p0": 40},
        "JEPI": {"ret": 0.06, "vol": 0.10, "yld": 0.075, "freq": "M", "p0": 50},
        "JEPQ": {"ret": 0.08, "vol": 0.13, "yld": 0.095, "freq": "M", "p0": 45},
        "SPYI": {"ret": 0.07, "vol": 0.12, "yld": 0.120, "freq": "M", "p0": 25},
        "QQQI": {"ret": 0.05, "vol": 0.14, "yld": 0.140, "freq": "M", "p0": 28},
    }
    data = {}
    for t in tickers:
        p = params.get(t)
        if p is None:
            continue
        n = len(dates)
        rets = np.random.normal(p["ret"] / 252, p["vol"] / np.sqrt(252), n)
        prices = p["p0"] * np.cumprod(1 + rets)
        close = pd.Series(prices, index=dates)
        divs = pd.Series(0.0, index=dates)
        if p["freq"] == "Q":
            for yr in range(start.year, end.year + 1):
                for mo in [3, 6, 9, 12]:
                    dt = pd.Timestamp(yr, mo, 20)
                    cands = dates[(dates >= dt - pd.Timedelta(days=5)) & (dates <= dt + pd.Timedelta(days=5))]
                    if len(cands) > 0:
                        d = cands[len(cands) // 2]
                        divs[d] = max(close.get(d, p["p0"]) * p["yld"] / 4 * np.random.uniform(0.85, 1.15), 0.01)
        else:
            for yr in range(start.year, end.year + 1):
                for mo in range(1, 13):
                    try:
                        dt = pd.Timestamp(yr, mo, 15)
                    except ValueError:
                        continue
                    cands = dates[(dates >= dt - pd.Timedelta(days=5)) & (dates <= dt + pd.Timedelta(days=5))]
                    if len(cands) > 0:
                        d = cands[len(cands) // 2]
                        divs[d] = max(close.get(d, p["p0"]) * p["yld"] / 12 * np.random.uniform(0.85, 1.15), 0.01)
        data[t] = {"close": close, "dividends": divs}
    return data


def find_common_start(data):
    starts = [data[t]["close"].index[0] for t in data]
    return max(starts)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(data, start_date, starting_age=57, end_date=None):
    inc_tickers = [t for t in INCOME_ETFS if t in data]
    gro_tickers = [t for t in GROWTH_ETFS if t in data]
    dvg_tickers = [t for t in DIV_GROWTH_ETFS if t in data]
    all_available = inc_tickers + gro_tickers + dvg_tickers

    if len(inc_tickers) == 0 or len(gro_tickers) == 0:
        st.error("Not enough ticker data.")
        return None

    if end_date is None:
        end_date = min(data[t]["close"].index[-1] for t in all_available)

    all_dates = None
    for t in all_available:
        idx = data[t]["close"].loc[start_date:end_date].index
        all_dates = idx if all_dates is None else all_dates.union(idx)
    all_dates = all_dates.sort_values()

    # Initial share purchases
    inc_cap_each = (STARTING_CAPITAL * INCOME_PCT) / max(len(inc_tickers), 1)
    gro_cap_each = (STARTING_CAPITAL * GROWTH_PCT) / max(len(gro_tickers), 1)
    dvg_cap_each = (STARTING_CAPITAL * DIV_GROWTH_PCT) / max(len(dvg_tickers), 1)

    shares = {}
    etf_capital = {}
    for t in inc_tickers:
        p0 = data[t]["close"].loc[start_date:].iloc[0]
        shares[t] = inc_cap_each / p0
        etf_capital[t] = inc_cap_each
    for t in gro_tickers:
        p0 = data[t]["close"].loc[start_date:].iloc[0]
        shares[t] = gro_cap_each / p0
        etf_capital[t] = gro_cap_each
    for t in dvg_tickers:
        p0 = data[t]["close"].loc[start_date:].iloc[0]
        shares[t] = dvg_cap_each / p0
        etf_capital[t] = dvg_cap_each

    initial_gro_shares = {t: shares[t] for t in gro_tickers}

    records = []
    cum_income = 0.0
    cum_gro_divs = 0.0
    cum_dvg_divs = 0.0

    for date in all_dates:
        daily_income = 0.0
        inc_val = 0.0
        gro_val = 0.0
        dvg_val = 0.0

        # Income sleeve — pay out dividends
        for t in inc_tickers:
            price = data[t]["close"].get(date, np.nan)
            if np.isnan(price):
                continue
            div = data[t]["dividends"].get(date, 0.0)
            if div > 0:
                daily_income += div * shares[t]
            inc_val += shares[t] * price

        # Growth sleeve — reinvest dividends
        for t in gro_tickers:
            price = data[t]["close"].get(date, np.nan)
            if np.isnan(price):
                continue
            div = data[t]["dividends"].get(date, 0.0)
            if div > 0 and price > 0:
                div_cash = div * shares[t]
                shares[t] += div_cash / price
                cum_gro_divs += div_cash
            gro_val += shares[t] * price

        # Dividend growth sleeve — reinvest dividends
        for t in dvg_tickers:
            price = data[t]["close"].get(date, np.nan)
            if np.isnan(price):
                continue
            div = data[t]["dividends"].get(date, 0.0)
            if div > 0 and price > 0:
                div_cash = div * shares[t]
                shares[t] += div_cash / price
                cum_dvg_divs += div_cash
            dvg_val += shares[t] * price

        cum_income += daily_income
        total = inc_val + gro_val + dvg_val

        records.append({
            "date": date,
            "income_value": inc_val,
            "growth_value": gro_val,
            "div_growth_value": dvg_val,
            "total_portfolio_value": total,
            "daily_income": daily_income,
            "cumulative_income": cum_income,
            "cum_gro_divs": cum_gro_divs,
            "cum_dvg_divs": cum_dvg_divs,
        })

    df = pd.DataFrame(records).set_index("date")
    monthly_income_raw = df["daily_income"].resample("ME").sum()
    monthly_income_raw.name = "monthly_income_raw"

    # Income smoothing reservoir
    reservoir = 0.0
    smoothed_values = []
    monthly_wd = 0.0
    cal_months = 0
    for i, (me, raw) in enumerate(monthly_income_raw.items()):
        reservoir += raw
        if i == 2 and monthly_wd == 0:
            monthly_wd = monthly_income_raw.iloc[:3].mean() * 0.90
            cal_months = 0
        cal_months += 1
        if cal_months >= 12 and i >= 3:
            monthly_wd = monthly_income_raw.iloc[max(0, i - 2):i + 1].mean() * 0.90
            cal_months = 0
        actual = min(monthly_wd, reservoir) if monthly_wd > 0 else raw
        reservoir -= actual
        smoothed_values.append({"month": me, "raw_income": raw, "smoothed_income": actual, "reservoir_balance": reservoir})

    smoothed_df = pd.DataFrame(smoothed_values).set_index("month")
    monthly_income = smoothed_df["smoothed_income"]
    monthly_income.name = "monthly_income"

    # Per-ETF detail
    etf_detail = {}
    for t in all_available:
        bucket = "Income" if t in inc_tickers else ("Growth" if t in gro_tickers else "Div Growth")
        freq = "Monthly" if t in INCOME_ETFS else "Quarterly"
        divs = data[t]["dividends"].loc[start_date:end_date]
        divs = divs[divs > 0]
        if len(divs) > 0:
            if t in inc_tickers:
                payouts = divs * (etf_capital[t] / data[t]["close"].loc[start_date:].iloc[0])
            else:
                payouts = divs * shares[t]  # approximate
            p0 = data[t]["close"].loc[start_date:].iloc[0]
            etf_detail[t] = {
                "total_dividends": (divs * (etf_capital[t] / p0)).sum(),
                "avg_monthly": (divs * (etf_capital[t] / p0)).resample("ME").sum().mean(),
                "shares": shares[t],
                "capital_allocated": etf_capital[t],
                "bucket": bucket,
                "frequency": freq,
                "yield_on_cost": divs.sum() / p0 * 100,
                "monthly_series": (divs * (etf_capital[t] / p0)).resample("ME").sum(),
            }

    return {
        "df": df, "monthly_income": monthly_income, "monthly_income_raw": monthly_income_raw,
        "smoothed_df": smoothed_df, "etf_detail": etf_detail,
        "inc_tickers": inc_tickers, "gro_tickers": gro_tickers, "dvg_tickers": dvg_tickers,
        "initial_gro_shares": initial_gro_shares,
        "final_shares": {t: shares[t] for t in gro_tickers},
        "cum_gro_divs": cum_gro_divs, "cum_dvg_divs": cum_dvg_divs,
        "start_date": start_date, "end_date": end_date, "starting_age": starting_age,
    }


# ---------------------------------------------------------------------------
# Forward projection
# ---------------------------------------------------------------------------
def project_forward(results, target_age=90):
    starting_age = results["starting_age"]
    df = results["df"]
    monthly_income = results["monthly_income"]
    bt_years = (results["end_date"] - results["start_date"]).days / 365.25

    current_age = starting_age + bt_years
    portfolio_now = df["total_portfolio_value"].iloc[-1]
    income_now = df["income_value"].iloc[-1]
    growth_now = df["growth_value"].iloc[-1]
    dvg_now = df["div_growth_value"].iloc[-1]

    trailing_annual_income = monthly_income.iloc[-12:].sum() if len(monthly_income) >= 12 else monthly_income.sum()

    # Allocation percentages (starting)
    inc_pct = INCOME_PCT
    gro_pct = GROWTH_PCT
    dvg_pct = DIV_GROWTH_PCT

    rows = []
    port_val = portfolio_now
    annual_inc = trailing_annual_income
    age = current_age

    # Add current state
    rows.append({
        "Age": int(round(age)), "Portfolio": port_val,
        "Annual Income": annual_inc, "Monthly Income": annual_inc / 12,
        "Income %": inc_pct * 100, "Growth %": gro_pct * 100, "Div Growth %": dvg_pct * 100,
    })

    while int(round(age)) < target_age:
        age += 1
        # Every 5 years, shift 5% from growth to income
        age_rounded = int(round(age))
        if age_rounded % 5 == 0 and age_rounded > int(round(current_age)):
            shift = 0.05
            if gro_pct >= shift:
                gro_pct -= shift
                inc_pct += shift

        # Portfolio grows at blended rate (growth portion grows faster)
        growth_rate = gro_pct * 0.10 + dvg_pct * 0.08 + inc_pct * 0.02
        port_val = port_val * (1 + growth_rate)

        # Income grows ~4%/year (dividend growth + yield on growing assets)
        annual_inc = annual_inc * 1.04

        rows.append({
            "Age": age_rounded, "Portfolio": port_val,
            "Annual Income": annual_inc, "Monthly Income": annual_inc / 12,
            "Income %": inc_pct * 100, "Growth %": gro_pct * 100, "Div Growth %": dvg_pct * 100,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dashboard UI
# ---------------------------------------------------------------------------
def main():
    st.title("Total Return + Income Sleeve Dashboard")
    st.markdown(
        """
        **Institutional 3-Bucket Retirement Strategy** on **$3.5M**

        - **Bucket 1 — Income Engine (50%)**: JEPI, JEPQ, SPYI, QQQI — covered call ETFs, monthly cash flow
        - **Bucket 2 — Growth Engine (35%)**: SPY, VTI — broad market, dividends reinvested for inheritance
        - **Bucket 3 — Dividend Growth (15%)**: SCHD, DGRO — rising dividends, inflation protection
        """
    )

    data = download_data(ALL_TICKERS)
    if not data:
        st.error("Failed to download data.")
        return

    common_start = find_common_start(data)
    latest_end = min(data[t]["close"].index[-1] for t in data)

    # Sidebar
    st.sidebar.header("Backtest Settings")
    st.sidebar.markdown(f"**Earliest common start:** {common_start.strftime('%Y-%m-%d')}")
    st.sidebar.markdown(f"**Latest data:** {latest_end.strftime('%Y-%m-%d')}")
    use_common = st.sidebar.checkbox("Use earliest common start date", value=True)
    if use_common:
        bt_start = common_start
    else:
        bt_start = pd.Timestamp(st.sidebar.date_input("Start date", value=common_start, min_value=common_start))

    st.sidebar.divider()
    st.sidebar.header("Your Profile")
    starting_age = st.sidebar.number_input("Age at backtest start", min_value=30, max_value=80, value=57)
    target_age = st.sidebar.slider("Project income to age", min_value=70, max_value=100, value=90)

    st.sidebar.divider()
    st.sidebar.header("Bucket Allocation")
    st.sidebar.markdown(f"**Income Engine (50%):** {fmt_money(STARTING_CAPITAL * INCOME_PCT)}")
    st.sidebar.markdown("JEPI • JEPQ • SPYI • QQQI")
    st.sidebar.markdown(f"**Growth Engine (35%):** {fmt_money(STARTING_CAPITAL * GROWTH_PCT)}")
    st.sidebar.markdown("SPY • VTI")
    st.sidebar.markdown(f"**Dividend Growth (15%):** {fmt_money(STARTING_CAPITAL * DIV_GROWTH_PCT)}")
    st.sidebar.markdown("SCHD • DGRO")

    results = run_backtest(data, bt_start, starting_age=starting_age)
    if results is None:
        return

    df = results["df"]
    monthly_income = results["monthly_income"]
    monthly_income_raw = results["monthly_income_raw"]
    smoothed_df = results["smoothed_df"]
    detail = results["etf_detail"]
    years = (results["end_date"] - results["start_date"]).days / 365.25

    final_growth = df["growth_value"].iloc[-1]
    final_income_sleeve = df["income_value"].iloc[-1]
    final_dvg = df["div_growth_value"].iloc[-1]
    final_total = df["total_portfolio_value"].iloc[-1]
    total_income_paid = df["cumulative_income"].iloc[-1]
    avg_monthly = monthly_income.mean()
    latest_monthly = monthly_income.iloc[-1] if len(monthly_income) > 0 else 0
    total_return = (final_total + total_income_paid - STARTING_CAPITAL) / STARTING_CAPITAL * 100
    gro_init_cap = STARTING_CAPITAL * GROWTH_PCT
    growth_pct = (final_growth - gro_init_cap) / gro_init_cap * 100
    growth_cagr = ((final_growth / gro_init_cap) ** (1 / years) - 1) * 100 if years > 0 else 0
    income_min = monthly_income.min()
    income_max = monthly_income.max()
    income_cv = (monthly_income.std() / avg_monthly * 100) if avg_monthly > 0 else 0
    reservoir_bal = smoothed_df["reservoir_balance"].iloc[-1]

    # === KEY METRICS ===
    st.header("Key Portfolio Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio Value", fmt_money(final_total + total_income_paid))
    c2.metric("Growth Sleeve (Inheritance)", fmt_money(final_growth), f"{fmt_pct(growth_pct)} total")
    c3.metric("Income Sleeve Value", fmt_money(final_income_sleeve))
    c4.metric("Div Growth Sleeve", fmt_money(final_dvg))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Steady Monthly Paycheck", fmt_money(avg_monthly))
    c6.metric("Latest Month Paycheck", fmt_money(latest_monthly))
    c7.metric("Growth CAGR (w/ DRIP)", fmt_pct(growth_cagr))
    c8.metric("Total Return (incl. income)", fmt_pct(total_return))

    st.divider()
    st.subheader("Income Stability")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Lowest Monthly Paycheck", fmt_money(income_min))
    s2.metric("Highest Monthly Paycheck", fmt_money(income_max))
    s3.metric("Income Variability (CV)", fmt_pct(income_cv))
    s4.metric("Reservoir Buffer Balance", fmt_money(reservoir_bal))

    st.divider()
    st.subheader("Growth Sleeve Detail (Daughter's Inheritance)")
    g1, g2, g3, g4 = st.columns(4)
    init_shares_total = sum(results["initial_gro_shares"].values())
    final_shares_total = sum(results["final_shares"].values())
    g1.metric("Growth Shares (Initial)", f"{init_shares_total:,.2f}")
    g2.metric("Growth Shares (Final, DRIP)", f"{final_shares_total:,.2f}")
    g3.metric("Shares Gained via DRIP", f"{final_shares_total - init_shares_total:,.2f}")
    g4.metric("Growth Divs Reinvested", fmt_money(results["cum_gro_divs"]))

    # === PORTFOLIO GROWTH CHART ===
    st.header("Portfolio Growth Over Time")
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    fig1.add_trace(go.Scatter(x=df.index, y=df["total_portfolio_value"], name="Total Portfolio", line=dict(width=2.5, color="#1f77b4")), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df.index, y=df["growth_value"], name="Growth Sleeve (Inheritance)", line=dict(width=2, color="#2ca02c")), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df.index, y=df["income_value"], name="Income Sleeve", line=dict(width=2, color="#ff7f0e")), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df.index, y=df["div_growth_value"], name="Div Growth Sleeve", line=dict(width=2, color="#9467bd")), secondary_y=False)
    fig1.add_trace(go.Scatter(x=df.index, y=df["cumulative_income"], name="Cumulative Income Paid", line=dict(width=2, color="#d62728", dash="dot")), secondary_y=True)
    fig1.update_layout(height=550, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig1.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False, tickformat="$,.0f")
    fig1.update_yaxes(title_text="Cumulative Income ($)", secondary_y=True, tickformat="$,.0f")
    st.plotly_chart(fig1, use_container_width=True)

    # === MONTHLY INCOME RAW vs SMOOTHED ===
    st.header("Monthly Retirement Income — Raw vs Smoothed Paycheck")
    st.markdown(
        "> All income ETF dividends flow into a **cash reservoir**. You withdraw a **steady monthly "
        "paycheck** recalibrated annually. Grey = raw dividends. Green = your actual paycheck."
    )
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=monthly_income_raw.index, y=monthly_income_raw.values, marker_color="rgba(180,180,180,0.45)", name="Raw Dividends"))
    fig2.add_trace(go.Bar(x=monthly_income.index, y=monthly_income.values, marker_color="#2ca02c", name="Your Monthly Paycheck"))
    fig2.update_layout(height=450, yaxis_tickformat="$,.0f", barmode="overlay", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, use_container_width=True)

    # === RESERVOIR ===
    st.subheader("Income Reservoir Balance")
    st.markdown("Cash buffer — dividends flow in, steady paychecks flow out.")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=smoothed_df.index, y=smoothed_df["reservoir_balance"], fill="tozeroy", line=dict(color="#1f77b4", width=2), name="Reservoir"))
    fig3.update_layout(height=300, yaxis_tickformat="$,.0f", hovermode="x unified")
    st.plotly_chart(fig3, use_container_width=True)

    # === INCOME BY ETF ===
    st.header("Monthly Income Breakdown by ETF")
    monthly_by_etf = pd.DataFrame()
    for t in results["inc_tickers"]:
        if t in detail:
            monthly_by_etf[t] = detail[t]["monthly_series"]
    monthly_by_etf = monthly_by_etf.fillna(0)
    if not monthly_by_etf.empty:
        fig4 = go.Figure()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, t in enumerate(monthly_by_etf.columns):
            fig4.add_trace(go.Scatter(x=monthly_by_etf.index, y=monthly_by_etf[t], name=t, stackgroup="one", line=dict(width=0.5, color=colors[i % len(colors)])))
        fig4.update_layout(height=400, yaxis_tickformat="$,.0f", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig4, use_container_width=True)

    # === ETF DETAIL TABLE ===
    st.header("ETF Dividend Detail (All Buckets)")
    rows = []
    for t in ALL_TICKERS:
        if t in detail:
            d = detail[t]
            rows.append({"ETF": t, "Bucket": d["bucket"], "Frequency": d["frequency"],
                         "Capital Allocated": fmt_money(d["capital_allocated"]),
                         "Shares": f"{d['shares']:,.2f}",
                         "Total Dividends": fmt_money(d["total_dividends"]),
                         "Avg Monthly": fmt_money(d["avg_monthly"]),
                         "Yield on Cost": fmt_pct(d["yield_on_cost"])})
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("ETF"), use_container_width=True)

    # === ANNUAL SUMMARY ===
    st.header("Annual Summary")
    df_a = df.copy()
    df_a["year"] = df_a.index.year
    a_rows = []
    for yr, grp in df_a.groupby("year"):
        yr_raw = grp["daily_income"].sum()
        yr_sm = smoothed_df.loc[smoothed_df.index.year == yr, "smoothed_income"]
        a_rows.append({"Year": int(yr),
                       "Growth Sleeve": fmt_money(grp["growth_value"].iloc[-1]),
                       "Income Sleeve": fmt_money(grp["income_value"].iloc[-1]),
                       "Div Growth": fmt_money(grp["div_growth_value"].iloc[-1]),
                       "Total Portfolio": fmt_money(grp["total_portfolio_value"].iloc[-1]),
                       "Raw Income": fmt_money(yr_raw),
                       "Smoothed Income": fmt_money(yr_sm.sum() if len(yr_sm) > 0 else 0),
                       "Monthly Paycheck": fmt_money(yr_sm.mean() if len(yr_sm) > 0 else 0),
                       "Cumulative Income": fmt_money(grp["cumulative_income"].iloc[-1])})
    if a_rows:
        st.dataframe(pd.DataFrame(a_rows).set_index("Year"), use_container_width=True)

    # === DRAWDOWN ===
    st.header("Drawdown Analysis")
    tv = df["total_portfolio_value"]
    dd = (tv - tv.cummax()) / tv.cummax() * 100
    gv = df["growth_value"]
    gdd = (gv - gv.cummax()) / gv.cummax() * 100
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Total Portfolio", fill="tozeroy", line=dict(color="#1f77b4", width=1)))
    fig5.add_trace(go.Scatter(x=gdd.index, y=gdd.values, name="Growth Sleeve", fill="tozeroy", line=dict(color="#2ca02c", width=1)))
    fig5.update_layout(height=350, yaxis_title="Drawdown (%)", yaxis_tickformat=".1f", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig5, use_container_width=True)
    dd1, dd2 = st.columns(2)
    dd1.metric("Max Portfolio Drawdown", fmt_pct(dd.min()))
    dd2.metric("Max Drawdown Date", dd.idxmin().strftime("%Y-%m-%d"))

    # === FORWARD PROJECTION ===
    st.header(f"Forward Projection to Age {target_age}")
    st.markdown(
        "Projection assumes **5.5% blended portfolio growth**, **4% annual income growth**, "
        "and **5% shift from Growth to Income every 5 years** (institutional rebalancing)."
    )
    proj = project_forward(results, target_age=target_age)

    # Milestone metrics
    milestones = [65, 70, 75, 80, 85, 90]
    avail_milestones = [m for m in milestones if m in proj["Age"].values and m <= target_age]
    if avail_milestones:
        cols = st.columns(len(avail_milestones))
        for i, age in enumerate(avail_milestones):
            row = proj[proj["Age"] == age].iloc[0]
            cols[i].metric(f"Age {age}", fmt_money(row["Monthly Income"]) + "/mo", fmt_money(row["Portfolio"]) + " portfolio")

    # Projection chart
    fig6 = make_subplots(specs=[[{"secondary_y": True}]])
    fig6.add_trace(go.Scatter(x=proj["Age"], y=proj["Portfolio"], name="Portfolio Value", line=dict(width=2.5, color="#1f77b4"), fill="tozeroy"), secondary_y=False)
    fig6.add_trace(go.Scatter(x=proj["Age"], y=proj["Monthly Income"], name="Monthly Income", line=dict(width=2.5, color="#2ca02c")), secondary_y=True)
    fig6.update_layout(height=450, hovermode="x unified", xaxis_title="Age", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig6.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False, tickformat="$,.0f")
    fig6.update_yaxes(title_text="Monthly Income ($)", secondary_y=True, tickformat="$,.0f")
    st.plotly_chart(fig6, use_container_width=True)

    # Projection table
    proj_display = proj.copy()
    proj_display["Portfolio"] = proj_display["Portfolio"].apply(fmt_money)
    proj_display["Annual Income"] = proj_display["Annual Income"].apply(fmt_money)
    proj_display["Monthly Income"] = proj_display["Monthly Income"].apply(fmt_money)
    proj_display["Income %"] = proj_display["Income %"].apply(lambda x: f"{x:.0f}%")
    proj_display["Growth %"] = proj_display["Growth %"].apply(lambda x: f"{x:.0f}%")
    proj_display["Div Growth %"] = proj_display["Div Growth %"].apply(lambda x: f"{x:.0f}%")
    st.dataframe(proj_display.set_index("Age"), use_container_width=True)

    # === STRATEGY SUMMARY ===
    st.divider()
    st.header("Strategy Summary")
    raw_avg = monthly_income_raw.mean()
    st.markdown(
        f"""
        | Metric | Value |
        |--------|-------|
        | **Starting Capital** | {fmt_money(STARTING_CAPITAL)} |
        | **Backtest Period** | {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} ({years:.1f} years) |
        | **Growth Sleeve (Inheritance)** | {fmt_money(final_growth)} |
        | **Growth CAGR** | {fmt_pct(growth_cagr)} |
        | **Income Sleeve Value** | {fmt_money(final_income_sleeve)} |
        | **Div Growth Sleeve Value** | {fmt_money(final_dvg)} |
        | **Total Income Collected** | {fmt_money(total_income_paid)} |
        | **Avg Monthly Paycheck** | {fmt_money(avg_monthly)} |
        | **Lowest Monthly Paycheck** | {fmt_money(income_min)} |
        | **Income Variability (CV)** | {fmt_pct(income_cv)} |
        | **Reservoir Buffer** | {fmt_money(reservoir_bal)} |
        | **Total Wealth Created** | {fmt_money(final_total + total_income_paid)} |
        | **Total Return** | {fmt_pct(total_return)} |
        """
    )


if __name__ == "__main__":
    main()
