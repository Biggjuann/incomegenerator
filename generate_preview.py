"""Generate a standalone HTML preview of the dashboard."""
import sys
sys.path.insert(0, "/home/user/incomegenerator")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import data generation and backtest logic
from app import _generate_demo_data, ALL_TICKERS, INCOME_ETFS, STARTING_CAPITAL, SPY_ALLOCATION, INCOME_ALLOCATION, find_common_start, run_backtest

def fmt_money(val):
    return f"${val:,.0f}"

def fmt_pct(val):
    return f"{val:,.2f}%"

# Generate data and run backtest
data = _generate_demo_data(ALL_TICKERS)
start = find_common_start(data)
results = run_backtest(data, start)

df = results["df"]
monthly_income = results["monthly_income"]
monthly_income_raw = results["monthly_income_raw"]
smoothed_df = results["smoothed_df"]
detail = results["etf_dividend_detail"]
income_tickers = results["income_tickers"]
years = (results["end_date"] - results["start_date"]).days / 365.25

final_spy = df["spy_value"].iloc[-1]
final_income_sleeve = df["income_sleeve_value"].iloc[-1]
final_total = df["total_portfolio_value"].iloc[-1]
total_income = df["cumulative_income"].iloc[-1]
avg_monthly = monthly_income.mean()
latest_monthly = monthly_income.iloc[-1]
total_return = (final_total + total_income - STARTING_CAPITAL) / STARTING_CAPITAL * 100
spy_growth = (final_spy - STARTING_CAPITAL * SPY_ALLOCATION) / (STARTING_CAPITAL * SPY_ALLOCATION) * 100
spy_cagr = ((final_spy / (STARTING_CAPITAL * SPY_ALLOCATION)) ** (1 / years) - 1) * 100
income_min = monthly_income.min()
income_max = monthly_income.max()
income_std = monthly_income.std()
income_cv = (income_std / avg_monthly * 100) if avg_monthly > 0 else 0
reservoir_balance = smoothed_df["reservoir_balance"].iloc[-1]

# ---- Build charts ----

# 1. Portfolio Growth
fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=df.index, y=df["total_portfolio_value"], name="Total Portfolio", line=dict(width=2.5, color="#1f77b4")), secondary_y=False)
fig1.add_trace(go.Scatter(x=df.index, y=df["spy_value"], name="SPY Sleeve (Inheritance)", line=dict(width=2, color="#2ca02c")), secondary_y=False)
fig1.add_trace(go.Scatter(x=df.index, y=df["income_sleeve_value"], name="Income Sleeve", line=dict(width=2, color="#ff7f0e")), secondary_y=False)
fig1.add_trace(go.Scatter(x=df.index, y=df["cumulative_income"], name="Cumulative Income Paid", line=dict(width=2, color="#d62728", dash="dot")), secondary_y=True)
fig1.update_layout(height=500, hovermode="x unified", title="Portfolio Growth Over Time", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig1.update_yaxes(title_text="Portfolio Value ($)", secondary_y=False, tickformat="$,.0f")
fig1.update_yaxes(title_text="Cumulative Income ($)", secondary_y=True, tickformat="$,.0f")

# 2. Monthly Income: Raw vs Smoothed
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=monthly_income_raw.index, y=monthly_income_raw.values, marker_color="rgba(180,180,180,0.45)", name="Raw Dividends Received"))
fig2.add_trace(go.Bar(x=monthly_income.index, y=monthly_income.values, marker_color="#2ca02c", name="Your Monthly Paycheck (Smoothed)"))
fig2.update_layout(height=420, yaxis_tickformat="$,.0f", barmode="overlay", hovermode="x unified", title="Monthly Retirement Income — Raw vs Smoothed Paycheck", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# 3. Reservoir Balance
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=smoothed_df.index, y=smoothed_df["reservoir_balance"], fill="tozeroy", line=dict(color="#1f77b4", width=2), name="Reservoir Balance"))
fig3.update_layout(height=280, yaxis_tickformat="$,.0f", hovermode="x unified", title="Income Reservoir Balance (Cash Buffer)")

# 4. Income Breakdown by ETF
monthly_by_etf = pd.DataFrame()
for t in income_tickers:
    if t in detail:
        monthly_by_etf[t] = detail[t]["monthly_series"]
monthly_by_etf = monthly_by_etf.fillna(0)
fig4 = go.Figure()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
for i, t in enumerate(monthly_by_etf.columns):
    fig4.add_trace(go.Scatter(x=monthly_by_etf.index, y=monthly_by_etf[t], name=t, stackgroup="one", line=dict(width=0.5, color=colors[i % len(colors)])))
fig4.update_layout(height=400, yaxis_tickformat="$,.0f", hovermode="x unified", title="Monthly Income Breakdown by ETF", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# 5. Drawdown
total_val = df["total_portfolio_value"]
running_max = total_val.cummax()
drawdown = (total_val - running_max) / running_max * 100
spy_val = df["spy_value"]
spy_max = spy_val.cummax()
spy_dd = (spy_val - spy_max) / spy_max * 100
fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=drawdown.index, y=drawdown.values, name="Total Portfolio", fill="tozeroy", line=dict(color="#1f77b4", width=1)))
fig5.add_trace(go.Scatter(x=spy_dd.index, y=spy_dd.values, name="SPY Sleeve", fill="tozeroy", line=dict(color="#2ca02c", width=1)))
fig5.update_layout(height=320, yaxis_title="Drawdown (%)", yaxis_tickformat=".1f", hovermode="x unified", title="Drawdown Analysis", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

# ---- ETF Detail Table ----
etf_table_rows = ""
for t in income_tickers:
    if t in detail:
        d = detail[t]
        etf_table_rows += f"<tr><td><b>{t}</b></td><td>{d['shares']:,.2f}</td><td>{fmt_money(d['total_dividends'])}</td><td>{fmt_money(d['avg_monthly'])}</td><td>{fmt_pct(d['yield_on_cost'])}</td></tr>"

# ---- Annual Summary Table ----
df_annual = df.copy()
df_annual["year"] = df_annual.index.year
annual_table_rows = ""
for year, grp in df_annual.groupby("year"):
    yr_raw = grp["daily_income"].sum()
    yr_smooth = smoothed_df.loc[smoothed_df.index.year == year, "smoothed_income"]
    yr_s_total = yr_smooth.sum() if len(yr_smooth) > 0 else 0
    yr_s_monthly = yr_smooth.mean() if len(yr_smooth) > 0 else 0
    annual_table_rows += f"<tr><td><b>{int(year)}</b></td><td>{fmt_money(grp['spy_value'].iloc[-1])}</td><td>{fmt_money(grp['income_sleeve_value'].iloc[-1])}</td><td>{fmt_money(grp['total_portfolio_value'].iloc[-1])}</td><td>{fmt_money(yr_raw)}</td><td>{fmt_money(yr_s_total)}</td><td>{fmt_money(yr_s_monthly)}</td><td>{fmt_money(grp['cumulative_income'].iloc[-1])}</td></tr>"

# ---- Trailing 12m projection ----
trailing_12 = monthly_income.iloc[-12:].mean() if len(monthly_income) >= 12 else monthly_income.mean()

# ---- Build HTML ----
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Income & Growth ETF Backtest Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0e1117; color: #fafafa; padding: 20px 40px; }}
  h1 {{ font-size: 2em; margin-bottom: 8px; }}
  h2 {{ font-size: 1.4em; margin: 30px 0 12px; color: #ccc; border-bottom: 1px solid #333; padding-bottom: 6px; }}
  h3 {{ font-size: 1.1em; margin: 20px 0 10px; color: #aaa; }}
  .subtitle {{ color: #888; margin-bottom: 24px; font-size: 0.95em; line-height: 1.5; }}
  .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 16px; }}
  .metric {{ background: #1a1d23; border-radius: 8px; padding: 16px; border: 1px solid #2a2d33; }}
  .metric .label {{ font-size: 0.8em; color: #888; text-transform: uppercase; letter-spacing: 0.5px; }}
  .metric .value {{ font-size: 1.5em; font-weight: 700; margin-top: 4px; color: #fff; }}
  .metric .delta {{ font-size: 0.85em; color: #2ca02c; margin-top: 2px; }}
  .chart {{ margin: 16px 0; }}
  .note {{ background: #1a2332; border-left: 3px solid #1f77b4; padding: 12px 16px; margin: 12px 0; border-radius: 4px; color: #aac; font-size: 0.9em; line-height: 1.6; }}
  table {{ width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 0.9em; }}
  th {{ background: #1a1d23; color: #888; text-align: left; padding: 10px 12px; border-bottom: 2px solid #333; font-weight: 600; text-transform: uppercase; font-size: 0.8em; letter-spacing: 0.5px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #222; }}
  tr:hover td {{ background: #1a1d23; }}
  .divider {{ border-top: 1px solid #333; margin: 24px 0; }}
  .summary-table {{ max-width: 700px; }}
  .summary-table td:first-child {{ color: #888; font-weight: 600; }}
  .summary-table td:last-child {{ color: #fff; font-weight: 500; }}
</style>
</head>
<body>

<h1>Income & Growth ETF Backtest Dashboard</h1>
<p class="subtitle">
  <b>Strategy</b>: $3M starting capital &mdash; <b>50% SPY</b> (dividends reinvested for your daughter's inheritance)
  and <b>50% High-Yield Income ETFs</b> (dividends paid out as monthly retirement income).<br>
  Income ETFs: <b>SCHD</b> &bull; <b>VYM</b> &bull; <b>HDV</b> &bull; <b>JEPI</b> &bull; <b>QYLD</b> &bull; <b>SDIV</b>
  &nbsp;&nbsp;|&nbsp;&nbsp; <span style="color:#ff7f0e">Demo data shown (yfinance unavailable) &mdash; run locally for real data</span>
</p>

<h2>Key Portfolio Metrics</h2>
<div class="metrics">
  <div class="metric"><div class="label">Total Portfolio Value</div><div class="value">{fmt_money(final_total + total_income)}</div></div>
  <div class="metric"><div class="label">SPY Sleeve (Inheritance)</div><div class="value">{fmt_money(final_spy)}</div><div class="delta">{fmt_pct(spy_growth)} total</div></div>
  <div class="metric"><div class="label">Income Sleeve Value</div><div class="value">{fmt_money(final_income_sleeve)}</div></div>
  <div class="metric"><div class="label">Total Income Collected</div><div class="value">{fmt_money(total_income)}</div></div>
</div>
<div class="metrics">
  <div class="metric"><div class="label">Steady Monthly Paycheck</div><div class="value">{fmt_money(avg_monthly)}</div></div>
  <div class="metric"><div class="label">Latest Month Paycheck</div><div class="value">{fmt_money(latest_monthly)}</div></div>
  <div class="metric"><div class="label">SPY CAGR (w/ reinvested divs)</div><div class="value">{fmt_pct(spy_cagr)}</div></div>
  <div class="metric"><div class="label">Total Return (incl. income)</div><div class="value">{fmt_pct(total_return)}</div></div>
</div>

<h3>Income Stability (Smoothed Paycheck)</h3>
<div class="metrics">
  <div class="metric"><div class="label">Lowest Monthly Paycheck</div><div class="value">{fmt_money(income_min)}</div></div>
  <div class="metric"><div class="label">Highest Monthly Paycheck</div><div class="value">{fmt_money(income_max)}</div></div>
  <div class="metric"><div class="label">Income Variability (CV)</div><div class="value">{fmt_pct(income_cv)}</div></div>
  <div class="metric"><div class="label">Reservoir Buffer Balance</div><div class="value">{fmt_money(reservoir_balance)}</div></div>
</div>

<div class="divider"></div>

<h3>SPY Inheritance Detail</h3>
<div class="metrics">
  <div class="metric"><div class="label">SPY Shares (Initial)</div><div class="value">{results['spy_shares_initial']:,.2f}</div></div>
  <div class="metric"><div class="label">SPY Shares (Final, DRIP)</div><div class="value">{results['spy_shares_final']:,.2f}</div></div>
  <div class="metric"><div class="label">Shares Gained via DRIP</div><div class="value">{results['spy_shares_final'] - results['spy_shares_initial']:,.2f}</div></div>
  <div class="metric"><div class="label">SPY Divs Reinvested</div><div class="value">{fmt_money(results['spy_divs_reinvested'])}</div></div>
</div>

<h2>Portfolio Growth Over Time</h2>
<div class="chart" id="chart1"></div>

<h2>Monthly Retirement Income &mdash; Raw vs Smoothed Paycheck</h2>
<div class="note">
  <b>The problem</b>: SCHD, VYM, HDV, and SDIV pay dividends <b>quarterly</b> (Mar/Jun/Sep/Dec),
  creating huge spikes and near-zero income in between.<br>
  <b>The fix</b>: All dividends flow into a <b>reservoir</b> (cash buffer). You withdraw a <b>steady
  monthly paycheck</b> recalibrated annually. Grey bars = raw spiky dividends. Green bars = your actual smooth paycheck.
</div>
<div class="chart" id="chart2"></div>

<h2>Income Reservoir Balance</h2>
<p style="color:#888;font-size:0.9em;margin-bottom:8px;">Cash buffer &mdash; dividends flow in, steady paychecks flow out. Positive balance = safety cushion.</p>
<div class="chart" id="chart3"></div>

<h2>Monthly Income Breakdown by ETF</h2>
<div class="chart" id="chart4"></div>

<h2>Income ETF Dividend Detail</h2>
<table>
  <tr><th>ETF</th><th>Shares Held</th><th>Total Dividends</th><th>Avg Monthly Income</th><th>Yield on Cost</th></tr>
  {etf_table_rows}
</table>

<h2>Annual Summary</h2>
<table>
  <tr><th>Year</th><th>SPY Sleeve</th><th>Income Sleeve</th><th>Total Portfolio</th><th>Raw Annual Income</th><th>Smoothed Annual Income</th><th>Monthly Paycheck</th><th>Cumulative Income</th></tr>
  {annual_table_rows}
</table>

<h2>Drawdown Analysis</h2>
<div class="chart" id="chart5"></div>
<div class="metrics" style="max-width:500px;">
  <div class="metric"><div class="label">Max Portfolio Drawdown</div><div class="value">{fmt_pct(drawdown.min())}</div></div>
  <div class="metric"><div class="label">Max Drawdown Date</div><div class="value">{drawdown.idxmin().strftime('%Y-%m-%d')}</div></div>
</div>

<h2>Forward Income Projection (trailing 12-month avg)</h2>
<div class="metrics" style="max-width:700px;">
  <div class="metric"><div class="label">Projected Monthly Income</div><div class="value">{fmt_money(trailing_12)}</div></div>
  <div class="metric"><div class="label">Projected Annual Income</div><div class="value">{fmt_money(trailing_12 * 12)}</div></div>
  <div class="metric"><div class="label">Yield on Original Capital</div><div class="value">{fmt_pct(trailing_12 * 12 / (STARTING_CAPITAL * INCOME_ALLOCATION) * 100)}</div></div>
</div>

<div class="divider"></div>
<h2>Strategy Summary</h2>
<table class="summary-table">
  <tr><td>Starting Capital</td><td>{fmt_money(STARTING_CAPITAL)}</td></tr>
  <tr><td>Backtest Period</td><td>{results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')} ({years:.1f} years)</td></tr>
  <tr><td>SPY Sleeve (Daughter's Inheritance)</td><td>{fmt_money(final_spy)}</td></tr>
  <tr><td>SPY CAGR</td><td>{fmt_pct(spy_cagr)}</td></tr>
  <tr><td>Income Sleeve Current Value</td><td>{fmt_money(final_income_sleeve)}</td></tr>
  <tr><td>Total Retirement Income Collected</td><td>{fmt_money(total_income)}</td></tr>
  <tr><td>Avg Smoothed Monthly Paycheck</td><td>{fmt_money(avg_monthly)}</td></tr>
  <tr><td>Lowest Monthly Paycheck</td><td>{fmt_money(income_min)}</td></tr>
  <tr><td>Income Variability (CV)</td><td>{fmt_pct(income_cv)}</td></tr>
  <tr><td>Reservoir Buffer</td><td>{fmt_money(reservoir_balance)}</td></tr>
  <tr><td>Total Wealth Created</td><td>{fmt_money(final_total + total_income)}</td></tr>
  <tr><td>Total Return</td><td>{fmt_pct(total_return)}</td></tr>
</table>

<script>
var dark = {{
  paper_bgcolor: '#0e1117',
  plot_bgcolor: '#0e1117',
  font: {{ color: '#ccc' }},
  xaxis: {{ gridcolor: '#222' }},
  yaxis: {{ gridcolor: '#222' }}
}};
var cfg = {{ responsive: true, displayModeBar: false }};
</script>
<script>
Plotly.newPlot('chart1', {fig1.to_json()}.data, Object.assign({{}}, {fig1.to_json()}.layout, dark), cfg);
Plotly.newPlot('chart2', {fig2.to_json()}.data, Object.assign({{}}, {fig2.to_json()}.layout, dark), cfg);
Plotly.newPlot('chart3', {fig3.to_json()}.data, Object.assign({{}}, {fig3.to_json()}.layout, dark), cfg);
Plotly.newPlot('chart4', {fig4.to_json()}.data, Object.assign({{}}, {fig4.to_json()}.layout, dark), cfg);
Plotly.newPlot('chart5', {fig5.to_json()}.data, Object.assign({{}}, {fig5.to_json()}.layout, dark), cfg);
</script>

</body>
</html>
"""

with open("/home/user/incomegenerator/dashboard_preview.html", "w") as f:
    f.write(html)

print("Dashboard preview generated: dashboard_preview.html")
print(f"\nKey numbers:")
print(f"  Total Portfolio Value: {fmt_money(final_total + total_income)}")
print(f"  SPY Inheritance:       {fmt_money(final_spy)} ({fmt_pct(spy_cagr)} CAGR)")
print(f"  Monthly Paycheck:      {fmt_money(avg_monthly)} (smoothed)")
print(f"  Income Min/Max:        {fmt_money(income_min)} / {fmt_money(income_max)}")
print(f"  Income CV:             {fmt_pct(income_cv)}")
print(f"  Reservoir Buffer:      {fmt_money(reservoir_balance)}")
print(f"  Total Income Earned:   {fmt_money(total_income)}")
