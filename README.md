# Income & Growth ETF Backtest Dashboard

Backtest a $3,000,000 retirement income strategy:

- **50% SPY** — Growth engine for inheritance (dividends reinvested/compounded)
- **50% High-Yield Income ETFs** — Monthly retirement income (dividends paid out as cash)

Income ETFs: SCHD, VYM, HDV, JEPI, QYLD, SDIV

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Features

- Full backtest since earliest common inception date
- Portfolio growth chart with SPY vs Income sleeve
- Monthly income bar chart with 12-month rolling average
- Income breakdown by ETF (stacked area)
- Per-ETF dividend detail table
- Annual summary table
- Drawdown analysis
- Forward income projection based on trailing 12-month average
