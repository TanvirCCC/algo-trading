# lnterqo v3

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)
![MT4](https://img.shields.io/badge/MetaTrader-4-lightgrey?logo=data:image/png;base64,)
![License](https://img.shields.io/badge/License-MIT-green)

**Automated Gold (XAUUSD) trading system** built in Python. Combines Smart Money Concepts with a machine learning signal filter and deploys via a MetaTrader 4 Expert Advisor. Includes a real-time Streamlit analytics dashboard.

**[→ Live Dashboard](https://tanvirccc-algo-trading-dashboard-app.streamlit.app)**

---

## Performance (Out-of-Sample, 2020–2026)

| Metric | Result |
|--------|--------|
| Trades | 622 |
| Win Rate | 39.4% |
| EV per Trade | **+2.13R** |
| Profit Factor | **4.74** |
| Max Drawdown | −5.37% |
| Sharpe Ratio | 2.8 |
| FTMO 100k Pass Rate | **99.7%** *(Monte Carlo, 10k sims)* |

> Validated across 5 independent walk-forward windows — no look-ahead bias.

---

## How It Works

```
MT4 (Gold M5 chart)
  └─ exports bar data every 60s (CSV)
       └─ signal_engine.py
            ├─ scans for CISD + zone setups
            ├─ applies ML filter (RF, threshold 0.60)
            └─ writes signal to CSV bridge
                 └─ MT4 EA reads signal
                      └─ places order with SL + TP
                           └─ results feed into dashboard
```

### Strategy Logic

| Component | Role |
|-----------|------|
| **CISD** | Detects institutional order flow shifts (Change in State of Delivery) |
| **Supply & Demand Zones** | Unmitigated institutional order blocks |
| **Market Structure** | BOS / CHoCH for trend alignment |
| **ML Filter** | Random Forest — rejects low-probability setups |
| **News Filter** | Blocks 15 min before / 2 hrs after high-impact USD news |
| **CUSUM Regime** | Filters out ranging markets |

Trading windows: **London 07:00–10:00 UTC** · **New York 12:00–15:00 UTC**

---

## Dashboard

Five tabs — all charts built with Plotly:

| Tab | Contents |
|-----|----------|
| **Overview** | Live equity curve vs backtest, drawdown, upcoming news |
| **Performance** | Cumulative R, rolling Sharpe, monthly PnL heatmap, R-distribution |
| **Analytics** | Win rate by zone / confidence / direction, trade duration, PnL by weekday |
| **Signals** | Live signal scatter, entry/SL/TP levels, signal log table |
| **Market** | Hour×weekday heatmaps, intraday return profile, autocorrelation (ACF) |

---

## Stack

```
Python 3.12 · Pandas · NumPy · scikit-learn
Streamlit · Plotly
MQL4 / MetaTrader 4
```

---

## Project Structure

```
├── strategy/
│   ├── lnterqo_strategy.py   # Signal scanner
│   ├── ml_filter.py          # Random Forest filter
│   └── risk_manager.py       # Position sizing & risk limits
├── detectors/
│   ├── cisd.py               # Change in State of Delivery
│   ├── market_structure.py   # BOS / CHoCH
│   ├── order_blocks.py       # Supply & demand zones
│   └── regime.py             # CUSUM regime filter
├── backtest/
│   ├── engine.py             # Backtest + Monte Carlo engine
│   └── lnterqo_v3_oos_trades.csv
├── live/
│   ├── signal_engine.py      # Live 60s scan loop
│   ├── config.py             # Parameters
│   └── ea/LnterqoV3.mq4     # MT4 Expert Advisor
├── data/
│   └── news_calendar.py      # ForexFactory news feed
└── dashboard/
    └── app.py                # Streamlit dashboard
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Dashboard only (no MT4 needed)
streamlit run dashboard/app.py

# Live signal engine (requires MT4 EA running)
python live/signal_engine.py
```

MT4 setup guide: [`live/ea/SETUP.md`](live/ea/SETUP.md)
