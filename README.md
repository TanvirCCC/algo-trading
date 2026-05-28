# lnterqo v3 — Algorithmic Gold Trading System

> An end-to-end quantitative trading system for Gold (XAUUSD) built during my MSc Mathematical Trading & Finance dissertation. Combines Smart Money Concepts (SMC) with machine learning signal filtering, deployed as an automated MT4 Expert Advisor with a real-time Streamlit dashboard.

---

## Live Dashboard

**[→ View Live Dashboard](https://tanvirccc-algo-trading-dashboard-app.streamlit.app)**

Strategy performance, equity curve, heatmaps, signal log, and backtest analytics — all in one place.

---

## Strategy Overview

lnterqo v3 is a systematic intraday strategy for Gold, trading the **London (07:00–10:00 UTC)** and **New York (12:00–15:00 UTC)** kill zones. It identifies high-probability reversal setups using institutional order flow concepts:

| Component | Description |
|-----------|-------------|
| **CISD** | Change in State of Delivery — detects institutional order flow shifts |
| **Supply & Demand Zones** | Identifies unmitigated institutional order blocks |
| **Market Structure** | BOS / CHoCH detection for trend alignment |
| **ML Filter** | Random Forest classifier trained on 622 OOS trades to reject low-probability setups |
| **News Filter** | Blocks trading 15 min before and 2 hrs after high-impact USD news (ForexFactory API) |
| **CUSUM Regime** | Detects trending vs ranging regimes using cumulative sum statistics |

---

## Backtest Results (v3 — Rolling Walk-Forward Validation)

Validated across **5 independent out-of-sample windows** from 2020–2026 (zero look-ahead bias):

| Metric | Value |
|--------|-------|
| Total OOS Trades | 622 |
| Win Rate | 39.4% |
| EV/R | **+2.129R** |
| Profit Factor | **4.74** |
| Max Drawdown | −5.37% |
| Sharpe Ratio | 2.8 |
| FTMO 100k Pass Rate | **99.7%** (Monte Carlo, 10,000 simulations) |
| Barrier Probability | **100%** across all 5 windows |

> Starting equity £10,000 → **£142,000** over OOS period (2020–2026)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MT4 (CMC Markets)                     │
│  GOLD M5 chart + LnterqoV3.mq4 Expert Advisor           │
│  Exports: gold_5m_live.csv, gold_d1_live.csv every 60s  │
└─────────────────────┬───────────────────────────────────┘
                      │ CSV bridge (file system)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              signal_engine.py (Python)                   │
│  1. Loads live bar data from MT4                         │
│  2. Runs lnterqo v3 scanner (CISD + zones)               │
│  3. Applies ML filter (Random Forest, threshold=0.60)    │
│  4. Writes new signals → lnterqo_signals.csv             │
└─────────────────────┬───────────────────────────────────┘
                      │ CSV bridge (file system)
                      ▼
┌─────────────────────────────────────────────────────────┐
│              MT4 EA reads signal CSV                     │
│  Calculates lot size (0.5% risk / 1.0% high confidence)  │
│  Places OrderSend() with SL and TP                       │
│  Writes trade results → lnterqo_trades.csv               │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│         Streamlit Dashboard (dashboard/app.py)           │
│  Real-time equity curve · Drawdown · Heatmaps            │
│  Signal log · Analytics · Live vs Backtest comparison    │
└─────────────────────────────────────────────────────────┘
```

---

## Dashboard Features

- **Overview** — Live equity curve overlaid on backtest baseline, drawdown chart, news calendar
- **Performance** — Cumulative R-multiple (live vs backtest), rolling Sharpe, monthly PnL heatmap
- **Analytics** — Win rate by zone type / confidence / direction, trade duration, PnL by weekday
- **Signals** — Real-time signal scatter plot, latest signal entry/SL/TP levels, signal log table
- **Market** — Weekday × hour return heatmap, volume heatmap, intraday return profile, autocorrelation (ACF)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Strategy | Python 3.12, Pandas, NumPy |
| ML Filter | scikit-learn (Random Forest) |
| Backtesting | Custom rolling WFV engine with Monte Carlo simulation |
| Execution | MQL4 / MetaTrader 4 Expert Advisor |
| Dashboard | Streamlit + Plotly |
| Data | HistData (tick), Dukascopy, yFinance, ForexFactory API |
| Optimisation | Genetic Algorithm (v4) — pop=30, gen=20, tournament selection |

---

## Strategy Versions

| Version | Risk Model | Optimisation | OOS EV/R | Max DD |
|---------|-----------|--------------|----------|--------|
| v1 | Fixed tiers (0.5% / 1%) | Grid search | +1.8R | −6.2% |
| v2 | EG (exponential gradient) | Grid search | +2.122R | −5.19% |
| **v3** ✓ | Fixed tiers (0.5% / 1%) | Grid search + ML | **+2.129R** | **−5.37%** |
| v4 | Fixed tiers | Genetic algorithm | +9.5R | −13.1% |

v3 selected as primary: best risk-adjusted performance, lowest drawdown, most robust across windows.

---

## Project Structure

```
algo-trading/
├── strategy/
│   ├── lnterqo_strategy.py   # Core signal scanner (CISD, zones, structure)
│   ├── ml_filter.py          # Random Forest signal filter
│   └── risk_manager.py       # Position sizing & daily risk limits
├── detectors/
│   ├── cisd.py               # Change in State of Delivery
│   ├── market_structure.py   # BOS / CHoCH detection
│   ├── order_blocks.py       # Supply & demand zones
│   └── regime.py             # CUSUM regime detection
├── backtest/
│   ├── engine.py             # Backtest engine with Monte Carlo
│   └── lnterqo_v3_oos_trades.csv  # 622 OOS trades (2020–2026)
├── live/
│   ├── signal_engine.py      # Live trading loop (60s scan)
│   ├── ea/LnterqoV3.mq4      # MT4 Expert Advisor
│   └── config.py             # All parameters
├── data/
│   └── news_calendar.py      # ForexFactory high-impact news feed
├── dashboard/
│   └── app.py                # Streamlit dashboard
└── backtest_lnterqo_v3_wfv.py   # Full rolling WFV backtest script
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard (backtest analytics — no MT4 required)
streamlit run dashboard/app.py

# Run live signal engine (requires MT4 EA running)
python live/signal_engine.py
```

MT4 EA setup: see [`live/ea/SETUP.md`](live/ea/SETUP.md)

---

## Key Concepts

- **Rolling Anchored WFV**: 5 windows with expanding in-sample (50/60/70/80/90%) and fixed 10% OOS — tests whether edge is stable across time
- **CISD**: Institutional concept — detects when a delivery imbalance flips direction, signalling a high-probability reversal zone
- **R-Multiple**: All performance measured in units of risk (1R = distance entry to stop) — size-independent and comparable across strategies
- **Barrier Probability**: Monte Carlo test of whether the equity curve is statistically unlikely to be random (p < 0.05)

---

*MSc Mathematical Trading & Finance — Dissertation Project*
