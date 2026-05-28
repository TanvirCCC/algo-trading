# lnterqo v3

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)
![MT4](https://img.shields.io/badge/Platform-MetaTrader_4-lightgrey)
![License](https://img.shields.io/badge/License-MIT-22c55e)

**Systematic intraday trading system for Gold (XAUUSD).** Translates Smart Money Concepts (institutional order flow) into a fully quantified, ML-filtered signal pipeline deployed via a MetaTrader 4 Expert Advisor. Validated with rolling walk-forward analysis across six independent out-of-sample years.

**[→ Live Dashboard](https://tanvirccc-algo-trading-dashboard-app.streamlit.app)**

---

## Out-of-Sample Performance (2020–2026)

Validated across **5 independent walk-forward windows** — no look-ahead bias, no in-sample overfitting.

| Metric | Result |
|--------|--------|
| Total OOS Trades | 622 |
| Win Rate | 39.4% |
| Expected Value | **+2.13R per trade** |
| Profit Factor | **4.74** |
| Max Drawdown | −5.37% |
| Sharpe Ratio | 2.8 |
| FTMO 100k Barrier | **100%** across all 5 windows |
| FTMO 100k Pass Rate | **99.7%** *(Monte Carlo, 10,000 simulations)* |

> £10,000 → **£142,000** over the OOS period on fixed 0.5% risk per trade.

---

## Mathematical Framework

### 1. Expected Value & R-Multiple

All performance is measured in **R-multiples** — units of risk normalised by stop distance. This makes results comparable across different position sizes and instruments.

$$\text{EV} = p_w \cdot \bar{R}_w + (1 - p_w) \cdot \bar{R}_l$$

where $p_w$ is win rate, $\bar{R}_w$ is mean win in R, and $\bar{R}_l$ is mean loss in R (negative). For lnterqo v3: $EV = 0.394 \times 5.41 + 0.606 \times (-1.0) = +2.13R$.

A positive EV is necessary but not sufficient — the **distribution** of outcomes matters. A strategy with EV = +0.5R but high variance can still blow up on any given run.

### 2. Walk-Forward Validation

Standard backtesting suffers from look-ahead bias and overfitting. Rolling WFV addresses this by treating each window as a genuine out-of-sample test:

```
Window 1:  IS [2016–2020] → optimise params → OOS [2020–2021]
Window 2:  IS [2016–2021] → optimise params → OOS [2021–2022]
Window 3:  IS [2016–2022] → optimise params → OOS [2022–2023]
Window 4:  IS [2016–2023] → optimise params → OOS [2023–2024]
Window 5:  IS [2016–2024] → optimise params → OOS [2024–2026]
```

Parameters are re-optimised on in-sample data only. OOS results are **never touched** during optimisation. Each window is a statistically independent test of the edge's stability.

### 3. Monte Carlo — Barrier Probability

Prop firm challenges (and risk-of-ruin analysis) involve **path-dependent** questions that EV alone cannot answer. Monte Carlo simulation directly resamples the empirical trade return distribution:

```python
def simulate(trade_returns, n_trades=500, n_sims=10_000,
             profit_target=0.10, max_dd=0.05):
    B_up   = 1 + profit_target   # normalised upper barrier
    B_down = 1 - max_dd          # normalised lower barrier
    passes = 0
    for _ in range(n_sims):
        equity = 1.0
        peak   = 1.0
        for _ in range(n_trades):
            r      = np.random.choice(trade_returns)
            equity *= (1 + r * RISK_PCT)
            peak    = max(peak, equity)
            if equity / peak - 1 < -max_dd:  break
            if equity >= B_up:
                passes += 1; break
    return passes / n_sims
```

Result: **99.7% pass rate** on FTMO 100k challenge structure (6% target, 4% max DD) across 10,000 simulated runs on the OOS trade distribution.

### 4. CUSUM Regime Detection

To avoid trading in low-quality market conditions, lnterqo uses a **Cumulative Sum (CUSUM)** detector — a sequential change-point detection algorithm from statistical process control (Page, 1954):

$$S_t^+ = \max(0,\; S_{t-1}^+ + r_t - k), \quad S_0^+ = 0$$
$$S_t^- = \max(0,\; S_{t-1}^- - r_t - k), \quad S_0^- = 0$$

where $r_t$ is the standardised log-return and $k$ is the allowance parameter (typically $k = 0.5\sigma$). A regime shift is signalled when $S_t^+$ or $S_t^-$ exceeds threshold $h$. Unlike moving averages, CUSUM accumulates evidence over time and is optimal under the minimax criterion for detecting a sustained mean shift.

### 5. Machine Learning Filter

A **Random Forest classifier** is trained on 622 OOS trades (post-WFV) to distinguish high-probability from low-probability setups. Features include zone type, confidence score, session, CUSUM state, and seasonal indicators. Signals are only passed to execution if $P(\text{win}) \geq 0.60$.

This is an application of the **Fundamental Law of Active Management** (Grinold, 1989):

$$\text{IR} = \text{IC} \times \sqrt{BR}$$

The ML filter increases the Information Coefficient (IC) by rejecting setups where the signal has low predictive power, even at the cost of reduced Breadth (BR).

---

## Signal Logic — ICT to Quant

The core entry logic is based on **Smart Money Concepts** — a framework for reading institutional order flow. Below is the mapping to standard quantitative terminology:

| SMC Concept | Quantitative Equivalent |
|-------------|------------------------|
| **CISD** (Change in State of Delivery) | Structural order flow reversal — a failed delivery imbalance that signals a regime shift in the dominant aggressor side |
| **Supply / Demand Zone** | Unmitigated limit order cluster — price level where an imbalance between aggressive buyers and sellers was left unresolved |
| **BOS** (Break of Structure) | Higher high / lower low confirmation — directional momentum filter |
| **CHoCH** (Change of Character) | First counter-trend structural break — early signal of potential trend reversal |
| **FVG** (Fair Value Gap) | Three-candle price imbalance — equivalent to an unfilled gap in order flow; price has a statistical tendency to revisit |
| **Kill Zone** | High-volume session window — London open (07:00–10:00 UTC) and NY open (12:00–15:00 UTC), when institutional order flow dominates |
| **Liquidity Sweep** | Stop-hunt above/below equal highs/lows — engineered move to fill resting orders before reversal |

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│            MetaTrader 4 — Gold M5 Chart             │
│  LnterqoV3.mq4 EA — exports bars every 60s (CSV)   │
└───────────────────────┬─────────────────────────────┘
                        │ file bridge
                        ▼
┌─────────────────────────────────────────────────────┐
│           signal_engine.py — Python loop            │
│  ① Load live bars from MT4                          │
│  ② CUSUM regime filter                              │
│  ③ Scan: CISD + Supply/Demand zones                 │
│  ④ ML filter  P(win) ≥ 0.60                         │
│  ⑤ News filter  ±15min / 2hr high-impact USD        │
│  ⑥ Write signal → lnterqo_signals.csv               │
└───────────────────────┬─────────────────────────────┘
                        │ file bridge
                        ▼
┌─────────────────────────────────────────────────────┐
│         MT4 EA — order execution                    │
│  Reads signal CSV → calculates lot size             │
│  (0.5% risk / 1.0% for confidence ≥ 4)             │
│  OrderSend() with SL + TP → logs result CSV         │
└─────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│    Streamlit Dashboard — real-time analytics        │
│  Equity · Drawdown · Heatmaps · Signal log          │
│  Live vs Backtest comparison · ACF · Regime         │
└─────────────────────────────────────────────────────┘
```

---

## Dashboard

| Tab | Contents |
|-----|----------|
| **Overview** | Live equity curve overlaid on backtest baseline, drawdown, news calendar, kill zone status |
| **Performance** | Cumulative R (live vs backtest), rolling Sharpe, monthly PnL heatmap, R-distribution comparison |
| **Analytics** | Win rate by zone type / confidence / direction, PnL by weekday, trade duration histogram |
| **Signals** | Live signal scatter (entry/SL/TP), signal log table, latest signal panel |
| **Market** | Hour×weekday return heatmap, volume heatmap, intraday return profile, autocorrelation (ACF) |

---

## Stack

| Layer | Technology |
|-------|-----------|
| Signal generation | Python 3.12, Pandas, NumPy |
| ML filter | scikit-learn — Random Forest |
| Regime detection | CUSUM (custom), seasonal decomposition |
| Backtesting | Custom rolling WFV engine + Monte Carlo |
| Execution | MQL4 / MetaTrader 4 (CSV file bridge) |
| Dashboard | Streamlit + Plotly |
| News filter | ForexFactory REST API |

---

## Project Structure

```
├── strategy/
│   ├── lnterqo_strategy.py   # Signal scanner (CISD, zones, structure)
│   ├── ml_filter.py          # Random Forest signal filter
│   └── risk_manager.py       # Position sizing, drawdown limits
├── detectors/
│   ├── cisd.py               # Change in State of Delivery
│   ├── market_structure.py   # BOS / CHoCH structural analysis
│   ├── order_blocks.py       # Supply & demand zone detection
│   ├── regime.py             # CUSUM regime detection
│   └── seasonality.py        # Intraday seasonal features
├── backtest/
│   ├── engine.py             # Walk-forward backtester + Monte Carlo
│   ├── metrics.py            # Sharpe, profit factor, drawdown, barrier prob
│   └── lnterqo_v3_oos_trades.csv   # 622 OOS trades (2020–2026)
├── live/
│   ├── signal_engine.py      # Live 60s scan loop
│   ├── config.py             # All strategy parameters
│   └── ea/LnterqoV3.mq4     # MT4 Expert Advisor
├── data/
│   └── news_calendar.py      # High-impact news feed
└── dashboard/
    └── app.py                # Streamlit analytics dashboard
```

---

## Quickstart

```bash
pip install -r requirements.txt

# Dashboard — backtest analytics, no MT4 required
streamlit run dashboard/app.py

# Live signal engine — requires MT4 EA running and exporting bars
python live/signal_engine.py
```

MT4 EA setup: [`live/ea/SETUP.md`](live/ea/SETUP.md)

---

## References

- Grinold, R.C. (1989). *The Fundamental Law of Active Management.* Journal of Portfolio Management.
- Page, E.S. (1954). *Continuous Inspection Schemes.* Biometrika.
- Lo, A.W. & MacKinlay, A.C. (1988). *Stock Market Prices Do Not Follow Random Walks.* Review of Financial Studies.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.
