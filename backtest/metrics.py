"""
Performance Metrics
Trading Game report requires: win rate, R:R, profit factor, Sharpe ratio, max drawdown.
"""

import pandas as pd
import numpy as np
from backtest.engine import Trade


def _max_consecutive(outcomes: list[int], target: int) -> int:
    max_c = count = 0
    for x in outcomes:
        if x == target:
            count += 1
            max_c = max(max_c, count)
        else:
            count = 0
    return max_c


def compute_metrics(trades: list[Trade], equity_curve: pd.Series) -> dict:
    if not trades:
        return {}

    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if not closed:
        return {"message": "No closed trades"}

    wins = [t for t in closed if t.outcome == "win"]
    losses = [t for t in closed if t.outcome == "loss"]

    win_rate = len(wins) / len(closed)
    avg_win = np.mean([t.pnl for t in wins]) if wins else 0
    avg_loss = abs(np.mean([t.pnl for t in losses])) if losses else 0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    avg_rr = np.mean([t.signal.risk_reward for t in closed])
    total_pnl = sum(t.pnl for t in closed)

    r_mults = np.array([t.r_multiple for t in closed])
    expectancy_r = float(np.mean(r_mults))

    # Drawdown
    eq = equity_curve.values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_drawdown = dd.min()

    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    total_return_pct = (final - initial) / initial * 100

    # Date range from trades for CAGR / trades-per-year
    entry_dates = [t.signal.timestamp for t in closed]
    exit_dates = [t.exit_time for t in closed if t.exit_time is not None]
    if entry_dates and exit_dates:
        years = max((max(exit_dates) - min(entry_dates)).days / 365.25, 1 / 252)
    else:
        years = 1.0

    cagr = ((final / initial) ** (1 / years) - 1) * 100 if initial > 0 else 0.0
    calmar = cagr / abs(max_drawdown * 100) if max_drawdown != 0 else np.inf

    # Sharpe / Sortino (annualised from equity curve)
    returns = equity_curve.pct_change().dropna()
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    neg_returns = returns[returns < 0]
    downside_std = neg_returns.std() if len(neg_returns) > 1 else returns.std()
    sortino = (returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

    # Consecutive streaks
    seq = [1 if t.outcome == "win" else 0 for t in closed]
    max_consec_wins = _max_consecutive(seq, 1)
    max_consec_losses = _max_consecutive(seq, 0)

    trades_per_year = len(closed) / years
    avg_bars_held = float(np.mean([t.bars_held for t in closed]))

    return {
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": f"{win_rate*100:.1f}%",
        "expectancy_r": round(expectancy_r, 3),
        "avg_win_gbp": round(avg_win, 2),
        "avg_loss_gbp": round(avg_loss, 2),
        "avg_risk_reward": round(avg_rr, 2),
        "profit_factor": round(profit_factor, 2),
        "total_pnl_gbp": round(total_pnl, 2),
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 2) if not np.isinf(calmar) else "∞",
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "trades_per_year": round(trades_per_year, 1),
        "avg_bars_held": round(avg_bars_held, 1),
        "initial_equity": round(initial, 2),
        "final_equity": round(final, 2),
    }


def print_metrics(metrics: dict, asset: str = ""):
    header = f"  BACKTEST RESULTS — {asset}  " if asset else "  BACKTEST RESULTS  "
    print("=" * 50)
    print(header.center(50))
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<25} {v}")
    print("=" * 50)


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    return pd.DataFrame([t.to_dict() for t in trades])


def report_summary(trades: list[Trade], equity_curve: pd.Series, asset: str) -> str:
    """
    Generate the back-test paragraph for the Trading Game report.
    Uses CDT03-aligned language per the bridge document.
    """
    m = compute_metrics(trades, equity_curve)
    if not m or "message" in m:
        return "Insufficient data for back-test summary."

    return (
        f"We back-tested our false-breakout and gap-fill entry rules on {asset} "
        f"across the historical data period using our Python implementation. "
        f"The strategy identified {m['total_trades']} trade setups. "
        f"The strategy produced a win rate of {m['win_rate']} and an average "
        f"risk/reward ratio of {m['avg_risk_reward']}:1 over the test period. "
        f"Profit factor was {m['profit_factor']} (gross profit / gross loss). "
        f"The Sharpe ratio was {m['sharpe_ratio']} (annualised). "
        f"Maximum drawdown was {m['max_drawdown_pct']}% of peak equity. "
        f"Total return over the test period: {m['total_return_pct']}%. "
        f"All entries were restricted to London and New York open sessions "
        f"and required three-signal confluence: Fibonacci/discount zone, "
        f"open gap or demand zone, and RSI/CCI confirmation."
    )
