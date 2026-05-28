"""
Backtesting Engine
Simulates the ICT strategy on historical commodity data.

Logic:
  - Walk through intraday bars sequentially
  - Enter when a Signal is generated
  - Exit at TP or SL, whichever hits first
  - One position per asset at a time
  - Uses RiskManager for position sizing

Output:
  - Trade log (for the report)
  - Equity curve
  - Performance metrics
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from strategy.ict_strategy import Signal, scan_for_signals
from strategy.risk_manager import RiskManager


@dataclass
class Trade:
    signal: Signal
    size: float
    entry_price: float
    exit_price: float = 0.0
    exit_time: pd.Timestamp = None
    pnl: float = 0.0
    outcome: str = ""     # "win", "loss", "open"
    bars_held: int = 0

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry_price - self.signal.stop)
        if risk == 0:
            return 0.0
        return self.pnl / (risk * self.size)

    def to_dict(self) -> dict:
        s = self.signal
        return {
            "date": s.timestamp.strftime("%Y-%m-%d %H:%M"),
            "asset": s.asset,
            "direction": s.direction,
            "entry": round(self.entry_price, 4),
            "stop": round(s.stop, 4),
            "target": round(s.target, 4),
            "exit": round(self.exit_price, 4),
            "size": round(self.size, 4),
            "pnl_gbp": round(self.pnl, 2),
            "r_multiple": round(self.r_multiple, 2),
            "outcome": self.outcome,
            "bars_held": self.bars_held,
            "risk_reward": s.risk_reward,
            "confidence": s.confidence,
            "zone_type": s.zone_type,
            "rationale": s.report_rationale,
        }


def run_backtest(
    asset: str,
    df_daily: pd.DataFrame,
    df_intraday: pd.DataFrame,
    initial_equity: float = 10_000.0,
    risk_pct: float = 0.02,
    min_rr: float = 2.0,
    signal_filter=None,
    use_kill_zones: bool = True,
    zone_lookback: int = 80,
    zone_tolerance_atr: float = 0.0,
    use_news_filter: bool = True,
    use_seasonal_filter: bool = False,
    force_neutral_bias: bool = False,
    df_enriched_for_filter: pd.DataFrame = None,
) -> tuple[list[Trade], pd.Series]:
    """
    Run a full backtest. Returns (trade_list, equity_curve).
    Pass a fitted SignalFilter to enable ML-gated entries.
    Set use_kill_zones=False for daily-bar backtests where session timing doesn't apply.
    zone_lookback: bars to search back for active FVG/OB zones (use 200+ for daily).
    zone_tolerance_atr: extend zone boundaries by N × ATR (use 0.5 for daily bars).
    """
    rm = RiskManager(equity=initial_equity, risk_pct=risk_pct)
    signals = scan_for_signals(
        asset, df_daily, df_intraday,
        min_rr=min_rr,
        use_kill_zones=use_kill_zones,
        zone_lookback=zone_lookback,
        zone_tolerance_atr=zone_tolerance_atr,
        bar_range_check=not use_kill_zones,
        use_news_filter=use_news_filter,
        use_seasonal_filter=use_seasonal_filter,
        force_neutral_bias=force_neutral_bias,
    )

    if not signals:
        print(f"No signals found for {asset}")
        return [], pd.Series([initial_equity], name="equity")

    trades = []
    equity_curve = [initial_equity]
    signal_map = {s.timestamp: s for s in signals}
    open_trade: Trade | None = None
    current_date = None

    for i in range(len(df_intraday)):
        row = df_intraday.iloc[i]
        ts = df_intraday.index[i]

        # Reset daily risk limits at the start of each new calendar day
        bar_date = ts.date()
        if bar_date != current_date:
            rm.reset_day()
            current_date = bar_date

        # Check for exit on open trade
        if open_trade is not None:
            sig = open_trade.signal
            high = row["high"]
            low = row["low"]
            bearish_bar = row["close"] < row["open"]
            open_trade.bars_held += 1
            outcome = None
            exit_p = 0.0

            if sig.direction == "long":
                sl_hit = low <= sig.stop
                tp_hit = high >= sig.target
                if sl_hit and tp_hit:
                    # Both levels on same bar — use candle direction to infer order
                    outcome, exit_p = ("loss", sig.stop) if bearish_bar else ("win", sig.target)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target

            elif sig.direction == "short":
                sl_hit = high >= sig.stop
                tp_hit = low <= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("win", sig.target) if bearish_bar else ("loss", sig.stop)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target

            if outcome is not None:
                open_trade.exit_price = exit_p
                open_trade.exit_time = ts
                open_trade.outcome = outcome
                if sig.direction == "long":
                    open_trade.pnl = (exit_p - open_trade.entry_price) * open_trade.size
                else:
                    open_trade.pnl = (open_trade.entry_price - exit_p) * open_trade.size
                rm.record_trade(open_trade.pnl)
                trades.append(open_trade)
                equity_curve.append(rm.equity)
                open_trade = None

        # Check for new signal entry
        if open_trade is None and ts in signal_map:
            ok, reason = rm.can_trade()
            if ok:
                sig = signal_map[ts]
                # ML filter gate — RF must accept the signal (P(win) ≥ threshold)
                _df_for_filter = df_enriched_for_filter if df_enriched_for_filter is not None else df_intraday
                if signal_filter is not None and not signal_filter.accept(sig, _df_for_filter):
                    continue
                size = rm.position_size(sig.entry, sig.stop, sig.confidence)
                if size > 0:
                    open_trade = Trade(signal=sig, size=size, entry_price=sig.entry)

    # Close any open trade at last price
    if open_trade is not None:
        last_price = df_intraday.iloc[-1]["close"]
        open_trade.exit_price = last_price
        open_trade.exit_time = df_intraday.index[-1]
        open_trade.outcome = "open"
        if open_trade.signal.direction == "long":
            open_trade.pnl = (last_price - open_trade.entry_price) * open_trade.size
        else:
            open_trade.pnl = (open_trade.entry_price - last_price) * open_trade.size
        trades.append(open_trade)
        equity_curve.append(rm.equity + open_trade.pnl)

    eq_series = pd.Series(equity_curve, name="equity")
    return trades, eq_series
