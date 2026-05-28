"""
Commodity Strategy Optimizer — Daily Trend-Following
======================================================
Runs the EMA/RSI daily trend strategy on Oil, Cocoa, Corn, Wheat.
Weekly bars = HTF bias. Daily bars = entry timeframe.

Usage:
    python3 optimize_commodities.py
    python3 optimize_commodities.py --assets Oil Cocoa
    python3 optimize_commodities.py --start 2016-01-01 --end 2023-12-31
"""

import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data.yfinance_fetcher import load_parquet
from strategy.daily_trend_strategy import scan_daily_signals
from backtest.metrics import compute_metrics
from backtest.monte_carlo import run_prop_firm_simulations
from strategy.risk_manager import RiskManager

INITIAL_EQUITY = 10_000.0
RISK_PCT       = 0.02
MC_N_SIMS      = 10_000
MIN_TRADES     = 15

ASSETS = ["Oil", "Cocoa", "Corn", "Wheat"]

# Variations: (label, min_rr, rsi_entry_long, rsi_entry_short)
VARIATIONS = [
    ("RR=2 / RSI=45",   2.0, 45.0, 55.0),
    ("RR=2 / RSI=50",   2.0, 50.0, 50.0),
    ("RR=1.5 / RSI=45", 1.5, 45.0, 55.0),
    ("RR=1.5 / RSI=50", 1.5, 50.0, 50.0),
    ("RR=3 / RSI=45",   3.0, 45.0, 55.0),
]


# ── Minimal backtest engine for DailySignal objects ───────────────────────────

from dataclasses import dataclass

@dataclass
class DailyTrade:
    signal:      object
    size:        float
    entry_price: float
    exit_price:  float = 0.0
    exit_time:   pd.Timestamp = None
    pnl:         float = 0.0
    outcome:     str = ""
    bars_held:   int = 0

    @property
    def r_multiple(self) -> float:
        risk = abs(self.entry_price - self.signal.stop)
        if risk == 0:
            return 0.0
        return self.pnl / (risk * self.size)


def run_daily_backtest(
    signals: list,
    df_daily: pd.DataFrame,
    initial_equity: float = 10_000.0,
    risk_pct: float = 0.02,
) -> tuple[list, pd.Series]:
    if not signals:
        return [], pd.Series([initial_equity])

    rm = RiskManager(equity=initial_equity, risk_pct=risk_pct)
    sig_map = {s.timestamp: s for s in signals}

    trades = []
    equity_curve = [initial_equity]
    open_trade = None
    current_date = None

    for i in range(len(df_daily)):
        row = df_daily.iloc[i]
        ts  = df_daily.index[i]

        bar_date = ts.date() if hasattr(ts, 'date') else ts
        if bar_date != current_date:
            rm.reset_day()
            current_date = bar_date

        if open_trade is not None:
            sig = open_trade.signal
            high, low = row["high"], row["low"]
            open_trade.bars_held += 1
            bearish = row["close"] < row["open"]
            outcome = None

            if sig.direction == "long":
                sl_hit = low  <= sig.stop
                tp_hit = high >= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("loss", sig.stop) if bearish else ("win", sig.target)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target
            else:
                sl_hit = high >= sig.stop
                tp_hit = low  <= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("win", sig.target) if bearish else ("loss", sig.stop)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target

            if outcome:
                open_trade.exit_price = exit_p
                open_trade.exit_time  = ts
                open_trade.outcome    = outcome
                if sig.direction == "long":
                    open_trade.pnl = (exit_p - open_trade.entry_price) * open_trade.size
                else:
                    open_trade.pnl = (open_trade.entry_price - exit_p) * open_trade.size
                rm.record_trade(open_trade.pnl)
                trades.append(open_trade)
                equity_curve.append(rm.equity)
                open_trade = None

        if open_trade is None and ts in sig_map:
            ok, _ = rm.can_trade()
            if ok:
                sig  = sig_map[ts]
                size = rm.position_size(sig.entry, sig.stop)
                if size > 0:
                    open_trade = DailyTrade(signal=sig, size=size, entry_price=sig.entry)

    if open_trade is not None:
        last = df_daily.iloc[-1]["close"]
        open_trade.exit_price = last
        open_trade.exit_time  = df_daily.index[-1]
        open_trade.outcome    = "open"
        sig = open_trade.signal
        open_trade.pnl = (last - open_trade.entry_price) * open_trade.size \
                         if sig.direction == "long" \
                         else (open_trade.entry_price - last) * open_trade.size
        trades.append(open_trade)
        equity_curve.append(rm.equity + open_trade.pnl)

    return trades, pd.Series(equity_curve)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_trade_compat(daily_trades):
    """Convert DailyTrade list to something compute_metrics understands."""
    class _T:
        pass
    out = []
    for dt in daily_trades:
        t = _T()
        t.outcome    = dt.outcome
        t.pnl        = dt.pnl
        t.r_multiple = dt.r_multiple
        t.signal     = dt.signal
        t.exit_time  = dt.exit_time
        t.bars_held  = dt.bars_held
        out.append(t)
    return out


def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def _short_metrics(trades, equity):
    compat = _to_trade_compat(trades)
    m = compute_metrics(compat, equity)
    if not m or "message" in m:
        return {}
    return m


# ── Per-asset optimizer ───────────────────────────────────────────────────────

def run_asset(name: str, start: str, end: str) -> tuple | None:
    print(f"\n{'='*62}")
    print(f"  {name.upper()}  (daily bars  {start[:4]}–{end[:4]})".center(62))
    print(f"{'='*62}")

    try:
        df_d = load_parquet(name, "daily").loc[start:end]
        df_w = load_parquet(name, "weekly").loc[start:end]
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        return None

    print(f"  Daily bars : {len(df_d):,}  "
          f"({df_d.index[0].date()} → {df_d.index[-1].date()})")

    results = []

    for label, min_rr, rsi_long, rsi_short in VARIATIONS:
        sigs = scan_daily_signals(
            name, df_w, df_d,
            min_rr=min_rr,
            rsi_entry_long=rsi_long,
            rsi_entry_short=rsi_short,
            rsi_dip_long=rsi_long,
            rsi_dip_short=rsi_short,
        )
        trades, equity = run_daily_backtest(sigs, df_d, INITIAL_EQUITY, RISK_PCT)
        closed = [t for t in trades if t.outcome in ("win","loss")]
        n = len(closed)

        if n < MIN_TRADES:
            print(f"  {label:<26}  only {n} trades — skip")
            results.append((label, min_rr, rsi_long, rsi_short, n, None))
            continue

        m = _short_metrics(trades, equity)
        print(f"  ── {label} ──")
        print(f"     Trades:{n:>4}  WR:{m.get('win_rate','—'):>7}  "
              f"EV/R:{m.get('expectancy_r','—'):>6}  PF:{m.get('profit_factor','—'):>6}")
        print(f"     Return:{m.get('total_return_pct','—'):>7}  "
              f"MaxDD:{m.get('max_drawdown_pct','—'):>7}  "
              f"Sharpe:{m.get('sharpe_ratio','—'):>6}  "
              f"Calmar:{m.get('calmar_ratio','—'):>6}")
        results.append((label, min_rr, rsi_long, rsi_short, n, m))

    valid = [(l, rr, rl, rs, n, m) for l, rr, rl, rs, n, m in results if m]
    if not valid:
        print(f"  No valid variations for {name}")
        return None

    def _calmar(m):
        try:
            return float(str(m.get("calmar_ratio", 0)).replace("%",""))
        except Exception:
            return 0.0

    best = max(valid, key=lambda x: _calmar(x[5]))
    b_label, b_rr, b_rl, b_rs, b_n, b_m = best
    print(f"\n  ★ Best: {b_label}  (min_rr={b_rr})")

    # MC prop firm sims on best variation
    sigs_best = scan_daily_signals(name, df_w, df_d, min_rr=b_rr,
                                   rsi_entry_long=b_rl, rsi_entry_short=b_rs,
                                   rsi_dip_long=b_rl, rsi_dip_short=b_rs)
    trades_best, equity_best = run_daily_backtest(sigs_best, df_d, INITIAL_EQUITY, RISK_PCT)
    r_mults = _r_multiples(trades_best)
    if len(r_mults) >= MIN_TRADES:
        print(f"\n  Prop firm sims ({MC_N_SIMS:,} MC)...")
        run_prop_firm_simulations(r_mults, r_to_pct=RISK_PCT, n_sims=MC_N_SIMS)

    return best


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=ASSETS)
    parser.add_argument("--start",  default="2016-01-01")
    parser.add_argument("--end",    default="2023-12-31")
    args = parser.parse_args()

    print("=" * 62)
    print("  COMMODITY OPTIMIZER — Daily Trend Strategy".center(62))
    print("=" * 62)
    print(f"  Assets : {', '.join(args.assets)}")
    print(f"  Period : {args.start} → {args.end}")
    print(f"  TF     : Weekly (HTF bias) + Daily (entries)")

    winners = {}
    for name in args.assets:
        result = run_asset(name, args.start, args.end)
        if result:
            winners[name] = result

    print(f"\n{'='*62}")
    print("  SUMMARY — Best config per asset".center(62))
    print(f"{'='*62}")
    hdr = f"  {'Asset':<10} {'Config':<24} {'Trades':>7} {'WR':>8} {'Return':>9} {'MaxDD':>8} {'Calmar':>8}"
    print(hdr)
    print("  " + "-"*60)
    for name, (label, rr, rl, rs, n, m) in winners.items():
        if m:
            print(f"  {name:<10} {label:<24} {n:>7} "
                  f"{str(m.get('win_rate','—')):>8} "
                  f"{str(m.get('total_return_pct','—')):>9} "
                  f"{str(m.get('max_drawdown_pct','—')):>8} "
                  f"{str(m.get('calmar_ratio','—')):>8}")
    print("\nDone.")


if __name__ == "__main__":
    main()
