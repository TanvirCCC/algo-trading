"""
ICT/MMXM Commodity Trading Strategy — Full Quant Workflow
SMM591 Commodity Derivatives & Trading — Trading Game
Bayes Business School, City St George's 2025/26

Workflow (Thomas DeltaTrend methodology):
  1. Generate ICT/MMXM signals (FVG, OB, structure, kill zones)
  2. Backtest on historical data
  3. Statistical validation: Bootstrap CI on EV, null model test, IC
  4. Regime analysis: ADX + Hurst + Markov steady state
  5. Monte Carlo: reshuffling, regime-switching, barrier simulation
  6. Generate report paragraphs in SMM591/CDT03 academic language

Assets: Crude Oil (CL=F), Gold (GC=F), Corn (ZC=F)
Account: £10,000 demo | Risk: 2% per trade | Report due: 6 July 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from data.fetcher import fetch, fetch_multi_tf
from backtest.engine import run_backtest, Trade
from backtest.metrics import compute_metrics, print_metrics, trades_to_dataframe, report_summary
from backtest.statistics import validate, required_sample_size, bootstrap_ci
from backtest.monte_carlo import (
    monte_carlo_reshuffle,
    monte_carlo_regime_switching,
    monte_carlo_barrier,
    assign_trade_regimes,
)
from detectors.regime import classify_regime, regime_markov_analysis, add_adx, cusum_events
from strategy.ml_filter import SignalFilter


COMMODITIES = {
    "Crude Oil": "crude",
    "Gold":      "gold",
    "Corn":      "ZC=F",
}


def run_all(save_results: bool = True, run_stats: bool = True):
    all_results = {}

    for name, symbol in COMMODITIES.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}  ({symbol})")
        print(f"{'='*60}")

        try:
            tf = fetch_multi_tf(symbol)
            df_daily = tf["daily"]
            df_1h    = tf["1h"]
            df_15m   = tf.get("15m", df_1h)
        except Exception as e:
            print(f"  Error fetching data: {e}")
            continue

        print(f"  Data: {len(df_daily)} daily | {len(df_1h)} hourly | {len(df_15m)} 15-min bars")

        # ── 1. Backtest (use 15m intraday for more signals) ──────────────────
        trades, equity_curve = run_backtest(
            asset=name,
            df_daily=df_daily,
            df_intraday=df_15m,
            initial_equity=10_000.0,
            risk_pct=0.02,
            min_rr=2.0,
        )
        df_1h = df_15m  # use 15m for all downstream analysis

        metrics = compute_metrics(trades, equity_curve)
        print_metrics(metrics, asset=name)

        if not trades:
            continue

        closed = [t for t in trades if t.outcome in ("win", "loss")]
        r_mults = np.array([t.r_multiple for t in closed]) if closed else np.array([])

        # Save trade log
        if save_results and trades:
            df_trades = trades_to_dataframe(trades)
            out_csv = f"backtest/{name.lower().replace(' ', '_')}_trades.csv"
            df_trades.to_csv(out_csv, index=False)

        if not run_stats or len(closed) < 3:
            print(f"  Skipping statistical validation (need ≥ 3 closed trades, have {len(closed)}).")
            all_results[name] = {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}
            continue

        # ── 2. Statistical Validation ────────────────────────────────────────
        print(f"\n  Running statistical validation...")
        stat_summary = validate(trades, df_1h, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
        if stat_summary:
            print(stat_summary.report())

        # Required sample size for this edge
        if stat_summary and abs(stat_summary.ev_per_trade) > 1e-6:
            n_needed = required_sample_size(delta=abs(stat_summary.ev_per_trade), sigma=1.0)
            print(f"\n  Sample size needed to confirm {abs(stat_summary.ev_per_trade):.2f}R edge: {n_needed} trades")
            print(f"  Current sample: {len(closed)} trades — {'sufficient' if len(closed) >= n_needed else 'INSUFFICIENT'}")

        # ── 3. Regime Analysis ───────────────────────────────────────────────
        print(f"\n  Running regime analysis...")
        df_1h_regime = classify_regime(df_1h, adx_threshold=25.0, use_hurst=True)
        regime_series = df_1h_regime["regime"].values

        # Tag each trade with its regime
        trade_regimes = assign_trade_regimes(closed, df_1h_regime, regime_series)
        if len(set(trade_regimes)) > 1:
            markov = regime_markov_analysis(trade_regimes)
        else:
            print(f"  All trades in same regime: {trade_regimes[0] if len(trade_regimes) > 0 else 'N/A'}")
            markov = None

        # Regime-conditioned EV
        for regime in set(trade_regimes):
            mask = trade_regimes == regime
            if mask.sum() > 0:
                ev_regime = np.mean(r_mults[mask])
                print(f"  EV in {regime} regime: {ev_regime:+.4f}R  ({mask.sum()} trades)")

        # ── 4. ML Signal Filter (SMM748) ─────────────────────────────────────
        print(f"\n  Running ML signal filter (Random Forest)...")

        from strategy.ict_strategy import prepare_data as _prepare_data
        _, df_1h_prep = _prepare_data(df_daily.copy(), df_1h.copy())

        if "cusum_up" not in df_1h_prep.columns:
            df_1h_prep = cusum_events(df_1h_prep)

        ml_filter = SignalFilter(probability_threshold=0.55)
        ml_result = ml_filter.fit(closed, df_1h_prep, train_ratio=0.70)

        if ml_result:
            print(ml_result.report())

            # Re-run backtest with RF gating entries
            trades_ml, equity_ml = run_backtest(
                asset=name,
                df_daily=df_daily,
                df_intraday=df_1h_prep,
                initial_equity=10_000.0,
                risk_pct=0.02,
                min_rr=2.0,
                signal_filter=ml_filter,
            )
            metrics_ml = compute_metrics(trades_ml, equity_ml)
            _print_ml_comparison(metrics, metrics_ml, name)
        else:
            ml_filter = None

        # ── 5. Monte Carlo ───────────────────────────────────────────────────
        print(f"\n  Running Monte Carlo simulations...")

        mc1 = monte_carlo_reshuffle(r_mults, n_sims=5_000, r_to_pct=0.02)
        print(mc1.report())

        if markov and len(set(trade_regimes)) > 1 and len(r_mults) >= len(set(trade_regimes)):
            try:
                mc2 = monte_carlo_regime_switching(
                    r_mults, trade_regimes, n_sims=5_000, r_to_pct=0.02
                )
                print(mc2.report())
            except (ValueError, ZeroDivisionError):
                print("  Regime-switching Monte Carlo skipped (degenerate Markov matrix).")

        mc3 = monte_carlo_barrier(
            r_mults,
            initial_balance=10_000.0,
            profit_target_pct=0.20,
            max_drawdown_pct=0.20,
            challenge_fee=0.0,
            n_sims=5_000,
            r_to_pct=0.02,
        )
        print(mc3.report())

        # ── 6. Report Paragraphs ─────────────────────────────────────────────
        report_para = report_summary(trades, equity_curve, name)
        rpt_path = f"backtest/{name.lower().replace(' ', '_')}_report_paragraph.txt"
        with open(rpt_path, "w") as f:
            f.write(report_para)
            if stat_summary:
                f.write(f"\n\nStatistical Validation:\n")
                f.write(f"Bootstrap 95% CI on EV: [{stat_summary.ci_lower:+.4f}R, {stat_summary.ci_upper:+.4f}R]. ")
                f.write(f"Null model p-value: {stat_summary.p_value:.4f}. ")
                f.write(f"Information Coefficient: {stat_summary.ic:.4f}. ")
                f.write(f"Timing {'is' if stat_summary.timing_significant else 'is not'} statistically significant vs random entry.")
        print(f"  Report paragraph saved: {rpt_path}")

        all_results[name] = {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "stat_summary": stat_summary,
            "mc_basic": mc1,
            "mc_barrier": mc3,
        }

    _print_portfolio_summary(all_results)
    return all_results


def _print_ml_comparison(m_raw: dict, m_ml: dict, asset: str):
    """
    Side-by-side comparison of raw strategy vs ML-filtered strategy.
    Mirrors Thomas DeltaTrend's Sharpe/Sortino comparison from TikTok.
    SMM748: model evaluation — does the RF filter improve risk-adjusted returns?
    """
    print(f"\n{'─'*57}")
    print(f"  ML FILTER COMPARISON — {asset}  (SMM748)")
    print(f"{'─'*57}")
    print(f"  {'Metric':<28} {'Raw Strategy':>12}  {'+ ML Filter':>12}")
    print(f"  {'─'*54}")

    def fmt(d, key):
        v = d.get(key, "N/A")
        return str(v) if v != "N/A" else "N/A"

    rows = [
        ("Trades taken",    "total_trades"),
        ("Win rate",        "win_rate"),
        ("Sharpe ratio",    "sharpe_ratio"),
        ("Sortino ratio",   "sortino_ratio"),
        ("Profit factor",   "profit_factor"),
        ("Total PnL (£)",   "total_pnl_gbp"),
        ("Max drawdown %",  "max_drawdown_pct"),
    ]
    for label, key in rows:
        print(f"  {label:<28} {fmt(m_raw, key):>12}  {fmt(m_ml, key):>12}")

    # Highlight Sharpe improvement
    raw_sh = m_raw.get("sharpe_ratio", 0) or 0
    ml_sh  = m_ml.get("sharpe_ratio", 0) or 0
    delta  = ml_sh - raw_sh
    arrow  = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
    print(f"  {'─'*54}")
    print(f"  Sharpe delta: {delta:+.2f}  {arrow}  {'IMPROVEMENT' if delta > 0 else 'NO IMPROVEMENT'}")
    print(f"{'─'*57}")


def _print_portfolio_summary(results: dict):
    print(f"\n{'='*60}")
    print(f"  PORTFOLIO SUMMARY — ALL COMMODITIES".center(60))
    print(f"{'='*60}")

    total_trades = total_wins = 0
    total_pnl = 0.0

    for name, r in results.items():
        m = r.get("metrics", {})
        s = r.get("stat_summary")
        if not m or "message" in m:
            print(f"  {name:<15} — no trades")
            continue
        ci_str = f"CI [{s.ci_lower:+.3f}, {s.ci_upper:+.3f}]" if s else "CI N/A"
        print(f"  {name:<15}  WR {m.get('win_rate','?'):<8}  PnL: £{m.get('total_pnl_gbp','?'):<10}  {ci_str}")
        total_trades += m.get("total_trades", 0)
        total_wins += m.get("wins", 0)
        pnl = m.get("total_pnl_gbp", 0)
        if isinstance(pnl, (int, float)):
            total_pnl += pnl

    print(f"{'─'*60}")
    if total_trades > 0:
        print(f"  Total trades: {total_trades}  |  Combined PnL: £{total_pnl:.2f}")
        print(f"  Portfolio win rate: {total_wins/total_trades*100:.1f}%")
        n_needed = required_sample_size(delta=0.1, sigma=1.0)
        print(f"  Trades needed to confirm 0.1R edge: {n_needed} (currently have {total_trades})")
    print(f"{'='*60}")


def run_long_backtest(start: str = "2010-01-01", run_stats: bool = True, save_results: bool = True):
    """
    Long-term backtest on daily bars from `start` to today.
    Uses weekly bars for HTF bias (EMA50/200), daily bars for signal detection.
    Kill zones are disabled — entries fire on end-of-day bar close.

    yfinance note: intraday (1h, 15m) is capped at 730 days regardless of start.
    Daily data is available back to ~2000 for GC=F, CL=F, ZC=F.
    """
    from data.fetcher import fetch_long_history

    print(f"\n{'='*60}")
    print(f"  LONG-TERM BACKTEST  {start} → TODAY".center(60))
    print(f"  Crude Oil | Gold | Corn  —  Daily bars")
    print(f"  HTF bias: weekly EMA50/200  |  Entry TF: daily FVG/OB")
    print(f"  Kill zones disabled (daily bars have no session time)")
    print(f"{'='*60}")

    all_results = {}

    for name, symbol in COMMODITIES.items():
        print(f"\n{'='*60}")
        print(f"  {name.upper()}  ({symbol})")
        print(f"{'='*60}")

        try:
            tf = fetch_long_history(symbol, start=start)
            df_weekly = tf["weekly"]
            df_daily  = tf["daily"]
        except Exception as e:
            print(f"  Error fetching data: {e}")
            continue

        print(f"  Data: {len(df_weekly)} weekly bars | {len(df_daily)} daily bars")
        print(f"  Range: {df_daily.index[0].date()} → {df_daily.index[-1].date()}")

        trades, equity_curve = run_backtest(
            asset=name,
            df_daily=df_weekly,
            df_intraday=df_daily,
            initial_equity=10_000.0,
            risk_pct=0.02,
            min_rr=2.0,
            use_kill_zones=False,
            zone_lookback=80,       # ~16 trading weeks — enough depth in trend context
            zone_tolerance_atr=2.0, # 2 ATR proximity for daily zone approach
        )

        metrics = compute_metrics(trades, equity_curve)
        print_metrics(metrics, asset=name)

        if not trades:
            continue

        closed = [t for t in trades if t.outcome in ("win", "loss")]
        r_mults = np.array([t.r_multiple for t in closed]) if closed else np.array([])

        if save_results and trades:
            df_trades = trades_to_dataframe(trades)
            out_csv = f"backtest/{name.lower().replace(' ', '_')}_long_trades.csv"
            df_trades.to_csv(out_csv, index=False)
            print(f"  Trade log: {out_csv}")

        if not run_stats or len(closed) < 3:
            print(f"  Skipping stats (need ≥ 3 closed trades, have {len(closed)}).")
            all_results[name] = {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}
            continue

        # ── Statistical Validation ───────────────────────────────────────────
        print(f"\n  Running statistical validation...")

        # prepare_data enriches df_daily with RSI/CCI/ATR etc. needed by stats + ML filter
        from strategy.ict_strategy import prepare_data as _prepare_data
        _, df_daily_prep = _prepare_data(df_weekly.copy(), df_daily.copy())

        stat_summary = validate(trades, df_daily_prep, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
        if stat_summary:
            print(stat_summary.report())
            if abs(stat_summary.ev_per_trade) > 1e-6:
                n_needed = required_sample_size(delta=abs(stat_summary.ev_per_trade), sigma=1.0)
                print(f"\n  Trades to confirm {abs(stat_summary.ev_per_trade):.2f}R edge: {n_needed}")
                print(f"  Current sample: {len(closed)} — {'sufficient' if len(closed) >= n_needed else 'INSUFFICIENT'}")

        # ── Regime Analysis ──────────────────────────────────────────────────
        print(f"\n  Running regime analysis...")
        df_daily_regime = classify_regime(df_daily_prep, adx_threshold=25.0, use_hurst=True)
        regime_series = df_daily_regime["regime"].values
        trade_regimes = assign_trade_regimes(closed, df_daily_regime, regime_series)

        if len(set(trade_regimes)) > 1:
            markov = regime_markov_analysis(trade_regimes)
        else:
            markov = None
            print(f"  All trades in single regime: {trade_regimes[0] if len(trade_regimes) > 0 else 'N/A'}")

        for regime in sorted(set(trade_regimes)):
            mask = trade_regimes == regime
            if mask.sum() > 0 and len(r_mults) > 0:
                ev_r = np.mean(r_mults[mask])
                print(f"  EV in {regime} regime: {ev_r:+.4f}R  ({mask.sum()} trades)")

        # ── ML Signal Filter ─────────────────────────────────────────────────
        print(f"\n  Running ML signal filter (Random Forest)...")
        if "cusum_up" not in df_daily_prep.columns:
            df_daily_prep = cusum_events(df_daily_prep)

        ml_filter = SignalFilter(probability_threshold=0.55)
        ml_result = ml_filter.fit(closed, df_daily_prep, train_ratio=0.70)

        if ml_result:
            print(ml_result.report())
            trades_ml, equity_ml = run_backtest(
                asset=name,
                df_daily=df_weekly,
                df_intraday=df_daily_prep,  # enriched df needed by ml_filter.accept()
                initial_equity=10_000.0,
                risk_pct=0.02,
                min_rr=2.0,
                use_kill_zones=False,
                zone_lookback=80,
                zone_tolerance_atr=2.0,
                signal_filter=ml_filter,
            )
            metrics_ml = compute_metrics(trades_ml, equity_ml)
            _print_ml_comparison(metrics, metrics_ml, name)
        else:
            ml_filter = None

        # ── Monte Carlo ──────────────────────────────────────────────────────
        print(f"\n  Running Monte Carlo simulations...")
        mc1 = monte_carlo_reshuffle(r_mults, n_sims=5_000, r_to_pct=0.02)
        print(mc1.report())

        n_regimes = len(set(trade_regimes))
        if markov and n_regimes > 1 and len(r_mults) >= n_regimes:
            try:
                mc2 = monte_carlo_regime_switching(r_mults, trade_regimes, n_sims=5_000, r_to_pct=0.02)
                print(mc2.report())
            except (ValueError, ZeroDivisionError):
                print("  Regime-switching Monte Carlo skipped (degenerate Markov matrix).")

        mc3 = monte_carlo_barrier(
            r_mults,
            initial_balance=10_000.0,
            profit_target_pct=0.20,
            max_drawdown_pct=0.20,
            challenge_fee=0.0,
            n_sims=5_000,
            r_to_pct=0.02,
        )
        print(mc3.report())

        # ── Report Paragraph ─────────────────────────────────────────────────
        m = metrics
        report_para = (
            f"We back-tested our false-breakout and gap-fill entry rules on {name} "
            f"using daily OHLCV data from {start} to present ({len(df_daily)} trading days). "
            f"The strategy generated {m.get('total_trades', 0)} trade setups over this period "
            f"({m.get('trades_per_year', 0):.1f} trades per year). "
            f"Win rate: {m.get('win_rate', 'N/A')}. "
            f"Expectancy: {m.get('expectancy_r', 0):+.3f}R per trade. "
            f"Profit factor: {m.get('profit_factor', 'N/A')}. "
            f"Sharpe ratio (annualised): {m.get('sharpe_ratio', 'N/A')}. "
            f"Sortino ratio: {m.get('sortino_ratio', 'N/A')}. "
            f"CAGR: {m.get('cagr_pct', 'N/A')}%. "
            f"Calmar ratio: {m.get('calmar_ratio', 'N/A')}. "
            f"Maximum drawdown: {m.get('max_drawdown_pct', 'N/A')}%. "
            f"Max consecutive losses: {m.get('max_consec_losses', 'N/A')}. "
            f"Total return: {m.get('total_return_pct', 'N/A')}% on £10,000 initial equity."
        )
        rpt_path = f"backtest/{name.lower().replace(' ', '_')}_long_report_paragraph.txt"
        with open(rpt_path, "w") as f:
            f.write(report_para)
            if stat_summary:
                f.write(f"\n\nStatistical Validation:\n")
                f.write(f"Bootstrap 95% CI on EV: [{stat_summary.ci_lower:+.4f}R, {stat_summary.ci_upper:+.4f}R]. ")
                f.write(f"Null model p-value: {stat_summary.p_value:.4f}. ")
                f.write(f"IC: {stat_summary.ic:.4f}. ")
                f.write(f"Timing {'IS' if stat_summary.timing_significant else 'IS NOT'} significant vs random entry.")
        print(f"  Report paragraph: {rpt_path}")

        all_results[name] = {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
            "stat_summary": stat_summary if run_stats else None,
        }

    _print_portfolio_summary(all_results)
    return all_results


def run_ib_backtest(start: str = "20150101", run_stats: bool = True, save_results: bool = True):
    """
    Long-term intraday backtest using Interactive Brokers data.
    Requires data pre-downloaded via: python3 data/ib_fetcher.py --start 20150101

    Uses 5-min bars for signal detection + kill zones.
    Uses daily bars (also from IB) for HTF EMA50/200 bias.
    """
    from data.ib_fetcher import load_parquet, ASSET_SPECS as IB_ASSETS

    print(f"\n{'='*60}")
    print(f"  IB INTRADAY BACKTEST  (5-min bars, kill zones ON)".center(60))
    print(f"  Assets: {', '.join(IB_ASSETS.keys())}")
    print(f"{'='*60}")

    all_results = {}

    for name in IB_ASSETS:
        print(f"\n{'='*60}")
        print(f"  {name.upper()}")
        print(f"{'='*60}")

        try:
            df_5m    = load_parquet(name, "5m")
            df_daily = load_parquet(name, "daily")
        except FileNotFoundError as e:
            print(f"  {e}")
            continue

        print(f"  5-min bars : {len(df_5m):,}  ({df_5m.index[0].date()} → {df_5m.index[-1].date()})")
        print(f"  Daily bars : {len(df_daily):,}")

        # ── News calendar ────────────────────────────────────────────────────
        from data.news_calendar import get_high_impact_events, mark_news_windows
        from detectors.seasonality import add_seasonal_columns

        news_events = get_high_impact_events(df_5m.index[0], df_5m.index[-1])
        df_5m = mark_news_windows(df_5m, news_events, pre_minutes=10, post_minutes=30)
        df_5m = add_seasonal_columns(df_5m, asset=name)
        n_news = int(df_5m["news_entry"].sum())
        print(f"  News events: {len(news_events)} high-impact USD events marked "
              f"({n_news} bars in post-news entry windows)")

        # Aggregate daily → weekly for HTF bias
        df_weekly = df_daily.resample("1W").agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"), close=("close","last"), volume=("volume","sum")
        ).dropna()

        trades, equity_curve = run_backtest(
            asset=name,
            df_daily=df_daily,
            df_intraday=df_5m,
            initial_equity=10_000.0,
            risk_pct=0.02,
            min_rr=2.0,
            use_kill_zones=True,
            zone_lookback=200,
            zone_tolerance_atr=0.3,
        )

        metrics = compute_metrics(trades, equity_curve)
        print_metrics(metrics, asset=name)

        if not trades:
            continue

        closed = [t for t in trades if t.outcome in ("win", "loss")]
        r_mults = np.array([t.r_multiple for t in closed]) if closed else np.array([])

        if save_results and trades:
            df_trades = trades_to_dataframe(trades)
            out_csv = f"backtest/{name.lower()}_ib_trades.csv"
            df_trades.to_csv(out_csv, index=False)
            print(f"  Trade log: {out_csv}")

        if not run_stats or len(closed) < 6:
            print(f"  Skipping stats (need ≥ 6 closed trades, have {len(closed)}).")
            all_results[name] = {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}
            continue

        # ── Statistical Validation ───────────────────────────────────────────
        print(f"\n  Running statistical validation...")
        from strategy.ict_strategy import prepare_data as _prepare_data
        _, df_5m_prep = _prepare_data(df_daily.copy(), df_5m.copy())

        stat_summary = validate(trades, df_5m_prep, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
        if stat_summary:
            print(stat_summary.report())
            if abs(stat_summary.ev_per_trade) > 1e-6:
                n_needed = required_sample_size(delta=abs(stat_summary.ev_per_trade), sigma=1.0)
                print(f"\n  Trades to confirm {abs(stat_summary.ev_per_trade):.2f}R edge: {n_needed}")
                print(f"  Current sample: {len(closed)} — {'sufficient' if len(closed) >= n_needed else 'INSUFFICIENT'}")

        # ── Regime Analysis ──────────────────────────────────────────────────
        print(f"\n  Running regime analysis...")
        df_regime = classify_regime(df_5m_prep, adx_threshold=25.0, use_hurst=True)
        regime_series = df_regime["regime"].values
        trade_regimes = assign_trade_regimes(closed, df_regime, regime_series)

        if len(set(trade_regimes)) > 1:
            markov = regime_markov_analysis(trade_regimes)
        else:
            markov = None
            print(f"  All trades in single regime: {trade_regimes[0] if len(trade_regimes) > 0 else 'N/A'}")

        for regime in sorted(set(trade_regimes)):
            mask = trade_regimes == regime
            if mask.sum() > 0 and len(r_mults) > 0:
                print(f"  EV in {regime}: {np.mean(r_mults[mask]):+.4f}R  ({mask.sum()} trades)")

        # ── ML Signal Filter ─────────────────────────────────────────────────
        print(f"\n  Running ML signal filter...")
        if "cusum_up" not in df_5m_prep.columns:
            df_5m_prep = cusum_events(df_5m_prep)

        ml_filter = SignalFilter(probability_threshold=0.55)
        ml_result = ml_filter.fit(closed, df_5m_prep, train_ratio=0.70)
        if ml_result:
            print(ml_result.report())
            trades_ml, equity_ml = run_backtest(
                asset=name, df_daily=df_daily, df_intraday=df_5m_prep,
                initial_equity=10_000.0, risk_pct=0.02, min_rr=2.0,
                use_kill_zones=True, zone_lookback=200, zone_tolerance_atr=0.3,
                signal_filter=ml_filter,
            )
            _print_ml_comparison(metrics, compute_metrics(trades_ml, equity_ml), name)

        # ── Monte Carlo ──────────────────────────────────────────────────────
        if len(r_mults) >= 3:
            print(f"\n  Running Monte Carlo simulations...")
            mc1 = monte_carlo_reshuffle(r_mults, n_sims=5_000, r_to_pct=0.02)
            print(mc1.report())
            n_regimes = len(set(trade_regimes))
            if markov and n_regimes > 1 and len(r_mults) >= n_regimes:
                try:
                    mc2 = monte_carlo_regime_switching(r_mults, trade_regimes, n_sims=5_000, r_to_pct=0.02)
                    print(mc2.report())
                except (ValueError, ZeroDivisionError):
                    print("  Regime-switching MC skipped.")

            # Trading Game barrier (SMM591)
            mc3 = monte_carlo_barrier(
                r_mults, initial_balance=10_000.0,
                profit_target_pct=0.20, max_drawdown_pct=0.20,
                challenge_fee=0.0, n_sims=5_000, r_to_pct=0.02,
            )
            print(mc3.report())

            # Prop firm challenge analysis (DeltaTrend Section 11)
            from backtest.monte_carlo import run_prop_firm_simulations
            run_prop_firm_simulations(r_mults, r_to_pct=0.02, n_sims=5_000)

        all_results[name] = {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}

    _print_portfolio_summary(all_results)
    return all_results


def run_bloomberg_backtest(data_dir: str = "Bloomberg Data", run_stats: bool = True, save_results: bool = True):
    """
    Intraday backtest on Bloomberg 5-min bars (Jan–May 2026).
    Uses yfinance daily/weekly for HTF bias (EMA50/200).
    Kill zones are enabled — entries restricted to London + NY opens.
    """
    from data.bloomberg import load_bloomberg_excel, BLOOMBERG_ASSETS
    from data.fetcher import fetch_long_history

    print(f"\n{'='*60}")
    print(f"  BLOOMBERG INTRADAY BACKTEST  (5-min bars)".center(60))
    print(f"  Kill zones: ON  |  Timeframe: 5-min")
    print(f"  London 07:00-10:00 UTC | NY 12:00-15:00 UTC")
    print(f"{'='*60}")

    all_results = {}

    for name, (filename, yf_symbol) in BLOOMBERG_ASSETS.items():
        filepath = f"{data_dir}/{filename}"
        print(f"\n{'='*60}")
        print(f"  {name.upper()}  ({filename})")
        print(f"{'='*60}")

        # ── Load 5-min intraday bars from Bloomberg ──────────────────────────
        try:
            df_5m = load_bloomberg_excel(filepath)
        except Exception as e:
            print(f"  Error loading Bloomberg file: {e}")
            continue

        if df_5m.empty:
            print(f"  No data parsed from {filename}")
            continue

        start_date = df_5m.index[0].strftime("%Y-%m-%d")
        end_date   = df_5m.index[-1].strftime("%Y-%m-%d")
        print(f"  5-min bars: {len(df_5m):,}  |  {start_date} → {end_date}")

        # ── Fetch HTF daily/weekly from yfinance for EMA50/200 bias ─────────
        try:
            tf = fetch_long_history(yf_symbol, start="2022-01-01")
            df_weekly = tf["weekly"]
            df_daily  = tf["daily"]
            print(f"  HTF: {len(df_daily)} daily bars (yfinance, for EMA bias)")
        except Exception as e:
            print(f"  Warning — could not fetch HTF data: {e}. Using 5-min as HTF.")
            df_daily  = df_5m.resample("1D").agg(
                open=("open","first"), high=("high","max"),
                low=("low","min"), close=("close","last"), volume=("volume","sum")
            ).dropna()
            df_weekly = df_5m.resample("1W").agg(
                open=("open","first"), high=("high","max"),
                low=("low","min"), close=("close","last"), volume=("volume","sum")
            ).dropna()

        # ── Backtest ──────────────────────────────────────────────────────────
        trades, equity_curve = run_backtest(
            asset=name,
            df_daily=df_daily,
            df_intraday=df_5m,
            initial_equity=10_000.0,
            risk_pct=0.02,
            min_rr=2.0,
            use_kill_zones=True,    # re-enabled for intraday
            zone_lookback=200,      # look back 200 × 5-min bars (~16h) for active zones
            zone_tolerance_atr=0.5,
        )

        metrics = compute_metrics(trades, equity_curve)
        print_metrics(metrics, asset=name)

        if not trades:
            continue

        closed = [t for t in trades if t.outcome in ("win", "loss")]
        r_mults = np.array([t.r_multiple for t in closed]) if closed else np.array([])

        if save_results and trades:
            df_trades = trades_to_dataframe(trades)
            out_csv = f"backtest/{name.lower()}_bloomberg_trades.csv"
            df_trades.to_csv(out_csv, index=False)
            print(f"  Trade log: {out_csv}")

        if not run_stats or len(closed) < 6:
            note = f"need ≥ 6, have {len(closed)} — export more Bloomberg data for full stats"
            print(f"  Skipping stats ({note}).")
            all_results[name] = {"trades": trades, "equity_curve": equity_curve, "metrics": metrics}
            continue

        # ── Statistical Validation ───────────────────────────────────────────
        print(f"\n  Running statistical validation...")
        from strategy.ict_strategy import prepare_data as _prepare_data
        _, df_5m_prep = _prepare_data(df_daily.copy(), df_5m.copy())

        stat_summary = validate(trades, df_5m_prep, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
        if stat_summary:
            print(stat_summary.report())
            if abs(stat_summary.ev_per_trade) > 1e-6:
                n_needed = required_sample_size(delta=abs(stat_summary.ev_per_trade), sigma=1.0)
                print(f"\n  Trades to confirm {abs(stat_summary.ev_per_trade):.2f}R edge: {n_needed}")
                print(f"  Current sample: {len(closed)} — {'sufficient' if len(closed) >= n_needed else 'INSUFFICIENT'}")

        # ── Regime Analysis ──────────────────────────────────────────────────
        print(f"\n  Running regime analysis...")
        df_regime = classify_regime(df_5m_prep, adx_threshold=25.0, use_hurst=True)
        regime_series = df_regime["regime"].values
        trade_regimes = assign_trade_regimes(closed, df_regime, regime_series)

        if len(set(trade_regimes)) > 1:
            markov = regime_markov_analysis(trade_regimes)
        else:
            markov = None
            print(f"  All trades in single regime: {trade_regimes[0] if len(trade_regimes) > 0 else 'N/A'}")

        for regime in sorted(set(trade_regimes)):
            mask = trade_regimes == regime
            if mask.sum() > 0 and len(r_mults) > 0:
                ev_r = np.mean(r_mults[mask])
                print(f"  EV in {regime} regime: {ev_r:+.4f}R  ({mask.sum()} trades)")

        # ── ML Signal Filter ─────────────────────────────────────────────────
        print(f"\n  Running ML signal filter (Random Forest)...")
        if "cusum_up" not in df_5m_prep.columns:
            df_5m_prep = cusum_events(df_5m_prep)

        ml_filter = SignalFilter(probability_threshold=0.55)
        ml_result = ml_filter.fit(closed, df_5m_prep, train_ratio=0.70)

        if ml_result:
            print(ml_result.report())
            trades_ml, equity_ml = run_backtest(
                asset=name,
                df_daily=df_daily,
                df_intraday=df_5m_prep,
                initial_equity=10_000.0,
                risk_pct=0.02,
                min_rr=2.0,
                use_kill_zones=True,
                zone_lookback=200,
                zone_tolerance_atr=0.5,
                signal_filter=ml_filter,
            )
            metrics_ml = compute_metrics(trades_ml, equity_ml)
            _print_ml_comparison(metrics, metrics_ml, name)

        # ── Monte Carlo ──────────────────────────────────────────────────────
        if len(r_mults) >= 3:
            print(f"\n  Running Monte Carlo simulations...")
            mc1 = monte_carlo_reshuffle(r_mults, n_sims=5_000, r_to_pct=0.02)
            print(mc1.report())

            n_regimes = len(set(trade_regimes))
            if markov and n_regimes > 1 and len(r_mults) >= n_regimes:
                try:
                    mc2 = monte_carlo_regime_switching(r_mults, trade_regimes, n_sims=5_000, r_to_pct=0.02)
                    print(mc2.report())
                except (ValueError, ZeroDivisionError):
                    print("  Regime-switching Monte Carlo skipped (degenerate Markov matrix).")

            mc3 = monte_carlo_barrier(
                r_mults, initial_balance=10_000.0,
                profit_target_pct=0.20, max_drawdown_pct=0.20,
                challenge_fee=0.0, n_sims=5_000, r_to_pct=0.02,
            )
            print(mc3.report())

        all_results[name] = {
            "trades": trades,
            "equity_curve": equity_curve,
            "metrics": metrics,
        }

    _print_portfolio_summary(all_results)
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ICT/MMXM Commodity Backtest — SMM591")
    parser.add_argument("--long",      action="store_true", help="Run long-term daily backtest (2010-today)")
    parser.add_argument("--short",     action="store_true", help="Run short-term 60-day intraday backtest")
    parser.add_argument("--both",      action="store_true", help="Run both long and short backtests")
    parser.add_argument("--bloomberg", action="store_true", help="Run Bloomberg 5-min intraday backtest")
    parser.add_argument("--ib",        action="store_true", help="Run IB 5-min intraday backtest (best data, default)")
    parser.add_argument("--start", default="20150101", help="Start date for IB/long backtest (YYYYMMDD)")
    args = parser.parse_args()

    if args.both:
        run_long_backtest(start=args.start)
        run_all()
    elif args.short:
        run_all()
    elif args.long:
        run_long_backtest(start=args.start)
    elif args.bloomberg:
        run_bloomberg_backtest()
    else:
        # Default: IB intraday (best data)
        run_ib_backtest(start=args.start)
