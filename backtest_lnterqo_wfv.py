"""
lnterqo Strategy — Walk-Forward Validation (80/20 IS/OOS)
==========================================================
Data  : data/historical/gold_5m_combined.parquet  (2016–2026, 713k bars)
Split : 80% in-sample (parameter search + ML train) / 20% OOS (frozen, unbiased)

lnterqo methodology:
  - CISD primary entry trigger (Change in State of Delivery)
  - AMD phase: Asia accumulation → London/NY manipulation + distribution
  - Entry zones: BKR CE / IFVG / FVG
  - Session timing: London 07–10 UTC, NY 12–15 UTC
"""

import warnings, sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from detectors.regime import classify_regime, regime_markov_analysis, cusum_events
from backtest.engine import run_backtest as _ict_run   # reuse execution engine
from backtest.metrics import compute_metrics, print_metrics, trades_to_dataframe, report_summary
from backtest.statistics import validate, required_sample_size
from backtest.monte_carlo import (
    monte_carlo_reshuffle, monte_carlo_regime_switching, monte_carlo_barrier,
    assign_trade_regimes, run_prop_firm_simulations, PROP_FIRM_CONFIGS,
)
from strategy.ml_filter import SignalFilter
from strategy.lnterqo_strategy import prepare_data, scan_for_signals, Signal
from strategy.risk_manager import RiskManager


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("data/historical")
COMBINED_5M     = DATA_DIR / "gold_5m_combined.parquet"
IN_SAMPLE_RATIO = 0.80
MC_N_SIMS       = 100_000
ML_THRESHOLD    = 0.60
MIN_TRADES_ML   = 15
MIN_TRADES_MC   = 3
INITIAL_EQUITY  = 10_000.0
R_TO_PCT        = 0.005   # 0.5% default risk (matches RiskManager)

# Variations: (label, min_rr, zone_lookback, cisd_lookback)
VARIATIONS = [
    ("CISD_RR2_ZL50",  2.0, 50, 40),
    ("CISD_RR2_ZL80",  2.0, 80, 40),
    ("CISD_RR3_ZL50",  3.0, 50, 40),
]

_FIRM_KEYS = ["FTMO_50k", "The5ers_100k", "TopStep_50k"]


# ── Custom backtest runner (lnterqo strategy) ──────────────────────────────────

def run_lnterqo_backtest(
    asset, df_daily, df_5m,
    min_rr, zone_lookback, cisd_lookback,
    initial_equity=INITIAL_EQUITY,
    signal_filter=None,
    df_enriched_for_filter=None,
    use_news_filter=True,
    force_neutral_bias=False,
):
    """Run backtest using lnterqo strategy signals through the standard engine."""
    from backtest.engine import Trade

    rm = RiskManager(equity=initial_equity)
    signals = scan_for_signals(
        asset, df_daily, df_5m,
        min_rr=min_rr,
        use_news_filter=use_news_filter,
        zone_lookback=zone_lookback,
        cisd_lookback=cisd_lookback,
        force_neutral_bias=force_neutral_bias,
    )

    if not signals:
        return [], pd.Series([initial_equity], name="equity")

    trades       = []
    equity_curve = [initial_equity]
    signal_map   = {s.timestamp: s for s in signals}
    open_trade: Trade | None = None
    current_date = None

    for i in range(len(df_5m)):
        row = df_5m.iloc[i]
        ts  = df_5m.index[i]

        bar_date = ts.date()
        if bar_date != current_date:
            rm.reset_day()
            current_date = bar_date

        # Exit open trade
        if open_trade is not None:
            sig  = open_trade.signal
            high = row["high"]
            low  = row["low"]
            bearish_bar = row["close"] < row["open"]
            open_trade.bars_held += 1
            outcome = None
            exit_p  = 0.0

            if sig.direction == "long":
                sl_hit = low  <= sig.stop
                tp_hit = high >= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("loss", sig.stop) if bearish_bar else ("win", sig.target)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target
            else:
                sl_hit = high >= sig.stop
                tp_hit = low  <= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("win", sig.target) if bearish_bar else ("loss", sig.stop)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target

            if outcome is not None:
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

        # New entry
        if open_trade is None and ts in signal_map:
            ok, _ = rm.can_trade()
            if ok:
                sig = signal_map[ts]
                _df_filter = df_enriched_for_filter if df_enriched_for_filter is not None else df_5m
                if signal_filter is not None and not signal_filter.accept(sig, _df_filter):
                    continue
                size = rm.position_size(sig.entry, sig.stop, sig.confidence)
                if size > 0:
                    open_trade = Trade(signal=sig, size=size, entry_price=sig.entry)

    # Close open trade at last bar
    if open_trade is not None:
        last_price = df_5m.iloc[-1]["close"]
        open_trade.exit_price = last_price
        open_trade.exit_time  = df_5m.index[-1]
        open_trade.outcome    = "open"
        if open_trade.signal.direction == "long":
            open_trade.pnl = (last_price - open_trade.entry_price) * open_trade.size
        else:
            open_trade.pnl = (open_trade.entry_price - last_price) * open_trade.size
        trades.append(open_trade)
        equity_curve.append(rm.equity + open_trade.pnl)

    return trades, pd.Series(equity_curve, name="equity")


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def _pass_rates(r_mults, n_sims):
    results = {}
    for key in _FIRM_KEYS:
        cfg = PROP_FIRM_CONFIGS[key]
        br  = monte_carlo_barrier(
            r_mults,
            initial_balance=cfg["initial_balance"],
            profit_target_pct=cfg["profit_target_pct"],
            max_drawdown_pct=cfg["max_drawdown_pct"],
            daily_loss_limit_pct=cfg.get("daily_loss_limit_pct"),
            challenge_fee=cfg["challenge_fee"],
            r_to_pct=R_TO_PCT,
            n_sims=n_sims,
        )
        results[key] = br.pass_rate
    return results


def _fmt(v):
    return f"{v*100:6.1f}%" if v is not None else "   N/A"


def _run_variation(label, min_rr, zone_lb, cisd_lb,
                   df_daily_is, df_5m_is):
    trades_raw, equity_raw = run_lnterqo_backtest(
        "Gold", df_daily_is, df_5m_is,
        min_rr=min_rr, zone_lookback=zone_lb, cisd_lookback=cisd_lb,
        force_neutral_bias=True,
    )
    n_raw = len([t for t in trades_raw if t.outcome in ("win", "loss")])

    if n_raw < MIN_TRADES_ML:
        return trades_raw, equity_raw, n_raw, None, None

    ml = SignalFilter(probability_threshold=ML_THRESHOLD)
    ml_result = ml.fit(trades_raw, df_5m_is)
    if ml_result is None:
        return trades_raw, equity_raw, n_raw, None, None

    trades_ml, equity_ml = run_lnterqo_backtest(
        "Gold", df_daily_is, df_5m_is,
        min_rr=min_rr, zone_lookback=zone_lb, cisd_lookback=cisd_lb,
        force_neutral_bias=True,
        signal_filter=ml, df_enriched_for_filter=df_5m_is,
    )
    return trades_ml, equity_ml, n_raw, ml_result.acceptance_rate, ml


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*65}")
    print("  lnterqo STRATEGY — WALK-FORWARD VALIDATION  (80/20 IS/OOS)")
    print(f"  Data: {COMBINED_5M.name}")
    print(f"{'='*65}")

    if not COMBINED_5M.exists():
        print(f"ERROR: {COMBINED_5M} not found.")
        sys.exit(1)

    df_5m = pd.read_parquet(COMBINED_5M)
    df_5m.columns = [c.lower() for c in df_5m.columns]
    df_5m = df_5m[["open", "high", "low", "close", "volume"]]

    df_daily = (
        df_5m.resample("D")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .dropna(subset=["open"])
    )

    total_start = df_5m.index[0]
    total_end   = df_5m.index[-1]
    total_days  = (total_end - total_start).days
    split_date  = total_start + timedelta(days=int(total_days * IN_SAMPLE_RATIO))

    print(f"\n  Full range  : {total_start.date()} → {total_end.date()}  ({len(df_5m):,} bars)")
    print(f"  Split date  : {split_date.date()}  (80% IS / 20% OOS)")

    df_5m_is    = df_5m[df_5m.index <  split_date]
    df_5m_oos   = df_5m[df_5m.index >= split_date]
    df_daily_is = df_daily[df_daily.index <  split_date]
    df_daily_oos= df_daily[df_daily.index >= split_date]

    print(f"  In-sample   : {df_5m_is.index[0].date()} → {df_5m_is.index[-1].date()}  ({len(df_5m_is):,} bars)")
    print(f"  Out-of-sample: {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}  ({len(df_5m_oos):,} bars)")

    # Enrich IS
    print("\n  Enriching in-sample data...")
    news_is = get_high_impact_events(df_5m_is.index[0], df_5m_is.index[-1])
    df_5m_is = mark_news_windows(df_5m_is, news_is, pre_minutes=15, post_hours=2)
    df_5m_is = add_seasonal_columns(df_5m_is, asset="Gold")
    df_daily_is_enr, df_5m_is_enr = prepare_data(df_daily_is.copy(), df_5m_is.copy())
    if "cusum_up" not in df_5m_is_enr.columns:
        df_5m_is_enr = cusum_events(df_5m_is_enr)
    print(f"  News events (IS): {len(news_is)}")

    # Grid search
    print(f"\n{'='*65}")
    print("  IN-SAMPLE GRID SEARCH  (lnterqo variations, ML-filtered)")
    print(f"  ML threshold: {ML_THRESHOLD}  |  MC sims: {MC_N_SIMS:,}")
    print(f"{'='*65}")

    COL = 20
    hdr = (f"{'Variation':<{COL}} | {'Raw':>5} | {'ML%':>5} | {'Trades':>6} | "
           f"{'WR%':>6} | {'EV/R':>6} | {'MaxDD%':>7} | "
           f"{'FTMO%':>7} | {'The5%':>7} | {'Top%':>7} | {'PnL':>8}")
    sep = "-" * len(hdr)
    print(sep); print(hdr); print(sep)

    rows = []
    best_ml_filters = {}

    for idx, (label, min_rr, zone_lb, cisd_lb) in enumerate(VARIATIONS):
        print(f"  [{idx+1:02d}/{len(VARIATIONS)}] {label} ...", end="", flush=True)

        trades, equity, n_raw, ml_accept, ml_filter = _run_variation(
            label, min_rr, zone_lb, cisd_lb,
            df_daily_is_enr, df_5m_is_enr,
        )

        closed   = [t for t in trades if t.outcome in ("win", "loss")]
        n_trades = len(closed)
        metrics  = compute_metrics(trades, equity)

        if ml_filter:
            best_ml_filters[label] = ml_filter

        if not metrics or "message" in metrics or n_trades == 0:
            print(f"  → {n_raw} raw, 0 ML trades")
            rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                         "cisd_lb": cisd_lb, "n_raw": n_raw, "n_trades": 0,
                         "win_rate": None, "ev_r": None, "max_dd_pct": None,
                         "pnl": None, "the5ers": None, "ftmo_50k": None,
                         "topstep": None, "ml_accept": ml_accept})
            print()
            continue

        wr   = float(metrics["win_rate"].rstrip("%")) / 100.0
        ev_r = metrics.get("expectancy_r", 0)
        dd   = metrics.get("max_drawdown_pct", 0)
        pnl  = metrics.get("total_pnl_gbp", 0)

        pass_rates = {}
        if n_trades >= MIN_TRADES_MC:
            r_mults    = _r_multiples(trades)
            pass_rates = _pass_rates(r_mults, n_sims=min(MC_N_SIMS, 10_000))
        else:
            pass_rates = {k: None for k in _FIRM_KEYS}

        ml_str = f"{ml_accept*100:5.0f}%" if ml_accept is not None else "  N/A"
        print(
            f"\r  {label:<{COL}} | {n_raw:>5} | {ml_str:>5} | {n_trades:>6} | "
            f"{wr*100:>5.1f}% | {ev_r:>+6.3f} | {dd:>+7.2f}% | "
            f"{_fmt(pass_rates.get('FTMO_50k')):>7} | "
            f"{_fmt(pass_rates.get('The5ers_100k')):>7} | "
            f"{_fmt(pass_rates.get('TopStep_50k')):>7} | "
            f"{pnl:>+8.0f}"
        )

        rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                     "cisd_lb": cisd_lb, "n_raw": n_raw, "n_trades": n_trades,
                     "win_rate": wr, "ev_r": ev_r, "max_dd_pct": dd, "pnl": pnl,
                     "the5ers": pass_rates.get("The5ers_100k"),
                     "ftmo_50k": pass_rates.get("FTMO_50k"),
                     "topstep": pass_rates.get("TopStep_50k"),
                     "ml_accept": ml_accept})

    print(sep)

    valid  = [r for r in rows if r["n_trades"] >= MIN_TRADES_MC and r["win_rate"] is not None]
    if not valid:
        print("\n  No variation had enough trades. Exiting.")
        sys.exit(1)

    ranked = sorted(valid, key=lambda r: (r["the5ers"] or 0, r["win_rate"] or 0), reverse=True)

    print(f"\n{'='*65}")
    print("  IN-SAMPLE RANKING  (by The5%ers pass rate)")
    print(f"{'='*65}")
    rank_hdr = (f"{'#':>3}  {'Variation':<{COL}}  {'The5%':>7}  {'FTMO%':>7}  "
                f"{'WR%':>6}  {'EV/R':>6}  {'MaxDD%':>7}  {'Trades':>6}")
    print(rank_hdr)
    print("-" * len(rank_hdr))
    for ri, r in enumerate(ranked, 1):
        print(f"{ri:>3}  {r['label']:<{COL}}  "
              f"{_fmt(r['the5ers']):>7}  {_fmt(r['ftmo_50k']):>7}  "
              f"{r['win_rate']*100:>5.1f}%  {r['ev_r']:>+6.3f}  "
              f"{r['max_dd_pct']:>+7.2f}%  {r['n_trades']:>6}")

    best = ranked[0]
    print(f"\n  Best: [{best['label']}]  WR={best['win_rate']*100:.1f}%  "
          f"The5%={_fmt(best['the5ers'])}  Trades(IS)={best['n_trades']}")

    # ── OOS validation ────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OUT-OF-SAMPLE VALIDATION  [{best['label']}]")
    print(f"  {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
    print(f"  Frozen ML filter (trained on IS only).")
    print(f"{'='*65}")

    print("\n  Enriching OOS data...")
    news_oos = get_high_impact_events(df_5m_oos.index[0], df_5m_oos.index[-1])
    df_5m_oos = mark_news_windows(df_5m_oos, news_oos, pre_minutes=15, post_hours=2)
    df_5m_oos = add_seasonal_columns(df_5m_oos, asset="Gold")
    df_daily_oos_enr, df_5m_oos_enr = prepare_data(df_daily_oos.copy(), df_5m_oos.copy())
    if "cusum_up" not in df_5m_oos_enr.columns:
        df_5m_oos_enr = cusum_events(df_5m_oos_enr)

    frozen_ml = best_ml_filters.get(best["label"])

    trades_oos, equity_oos = run_lnterqo_backtest(
        "Gold", df_daily_oos_enr, df_5m_oos_enr,
        min_rr=best["min_rr"], zone_lookback=best["zone_lb"],
        cisd_lookback=best["cisd_lb"],
        force_neutral_bias=True,
        signal_filter=frozen_ml, df_enriched_for_filter=df_5m_oos_enr,
    )

    metrics_oos = compute_metrics(trades_oos, equity_oos)
    print_metrics(metrics_oos, asset="Gold — lnterqo (OOS)")

    closed_oos = [t for t in trades_oos if t.outcome in ("win", "loss")]
    r_mults_oos = np.array([t.r_multiple for t in closed_oos]) if closed_oos else np.array([])

    if trades_oos:
        df_trades = trades_to_dataframe(trades_oos)
        df_trades.to_csv("backtest/lnterqo_oos_trades.csv", index=False)
        print(f"\n  Trade log saved: backtest/lnterqo_oos_trades.csv")

    if len(closed_oos) < MIN_TRADES_MC:
        print(f"\n  Only {len(closed_oos)} OOS trades — need ≥{MIN_TRADES_MC}. Done.")
        return

    # Statistical validation
    print(f"\n  Running statistical validation (OOS)...")
    stat = validate(trades_oos, df_5m_oos_enr, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
    if stat:
        print(stat.report())
        if abs(stat.ev_per_trade) > 1e-6:
            n_needed = required_sample_size(delta=abs(stat.ev_per_trade), sigma=1.0)
            print(f"  Trades to confirm {abs(stat.ev_per_trade):.2f}R edge: {n_needed}")
            print(f"  Current OOS: {len(closed_oos)} — {'sufficient' if len(closed_oos) >= n_needed else 'INSUFFICIENT'}")

    # Regime analysis
    print(f"\n  Running regime analysis (OOS)...")
    df_regime = classify_regime(df_5m_oos_enr, adx_threshold=25.0, use_hurst=True)
    regime_series = df_regime["regime"].values
    trade_regimes = assign_trade_regimes(closed_oos, df_regime, regime_series)
    markov = None
    if len(set(trade_regimes)) > 1:
        markov = regime_markov_analysis(trade_regimes)
    for regime in sorted(set(trade_regimes)):
        mask = trade_regimes == regime
        if mask.sum() > 0 and len(r_mults_oos) > 0:
            print(f"  EV in {regime}: {np.mean(r_mults_oos[mask]):+.4f}R  ({mask.sum()} trades)")

    # Monte Carlo
    print(f"\n  Running Monte Carlo (OOS)  —  {MC_N_SIMS:,} simulations ...")

    mc1 = monte_carlo_reshuffle(r_mults_oos, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
    print(mc1.report())

    n_reg = len(set(trade_regimes))
    if markov and n_reg > 1 and len(r_mults_oos) >= n_reg:
        try:
            mc2 = monte_carlo_regime_switching(r_mults_oos, trade_regimes, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
            print(mc2.report())
        except (ValueError, ZeroDivisionError):
            pass

    mc3 = monte_carlo_barrier(
        r_mults_oos, initial_balance=INITIAL_EQUITY,
        profit_target_pct=0.20, max_drawdown_pct=0.20,
        challenge_fee=0.0, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT,
    )
    print(mc3.report())

    print(f"\n  Prop Firm Pass Rates (OOS, {MC_N_SIMS:,} sims):")
    run_prop_firm_simulations(r_mults_oos, r_to_pct=R_TO_PCT, n_sims=MC_N_SIMS)

    # Summary
    m = metrics_oos
    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — lnterqo OOS (unbiased)")
    print(f"{'='*65}")
    print(f"  Variation  : {best['label']}")
    print(f"  OOS period : {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
    print(f"  Trades     : {len(closed_oos)}")
    if m and "message" not in m:
        print(f"  Win rate   : {m.get('win_rate', 'N/A')}")
        print(f"  Expectancy : {m.get('expectancy_r', 0):+.3f}R")
        print(f"  Sharpe     : {m.get('sharpe_ratio', 'N/A')}")
        print(f"  Max DD     : {m.get('max_drawdown_pct', 'N/A')}%")
        print(f"  Total PnL  : £{m.get('total_pnl_gbp', 0):+,.0f}")
    print(f"  MC EV      : {mc1.ev_per_trade:+.4f}R")
    print(f"  MC med bal : £{INITIAL_EQUITY*(1+mc1.p50_terminal):,.0f}")
    print(f"  Barrier    : {mc3.pass_rate:.1%}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
