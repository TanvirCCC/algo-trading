"""
Gold Walk-Forward Validation — 80/20 In-Sample / Out-of-Sample
==============================================================
Data  : data/historical/gold_5m_combined.parquet  (2016–2026, 713k bars)
Split : first 80% of date range = in-sample (parameter search + ML train)
        last  20% = out-of-sample (frozen params + frozen ML, unbiased eval)

Workflow per variation (in-sample):
  1. Raw backtest
  2. Train Random Forest ML filter on raw trades (temporal 70/30 split within IS)
  3. Re-run with ML filter → IS metrics

Then:
  4. Pick best variation by The5ers pass rate (tiebreak: win rate)
  5. Re-run best config on OOS data with frozen ML weights
  6. 100k Monte Carlo on OOS trades (reshuffle + regime + barrier)
  7. Print full OOS report
"""

import warnings
import sys, os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from detectors.regime import classify_regime, regime_markov_analysis, cusum_events
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics, print_metrics, trades_to_dataframe, report_summary
from backtest.statistics import validate, required_sample_size
from backtest.monte_carlo import (
    monte_carlo_reshuffle,
    monte_carlo_regime_switching,
    monte_carlo_barrier,
    assign_trade_regimes,
    run_prop_firm_simulations,
    PROP_FIRM_CONFIGS,
)
from strategy.ml_filter import SignalFilter
from strategy.ict_strategy import prepare_data

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR          = Path("data/historical")
COMBINED_5M       = DATA_DIR / "gold_5m_combined.parquet"
IN_SAMPLE_RATIO   = 0.80
MC_N_SIMS         = 100_000
ML_THRESHOLD      = 0.65
R_TO_PCT          = 0.005  # 1R = 0.5% equity (default risk; quality trades use 1% via RiskManager)
MIN_TRADES_FOR_ML = 15
MIN_TRADES_FOR_MC = 3
INITIAL_EQUITY    = 10_000.0

# Strategy variations: (Label, min_rr, zone_lookback, use_kill_zones, use_news, zone_tolerance_atr)
VARIATIONS = [
    ("3.0+LB=200",         3.0, 200, True,  True,  0.3),
    ("3.0+LB=150",         3.0, 150, True,  True,  0.3),
    ("3.0+LB=300",         3.0, 300, True,  True,  0.3),
]

_FIRM_KEYS = ["FTMO_50k", "The5ers_100k", "TopStep_50k"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def _pass_rates(r_mults: np.ndarray, n_sims: int) -> dict:
    results = {}
    for key in _FIRM_KEYS:
        cfg = PROP_FIRM_CONFIGS[key]
        br = monte_carlo_barrier(
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


def _fmt_pct(v):
    return f"{v*100:6.1f}%" if v is not None else "   N/A"


def _run_variation(label, min_rr, lookback, use_kz, use_news, tol,
                   df_daily_enriched, df_5m_enriched, df_5m_enriched_for_ml):
    trades_raw, equity_raw = run_backtest(
        "Gold", df_daily_enriched, df_5m_enriched,
        initial_equity=INITIAL_EQUITY, risk_pct=R_TO_PCT,
        min_rr=min_rr, use_kill_zones=use_kz,
        zone_lookback=lookback, zone_tolerance_atr=tol,
        use_news_filter=use_news,
        force_neutral_bias=True,
    )
    n_raw = len([t for t in trades_raw if t.outcome in ("win", "loss")])

    if n_raw < MIN_TRADES_FOR_ML:
        return trades_raw, equity_raw, n_raw, None, None

    ml_filter = SignalFilter(probability_threshold=ML_THRESHOLD)
    ml_result = ml_filter.fit(trades_raw, df_5m_enriched_for_ml)
    if ml_result is None:
        return trades_raw, equity_raw, n_raw, None, None

    trades_ml, equity_ml = run_backtest(
        "Gold", df_daily_enriched, df_5m_enriched,
        initial_equity=INITIAL_EQUITY, risk_pct=R_TO_PCT,
        min_rr=min_rr, use_kill_zones=use_kz,
        zone_lookback=lookback, zone_tolerance_atr=tol,
        use_news_filter=use_news,
        force_neutral_bias=True,
        signal_filter=ml_filter,
        df_enriched_for_filter=df_5m_enriched_for_ml,
    )
    return trades_ml, equity_ml, n_raw, ml_result.acceptance_rate, ml_filter


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load combined data ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  GOLD WALK-FORWARD VALIDATION  (80/20 IS/OOS)")
    print(f"  Data: {COMBINED_5M.name}")
    print(f"{'='*65}")

    if not COMBINED_5M.exists():
        print(f"ERROR: {COMBINED_5M} not found.")
        sys.exit(1)

    df_5m_full = pd.read_parquet(COMBINED_5M)
    df_5m_full.columns = [c.lower() for c in df_5m_full.columns]
    df_5m_full = df_5m_full[["open", "high", "low", "close", "volume"]]
    # full dataset — no date restriction

    # Derive daily from 5m
    df_daily_full = (
        df_5m_full.resample("D")
        .agg(open=("open","first"), high=("high","max"),
             low=("low","min"), close=("close","last"), volume=("volume","sum"))
        .dropna(subset=["open"])
    )

    total_start = df_5m_full.index[0]
    total_end   = df_5m_full.index[-1]
    total_days  = (total_end - total_start).days
    split_date  = total_start + timedelta(days=int(total_days * IN_SAMPLE_RATIO))

    print(f"\n  Full range  : {total_start.date()} → {total_end.date()}  ({len(df_5m_full):,} bars)")
    print(f"  Split date  : {split_date.date()}  (80% IS / 20% OOS)")

    # ── 2. Split ─────────────────────────────────────────────────────────────
    df_5m_is    = df_5m_full[df_5m_full.index <  split_date]
    df_5m_oos   = df_5m_full[df_5m_full.index >= split_date]
    df_daily_is = df_daily_full[df_daily_full.index <  split_date]
    df_daily_oos= df_daily_full[df_daily_full.index >= split_date]

    print(f"  In-sample   : {df_5m_is.index[0].date()} → {df_5m_is.index[-1].date()}  ({len(df_5m_is):,} bars)")
    print(f"  Out-of-sample: {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}  ({len(df_5m_oos):,} bars)")

    # ── 3. Enrich in-sample data ─────────────────────────────────────────────
    print("\n  Enriching in-sample data (news, seasonality, indicators)...")
    news_is = get_high_impact_events(df_5m_is.index[0], df_5m_is.index[-1])
    df_5m_is = mark_news_windows(df_5m_is, news_is, pre_minutes=10, post_hours=2)
    df_5m_is = add_seasonal_columns(df_5m_is, asset="Gold")
    df_daily_is_enriched, df_5m_is_enriched = prepare_data(df_daily_is.copy(), df_5m_is.copy())
    if "cusum_up" not in df_5m_is_enriched.columns:
        df_5m_is_enriched = cusum_events(df_5m_is_enriched)
    print(f"  News events (IS): {len(news_is)}")

    # ── 4. Grid search on in-sample ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  IN-SAMPLE GRID SEARCH  (13 variations, ML-filtered)")
    print(f"  MC sims: {MC_N_SIMS:,}  |  ML threshold: {ML_THRESHOLD}")
    print(f"{'='*65}")

    COL = 20
    hdr = (f"{'Variation':<{COL}} | {'Raw':>5} | {'ML%':>5} | {'Trades':>6} | "
           f"{'WR%':>6} | {'EV/R':>6} | {'MaxDD%':>7} | "
           f"{'FTMO%':>7} | {'The5%':>7} | {'Top%':>7} | {'PnL':>8}")
    sep = "-" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    rows = []
    best_ml_filters = {}  # label → trained SignalFilter

    for i, (label, min_rr, lookback, use_kz, use_news, tol) in enumerate(VARIATIONS):
        print(f"  [{i+1:02d}/{len(VARIATIONS)}] {label} ...", end="", flush=True)

        trades, equity, n_raw, ml_accept, ml_filter = _run_variation(
            label, min_rr, lookback, use_kz, use_news, tol,
            df_daily_is_enriched, df_5m_is_enriched, df_5m_is_enriched
        )

        closed  = [t for t in trades if t.outcome in ("win", "loss")]
        n_trades = len(closed)
        metrics  = compute_metrics(trades, equity)

        if ml_filter:
            best_ml_filters[label] = ml_filter

        if not metrics or "message" in metrics or n_trades == 0:
            print(f"  → {n_raw} raw, 0 ML trades")
            rows.append({"label": label, "min_rr": min_rr, "lookback": lookback,
                         "use_kz": use_kz, "use_news": use_news, "tol": tol,
                         "n_raw": n_raw, "n_trades": 0, "win_rate": None,
                         "ev_r": None, "max_dd_pct": None, "pnl": None,
                         "the5ers": None, "ftmo_50k": None, "topstep": None,
                         "ml_accept": ml_accept})
            print()
            continue

        wr   = float(metrics["win_rate"].rstrip("%")) / 100.0
        ev_r = metrics.get("expectancy_r", 0)
        dd   = metrics.get("max_drawdown_pct", 0)
        pnl  = metrics.get("total_pnl_gbp", 0)

        pass_rates = {}
        if n_trades >= MIN_TRADES_FOR_MC:
            r_mults    = _r_multiples(trades)
            pass_rates = _pass_rates(r_mults, n_sims=min(MC_N_SIMS, 10_000))
        else:
            pass_rates = {k: None for k in _FIRM_KEYS}

        ml_str = f"{ml_accept*100:5.0f}%" if ml_accept is not None else "  N/A"
        print(
            f"\r  {label:<{COL}} | {n_raw:>5} | {ml_str:>5} | {n_trades:>6} | "
            f"{wr*100:>5.1f}% | {ev_r:>+6.3f} | {dd:>+7.2f}% | "
            f"{_fmt_pct(pass_rates.get('FTMO_50k')):>7} | "
            f"{_fmt_pct(pass_rates.get('The5ers_100k')):>7} | "
            f"{_fmt_pct(pass_rates.get('TopStep_50k')):>7} | "
            f"{pnl:>+8.0f}"
        )

        rows.append({"label": label, "min_rr": min_rr, "lookback": lookback,
                     "use_kz": use_kz, "use_news": use_news, "tol": tol,
                     "n_raw": n_raw, "n_trades": n_trades,
                     "win_rate": wr, "ev_r": ev_r, "max_dd_pct": dd, "pnl": pnl,
                     "the5ers": pass_rates.get("The5ers_100k"),
                     "ftmo_50k": pass_rates.get("FTMO_50k"),
                     "topstep": pass_rates.get("TopStep_50k"),
                     "ml_accept": ml_accept})

    print(sep)

    # ── 5. Rank by The5ers, tiebreak win rate ────────────────────────────────
    valid = [r for r in rows if r["n_trades"] >= MIN_TRADES_FOR_MC and r["win_rate"] is not None]

    if not valid:
        print("\n  No variation had enough trades. Exiting.")
        sys.exit(1)

    ranked = sorted(valid, key=lambda r: (r["the5ers"] or 0, r["win_rate"] or 0), reverse=True)

    print(f"\n{'='*65}")
    print("  IN-SAMPLE RANKING  (by The5%ers pass rate, tiebreak: win rate)")
    print(f"{'='*65}")
    rank_hdr = (f"{'#':>3}  {'Variation':<{COL}}  {'The5%':>7}  {'FTMO%':>7}  "
                f"{'WR%':>6}  {'EV/R':>6}  {'MaxDD%':>7}  {'Trades':>6}")
    print(rank_hdr)
    print("-" * len(rank_hdr))
    for ri, r in enumerate(ranked, 1):
        print(f"{ri:>3}  {r['label']:<{COL}}  "
              f"{_fmt_pct(r['the5ers']):>7}  {_fmt_pct(r['ftmo_50k']):>7}  "
              f"{r['win_rate']*100:>5.1f}%  {r['ev_r']:>+6.3f}  "
              f"{r['max_dd_pct']:>+7.2f}%  {r['n_trades']:>6}")

    best = ranked[0]
    print(f"\n  Best config: [{best['label']}]  "
          f"WR={best['win_rate']*100:.1f}%  The5%={_fmt_pct(best['the5ers'])}  "
          f"Trades(IS)={best['n_trades']}")

    # ── 6. Out-of-sample validation ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  OUT-OF-SAMPLE VALIDATION  [{best['label']}]")
    print(f"  {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
    print(f"  Params frozen from IS. ML filter trained on IS trades only.")
    print(f"{'='*65}")

    # Enrich OOS data
    print("\n  Enriching OOS data...")
    news_oos = get_high_impact_events(df_5m_oos.index[0], df_5m_oos.index[-1])
    df_5m_oos = mark_news_windows(df_5m_oos, news_oos, pre_minutes=10, post_hours=2)
    df_5m_oos = add_seasonal_columns(df_5m_oos, asset="Gold")
    df_daily_oos_enriched, df_5m_oos_enriched = prepare_data(df_daily_oos.copy(), df_5m_oos.copy())
    if "cusum_up" not in df_5m_oos_enriched.columns:
        df_5m_oos_enriched = cusum_events(df_5m_oos_enriched)

    # Use frozen ML filter from IS (trained on in-sample trades)
    frozen_ml = best_ml_filters.get(best["label"])

    trades_oos, equity_oos = run_backtest(
        "Gold", df_daily_oos_enriched, df_5m_oos_enriched,
        initial_equity=INITIAL_EQUITY, risk_pct=R_TO_PCT,
        min_rr=best["min_rr"], use_kill_zones=best["use_kz"],
        zone_lookback=best["lookback"], zone_tolerance_atr=best["tol"],
        use_news_filter=best["use_news"],
        force_neutral_bias=True,
        signal_filter=frozen_ml,
        df_enriched_for_filter=df_5m_oos_enriched,
    )

    metrics_oos = compute_metrics(trades_oos, equity_oos)
    print_metrics(metrics_oos, asset="Gold (OOS)")

    closed_oos = [t for t in trades_oos if t.outcome in ("win", "loss")]
    r_mults_oos = np.array([t.r_multiple for t in closed_oos]) if closed_oos else np.array([])

    if trades_oos:
        df_trades_oos = trades_to_dataframe(trades_oos)
        df_trades_oos.to_csv("backtest/gold_wfv_oos_trades.csv", index=False)
        print(f"\n  OOS trade log saved: backtest/gold_wfv_oos_trades.csv")

    if len(closed_oos) < MIN_TRADES_FOR_MC:
        print(f"\n  Only {len(closed_oos)} OOS trades — need ≥{MIN_TRADES_FOR_MC} for MC. Done.")
        return

    # ── 7. Statistical validation (OOS) ─────────────────────────────────────
    print(f"\n  Running statistical validation (OOS)...")
    stat_oos = validate(trades_oos, df_5m_oos_enriched,
                        confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
    if stat_oos:
        print(stat_oos.report())
        if abs(stat_oos.ev_per_trade) > 1e-6:
            n_needed = required_sample_size(delta=abs(stat_oos.ev_per_trade), sigma=1.0)
            print(f"  Trades to confirm {abs(stat_oos.ev_per_trade):.2f}R edge: {n_needed}")
            print(f"  Current OOS: {len(closed_oos)} — {'sufficient' if len(closed_oos) >= n_needed else 'INSUFFICIENT'}")

    # ── 8. Regime analysis (OOS) ─────────────────────────────────────────────
    print(f"\n  Running regime analysis (OOS)...")
    df_regime_oos  = classify_regime(df_5m_oos_enriched, adx_threshold=25.0, use_hurst=True)
    regime_series  = df_regime_oos["regime"].values
    trade_regimes  = assign_trade_regimes(closed_oos, df_regime_oos, regime_series)

    markov = None
    if len(set(trade_regimes)) > 1:
        markov = regime_markov_analysis(trade_regimes)
    else:
        label_r = trade_regimes[0] if len(trade_regimes) > 0 else "N/A"
        print(f"  All OOS trades in single regime: {label_r}")

    for regime in sorted(set(trade_regimes)):
        mask = trade_regimes == regime
        if mask.sum() > 0 and len(r_mults_oos) > 0:
            print(f"  EV in {regime}: {np.mean(r_mults_oos[mask]):+.4f}R  ({mask.sum()} trades)")

    # ── 9. Monte Carlo — 100k sims (OOS) ────────────────────────────────────
    print(f"\n  Running Monte Carlo (OOS)  —  {MC_N_SIMS:,} simulations ...")

    # Method 1: Reshuffling
    mc1 = monte_carlo_reshuffle(r_mults_oos, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
    print(mc1.report())

    # Method 2: Regime-switching (if multiple regimes)
    n_reg = len(set(trade_regimes))
    if markov and n_reg > 1 and len(r_mults_oos) >= n_reg:
        try:
            mc2 = monte_carlo_regime_switching(
                r_mults_oos, trade_regimes, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT
            )
            print(mc2.report())
        except (ValueError, ZeroDivisionError):
            print("  Regime-switching MC skipped (degenerate matrix).")

    # Method 3: Barrier — Trading Game account (£10k)
    mc3 = monte_carlo_barrier(
        r_mults_oos,
        initial_balance=INITIAL_EQUITY,
        profit_target_pct=0.20,
        max_drawdown_pct=0.20,
        challenge_fee=0.0,
        n_sims=MC_N_SIMS,
        r_to_pct=R_TO_PCT,
    )
    print(mc3.report())

    # Prop firm pass rates (100k sims)
    print(f"\n  Prop Firm Pass Rates (OOS, {MC_N_SIMS:,} sims):")
    run_prop_firm_simulations(r_mults_oos, r_to_pct=R_TO_PCT, n_sims=MC_N_SIMS)

    # ── 10. Save report paragraph ────────────────────────────────────────────
    m = metrics_oos
    report = report_summary(trades_oos, equity_oos, "Gold")
    rpt_path = "backtest/gold_wfv_oos_report.txt"
    with open(rpt_path, "w") as f:
        f.write(f"GOLD — WALK-FORWARD VALIDATION REPORT\n")
        f.write(f"Configuration: {best['label']}\n")
        f.write(f"In-sample  : {df_5m_is.index[0].date()} → {df_5m_is.index[-1].date()}\n")
        f.write(f"Out-of-sample: {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}\n")
        f.write(f"ML filter: trained on IS trades only (P(win) ≥ {ML_THRESHOLD})\n\n")
        f.write(report)
        if stat_oos:
            f.write(f"\n\nStatistical Validation (OOS):\n")
            f.write(f"Bootstrap 95% CI on EV: [{stat_oos.ci_lower:+.4f}R, {stat_oos.ci_upper:+.4f}R]\n")
            f.write(f"Null model p-value: {stat_oos.p_value:.4f}\n")
            f.write(f"Information Coefficient: {stat_oos.ic:.4f}\n")
            f.write(f"Timing {'IS' if stat_oos.timing_significant else 'IS NOT'} significant vs random entry.\n")
        f.write(f"\nMonte Carlo (OOS, {MC_N_SIMS:,} sims):\n")
        f.write(f"  Reshuffle: EV={mc1.ev_per_trade:+.4f}R  "
                f"Median balance: £{INITIAL_EQUITY*(1+mc1.p50_terminal):,.0f}  "
                f"Profitable paths: {mc1.pct_profitable:.1%}\n")
        f.write(f"  Barrier pass rate: {mc3.pass_rate:.1%}  "
                f"Net EV per attempt: £{mc3.net_ev:+,.0f}\n")
    print(f"\n  Report saved: {rpt_path}")

    print(f"\n{'='*65}")
    print("  FINAL SUMMARY — OOS (unbiased)")
    print(f"{'='*65}")
    print(f"  Variation     : {best['label']}")
    print(f"  OOS period    : {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
    print(f"  Trades        : {len(closed_oos)}")
    if m and "message" not in m:
        print(f"  Win rate      : {m.get('win_rate', 'N/A')}")
        print(f"  Expectancy    : {m.get('expectancy_r', 0):+.3f}R")
        print(f"  Sharpe        : {m.get('sharpe_ratio', 'N/A')}")
        print(f"  Max drawdown  : {m.get('max_drawdown_pct', 'N/A')}%")
        print(f"  Total PnL     : £{m.get('total_pnl_gbp', 0):+,.0f}")
    print(f"  MC EV (OOS)   : {mc1.ev_per_trade:+.4f}R")
    print(f"  MC median bal : £{INITIAL_EQUITY*(1+mc1.p50_terminal):,.0f}")
    print(f"  Barrier pass  : {mc3.pass_rate:.1%}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
