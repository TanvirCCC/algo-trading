"""
Out-of-Sample Validator
=======================
Runs the winning strategy config on held-out OOS data.

Workflow (proper walk-forward):
  1. Load OOS data (IB parquet: Aug 2024 → May 2026)
  2. Run raw backtest with winning config
  3. Train ML filter on first 70% of OOS trades (temporal split — no lookahead)
  4. Apply trained ML filter to ALL OOS trades
  5. Report unbiased metrics + prop firm pass rates

The strategy PARAMETERS (min_rr, lookback, etc.) were selected on in-sample
data. The ML filter is re-trained within OOS to avoid leakage.

Usage:
    python3 validate.py                    # default winning config
    python3 validate.py --min-rr 1.5      # override a parameter
"""

import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from data.ib_fetcher import load_parquet
from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.monte_carlo import monte_carlo_barrier, run_prop_firm_simulations, PROP_FIRM_CONFIGS
from strategy.ml_filter import SignalFilter
from strategy.ict_strategy import prepare_data

# ── Winning config (from in-sample ML-filtered optimiser) ─────────────────────
# LowRR+LB=80: 100% The5ers, 96.2% WR, -2% MaxDD, +2.746 EV/R
WINNING_CONFIG = dict(
    min_rr             = 1.5,
    zone_lookback      = 80,
    use_kill_zones     = True,
    use_news_filter    = True,
    zone_tolerance_atr = 0.3,
    ml_threshold       = 0.55,
)

MC_N_SIMS  = 100_000
R_TO_PCT   = 0.02


def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rr",    type=float, default=WINNING_CONFIG["min_rr"])
    parser.add_argument("--lookback",  type=int,   default=WINNING_CONFIG["zone_lookback"])
    parser.add_argument("--no-kz",     action="store_true")
    parser.add_argument("--no-news",   action="store_true")
    parser.add_argument("--tol",       type=float, default=WINNING_CONFIG["zone_tolerance_atr"])
    parser.add_argument("--threshold", type=float, default=WINNING_CONFIG["ml_threshold"])
    args = parser.parse_args()

    cfg = dict(
        min_rr             = args.min_rr,
        zone_lookback      = args.lookback,
        use_kill_zones     = not args.no_kz,
        use_news_filter    = not args.no_news,
        zone_tolerance_atr = args.tol,
        ml_threshold       = args.threshold,
    )

    print("=" * 62)
    print("  OUT-OF-SAMPLE VALIDATION — Gold IB Data".center(62))
    print("=" * 62)
    print(f"  Config : min_rr={cfg['min_rr']}  LB={cfg['zone_lookback']}  "
          f"KZ={cfg['use_kill_zones']}  News={cfg['use_news_filter']}  "
          f"Tol={cfg['zone_tolerance_atr']}  MLt={cfg['ml_threshold']}")

    # ── 1. Load OOS data (IB) ─────────────────────────────────────────────────
    print("\nLoading OOS data (IB)...")
    df_5m    = load_parquet("Gold", "5m")
    df_daily = load_parquet("Gold", "daily")
    print(f"  5m bars : {len(df_5m):,}  ({df_5m.index[0].date()} → {df_5m.index[-1].date()})")

    # ── 2. Enrich ─────────────────────────────────────────────────────────────
    print("Marking news windows...")
    news_events = get_high_impact_events(df_5m.index[0], df_5m.index[-1])
    df_5m = mark_news_windows(df_5m, news_events)
    print(f"  {len(news_events)} high-impact events")

    print("Adding seasonal columns...")
    df_5m = add_seasonal_columns(df_5m, asset="Gold")

    print("Pre-computing indicators for ML filter...")
    _, df_5m_enriched = prepare_data(df_daily.copy(), df_5m.copy())

    # ── 3. Raw backtest ───────────────────────────────────────────────────────
    print("\nRunning raw backtest...")
    trades_raw, equity_raw = run_backtest(
        "Gold", df_daily, df_5m,
        initial_equity = 10_000.0,
        risk_pct       = R_TO_PCT,
        min_rr         = cfg["min_rr"],
        use_kill_zones = cfg["use_kill_zones"],
        zone_lookback  = cfg["zone_lookback"],
        zone_tolerance_atr = cfg["zone_tolerance_atr"],
        use_news_filter    = cfg["use_news_filter"],
    )

    closed_raw = [t for t in trades_raw if t.outcome in ("win", "loss")]
    print(f"  Raw trades: {len(closed_raw)}")

    if len(closed_raw) < 10:
        print("  Too few trades for meaningful OOS validation.")
        return

    # ── 4. Train ML on first 70%, apply to all ────────────────────────────────
    print(f"\nTraining ML filter on first 70% of OOS trades "
          f"(threshold={cfg['ml_threshold']})...")
    ml_filter = SignalFilter(probability_threshold=cfg["ml_threshold"])
    ml_result = ml_filter.fit(trades_raw, df_5m_enriched, train_ratio=0.70)

    if ml_result:
        print(ml_result.report())

    # ── 5. ML-filtered backtest ───────────────────────────────────────────────
    print("\nRe-running with ML filter...")
    trades_ml, equity_ml = run_backtest(
        "Gold", df_daily, df_5m,
        initial_equity     = 10_000.0,
        risk_pct           = R_TO_PCT,
        min_rr             = cfg["min_rr"],
        use_kill_zones     = cfg["use_kill_zones"],
        zone_lookback      = cfg["zone_lookback"],
        zone_tolerance_atr = cfg["zone_tolerance_atr"],
        use_news_filter    = cfg["use_news_filter"],
        signal_filter      = ml_filter,
        df_enriched_for_filter = df_5m_enriched,
    )

    closed_ml = [t for t in trades_ml if t.outcome in ("win", "loss")]
    accept_pct = len(closed_ml) / len(closed_raw) * 100 if closed_raw else 0

    print(f"\n  ML accepted: {len(closed_ml)}/{len(closed_raw)} trades "
          f"({accept_pct:.0f}%)")

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  RAW STRATEGY  (unfiltered OOS)".center(62))
    print("=" * 62)
    metrics_raw = compute_metrics(trades_raw, equity_raw)
    if metrics_raw and "message" not in metrics_raw:
        for k, v in metrics_raw.items():
            print(f"  {k:<28} {v}")

    print("\n" + "=" * 62)
    print("  ML-FILTERED STRATEGY  (OOS)".center(62))
    print("=" * 62)
    metrics_ml = compute_metrics(trades_ml, equity_ml)
    if metrics_ml and "message" not in metrics_ml:
        for k, v in metrics_ml.items():
            print(f"  {k:<28} {v}")
    else:
        print("  Insufficient trades after ML filtering.")
        return

    # ── 7. Prop firm pass rates ───────────────────────────────────────────────
    print("\nRunning prop firm simulations (100k MC)...")
    r_mults = _r_multiples(trades_ml)
    run_prop_firm_simulations(r_mults, r_to_pct=R_TO_PCT, n_sims=MC_N_SIMS)


if __name__ == "__main__":
    main()
