"""
Grid Search Parameter Optimiser — Gold (ML-filtered)
=====================================================
For each variation:
  1. Run raw backtest
  2. Train Random Forest signal filter on raw trades (70/30 temporal split)
  3. Re-run backtest with ML filter applied
  4. Report ML-filtered metrics + prop firm pass rates (10k MC sims)

Usage:
    python3 optimize.py                        # IB data (default)
    python3 optimize.py --source dukascopy     # Dukascopy in-sample data
    python3 optimize.py --source dukascopy --start 2016-01-01 --end 2023-12-31
"""

import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from data.ib_fetcher import load_parquet
from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.monte_carlo import monte_carlo_barrier, PROP_FIRM_CONFIGS
from strategy.ml_filter import SignalFilter
from strategy.ict_strategy import prepare_data

# ── Variations ────────────────────────────────────────────────────────────────
# Each tuple: (Label, min_rr, zone_lookback, use_kill_zones, use_news_filter, zone_tolerance_atr)
VARIATIONS = [
    # Label,              min_rr, lookback, kill_zones, news,  tolerance
    # ── Already run (kept for reference) ──────────────────────────────────
    ("Baseline",           2.0,   200,      True,       True,  0.3),
    ("minRR=2.5",          2.5,   200,      True,       True,  0.3),
    ("minRR=3.0",          3.0,   200,      True,       True,  0.3),
    ("LB=150",             2.0,   150,      True,       True,  0.3),
    ("LB=300",             2.0,   300,      True,       True,  0.3),
    # ── High-value combos suggested by 8-year data ─────────────────────────
    ("2.5+LB=150",         2.5,   150,      True,       True,  0.3),
    ("2.5+LB=300",         2.5,   300,      True,       True,  0.3),
    ("3.0+LB=150",         3.0,   150,      True,       True,  0.3),
    ("3.0+LB=300",         3.0,   300,      True,       True,  0.3),
    ("2.0+Tol=0.5",        2.0,   200,      True,       True,  0.5),
    ("2.5+Tol=0.5",        2.5,   200,      True,       True,  0.5),
    ("NoNews+LB=300",      2.0,   300,      True,       False, 0.3),
    ("2.5+LB=300+NoNews",  2.5,   300,      True,       False, 0.3),
]

# Prop firm keys
_FIRM_KEYS = ["FTMO_50k", "The5ers_100k", "TopStep_50k"]

MIN_TRADES_FOR_MC  = 3
MIN_TRADES_FOR_ML  = 15     # need at least this many raw trades to train RF
MC_N_SIMS          = 10_000  # 10k sims — ±1% accuracy, fast
ML_THRESHOLD       = 0.55   # P(win) cutoff for accepting a signal
R_TO_PCT           = 0.02   # 1R = 2% of account


# ── Helpers ───────────────────────────────────────────────────────────────────

def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def _compute_pass_rates(r_mults: np.ndarray) -> dict[str, float]:
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
            n_sims=MC_N_SIMS,
        )
        results[key] = br.pass_rate
    return results


def _fmt_pct(val) -> str:
    if val is None:
        return "  N/A  "
    return f"{val * 100:6.1f}%"


def _fmt_float(val, fmt=".3f") -> str:
    if val is None:
        return "  N/A "
    return format(val, fmt)


def _run_variation(label, min_rr, lookback, use_kz, use_news, tol, df_daily, df_5m, df_5m_enriched):
    """
    Run one variation: raw backtest → train ML → re-run with filter.
    df_5m_enriched: pre-computed df with RSI/CCI/ATR columns for ML feature extraction.
    Returns (trades_final, equity_final, n_raw_trades, ml_accept_pct).
    Falls back to raw trades if ML can't be trained.
    """
    # Step 1: raw backtest
    trades_raw, equity_raw = run_backtest(
        "Gold", df_daily, df_5m,
        initial_equity=10_000.0,
        risk_pct=R_TO_PCT,
        min_rr=min_rr,
        use_kill_zones=use_kz,
        zone_lookback=lookback,
        zone_tolerance_atr=tol,
        use_news_filter=use_news,
    )

    n_raw = len([t for t in trades_raw if t.outcome in ("win", "loss")])

    if n_raw < MIN_TRADES_FOR_ML:
        return trades_raw, equity_raw, n_raw, None

    # Step 2: train ML filter using enriched df (has RSI, CCI, ATR etc.)
    ml_filter = SignalFilter(probability_threshold=ML_THRESHOLD)
    ml_result = ml_filter.fit(trades_raw, df_5m_enriched)

    if ml_result is None:
        return trades_raw, equity_raw, n_raw, None

    # Step 3: re-run with ML filter — pass enriched df so accept() can read RSI/CCI etc.
    trades_ml, equity_ml = run_backtest(
        "Gold", df_daily, df_5m,
        initial_equity=10_000.0,
        risk_pct=R_TO_PCT,
        min_rr=min_rr,
        signal_filter=ml_filter,
        use_kill_zones=use_kz,
        zone_lookback=lookback,
        zone_tolerance_atr=tol,
        use_news_filter=use_news,
        df_enriched_for_filter=df_5m_enriched,
    )

    ml_accept_pct = ml_result.acceptance_rate
    return trades_ml, equity_ml, n_raw, ml_accept_pct


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_data(source: str, start: str | None, end: str | None):
    """Load 5m + daily data from IB or Dukascopy, with optional date clipping."""
    DATA_DIR = Path("data/historical")

    if source == "dukascopy":
        p5m    = DATA_DIR / "gold_5m_dukascopy_insample.parquet"
        pdaily = DATA_DIR / "gold_daily_dukascopy_insample.parquet"
        if not p5m.exists():
            raise FileNotFoundError(
                f"Dukascopy data not found at {p5m}.\n"
                "Run: python3 data/dukascopy_fetcher.py"
            )
        df_5m    = pd.read_parquet(p5m)
        df_daily = pd.read_parquet(pdaily)
    else:
        df_5m    = load_parquet("Gold", "5m")
        df_daily = load_parquet("Gold", "daily")

    if start:
        df_5m    = df_5m[df_5m.index >= start]
        df_daily = df_daily[df_daily.index >= start]
    if end:
        df_5m    = df_5m[df_5m.index <= end]
        df_daily = df_daily[df_daily.index <= end]

    return df_5m, df_daily


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["ib", "dukascopy"], default="ib",
                        help="Data source: 'ib' (default) or 'dukascopy'")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"Loading Gold data (source={args.source})...")
    df_5m, df_daily = _load_data(args.source, args.start, args.end)
    print(f"  5m bars   : {len(df_5m):,}  ({df_5m.index[0].date()} → {df_5m.index[-1].date()})")
    print(f"  Daily bars: {len(df_daily):,}  ({df_daily.index[0].date()} → {df_daily.index[-1].date()})")

    # ── 2. News windows ───────────────────────────────────────────────────────
    print("Fetching high-impact news events...")
    news_events = get_high_impact_events(df_5m.index[0], df_5m.index[-1])
    print(f"  Found {len(news_events)} high-impact events")
    df_5m = mark_news_windows(df_5m, news_events)

    # ── 3. Seasonal columns ───────────────────────────────────────────────────
    print("Adding seasonal columns...")
    df_5m = add_seasonal_columns(df_5m, asset="Gold")

    # ── 4. Pre-enrich df for ML feature extraction (RSI, CCI, ATR etc.) ──────
    print("Pre-computing technical indicators for ML filter...")
    _, df_5m_enriched = prepare_data(df_daily.copy(), df_5m.copy())
    print(f"  MC sims per firm : {MC_N_SIMS:,}  |  ML threshold : {ML_THRESHOLD}")
    print()

    # ── 4. Grid search ────────────────────────────────────────────────────────
    col_w = 18

    header = (
        f"{'Label':<{col_w}} | "
        f"{'Raw':>5} | "
        f"{'ML%':>5} | "
        f"{'Trades':>6} | "
        f"{'WR%':>5} | "
        f"{'EV/R':>6} | "
        f"{'MaxDD%':>7} | "
        f"{'FTMO50k%':>9} | "
        f"{'The5ers%':>9} | "
        f"{'TopStep%':>9} | "
        f"{'PnL':>8}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    rows = []

    for i, var in enumerate(VARIATIONS):
        label, min_rr, lookback, use_kz, use_news, tol = var
        print(f"[{i+1}/{len(VARIATIONS)}] {label} ...")

        trades, equity_curve, n_raw, ml_accept_pct = _run_variation(
            label, min_rr, lookback, use_kz, use_news, tol, df_daily, df_5m, df_5m_enriched
        )

        metrics = compute_metrics(trades, equity_curve)
        closed  = [t for t in trades if t.outcome in ("win", "loss")]
        n_trades = len(closed)

        raw_str = f"{n_raw:>5}"
        ml_str  = f"{ml_accept_pct*100:5.0f}%" if ml_accept_pct is not None else "  N/A"

        if not metrics or "message" in metrics or n_trades == 0:
            wr_str  = "  N/A"
            ev_str  = "  N/A"
            dd_str  = "  N/A"
            pnl_str = "  N/A"
            pass_rates = {k: None for k in _FIRM_KEYS}
        else:
            wr_val  = float(metrics["win_rate"].rstrip("%")) / 100.0
            wr_str  = f"{wr_val * 100:5.1f}%"
            ev_str  = f"{metrics['expectancy_r']:+6.3f}"
            dd_str  = f"{metrics['max_drawdown_pct']:+7.2f}%"
            pnl_str = f"{metrics['total_pnl_gbp']:+8.0f}"

            if n_trades >= MIN_TRADES_FOR_MC:
                r_mults    = _r_multiples(trades)
                pass_rates = _compute_pass_rates(r_mults)
            else:
                pass_rates = {k: None for k in _FIRM_KEYS}

        ftmo_str    = _fmt_pct(pass_rates.get("FTMO_50k"))
        the5_str    = _fmt_pct(pass_rates.get("The5ers_100k"))
        topstep_str = _fmt_pct(pass_rates.get("TopStep_50k"))

        row_str = (
            f"{label:<{col_w}} | "
            f"{raw_str:>5} | "
            f"{ml_str:>5} | "
            f"{n_trades:>6} | "
            f"{wr_str:>5} | "
            f"{ev_str:>6} | "
            f"{dd_str:>7} | "
            f"{ftmo_str:>9} | "
            f"{the5_str:>9} | "
            f"{topstep_str:>9} | "
            f"{pnl_str:>8}"
        )
        print(row_str)

        rows.append({
            "label":        label,
            "n_raw":        n_raw,
            "n_trades":     n_trades,
            "ml_accept":    ml_accept_pct,
            "win_rate":     float(metrics.get("win_rate", "0%").rstrip("%")) / 100.0 if metrics and "message" not in metrics else None,
            "ev_r":         metrics.get("expectancy_r") if metrics and "message" not in metrics else None,
            "max_dd_pct":   metrics.get("max_drawdown_pct") if metrics and "message" not in metrics else None,
            "pnl":          metrics.get("total_pnl_gbp") if metrics and "message" not in metrics else None,
            "ftmo_50k":     pass_rates.get("FTMO_50k"),
            "the5ers":      pass_rates.get("The5ers_100k"),
            "topstep":      pass_rates.get("TopStep_50k"),
        })

    print(sep)
    print()

    # ── 5. Ranking by The5ers pass rate ───────────────────────────────────────
    ranked = sorted(
        [r for r in rows if r["the5ers"] is not None],
        key=lambda r: r["the5ers"],
        reverse=True,
    )

    if ranked:
        print("=" * len(sep))
        print("  RANKING BY The5%ers PASS RATE — ML FILTERED (desc)".center(len(sep)))
        print("=" * len(sep))
        rank_header = (
            f"{'Rank':>4}  {'Label':<{col_w}}  "
            f"{'The5ers%':>9}  {'FTMO50k%':>9}  "
            f"{'TopStep%':>9}  {'WR%':>5}  "
            f"{'EV/R':>6}  {'MaxDD%':>7}  {'ML%':>5}  {'Trades':>6}  {'PnL':>8}"
        )
        print(rank_header)
        print("-" * len(rank_header))
        for rank_i, r in enumerate(ranked, 1):
            ml_col = f"{r['ml_accept']*100:5.0f}%" if r["ml_accept"] is not None else "  N/A"
            print(
                f"{rank_i:>4}  {r['label']:<{col_w}}  "
                f"{_fmt_pct(r['the5ers']):>9}  "
                f"{_fmt_pct(r['ftmo_50k']):>9}  "
                f"{_fmt_pct(r['topstep']):>9}  "
                f"{r['win_rate'] * 100:5.1f}%  "
                f"{r['ev_r']:+6.3f}  "
                f"{r['max_dd_pct']:+7.2f}%  "
                f"{ml_col:>5}  "
                f"{r['n_trades']:>6}  "
                f"{r['pnl']:+8.0f}"
            )
        print()
    else:
        print("  No variations had enough trades for Monte Carlo ranking.")
        print()

    # ── 6. Winner summary ─────────────────────────────────────────────────────
    valid_rows = [r for r in rows if r["n_trades"] >= MIN_TRADES_FOR_MC and r["win_rate"] is not None]

    print("=" * 60)
    print("  WINNER SUMMARY  (ML-filtered)".center(60))
    print("=" * 60)

    if not valid_rows:
        print("  Insufficient data.")
    else:
        best_wr   = max(valid_rows, key=lambda r: r["win_rate"])
        rows_mc   = [r for r in valid_rows if r["the5ers"] is not None]
        best_the5 = max(rows_mc, key=lambda r: r["the5ers"]) if rows_mc else None
        best_dd   = min(valid_rows, key=lambda r: abs(r["max_dd_pct"]))
        best_ev   = max(valid_rows, key=lambda r: r["ev_r"] if r["ev_r"] is not None else -999)

        print(f"  Best Win Rate    : {best_wr['label']:<{col_w}}  WR={best_wr['win_rate']*100:.1f}%  Trades={best_wr['n_trades']}")
        if best_the5:
            print(f"  Best The5%ers    : {best_the5['label']:<{col_w}}  Pass={best_the5['the5ers']*100:.1f}%  Trades={best_the5['n_trades']}")
        print(f"  Best Drawdown    : {best_dd['label']:<{col_w}}  MaxDD={best_dd['max_dd_pct']:.2f}%  Trades={best_dd['n_trades']}")
        print(f"  Best EV/R        : {best_ev['label']:<{col_w}}  EV={best_ev['ev_r']:+.3f}R  Trades={best_ev['n_trades']}")

    print("=" * 60)


if __name__ == "__main__":
    main()
