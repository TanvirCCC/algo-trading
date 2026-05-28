"""
Oil (Brent) ICT Strategy Backtest
===================================
Runs the ICT FVG/OB strategy on Brent Crude 5m histdata (2010-2024)
extended with yfinance 1h for 2025-2026.

Uses same approach as Gold optimizer — grid search + ML filter + MC prop firm sims.

Usage:
    python3 backtest_oil.py
    python3 backtest_oil.py --start 2016-01-01 --end 2023-12-31
    python3 backtest_oil.py --start 2016-01-01 --end 2023-12-31 --no-ml
"""

import warnings
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from backtest.monte_carlo import monte_carlo_barrier, run_prop_firm_simulations, PROP_FIRM_CONFIGS
from strategy.ml_filter import SignalFilter
from strategy.ict_strategy import prepare_data

DATA_DIR = Path("data/historical")

INITIAL_EQUITY = 10_000.0
RISK_PCT       = 0.02
MC_N_SIMS      = 10_000
ML_THRESHOLD   = 0.55
MIN_TRADES_ML  = 15
MIN_TRADES_MC  = 5

_FIRM_KEYS = ["FTMO_50k", "The5ers_100k", "TopStep_50k"]

VARIATIONS = [
    # Label,         min_rr, lookback, kill_zones, news,  tol
    ("Baseline",      2.0,   200,      True,       True,  0.3),
    ("LB=300",        2.0,   300,      True,       True,  0.3),
    ("LB=150",        2.0,   150,      True,       True,  0.3),
    ("RR=2.5",        2.5,   200,      True,       True,  0.3),
    ("RR=2.5+LB=300", 2.5,   300,      True,       True,  0.3),
    ("RR=3+LB=300",   3.0,   300,      True,       True,  0.3),
    ("Tol=0.5",       2.0,   200,      True,       True,  0.5),
    ("NoNews+LB=300", 2.0,   300,      True,       False, 0.3),
]


def _r_multiples(trades) -> np.ndarray:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    return np.array([t.r_multiple for t in closed])


def _compute_pass_rates(r_mults: np.ndarray) -> dict[str, float]:
    out = {}
    for key in _FIRM_KEYS:
        cfg = PROP_FIRM_CONFIGS[key]
        br = monte_carlo_barrier(
            r_mults,
            initial_balance=cfg["initial_balance"],
            profit_target_pct=cfg["profit_target_pct"],
            max_drawdown_pct=cfg["max_drawdown_pct"],
            daily_loss_limit_pct=cfg.get("daily_loss_limit_pct"),
            challenge_fee=cfg["challenge_fee"],
            r_to_pct=RISK_PCT,
            n_sims=MC_N_SIMS,
        )
        out[key] = br.pass_rate
    return out


def _run_variation(label, min_rr, lookback, use_kz, use_news, tol,
                   df_daily, df_5m, df_5m_enriched, use_ml=True):
    trades_raw, equity_raw = run_backtest(
        "Oil", df_daily, df_5m,
        initial_equity=INITIAL_EQUITY,
        risk_pct=RISK_PCT,
        min_rr=min_rr,
        use_kill_zones=use_kz,
        zone_lookback=lookback,
        zone_tolerance_atr=tol,
        use_news_filter=use_news,
    )

    n_raw = len([t for t in trades_raw if t.outcome in ("win", "loss")])

    if not use_ml or n_raw < MIN_TRADES_ML:
        return trades_raw, equity_raw, n_raw, None

    ml_filter = SignalFilter(probability_threshold=ML_THRESHOLD)
    ml_result = ml_filter.fit(trades_raw, df_5m_enriched)
    if ml_result is None:
        return trades_raw, equity_raw, n_raw, None

    trades_ml, equity_ml = run_backtest(
        "Oil", df_daily, df_5m,
        initial_equity=INITIAL_EQUITY,
        risk_pct=RISK_PCT,
        min_rr=min_rr,
        signal_filter=ml_filter,
        use_kill_zones=use_kz,
        zone_lookback=lookback,
        zone_tolerance_atr=tol,
        use_news_filter=use_news,
        df_enriched_for_filter=df_5m_enriched,
    )
    return trades_ml, equity_ml, n_raw, ml_result.acceptance_rate


def _trades_per_week(trades, df_5m) -> float:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if not closed:
        return 0.0
    n_weeks = len(df_5m) / (5 * 24 * 12)  # 5d × 24h × 12 bars/h for 5m bars
    return len(closed) / max(n_weeks, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end",   default="2023-12-31")
    parser.add_argument("--no-ml", action="store_true")
    args = parser.parse_args()

    use_ml = not args.no_ml

    print("=" * 64)
    print("  OIL (BRENT) — ICT Strategy Backtest".center(64))
    print("=" * 64)
    print(f"  Period : {args.start} → {args.end}")
    print(f"  ML filter: {'ON' if use_ml else 'OFF'}")

    # ── Load data ─────────────────────────────────────────────────────────────
    p5m    = DATA_DIR / "brent_5m_histdata.parquet"
    pdaily = DATA_DIR / "brent_daily_combined.parquet"

    if not p5m.exists():
        print(f"ERROR: {p5m} not found. Run data/histdata_fetcher.py first.")
        return
    if not pdaily.exists():
        print(f"ERROR: {pdaily} not found.")
        return

    df_5m_full    = pd.read_parquet(p5m)
    df_daily_full = pd.read_parquet(pdaily)

    # Clip to requested period
    df_5m    = df_5m_full[(df_5m_full.index >= args.start) & (df_5m_full.index <= args.end)]
    df_daily = df_daily_full[(df_daily_full.index >= args.start) & (df_daily_full.index <= args.end)]

    print(f"\n  5m bars : {len(df_5m):,}  ({df_5m.index[0].date()} → {df_5m.index[-1].date()})")
    print(f"  Daily   : {len(df_daily):,}  ({df_daily.index[0].date()} → {df_daily.index[-1].date()})")

    # Enrich 5m for ML features
    try:
        from detectors import indicators
        df_5m_enriched = indicators.add_all(df_5m.copy())
    except Exception:
        df_5m_enriched = df_5m.copy()

    # Mark news windows
    try:
        events = get_high_impact_events(start=args.start, end=args.end)
        df_5m_enriched = mark_news_windows(df_5m_enriched, events)
    except Exception:
        pass

    # ── Grid search ───────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    best_calmar = -999
    best_result = None

    for label, min_rr, lookback, use_kz, use_news, tol in VARIATIONS:
        trades, equity, n_raw, ml_acc = _run_variation(
            label, min_rr, lookback, use_kz, use_news, tol,
            df_daily, df_5m, df_5m_enriched, use_ml=use_ml
        )
        closed = [t for t in trades if t.outcome in ("win", "loss")]
        n = len(closed)

        if n < MIN_TRADES_MC:
            print(f"  {label:<22}  only {n} closed trades — skip")
            continue

        m = compute_metrics(closed, equity)
        if not m or "message" in m:
            print(f"  {label:<22}  metrics unavailable")
            continue

        tpw = _trades_per_week(trades, df_5m)
        ml_str = f"  ML:{ml_acc*100:.0f}%" if ml_acc is not None else ""

        print(f"  {label:<22}  N:{n:>4}{ml_str}  WR:{m.get('win_rate','—'):>7}  "
              f"EV/R:{m.get('expectancy_r','—'):>6}  PF:{m.get('profit_factor','—'):>6}  "
              f"Return:{m.get('total_return_pct','—'):>8}  "
              f"MaxDD:{m.get('max_drawdown_pct','—'):>7}  "
              f"Calmar:{m.get('calmar_ratio','—'):>6}  "
              f"T/wk:{tpw:.1f}")

        calmar = 0.0
        try:
            calmar = float(str(m.get("calmar_ratio", 0)).replace("%",""))
        except Exception:
            pass

        if calmar > best_calmar:
            best_calmar = calmar
            best_result = (label, min_rr, lookback, use_kz, use_news, tol, trades, equity, m, tpw)

    if best_result is None:
        print("\n  No valid variations found.")
        return

    b_label, b_rr, b_lb, b_kz, b_news, b_tol, b_trades, b_equity, b_m, b_tpw = best_result
    print(f"\n  ★ Best: {b_label}  (min_rr={b_rr}, lb={b_lb})")
    print(f"     WR={b_m.get('win_rate')}  Return={b_m.get('total_return_pct')}  "
          f"MaxDD={b_m.get('max_drawdown_pct')}  Calmar={b_m.get('calmar_ratio')}  "
          f"T/wk={b_tpw:.1f}")

    # ── MC Prop firm sims ─────────────────────────────────────────────────────
    r_mults = _r_multiples(b_trades)
    if len(r_mults) >= MIN_TRADES_MC:
        print(f"\n  Prop firm sims ({MC_N_SIMS:,} MC)...")
        run_prop_firm_simulations(r_mults, r_to_pct=RISK_PCT, n_sims=MC_N_SIMS)

    # ── Weekly trade frequency ─────────────────────────────────────────────────
    print(f"\n  Average trades/week (best config): {b_tpw:.2f}")

    # ── Save trade log ────────────────────────────────────────────────────────
    rows = [t.to_dict() for t in b_trades if hasattr(t, "to_dict")]
    if rows:
        df_log = pd.DataFrame(rows)
        out = Path("backtest") / "oil_ict_trades.csv"
        df_log.to_csv(out, index=False)
        print(f"  Trade log → {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
