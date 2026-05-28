"""
lnterqo v2 — Rolling Walk-Forward Validation
=============================================
NYU Stern (2025): "Online Quantitative Trading Strategies" methodology
  - 5 anchored rolling windows (expanding IS, fixed 10% OOS per window)
  - Per-window: grid search → best variation → frozen ML → OOS
  - EG (Exponential Gradient) position sizing (Kivinen & Warmuth, 1997)
  - Aggregated OOS trades across all 5 windows for combined edge estimate
  - Final comparison vs ICT v1 and lnterqo v1

Anchored window schedule (50% IS minimum → 10% OOS each):
  Win 1: IS=0–50%   OOS=50–60%
  Win 2: IS=0–60%   OOS=60–70%
  Win 3: IS=0–70%   OOS=70–80%
  Win 4: IS=0–80%   OOS=80–90%
  Win 5: IS=0–90%   OOS=90–100%
"""

import warnings, sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from datetime import timedelta

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from detectors.regime import classify_regime, regime_markov_analysis, cusum_events
from backtest.metrics import compute_metrics, print_metrics, trades_to_dataframe
from backtest.statistics import validate, required_sample_size
from backtest.monte_carlo import (
    monte_carlo_reshuffle, monte_carlo_regime_switching, monte_carlo_barrier,
    assign_trade_regimes, run_prop_firm_simulations, PROP_FIRM_CONFIGS,
)
from backtest.engine import Trade
from strategy.ml_filter import SignalFilter
from strategy.lnterqo_strategy import prepare_data, scan_for_signals
from strategy.risk_manager import RiskManager, INITIAL_EQUITY


# ── Config ─────────────────────────────────────────────────────────────────────

DATA_DIR        = Path("data/historical")
COMBINED_5M     = DATA_DIR / "gold_5m_combined.parquet"
N_WINDOWS       = 5
INITIAL_IS_PCT  = 0.50   # first window IS fraction
OOS_WINDOW_PCT  = 0.10   # each OOS window fraction
MC_N_SIMS       = 100_000
MC_N_SIMS_IS    = 5_000  # reduced for per-window IS grid search speed
ML_THRESHOLD    = 0.60
MIN_TRADES_ML   = 15
MIN_TRADES_MC   = 3
R_TO_PCT        = 0.005

# EG sizing params
EG_ETA          = 0.02   # learning rate — dampened to avoid oversizing
EG_MIN_RISK     = 0.0025 # 0.25% floor
EG_MAX_RISK     = 0.015  # 1.5% ceiling
EG_BASE_RISK    = 0.005  # 0.5% starting point

# Variations: (label, min_rr, zone_lookback, cisd_lookback)
VARIATIONS = [
    ("CISD_RR2_ZL50",  2.0, 50, 40),
    ("CISD_RR2_ZL80",  2.0, 80, 40),
    ("CISD_RR3_ZL50",  3.0, 50, 40),
]

_FIRM_KEYS = ["FTMO_100k", "FTMO_50k", "The5ers_100k", "TopStep_50k", "TradingGame"]

# Published results from earlier runs (for final comparison table)
_ICT_V1 = dict(trades=47, wr=37.5, ev_r=0.684, sharpe=5.44, max_dd=-2.48, barrier=100.0)
_LNT_V1 = dict(trades=93, wr=38.7, ev_r=1.421, sharpe=4.08, max_dd=-4.39, barrier=100.0)


# ── Exponential Gradient Risk Manager ─────────────────────────────────────────

@dataclass
class EGRiskManager(RiskManager):
    """
    Extends RiskManager with EG multiplicative position-size updates.
    After each trade: risk_pct ← risk_pct * exp(η * r_proxy)
    Replaces the fixed-tier sizing while keeping the loss ladder safety rules.
    """
    eta: float = EG_ETA
    min_risk: float = EG_MIN_RISK
    max_risk: float = EG_MAX_RISK

    def __post_init__(self):
        self._eg_risk: float = self.risk_pct   # starts at base 0.5%

    def position_size(self, entry: float, stop: float, confidence: int = 1) -> float:
        distance = abs(entry - stop)
        if distance == 0:
            return 0.0
        # EG fraction, still gated by loss-ladder cap
        ladder_cap = super().current_risk_pct(confidence)
        effective = min(self._eg_risk, ladder_cap) if ladder_cap > 0 else 0.0
        return (self.equity * effective) / distance

    def record_trade(self, pnl: float):
        # Compute r_proxy before super() updates equity
        r_proxy = pnl / (self.equity * self._eg_risk) if self._eg_risk > 0 and self.equity > 0 else 0.0
        super().record_trade(pnl)
        # Multiplicative EG update
        self._eg_risk *= float(np.exp(self.eta * r_proxy))
        self._eg_risk = float(np.clip(self._eg_risk, self.min_risk, self.max_risk))

    def reset_day(self):
        super().reset_day()
        # EG risk persists across days — only loss ladder resets


# ── Backtest runner (EG-aware) ─────────────────────────────────────────────────

def run_lnterqo_backtest_v2(
    asset, df_daily, df_5m,
    min_rr, zone_lookback, cisd_lookback,
    initial_equity=INITIAL_EQUITY,
    signal_filter=None,
    df_enriched_for_filter=None,
    use_news_filter=True,
    force_neutral_bias=False,
    use_eg=True,
):
    rm = EGRiskManager(equity=initial_equity) if use_eg else RiskManager(equity=initial_equity)

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

        if open_trade is not None:
            sig  = open_trade.signal
            high = row["high"]
            low  = row["low"]
            bearish = row["close"] < row["open"]
            open_trade.bars_held += 1
            outcome = None; exit_p = 0.0

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

        if open_trade is None and ts in signal_map:
            ok, _ = rm.can_trade()
            if ok:
                sig = signal_map[ts]
                _df_f = df_enriched_for_filter if df_enriched_for_filter is not None else df_5m
                if signal_filter is not None and not signal_filter.accept(sig, _df_f):
                    continue
                size = rm.position_size(sig.entry, sig.stop, sig.confidence)
                if size > 0:
                    open_trade = Trade(signal=sig, size=size, entry_price=sig.entry)

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

def _enrich(df_5m, df_daily):
    news = get_high_impact_events(df_5m.index[0], df_5m.index[-1])
    df_5m = mark_news_windows(df_5m, news, pre_minutes=15, post_hours=2)
    df_5m = add_seasonal_columns(df_5m, asset="Gold")
    df_daily_enr, df_5m_enr = prepare_data(df_daily.copy(), df_5m.copy())
    if "cusum_up" not in df_5m_enr.columns:
        df_5m_enr = cusum_events(df_5m_enr)
    return df_daily_enr, df_5m_enr, len(news)


def _r_multiples(trades) -> np.ndarray:
    return np.array([t.r_multiple for t in trades if t.outcome in ("win", "loss")])


def _pass_rate_single(r_mults, key, n_sims):
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
    return br.pass_rate


def _fmt(v):
    return f"{v*100:6.1f}%" if v is not None else "   N/A"


def _run_is_grid(df_daily_is, df_5m_is, window_idx):
    """Grid search on IS — returns (best_label, best_params, best_ml, rows)."""
    rows = []
    best_ml_filters = {}
    hdr = (f"  {'Variation':<20} | {'Raw':>5} | {'Trades':>6} | "
           f"{'WR%':>6} | {'EV/R':>6} | {'MaxDD%':>7} | {'The5%':>7}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for label, min_rr, zone_lb, cisd_lb in VARIATIONS:
        # Raw IS run (no ML)
        trades_raw, eq_raw = run_lnterqo_backtest_v2(
            "Gold", df_daily_is, df_5m_is,
            min_rr=min_rr, zone_lookback=zone_lb, cisd_lookback=cisd_lb,
            force_neutral_bias=True, use_eg=False,
        )
        n_raw = len([t for t in trades_raw if t.outcome in ("win", "loss")])

        if n_raw < MIN_TRADES_ML:
            print(f"  {label:<20} | {n_raw:>5} | SKIP (insufficient)")
            rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                         "cisd_lb": cisd_lb, "n_trades": 0, "the5ers": None})
            continue

        ml = SignalFilter(probability_threshold=ML_THRESHOLD)
        ml_result = ml.fit(trades_raw, df_5m_is)
        if ml_result is None:
            rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                         "cisd_lb": cisd_lb, "n_trades": 0, "the5ers": None})
            continue

        trades_ml, eq_ml = run_lnterqo_backtest_v2(
            "Gold", df_daily_is, df_5m_is,
            min_rr=min_rr, zone_lookback=zone_lb, cisd_lookback=cisd_lb,
            force_neutral_bias=True, use_eg=False,
            signal_filter=ml, df_enriched_for_filter=df_5m_is,
        )
        best_ml_filters[label] = ml
        closed = [t for t in trades_ml if t.outcome in ("win", "loss")]
        n_closed = len(closed)

        if n_closed < MIN_TRADES_MC:
            rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                         "cisd_lb": cisd_lb, "n_trades": n_closed, "the5ers": None})
            continue

        metrics = compute_metrics(trades_ml, eq_ml)
        if not metrics or "message" in metrics:
            rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                         "cisd_lb": cisd_lb, "n_trades": n_closed, "the5ers": None})
            continue

        wr   = float(metrics["win_rate"].rstrip("%")) / 100.0
        ev_r = metrics.get("expectancy_r", 0)
        dd   = metrics.get("max_drawdown_pct", 0)

        r_mults = _r_multiples(trades_ml)
        the5ers = _pass_rate_single(r_mults, "The5ers_100k", MC_N_SIMS_IS) if len(r_mults) >= 3 else None

        print(f"  {label:<20} | {n_raw:>5} | {n_closed:>6} | "
              f"{wr*100:>5.1f}% | {ev_r:>+6.3f} | {dd:>+7.2f}% | {_fmt(the5ers):>7}")

        rows.append({"label": label, "min_rr": min_rr, "zone_lb": zone_lb,
                     "cisd_lb": cisd_lb, "n_trades": n_closed,
                     "win_rate": wr, "ev_r": ev_r, "max_dd_pct": dd,
                     "the5ers": the5ers})

    valid = [r for r in rows if r.get("the5ers") is not None]
    if not valid:
        # Fallback: best by trade count
        valid = [r for r in rows if r["n_trades"] > 0]
        if not valid:
            return VARIATIONS[0][0], VARIATIONS[0], None
        best_row = max(valid, key=lambda r: r["n_trades"])
    else:
        best_row = max(valid, key=lambda r: (r["the5ers"] or 0, r.get("ev_r") or 0))

    best_label  = best_row["label"]
    best_params = (best_row["label"], best_row["min_rr"], best_row["zone_lb"], best_row["cisd_lb"])
    best_ml     = best_ml_filters.get(best_label)
    return best_label, best_params, best_ml


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  lnterqo v2 — ROLLING WALK-FORWARD VALIDATION  (NYU Stern 2025)")
    print(f"  Data: {COMBINED_5M.name}")
    print(f"  Windows: {N_WINDOWS}  |  IS: anchored  |  OOS per window: {OOS_WINDOW_PCT*100:.0f}%")
    print(f"  Sizing: Exponential Gradient (η={EG_ETA}, [{EG_MIN_RISK*100:.2f}%–{EG_MAX_RISK*100:.1f}%])")
    print(f"{'='*70}")

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

    total_bars = len(df_5m)
    total_start = df_5m.index[0]
    total_end   = df_5m.index[-1]
    print(f"\n  Full range  : {total_start.date()} → {total_end.date()}  ({total_bars:,} bars)")

    # ── Build window boundaries ────────────────────────────────────────────────
    windows = []
    for w in range(N_WINDOWS):
        is_end_frac  = INITIAL_IS_PCT + w * OOS_WINDOW_PCT
        oos_end_frac = is_end_frac + OOS_WINDOW_PCT
        is_end_bar   = int(total_bars * is_end_frac)
        oos_end_bar  = min(int(total_bars * oos_end_frac), total_bars)
        windows.append((is_end_bar, oos_end_bar))

    print(f"\n  {'Win':>3}  {'IS start':>12}  {'IS end':>12}  {'OOS start':>12}  {'OOS end':>12}  {'IS bars':>8}  {'OOS bars':>8}")
    print("  " + "-" * 80)
    for w, (is_end, oos_end) in enumerate(windows):
        is_start_bar = 0
        oos_start_bar = is_end
        is_start  = df_5m.index[is_start_bar]
        is_end_ts = df_5m.index[is_end - 1]
        oos_s     = df_5m.index[oos_start_bar]
        oos_e     = df_5m.index[oos_end - 1]
        print(f"  {w+1:>3}  {str(is_start.date()):>12}  {str(is_end_ts.date()):>12}  "
              f"{str(oos_s.date()):>12}  {str(oos_e.date()):>12}  "
              f"{is_end:>8,}  {oos_end - oos_start_bar:>8,}")

    # ── Run each window ────────────────────────────────────────────────────────
    all_oos_trades   = []
    window_summaries = []

    for w, (is_end_bar, oos_end_bar) in enumerate(windows):
        oos_start_bar = is_end_bar

        df_5m_is    = df_5m.iloc[:is_end_bar]
        df_5m_oos   = df_5m.iloc[oos_start_bar:oos_end_bar]
        df_daily_is = df_daily[df_daily.index <= df_5m_is.index[-1]]
        df_daily_oos= df_daily[
            (df_daily.index > df_5m_is.index[-1]) &
            (df_daily.index <= df_5m_oos.index[-1])
        ]

        print(f"\n{'─'*70}")
        print(f"  WINDOW {w+1}/{N_WINDOWS}  "
              f"IS: {df_5m_is.index[0].date()} → {df_5m_is.index[-1].date()}  "
              f"OOS: {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
        print(f"{'─'*70}")

        # Enrich IS
        print(f"  Enriching IS...")
        df_daily_is_enr, df_5m_is_enr, n_news_is = _enrich(df_5m_is, df_daily_is)
        print(f"  IS news events: {n_news_is}")

        # Grid search on IS
        print(f"  IS grid search:")
        best_label, best_params, best_ml = _run_is_grid(df_daily_is_enr, df_5m_is_enr, w)
        _, best_rr, best_zlb, best_clb = best_params
        print(f"\n  Selected: [{best_label}]  RR={best_rr}  ZL={best_zlb}  CL={best_clb}")

        # Enrich OOS
        print(f"  Enriching OOS...")
        df_daily_oos_enr, df_5m_oos_enr, n_news_oos = _enrich(df_5m_oos, df_daily_oos)
        print(f"  OOS news events: {n_news_oos}")

        # OOS backtest with frozen ML + EG sizing
        trades_oos, equity_oos = run_lnterqo_backtest_v2(
            "Gold", df_daily_oos_enr, df_5m_oos_enr,
            min_rr=best_rr, zone_lookback=best_zlb, cisd_lookback=best_clb,
            force_neutral_bias=True,
            signal_filter=best_ml,
            df_enriched_for_filter=df_5m_oos_enr,
            use_eg=True,
        )

        closed_oos = [t for t in trades_oos if t.outcome in ("win", "loss")]
        n_trades   = len(closed_oos)
        r_mults    = _r_multiples(trades_oos)
        metrics    = compute_metrics(trades_oos, equity_oos)

        if metrics and "message" not in metrics and n_trades > 0:
            wr   = float(metrics["win_rate"].rstrip("%")) / 100.0
            ev_r = metrics.get("expectancy_r", 0.0)
            dd   = metrics.get("max_drawdown_pct", 0.0)
            sh   = metrics.get("sharpe_ratio", 0.0)
        else:
            wr = ev_r = dd = sh = 0.0

        print(f"\n  Window {w+1} OOS: {n_trades} trades | "
              f"WR={wr*100:.1f}% | EV={ev_r:+.3f}R | "
              f"DD={dd:+.2f}% | Sharpe={sh:.2f}")

        all_oos_trades.extend(closed_oos)
        window_summaries.append({
            "window": w + 1,
            "is_start": df_5m_is.index[0].date(),
            "oos_start": df_5m_oos.index[0].date(),
            "oos_end":   df_5m_oos.index[-1].date(),
            "variation": best_label,
            "n_trades":  n_trades,
            "win_rate":  wr,
            "ev_r":      ev_r,
            "max_dd":    dd,
            "sharpe":    sh,
        })

    # ── Aggregate OOS ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ROLLING WFV — WINDOW-BY-WINDOW SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Win':>3}  {'OOS Period':>24}  {'Var':<20}  "
          f"{'Trades':>6}  {'WR%':>6}  {'EV/R':>6}  {'DD%':>7}  {'Sharpe':>7}")
    print("  " + "-" * 85)
    for s in window_summaries:
        print(f"  {s['window']:>3}  "
              f"{str(s['oos_start'])+' → '+str(s['oos_end']):>24}  "
              f"{s['variation']:<20}  "
              f"{s['n_trades']:>6}  "
              f"{s['win_rate']*100:>5.1f}%  "
              f"{s['ev_r']:>+6.3f}  "
              f"{s['max_dd']:>+7.2f}%  "
              f"{s['sharpe']:>7.2f}")

    all_r_mults = _r_multiples(all_oos_trades)
    n_total_trades = len(all_oos_trades)
    wins_total     = sum(1 for t in all_oos_trades if t.outcome == "win")

    print(f"\n  {'='*70}")
    print(f"  COMBINED OOS: {n_total_trades} trades  |  "
          f"WR={wins_total/n_total_trades*100:.1f}%  |  "
          f"EV={np.mean(all_r_mults):+.3f}R" if n_total_trades > 0 else
          f"  COMBINED OOS: 0 trades")

    if n_total_trades == 0:
        print("\n  No OOS trades collected. Exiting.")
        return

    # Save combined trade log
    os.makedirs("backtest", exist_ok=True)
    df_all = pd.DataFrame([{
        "entry_time":  t.signal.timestamp,
        "direction":   t.signal.direction,
        "entry":       t.entry_price,
        "stop":        t.signal.stop,
        "target":      t.signal.target,
        "exit_price":  t.exit_price,
        "exit_time":   t.exit_time,
        "outcome":     t.outcome,
        "r_multiple":  t.r_multiple,
        "pnl":         t.pnl,
        "confidence":  t.signal.confidence,
        "zone_type":   t.signal.zone_type,
    } for t in all_oos_trades])
    df_all.to_csv("backtest/lnterqo_v2_oos_trades.csv", index=False)
    print(f"  Trade log saved: backtest/lnterqo_v2_oos_trades.csv")

    if n_total_trades < MIN_TRADES_MC:
        print(f"  Only {n_total_trades} trades — need ≥{MIN_TRADES_MC}. Done.")
        return

    # ── Statistical validation on combined OOS ────────────────────────────────
    print(f"\n{'='*70}")
    print("  STATISTICAL VALIDATION — COMBINED OOS (all windows)")
    print(f"{'='*70}")

    # Use the last OOS window's enriched df for IC calculation
    # (proxy — IC is computed on the most recent period)
    oos_end_bar_last = windows[-1][1]
    oos_start_bar_last = windows[-1][0]
    df_5m_last_oos = df_5m.iloc[oos_start_bar_last:oos_end_bar_last]
    if oos_start_bar_last > 0:
        df_daily_last = df_daily[df_daily.index > df_5m.iloc[oos_start_bar_last - 1].name]
    else:
        df_daily_last = df_daily
    _, df_5m_last_enr, _ = _enrich(df_5m_last_oos, df_daily_last)

    stat = validate(all_oos_trades, df_5m_last_enr, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
    if stat:
        print(stat.report())
        if abs(stat.ev_per_trade) > 1e-6:
            n_needed = required_sample_size(delta=abs(stat.ev_per_trade), sigma=1.0)
            print(f"  Trades to confirm {abs(stat.ev_per_trade):.2f}R edge: {n_needed}")
            print(f"  Combined OOS total: {n_total_trades} — "
                  f"{'sufficient' if n_total_trades >= n_needed else 'INSUFFICIENT'}")

    # ── Monte Carlo on combined OOS ────────────────────────────────────────────
    print(f"\n  Running Monte Carlo (combined OOS)  —  {MC_N_SIMS:,} simulations ...")

    mc1 = monte_carlo_reshuffle(all_r_mults, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
    print(mc1.report())

    # Regime switching MC
    try:
        df_regime = classify_regime(df_5m_last_enr, adx_threshold=25.0, use_hurst=True)
        regime_series = df_regime["regime"].values
        # Only use trades from last window for regime labelling (for brevity)
        last_win_trades = [t for t in all_oos_trades
                           if t.signal.timestamp >= df_5m_last_oos.index[0]]
        if len(last_win_trades) >= MIN_TRADES_MC:
            trade_regimes = assign_trade_regimes(last_win_trades, df_regime, regime_series)
            if len(set(trade_regimes)) > 1:
                r_last = _r_multiples(last_win_trades)
                mc2 = monte_carlo_regime_switching(r_last, trade_regimes, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
                print(mc2.report())
    except Exception:
        pass

    # Trading Game barrier
    mc3 = monte_carlo_barrier(
        all_r_mults, initial_balance=INITIAL_EQUITY,
        profit_target_pct=0.20, max_drawdown_pct=0.20,
        challenge_fee=0.0, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT,
    )
    print(mc3.report())

    print(f"\n  Prop Firm Pass Rates (combined OOS, {MC_N_SIMS:,} sims):")
    run_prop_firm_simulations(all_r_mults, r_to_pct=R_TO_PCT, n_sims=MC_N_SIMS)

    # ── Comparison table ──────────────────────────────────────────────────────
    combined_wr  = wins_total / n_total_trades if n_total_trades else 0
    combined_ev  = float(np.mean(all_r_mults)) if len(all_r_mults) else 0
    combined_dd  = min(s["max_dd"] for s in window_summaries) if window_summaries else 0
    combined_sh  = float(np.mean([s["sharpe"] for s in window_summaries if s["sharpe"] != 0]))
    combined_bar = mc3.pass_rate * 100

    # Trades/year: combined trades over the OOS span
    oos_start_overall = df_5m.iloc[windows[0][0]].name
    oos_end_overall   = df_5m.iloc[windows[-1][1] - 1].name
    oos_years = max((oos_end_overall - oos_start_overall).days / 365.25, 0.01)
    trades_per_yr = n_total_trades / oos_years

    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*70}")
    hdr = (f"  {'Strategy':<22}  {'Trades/yr':>9}  {'WR%':>6}  {'EV/R':>6}  "
           f"{'MaxDD%':>7}  {'Sharpe':>7}  {'Barrier%':>9}")
    sep = "  " + "-" * (len(hdr) - 2)
    print(hdr); print(sep)

    def _row(name, tpy, wr, ev, dd, sh, bar):
        return (f"  {name:<22}  {tpy:>9.1f}  {wr:>5.1f}%  {ev:>+6.3f}  "
                f"{dd:>+7.2f}%  {sh:>7.2f}  {bar:>8.1f}%")

    print(_row("ICT v1",
               _ICT_V1["trades"], _ICT_V1["wr"], _ICT_V1["ev_r"],
               _ICT_V1["max_dd"], _ICT_V1["sharpe"], _ICT_V1["barrier"]))
    print(_row("lnterqo v1 (80/20)",
               _LNT_V1["trades"], _LNT_V1["wr"], _LNT_V1["ev_r"],
               _LNT_V1["max_dd"], _LNT_V1["sharpe"], _LNT_V1["barrier"]))
    print(_row(f"lnterqo v2 (rolling WFV)",
               trades_per_yr, combined_wr * 100, combined_ev,
               combined_dd, combined_sh, combined_bar))
    print(sep)

    print(f"\n  Notes:")
    print(f"  - v2 covers {n_total_trades} trades across {N_WINDOWS} non-overlapping OOS windows")
    print(f"  - OOS span: {oos_start_overall.date()} → {oos_end_overall.date()} ({oos_years:.1f} yrs)")
    print(f"  - EG sizing: η={EG_ETA}, risk range [{EG_MIN_RISK*100:.2f}%–{EG_MAX_RISK*100:.1f}%]")
    print(f"  - ICT v1 / lnterqo v1 from single 80/20 OOS (2024-04-23 → 2026-05-22)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
