"""
lnterqo v4 — Rolling WFV + Genetic Parameter Optimisation
==========================================================
Extends v3 (rolling WFV, fixed risk tiers) by replacing the manual
3-variation grid search with a genetic algorithm (inspired by the
RationalEdge / NYU paper workflow).

Key innovation:
  - Signals are pre-computed ONCE per window (relaxed min_rr=1.0)
  - Raw outcomes (TP/SL hit) computed without risk management
  - ML model fitted on raw outcomes
  - GA evolves [min_rr, ml_threshold, min_confidence] in O(milliseconds)
    per fitness call — no re-scanning, no re-running the full backtest
  - Zone lookback [50, 80] run in parallel; GA finds best params for each;
    best overall combination selected for OOS

GA setup:
  Population: 30  |  Generations: 20  |  Tournament k: 3
  Fitness: profit_factor × ln(1 + n_trades)

Risk profile: v1 fixed tiers (0.5% / 1% conf≥4, loss ladder)
"""

import warnings, sys, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
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

DATA_DIR       = Path("data/historical")
COMBINED_5M    = DATA_DIR / "gold_5m_combined.parquet"
N_WINDOWS      = 5
INITIAL_IS_PCT = 0.50
OOS_WINDOW_PCT = 0.10
MC_N_SIMS      = 100_000
MIN_TRADES_ML  = 15
MIN_TRADES_MC  = 3
R_TO_PCT       = 0.005

# Genetic algorithm
GA_POP         = 30
GA_GEN         = 20
GA_CR          = 0.75   # crossover rate
GA_MR          = 0.20   # mutation rate
GA_K           = 3      # tournament size

ZONE_LBS       = [50, 80]   # pre-compute for both; GA picks best
CISD_LB        = 40

# Published results for comparison table
_ICT_V1 = dict(trades=47.0,  wr=37.5, ev_r=0.684, sharpe=5.44, max_dd=-2.48, barrier=100.0)
_LNT_V1 = dict(trades=93.0,  wr=38.7, ev_r=1.421, sharpe=4.08, max_dd=-4.39, barrier=100.0)
_LNT_V2 = dict(trades=112.5, wr=39.7, ev_r=2.122, sharpe=3.75, max_dd=-5.19, barrier=100.0)
_LNT_V3 = dict(trades=110.4, wr=39.4, ev_r=2.129, sharpe=3.61, max_dd=-5.37, barrier=100.0)


# ── Raw outcome computation (no risk management) ───────────────────────────────

def _compute_raw_outcomes(signals: list, df_5m: pd.DataFrame) -> list[Trade]:
    """
    For every signal, scan forward up to 300 bars to find TP or SL hit.
    Returns Trade objects with outcome/pnl set, using unit size so that
    r_multiple == theoretical R (needed by ML filter's prepare_dataset).
    """
    idx_map = {ts: i for i, ts in enumerate(df_5m.index)}
    trades = []

    for sig in signals:
        if sig.timestamp not in idx_map:
            continue
        start = idx_map[sig.timestamp]
        risk = abs(sig.entry - sig.stop)
        if risk == 0:
            continue
        size = 1.0 / risk   # unit sizing → pnl = R multiple

        outcome = None
        exit_p  = 0.0
        exit_j  = start

        for j in range(start + 1, min(start + 300, len(df_5m))):
            row = df_5m.iloc[j]
            h, l = row["high"], row["low"]
            bearish = row["close"] < row["open"]

            if sig.direction == "long":
                sl_hit = l <= sig.stop
                tp_hit = h >= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("loss", sig.stop) if bearish else ("win", sig.target)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target
            else:
                sl_hit = h >= sig.stop
                tp_hit = l <= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("win", sig.target) if bearish else ("loss", sig.stop)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target

            if outcome:
                exit_j = j
                break

        if not outcome:
            continue

        t = Trade(signal=sig, size=size, entry_price=sig.entry)
        t.exit_price = exit_p
        t.exit_time  = df_5m.index[exit_j]
        t.outcome    = outcome
        if sig.direction == "long":
            t.pnl = (exit_p - sig.entry) * size
        else:
            t.pnl = (sig.entry - exit_p) * size
        trades.append(t)

    return trades


# ── Pre-computed signal pool (GA fitness oracle) ───────────────────────────────

class PrecomputedSignalPool:
    """
    Holds pre-computed (signal, outcome, r_multiple, ml_probability) for every
    raw signal. The GA calls fitness() millions of times — each call is just an
    array filter, taking microseconds.
    """

    def __init__(self, raw_trades: list[Trade], ml: SignalFilter, df: pd.DataFrame):
        extractor = ml.extractor
        self._rr    = []
        self._conf  = []
        self._prob  = []
        self._rmult = []

        for t in raw_trades:
            if t.outcome not in ("win", "loss"):
                continue
            features = extractor.extract(t.signal, df)
            if features is None:
                prob = 0.5
            else:
                prob = float(ml.model.predict_proba(features.reshape(1, -1))[0, 1])
            self._rr.append(t.signal.risk_reward)
            self._conf.append(t.signal.confidence)
            self._prob.append(prob)
            self._rmult.append(t.r_multiple)

        self._rr    = np.array(self._rr,    dtype=float)
        self._conf  = np.array(self._conf,  dtype=int)
        self._prob  = np.array(self._prob,  dtype=float)
        self._rmult = np.array(self._rmult, dtype=float)

    def __len__(self):
        return len(self._rr)

    def fitness(self, min_rr: float, ml_threshold: float, min_confidence: int) -> float:
        mask = (self._rr >= min_rr) & (self._prob >= ml_threshold) & (self._conf >= min_confidence)
        r = self._rmult[mask]
        n = len(r)
        if n < 15:
            return -1.0
        wins   = r[r > 0].sum()
        losses = abs(r[r <= 0].sum())
        pf = wins / losses if losses > 0 else wins
        return float(pf * np.log1p(n))


# ── Genetic optimiser ──────────────────────────────────────────────────────────

class GeneticOptimizer:
    """
    Evolves [min_rr, ml_threshold, min_confidence] using a standard GA.
    Fitness oracle is PrecomputedSignalPool — each eval is O(n_signals).
    """

    PARAM_GRID = {
        "min_rr":         [1.5, 2.0, 2.5, 3.0, 3.5],
        "ml_threshold":   [0.50, 0.55, 0.60, 0.65, 0.70],
        "min_confidence": [1, 2, 3],
    }

    def __init__(self, pool: PrecomputedSignalPool,
                 pop_size=GA_POP, n_gen=GA_GEN,
                 crossover_rate=GA_CR, mutation_rate=GA_MR,
                 tournament_k=GA_K, seed=42):
        self.pool = pool
        self.pop_size = pop_size
        self.n_gen = n_gen
        self.cr = crossover_rate
        self.mr = mutation_rate
        self.k  = tournament_k
        self.rng = np.random.default_rng(seed)
        self.keys  = list(self.PARAM_GRID.keys())
        self.sizes = [len(self.PARAM_GRID[k]) for k in self.keys]

    def _decode(self, c: np.ndarray) -> dict:
        return {k: self.PARAM_GRID[k][int(c[i])] for i, k in enumerate(self.keys)}

    def _eval(self, c: np.ndarray) -> float:
        return self.pool.fitness(**self._decode(c))

    def _select(self, pop: np.ndarray, fits: np.ndarray) -> np.ndarray:
        idxs = self.rng.choice(len(pop), self.k, replace=False)
        return pop[max(idxs, key=lambda i: fits[i])].copy()

    def _cross(self, p1: np.ndarray, p2: np.ndarray):
        if self.rng.random() > self.cr:
            return p1.copy(), p2.copy()
        pt = self.rng.integers(1, len(p1))
        return np.concatenate([p1[:pt], p2[pt:]]), np.concatenate([p2[:pt], p1[pt:]])

    def _mutate(self, c: np.ndarray) -> np.ndarray:
        c = c.copy()
        for i in range(len(c)):
            if self.rng.random() < self.mr:
                c[i] = self.rng.integers(0, self.sizes[i])
        return c

    def run(self, verbose=True) -> tuple[dict, float]:
        pop = np.array([[self.rng.integers(0, s) for s in self.sizes]
                        for _ in range(self.pop_size)])
        best_c = pop[0].copy()
        best_f = -999.0

        for gen in range(self.n_gen):
            fits = np.array([self._eval(c) for c in pop])
            bi = int(np.argmax(fits))
            if fits[bi] > best_f:
                best_f = fits[bi]
                best_c = pop[bi].copy()

            if verbose and (gen == 0 or (gen + 1) % 5 == 0 or gen == self.n_gen - 1):
                p = self._decode(best_c)
                print(f"      gen {gen+1:>2}/{self.n_gen}  "
                      f"fitness={best_f:.3f}  "
                      f"rr={p['min_rr']}  "
                      f"ml={p['ml_threshold']}  "
                      f"conf≥{p['min_confidence']}", flush=True)

            new_pop = [best_c.copy()]
            while len(new_pop) < self.pop_size:
                c1, c2 = self._cross(self._select(pop, fits), self._select(pop, fits))
                new_pop.append(self._mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self._mutate(c2))
            pop = np.array(new_pop)

        # Final evaluation
        fits = np.array([self._eval(c) for c in pop])
        bi   = int(np.argmax(fits))
        if fits[bi] > best_f:
            best_f = fits[bi]
            best_c = pop[bi].copy()

        return self._decode(best_c), float(best_f)


# ── Backtest runner (v1 fixed risk) ───────────────────────────────────────────

def run_lnterqo_backtest_v4(
    asset, df_daily, df_5m,
    min_rr, zone_lookback, cisd_lookback,
    min_confidence=1,
    initial_equity=INITIAL_EQUITY,
    signal_filter=None,
    df_enriched_for_filter=None,
    use_news_filter=True,
    force_neutral_bias=False,
):
    rm = RiskManager(equity=initial_equity)

    signals = scan_for_signals(
        asset, df_daily, df_5m,
        min_rr=min_rr,
        use_news_filter=use_news_filter,
        zone_lookback=zone_lookback,
        cisd_lookback=cisd_lookback,
        force_neutral_bias=force_neutral_bias,
    )
    # Apply minimum confidence filter
    if min_confidence > 1:
        signals = [s for s in signals if s.confidence >= min_confidence]

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
            sig     = open_trade.signal
            h, l    = row["high"], row["low"]
            bearish = row["close"] < row["open"]
            open_trade.bars_held += 1
            outcome = None; exit_p = 0.0

            if sig.direction == "long":
                sl_hit = l <= sig.stop
                tp_hit = h >= sig.target
                if sl_hit and tp_hit:
                    outcome, exit_p = ("loss", sig.stop) if bearish else ("win", sig.target)
                elif sl_hit:
                    outcome, exit_p = "loss", sig.stop
                elif tp_hit:
                    outcome, exit_p = "win", sig.target
            else:
                sl_hit = h >= sig.stop
                tp_hit = l <= sig.target
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
                sig  = signal_map[ts]
                _df  = df_enriched_for_filter if df_enriched_for_filter is not None else df_5m
                if signal_filter is not None:
                    signal_filter.threshold = signal_filter.threshold  # keep evolved threshold
                    if not signal_filter.accept(sig, _df):
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


def _pass_rate(r_mults, key, n_sims):
    cfg = PROP_FIRM_CONFIGS[key]
    return monte_carlo_barrier(
        r_mults,
        initial_balance=cfg["initial_balance"],
        profit_target_pct=cfg["profit_target_pct"],
        max_drawdown_pct=cfg["max_drawdown_pct"],
        daily_loss_limit_pct=cfg.get("daily_loss_limit_pct"),
        challenge_fee=cfg["challenge_fee"],
        r_to_pct=R_TO_PCT,
        n_sims=n_sims,
    ).pass_rate


def _fmt(v):
    return f"{v*100:6.1f}%" if v is not None else "   N/A"


# ── Per-window genetic search ──────────────────────────────────────────────────

def _run_window_genetic(df_daily_is, df_5m_is):
    """
    Pre-compute signals + outcomes for each zone_lb, fit ML, build pool,
    run GA. Returns (best_zone_lb, best_params, best_ml).
    """
    candidates = []

    for zone_lb in ZONE_LBS:
        print(f"\n    [ZL={zone_lb}] Scanning signals (relaxed min_rr=1.0)...", flush=True)

        signals_all = scan_for_signals(
            "Gold", df_daily_is, df_5m_is,
            min_rr=1.0,
            zone_lookback=zone_lb, cisd_lookback=CISD_LB,
            force_neutral_bias=True, use_news_filter=True,
        )
        print(f"           {len(signals_all)} candidate signals", flush=True)

        if len(signals_all) < 30:
            print(f"           Too few signals — skipping ZL={zone_lb}", flush=True)
            continue

        raw_trades = _compute_raw_outcomes(signals_all, df_5m_is)
        print(f"           {len(raw_trades)} resolved outcomes", flush=True)

        if len(raw_trades) < MIN_TRADES_ML:
            continue

        ml = SignalFilter(probability_threshold=0.60)
        ml_result = ml.fit(raw_trades, df_5m_is)
        if ml_result is None:
            continue

        pool = PrecomputedSignalPool(raw_trades, ml, df_5m_is)
        print(f"           Pool size: {len(pool)}  |  Running GA (pop={GA_POP}, gen={GA_GEN}):", flush=True)

        ga = GeneticOptimizer(pool, pop_size=GA_POP, n_gen=GA_GEN, seed=42)
        best_params, best_fitness = ga.run(verbose=True)

        # Apply evolved ML threshold to the filter object
        ml.threshold = best_params["ml_threshold"]

        # Quick IS validation with evolved params
        r_pool = pool._rmult[
            (pool._rr >= best_params["min_rr"]) &
            (pool._prob >= best_params["ml_threshold"]) &
            (pool._conf >= best_params["min_confidence"])
        ]
        n_pool = len(r_pool)
        ev_pool = float(np.mean(r_pool)) if n_pool > 0 else 0.0
        wr_pool = float(np.mean(r_pool > 0)) if n_pool > 0 else 0.0

        print(f"           → Evolved: rr≥{best_params['min_rr']}  "
              f"ml≥{best_params['ml_threshold']}  "
              f"conf≥{best_params['min_confidence']}  "
              f"fitness={best_fitness:.3f}  "
              f"n={n_pool}  ev={ev_pool:+.3f}R  wr={wr_pool*100:.1f}%", flush=True)

        candidates.append({
            "zone_lb":    zone_lb,
            "params":     best_params,
            "fitness":    best_fitness,
            "ml":         ml,
            "n_pool":     n_pool,
            "ev_pool":    ev_pool,
        })

    if not candidates:
        print("    No candidates — using defaults.", flush=True)
        return 50, {"min_rr": 2.0, "ml_threshold": 0.60, "min_confidence": 1}, None

    best = max(candidates, key=lambda c: c["fitness"])
    print(f"\n    Selected: ZL={best['zone_lb']}  "
          f"rr≥{best['params']['min_rr']}  "
          f"ml≥{best['params']['ml_threshold']}  "
          f"conf≥{best['params']['min_confidence']}  "
          f"(fitness={best['fitness']:.3f})", flush=True)
    return best["zone_lb"], best["params"], best["ml"]


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*70}")
    print("  lnterqo v4 — ROLLING WFV + GENETIC PARAMETER OPTIMISATION")
    print(f"  Data: {COMBINED_5M.name}")
    print(f"  Windows: {N_WINDOWS}  |  IS: anchored  |  OOS per window: {OOS_WINDOW_PCT*100:.0f}%")
    print(f"  GA: pop={GA_POP}  gen={GA_GEN}  cr={GA_CR}  mr={GA_MR}")
    print(f"  Risk: Fixed tiers (0.5% / 1% conf≥4, loss ladder)")
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

    total_bars  = len(df_5m)
    total_start = df_5m.index[0]
    total_end   = df_5m.index[-1]
    print(f"\n  Full range: {total_start.date()} → {total_end.date()}  ({total_bars:,} bars)")

    # Window boundaries
    windows = []
    for w in range(N_WINDOWS):
        is_end_frac  = INITIAL_IS_PCT + w * OOS_WINDOW_PCT
        oos_end_frac = is_end_frac + OOS_WINDOW_PCT
        is_end_bar   = int(total_bars * is_end_frac)
        oos_end_bar  = min(int(total_bars * oos_end_frac), total_bars)
        windows.append((is_end_bar, oos_end_bar))

    print(f"\n  {'Win':>3}  {'IS end':>12}  {'OOS start':>12}  {'OOS end':>12}  {'IS bars':>8}  {'OOS bars':>8}")
    print("  " + "-" * 62)
    for w, (is_end, oos_end) in enumerate(windows):
        print(f"  {w+1:>3}  "
              f"{str(df_5m.index[is_end-1].date()):>12}  "
              f"{str(df_5m.index[is_end].date()):>12}  "
              f"{str(df_5m.index[oos_end-1].date()):>12}  "
              f"{is_end:>8,}  {oos_end-is_end:>8,}")

    all_oos_trades   = []
    window_summaries = []

    for w, (is_end_bar, oos_end_bar) in enumerate(windows):
        oos_start_bar = is_end_bar

        df_5m_is     = df_5m.iloc[:is_end_bar]
        df_5m_oos    = df_5m.iloc[oos_start_bar:oos_end_bar]
        df_daily_is  = df_daily[df_daily.index <= df_5m_is.index[-1]]
        df_daily_oos = df_daily[
            (df_daily.index > df_5m_is.index[-1]) &
            (df_daily.index <= df_5m_oos.index[-1])
        ]

        print(f"\n{'─'*70}")
        print(f"  WINDOW {w+1}/{N_WINDOWS}  "
              f"IS: {df_5m_is.index[0].date()} → {df_5m_is.index[-1].date()}  "
              f"OOS: {df_5m_oos.index[0].date()} → {df_5m_oos.index[-1].date()}")
        print(f"{'─'*70}")

        print("  Enriching IS...", flush=True)
        df_daily_is_enr, df_5m_is_enr, n_news_is = _enrich(df_5m_is, df_daily_is)
        print(f"  IS news events: {n_news_is}")

        print("  Genetic optimisation on IS:", flush=True)
        best_zone_lb, best_params, best_ml = _run_window_genetic(df_daily_is_enr, df_5m_is_enr)

        print("\n  Enriching OOS...", flush=True)
        df_daily_oos_enr, df_5m_oos_enr, n_news_oos = _enrich(df_5m_oos, df_daily_oos)
        print(f"  OOS news events: {n_news_oos}")

        trades_oos, equity_oos = run_lnterqo_backtest_v4(
            "Gold", df_daily_oos_enr, df_5m_oos_enr,
            min_rr=best_params["min_rr"],
            zone_lookback=best_zone_lb,
            cisd_lookback=CISD_LB,
            min_confidence=best_params["min_confidence"],
            force_neutral_bias=True,
            signal_filter=best_ml,
            df_enriched_for_filter=df_5m_oos_enr,
        )

        closed_oos = [t for t in trades_oos if t.outcome in ("win", "loss")]
        metrics    = compute_metrics(trades_oos, equity_oos)
        wr  = float(metrics["win_rate"].rstrip("%")) / 100.0 if metrics and "message" not in metrics and closed_oos else 0.0
        ev  = metrics.get("expectancy_r", 0.0) if metrics and "message" not in metrics else 0.0
        dd  = metrics.get("max_drawdown_pct", 0.0) if metrics and "message" not in metrics else 0.0
        sh  = metrics.get("sharpe_ratio", 0.0) if metrics and "message" not in metrics else 0.0

        print(f"\n  Window {w+1} OOS: {len(closed_oos)} trades | "
              f"WR={wr*100:.1f}% | EV={ev:+.3f}R | DD={dd:+.2f}% | Sharpe={sh:.2f}")

        all_oos_trades.extend(closed_oos)
        window_summaries.append({
            "window":    w + 1,
            "oos_start": df_5m_oos.index[0].date(),
            "oos_end":   df_5m_oos.index[-1].date(),
            "zone_lb":   best_zone_lb,
            "params":    best_params,
            "n_trades":  len(closed_oos),
            "win_rate":  wr,
            "ev_r":      ev,
            "max_dd":    dd,
            "sharpe":    sh,
        })

    # ── Aggregate ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ROLLING WFV — WINDOW-BY-WINDOW SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Win':>3}  {'OOS Period':>24}  {'ZL':>4}  {'rr':>4}  {'ml':>5}  {'c':>2}  "
          f"{'Trd':>5}  {'WR%':>5}  {'EV/R':>6}  {'DD%':>7}  {'Sharpe':>6}")
    print("  " + "-" * 80)
    for s in window_summaries:
        p = s["params"]
        print(f"  {s['window']:>3}  "
              f"{str(s['oos_start'])+' → '+str(s['oos_end']):>24}  "
              f"{s['zone_lb']:>4}  {p['min_rr']:>4}  {p['ml_threshold']:>5}  "
              f"{p['min_confidence']:>2}  "
              f"{s['n_trades']:>5}  {s['win_rate']*100:>4.1f}%  "
              f"{s['ev_r']:>+6.3f}  {s['max_dd']:>+7.2f}%  {s['sharpe']:>6.2f}")

    n_total = len(all_oos_trades)
    wins    = sum(1 for t in all_oos_trades if t.outcome == "win")
    all_r   = _r_multiples(all_oos_trades)

    print(f"\n  Combined OOS: {n_total} trades  |  "
          + (f"WR={wins/n_total*100:.1f}%  |  EV={np.mean(all_r):+.3f}R" if n_total > 0 else "0 trades"))

    if n_total == 0:
        print("\n  No OOS trades. Exiting.")
        return

    os.makedirs("backtest", exist_ok=True)
    pd.DataFrame([{
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
    } for t in all_oos_trades]).to_csv("backtest/lnterqo_v4_oos_trades.csv", index=False)
    print("  Trade log saved: backtest/lnterqo_v4_oos_trades.csv")

    if n_total < MIN_TRADES_MC:
        print(f"  Only {n_total} trades — need ≥{MIN_TRADES_MC}. Done.")
        return

    # ── Statistical validation ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STATISTICAL VALIDATION — COMBINED OOS")
    print(f"{'='*70}")
    oos_start_bar_last = windows[-1][0]
    oos_end_bar_last   = windows[-1][1]
    df_5m_last = df_5m.iloc[oos_start_bar_last:oos_end_bar_last]
    if oos_start_bar_last > 0:
        df_daily_last = df_daily[df_daily.index > df_5m.iloc[oos_start_bar_last - 1].name]
    else:
        df_daily_last = df_daily
    _, df_5m_last_enr, _ = _enrich(df_5m_last, df_daily_last)

    stat = validate(all_oos_trades, df_5m_last_enr, confidence=0.95, n_bootstrap=5_000, n_null_sims=2_000)
    if stat:
        print(stat.report())
        if abs(stat.ev_per_trade) > 1e-6:
            n_needed = required_sample_size(delta=abs(stat.ev_per_trade), sigma=1.0)
            print(f"  Trades to confirm {abs(stat.ev_per_trade):.2f}R edge: {n_needed}")
            print(f"  Combined OOS: {n_total} — {'sufficient' if n_total >= n_needed else 'INSUFFICIENT'}")

    # ── Monte Carlo ───────────────────────────────────────────────────────────
    print(f"\n  Running Monte Carlo (combined OOS)  —  {MC_N_SIMS:,} simulations ...")
    mc1 = monte_carlo_reshuffle(all_r, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
    print(mc1.report())

    try:
        df_regime     = classify_regime(df_5m_last_enr, adx_threshold=25.0, use_hurst=True)
        regime_series = df_regime["regime"].values
        last_trades   = [t for t in all_oos_trades if t.signal.timestamp >= df_5m_last.index[0]]
        if len(last_trades) >= MIN_TRADES_MC:
            trade_regimes = assign_trade_regimes(last_trades, df_regime, regime_series)
            if len(set(trade_regimes)) > 1:
                r_last = _r_multiples(last_trades)
                mc2 = monte_carlo_regime_switching(r_last, trade_regimes, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT)
                print(mc2.report())
    except Exception:
        pass

    mc3 = monte_carlo_barrier(
        all_r, initial_balance=INITIAL_EQUITY,
        profit_target_pct=0.20, max_drawdown_pct=0.20,
        challenge_fee=0.0, n_sims=MC_N_SIMS, r_to_pct=R_TO_PCT,
    )
    print(mc3.report())

    print(f"\n  Prop Firm Pass Rates (combined OOS, {MC_N_SIMS:,} sims):")
    run_prop_firm_simulations(all_r, r_to_pct=R_TO_PCT, n_sims=MC_N_SIMS)

    # ── Comparison table ──────────────────────────────────────────────────────
    oos_start_ts  = df_5m.iloc[windows[0][0]].name
    oos_end_ts    = df_5m.iloc[windows[-1][1] - 1].name
    oos_years     = max((oos_end_ts - oos_start_ts).days / 365.25, 0.01)
    trades_per_yr = n_total / oos_years
    combined_wr   = wins / n_total if n_total else 0
    combined_ev   = float(np.mean(all_r)) if len(all_r) else 0
    combined_dd   = min(s["max_dd"] for s in window_summaries) if window_summaries else 0
    combined_sh   = float(np.mean([s["sharpe"] for s in window_summaries if s["sharpe"] != 0]))
    combined_bar  = mc3.pass_rate * 100

    print(f"\n{'='*70}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*70}")
    hdr = (f"  {'Strategy':<26}  {'Trd/yr':>6}  {'WR%':>6}  {'EV/R':>6}  "
           f"{'MaxDD%':>7}  {'Sharpe':>6}  {'Barrier':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    def _row(name, tpy, wr, ev, dd, sh, bar):
        return (f"  {name:<26}  {tpy:>6.1f}  {wr:>5.1f}%  {ev:>+6.3f}  "
                f"{dd:>+7.2f}%  {sh:>6.2f}  {bar:>7.1f}%")

    def _d(cfg): return cfg["trades"], cfg["wr"], cfg["ev_r"], cfg["max_dd"], cfg["sharpe"], cfg["barrier"]
    print(_row("ICT v1",                   *_d(_ICT_V1)))
    print(_row("lnterqo v1 (80/20)",       *_d(_LNT_V1)))
    print(_row("lnterqo v2 (rolling+EG)",  *_d(_LNT_V2)))
    print(_row("lnterqo v3 (rolling+fix)", *_d(_LNT_V3)))
    print(_row("lnterqo v4 (rolling+GA)",  trades_per_yr, combined_wr * 100, combined_ev,
               combined_dd, combined_sh, combined_bar))
    print("  " + "-" * (len(hdr) - 2))

    print(f"\n  Notes:")
    print(f"  - v4 covers {n_total} trades across {N_WINDOWS} OOS windows ({oos_years:.1f} yrs)")
    print(f"  - GA evolved params per window instead of fixed 3-variation grid")
    print(f"  - OOS span: {oos_start_ts.date()} → {oos_end_ts.date()}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
