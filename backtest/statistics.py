"""
Statistical Validation Layer
DeltaTrend Notes: DeltaTrend — Mathematics, Theory & Application.md

Implements Thomas's quant workflow:
  1. Expected Value with Bootstrap CI  (SMM272 Bootstrap VaR)
  2. Null model test — does timing beat random?  (SMM270 Econometrics)
  3. Sample size requirements (how many trades do we need?)
  4. Return distribution analysis (skewness, kurtosis, full moments)
  5. Information Coefficient estimation  (SMM282 Quant Trading)

Reference: Grinold (1989) Fundamental Law of Active Management:
  IR = IC × sqrt(BR)   — information ratio = skill × breadth
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from backtest.engine import Trade


@dataclass
class StatSummary:
    n_trades: int
    ev_per_trade: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    ci_spans_zero: bool
    win_rate: float
    profit_factor: float
    sharpe: float
    skewness: float
    kurtosis: float
    null_model_ev: float
    null_model_std: float
    p_value: float
    timing_significant: bool
    ic: float

    def report(self) -> str:
        verdict = "SIGNIFICANT EDGE" if not self.ci_spans_zero and self.timing_significant else (
            "WEAK SIGNAL (timing real, CI spans zero)" if self.timing_significant else "NO EDGE DETECTED"
        )
        return (
            f"\n{'='*55}\n"
            f"  STATISTICAL VALIDATION — {verdict}\n"
            f"{'='*55}\n"
            f"  Trades analysed          : {self.n_trades}\n"
            f"  EV per trade             : {self.ev_per_trade:+.4f}R\n"
            f"  {self.ci_level*100:.0f}% Bootstrap CI         : [{self.ci_lower:+.4f}R, {self.ci_upper:+.4f}R]\n"
            f"  CI spans zero            : {self.ci_spans_zero}  ← must be False for valid edge\n"
            f"  Null model p-value       : {self.p_value:.4f}  ← must be < 0.05\n"
            f"  Timing significant       : {self.timing_significant}\n"
            f"  Information Coefficient  : {self.ic:.4f}  ← IC > 0.02 is meaningful\n"
            f"  Win rate                 : {self.win_rate:.1%}\n"
            f"  Profit factor            : {self.profit_factor:.2f}\n"
            f"  Skewness                 : {self.skewness:.3f}  (>0 = right-skewed, rare big wins)\n"
            f"  Excess kurtosis          : {self.kurtosis:.3f}  (>0 = fat tails)\n"
            f"  Sharpe ratio (ann.)      : {self.sharpe:.3f}\n"
            f"{'='*55}"
        )


# ─── Expected Value ────────────────────────────────────────────────────────────

def expected_value(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """EV = P(win) × avg_win + P(loss) × avg_loss  (Thomas's formula)"""
    return win_rate * avg_win + (1 - win_rate) * avg_loss


def ev_from_trades(trades: list[Trade]) -> float:
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if not closed:
        return 0.0
    return np.mean([t.r_multiple for t in closed])


# ─── Bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    r_multiples: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for EV per trade.
    SMM272: Bootstrap VaR & Backtesting — non-parametric CI without normality assumption.

    Returns: (lower, upper, sample_ev)
    If CI spans zero → cannot reject H₀: EV = 0 at (1-confidence) significance level.
    """
    n = len(r_multiples)
    sample_ev = float(np.mean(r_multiples))
    boot_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boot_means[i] = np.mean(np.random.choice(r_multiples, size=n, replace=True))
    alpha = 1 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper, sample_ev


def required_sample_size(
    delta: float = 0.1,
    sigma: float = 1.0,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """
    Minimum number of trades to detect edge of size delta (in R-multiples).
    Formula: n ≈ (z_α + z_β)² × σ² / δ²   (Thomas's derivation)

    Default: detect a 0.1R edge at 5% significance with 80% power ≈ 618 trades.
    """
    z_alpha = stats.norm.ppf(1 - alpha)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil((z_alpha + z_beta) ** 2 * sigma ** 2 / delta ** 2))


# ─── Return Distribution Analysis ─────────────────────────────────────────────

def distribution_stats(r_multiples: np.ndarray) -> dict:
    """Full moments of the trade return distribution."""
    return {
        "n": len(r_multiples),
        "mean_ev": float(np.mean(r_multiples)),
        "std": float(np.std(r_multiples)),
        "win_rate": float(np.mean(r_multiples > 0)),
        "skewness": float(stats.skew(r_multiples)),
        "kurtosis": float(stats.kurtosis(r_multiples)),  # excess kurtosis
        "min": float(np.min(r_multiples)),
        "max": float(np.max(r_multiples)),
        "p25": float(np.percentile(r_multiples, 25)),
        "p75": float(np.percentile(r_multiples, 75)),
    }


# ─── Null Model Test ───────────────────────────────────────────────────────────

def null_model_test(
    actual_ev: float,
    n_trades: int,
    df: pd.DataFrame,
    stop_r: float = 1.0,
    target_r: float = 2.0,
    n_sims: int = 5_000,
) -> tuple[float, np.ndarray]:
    """
    Test whether strategy timing beats random entry.
    SMM270 Econometrics: p-value, null hypothesis H₀: timing has no signal.

    Randomly samples n_trades timestamps from df and computes EV using the same
    stop/target structure. p-value = P(random EV ≥ actual EV).

    Returns: (p_value, null_ev_distribution)
    """
    if "atr" not in df.columns:
        return 1.0, np.zeros(n_sims)

    valid_idx = np.where(~df["atr"].isna())[0]
    if len(valid_idx) < n_trades + 20:
        return 1.0, np.zeros(n_sims)

    null_evs = np.empty(n_sims)

    for sim in range(n_sims):
        chosen = np.random.choice(valid_idx[:-20], size=n_trades, replace=False)
        r_mults = []
        for pos in chosen:
            entry = df.iloc[pos]["close"]
            atr = df.iloc[pos]["atr"]
            if atr == 0 or pd.isna(atr):
                continue
            stop = entry - stop_r * atr
            target = entry + target_r * atr
            future = df.iloc[pos + 1: pos + 21]
            outcome = 0.0
            for _, bar in future.iterrows():
                if bar["low"] <= stop:
                    outcome = -stop_r
                    break
                if bar["high"] >= target:
                    outcome = target_r
                    break
            r_mults.append(outcome)
        null_evs[sim] = np.mean(r_mults) if r_mults else 0.0

    p_value = float(np.mean(null_evs >= actual_ev))
    return p_value, null_evs


# ─── Information Coefficient ───────────────────────────────────────────────────

def information_coefficient(signals: np.ndarray, forward_returns: np.ndarray) -> float:
    """
    IC = Pearson correlation between signal values and forward returns.
    SMM282 Quant Trading / Fundamental Law: IR = IC × sqrt(BR).

    signals: array of signal values (+1/-1 or continuous forecast)
    forward_returns: array of actual returns over the forward window
    """
    if len(signals) < 5:
        return 0.0
    mask = ~(np.isnan(signals) | np.isnan(forward_returns))
    if mask.sum() < 5:
        return 0.0
    return float(np.corrcoef(signals[mask], forward_returns[mask])[0, 1])


def fundamental_law(ic: float, breadth: float) -> float:
    """
    Grinold (1989): IR = IC × sqrt(BR)
    ir = IC × sqrt(number of independent bets per year)
    """
    return ic * np.sqrt(max(breadth, 0))


# ─── Full Validation Pipeline ─────────────────────────────────────────────────

def validate(
    trades: list[Trade],
    df: pd.DataFrame,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    n_null_sims: int = 3_000,
) -> StatSummary:
    """
    Run the full Thomas-style statistical validation on a set of trades.
    """
    closed = [t for t in trades if t.outcome in ("win", "loss")]
    if len(closed) < 3:
        print("  Too few closed trades for statistical validation (need ≥ 3).")
        return None

    r_mults = np.array([t.r_multiple for t in closed])
    wins = r_mults[r_mults > 0]
    losses = r_mults[r_mults <= 0]
    win_rate = len(wins) / len(r_mults)
    gross_profit = wins.sum() if len(wins) else 0.0
    gross_loss = abs(losses.sum()) if len(losses) else 1e-9
    profit_factor = gross_profit / gross_loss

    # Bootstrap CI
    ci_lower, ci_upper, ev = bootstrap_ci(r_mults, confidence, n_bootstrap)
    ci_spans_zero = ci_lower < 0 < ci_upper

    # Sharpe (annualised from R-multiples, assuming 252 trading days)
    sharpe = (np.mean(r_mults) / np.std(r_mults) * np.sqrt(252)) if np.std(r_mults) > 0 else 0.0

    # Distribution
    dist = distribution_stats(r_mults)

    # Null model
    p_value, null_dist = null_model_test(
        actual_ev=ev,
        n_trades=len(closed),
        df=df,
        stop_r=1.0,
        target_r=2.0,
        n_sims=n_null_sims,
    )
    timing_significant = p_value < 0.05

    # IC (use +1/-1 as signal based on direction)
    signal_vals = np.array([1 if t.signal.direction == "long" else -1 for t in closed])
    ic = information_coefficient(signal_vals, r_mults)

    return StatSummary(
        n_trades=len(closed),
        ev_per_trade=round(ev, 4),
        ci_lower=round(ci_lower, 4),
        ci_upper=round(ci_upper, 4),
        ci_level=confidence,
        ci_spans_zero=ci_spans_zero,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 3),
        sharpe=round(sharpe, 3),
        skewness=round(dist["skewness"], 3),
        kurtosis=round(dist["kurtosis"], 3),
        null_model_ev=round(float(np.mean(null_dist)), 4),
        null_model_std=round(float(np.std(null_dist)), 4),
        p_value=round(p_value, 4),
        timing_significant=timing_significant,
        ic=round(ic, 4),
    )
