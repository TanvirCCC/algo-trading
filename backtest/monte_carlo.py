"""
Monte Carlo Simulation — Three Methods
DeltaTrend Notes: Methods 1, 2, 3 from Mathematics & Theory file

Method 1: Basic Reshuffling (Bootstrap Resampling)
  — shows path dependence; same EV can produce very different equity curves
  — SMM302 Stochastic Modelling: stochastic processes, path dependence

Method 2: Regime-Switching Monte Carlo
  — preserves clustering of trade returns by regime (Markov chain)
  — SMM302: Markov chains, transition matrix, steady-state vector

Method 3: Barrier Simulation (Trading Game / Prop Firm)
  — models £10,000 account with profit target and max drawdown as barriers
  — SMM254 Derivatives: barrier option / first-passage-time problem
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class MCResult:
    method: str
    n_sims: int
    mean_terminal_pnl: float
    pct_profitable: float
    p5_terminal: float
    p50_terminal: float
    p95_terminal: float
    p90_max_drawdown: float
    p50_max_drawdown: float
    ev_per_trade: float
    ci_lower: float
    ci_upper: float

    def report(self, initial_equity: float = 10_000.0) -> str:
        lines = [
            f"\n{'─'*50}",
            f"  Monte Carlo — {self.method}",
            f"{'─'*50}",
            f"  Simulations          : {self.n_sims:,}",
            f"  EV per trade         : {self.ev_per_trade:+.4f}R",
            f"  95% CI on EV         : [{self.ci_lower:+.4f}R, {self.ci_upper:+.4f}R]",
            f"  % paths profitable   : {self.pct_profitable:.1%}",
            f"  5th pct balance      : £{initial_equity * (1 + self.p5_terminal):,.0f}",
            f"  Median balance       : £{initial_equity * (1 + self.p50_terminal):,.0f}",
            f"  95th pct balance     : £{initial_equity * (1 + self.p95_terminal):,.0f}",
            f"  Median max drawdown  : {self.p50_max_drawdown:.1%}",
            f"  90th pct max drawdown: {self.p90_max_drawdown:.1%}  ← realistic worst case",
            f"{'─'*50}",
        ]
        return "\n".join(lines)


# ─── Method 1: Basic Reshuffling ──────────────────────────────────────────────

def monte_carlo_reshuffle(
    r_multiples: np.ndarray,
    n_sims: int = 10_000,
    n_trades: int = None,
    r_to_pct: float = 0.02,   # 1R = 2% of account (2% risk per trade)
) -> MCResult:
    """
    Method 1: Bootstrap resampling of trade returns.
    Samples with replacement to show distribution of possible equity paths.
    SMM302: one backtest = one realisation of a stochastic process.

    r_to_pct: converts R-multiples to % account moves (1R = risk_pct of account)
    """
    if n_trades is None:
        n_trades = len(r_multiples)

    returns_pct = r_multiples * r_to_pct

    terminal = np.empty(n_sims)
    max_dd = np.empty(n_sims)

    for i in range(n_sims):
        path_r = np.random.choice(returns_pct, size=n_trades, replace=True)
        equity = np.cumprod(1 + path_r)
        terminal[i] = equity[-1] - 1
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd[i] = dd.min()

    # Bootstrap CI on EV
    ev = np.mean(r_multiples)
    n = len(r_multiples)
    boot_evs = np.array([np.mean(np.random.choice(r_multiples, n, replace=True))
                         for _ in range(2_000)])
    ci_lo = float(np.percentile(boot_evs, 2.5))
    ci_hi = float(np.percentile(boot_evs, 97.5))

    return MCResult(
        method="Basic Reshuffling",
        n_sims=n_sims,
        mean_terminal_pnl=float(np.mean(terminal)),
        pct_profitable=float(np.mean(terminal > 0)),
        p5_terminal=float(np.percentile(terminal, 5)),
        p50_terminal=float(np.percentile(terminal, 50)),
        p95_terminal=float(np.percentile(terminal, 95)),
        p90_max_drawdown=float(np.percentile(max_dd, 90)),
        p50_max_drawdown=float(np.percentile(max_dd, 50)),
        ev_per_trade=round(ev, 4),
        ci_lower=round(ci_lo, 4),
        ci_upper=round(ci_hi, 4),
    )


# ─── Method 2: Regime-Switching Monte Carlo ───────────────────────────────────

def build_transition_matrix(regime_sequence: np.ndarray) -> tuple[np.ndarray, list]:
    """
    Build Markov transition matrix from regime label sequence.
    SMM302: P_ij = P(next = j | current = i). Each row sums to 1.
    """
    regimes = sorted(set(regime_sequence))
    idx = {r: i for i, r in enumerate(regimes)}
    K = len(regimes)
    counts = np.zeros((K, K))
    for t in range(len(regime_sequence) - 1):
        i, j = idx[regime_sequence[t]], idx[regime_sequence[t + 1]]
        counts[i, j] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / np.where(row_sums > 0, row_sums, 1)
    return P, regimes


def markov_steady_state(P: np.ndarray) -> np.ndarray:
    """
    Compute steady-state distribution π* where π* = π* × P.
    SMM302: left eigenvector of P corresponding to eigenvalue 1.
    Solving: π*(P - I) = 0 with Σπ*_i = 1.
    """
    from scipy.linalg import eig
    eigenvalues, eigenvectors = eig(P.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    steady = np.real(eigenvectors[:, idx])
    steady = steady / steady.sum()
    return steady


def monte_carlo_regime_switching(
    r_multiples: np.ndarray,
    regime_labels: np.ndarray,
    n_sims: int = 10_000,
    n_trades: int = None,
    r_to_pct: float = 0.02,
) -> MCResult:
    """
    Method 2: Regime-aware Monte Carlo.
    Preserves clustering of returns by regime using Markov transition matrix.
    More realistic than pure reshuffling when trending/choppy regimes persist.
    SMM302: Markov chains — next state depends only on current state (memoryless).
    """
    if n_trades is None:
        n_trades = len(r_multiples)

    P, regimes = build_transition_matrix(regime_labels)
    n_regimes = len(regimes)
    regime_to_idx = {r: i for i, r in enumerate(regimes)}
    pools = {regime_to_idx[r]: r_multiples[regime_labels == r] * r_to_pct
             for r in regimes if (regime_labels == r).sum() > 0}

    steady = markov_steady_state(P)

    terminal = np.empty(n_sims)
    max_dd = np.empty(n_sims)

    for sim in range(n_sims):
        current = np.random.choice(n_regimes, p=steady)
        equity = [1.0]
        for _ in range(n_trades):
            pool = pools.get(current, np.array([0.0]))
            r = np.random.choice(pool)
            equity.append(equity[-1] * (1 + r))
            current = np.random.choice(n_regimes, p=P[current])
        equity = np.array(equity)
        terminal[sim] = equity[-1] - 1
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dd[sim] = dd.min()

    ev = np.mean(r_multiples)
    n = len(r_multiples)
    boot_evs = np.array([np.mean(np.random.choice(r_multiples, n, replace=True))
                         for _ in range(2_000)])

    return MCResult(
        method="Regime-Switching (Markov)",
        n_sims=n_sims,
        mean_terminal_pnl=float(np.mean(terminal)),
        pct_profitable=float(np.mean(terminal > 0)),
        p5_terminal=float(np.percentile(terminal, 5)),
        p50_terminal=float(np.percentile(terminal, 50)),
        p95_terminal=float(np.percentile(terminal, 95)),
        p90_max_drawdown=float(np.percentile(max_dd, 90)),
        p50_max_drawdown=float(np.percentile(max_dd, 50)),
        ev_per_trade=round(ev, 4),
        ci_lower=round(float(np.percentile(boot_evs, 2.5)), 4),
        ci_upper=round(float(np.percentile(boot_evs, 97.5)), 4),
    )


# ─── Method 3: Barrier Simulation (Trading Game Account) ─────────────────────

@dataclass
class BarrierResult:
    pass_rate: float
    mean_payout_on_pass: float
    net_ev: float
    break_even_pass_rate: float
    p90_max_drawdown: float
    n_sims: int

    def report(self, initial: float = 10_000.0) -> str:
        verdict = "POSITIVE EV" if self.net_ev > 0 else "NEGATIVE EV"
        lines = [
            f"\n{'─'*50}",
            f"  Trading Game Account — Barrier Simulation",
            f"  ({verdict})",
            f"{'─'*50}",
            f"  Simulations          : {self.n_sims:,}",
            f"  Pass rate            : {self.pass_rate:.1%}",
            f"  Break-even pass rate : {self.break_even_pass_rate:.1%}",
            f"  Mean payout if pass  : £{self.mean_payout_on_pass:,.0f}",
            f"  Net EV per attempt   : £{self.net_ev:+,.0f}",
            f"  90th pct drawdown    : {self.p90_max_drawdown:.1%}",
            f"{'─'*50}",
        ]
        return "\n".join(lines)


# ─── Prop Firm Challenge Configs ─────────────────────────────────────────────
PROP_FIRM_CONFIGS = {
    "FTMO_100k": dict(
        initial_balance=100_000, profit_target_pct=0.10,
        max_drawdown_pct=0.10, daily_loss_limit_pct=0.05,
        challenge_fee=540, label="FTMO 100k Phase 1",
    ),
    "FTMO_50k": dict(
        initial_balance=50_000, profit_target_pct=0.10,
        max_drawdown_pct=0.10, daily_loss_limit_pct=0.05,
        challenge_fee=345, label="FTMO 50k Phase 1",
    ),
    "TopStep_50k": dict(
        initial_balance=50_000, profit_target_pct=0.06,
        max_drawdown_pct=0.04, daily_loss_limit_pct=0.02,
        challenge_fee=165, label="TopStep 50k",
    ),
    "The5ers_100k": dict(
        initial_balance=100_000, profit_target_pct=0.06,
        max_drawdown_pct=0.04, daily_loss_limit_pct=None,
        challenge_fee=299, label="The5%ers 100k",
    ),
    "TradingGame": dict(
        initial_balance=10_000, profit_target_pct=0.20,
        max_drawdown_pct=0.20, daily_loss_limit_pct=None,
        challenge_fee=0.0, label="Trading Game (SMM591)",
    ),
}


def monte_carlo_barrier(
    r_multiples: np.ndarray,
    initial_balance: float = 10_000.0,
    profit_target_pct: float = 0.20,   # 20% profit target (Trading Game)
    max_drawdown_pct: float = 0.20,    # 20% drawdown limit (lose all)
    daily_loss_limit_pct: float = None, # None = no daily limit
    challenge_fee: float = 0.0,        # no fee for Trading Game (demo account)
    max_trades: int = 200,
    r_to_pct: float = 0.02,
    n_sims: int = 10_000,
) -> BarrierResult:
    """
    Method 3: Barrier/first-passage-time simulation.
    Models account challenge as: pass = hit upper barrier; fail = hit lower barrier.
    SMM254 Derivatives: barrier option / knock-out option analogue.

    For the Trading Game: upper barrier = performance prize (top 5 mark bonus),
    lower barrier = embarrassing loss (bottom 5 mark penalty).
    """
    B_up   = initial_balance * (1 + profit_target_pct)
    B_down = initial_balance * (1 - max_drawdown_pct)
    daily_limit = initial_balance * daily_loss_limit_pct if daily_loss_limit_pct else None

    returns_pct = r_multiples * r_to_pct
    passed_list = []
    final_balances = []
    max_dd_list = []

    trades_per_day = max(1, max_trades // 20)  # rough estimate for daily grouping

    for _ in range(n_sims):
        balance = initial_balance
        peak = initial_balance
        max_dd = 0.0
        passed = False
        day_start_balance = initial_balance
        trades_today = 0

        for trade_num in range(max_trades):
            # Reset daily loss tracking
            if trades_today >= trades_per_day:
                day_start_balance = balance
                trades_today = 0

            r = np.random.choice(returns_pct)
            balance *= (1 + r)
            trades_today += 1

            if balance > peak:
                peak = balance
            dd = (balance - peak) / peak
            max_dd = min(max_dd, dd)

            # Daily loss limit breach
            if daily_limit and (day_start_balance - balance) >= daily_limit:
                break

            if balance >= B_up:
                passed = True
                break
            if balance <= B_down:
                break

        passed_list.append(passed)
        final_balances.append(balance)
        max_dd_list.append(max_dd)

    pass_rate = np.mean(passed_list)
    passing_balances = [b for b, p in zip(final_balances, passed_list) if p]
    mean_payout = np.mean(passing_balances) - initial_balance if passing_balances else 0.0

    # Break-even: P(pass) × payout > P(fail) × fee
    # → P(pass) > fee / (payout + fee)
    break_even = challenge_fee / (mean_payout + challenge_fee) if (mean_payout + challenge_fee) > 0 else 0.0
    net_ev = pass_rate * mean_payout - (1 - pass_rate) * challenge_fee

    return BarrierResult(
        pass_rate=round(pass_rate, 4),
        mean_payout_on_pass=round(mean_payout, 2),
        net_ev=round(net_ev, 2),
        break_even_pass_rate=round(break_even, 4),
        p90_max_drawdown=round(float(np.percentile(max_dd_list, 90)), 4),
        n_sims=n_sims,
    )


def run_prop_firm_simulations(
    r_multiples: np.ndarray,
    r_to_pct: float = 0.02,
    n_sims: int = 5_000,
) -> None:
    """Print prop firm pass-rate table for all configured firms."""
    print(f"\n{'═'*62}")
    print(f"  PROP FIRM CHALLENGE ANALYSIS  (DeltaTrend barrier model)")
    print(f"{'═'*62}")
    print(f"  {'Firm':<22} {'Fee':>6}  {'Pass%':>6}  {'Net EV':>10}  {'DD 90th':>8}")
    print(f"  {'─'*56}")
    for key, cfg in PROP_FIRM_CONFIGS.items():
        result = monte_carlo_barrier(
            r_multiples,
            initial_balance=cfg["initial_balance"],
            profit_target_pct=cfg["profit_target_pct"],
            max_drawdown_pct=cfg["max_drawdown_pct"],
            daily_loss_limit_pct=cfg.get("daily_loss_limit_pct"),
            challenge_fee=cfg["challenge_fee"],
            r_to_pct=r_to_pct,
            n_sims=n_sims,
        )
        ev_str = f"£{result.net_ev:+,.0f}" if result.net_ev != 0 else "N/A"
        print(f"  {cfg['label']:<22} £{cfg['challenge_fee']:>5,.0f}  "
              f"{result.pass_rate:>5.1%}  {ev_str:>10}  "
              f"{result.p90_max_drawdown:>7.1%}")
    print(f"{'═'*62}")


# ─── Regime label generator (for Method 2) ────────────────────────────────────

def label_regimes_by_adx(df: pd.DataFrame, adx_threshold: float = 25.0) -> np.ndarray:
    """
    Classify bars as Trending or Choppy using ADX.
    ADX > threshold → Trending; else Choppy.
    SMM748 ML / SMM282 Quant Trading: regime filtering.
    """
    if "ADX" not in df.columns:
        from detectors.regime import add_adx
        df = add_adx(df)
    regime = np.where(df["ADX"].fillna(0) > adx_threshold, "Trending", "Choppy")
    return regime


def assign_trade_regimes(
    trades: list,
    df: pd.DataFrame,
    regime_series: np.ndarray,
) -> np.ndarray:
    """Map each trade's timestamp to its regime label."""
    labels = []
    for t in trades:
        if t.outcome not in ("win", "loss"):
            continue
        ts = t.signal.timestamp
        if ts in df.index:
            pos = df.index.get_loc(ts)
            if pos < len(regime_series):
                labels.append(regime_series[pos])
                continue
        labels.append("Choppy")
    return np.array(labels)
