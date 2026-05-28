"""
Regime Detection
DeltaTrend Notes: Section 12 — Regime Filtering Mathematics

Implements:
  1. ADX (Average Directional Index) — trend strength  (CDT03 + SMM282)
  2. Hurst Exponent — trending vs mean-reverting        (SMM302 + Thomas)
  3. CUSUM Event Detector — directional pressure events (Thomas Ep03)
  4. Regime classification from ADX + Hurst             (SMM748 ML)
  5. Markov transition matrix from regime sequence      (SMM302 Stochastic)

These are used to:
  (a) filter ICT signals to only take trades in the right regime
  (b) label trades for regime-switching Monte Carlo
  (c) compute steady-state time spent in trending vs choppy
"""

import numpy as np
import pandas as pd
from scipy.linalg import eig


# ─── ADX ──────────────────────────────────────────────────────────────────────

def add_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI.
    Thomas's full derivation: measures trend STRENGTH (not direction).
    ADX > 25 = strong trend; ADX < 20 = choppy/ranging.
    CDT03: directional movement indicator.
    SMM282 Quant Trading: trend-following filter.
    """
    df = df.copy()
    df["+DM"] = np.where(
        (df["high"] - df["high"].shift(1)) > (df["low"].shift(1) - df["low"]),
        np.maximum(df["high"] - df["high"].shift(1), 0), 0,
    )
    df["-DM"] = np.where(
        (df["low"].shift(1) - df["low"]) > (df["high"] - df["high"].shift(1)),
        np.maximum(df["low"].shift(1) - df["low"], 0), 0,
    )
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)

    alpha = 1 / window
    atr_ = tr.ewm(alpha=alpha, adjust=False).mean()
    pdm  = df["+DM"].ewm(alpha=alpha, adjust=False).mean()
    mdm  = df["-DM"].ewm(alpha=alpha, adjust=False).mean()

    df["+DI"] = 100 * pdm / atr_.replace(0, np.nan)
    df["-DI"] = 100 * mdm / atr_.replace(0, np.nan)
    df["DX"]  = 100 * (df["+DI"] - df["-DI"]).abs() / (df["+DI"] + df["-DI"]).replace(0, np.nan)
    df["ADX"] = df["DX"].ewm(alpha=alpha, adjust=False).mean()
    df.drop(columns=["+DM", "-DM"], inplace=True)
    return df


# ─── Hurst Exponent ───────────────────────────────────────────────────────────

def hurst_exponent(series: np.ndarray, min_lag: int = 10, max_lag: int = None) -> float:
    """
    Estimate Hurst exponent via R/S (Rescaled Range) analysis.
    SMM302 Stochastic Modelling: long-range dependence.

    H > 0.5 → trending / persistent (momentum works)
    H = 0.5 → random walk (GBM)
    H < 0.5 → mean-reverting / anti-persistent

    Applied to 20–50 bar rolling windows to classify current regime.
    """
    n = len(series)
    if max_lag is None:
        max_lag = n // 4
    max_lag = max(max_lag, min_lag + 1)
    if n < max_lag:
        return 0.5

    rs_vals, lags = [], []
    for lag in range(min_lag, max_lag):
        n_blocks = n // lag
        if n_blocks == 0:
            continue
        rs_block = []
        for b in range(n_blocks):
            block = series[b * lag: (b + 1) * lag]
            mean = np.mean(block)
            dev = np.cumsum(block - mean)
            R = dev.max() - dev.min()
            S = np.std(block, ddof=1)
            if S > 0:
                rs_block.append(R / S)
        if rs_block:
            rs_vals.append(np.mean(rs_block))
            lags.append(lag)

    if len(lags) < 2:
        return 0.5

    log_lags = np.log(lags)
    log_rs   = np.log(rs_vals)
    H = float(np.polyfit(log_lags, log_rs, 1)[0])
    return np.clip(H, 0.0, 1.0)


def rolling_hurst(prices: pd.Series, window: int = 50) -> pd.Series:
    """Rolling Hurst exponent over a lookback window."""
    returns = prices.pct_change().dropna()
    h_vals = []
    for i in range(len(returns)):
        if i < window:
            h_vals.append(0.5)
        else:
            chunk = returns.iloc[i - window: i].values
            h_vals.append(hurst_exponent(chunk))
    result = pd.Series(h_vals, index=returns.index)
    # Align back to original price index
    aligned = pd.Series(np.nan, index=prices.index)
    aligned.loc[result.index] = result.values
    return aligned.ffill()


# ─── Regime Classification ────────────────────────────────────────────────────

def classify_regime(
    df: pd.DataFrame,
    adx_threshold: float = 25.0,
    hurst_threshold: float = 0.55,
    use_hurst: bool = True,
) -> pd.DataFrame:
    """
    Classify each bar as Trending or Choppy.

    Trending:  ADX > adx_threshold  AND (Hurst > hurst_threshold if use_hurst)
    Choppy:    otherwise

    For ICT strategy: only take signals when regime = Trending
    (false breakouts are more reliable in trending, not choppy, markets).

    SMM748 ML: regime classification as a categorical feature.
    SMM282: regime filter to improve strategy EV.
    """
    df = df.copy()
    if "ADX" not in df.columns:
        df = add_adx(df)

    adx_trending = df["ADX"] > adx_threshold

    if use_hurst:
        df["hurst"] = rolling_hurst(df["close"], window=50)
        hurst_trending = df["hurst"] > hurst_threshold
        df["regime"] = np.where(adx_trending & hurst_trending, "Trending", "Choppy")
    else:
        df["regime"] = np.where(adx_trending, "Trending", "Choppy")

    df["is_trending"] = df["regime"] == "Trending"
    return df


# ─── CUSUM Event Detector ─────────────────────────────────────────────────────

def cusum_events(df: pd.DataFrame, k_multiple: float = 1.0, atr_window: int = 14) -> pd.DataFrame:
    """
    ATR-normalised CUSUM (Cumulative Sum) event detector.
    Thomas Ep03: fires events when accumulated directional pressure exceeds ATR threshold.
    Resets after each event.

    S+_t = max(0, S+_{t-1} + r_t)   — upward pressure accumulator
    S-_t = max(0, S-_{t-1} - r_t)   — downward pressure accumulator
    Fire event when S > k × ATR. Reset to 0 after firing.

    Returns df with:
      cusum_up   : 1 when upward event fires
      cusum_down : -1 when downward event fires
    """
    df = df.copy()
    if "atr" not in df.columns:
        from detectors.indicators import add_atr
        df = add_atr(df, atr_window)

    returns = df["close"].pct_change().fillna(0).values
    atr_norm = (df["atr"] / df["close"]).fillna(0).values
    threshold = k_multiple * atr_norm

    n = len(df)
    s_up = np.zeros(n)
    s_down = np.zeros(n)
    events = np.zeros(n, dtype=int)

    for i in range(1, n):
        r = returns[i]
        h = threshold[i]
        s_up[i]   = max(0, s_up[i - 1]   + r)
        s_down[i] = max(0, s_down[i - 1] - r)
        if s_up[i] > h:
            events[i] = 1
            s_up[i] = 0
        elif s_down[i] > h:
            events[i] = -1
            s_down[i] = 0

    df["cusum_up"]   = (events == 1).astype(int)
    df["cusum_down"] = (events == -1).astype(int)
    df["cusum_event"] = events
    return df


# ─── Markov Steady State ──────────────────────────────────────────────────────

def regime_markov_analysis(regime_series: np.ndarray) -> dict:
    """
    Build transition matrix and compute steady state for regime sequence.
    SMM302 Stochastic Modelling: π* = π* × P, Σπ*_i = 1.
    Left eigenvector of P with eigenvalue 1.

    Returns:
      transition_matrix: P (K×K)
      steady_state: π* (K,)
      regimes: list of state names
      state_evolution: shows convergence to π* over 10 steps
    """
    regimes = sorted(set(regime_series))
    idx = {r: i for i, r in enumerate(regimes)}
    K = len(regimes)

    counts = np.zeros((K, K))
    for t in range(len(regime_series) - 1):
        i, j = idx[regime_series[t]], idx[regime_series[t + 1]]
        counts[i, j] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    P = counts / np.where(row_sums > 0, row_sums, 1)

    # Steady state: left eigenvector for eigenvalue 1
    eigenvalues, eigenvectors = eig(P.T)
    idx_ev = np.argmin(np.abs(eigenvalues - 1.0))
    pi_star = np.real(eigenvectors[:, idx_ev])
    pi_star = pi_star / pi_star.sum()

    # State evolution over 10 steps from uniform start
    pi_0 = np.ones(K) / K
    evolution = [pi_0]
    for _ in range(10):
        evolution.append(evolution[-1] @ P)

    print(f"\n  Regime Markov Analysis")
    print(f"  States: {regimes}")
    print(f"  Transition matrix P:")
    for i, r in enumerate(regimes):
        row = "  " + " | ".join(f"{P[i,j]:.3f}" for j in range(K))
        print(f"    {r:10s}: {row}")
    print(f"  Steady state π*: " + " | ".join(f"{r}: {pi_star[i]:.3f}" for i, r in enumerate(regimes)))

    return {
        "transition_matrix": P,
        "steady_state": pi_star,
        "regimes": regimes,
        "state_evolution": np.array(evolution),
    }
