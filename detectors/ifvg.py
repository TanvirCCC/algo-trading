"""
IFVG / IOF — Inverted Fair Value Gap / Inversion of FVG
lnterqo Notes: Trading Notes/lnterqo — Strategy Analysis.md

When price fully closes THROUGH a bullish FVG (close < FVG bot) → the zone inverts
to a bearish resistance zone (IFVG bear).

When price fully closes THROUGH a bearish FVG (close > FVG top) → inverts to
bullish support zone (IFVG bull).

lnterqo prefers IFVG over raw FVG for retracement entries because:
  - The inversion confirms change of character — mitigated and flipped
  - Higher probability than an untested FVG because the zone has already been tested
  - He marks IFVG in pink/salmon; IOF in blue on his charts
"""

import numpy as np
import pandas as pd
from collections import deque


def build_ifvg_zones(df: pd.DataFrame, lookback: int = 200, atr_mult: float = 0.3) -> dict:
    """
    Scan `lookback` bars and return active IFVG zones (inverted FVGs still valid).

    A zone is invalidated when price closes through the far edge.

    Returns:
      {
        "bull": [{"top", "bot", "formed_i", "formed_ts"}, ...],  # inverted bearish → now support
        "bear": [{"top", "bot", "formed_i", "formed_ts"}, ...],  # inverted bullish → now resistance
      }
    """
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    n      = len(df)

    atr = _atr_values(df)

    bull_ifvg = []
    bear_ifvg = []

    start = max(2, n - lookback)

    # First pass: collect all FVGs in window
    raw_bull_fvgs = []  # (i, top, bot)
    raw_bear_fvgs = []

    for i in range(start, n - 1):
        mid_body = abs(closes[i - 1] - opens[i - 1])
        if mid_body < atr_mult * atr[i - 1]:
            continue

        # Bullish FVG: candle[i].low > candle[i-2].high
        if i >= 2 and lows[i] > highs[i - 2]:
            raw_bull_fvgs.append((i, lows[i], highs[i - 2]))   # (index, top, bot)

        # Bearish FVG: candle[i].high < candle[i-2].low
        if i >= 2 and highs[i] < lows[i - 2]:
            raw_bear_fvgs.append((i, lows[i - 2], highs[i]))   # (index, top, bot)

    current_close = closes[-1]

    # Check which bull FVGs have been inverted (closed below bot) → bear IFVG
    for (fi, top, bot) in raw_bull_fvgs:
        inverted = False
        invert_i = None
        for j in range(fi + 1, n):
            if closes[j] < bot:
                inverted = True
                invert_i = j
                break
        if inverted and invert_i is not None:
            # IFVG bear zone = original FVG range (now resistance)
            # Active until price closes above the original FVG top
            if current_close < top:
                bear_ifvg.append({
                    "top":       top,
                    "bot":       bot,
                    "formed_i":  invert_i,
                    "formed_ts": df.index[invert_i],
                })

    # Check which bear FVGs have been inverted (closed above top) → bull IFVG
    for (fi, top, bot) in raw_bear_fvgs:
        inverted = False
        invert_i = None
        for j in range(fi + 1, n):
            if closes[j] > top:
                inverted = True
                invert_i = j
                break
        if inverted and invert_i is not None:
            if current_close > bot:
                bull_ifvg.append({
                    "top":       top,
                    "bot":       bot,
                    "formed_i":  invert_i,
                    "formed_ts": df.index[invert_i],
                })

    return {"bull": bull_ifvg, "bear": bear_ifvg}


def price_in_ifvg(price: float, ifvgs: list[dict]) -> dict | None:
    """Return the first IFVG that contains price."""
    for zone in ifvgs:
        if zone["bot"] <= price <= zone["top"]:
            return zone
    return None


def _atr_values(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].values
    low  = df["low"].values
    prev_close = np.roll(df["close"].values, 1)
    prev_close[0] = df["close"].values[0]
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - prev_close), np.abs(low - prev_close)
    ))
    atr = np.full(len(tr), np.nan)
    if len(tr) >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr
