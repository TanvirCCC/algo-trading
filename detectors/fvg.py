"""
Fair Value Gap (FVG) Detector
ICT Notes: Fair Value Gap.md

A FVG is a 3-candle imbalance where:
  - Bullish FVG: candle[i].low > candle[i-2].high  (gap between candle -2 high and candle 0 low)
  - Bearish FVG: candle[i].high < candle[i-2].low  (gap between candle -2 low and candle 0 high)

Requirements (per notes):
  1. Energetic displacement candle in the middle (large body relative to ATR)
  2. Gap must exist between candle 1 and candle 3
  3. Context: prior liquidity sweep needed for highest-probability FVG

Academic equivalent (CDT03): price gap — "A gap is closed or 'filled' when the price comes back
and retraces the whole range of the gap." Runaway gap = mid-trend displacement FVG.
"""

import pandas as pd
import numpy as np


def detect_fvgs(df: pd.DataFrame, atr_mult: float = 0.5) -> pd.DataFrame:
    """
    Identify bullish and bearish Fair Value Gaps.

    atr_mult: minimum body size of displacement candle as a multiple of ATR(14).
              Filters out small, lethargic gaps per ICT rules.

    Adds columns:
      fvg_bull      — True if candle is the entry (3rd) candle of a bullish FVG
      fvg_bear      — True if candle is the entry (3rd) candle of a bearish FVG
      fvg_bull_top  — Top of bullish FVG zone
      fvg_bull_bot  — Bottom of bullish FVG zone
      fvg_bear_top  — Top of bearish FVG zone
      fvg_bear_bot  — Bottom of bearish FVG zone
    """
    df = df.copy()
    atr = _atr(df, 14)

    n = len(df)
    fvg_bull = [False] * n
    fvg_bear = [False] * n
    fvg_bull_top = [np.nan] * n
    fvg_bull_bot = [np.nan] * n
    fvg_bear_top = [np.nan] * n
    fvg_bear_bot = [np.nan] * n

    for i in range(2, n):
        mid_body = abs(df["close"].iloc[i - 1] - df["open"].iloc[i - 1])
        min_displacement = atr_mult * atr.iloc[i - 1]

        if mid_body < min_displacement:
            continue

        # Bullish FVG: candle[i].low > candle[i-2].high
        if df["low"].iloc[i] > df["high"].iloc[i - 2]:
            fvg_bull[i] = True
            fvg_bull_bot[i] = df["high"].iloc[i - 2]
            fvg_bull_top[i] = df["low"].iloc[i]

        # Bearish FVG: candle[i].high < candle[i-2].low
        if df["high"].iloc[i] < df["low"].iloc[i - 2]:
            fvg_bear[i] = True
            fvg_bear_top[i] = df["low"].iloc[i - 2]
            fvg_bear_bot[i] = df["high"].iloc[i]

    df["fvg_bull"] = fvg_bull
    df["fvg_bear"] = fvg_bear
    df["fvg_bull_top"] = fvg_bull_top
    df["fvg_bull_bot"] = fvg_bull_bot
    df["fvg_bear_top"] = fvg_bear_top
    df["fvg_bear_bot"] = fvg_bear_bot
    return df


def get_active_fvgs(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Return a list of unfilled (active) FVG zones from the last `lookback` candles.
    A FVG is filled when price trades through the entire gap (both body levels).
    """
    active_bull = []
    active_bear = []

    recent = df.tail(lookback).copy()
    closes = df["close"].values
    n_full = len(closes)

    for idx, row in recent.iterrows():
        pos = df.index.get_loc(idx)

        if row["fvg_bull"] and not np.isnan(row["fvg_bull_bot"]):
            top = row["fvg_bull_top"]
            bot = row["fvg_bull_bot"]
            # Check if filled: any subsequent close went below bot (full fill)
            subsequent_lows = df["low"].iloc[pos + 1:]
            if subsequent_lows.empty or subsequent_lows.min() > bot:
                active_bull.append({"top": top, "bot": bot, "formed_at": idx})

        if row["fvg_bear"] and not np.isnan(row["fvg_bear_top"]):
            top = row["fvg_bear_top"]
            bot = row["fvg_bear_bot"]
            subsequent_highs = df["high"].iloc[pos + 1:]
            if subsequent_highs.empty or subsequent_highs.max() < top:
                active_bear.append({"top": top, "bot": bot, "formed_at": idx})

    return {"bull": active_bull, "bear": active_bear}


def price_in_fvg(price: float, fvgs: list[dict]) -> dict | None:
    """Return the FVG zone if price is currently inside it, else None."""
    for zone in fvgs:
        if zone["bot"] <= price <= zone["top"]:
            return zone
    return None


def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()
