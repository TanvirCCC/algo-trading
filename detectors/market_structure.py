"""
Market Structure Detector
ICT Notes: Market Structure.md, Liquidity Concepts.md

Detects:
- Swing highs / swing lows (3-candle pattern)
- Break of Structure (BOS)
- Change of Character (CHoCH / MSS)
- Premium / Discount zones (50% of dealing range)
"""

import pandas as pd
import numpy as np


def find_swings(df: pd.DataFrame, lookback: int = 2) -> pd.DataFrame:
    """
    Identify swing highs and swing lows using a lookback-candle confirmation.
    ICT definition: swing high = candle with higher highs on both sides (lookback bars each side).
    """
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)

    swing_high = [False] * n
    swing_low = [False] * n

    for i in range(lookback, n - lookback):
        if all(highs[i] > highs[i - j] for j in range(1, lookback + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, lookback + 1)):
            swing_high[i] = True
        if all(lows[i] < lows[i - j] for j in range(1, lookback + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, lookback + 1)):
            swing_low[i] = True

    df = df.copy()
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    df["sh_price"] = np.where(df["swing_high"], df["high"], np.nan)
    df["sl_price"] = np.where(df["swing_low"], df["low"], np.nan)
    return df


def find_bos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Break of Structure (BOS): price closes beyond the most recent confirmed swing.
    - Bullish BOS: close above previous swing high → continuation up
    - Bearish BOS: close below previous swing low → continuation down

    MSS (Market Structure Shift / CHoCH): BOS in opposite direction of current trend.
    """
    df = df.copy()
    df["bos_bull"] = False
    df["bos_bear"] = False
    df["mss_bull"] = False   # CHoCH up (end of downtrend)
    df["mss_bear"] = False   # CHoCH down (end of uptrend)

    last_sh = np.nan
    last_sl = np.nan
    trend = None  # "up" or "down"

    for i in range(len(df)):
        row = df.iloc[i]

        if df["swing_high"].iloc[i]:
            last_sh = row["high"]
        if df["swing_low"].iloc[i]:
            last_sl = row["low"]

        if not np.isnan(last_sh) and row["close"] > last_sh:
            if trend == "down":
                df.iloc[i, df.columns.get_loc("mss_bull")] = True
            else:
                df.iloc[i, df.columns.get_loc("bos_bull")] = True
            trend = "up"
            last_sh = np.nan

        if not np.isnan(last_sl) and row["close"] < last_sl:
            if trend == "up":
                df.iloc[i, df.columns.get_loc("mss_bear")] = True
            else:
                df.iloc[i, df.columns.get_loc("bos_bear")] = True
            trend = "down"
            last_sl = np.nan

    return df


def add_premium_discount(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Premium / Discount: 50% level of the recent dealing range (rolling high/low).
    Above 50% = premium (look to sell), below 50% = discount (look to buy).
    """
    df = df.copy()
    roll_high = df["high"].rolling(window).max()
    roll_low = df["low"].rolling(window).min()
    equilibrium = (roll_high + roll_low) / 2
    df["equilibrium"] = equilibrium
    df["range_high"] = roll_high
    df["range_low"] = roll_low
    df["in_discount"] = df["close"] < equilibrium
    df["in_premium"] = df["close"] > equilibrium
    return df


def analyze(df: pd.DataFrame, lookback: int = 2, pd_window: int = 20) -> pd.DataFrame:
    df = find_swings(df, lookback)
    df = find_bos(df)
    df = add_premium_discount(df, pd_window)
    return df
