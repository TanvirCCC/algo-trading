"""
Liquidity Levels Detector
ICT Notes: Liquidity Concepts.md, Market Structure.md

Key levels:
  - Previous day high/low (PDH/PDL) — easiest draw on liquidity
  - Previous week high/low (PWH/PWL) — intermediate draw
  - Equal highs/lows (within tolerance) — engineered liquidity clusters
  - IRL/ERL cycle: after external liquidity is taken, internal draw; vice versa

Academic equivalent (CDT03): prior swing highs/lows as support and resistance.
"""

import pandas as pd
import numpy as np


def add_prev_day_levels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous day high, low, and open to the intraday (1h/15m) dataframe.
    These are the most common draw-on-liquidity targets (PDH/PDL).
    """
    if "day_high" in df.columns:
        return df  # already prepared — avoid duplicate join
    df = df.copy()
    df["date"] = df.index.date

    daily_hl = df.groupby("date").agg(
        day_high=("high", "max"),
        day_low=("low", "min"),
        day_open=("open", "first"),
    ).shift(1)

    df = df.join(daily_hl, on="date")
    df.drop(columns=["date"], inplace=True)
    return df


def add_prev_week_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous week high and low."""
    df = df.copy()
    df["week"] = df.index.to_period("W")

    weekly_hl = df.groupby("week").agg(
        week_high=("high", "max"),
        week_low=("low", "min"),
    ).shift(1)

    df = df.join(weekly_hl, on="week")
    df.drop(columns=["week"], inplace=True)
    return df


def find_equal_highs_lows(df: pd.DataFrame, tolerance_pct: float = 0.002) -> pd.DataFrame:
    """
    Identify equal highs and equal lows (engineered liquidity).
    Two swing highs/lows are 'equal' if within tolerance_pct of each other.
    These represent stop clusters that smart money targets.
    """
    df = df.copy()
    df["equal_high"] = False
    df["equal_low"] = False

    sh_idx = df.index[df["swing_high"]] if "swing_high" in df.columns else []
    sl_idx = df.index[df["swing_low"]] if "swing_low" in df.columns else []

    for i in range(1, len(sh_idx)):
        h1 = df.loc[sh_idx[i - 1], "high"]
        h2 = df.loc[sh_idx[i], "high"]
        if abs(h1 - h2) / h1 < tolerance_pct:
            df.loc[sh_idx[i], "equal_high"] = True

    for i in range(1, len(sl_idx)):
        l1 = df.loc[sl_idx[i - 1], "low"]
        l2 = df.loc[sl_idx[i], "low"]
        if abs(l1 - l2) / l1 < tolerance_pct:
            df.loc[sl_idx[i], "equal_low"] = True

    return df


def get_liquidity_targets(df: pd.DataFrame) -> dict:
    """
    Return the current active liquidity targets above and below price.

    Returns:
        buy_side  — levels above current price (targets for bearish sweep)
        sell_side — levels below current price (targets for bullish sweep)
    """
    last = df.iloc[-1]
    price = last["close"]

    levels = []

    for col in ["day_high", "day_low", "week_high", "week_low"]:
        if col in df.columns and not pd.isna(last.get(col, np.nan)):
            levels.append({"level": last[col], "type": col})

    if "sh_price" in df.columns:
        recent_sh = df["sh_price"].dropna().tail(5)
        for val in recent_sh:
            levels.append({"level": val, "type": "swing_high"})

    if "sl_price" in df.columns:
        recent_sl = df["sl_price"].dropna().tail(5)
        for val in recent_sl:
            levels.append({"level": val, "type": "swing_low"})

    buy_side = sorted([l for l in levels if l["level"] > price], key=lambda x: x["level"])
    sell_side = sorted([l for l in levels if l["level"] < price], key=lambda x: x["level"], reverse=True)

    return {
        "buy_side": buy_side,
        "sell_side": sell_side,
        "nearest_above": buy_side[0] if buy_side else None,
        "nearest_below": sell_side[0] if sell_side else None,
    }
