"""
Seasonal Tendencies Detector
Gold/commodity seasonality based on documented COT + demand cycles.

Gold seasonal pattern (empirical, 1980–2025):
  Jan–Feb  : Bullish  — post-holiday safe-haven demand, central bank buying
  Mar–Apr  : Neutral  — spring consolidation
  May      : Bearish  — sell in May effect, pre-summer lull
  Jun–Aug  : Bearish  — summer doldrums, lowest physical demand
  Sep–Oct  : Neutral/Bull — Indian festive season begins, Q4 repositioning
  Nov–Dec  : Bullish  — year-end safe-haven, holiday demand, Indian wedding peak

ICT AMD (Accumulation–Manipulation–Distribution) weekly pattern:
  Monday   : Accumulation / liquidity grab — often fake directional move
  Tuesday  : Primary displacement — highest probability ICT entry day
  Wednesday: FOMC weeks → high-impact; otherwise continuation or reversal
  Thursday : Distribution / manipulation — often reversal of Tue/Wed move
  Friday   : Close-out — choppy, avoid new positions late day
"""

import pandas as pd
import numpy as np

# Month → (bias, strength)   strength 0–1
_GOLD_MONTH_BIAS: dict[int, tuple[str, float]] = {
    1:  ("bull",    0.7),   # January
    2:  ("bull",    0.6),   # February
    3:  ("neutral", 0.3),   # March
    4:  ("neutral", 0.3),   # April
    5:  ("bear",    0.4),   # May
    6:  ("bear",    0.5),   # June
    7:  ("bear",    0.5),   # July
    8:  ("bear",    0.4),   # August
    9:  ("neutral", 0.4),   # September
    10: ("bull",    0.5),   # October
    11: ("bull",    0.6),   # November
    12: ("bull",    0.7),   # December
}

# Day of week → priority (0=skip, 1=low, 2=normal, 3=high)
_DOW_PRIORITY: dict[int, int] = {
    0: 1,   # Monday   — accumulation, lower probability
    1: 3,   # Tuesday  — primary displacement day
    2: 2,   # Wednesday — normal (FOMC risk if in week)
    3: 2,   # Thursday  — continuation possible
    4: 1,   # Friday   — avoid late entries, close risk
}

# Oil seasonal
_OIL_MONTH_BIAS: dict[int, tuple[str, float]] = {
    1:  ("neutral", 0.3),
    2:  ("bull",    0.4),   # refinery restarts
    3:  ("bull",    0.5),   # driving season build
    4:  ("bull",    0.6),   # peak spring demand
    5:  ("bull",    0.5),   # summer travel
    6:  ("bull",    0.4),
    7:  ("neutral", 0.3),
    8:  ("bear",    0.4),   # hurricane risk / end of driving season
    9:  ("bear",    0.4),
    10: ("bear",    0.5),   # inventory builds
    11: ("neutral", 0.3),
    12: ("bear",    0.4),
}

# Corn seasonal
_CORN_MONTH_BIAS: dict[int, tuple[str, float]] = {
    1:  ("neutral", 0.3),
    2:  ("neutral", 0.3),
    3:  ("bull",    0.5),   # planting intentions
    4:  ("bull",    0.6),   # planting season
    5:  ("bull",    0.5),
    6:  ("neutral", 0.4),   # pollination uncertainty
    7:  ("bear",    0.5),   # good/bad crop report
    8:  ("neutral", 0.4),
    9:  ("bear",    0.5),   # harvest pressure
    10: ("bear",    0.6),
    11: ("neutral", 0.3),
    12: ("neutral", 0.3),
}

_ASSET_BIAS: dict[str, dict] = {
    "Gold":  _GOLD_MONTH_BIAS,
    "Oil":   _OIL_MONTH_BIAS,
    "Corn":  _CORN_MONTH_BIAS,
    "Cocoa": _GOLD_MONTH_BIAS,  # use gold as proxy for now
}


def get_seasonal_bias(ts: pd.Timestamp, asset: str = "Gold") -> tuple[str, float]:
    """
    Return (bias, strength) for a timestamp based on seasonal tendencies.
    bias: 'bull', 'bear', or 'neutral'
    strength: 0.0–1.0 (how strong the seasonal signal is)
    """
    table = _ASSET_BIAS.get(asset, _GOLD_MONTH_BIAS)
    return table.get(ts.month, ("neutral", 0.0))


def get_dow_priority(ts: pd.Timestamp) -> int:
    """
    Return day-of-week entry priority (0=skip, 1=low, 2=normal, 3=high).
    Based on ICT AMD weekly cycle.
    """
    return _DOW_PRIORITY.get(ts.weekday(), 2)


def seasonal_agrees_with_direction(
    ts: pd.Timestamp,
    direction: str,
    asset: str = "Gold",
    min_strength: float = 0.4,
) -> bool:
    """
    True if the seasonal bias agrees with the trade direction.
    Returns True if seasonal is neutral (no override).
    Only filters when seasonal strength >= min_strength.
    """
    bias, strength = get_seasonal_bias(ts, asset)
    if strength < min_strength or bias == "neutral":
        return True   # seasonal not strong enough to filter
    if direction == "long"  and bias == "bear":
        return False
    if direction == "short" and bias == "bull":
        return False
    return True


def add_seasonal_columns(df: pd.DataFrame, asset: str = "Gold") -> pd.DataFrame:
    """
    Add seasonal_bias and dow_priority columns to a DataFrame.
    Useful for the ML filter feature set.
    """
    df = df.copy()
    biases = [get_seasonal_bias(ts, asset) for ts in df.index]
    df["seasonal_bias"]     = [b[0] for b in biases]
    df["seasonal_strength"] = [b[1] for b in biases]
    df["dow_priority"]      = [get_dow_priority(ts) for ts in df.index]

    # Numeric encoding for ML
    bias_map = {"bull": 1, "neutral": 0, "bear": -1}
    df["seasonal_bias_num"] = df["seasonal_bias"].map(bias_map)
    return df
