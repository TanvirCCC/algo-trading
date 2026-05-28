"""
CISD — Change in State of Delivery
lnterqo Notes: Trading Notes/lnterqo — Strategy Analysis.md

Primary entry trigger. A CISD occurs when price closes beyond a significant
structural level in a way that signals the algorithm has shifted delivery direction.

Bearish CISD:
  - Lower high pattern confirmed (recent SH below prior SH)
  - Candle closes BELOW the most recent swing low (SSL)
  - Entry on CISD candle close; CISD body = re-entry zone if missed

Bullish CISD:
  - Higher low pattern confirmed (recent SL above prior SL)
  - Candle closes ABOVE the most recent swing high (BSH)
  - Entry on CISD candle close; CISD body = re-entry zone if missed

Why CISD > BOS/CHoCH:
  BOS/CHoCH is a label applied after the fact.
  CISD is the candle close that CREATES the structural shift — it is the entry trigger itself.
"""

import numpy as np
import pandas as pd


def _find_recent_swings(highs, lows, i: int, lookback: int = 30, confirm: int = 2):
    """
    Return the two most recent swing highs and swing lows in window [i-lookback, i-confirm].
    confirm: bars needed on each side to confirm a swing (default 2).
    """
    swing_highs = []  # (bar_index, price)
    swing_lows  = []

    start = max(confirm, i - lookback)
    end   = i - confirm

    for j in range(start, end):
        # Swing high: highest in [j-confirm, j+confirm]
        if all(highs[j] >= highs[j - k] for k in range(1, confirm + 1)) and \
           all(highs[j] >= highs[j + k] for k in range(1, confirm + 1)):
            swing_highs.append((j, highs[j]))
        if all(lows[j] <= lows[j - k] for k in range(1, confirm + 1)) and \
           all(lows[j] <= lows[j + k] for k in range(1, confirm + 1)):
            swing_lows.append((j, lows[j]))

    return swing_highs, swing_lows


def detect_cisd_bearish(df: pd.DataFrame, i: int, lookback: int = 40) -> dict | None:
    """
    Detect a bearish CISD at bar i.

    Conditions:
      1. Lower high pattern: most recent SH < prior SH (confirms bearish delivery)
      2. Current bar closes BELOW the most recent swing low
      3. Bar i is the displacement candle (close < prior SSL)

    Returns a zone dict with:
      top, bot   — CISD candle body (re-entry zone)
      sl_price   — suggested SL above the CISD candle high
      formed_i   — bar index
    """
    if i < lookback + 4:
        return None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values

    swing_highs, swing_lows = _find_recent_swings(highs, lows, i, lookback)

    if len(swing_highs) < 2 or len(swing_lows) < 1:
        return None

    # Lower high pattern: the most recent SH must be lower than the one before it
    sh_recent = swing_highs[-1]
    sh_prior  = swing_highs[-2]
    if sh_recent[1] >= sh_prior[1]:
        return None  # not a lower high

    # Most recent swing low = structural level to break below
    ssl = swing_lows[-1][1]

    # CISD: current close must break below the SSL
    if closes[i] >= ssl:
        return None

    # Candle body
    body_top = max(opens[i], closes[i])
    body_bot = min(opens[i], closes[i])

    return {
        "direction": "bear",
        "top":       body_top,
        "bot":       body_bot,
        "sl_price":  highs[i],           # SL above CISD candle high
        "ssl":       ssl,                 # the level that was broken
        "formed_i":  i,
        "formed_ts": df.index[i],
    }


def detect_cisd_bullish(df: pd.DataFrame, i: int, lookback: int = 40) -> dict | None:
    """
    Detect a bullish CISD at bar i.

    Conditions:
      1. Higher low pattern: most recent SL > prior SL
      2. Current bar closes ABOVE the most recent swing high
    """
    if i < lookback + 4:
        return None

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values
    opens  = df["open"].values

    swing_highs, swing_lows = _find_recent_swings(highs, lows, i, lookback)

    if len(swing_lows) < 2 or len(swing_highs) < 1:
        return None

    sl_recent = swing_lows[-1]
    sl_prior  = swing_lows[-2]
    if sl_recent[1] <= sl_prior[1]:
        return None  # not a higher low

    bsl = swing_highs[-1][1]

    if closes[i] <= bsl:
        return None

    body_top = max(opens[i], closes[i])
    body_bot = min(opens[i], closes[i])

    return {
        "direction": "bull",
        "top":       body_top,
        "bot":       body_bot,
        "sl_price":  lows[i],            # SL below CISD candle low
        "bsl":       bsl,                 # the level that was broken
        "formed_i":  i,
        "formed_ts": df.index[i],
    }
