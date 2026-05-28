"""
Session AMD + OHLC Reference Levels
lnterqo Notes: Trading Notes/lnterqo — Strategy Analysis.md

AMD (Accumulation / Manipulation / Distribution) mapped to sessions (UTC):
  Asia        00:00 – 06:00  →  Accumulation (range, do not trade)
  London      07:00 – 10:00  →  Manipulation + early Distribution
  NY          12:00 – 15:00  →  Distribution (secondary session)
  Lunch       15:00 – 17:00  →  Avoid (low liquidity)

OHLC reference levels used by lnterqo:
  Midnight open  — 00:00 UTC open (NY midnight = 00:00 UTC in winter)
  9AM NY         — 14:00 UTC open (NY 9AM = 14:00 UTC)
  PDH / PDL      — Previous day's high / low (BSL / SSL)
  Asia H / L     — Range of the 00:00–06:00 UTC session

Asia Purge (Manipulation signal):
  When London open price sweeps above Asia High (bearish setup catalyst)
  or below Asia Low (bullish setup catalyst).
  After a purge: distribution trade expected in the opposite direction.
"""

from datetime import time, date, timedelta
import pandas as pd
import numpy as np


# ── Session boundaries (UTC) ──────────────────────────────────────────────────

ASIA_START    = time(0,  0)
ASIA_END      = time(6,  0)
LONDON_START  = time(7,  0)
LONDON_END    = time(10, 0)
NY_START      = time(12, 0)
NY_END        = time(15, 0)
LUNCH_START   = time(15, 0)
LUNCH_END     = time(17, 0)

# lnterqo's "intra-hour kill zone" — last 10 + first 10 min of each hour
_KILL_MINUTES = set(range(50, 60)) | set(range(0, 10))


def get_session(ts: pd.Timestamp) -> str:
    """Return the session label for a given UTC timestamp."""
    t = ts.time()
    if ASIA_START   <= t < ASIA_END:    return "asia"
    if LONDON_START <= t < LONDON_END:  return "london"
    if NY_START     <= t < NY_END:      return "ny"
    if LUNCH_START  <= t < LUNCH_END:   return "lunch"
    return "off"


def is_trading_session(ts: pd.Timestamp) -> bool:
    """True if the bar falls in London or NY session (not Asia, lunch, or off)."""
    s = get_session(ts)
    return s in ("london", "ny")


def is_intra_hour_kill(ts: pd.Timestamp) -> bool:
    """True if bar is in the :50–:09 intra-hour kill zone."""
    return ts.minute in _KILL_MINUTES


def get_asia_range(df_5m: pd.DataFrame, bar_date: date) -> dict | None:
    """
    Return the Asia session high/low for the given date (00:00–06:00 UTC).
    Returns None if insufficient data.
    """
    day_start = pd.Timestamp(bar_date)
    day_asia_end = day_start + pd.Timedelta(hours=6)

    mask = (df_5m.index >= day_start) & (df_5m.index < day_asia_end)
    asia_bars = df_5m[mask]

    if len(asia_bars) < 4:
        return None

    return {
        "high": asia_bars["high"].max(),
        "low":  asia_bars["low"].min(),
        "date": bar_date,
    }


def detect_asia_purge(
    row: pd.Series,
    asia_range: dict | None,
    already_purged_direction: str | None = None,
) -> str | None:
    """
    Returns 'bull' if price has swept BELOW Asia Low (manipulation → expect bullish D)
    Returns 'bear' if price has swept ABOVE Asia High (manipulation → expect bearish D)
    Returns None if no purge.

    already_purged_direction: prevents double-counting a purge already registered today.
    """
    if asia_range is None:
        return None

    if already_purged_direction == "bear":
        return "bear"
    if already_purged_direction == "bull":
        return "bull"

    if row["low"] < asia_range["low"]:
        return "bull"   # SSL swept → expect bullish distribution
    if row["high"] > asia_range["high"]:
        return "bear"   # BSL swept → expect bearish distribution
    return None


def get_midnight_open(df_5m: pd.DataFrame, bar_date: date) -> float | None:
    """
    Return the open price of the 00:00 UTC bar on bar_date.
    Midnight open = start of the new trading day reference.
    """
    midnight = pd.Timestamp(bar_date)
    # Find the first bar at or after midnight
    idx = df_5m.index.searchsorted(midnight)
    if idx >= len(df_5m):
        return None
    bar_ts = df_5m.index[idx]
    if bar_ts.date() != bar_date:
        return None
    return float(df_5m["open"].iloc[idx])


def get_9am_ny_level(df_5m: pd.DataFrame, bar_date: date) -> dict | None:
    """
    Return the high/low of the 9AM NY candle (14:00 UTC ≈ 9AM ET winter / 13:00 BST summer).
    lnterqo uses the 9AM high/low as a manipulation sweep target.
    """
    # 9AM NY = 14:00 UTC (EST winter). Use 13:30–14:00 UTC as the opening window.
    day_ts = pd.Timestamp(bar_date)
    window_start = day_ts + pd.Timedelta(hours=13, minutes=30)
    window_end   = day_ts + pd.Timedelta(hours=14, minutes=30)

    mask = (df_5m.index >= window_start) & (df_5m.index <= window_end)
    bars = df_5m[mask]

    if bars.empty:
        return None

    return {
        "high": bars["high"].max(),
        "low":  bars["low"].min(),
        "open": float(bars["open"].iloc[0]),
    }


def get_prev_day_levels(df_daily: pd.DataFrame, bar_date: date) -> dict | None:
    """
    Return previous day's High and Low (DR H / DR L in lnterqo's notation).
    These serve as BSL (PDH) and SSL (PDL) targets.
    """
    prev = bar_date - timedelta(days=1)
    # Walk back to find last trading day
    for _ in range(7):
        ts = pd.Timestamp(prev)
        if ts in df_daily.index:
            row = df_daily.loc[ts]
            return {
                "pdh": float(row["high"]),
                "pdl": float(row["low"]),
                "date": prev,
            }
        prev -= timedelta(days=1)
    return None
