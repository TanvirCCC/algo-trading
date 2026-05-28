"""
Breaker Block (BKR) Detector
lnterqo Notes: Trading Notes/lnterqo — Strategy Analysis.md

A BKR forms when:
  1. Price creates a swing high/low
  2. A displacement candle closes THROUGH that swing (creating a FVG)
  3. Price later retraces back into the range of that original swing candle
  4. That original candle is now the BKR zone — premium (bear) or discount (bull)

BKR > OB because it requires a prior liquidity sweep, confirming institutional involvement.
BKR CE (Consequent Encroachment) = midpoint of the BKR body = preferred entry.
"""

import numpy as np
import pandas as pd


def detect_breakers(df: pd.DataFrame, lookback: int = 40, confirm: int = 2) -> dict:
    """
    Scan the last `lookback` bars for active breaker block zones.

    Returns:
      {
        "bull": [{"top", "bot", "ce", "formed_i", "formed_ts"}, ...],
        "bear": [{"top", "bot", "ce", "formed_i", "formed_ts"}, ...],
      }

    A zone is "active" until price trades through the far edge of the BKR body.
    """
    highs  = df["high"].values
    lows   = df["low"].values
    opens  = df["open"].values
    closes = df["close"].values
    n      = len(df)

    bull_bkrs = []
    bear_bkrs = []

    start = max(confirm + 1, n - lookback)

    for j in range(start, n - confirm - 1):
        # ── Bearish BKR: swing high → swept below by displacement ────────────
        is_sh = (
            all(highs[j] >= highs[j - k] for k in range(1, confirm + 1)) and
            all(highs[j] >= highs[j + k] for k in range(1, confirm + 1))
        )
        if is_sh:
            # Look for a displacement candle after j that closes BELOW j's low
            sh_low = lows[j]
            for k in range(j + 1, min(j + lookback, n)):
                if closes[k] < sh_low:
                    # BKR zone = the swing high candle body
                    bkr_top = max(opens[j], closes[j])
                    bkr_bot = min(opens[j], closes[j])
                    ce      = (bkr_top + bkr_bot) / 2
                    # Only active if current price hasn't closed above bkr_top
                    if closes[-1] < bkr_top:
                        bear_bkrs.append({
                            "top":       bkr_top,
                            "bot":       bkr_bot,
                            "ce":        ce,
                            "sh_high":   highs[j],   # SL anchor above
                            "formed_i":  j,
                            "formed_ts": df.index[j],
                        })
                    break

        # ── Bullish BKR: swing low → swept above by displacement ─────────────
        is_sl = (
            all(lows[j] <= lows[j - k] for k in range(1, confirm + 1)) and
            all(lows[j] <= lows[j + k] for k in range(1, confirm + 1))
        )
        if is_sl:
            sl_high = highs[j]
            for k in range(j + 1, min(j + lookback, n)):
                if closes[k] > sl_high:
                    bkr_top = max(opens[j], closes[j])
                    bkr_bot = min(opens[j], closes[j])
                    ce      = (bkr_top + bkr_bot) / 2
                    if closes[-1] > bkr_bot:
                        bull_bkrs.append({
                            "top":       bkr_top,
                            "bot":       bkr_bot,
                            "ce":        ce,
                            "sl_low":    lows[j],    # SL anchor below
                            "formed_i":  j,
                            "formed_ts": df.index[j],
                        })
                    break

    return {"bull": bull_bkrs, "bear": bear_bkrs}


def price_in_bkr(price: float, bkrs: list[dict]) -> dict | None:
    """Return the first BKR zone that contains price (prioritise CE proximity)."""
    candidates = [z for z in bkrs if z["bot"] <= price <= z["top"]]
    if not candidates:
        return None
    # Prefer zone whose CE is closest to current price
    return min(candidates, key=lambda z: abs(z["ce"] - price))
