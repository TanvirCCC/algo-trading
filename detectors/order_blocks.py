"""
Order Block Detector
ICT Notes: Order Blocks & Price Delivery.md

Order Block (OB) definition:
  Bullish OB — the last DOWN-CLOSE candle before a bullish displacement move that
               sweeps sell-side liquidity. Price returns to this zone for entry.
  Bearish OB — the last UP-CLOSE candle before a bearish displacement move that
               sweeps buy-side liquidity. Price returns to this zone for entry.

The two-part requirement:
  1. Liquidity raid (stop hunt) before the OB
  2. Key level context (discount for bull OB, premium for bear OB)

Academic equivalent (CDT03): demand/supply zone at prior swing — a historically
significant level where institutional orders were last placed.
"""

import pandas as pd
import numpy as np


def detect_order_blocks(df: pd.DataFrame, displacement_atr_mult: float = 1.5) -> pd.DataFrame:
    """
    Detect the most recent bullish and bearish order blocks.

    A valid OB requires a displacement candle immediately after it:
      body of the displacement candle >= displacement_atr_mult × ATR(14)

    Adds columns:
      ob_bull      — True at the OB candle (for annotation)
      ob_bear      — True at the OB candle
      ob_bull_top  — OB high (top of zone to sell into for entry)
      ob_bull_bot  — OB low (stop loss reference)
      ob_bear_top  — OB high (stop loss reference)
      ob_bear_bot  — OB low (top of zone to buy into for entry)
    """
    df = df.copy()
    atr = _atr(df, 14)

    n = len(df)
    ob_bull = [False] * n
    ob_bear = [False] * n
    ob_bull_top = [np.nan] * n
    ob_bull_bot = [np.nan] * n
    ob_bear_top = [np.nan] * n
    ob_bear_bot = [np.nan] * n

    for i in range(1, n):
        displacement_body = abs(df["close"].iloc[i] - df["open"].iloc[i])
        min_disp = displacement_atr_mult * atr.iloc[i]

        if displacement_body < min_disp:
            continue

        prev = df.iloc[i - 1]

        # Bullish OB: prev candle is a down-close; current is a strong up candle
        if df["close"].iloc[i] > df["open"].iloc[i] and prev["close"] < prev["open"]:
            ob_bull[i - 1] = True
            ob_bull_top[i - 1] = prev["high"]
            ob_bull_bot[i - 1] = prev["low"]

        # Bearish OB: prev candle is an up-close; current is a strong down candle
        if df["close"].iloc[i] < df["open"].iloc[i] and prev["close"] > prev["open"]:
            ob_bear[i - 1] = True
            ob_bear_top[i - 1] = prev["high"]
            ob_bear_bot[i - 1] = prev["low"]

    df["ob_bull"] = ob_bull
    df["ob_bear"] = ob_bear
    df["ob_bull_top"] = ob_bull_top
    df["ob_bull_bot"] = ob_bull_bot
    df["ob_bear_top"] = ob_bear_top
    df["ob_bear_bot"] = ob_bear_bot
    return df


def get_active_obs(df: pd.DataFrame, lookback: int = 50) -> dict:
    """
    Return unfilled order block zones from the last `lookback` bars.
    A bullish OB is mitigated (used up) when price trades below ob_bull_bot.
    A bearish OB is mitigated when price trades above ob_bear_top.
    """
    active_bull = []
    active_bear = []
    recent = df.tail(lookback)

    for idx, row in recent.iterrows():
        pos = df.index.get_loc(idx)

        if row["ob_bull"] and not np.isnan(row["ob_bull_bot"]):
            subsequent = df.iloc[pos + 1:]
            if subsequent.empty or subsequent["low"].min() > row["ob_bull_bot"]:
                active_bull.append({
                    "top": row["ob_bull_top"],
                    "bot": row["ob_bull_bot"],
                    "formed_at": idx,
                })

        if row["ob_bear"] and not np.isnan(row["ob_bear_top"]):
            subsequent = df.iloc[pos + 1:]
            if subsequent.empty or subsequent["high"].max() < row["ob_bear_top"]:
                active_bear.append({
                    "top": row["ob_bear_top"],
                    "bot": row["ob_bear_bot"],
                    "formed_at": idx,
                })

    return {"bull": active_bull, "bear": active_bear}


def price_in_ob(price: float, obs: list[dict]) -> dict | None:
    for zone in obs:
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
