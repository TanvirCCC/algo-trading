"""
Technical Indicators
SMM591 CDT03 — class-taught toolkit (RSI, CCI, EMA, Bollinger Bands, MACD)

These are the indicators used in the Trading Game report.
They serve as confirmation signals alongside the ICT detectors.

CDT03 mappings:
  RSI < 40  → oversold / discount confirmation
  RSI > 60  → overbought / premium confirmation
  CCI < -100 → deep discount (commodity-specific, designed for commodities)
  CCI > +100 → deep premium
  EMA 50/200 crossover → trend direction / HTF bias filter
  Bollinger Band squeeze → consolidation preceding breakout
"""

import pandas as pd
import numpy as np


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi_oversold"] = df["rsi"] < 40
    df["rsi_overbought"] = df["rsi"] > 60
    return df


def add_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    CCI = (1/0.015) × (typical_price − SMA(typical_price)) / σ(typical_price)
    Designed for commodity markets. CCI < -100 = oversold; > +100 = overbought.
    """
    df = df.copy()
    tp = (df["high"] + df["low"] + df["close"]) / 3
    sma_tp = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df["cci"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    df["cci_oversold"] = df["cci"] < -100
    df["cci_overbought"] = df["cci"] > 100
    return df


def add_ema(df: pd.DataFrame, periods: list[int] = [50, 200]) -> pd.DataFrame:
    df = df.copy()
    for p in periods:
        df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
    if 50 in periods and 200 in periods:
        df["bull_trend"] = df["ema_50"] > df["ema_200"]
        df["bear_trend"] = df["ema_50"] < df["ema_200"]
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    sma = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    df["bb_upper"] = sma + std_dev * std
    df["bb_lower"] = sma - std_dev * std
    df["bb_mid"] = sma
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma
    # Squeeze: bandwidth below 20th percentile of recent 50 bars
    df["bb_squeeze"] = df["bb_width"] < df["bb_width"].rolling(50).quantile(0.20)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_bull"] = (df["macd"] > df["macd_signal"]) & (df["macd"] < 0)
    df["macd_bear"] = (df["macd"] < df["macd_signal"]) & (df["macd"] > 0)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


def add_all(df: pd.DataFrame) -> pd.DataFrame:
    df = add_rsi(df)
    df = add_cci(df)
    df = add_ema(df, [50, 200])
    df = add_bollinger_bands(df)
    df = add_macd(df)
    df = add_atr(df)
    return df
