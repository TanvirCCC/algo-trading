import yfinance as yf
import pandas as pd
from pathlib import Path

QC_DIR = Path(__file__).parent / "qc"

ASSETS = {
    "gold":   "GC=F",
    "crude":  "CL=F",
    "corn":   "ZC=F",
    "nq":     "NQ=F",
    "es":     "ES=F",
    "eurusd": "EURUSD=X",
    "gbpusd": "GBPUSD=X",
    "btc":    "BTC-USD",
}


def _load_qc(symbol: str, interval: str) -> pd.DataFrame | None:
    path = QC_DIR / f"{symbol.lower()}_{interval}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.set_index("datetime")
    df.index = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch(symbol: str, interval: str = "1h", period: str = "60d") -> pd.DataFrame:
    """
    Fetch OHLCV data. Uses QuantConnect CSV (data/qc/{symbol}_{interval}.csv) if present,
    otherwise falls back to yfinance.
    """
    qc = _load_qc(symbol, interval)
    if qc is not None:
        return qc
    ticker = ASSETS.get(symbol.lower(), symbol)
    df = yf.download(ticker, interval=interval, period=period, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    # yfinance 1.3+ returns MultiIndex columns — flatten to single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def fetch_multi_tf(symbol: str) -> dict[str, pd.DataFrame]:
    """Fetch daily, 1h, and 15m data for multi-timeframe analysis."""
    return {
        "daily": fetch(symbol, interval="1d", period="1y"),
        "1h":    fetch(symbol, interval="1h", period="60d"),
        "15m":   fetch(symbol, interval="15m", period="30d"),
    }


def fetch_range(symbol: str, interval: str, start: str, end: str = None) -> pd.DataFrame:
    """
    Fetch OHLCV data for a specific date range.
    Note: yfinance caps intraday (1h, 15m) at ~730 days regardless of start.
    Daily/weekly intervals support arbitrary historical ranges (back to ~2000).
    """
    ticker = ASSETS.get(symbol.lower(), symbol)
    kwargs = dict(interval=interval, start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} ({interval}, start={start})")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    return df


def fetch_long_history(symbol: str, start: str = "2010-01-01") -> dict[str, pd.DataFrame]:
    """
    Fetch weekly (HTF bias) and daily (entry TF) data for a long-term backtest.
    Daily data from yfinance goes back to ~2000 for GC=F, CL=F, ZC=F.
    Weekly is used for EMA50/200 HTF bias; daily for FVG/OB/signal detection.
    """
    return {
        "weekly": fetch_range(symbol, "1wk", start=start),
        "daily":  fetch_range(symbol, "1d",  start=start),
    }
