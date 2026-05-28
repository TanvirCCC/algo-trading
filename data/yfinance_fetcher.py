"""
yfinance Commodity Data Fetcher
================================
Downloads OHLCV data for energy, metals and agriculture commodities.
Uses continuous front-month futures (auto-adjusted) from Yahoo Finance.

Daily data goes back to ~2000 for most contracts — plenty for backtesting.

Usage:
    python3 data/yfinance_fetcher.py                # all commodities
    python3 data/yfinance_fetcher.py --assets Oil Cocoa Wheat
    python3 data/yfinance_fetcher.py --start 2016-01-01 --end 2023-12-31
"""

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf

OUTPUT_DIR = Path("data/historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Asset catalogue ──────────────────────────────────────────────────────────
# Continuous front-month futures (auto-adjusted for rolls)
ASSETS = {
    # Energy
    "Oil":    "BZ=F",    # Brent Crude (IG/CMC default for Europe)
    "WTI":    "CL=F",    # WTI Crude
    "NatGas": "NG=F",    # Natural Gas

    # Metals (Dukascopy covers Gold better, but yfinance as fallback)
    "Silver": "SI=F",
    "Copper": "HG=F",

    # Agriculture
    "Wheat":  "ZW=F",    # Chicago SRW Wheat
    "Corn":   "ZC=F",    # Corn
    "Soy":    "ZS=F",    # Soybeans
    "Coffee": "KC=F",    # Arabica Coffee
    "Cocoa":  "CC=F",    # Cocoa
    "Sugar":  "SB=F",    # Raw Sugar
}

# Point value used by IG/CMC CFDs (for position sizing reference)
CFD_CONTRACT_SIZE = {
    "Oil":    1000,   # $1000 per lot, 1 lot = 1000 barrels
    "WTI":    1000,
    "Wheat":  50,     # 50 bushels per tick
    "Corn":   50,
    "Cocoa":  10,     # £10 per $1 move on IG
    "Coffee": 37_500, # 37,500 lbs per contract
}


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_daily(ticker: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    kwargs = dict(interval="1d", start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        raise ValueError(f"No daily data returned for {ticker}")
    df = _flatten_columns(df)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def fetch_weekly(ticker: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    kwargs = dict(interval="1wk", start=start, auto_adjust=True, progress=False)
    if end:
        kwargs["end"] = end
    df = yf.download(ticker, **kwargs)
    if df.empty:
        raise ValueError(f"No weekly data returned for {ticker}")
    df = _flatten_columns(df)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]].dropna()


def download_asset(name: str, start: str = "2010-01-01", end: str = None):
    ticker = ASSETS[name]
    print(f"\n[{name}]  {ticker}  ({start} → {end or 'today'})")

    # Daily
    try:
        df_d = fetch_daily(ticker, start=start, end=end)
        path_d = OUTPUT_DIR / f"{name.lower()}_daily_yf.parquet"
        df_d.to_parquet(path_d)
        print(f"  Daily : {len(df_d):,} bars  "
              f"({df_d.index[0].date()} → {df_d.index[-1].date()})  → {path_d.name}")
    except Exception as e:
        print(f"  Daily : ERROR — {e}")
        df_d = pd.DataFrame()

    # Weekly (HTF bias for daily-bar strategy)
    try:
        df_w = fetch_weekly(ticker, start=start, end=end)
        path_w = OUTPUT_DIR / f"{name.lower()}_weekly_yf.parquet"
        df_w.to_parquet(path_w)
        print(f"  Weekly: {len(df_w):,} bars  "
              f"({df_w.index[0].date()} → {df_w.index[-1].date()})  → {path_w.name}")
    except Exception as e:
        print(f"  Weekly: ERROR — {e}")

    return df_d


def load_parquet(name: str, bar_size: str = "daily") -> pd.DataFrame:
    """
    Load a saved parquet for a commodity.
    bar_size: 'daily', 'weekly', '5m'
    Tries yfinance files first, then IB files.
    """
    candidates = [
        OUTPUT_DIR / f"{name.lower()}_{bar_size}_yf.parquet",
        OUTPUT_DIR / f"{name.lower()}_{bar_size}.parquet",
        OUTPUT_DIR / f"{name.lower()}_{bar_size}_dukascopy_insample.parquet",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(
        f"No {bar_size} data for {name}. "
        f"Run: python3 data/yfinance_fetcher.py --assets {name}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=list(ASSETS.keys()),
                        choices=list(ASSETS.keys()),
                        help="Which assets to download")
    parser.add_argument("--start",  default="2010-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=None,         help="End date YYYY-MM-DD")
    args = parser.parse_args()

    print(f"yfinance commodity downloader")
    print(f"Period: {args.start} → {args.end or 'today'}")
    print("=" * 50)

    for name in args.assets:
        if name not in ASSETS:
            print(f"Unknown asset: {name}")
            continue
        try:
            download_asset(name, start=args.start, end=args.end)
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
