"""
Dukascopy XAU/USD Historical Data Downloader
=============================================
Downloads 1-minute bid candles from Dukascopy's free public API,
resamples to 5m and daily, saves as parquet.

Usage:
    python3 data/dukascopy_fetcher.py               # full in-sample 2016-2023
    python3 data/dukascopy_fetcher.py --test        # 1 month only (sanity check)
    python3 data/dukascopy_fetcher.py --year 2020   # single year
"""

import sys
import lzma
import struct
import time
import argparse
import requests
import pandas as pd
from datetime import datetime, date, timedelta
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOL       = "XAUUSD"
POINT_VALUE  = 1_000     # Dukascopy XAU/USD: prices stored as int(price * 1000)
BASE_URL     = "https://datafeed.dukascopy.com/datafeed"
OUTPUT_DIR   = Path("data/historical")
SLEEP_SEC    = 0.15      # polite rate limit between requests
MAX_RETRIES  = 3


# ── Core download + parse ─────────────────────────────────────────────────────

def _fetch_day(year: int, month: int, day: int) -> pd.DataFrame | None:
    """
    Download and parse 1-minute BID candles for one calendar day.
    month is 1-indexed (1=Jan). Dukascopy URL uses 0-indexed months.
    Returns DataFrame or None if no data (weekend / holiday / gap).
    """
    url = (
        f"{BASE_URL}/{SYMBOL}/{year}/{month-1:02d}/{day:02d}"
        f"/BID_candles_min_1.bi5"
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return None          # no data for this day (normal on weekends)
            if resp.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            if len(resp.content) < 24:
                return None          # empty file

            raw = lzma.decompress(resp.content)
            break
        except Exception:
            if attempt == MAX_RETRIES - 1:
                return None
            time.sleep(2 ** attempt)
    else:
        return None

    n = len(raw) // 24
    if n == 0:
        return None

    day_start_utc = datetime(year, month, day)
    rows = []
    for i in range(n):
        off = i * 24
        t_ms, o, h, l, c = struct.unpack_from(">IIIII", raw, off)
        vol = struct.unpack_from(">f", raw, off + 20)[0]
        ts = day_start_utc + timedelta(seconds=int(t_ms))
        rows.append((ts, o / POINT_VALUE, h / POINT_VALUE,
                     l / POINT_VALUE, c / POINT_VALUE, vol))

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close", "volume"])
    df = df.set_index("datetime")
    return df


# ── Range downloader ──────────────────────────────────────────────────────────

def download_range(start: date, end: date, label: str = "") -> pd.DataFrame:
    """Download all trading days in [start, end] and concatenate."""
    frames = []
    current = start
    total   = (end - start).days + 1
    done    = 0
    skipped = 0

    print(f"  Downloading {label or SYMBOL}: {start} → {end}  ({total} calendar days)")

    while current <= end:
        if current.weekday() < 5:   # skip Saturday / Sunday
            df_day = _fetch_day(current.year, current.month, current.day)
            if df_day is not None and not df_day.empty:
                frames.append(df_day)
            else:
                skipped += 1
        done += 1

        if done % 100 == 0 or current == end:
            pct = done / total * 100
            bars = sum(len(f) for f in frames)
            print(f"    {done}/{total} days ({pct:.0f}%)  bars so far: {bars:,}  skipped: {skipped}")

        current += timedelta(days=1)
        time.sleep(SLEEP_SEC)

    if not frames:
        print("  No data received.")
        return pd.DataFrame()

    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


# ── Resamplers ────────────────────────────────────────────────────────────────

def to_5m(df_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        df_1m.resample("5min")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
    )


def to_daily(df_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        df_1m.resample("D")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna(subset=["open"])
    )


# ── Sanity check ──────────────────────────────────────────────────────────────

def _validate(df_1m: pd.DataFrame):
    lo, hi = df_1m["close"].min(), df_1m["close"].max()
    print(f"\n  Sanity check:")
    print(f"    Close price range : ${lo:.2f} – ${hi:.2f}")
    if lo < 500 or hi > 10_000:
        print("  WARNING: prices look wrong — check POINT_VALUE constant")
    else:
        print("    Prices look correct ✓")
    print(f"    First bar : {df_1m.index[0]}")
    print(f"    Last bar  : {df_1m.index[-1]}")
    print(f"    Total bars: {len(df_1m):,}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test",  action="store_true", help="Download Jan 2020 only")
    parser.add_argument("--year",  type=int,            help="Download a single year")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.test:
        start, end = date(2020, 1, 1), date(2020, 1, 31)
        label = "TEST (Jan 2020)"
    elif args.year:
        start, end = date(args.year, 1, 1), date(args.year, 12, 31)
        label = f"{args.year}"
    else:
        # Full in-sample period
        start, end = date(2016, 1, 1), date(2023, 12, 31)
        label = "in-sample 2016–2023"

    print(f"\nDukascopy XAU/USD downloader — {label}")
    print("=" * 55)

    df_1m = download_range(start, end, label)

    if df_1m.empty:
        print("Aborting — no data.")
        sys.exit(1)

    _validate(df_1m)

    # Save 1m
    suffix = f"_{args.year}" if args.year else ("_test" if args.test else "_insample")
    p1m = OUTPUT_DIR / f"gold_1m_dukascopy{suffix}.parquet"
    df_1m.to_parquet(p1m)
    print(f"\n  Saved 1m  → {p1m}  ({len(df_1m):,} bars)")

    # Save 5m
    df_5m = to_5m(df_1m)
    p5m = OUTPUT_DIR / f"gold_5m_dukascopy{suffix}.parquet"
    df_5m.to_parquet(p5m)
    print(f"  Saved 5m  → {p5m}  ({len(df_5m):,} bars)")

    # Save daily
    df_d = to_daily(df_1m)
    pd_ = OUTPUT_DIR / f"gold_daily_dukascopy{suffix}.parquet"
    df_d.to_parquet(pd_)
    print(f"  Saved D   → {pd_}  ({len(df_d):,} bars)")

    print("\nDone.")


if __name__ == "__main__":
    main()
