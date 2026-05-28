"""
histdata.com 1-Minute Bar Downloader
======================================
Downloads 1-minute OHLCV data for Oil (Brent/WTI) from histdata.com,
resamples to 1h and daily, then stitches with yfinance for 2024–now.

Supported symbols (histdata.com naming):
  BCOUSD  — Brent Crude Oil
  WTIUSD  — WTI Crude Oil (West Texas Intermediate)

Usage:
    python3 data/histdata_fetcher.py                        # Brent 2000-2023 + yf 2024-now
    python3 data/histdata_fetcher.py --symbol WTIUSD        # WTI instead
    python3 data/histdata_fetcher.py --start 2016 --end 2023
    python3 data/histdata_fetcher.py --yf-only              # just update 2024-now from yfinance
"""

import io
import re
import sys
import time
import zipfile
import argparse
import requests
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("data/historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SEC   = 1.5    # polite rate limit between requests
MAX_RETRIES = 3

# histdata.com → yfinance symbol mapping (for 2024-now extension)
YF_SYMBOLS = {
    "BCOUSD": "BZ=F",   # Brent Crude
    "WTIUSD": "CL=F",   # WTI Crude
}

# Friendly asset names for file naming
ASSET_NAMES = {
    "BCOUSD": "brent",
    "WTIUSD": "wti",
}


# ── histdata.com downloader ───────────────────────────────────────────────────

def _parse_csv_bytes(raw_bytes: bytes) -> pd.DataFrame | None:
    """Parse histdata.com CSV bytes into a DataFrame."""
    raw = raw_bytes.decode("utf-8", errors="replace")
    rows = []
    for line in raw.splitlines():
        parts = line.strip().split(";")
        if len(parts) < 5:
            continue
        try:
            dt = datetime.strptime(parts[0], "%Y%m%d %H%M%S")
            o, h, l, c = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            vol = float(parts[5]) if len(parts) > 5 else 0.0
            rows.append((dt, o, h, l, c, vol))
        except (ValueError, IndexError):
            continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["datetime","open","high","low","close","volume"])
    return df.set_index("datetime").sort_index()


def _download_month(symbol: str, year: int, month: int) -> pd.DataFrame | None:
    """
    Download one month of 1-min data from histdata.com using Playwright
    to render the JS form, extract the token, and POST the download.
    """
    from playwright.sync_api import sync_playwright

    url = (
        f"https://www.histdata.com/download-free-forex-historical-data/"
        f"?/ascii/1-minute-bar-quotes/{symbol}/{year}/{month}"
    )

    zip_bytes = None

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()

            # Capture the download response
            download_content = {}

            def handle_response(response):
                if "get.php" in response.url and response.status == 200:
                    try:
                        download_content["bytes"] = response.body()
                    except Exception:
                        pass

            page.on("response", handle_response)
            page.goto(url, timeout=30000, wait_until="networkidle")

            # Extract form fields
            token   = page.input_value("#tk")   if page.query_selector("#tk")   else None
            fxpair  = page.input_value("#fp")   if page.query_selector("#fp")   else symbol
            tf      = page.input_value("#tf")   if page.query_selector("#tf")   else "M1"
            platform = page.input_value("#pt")  if page.query_selector("#pt")   else "ASCII"

            if not token:
                browser.close()
                return None

            # Submit the download form
            page.evaluate(f"""
                fetch('https://www.histdata.com/get.php', {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/x-www-form-urlencoded'}},
                    body: new URLSearchParams({{
                        tk: '{token}',
                        date: '{year}',
                        datemonth: '{month:02d}',
                        platform: '{platform}',
                        timeframe: '{tf}',
                        fxpair: '{fxpair}'
                    }})
                }}).then(r => r.arrayBuffer()).then(buf => {{
                    window._dlbuf = btoa(String.fromCharCode(...new Uint8Array(buf)));
                }});
            """)
            page.wait_for_timeout(4000)
            b64 = page.evaluate("window._dlbuf")
            browser.close()

            if b64:
                import base64
                zip_bytes = base64.b64decode(b64)

    except Exception:
        return None

    if not zip_bytes or len(zip_bytes) < 100:
        return None

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_name = next((n for n in z.namelist() if n.endswith(".csv")), None)
            if not csv_name:
                return None
            return _parse_csv_bytes(z.read(csv_name))
    except Exception:
        return None


def download_histdata(
    symbol: str,
    start_year: int = 2000,
    end_year: int   = 2023,
) -> pd.DataFrame:
    """Download all months from start_year to end_year, return combined 1m DataFrame."""
    name  = ASSET_NAMES.get(symbol, symbol.lower())
    cache = OUTPUT_DIR / f"{name}_1m_histdata.parquet"

    # Load existing cache to avoid re-downloading
    existing = pd.DataFrame()
    if cache.exists():
        existing = pd.read_parquet(cache)
        if not existing.empty:
            covered_end = existing.index[-1]
            print(f"  Cache: {len(existing):,} bars up to {covered_end.date()}")

    frames = [existing] if not existing.empty else []
    total_months = (end_year - start_year + 1) * 12
    done = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            done += 1
            # Skip months already in cache
            if not existing.empty:
                month_start = pd.Timestamp(year, month, 1)
                if month_start <= existing.index[-1]:
                    continue

            pct = done / total_months * 100
            print(f"  [{pct:5.1f}%] {year}-{month:02d} ...", end=" ", flush=True)

            df_m = _download_month(symbol, year, month)
            if df_m is not None and not df_m.empty:
                frames.append(df_m)
                print(f"{len(df_m):,} bars", flush=True)
            else:
                print("no data", flush=True)

            time.sleep(SLEEP_SEC)

            # Checkpoint every 12 months
            if done % 12 == 0 and frames:
                df_save = pd.concat(frames)
                df_save = df_save[~df_save.index.duplicated(keep="first")].sort_index()
                df_save.to_parquet(cache)
                frames = [df_save]
                print(f"  → Checkpoint: {len(df_save):,} bars total")

    if not frames:
        print("  No data downloaded.")
        return pd.DataFrame()

    df = pd.concat(frames)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.to_parquet(cache)
    print(f"\n  histdata download complete: {len(df):,} 1m bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── yfinance 2024-now extension ────────────────────────────────────────────────

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    return df


def fetch_yf_recent(symbol: str) -> pd.DataFrame:
    """Fetch 1h data from yfinance for the last ~730 days (2024-now)."""
    yf_sym = YF_SYMBOLS.get(symbol)
    if not yf_sym:
        print(f"  No yfinance mapping for {symbol}")
        return pd.DataFrame()

    print(f"  Fetching recent 1h data from yfinance ({yf_sym})...")
    df = yf.download(yf_sym, interval="1h", period="730d",
                     auto_adjust=True, progress=False)
    if df.empty:
        print("  No yfinance data returned.")
        return pd.DataFrame()

    df = _flatten(df)
    df.index = pd.to_datetime(df.index).tz_localize(None) \
               if df.index.tz is None \
               else pd.to_datetime(df.index).tz_convert(None)
    df = df[["open","high","low","close","volume"]].dropna()
    print(f"  yfinance: {len(df):,} 1h bars  "
          f"({df.index[0].date()} → {df.index[-1].date()})")
    return df


# ── Resamplers ────────────────────────────────────────────────────────────────

def to_1h(df_1m: pd.DataFrame) -> pd.DataFrame:
    return (
        df_1m.resample("1h")
        .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        .dropna(subset=["open"])
    )


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.resample("D")
        .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
        .dropna(subset=["open"])
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",   default="BCOUSD", choices=list(YF_SYMBOLS.keys()))
    parser.add_argument("--start",    type=int, default=2000)
    parser.add_argument("--end",      type=int, default=2023)
    parser.add_argument("--yf-only",  action="store_true",
                        help="Skip histdata — only update 2024-now from yfinance")
    args = parser.parse_args()

    name = ASSET_NAMES.get(args.symbol, args.symbol.lower())

    print(f"\nhistdata.com Oil downloader — {args.symbol} ({name})")
    print("=" * 55)

    # ── 1. histdata.com (2000-2023) ──────────────────────────────────────────
    if not args.yf_only:
        print(f"\nStep 1: Downloading {args.start}–{args.end} from histdata.com...")
        df_1m = download_histdata(args.symbol, args.start, args.end)
    else:
        cache = OUTPUT_DIR / f"{name}_1m_histdata.parquet"
        df_1m = pd.read_parquet(cache) if cache.exists() else pd.DataFrame()
        print(f"  Using cached 1m data: {len(df_1m):,} bars")

    # ── 2. yfinance 2024-now (1h) ────────────────────────────────────────────
    print(f"\nStep 2: Extending with yfinance 2024–now...")
    df_yf_1h = fetch_yf_recent(args.symbol)

    # ── 3. Resample histdata 1m → 1h ─────────────────────────────────────────
    if not df_1m.empty:
        print("\nStep 3: Resampling 1m → 1h...")
        df_hist_1h = to_1h(df_1m)
        print(f"  histdata 1h: {len(df_hist_1h):,} bars")
    else:
        df_hist_1h = pd.DataFrame()

    # ── 4. Combine ────────────────────────────────────────────────────────────
    print("\nStep 4: Combining histdata + yfinance...")
    pieces = []
    if not df_hist_1h.empty:
        pieces.append(df_hist_1h)
    if not df_yf_1h.empty:
        # Only keep yfinance bars that are newer than histdata
        if pieces:
            cutoff = pieces[0].index[-1]
            df_yf_trim = df_yf_1h[df_yf_1h.index > cutoff]
        else:
            df_yf_trim = df_yf_1h
        pieces.append(df_yf_trim)

    if not pieces:
        print("No data to save.")
        sys.exit(1)

    df_1h = pd.concat(pieces)
    df_1h = df_1h[~df_1h.index.duplicated(keep="first")].sort_index()

    # Sanity check
    lo, hi = df_1h["close"].min(), df_1h["close"].max()
    print(f"\n  Sanity check:")
    print(f"    Price range: ${lo:.2f} – ${hi:.2f}")
    if lo < 5 or hi > 300:
        print("    WARNING: prices look wrong")
    else:
        print("    Prices look correct ✓")

    # ── 5. Save ───────────────────────────────────────────────────────────────
    p_1h = OUTPUT_DIR / f"{name}_1h_combined.parquet"
    df_1h.to_parquet(p_1h)
    print(f"\n  Saved 1h  → {p_1h.name}  ({len(df_1h):,} bars)")
    print(f"  Range     : {df_1h.index[0].date()} → {df_1h.index[-1].date()}")

    # Daily (for HTF bias)
    df_d = to_daily(df_1h)
    p_d = OUTPUT_DIR / f"{name}_daily_combined.parquet"
    df_d.to_parquet(p_d)
    print(f"  Saved daily → {p_d.name}  ({len(df_d):,} bars)")

    print("\nDone.")


if __name__ == "__main__":
    main()
