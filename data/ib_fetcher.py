"""
Interactive Brokers 5-min Historical Data Fetcher
Pulls data from specific (non-continuous) futures contracts in 1-month chunks.

Confirmed working on IB paper accounts:
  - Active front-month contracts support past endDateTime ✓
  - 1-month chunks of 5-min data return ~5,800 bars in ~60s ✓
  - Data available back ~1 year from contract inception ✓

Usage:
  python3 data/ib_fetcher.py              # all assets
  python3 data/ib_fetcher.py --assets Gold Oil
"""

import time
import argparse
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
from ib_insync import IB, Future, util

PORTS     = [7497, 4002]
CLIENT_ID = 99
TIMEOUT   = 300        # seconds per request — 1M of 5-min takes ~60s
PACE_WAIT = 14         # seconds between requests

OUTPUT_DIR = Path("data/historical")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Assets ─────────────────────────────────────────────────────────────────
# List front-month contracts newest-first; fetcher tries each and walks back.
# Add older contracts below when they expire and you want more history.
ASSET_SPECS = {
    "Gold": dict(
        symbol="GC", exchange="COMEX", currency="USD",
        multiplier="100", tradingClass="GC",
        contracts=[
            "20260626",   # GCM6 — active
            "20260428",   # GCJ6 — recently expired
            "20260224",   # GCG6
            "20251224",   # GCZ5
            "20251027",   # GCV5
            "20250826",   # GCQ5
        ],
    ),
    "Oil": dict(
        symbol="CL", exchange="NYMEX", currency="USD",
        multiplier="1000", tradingClass="CL",
        contracts=[
            "20260622",   # CLN6 — active
            "20260520",   # CLM6
            "20260421",   # CLJ6
            "20260319",   # CLH6
            "20260219",   # CLG6
            "20260120",   # CLF6
            "20251119",   # CLX5
            "20251020",   # CLV5
        ],
    ),
    "Corn": dict(
        symbol="ZC", exchange="CBOT", currency="USD",
        multiplier="50", tradingClass="ZC",
        contracts=[
            "20260714",   # ZCN6 — active
            "20260514",   # ZCK6
            "20260313",   # ZCH6
            "20251212",   # ZCZ5
            "20250912",   # ZCU5
            "20250714",   # ZCN5
        ],
    ),
    "Cocoa": dict(
        symbol="CC", exchange="NYBOT", currency="USD",
        multiplier="10", tradingClass="CC",
        contracts=[
            "20260715",   # CCN6 — active
            "20260512",   # CCK6
            "20260312",   # CCH6
            "20251210",   # CCZ5
            "20250909",   # CCU5
        ],
    ),
}


def connect() -> IB:
    ib = IB()
    for port in PORTS:
        try:
            ib.connect("127.0.0.1", port, clientId=CLIENT_ID, timeout=15)
            print(f"  Connected on port {port}")
            return ib
        except Exception:
            continue
    raise ConnectionError("Cannot connect to TWS/Gateway.")


def _normalise(raw: list) -> pd.DataFrame:
    df = util.df(raw)
    df = df.rename(columns={"date": "timestamp"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    return df.set_index("timestamp").sort_index()


def fetch_contract(ib: IB, spec: dict, expiry: str,
                   stop_before: pd.Timestamp | None = None) -> pd.DataFrame:
    """
    Fetch 5-min bars for one contract, walking back in 1-month chunks.
    stop_before: stop as soon as we reach data earlier than this timestamp
                 (avoids re-downloading data we already have).
    """
    is_expired = datetime.strptime(expiry, "%Y%m%d").date() < date.today()
    c = Future(
        symbol=spec["symbol"],
        lastTradeDateOrContractMonth=expiry,
        exchange=spec["exchange"],
        currency=spec["currency"],
        multiplier=spec["multiplier"],
        tradingClass=spec["tradingClass"],
    )
    if is_expired:
        c.includeExpired = True

    exp_dt   = datetime.strptime(expiry, "%Y%m%d")
    cursor   = min(exp_dt, datetime.now())
    chunks   = []
    no_data_count = 0

    print(f"  {expiry} ({'expired' if is_expired else 'active'}) — walking back...")

    while True:
        end_str = cursor.strftime("%Y%m%d %H:%M:%S")
        try:
            bars = ib.reqHistoricalData(
                c,
                endDateTime=end_str,
                durationStr="1 M",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
                timeout=TIMEOUT,
            )
        except Exception as e:
            if "pacing" in str(e).lower():
                print(f"    Pacing — waiting 65s...")
                time.sleep(65)
                continue
            print(f"    Request error: {e}")
            break

        time.sleep(PACE_WAIT)

        if not bars:
            no_data_count += 1
            if no_data_count >= 2:
                print(f"    No data at {cursor.date()} — reached start of contract")
                break
            cursor -= timedelta(days=30)
            continue

        no_data_count = 0
        chunk = _normalise(bars)
        earliest = chunk.index[0]
        print(f"    {earliest.date()} → {chunk.index[-1].date()}  ({len(chunk)} bars)")
        chunks.append(chunk)

        # Stop if we've reached data we already have
        if stop_before is not None and earliest <= stop_before:
            print(f"    Reached existing data boundary — stopping")
            break

        cursor = earliest - timedelta(minutes=1)

    if not chunks:
        return pd.DataFrame()

    result = pd.concat(chunks)
    return result[~result.index.duplicated(keep="first")].sort_index()


def download_asset(ib: IB, name: str):
    spec     = ASSET_SPECS[name]
    out_path = OUTPUT_DIR / f"{name.lower()}_5m.parquet"
    existing = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()

    all_chunks = [existing] if not existing.empty else []

    # Track earliest date already saved so we don't re-fetch
    covered_from = existing.index[0] if not existing.empty else None
    if covered_from:
        print(f"\n[{name}] Existing data from {covered_from.date()}, extending backwards...")
    else:
        print(f"\n[{name}]  {spec['symbol']} @ {spec['exchange']}")

    for expiry in spec["contracts"]:
        exp_dt = datetime.strptime(expiry, "%Y%m%d")
        # Skip contracts whose entire period is already covered
        if covered_from is not None and exp_dt < pd.Timestamp(covered_from):
            print(f"  {expiry}: already covered")
            continue

        # Tell fetcher to stop once it reaches data we already have
        stop_before = covered_from if covered_from is not None else None
        chunk = fetch_contract(ib, spec, expiry, stop_before=stop_before)

        if chunk.empty:
            print(f"  {expiry}: no data")
            continue

        print(f"  {expiry}: {len(chunk):,} bars ({chunk.index[0].date()} → {chunk.index[-1].date()})")
        all_chunks.append(chunk)

        # Save checkpoint and update coverage boundary
        df_save = pd.concat(all_chunks)
        df_save = df_save[~df_save.index.duplicated(keep="last")].sort_index()
        df_save.to_parquet(out_path)
        covered_from = df_save.index[0]
        print(f"  Checkpoint: {len(df_save):,} bars total  ({covered_from.date()} → {df_save.index[-1].date()})")
        all_chunks = [df_save]

    if all_chunks:
        df = all_chunks[0] if len(all_chunks) == 1 else pd.concat(all_chunks)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        df.to_parquet(out_path)
        print(f"\n[{name}] DONE — {len(df):,} bars  ({df.index[0].date()} → {df.index[-1].date()})")
    else:
        print(f"\n[{name}] No data saved.")


def load_parquet(name: str, bar_size: str = "5m") -> pd.DataFrame:
    path = OUTPUT_DIR / f"{name.lower()}_{bar_size}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No data at {path}. Run: python3 data/ib_fetcher.py")
    return pd.read_parquet(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=list(ASSET_SPECS.keys()))
    args = parser.parse_args()

    print("Connecting to Interactive Brokers...")
    ib = connect()

    try:
        for name in args.assets:
            if name not in ASSET_SPECS:
                print(f"Unknown: '{name}'. Available: {list(ASSET_SPECS.keys())}")
                continue
            download_asset(ib, name)
        print("\nAll done.")
    except KeyboardInterrupt:
        print("\nInterrupted — progress saved to checkpoints.")
    finally:
        ib.disconnect()
        print("Disconnected.")
