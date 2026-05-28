"""
Bloomberg Terminal 5-min bar Excel parser.

Expected format (BDH intraday export):
  Row 0 : column headers (Time Interval, Close, Net Chg, Open, High, Low, ...)
  Row 1 : summary row
  Row N : date-header row  e.g. "01JAN2026_00:00:00.000000"
  Rows  : bar rows         e.g. "23:00 - 23:05"

Returns a normalised DataFrame identical to the shape the rest of the
codebase expects: DatetimeIndex, columns open/high/low/close/volume.
"""

import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Bloomberg month abbreviation → int
_MON = {m: i for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"], 1
)}

_DATE_RE = re.compile(r'^(\d{2})([A-Z]{3})(\d{4})_')
_BAR_RE  = re.compile(r'^(\d{1,2}):(\d{2})\s*-\s*\d')


def _to_float(val) -> float | None:
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (ValueError, TypeError):
        return None


def load_bloomberg_excel(filepath: str | Path) -> pd.DataFrame:
    """
    Parse a Bloomberg 5-min OHLCV Excel export.
    Returns a clean DataFrame with DatetimeIndex (UTC-naive) and
    columns: open, high, low, close, volume.
    """
    raw = pd.read_excel(filepath, sheet_name=0, header=None)

    records: list[tuple] = []
    current_date: date | None = None

    for _, row in raw.iterrows():
        cell = str(row[0]).strip()

        # Date header row
        dm = _DATE_RE.match(cell)
        if dm:
            day  = int(dm.group(1))
            mon  = _MON.get(dm.group(2))
            year = int(dm.group(3))
            if mon:
                current_date = date(year, mon, day)
            continue

        # 5-min bar row
        bm = _BAR_RE.match(cell)
        if bm and current_date is not None:
            hour   = int(bm.group(1))
            minute = int(bm.group(2))
            ts = pd.Timestamp(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                hour=hour,
                minute=minute,
            )
            # Columns: Time Interval | Close | Net Chg | Open | High | Low | Tick Count | Volume
            close = _to_float(row[1])
            open_ = _to_float(row[3])
            high  = _to_float(row[4])
            low   = _to_float(row[5])
            vol   = _to_float(row[7]) if str(row[7]).strip() != "N.A." else 0.0

            if None not in (open_, high, low, close):
                records.append((ts, open_, high, low, close, vol or 0.0))

    if not records:
        raise ValueError(f"No bars parsed from {filepath}")

    df = pd.DataFrame(records, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df.set_index("timestamp", inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


# Mapping: friendly name → (Bloomberg file, yfinance symbol for HTF context)
BLOOMBERG_ASSETS = {
    "Gold":   ("Gold jan-april.xlsx",   "gold"),
    "Oil":    ("oil jan-may.xlsx",      "crude"),
    "Cocoa":  ("cocao jan-may.xlsx",    "CC=F"),
    "Nasdaq": ("nasdaq jan-may.xlsx",   "NQ=F"),
}
