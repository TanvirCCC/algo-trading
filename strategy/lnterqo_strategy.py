"""
lnterqo Strategy
lnterqo Notes: Trading Notes/lnterqo — Strategy Analysis.md

ICT/MMXM implementation based on @lnterqo's methodology:
  - CISD (Change in State of Delivery) as primary entry trigger
  - AMD phase: only enter during Distribution after Manipulation completes
  - Asia session range purge as manipulation confirmation
  - BKR CE / IFVG / FVG as entry zone confluence
  - OHLC references: midnight open, 9AM NY, PDH/PDL
  - Session kill zones: London (07–10 UTC) and NY (12–15 UTC)
  - Intra-hour :50–:09 windows = highest-probability entry timing

Signal hierarchy (entry quality, highest first):
  1. CISD + BKR CE + Asia purge aligned           → confidence 5
  2. CISD + IFVG/IOF + Asia purge aligned          → confidence 4
  3. CISD + FVG + Asia purge aligned               → confidence 3
  4. CISD + BKR/FVG, no confirmed purge yet        → confidence 2
  5. CISD only (weakest)                           → confidence 1
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import time, timedelta
from collections import deque

from detectors.cisd import detect_cisd_bearish, detect_cisd_bullish
from detectors.breaker import detect_breakers, price_in_bkr
from detectors.ifvg import build_ifvg_zones, price_in_ifvg
from detectors.session_amd import (
    get_session, is_trading_session, is_intra_hour_kill,
    get_asia_range, detect_asia_purge,
    get_midnight_open, get_9am_ny_level, get_prev_day_levels,
)
from detectors.market_structure import analyze as ms_analyze
from detectors.indicators import add_all as add_indicators
from detectors import liquidity


# ── Signal dataclass (same interface as ICT strategy for engine compatibility) ─

@dataclass
class Signal:
    timestamp: pd.Timestamp
    asset: str
    direction: str         # "long" | "short"
    entry: float
    stop: float
    target: float
    confidence: int        # 1–5
    risk_reward: float
    zone_type: str         # "CISD+BKR" | "CISD+IFVG" | "CISD+FVG" | "CISD"
    report_rationale: str  = ""
    raw_signals: dict      = field(default_factory=dict)  # ML filter compat

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)


# ── Session kill-zone times (UTC) ─────────────────────────────────────────────

_TRADING_SESSIONS = [
    (time(7, 0),  time(10, 0)),   # London
    (time(12, 0), time(15, 0)),   # NY morning
]


def _in_session(ts: pd.Timestamp) -> bool:
    t = ts.time()
    return any(start <= t < end for start, end in _TRADING_SESSIONS)


# ── Data preparation ──────────────────────────────────────────────────────────

def prepare_data(df_daily: pd.DataFrame, df_5m: pd.DataFrame) -> tuple:
    """Add indicators and market structure to both frames."""
    if "atr" not in df_5m.columns:
        df_5m = add_indicators(df_5m)
    if "swing_high" not in df_5m.columns:
        df_5m = ms_analyze(df_5m)
    if "atr" not in df_daily.columns:
        df_daily = add_indicators(df_daily)
    if "swing_high" not in df_daily.columns:
        df_daily = ms_analyze(df_daily)
    return df_daily, df_5m


# ── Bias from daily/4H ────────────────────────────────────────────────────────

def _daily_bias(df_daily: pd.DataFrame, ts: pd.Timestamp) -> str:
    """
    HTF bias: 'bull', 'bear', or 'neutral'.
    Uses 20-bar premium/discount and recent BOS direction.
    """
    past = df_daily[df_daily.index <= ts]
    if len(past) < 20:
        return "neutral"

    last = past.iloc[-1]
    close = last["close"]
    eq    = last.get("equilibrium", np.nan)

    # Market structure trend from BOS
    bos_bull = past["bos_bull"].sum() if "bos_bull" in past.columns else 0
    bos_bear = past["bos_bear"].sum() if "bos_bear" in past.columns else 0
    mss_bull = past["mss_bull"].sum() if "mss_bull" in past.columns else 0
    mss_bear = past["mss_bear"].sum() if "mss_bear" in past.columns else 0

    # Most recent structural event within last 10 bars
    recent = past.tail(10)
    last_bull = (recent["bos_bull"].sum() + recent["mss_bull"].sum()) if "bos_bull" in recent else 0
    last_bear = (recent["bos_bear"].sum() + recent["mss_bear"].sum()) if "bos_bear" in recent else 0

    if last_bull > last_bear and (np.isnan(eq) or close < eq):
        return "bull"   # discount + recent bullish structure
    if last_bear > last_bull and (np.isnan(eq) or close > eq):
        return "bear"   # premium + recent bearish structure
    return "neutral"


# ── Liquidity targets ─────────────────────────────────────────────────────────

def _select_target(
    entry: float, direction: str, stop: float,
    liq: dict, asia_range: dict | None, prev_day: dict | None,
    min_rr: float,
) -> float | None:
    """
    Target selection hierarchy (nearest qualifying level):
      1. Asia High/Low (opposite side of purge)
      2. PDH / PDL
      3. Buy-side / sell-side liquidity from liquidity detector
    """
    risk = abs(entry - stop)
    if risk == 0:
        return None

    candidates = []

    # Asia range: opposite side
    if asia_range:
        if direction == "long":
            candidates.append(asia_range["high"])
        else:
            candidates.append(asia_range["low"])

    # Previous day levels
    if prev_day:
        if direction == "long":
            candidates.append(prev_day["pdh"])
        else:
            candidates.append(prev_day["pdl"])

    # Liquidity pools from detector
    if direction == "long":
        for lv in liq.get("buy_side", []):
            candidates.append(lv["level"])
    else:
        for lv in liq.get("sell_side", []):
            candidates.append(lv["level"])

    # Filter: must give >= min_rr and be in the right direction
    valid = []
    for t in candidates:
        if direction == "long"  and t > entry and (t - entry) >= min_rr * risk:
            valid.append(t)
        if direction == "short" and t < entry and (entry - t) >= min_rr * risk:
            valid.append(t)

    if not valid:
        return None
    return min(valid, key=lambda t: abs(t - entry))   # nearest qualifying


# ── Main signal scanner ───────────────────────────────────────────────────────

def scan_for_signals(
    asset: str,
    df_daily: pd.DataFrame,
    df_5m: pd.DataFrame,
    min_rr: float = 2.0,
    use_news_filter: bool = True,
    zone_lookback: int = 60,
    cisd_lookback: int = 40,
    force_neutral_bias: bool = False,
) -> list[Signal]:
    """
    Scan df_5m for lnterqo-style signals.

    Entry conditions (all required):
      1. In London or NY session
      2. Daily bias not opposed to trade direction
      3. Asia range purge confirms manipulation complete (or neutral bias bypass)
      4. CISD detected on 5m
      5. Entry zone: CISD body / BKR CE / IFVG / FVG in the right location
      6. Valid liquidity target at >= min_rr

    Signal quality:
      5 = CISD + BKR CE + confirmed purge
      4 = CISD + IFVG/IOF + confirmed purge
      3 = CISD + FVG + confirmed purge
      2 = CISD + zone, purge not yet confirmed
      1 = CISD only
    """
    df_daily, df_5m = prepare_data(df_daily, df_5m)

    signals: list[Signal] = []
    last_signal_bar = -10   # cooldown
    current_date = None
    asia_range: dict | None = None
    purge_direction: str | None = None   # 'bull' | 'bear' | None
    midnight_open: float | None = None

    closes = df_5m["close"].values
    highs  = df_5m["high"].values
    lows   = df_5m["low"].values

    for i in range(max(cisd_lookback + 5, zone_lookback + 5), len(df_5m)):
        ts  = df_5m.index[i]
        row = df_5m.iloc[i]

        # ── Day reset ─────────────────────────────────────────────────────────
        bar_date = ts.date()
        if bar_date != current_date:
            current_date    = bar_date
            purge_direction = None
            asia_range      = get_asia_range(df_5m, bar_date)
            midnight_open   = get_midnight_open(df_5m, bar_date)

        # ── Session filter ────────────────────────────────────────────────────
        if not _in_session(ts):
            continue

        # ── News filter ───────────────────────────────────────────────────────
        if use_news_filter and row.get("news_avoid", False):
            continue

        # ── Cooldown ──────────────────────────────────────────────────────────
        if i - last_signal_bar < 6:
            continue

        # ── Track Asia purge (manipulation) ───────────────────────────────────
        new_purge = detect_asia_purge(row, asia_range, purge_direction)
        if new_purge and not purge_direction:
            purge_direction = new_purge

        # ── Daily bias ────────────────────────────────────────────────────────
        bias = "neutral" if force_neutral_bias else _daily_bias(df_daily, ts)

        atr = row.get("atr", 0)
        if pd.isna(atr) or atr == 0:
            continue

        # ── Build entry-zone context (lookback window) ────────────────────────
        window   = df_5m.iloc[max(0, i - zone_lookback): i + 1]
        liq_slice = df_5m.iloc[max(0, i - 500): i + 1]
        liq      = liquidity.get_liquidity_targets(liq_slice)
        prev_day = get_prev_day_levels(df_daily, bar_date)

        # Breakers and IFVGs for this window
        bkrs  = detect_breakers(window, lookback=zone_lookback)
        ifvgs = build_ifvg_zones(window, lookback=zone_lookback)

        # ── LONG setup ────────────────────────────────────────────────────────
        if bias in ("bull", "neutral"):
            # Purge check: if Asia data exists, prefer purge confirmation
            purge_ok = (purge_direction == "bull") or (asia_range is None) or force_neutral_bias

            cisd = detect_cisd_bullish(df_5m, i, cisd_lookback)
            if cisd and purge_ok:
                entry  = closes[i]
                stop   = cisd["sl_price"] - 0.1 * atr   # just below CISD low

                target = _select_target(
                    entry, "long", stop, liq, asia_range, prev_day, min_rr
                )
                if target is None:
                    continue

                rr = (target - entry) / abs(entry - stop)

                # Entry zone confluence → confidence
                zone_type = "CISD"
                conf = 1
                bkr_hit  = price_in_bkr(entry, bkrs["bull"])
                ifvg_hit = price_in_ifvg(entry, ifvgs["bull"])

                # FVG check (simple: is price near any recent bull FVG)
                fvg_hit = _price_near_fvg(entry, window, "bull")

                if bkr_hit and purge_direction == "bull":
                    zone_type, conf = "CISD+BKR", 5
                elif ifvg_hit and purge_direction == "bull":
                    zone_type, conf = "CISD+IFVG", 4
                elif fvg_hit and purge_direction == "bull":
                    zone_type, conf = "CISD+FVG", 3
                elif bkr_hit or ifvg_hit or fvg_hit:
                    zone_type, conf = "CISD+zone", 2

                # Post-news boost
                if use_news_filter and row.get("news_entry", False):
                    conf = min(5, conf + 1)

                rationale = (
                    f"Bullish CISD at {ts}. "
                    f"Zone: {zone_type}. "
                    f"Asia purge: {purge_direction or 'none'}. "
                    f"RR: {rr:.1f}."
                )

                signals.append(Signal(
                    timestamp=ts, asset=asset, direction="long",
                    entry=entry, stop=stop, target=target,
                    confidence=conf, risk_reward=round(rr, 2),
                    zone_type=zone_type, report_rationale=rationale,
                ))
                last_signal_bar = i
                continue

        # ── SHORT setup ───────────────────────────────────────────────────────
        if bias in ("bear", "neutral"):
            purge_ok = (purge_direction == "bear") or (asia_range is None) or force_neutral_bias

            cisd = detect_cisd_bearish(df_5m, i, cisd_lookback)
            if cisd and purge_ok:
                entry  = closes[i]
                stop   = cisd["sl_price"] + 0.1 * atr

                target = _select_target(
                    entry, "short", stop, liq, asia_range, prev_day, min_rr
                )
                if target is None:
                    continue

                rr = (entry - target) / abs(entry - stop)

                zone_type = "CISD"
                conf = 1
                bkr_hit  = price_in_bkr(entry, bkrs["bear"])
                ifvg_hit = price_in_ifvg(entry, ifvgs["bear"])
                fvg_hit  = _price_near_fvg(entry, window, "bear")

                if bkr_hit and purge_direction == "bear":
                    zone_type, conf = "CISD+BKR", 5
                elif ifvg_hit and purge_direction == "bear":
                    zone_type, conf = "CISD+IFVG", 4
                elif fvg_hit and purge_direction == "bear":
                    zone_type, conf = "CISD+FVG", 3
                elif bkr_hit or ifvg_hit or fvg_hit:
                    zone_type, conf = "CISD+zone", 2

                if use_news_filter and row.get("news_entry", False):
                    conf = min(5, conf + 1)

                rationale = (
                    f"Bearish CISD at {ts}. "
                    f"Zone: {zone_type}. "
                    f"Asia purge: {purge_direction or 'none'}. "
                    f"RR: {rr:.1f}."
                )

                signals.append(Signal(
                    timestamp=ts, asset=asset, direction="short",
                    entry=entry, stop=stop, target=target,
                    confidence=conf, risk_reward=round(rr, 2),
                    zone_type=zone_type, report_rationale=rationale,
                ))
                last_signal_bar = i

    return signals


# ── Helpers ───────────────────────────────────────────────────────────────────

def _price_near_fvg(price: float, window: pd.DataFrame, direction: str) -> bool:
    """Quick check: is price within any active FVG zone in window."""
    closes = window["close"].values
    highs  = window["high"].values
    lows   = window["low"].values
    n = len(window)

    for i in range(2, n):
        if direction == "bull":
            if lows[i] > highs[i - 2]:
                top = lows[i]
                bot = highs[i - 2]
                # Not yet filled
                if closes[i + 1:].size == 0 or closes[i + 1:].min() > bot:
                    if bot <= price <= top:
                        return True
        else:
            if highs[i] < lows[i - 2]:
                top = lows[i - 2]
                bot = highs[i]
                if closes[i + 1:].size == 0 or closes[i + 1:].max() < top:
                    if bot <= price <= top:
                        return True
    return False
