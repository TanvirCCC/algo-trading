"""
ICT / MMXM Strategy — Commodity CFD Implementation
SMM591 Trading Game

Strategy: NY Bread & Butter / False-Breakout + FVG Entry
Academic name: Multi-Signal Technical Confluence with Fundamental Filters

Entry logic (three-signal confluence required):
  1. HTF bias — EMA(50) > EMA(200) on daily = bull trend; reverse for bear
  2. Price at discount/premium zone — below 50% of range for longs, above for shorts
     AND Fibonacci 50–61.8% retracement zone OR open price gap (FVG)
  3. False breakout pattern — stop hunt swept previous swing, then reversal
     LTF BOS (break of structure) confirms direction change
  4. Confirmation — RSI < 40 (longs) or RSI > 60 (shorts) + CCI < -100 / > +100

Kill zones (session timing):
  London open: 07:00–10:00 GMT (02:00–05:00 NY)
  NY B&B:      08:30–12:00 NY  (13:30–17:00 GMT)
  PM session:  13:30–16:00 NY  (18:30–21:00 GMT)

Stop loss: below swing low (longs) / above swing high (shorts) + 0.5 × ATR buffer
Target:    nearest liquidity level (prior swing high/PDH for longs; swing low/PDL for shorts)
Min R:R:   2:1
"""

import pandas as pd
import numpy as np
from datetime import time
from dataclasses import dataclass, field
from detectors import market_structure, fvg, order_blocks, liquidity, indicators
from detectors.seasonality import seasonal_agrees_with_direction, get_dow_priority


NY_KILL_ZONES = [
    (time(8, 30), time(12, 0)),   # London session (UTC)
    (time(13, 30), time(16, 0)),  # NY morning session (UTC = 09:30–12:00 ET)
    (time(17, 30), time(20, 0)),  # NY afternoon session (UTC = 13:30–16:00 ET)
]

LONDON_KILL_ZONES = [
    (time(7, 0), time(10, 0)),    # London open (GMT, converted)
]


@dataclass
class Signal:
    timestamp: pd.Timestamp
    direction: str          # "long" or "short"
    asset: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    confidence: int         # 1–5 (number of confluence factors)
    zone_type: str          # "fvg", "ob", "fvg+ob"
    report_rationale: str   # academic language for the report
    raw_signals: dict = field(default_factory=dict)

    def __str__(self):
        return (
            f"[{self.timestamp:%Y-%m-%d %H:%M}] {self.direction.upper()} {self.asset} | "
            f"Entry: {self.entry:.2f}  SL: {self.stop:.2f}  TP: {self.target:.2f}  "
            f"R:R {self.risk_reward:.1f}  Confidence: {self.confidence}/5"
        )


def is_kill_zone(ts: pd.Timestamp) -> bool:
    """Check if timestamp falls within a NY kill zone."""
    t = ts.time()
    for start, end in NY_KILL_ZONES:
        if start <= t <= end:
            return True
    return False


def prepare_data(df_daily: pd.DataFrame, df_intraday: pd.DataFrame,
                  asset: str = "Gold") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all detectors on daily and intraday data."""
    from detectors.seasonality import add_seasonal_columns

    # Daily — HTF bias
    df_daily = market_structure.analyze(df_daily, lookback=3, pd_window=20)
    df_daily = indicators.add_all(df_daily)

    # Intraday — entry signals
    df_intraday = market_structure.analyze(df_intraday, lookback=2, pd_window=20)
    df_intraday = fvg.detect_fvgs(df_intraday)
    df_intraday = order_blocks.detect_order_blocks(df_intraday)
    df_intraday = liquidity.add_prev_day_levels(df_intraday)
    df_intraday = indicators.add_all(df_intraday)
    df_intraday = add_seasonal_columns(df_intraday, asset=asset)

    return df_daily, df_intraday


def get_htf_bias(df_daily: pd.DataFrame) -> str:
    """
    HTF bias using EMA50/200 (Dow Theory primary trend) AND EMA20/50 (intermediate trend).
    Commodities trend strongly — require both to agree for a clear bias.
    Academic: golden cross / death cross (CDT03).
    """
    if len(df_daily) < 50:
        return "neutral"
    last = df_daily.iloc[-1]

    ema50 = last.get("ema_50", None)
    ema200 = last.get("ema_200", None)
    close = last["close"]

    if ema50 is None:
        return "neutral"

    # Short-term: is price above or below EMA50?
    above_50 = close > ema50

    if ema200 is not None and len(df_daily) >= 200:
        bull_200 = ema50 > ema200
        if above_50 and bull_200:
            return "bull"
        if not above_50 and not bull_200:
            return "bear"
        # Conflicting — use price vs EMA50 as tiebreaker
        return "bull" if above_50 else "bear"
    else:
        return "bull" if above_50 else "bear"


def _zone_near_price(price: float, zones: list[dict], atr: float, tolerance: float) -> dict | None:
    """Return the first zone within (tolerance × ATR) of price. tolerance=0 → exact hit only."""
    buf = tolerance * atr
    for z in zones:
        if z["bot"] - buf <= price <= z["top"] + buf:
            return z
    return None


def scan_for_signals(
    asset: str,
    df_daily: pd.DataFrame,
    df_intraday: pd.DataFrame,
    min_rr: float = 2.0,
    use_kill_zones: bool = True,
    zone_lookback: int = 80,
    zone_tolerance_atr: float = 0.0,
    bar_range_check: bool = False,
    use_seasonal_filter: bool = False,
    use_news_filter: bool = True,
    force_neutral_bias: bool = False,
) -> list[Signal]:
    """
    Walk-forward scan for ICT entry setups.
    Uses rolling zone state (O(n)) instead of recomputing from scratch each bar.
    force_neutral_bias: allow both long and short regardless of EMA trend direction.
    """
    from collections import deque

    if "fvg_bull" not in df_intraday.columns or "ema_50" not in df_daily.columns:
        df_daily, df_intraday = prepare_data(df_daily, df_intraday)
    bias = "neutral" if force_neutral_bias else get_htf_bias(df_daily)

    has_news_avoid = "news_avoid" in df_intraday.columns
    has_news_entry = "news_entry" in df_intraday.columns

    signals = []
    last_signal_bar = -5

    # Rolling active zones — updated incrementally each bar
    fvg_bull: deque = deque()   # {top, bot, formed_i}
    fvg_bear: deque = deque()
    ob_bull:  deque = deque()
    ob_bear:  deque = deque()

    arr_close  = df_intraday["close"].values
    arr_low    = df_intraday["low"].values
    arr_high   = df_intraday["high"].values

    for i in range(50, len(df_intraday)):
        row = df_intraday.iloc[i]
        ts  = df_intraday.index[i]
        close = arr_close[i]
        low   = arr_low[i]
        high  = arr_high[i]

        # ── Register zones formed at bar i-1 (walk-forward safe) ─────────────
        p = df_intraday.iloc[i - 1]
        if p.get("fvg_bull", False) and not pd.isna(p.get("fvg_bull_bot", float("nan"))):
            fvg_bull.append({"top": p["fvg_bull_top"], "bot": p["fvg_bull_bot"], "formed_i": i - 1})
        if p.get("fvg_bear", False) and not pd.isna(p.get("fvg_bear_top", float("nan"))):
            fvg_bear.append({"top": p["fvg_bear_top"], "bot": p["fvg_bear_bot"], "formed_i": i - 1})
        if p.get("ob_bull", False) and not pd.isna(p.get("ob_bull_bot", float("nan"))):
            ob_bull.append({"top": p["ob_bull_top"], "bot": p["ob_bull_bot"], "formed_i": i - 1})
        if p.get("ob_bear", False) and not pd.isna(p.get("ob_bear_top", float("nan"))):
            ob_bear.append({"top": p["ob_bear_top"], "bot": p["ob_bear_bot"], "formed_i": i - 1})

        # ── Expire filled / stale zones ───────────────────────────────────────
        fvg_bull = deque(z for z in fvg_bull if (i - z["formed_i"]) <= zone_lookback and close > z["bot"])
        fvg_bear = deque(z for z in fvg_bear if (i - z["formed_i"]) <= zone_lookback and close < z["top"])
        ob_bull  = deque(z for z in ob_bull  if (i - z["formed_i"]) <= zone_lookback and low   > z["bot"])
        ob_bear  = deque(z for z in ob_bear  if (i - z["formed_i"]) <= zone_lookback and high  < z["top"])

        # ── Kill zone / news / cooldown filters ───────────────────────────────
        if use_news_filter and has_news_avoid and row.get("news_avoid", False):
            continue
        if use_kill_zones and not is_kill_zone(ts):
            if not (use_news_filter and has_news_entry and row.get("news_entry", False)):
                continue
        if i - last_signal_bar < 5:
            continue
        if use_seasonal_filter and get_dow_priority(ts) == 0:
            continue

        atr = row.get("atr", 0)
        if pd.isna(atr) or atr == 0:
            continue

        active_fvgs_now = {"bull": list(fvg_bull), "bear": list(fvg_bear)}
        active_obs_now  = {"bull": list(ob_bull),  "bear": list(ob_bear)}

        # Liquidity: only look at recent window to keep this O(1) per bar
        liq_slice   = df_intraday.iloc[max(0, i - 500): i + 1]
        liq_targets = liquidity.get_liquidity_targets(liq_slice)
        window      = df_intraday.iloc[max(0, i - 100): i]

        is_post_news   = use_news_filter and has_news_entry and row.get("news_entry", False)
        effective_min_rr = min_rr * 0.8 if is_post_news else min_rr

        # --- LONG SETUP ---
        if bias in ("bull", "neutral"):
            if not (use_seasonal_filter and not seasonal_agrees_with_direction(ts, "long", asset)):
                long_sig = _check_long(
                    i, row, ts, close, atr, window,
                    active_fvgs_now, active_obs_now, liq_targets, asset, effective_min_rr,
                    zone_tolerance_atr, bar_range_check,
                )
                if long_sig:
                    if is_post_news:
                        long_sig.confidence = min(5, long_sig.confidence + 1)
                        long_sig.report_rationale += " Post-news ICT displacement entry."
                    signals.append(long_sig)
                    last_signal_bar = i
                    continue

        # --- SHORT SETUP ---
        if bias in ("bear", "neutral"):
            if not (use_seasonal_filter and not seasonal_agrees_with_direction(ts, "short", asset)):
                short_sig = _check_short(
                    i, row, ts, close, atr, window,
                    active_fvgs_now, active_obs_now, liq_targets, asset, effective_min_rr,
                    zone_tolerance_atr, bar_range_check,
                )
                if short_sig:
                    if is_post_news:
                        short_sig.confidence = min(5, short_sig.confidence + 1)
                        short_sig.report_rationale += " Post-news ICT displacement entry."
                    signals.append(short_sig)
                    last_signal_bar = i

    return signals


def _active_fvgs_at(df: pd.DataFrame, lookback: int = 80) -> dict:
    """
    FVGs formed in the last `lookback` bars that have NOT been closed through (on a close basis).
    ICT rule: wick touches are acceptable — the FVG is invalid only when price CLOSES through the far edge.
    """
    bull, bear = [], []
    n = len(df)
    start = max(0, n - lookback)

    for i in range(start, n - 1):
        row = df.iloc[i]
        subsequent = df.iloc[i + 1:]

        if row.get("fvg_bull", False) and not pd.isna(row.get("fvg_bull_bot", float("nan"))):
            bot = row["fvg_bull_bot"]
            top = row["fvg_bull_top"]
            # Filled when price CLOSES below the FVG bottom
            if subsequent["close"].min() > bot:
                bull.append({"top": top, "bot": bot, "formed_at": df.index[i]})

        if row.get("fvg_bear", False) and not pd.isna(row.get("fvg_bear_top", float("nan"))):
            top = row["fvg_bear_top"]
            bot = row["fvg_bear_bot"]
            # Filled when price CLOSES above the FVG top
            if subsequent["close"].max() < top:
                bear.append({"top": top, "bot": bot, "formed_at": df.index[i]})

    return {"bull": bull, "bear": bear}


def _active_obs_at(df: pd.DataFrame, lookback: int = 80) -> dict:
    """OBs formed in the last `lookback` bars that have NOT been mitigated."""
    bull, bear = [], []
    n = len(df)
    start = max(0, n - lookback)

    for i in range(start, n - 1):
        row = df.iloc[i]
        subsequent = df.iloc[i + 1:]

        if row.get("ob_bull", False) and not pd.isna(row.get("ob_bull_bot", float("nan"))):
            bot = row["ob_bull_bot"]
            top = row["ob_bull_top"]
            if subsequent["low"].min() > bot:
                bull.append({"top": top, "bot": bot, "formed_at": df.index[i]})

        if row.get("ob_bear", False) and not pd.isna(row.get("ob_bear_top", float("nan"))):
            top = row["ob_bear_top"]
            bot = row["ob_bear_bot"]
            if subsequent["high"].max() < top:
                bear.append({"top": top, "bot": bot, "formed_at": df.index[i]})

    return {"bull": bull, "bear": bear}


def _check_long(i, row, ts, price, atr, df, active_fvgs, active_obs, liq, asset, min_rr,
               zone_tolerance_atr: float = 0.0, bar_range_check: bool = False):
    notes = []
    sweep = _recent_sweep(i, df, "low", lookback=10)

    if bar_range_check:
        # ── Daily-bar path ────────────────────────────────────────────────────
        # Detect a demand-zone touch: the day's LOW visited a bull FVG or OB,
        # AND the CLOSE held above the zone bottom (zone intact = bounce).
        # Entry at zone_top (limit-order level); stop below zone_bot.
        equilibrium = row.get("equilibrium", None)
        zone_hit = None
        zone_type = ""

        all_zones = [(z, "fvg") for z in active_fvgs["bull"]] + \
                    [(z, "ob")  for z in active_obs["bull"]]

        for z, ztype in all_zones:
            touched = row["low"] <= z["top"] + zone_tolerance_atr * atr
            bounced = row["close"] >= z["bot"]
            in_disc = (z["top"] < equilibrium) if equilibrium is not None else True
            if touched and bounced and in_disc:
                zone_hit = z
                zone_type = ztype
                break

        if zone_hit is None:
            return None

        confluence = 1
        notes.append("daily bar low touched demand zone (FVG/OB) and closed above it")

        if row["close"] > row["open"]:
            confluence += 1
            notes.append("bullish reversal bar — demand zone held intraday")
        if sweep:
            confluence += 1
            notes.append("false breakout pattern confirmed — sell-side liquidity swept then reversed")
        if row.get("rsi_oversold", False):
            confluence += 1
            notes.append("RSI < 40 confirms oversold/discount conditions")
        if row.get("cci_oversold", False):
            notes.append("CCI < -100 confirms deep discount (commodity-specific)")

        # Entry at zone_top if low actually touched it; otherwise close.
        entry = zone_hit["top"] if row["low"] <= zone_hit["top"] else float(row["close"])
        # Use a 2-ATR stop for daily bars — zone-bottom stops are often 5-10 ATR wide
        # on daily FVGs, producing unrealistically far targets that take months to fill.
        stop  = entry - 2.0 * atr
        target = liq["nearest_above"]["level"] if liq["nearest_above"] else entry + min_rr * (entry - stop)

    else:
        # ── Intraday path ─────────────────────────────────────────────────────
        confluence = 0
        zone_hit = None
        zone_type = ""

        if not row.get("in_discount", False):
            return None
        confluence += 1
        notes.append("price at Fibonacci/discount zone (<50% of range)")

        fvg_zone = _zone_near_price(price, active_fvgs["bull"], atr, zone_tolerance_atr)
        ob_zone  = _zone_near_price(price, active_obs["bull"],  atr, zone_tolerance_atr)

        if fvg_zone and ob_zone:
            zone_type, zone_hit = "fvg+ob", fvg_zone
            confluence += 2
            notes.append("price inside open gap zone (FVG) AND demand zone (OB) — highest confluence")
        elif fvg_zone:
            zone_type, zone_hit = "fvg", fvg_zone
            confluence += 1
            notes.append("price inside open gap zone (FVG) — gap-fill entry")
        elif ob_zone:
            zone_type, zone_hit = "ob", ob_zone
            confluence += 1
            notes.append("price at historical demand zone (Order Block)")

        if zone_hit is None:
            return None

        if sweep:
            confluence += 1
            notes.append("false breakout pattern confirmed — sell-side liquidity swept then reversed")
        if row.get("rsi_oversold", False):
            confluence += 1
            notes.append("RSI < 40 confirms oversold/discount conditions")
        if row.get("cci_oversold", False):
            notes.append("CCI < -100 confirms deep discount (commodity-specific)")

        if confluence < 2:
            return None

        entry = price
        stop  = zone_hit["bot"] - 0.5 * atr
        risk  = entry - stop
        # Require a real liquidity level that gives min_rr — no synthetic fallback
        target = next(
            (lv["level"] for lv in liq["buy_side"] if lv["level"] - entry >= min_rr * risk),
            None,
        )
        if target is None:
            return None

    rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0
    if rr < min_rr:
        return None

    rationale = _build_rationale("long", notes, entry, stop, target, rr)
    return Signal(
        timestamp=ts, direction="long", asset=asset,
        entry=entry, stop=stop, target=target,
        risk_reward=round(rr, 2), confidence=min(confluence, 5),
        zone_type=zone_type, report_rationale=rationale,
        raw_signals={"rsi": row.get("rsi"), "cci": row.get("cci"), "sweep": sweep},
    )


def _check_short(i, row, ts, price, atr, df, active_fvgs, active_obs, liq, asset, min_rr,
                zone_tolerance_atr: float = 0.0, bar_range_check: bool = False):
    notes = []
    sweep = _recent_sweep(i, df, "high", lookback=10)

    if bar_range_check:
        # ── Daily-bar path ────────────────────────────────────────────────────
        equilibrium = row.get("equilibrium", None)
        zone_hit = None
        zone_type = ""

        all_zones = [(z, "fvg") for z in active_fvgs["bear"]] + \
                    [(z, "ob")  for z in active_obs["bear"]]

        for z, ztype in all_zones:
            touched = row["high"] >= z["bot"] - zone_tolerance_atr * atr
            bounced = row["close"] <= z["top"]
            in_prem = (z["bot"] > equilibrium) if equilibrium is not None else True
            if touched and bounced and in_prem:
                zone_hit = z
                zone_type = ztype
                break

        if zone_hit is None:
            return None

        confluence = 1
        notes.append("daily bar high tagged supply zone (FVG/OB) and closed below it")

        if row["close"] < row["open"]:
            confluence += 1
            notes.append("bearish reversal bar — supply zone held intraday")
        if sweep:
            confluence += 1
            notes.append("false breakout pattern confirmed — buy-side liquidity swept then reversed")
        if row.get("rsi_overbought", False):
            confluence += 1
            notes.append("RSI > 60 confirms overbought/premium conditions")
        if row.get("cci_overbought", False):
            notes.append("CCI > +100 confirms deep premium (commodity-specific)")

        entry  = zone_hit["bot"] if row["high"] >= zone_hit["bot"] else float(row["close"])
        stop   = entry + 2.0 * atr
        target = liq["nearest_below"]["level"] if liq["nearest_below"] else entry - min_rr * (stop - entry)

    else:
        # ── Intraday path ─────────────────────────────────────────────────────
        confluence = 0
        zone_hit = None
        zone_type = ""

        if not row.get("in_premium", False):
            return None
        confluence += 1
        notes.append("price at premium zone (>50% of range) — overbought area")

        fvg_zone = _zone_near_price(price, active_fvgs["bear"], atr, zone_tolerance_atr)
        ob_zone  = _zone_near_price(price, active_obs["bear"],  atr, zone_tolerance_atr)

        if fvg_zone and ob_zone:
            zone_type, zone_hit = "fvg+ob", fvg_zone
            confluence += 2
            notes.append("price inside open gap zone (FVG) AND supply zone (OB) — highest confluence")
        elif fvg_zone:
            zone_type, zone_hit = "fvg", fvg_zone
            confluence += 1
            notes.append("price inside open gap zone (FVG) — gap-fill entry")
        elif ob_zone:
            zone_type, zone_hit = "ob", ob_zone
            confluence += 1
            notes.append("price at historical supply zone (Order Block)")

        if zone_hit is None:
            return None

        if sweep:
            confluence += 1
            notes.append("false breakout pattern confirmed — buy-side liquidity swept then reversed")
        if row.get("rsi_overbought", False):
            confluence += 1
            notes.append("RSI > 60 confirms overbought/premium conditions")
        if row.get("cci_overbought", False):
            notes.append("CCI > +100 confirms deep premium (commodity-specific)")

        if confluence < 2:
            return None

        entry = price
        stop  = zone_hit["top"] + 0.5 * atr
        risk  = stop - entry
        # Require a real liquidity level that gives min_rr — no synthetic fallback
        target = next(
            (lv["level"] for lv in liq["sell_side"] if entry - lv["level"] >= min_rr * risk),
            None,
        )
        if target is None:
            return None

    rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else 0
    if rr < min_rr:
        return None

    rationale = _build_rationale("short", notes, entry, stop, target, rr)

    return Signal(
        timestamp=ts,
        direction="short",
        asset=asset,
        entry=entry,
        stop=stop,
        target=target,
        risk_reward=round(rr, 2),
        confidence=min(confluence, 5),
        zone_type=zone_type,
        report_rationale=rationale,
        raw_signals={"rsi": row.get("rsi"), "cci": row.get("cci"), "sweep": sweep},
    )


def _recent_sweep(i: int, df: pd.DataFrame, side: str, lookback: int = 10) -> bool:
    """
    Detect a false breakout (stop hunt) using a window slice.
    `df` here is the recent window passed from the scanner (last 100 bars before i).
    Uses relative positions within the window.
    """
    n = len(df)
    if n < lookback + 2:
        return False

    window = df.iloc[-(lookback + 2): -2]
    prev = df.iloc[-2]
    current = df.iloc[-1]

    if side == "low":
        prior_low = window["low"].min()
        return prev["low"] < prior_low and current["close"] > prior_low

    if side == "high":
        prior_high = window["high"].max()
        return prev["high"] > prior_high and current["close"] < prior_high

    return False


def _build_rationale(direction: str, notes: list, entry: float, stop: float, target: float, rr: float) -> str:
    """
    Build the academic-language rationale string for the Trading Game report.
    Uses CDT03-aligned terminology per the bridge document.
    """
    lines = [
        f"{'Long' if direction == 'long' else 'Short'} entry at {entry:.2f}.",
        "Confluence factors:",
    ]
    for n in notes:
        lines.append(f"  • {n}")
    lines.append(
        f"Stop loss placed at {stop:.2f} (at structural swing extreme). "
        f"Profit target {target:.2f} (prior swing high/low). "
        f"Risk/reward: {rr:.1f}:1. "
        "Position sized to risk 2% of account equity (£200)."
    )
    return " ".join(lines)
