"""
Daily Trend-Following Strategy for Commodity CFDs
===================================================
Academic framework: Donchian/EMA trend filter + RSI momentum entry

Entry logic (confluence required):
  1. HTF bias  — EMA50 > EMA200 on WEEKLY bars = bull trend (golden cross)
  2. Pullback  — RSI(14) on daily dipped below 45 for longs (/<55 for shorts)
                 i.e. entered discount/oversold zone but not crashed
  3. Reversal  — RSI crosses back above 45 (bulls) / below 55 (bears)
                 AND price closes above EMA20 on daily (confirmation)
  4. Structure — entry bar is inside the EMA20–EMA50 corridor (value area)

Stop: 2 × ATR(14) below entry
Target: min_rr × risk (or nearest prior swing high/low if calculable)

This generates ~20-50 trades per year per asset on daily bars.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class DailySignal:
    timestamp: pd.Timestamp
    direction: str       # "long" or "short"
    asset: str
    entry: float
    stop: float
    target: float
    risk_reward: float
    confidence: int      # 1–4
    report_rationale: str
    raw_signals: dict = field(default_factory=dict)

    @property
    def zone_type(self) -> str:
        return "trend_pullback"


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]

    df["ema_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["ema_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["ema_200"] = c.ewm(span=200, adjust=False).mean()

    # RSI
    delta = c.diff()
    up   = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    rs   = up.ewm(com=13, adjust=False).mean() / down.ewm(com=13, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + rs)

    # ATR
    hl  = df["high"] - df["low"]
    hpc = (df["high"] - df["close"].shift()).abs()
    lpc = (df["low"]  - df["close"].shift()).abs()
    df["atr"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).ewm(span=14, adjust=False).mean()

    # Swing highs / lows for liquidity targets (rolling 20-bar)
    df["swing_high"] = df["high"].rolling(20).max()
    df["swing_low"]  = df["low"].rolling(20).min()

    return df


def get_htf_bias(df_weekly: pd.DataFrame) -> str:
    """Weekly EMA50 vs EMA200 trend direction."""
    df = df_weekly.copy()
    c = df["close"]
    ema50  = c.ewm(span=50,  adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    df["w_ema50"]  = ema50
    df["w_ema200"] = ema200

    if len(df) < 50:
        return "neutral"

    last = df.iloc[-1]
    e50  = last["w_ema50"]
    e200 = last["w_ema200"]

    if e50 > e200 and last["close"] > e50:
        return "bull"
    if e50 < e200 and last["close"] < e50:
        return "bear"
    # Mixed — use price vs EMA50 as tiebreaker
    return "bull" if last["close"] > e50 else "bear"


def scan_daily_signals(
    asset: str,
    df_weekly: pd.DataFrame,
    df_daily: pd.DataFrame,
    min_rr: float = 2.0,
    rsi_entry_long: float = 45.0,   # RSI recross above this = long entry
    rsi_entry_short: float = 55.0,  # RSI recross below this = short entry
    rsi_dip_long: float = 45.0,     # RSI must have dipped below this recently
    rsi_dip_short: float = 55.0,    # RSI must have risen above this recently
) -> list[DailySignal]:
    """
    Walk-forward scan for daily trend-pullback entries.
    Only uses data available up to each bar (no lookahead).
    """
    df = _add_indicators(df_daily.copy())

    signals = []
    last_sig_bar = -3   # min gap between signals

    for i in range(210, len(df)):   # need 200 bars for EMA200
        row  = df.iloc[i]
        prev = df.iloc[i - 1]
        ts   = df.index[i]

        # HTF bias from weekly (use weekly data up to this date)
        week_slice = df_weekly[df_weekly.index <= ts]
        if len(week_slice) < 52:
            continue
        bias = get_htf_bias(week_slice)

        if i - last_sig_bar < 3:
            continue

        atr = row.get("atr")
        if pd.isna(atr) or atr <= 0:
            continue

        rsi      = row.get("rsi")
        rsi_prev = prev.get("rsi")
        if pd.isna(rsi) or pd.isna(rsi_prev):
            continue

        ema20  = row.get("ema_20")
        ema50  = row.get("ema_50")
        close  = row["close"]

        # ── LONG setup ──────────────────────────────────────────────────────────
        if bias in ("bull", "neutral"):
            # RSI crossed above rsi_entry_long (pullback-then-reversal)
            rsi_cross_up = rsi_prev < rsi_dip_long and rsi >= rsi_entry_long
            price_above_ema20 = close > ema20 if not pd.isna(ema20) else False
            in_value = (ema20 <= close <= ema50) if not pd.isna(ema50) else False

            if rsi_cross_up and price_above_ema20:
                conf = 1
                notes = [f"RSI crossed above {rsi_entry_long:.0f} from oversold territory"]

                if close > ema20:
                    conf += 1
                    notes.append("daily close above EMA20 (short-term trend confirmed)")
                if close > ema50:
                    conf += 1
                    notes.append("price above EMA50 (intermediate trend bullish)")
                if row.get("close") > prev.get("close"):
                    conf += 1
                    notes.append("bullish price action — higher close than prior day")

                entry  = float(close)
                stop   = entry - 2.0 * atr
                # Target: recent swing high OR fixed RR
                swing_h = row.get("swing_high")
                target = float(swing_h) if (not pd.isna(swing_h) and swing_h > entry + min_rr * (entry - stop)) \
                         else entry + min_rr * (entry - stop)

                rr = (target - entry) / (entry - stop) if (entry - stop) > 0 else 0
                if rr < min_rr:
                    target = entry + min_rr * (entry - stop)
                    rr = min_rr

                rationale = (
                    f"Long entry at {entry:.4f}. "
                    f"Confluence: {'; '.join(notes)}. "
                    f"Weekly EMA50 > EMA200 confirms primary uptrend (Dow Theory). "
                    f"Stop at {stop:.4f} (2×ATR = {2*atr:.4f}). "
                    f"Target {target:.4f}. R:R {rr:.1f}:1. "
                    f"Position sized to 2% account risk."
                )
                signals.append(DailySignal(
                    timestamp=ts, direction="long", asset=asset,
                    entry=entry, stop=stop, target=target,
                    risk_reward=round(rr, 2), confidence=min(conf, 4),
                    report_rationale=rationale,
                    raw_signals={"rsi": rsi, "atr": atr, "bias": bias},
                ))
                last_sig_bar = i
                continue

        # ── SHORT setup ─────────────────────────────────────────────────────────
        if bias in ("bear", "neutral"):
            rsi_cross_dn = rsi_prev > rsi_dip_short and rsi <= rsi_entry_short
            price_below_ema20 = close < ema20 if not pd.isna(ema20) else False

            if rsi_cross_dn and price_below_ema20:
                conf = 1
                notes = [f"RSI crossed below {rsi_entry_short:.0f} from overbought territory"]

                if close < ema20:
                    conf += 1
                    notes.append("daily close below EMA20 (short-term trend confirmed)")
                if close < ema50:
                    conf += 1
                    notes.append("price below EMA50 (intermediate trend bearish)")
                if row.get("close") < prev.get("close"):
                    conf += 1
                    notes.append("bearish price action — lower close than prior day")

                entry  = float(close)
                stop   = entry + 2.0 * atr
                swing_l = row.get("swing_low")
                target = float(swing_l) if (not pd.isna(swing_l) and swing_l < entry - min_rr * (stop - entry)) \
                         else entry - min_rr * (stop - entry)

                rr = (entry - target) / (stop - entry) if (stop - entry) > 0 else 0
                if rr < min_rr:
                    target = entry - min_rr * (stop - entry)
                    rr = min_rr

                rationale = (
                    f"Short entry at {entry:.4f}. "
                    f"Confluence: {'; '.join(notes)}. "
                    f"Weekly EMA50 < EMA200 confirms primary downtrend (Dow Theory). "
                    f"Stop at {stop:.4f} (2×ATR = {2*atr:.4f}). "
                    f"Target {target:.4f}. R:R {rr:.1f}:1. "
                    f"Position sized to 2% account risk."
                )
                signals.append(DailySignal(
                    timestamp=ts, direction="short", asset=asset,
                    entry=entry, stop=stop, target=target,
                    risk_reward=round(rr, 2), confidence=min(conf, 4),
                    report_rationale=rationale,
                    raw_signals={"rsi": rsi, "atr": atr, "bias": bias},
                ))
                last_sig_bar = i

    return signals
