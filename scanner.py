"""
Live Signal Scanner — SMM591 Trading Game
Run this to get current ICT signals on your 3 commodity positions.

Usage:
    python3 scanner.py
    python3 scanner.py --asset gold
    python3 scanner.py --chart
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from datetime import datetime
import pytz

from data.fetcher import fetch, fetch_multi_tf, ASSETS
from strategy.ict_strategy import scan_for_signals, get_htf_bias, prepare_data
from detectors.liquidity import get_liquidity_targets


TRADING_GAME_ASSETS = {
    "Gold":      "gold",    # GC=F — Metals
    "Crude Oil": "crude",   # CL=F — Energy
    "Corn":      "ZC=F",    # Agricultural
}

LONDON_TZ = pytz.timezone("Europe/London")
NY_TZ = pytz.timezone("America/New_York")


def scan_asset(name: str, symbol: str, chart: bool = False):
    print(f"\n{'='*55}")
    print(f"  {name.upper()} ({ASSETS.get(symbol, symbol)})")
    print(f"{'='*55}")

    try:
        tf = fetch_multi_tf(symbol)
        df_daily = tf["daily"]
        df_1h = tf["1h"]
    except Exception as e:
        print(f"  Error: {e}")
        return

    df_daily_p, df_1h_p = prepare_data(df_daily, df_1h)
    bias = get_htf_bias(df_daily_p)
    last = df_1h_p.iloc[-1]
    price = last["close"]
    liq = get_liquidity_targets(df_1h_p)

    # Current session info
    now_ny = datetime.now(NY_TZ)
    now_london = datetime.now(LONDON_TZ)

    print(f"  Current price   : {price:.2f}")
    print(f"  HTF bias        : {bias.upper()}")
    print(f"  EMA 50          : {last.get('ema_50', 'N/A'):.2f}" if last.get('ema_50') else "  EMA 50          : N/A")
    print(f"  RSI(14)         : {last.get('rsi', 'N/A'):.1f}" if last.get('rsi') else "  RSI(14)         : N/A")
    print(f"  CCI(20)         : {last.get('cci', 'N/A'):.1f}" if last.get('cci') else "  CCI(20)         : N/A")
    print(f"  In discount     : {last.get('in_discount', False)}")
    print(f"  In premium      : {last.get('in_premium', False)}")
    print(f"  NY time         : {now_ny.strftime('%H:%M')}  |  London: {now_london.strftime('%H:%M')}")

    # Nearest liquidity levels
    if liq["nearest_above"]:
        print(f"  Next target ↑   : {liq['nearest_above']['level']:.2f} ({liq['nearest_above']['type']})")
    if liq["nearest_below"]:
        print(f"  Next target ↓   : {liq['nearest_below']['level']:.2f} ({liq['nearest_below']['type']})")

    # Scan for signals (last 30 days of 1h data)
    df_1h_recent = df_1h.tail(30 * 16)
    signals = scan_for_signals(name, df_daily, df_1h_recent, min_rr=2.0)

    if not signals:
        print(f"\n  No active signals found in the last 30 days.")
    else:
        print(f"\n  Signals found: {len(signals)}")
        for sig in signals[-3:]:  # show last 3
            direction_icon = "▲ LONG" if sig.direction == "long" else "▼ SHORT"
            print(f"\n  [{sig.timestamp:%Y-%m-%d %H:%M}] {direction_icon}")
            print(f"    Entry:  {sig.entry:.4f}")
            print(f"    Stop:   {sig.stop:.4f}")
            print(f"    Target: {sig.target:.4f}")
            print(f"    R:R:    {sig.risk_reward}:1  |  Confidence: {sig.confidence}/5")
            print(f"    Zone:   {sig.zone_type}")
            print(f"    Report text:")
            print(f"      {sig.report_rationale[:250]}...")

    if chart:
        try:
            from backtest.charts import plot_trades
            plot_trades(name, df_1h_p, [], last_n_bars=150,
                        save_path=f"backtest/{name.lower().replace(' ','_')}_chart.png")
        except Exception as e:
            print(f"  Chart error: {e}")


def main():
    parser = argparse.ArgumentParser(description="ICT Commodity Signal Scanner — SMM591")
    parser.add_argument("--asset", type=str, default=None, help="Specific asset (gold/crude/corn)")
    parser.add_argument("--chart", action="store_true", help="Save chart PNG")
    args = parser.parse_args()

    print(f"\n ICT/MMXM Commodity Scanner — SMM591 Trading Game")
    print(f" {datetime.now(LONDON_TZ).strftime('%A %d %B %Y, %H:%M London')}")
    print(f" Strategy: False-breakout + FVG/OB entry | Kill zones: NY 08:30-12:00, 13:30-16:00")

    assets_to_scan = TRADING_GAME_ASSETS
    if args.asset:
        key = args.asset.lower()
        match = {k: v for k, v in TRADING_GAME_ASSETS.items() if key in k.lower()}
        if not match:
            print(f"Unknown asset: {args.asset}. Options: gold, crude, corn")
            return
        assets_to_scan = match

    for name, symbol in assets_to_scan.items():
        scan_asset(name, symbol, chart=args.chart)

    print(f"\n{'='*55}")
    print(f"  TRADING GAME REMINDER")
    print(f"{'='*55}")
    print(f"  Min contracts: 3 (energy + metals + agriculture)")
    print(f"  Risk per trade: 2% = £200 on £10,000 account")
    print(f"  Stop = structural swing extreme + 0.5×ATR")
    print(f"  Target = next prior swing high/low")
    print(f"  Report deadline: Monday 6 July 2026, 23:59 London")
    print(f"  Max report pages: 15 (no +10% allowance)")
    print()


if __name__ == "__main__":
    main()
