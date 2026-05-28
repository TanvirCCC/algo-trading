"""
Chart generation for the SMM591 Trading Game report.
Produces annotated candlestick charts showing:
  - FVG zones (shaded)
  - Order blocks (shaded)
  - Trade entries / SL / TP markers
  - EMAs (50, 200)
  - RSI and CCI subplots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from detectors import market_structure, fvg, order_blocks, indicators, liquidity
from backtest.engine import Trade


def plot_trades(
    asset: str,
    df: pd.DataFrame,
    trades: list[Trade],
    last_n_bars: int = 200,
    save_path: str | None = None,
):
    """
    Annotated chart showing price action with FVG zones, OBs, EMAs, and trade markers.
    """
    df = df.tail(last_n_bars).copy()
    df = indicators.add_all(df)
    df = fvg.detect_fvgs(df)
    df = order_blocks.detect_order_blocks(df)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[4, 1, 1, 1], hspace=0.05)
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax_price)
    ax_rsi = fig.add_subplot(gs[2], sharex=ax_price)
    ax_cci = fig.add_subplot(gs[3], sharex=ax_price)

    x = np.arange(len(df))
    idx = df.index

    # --- Candlesticks ---
    for i, (ts, row) in enumerate(df.iterrows()):
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax_price.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.8)
        ax_price.bar(i, abs(row["close"] - row["open"]),
                     bottom=min(row["open"], row["close"]),
                     color=color, width=0.6)

    # --- EMAs ---
    if "ema_50" in df.columns:
        ax_price.plot(x, df["ema_50"].values, color="#ff9800", linewidth=1.2, label="EMA 50")
    if "ema_200" in df.columns:
        ax_price.plot(x, df["ema_200"].values, color="#9c27b0", linewidth=1.2, label="EMA 200", linestyle="--")

    # --- FVG zones ---
    for i in range(len(df)):
        row = df.iloc[i]
        if row.get("fvg_bull", False) and not pd.isna(row.get("fvg_bull_bot", float("nan"))):
            ax_price.axhspan(row["fvg_bull_bot"], row["fvg_bull_top"],
                             xmin=i / len(df), xmax=min((i + 30) / len(df), 1.0),
                             color="#26a69a", alpha=0.12)
        if row.get("fvg_bear", False) and not pd.isna(row.get("fvg_bear_bot", float("nan"))):
            ax_price.axhspan(row["fvg_bear_bot"], row["fvg_bear_top"],
                             xmin=i / len(df), xmax=min((i + 30) / len(df), 1.0),
                             color="#ef5350", alpha=0.12)

    # --- Order blocks ---
    for i in range(len(df)):
        row = df.iloc[i]
        if row.get("ob_bull", False) and not pd.isna(row.get("ob_bull_bot", float("nan"))):
            ax_price.axhspan(row["ob_bull_bot"], row["ob_bull_top"],
                             xmin=i / len(df), xmax=min((i + 20) / len(df), 1.0),
                             color="#00bcd4", alpha=0.15, linewidth=0)
        if row.get("ob_bear", False) and not pd.isna(row.get("ob_bear_bot", float("nan"))):
            ax_price.axhspan(row["ob_bear_bot"], row["ob_bear_top"],
                             xmin=i / len(df), xmax=min((i + 20) / len(df), 1.0),
                             color="#ff5722", alpha=0.15, linewidth=0)

    # --- Trade markers ---
    for trade in trades:
        ts = trade.signal.timestamp
        if ts not in idx:
            continue
        xi = idx.get_loc(ts)
        color = "#26a69a" if trade.signal.direction == "long" else "#ef5350"
        marker = "^" if trade.signal.direction == "long" else "v"

        ax_price.scatter(xi, trade.entry_price, marker=marker, color=color, s=150, zorder=5)
        ax_price.axhline(trade.signal.stop, color="red", linewidth=0.8, linestyle=":", alpha=0.7)
        ax_price.axhline(trade.signal.target, color="#26a69a", linewidth=0.8, linestyle=":", alpha=0.7)

        label = f"{'L' if trade.signal.direction == 'long' else 'S'} {trade.outcome.upper()}"
        ax_price.annotate(label, (xi, trade.entry_price),
                          textcoords="offset points", xytext=(5, 8),
                          fontsize=7, color=color)

    # --- Volume ---
    vol_colors = ["#26a69a" if df.iloc[i]["close"] >= df.iloc[i]["open"] else "#ef5350"
                  for i in range(len(df))]
    ax_vol.bar(x, df["volume"].values, color=vol_colors, alpha=0.7)
    ax_vol.set_ylabel("Volume", fontsize=8)
    ax_vol.tick_params(labelbottom=False)

    # --- RSI ---
    if "rsi" in df.columns:
        ax_rsi.plot(x, df["rsi"].values, color="#2196f3", linewidth=1)
        ax_rsi.axhline(60, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_rsi.axhline(40, color="green", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_rsi.fill_between(x, df["rsi"].values, 40, where=df["rsi"].values < 40,
                            color="green", alpha=0.2)
        ax_rsi.fill_between(x, df["rsi"].values, 60, where=df["rsi"].values > 60,
                            color="red", alpha=0.2)
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI(14)", fontsize=8)
        ax_rsi.tick_params(labelbottom=False)

    # --- CCI ---
    if "cci" in df.columns:
        ax_cci.plot(x, df["cci"].values, color="#ff9800", linewidth=1)
        ax_cci.axhline(100, color="red", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_cci.axhline(-100, color="green", linewidth=0.7, linestyle="--", alpha=0.6)
        ax_cci.axhline(0, color="gray", linewidth=0.5, alpha=0.4)
        ax_cci.set_ylabel("CCI(20)", fontsize=8)

    # --- Labels ---
    tick_step = max(1, len(df) // 10)
    tick_positions = range(0, len(df), tick_step)
    ax_cci.set_xticks(list(tick_positions))
    ax_cci.set_xticklabels(
        [idx[i].strftime("%b %d") for i in tick_positions],
        rotation=30, fontsize=7
    )

    ax_price.set_title(
        f"{asset} — ICT Strategy Backtest\n"
        f"EMAs (50/200) | FVG zones (green/red) | OBs (cyan/orange) | Entries (▲▼)",
        fontsize=11
    )
    ax_price.legend(loc="upper left", fontsize=8)
    ax_price.set_ylabel("Price", fontsize=9)

    legend_elements = [
        mpatches.Patch(color="#26a69a", alpha=0.3, label="Bull FVG (gap zone)"),
        mpatches.Patch(color="#ef5350", alpha=0.3, label="Bear FVG (gap zone)"),
        mpatches.Patch(color="#00bcd4", alpha=0.3, label="Bull OB (demand zone)"),
        mpatches.Patch(color="#ff5722", alpha=0.3, label="Bear OB (supply zone)"),
    ]
    ax_price.legend(handles=legend_elements + ax_price.get_legend_handles_labels()[0],
                    loc="upper left", fontsize=7, ncol=2)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_equity_curve(
    equity_curves: dict[str, pd.Series],
    save_path: str | None = None,
):
    """Portfolio equity curve for all assets."""
    fig, ax = plt.subplots(figsize=(12, 5))

    colors = {"Gold": "#ffd700", "Crude Oil": "#ff6b35", "Corn": "#4caf50"}

    for name, curve in equity_curves.items():
        color = colors.get(name, "#2196f3")
        ax.plot(curve.values, label=name, color=color, linewidth=1.8)

    initial = list(equity_curves.values())[0].iloc[0]
    ax.axhline(initial, color="gray", linewidth=0.8, linestyle="--", label="Initial equity")

    ax.set_title("Equity Curve — ICT Commodity Strategy (SMM591 Trading Game)", fontsize=12)
    ax.set_ylabel("Account Equity (£)")
    ax.set_xlabel("Trade #")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Equity curve saved: {save_path}")
    else:
        plt.show()
    plt.close()
