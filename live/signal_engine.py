"""
lnterqo v3 Live Signal Engine
==============================
Runs every 60s during London/NY sessions.
  1. Reads live bar data exported by MT4 EA
  2. Runs lnterqo v3 scanner + frozen ML filter
  3. Writes new signals to CSV bridge for MT4 EA to execute
  4. Logs equity updates from MT4 trade results
"""

import sys, os, time, logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from live.config import *
from live.supabase_sync import push_signal, push_equity, push_status
from data.news_calendar import get_high_impact_events, mark_news_windows
from detectors.seasonality import add_seasonal_columns
from detectors.regime import cusum_events
from strategy.lnterqo_strategy import prepare_data, scan_for_signals
from strategy.ml_filter import SignalFilter
from strategy.risk_manager import RiskManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("live/signal_engine.log"),
    ],
)
log = logging.getLogger("lnterqo_live")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _in_session() -> bool:
    """True if current UTC time is within London or NY kill zone."""
    now = datetime.now(timezone.utc)
    h, m = now.hour, now.minute
    london = (7, 0) <= (h, m) < (10, 0)
    ny     = (12, 0) <= (h, m) < (15, 0)
    return london or ny


def _load_bars(path: Path, n: int = 1500) -> pd.DataFrame | None:
    """Load OHLCV CSV written by MT4 EA."""
    if not path.exists():
        log.warning(f"Bar file not found: {path}")
        return None
    try:
        df = pd.read_csv(path, parse_dates=["time"])
        df = df.rename(columns={"time": "timestamp"})
        df = df.set_index("timestamp").sort_index()
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].tail(n)
        df = df[df["close"] > 0]
        return df
    except Exception as e:
        log.error(f"Failed to load {path}: {e}")
        return None


def _enrich(df_5m: pd.DataFrame, df_daily: pd.DataFrame):
    """Add indicators, market structure, news windows."""
    news = get_high_impact_events(df_5m.index[0], df_5m.index[-1])
    df_5m = mark_news_windows(df_5m, news, pre_minutes=PRE_NEWS_MIN, post_hours=POST_NEWS_HRS)
    df_5m = add_seasonal_columns(df_5m, asset=ASSET_LABEL)
    df_daily_enr, df_5m_enr = prepare_data(df_daily.copy(), df_5m.copy())
    if "cusum_up" not in df_5m_enr.columns:
        df_5m_enr = cusum_events(df_5m_enr)
    return df_daily_enr, df_5m_enr


def _read_last_signal_id() -> int:
    if not SIGNAL_FILE.exists():
        return 0
    try:
        df = pd.read_csv(SIGNAL_FILE)
        return int(df["id"].max()) if len(df) > 0 else 0
    except Exception:
        return 0


def _write_signal(sig_id: int, signal) -> None:
    """Append signal to CSV bridge file."""
    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "id":          sig_id,
        "timestamp":   signal.timestamp,
        "direction":   signal.direction,
        "entry":       round(signal.entry, 5),
        "stop":        round(signal.stop, 5),
        "target":      round(signal.target, 5),
        "confidence":  signal.confidence,
        "zone_type":   signal.zone_type,
        "status":      "NEW",
        "rr":          round(signal.risk_reward, 2),
        "rationale":   signal.report_rationale.replace(",", ";"),
    }])
    header = not SIGNAL_FILE.exists()
    row.to_csv(SIGNAL_FILE, mode="a", header=header, index=False)
    push_signal(sig_id, signal)
    log.info(f"Signal written: id={sig_id} {signal.direction} {signal.zone_type} "
             f"entry={signal.entry:.2f} sl={signal.stop:.2f} tp={signal.target:.2f} "
             f"rr={signal.risk_reward:.1f} conf={signal.confidence}")


def _update_equity_history(equity: float) -> None:
    """Append current equity to history CSV for dashboard."""
    EQUITY_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"timestamp": datetime.now(timezone.utc), "equity": equity}])
    header = not EQUITY_HISTORY_FILE.exists()
    row.to_csv(EQUITY_HISTORY_FILE, mode="a", header=header, index=False)
    push_equity(equity)


def _read_equity_from_status() -> float:
    """Read last known equity from MT4 status file."""
    if not STATUS_FILE.exists():
        return INITIAL_EQUITY
    try:
        df = pd.read_csv(STATUS_FILE)
        return float(df["equity"].iloc[-1])
    except Exception:
        return INITIAL_EQUITY


# ── ML filter — trained on full historical IS, frozen for live use ─────────────

def load_or_train_ml_filter() -> SignalFilter | None:
    """
    Load a pre-trained ML filter saved from the v3 backtest, or train fresh
    on the backtest trade log if available.
    """
    trade_log = Path("backtest/lnterqo_v3_oos_trades.csv")
    parquet   = Path("data/historical/gold_5m_combined.parquet")

    if not trade_log.exists() or not parquet.exists():
        log.warning("No trade log or parquet found — ML filter disabled.")
        return None

    try:
        log.info("Training ML filter on v3 OOS trade log...")
        from backtest.engine import Trade
        from strategy.lnterqo_strategy import Signal as LnterqoSignal
        from dataclasses import field

        df_hist = pd.read_parquet(parquet)
        df_hist.columns = [c.lower() for c in df_hist.columns]
        df_hist = df_hist[["open", "high", "low", "close", "volume"]]

        news = get_high_impact_events(df_hist.index[0], df_hist.index[-1])
        df_hist = mark_news_windows(df_hist, news, pre_minutes=PRE_NEWS_MIN, post_hours=POST_NEWS_HRS)
        df_hist = add_seasonal_columns(df_hist, asset=ASSET_LABEL)
        df_daily_hist = df_hist.resample("D").agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"), close=("close","last"), volume=("volume","sum")
        ).dropna(subset=["open"])
        _, df_hist_enr = prepare_data(df_daily_hist.copy(), df_hist.copy())
        if "cusum_up" not in df_hist_enr.columns:
            df_hist_enr = cusum_events(df_hist_enr)

        trades_df = pd.read_csv(trade_log, parse_dates=["entry_time"])

        # Reconstruct minimal Trade objects for ML training
        trades = []
        for _, row in trades_df.iterrows():
            if row["outcome"] not in ("win", "loss"):
                continue
            ts = pd.Timestamp(row["entry_time"])
            if ts not in df_hist_enr.index:
                continue
            close = df_hist_enr.loc[ts, "close"] if ts in df_hist_enr.index else row["entry"]
            sig = LnterqoSignal(
                timestamp=ts, asset=ASSET_LABEL,
                direction=row["direction"],
                entry=row["entry"], stop=row["stop"], target=row["target"],
                confidence=int(row["confidence"]),
                risk_reward=float(row.get("rr", abs(row["target"]-row["entry"])/max(abs(row["entry"]-row["stop"]),1e-9))),
                zone_type=str(row.get("zone_type", "CISD")),
            )
            risk = abs(row["entry"] - row["stop"])
            size = 1.0 / risk if risk > 0 else 1.0
            t = Trade(signal=sig, size=size, entry_price=row["entry"])
            t.outcome = row["outcome"]
            t.pnl = size * (row["target"] - row["entry"] if row["outcome"] == "win" else row["stop"] - row["entry"])
            trades.append(t)

        if len(trades) < 10:
            log.warning(f"Only {len(trades)} trades for ML — filter disabled.")
            return None

        ml = SignalFilter(probability_threshold=ML_THRESHOLD)
        ml.fit(trades, df_hist_enr)
        log.info(f"ML filter trained on {len(trades)} trades.")
        return ml

    except Exception as e:
        log.error(f"ML filter training failed: {e}")
        return None


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("  lnterqo v3 Live Signal Engine — starting")
    log.info(f"  Symbol: {SYMBOL}  |  Min RR: {MIN_RR}  |  Conf≥: {MIN_CONFIDENCE}")
    log.info(f"  Bridge: {BRIDGE_DIR}")
    log.info("=" * 60)

    BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

    ml_filter = load_or_train_ml_filter()
    rm        = RiskManager(equity=_read_equity_from_status())
    last_signal_id   = _read_last_signal_id()
    last_signal_ts   = None
    current_date     = None
    scan_count       = 0

    log.info(f"Starting equity: £{rm.equity:,.2f}")
    log.info(f"ML filter: {'enabled' if ml_filter else 'disabled'}")

    while True:
        try:
            now = datetime.now(timezone.utc)

            # Daily reset
            if now.date() != current_date:
                rm.reset_day()
                current_date = now.date()
                log.info(f"Day reset — equity: £{rm.equity:,.2f}")

            # Update equity from MT4 status
            live_equity = _read_equity_from_status()
            if abs(live_equity - rm.equity) > 1.0:
                rm.equity = live_equity
                _update_equity_history(live_equity)

            # Push heartbeat to Supabase so dashboard shows RUNNING
            _spread = 0.0
            if STATUS_FILE.exists():
                try:
                    _spread = float(pd.read_csv(STATUS_FILE)["spread"].iloc[-1])
                except Exception:
                    pass
            push_status("RUNNING", SYMBOL, rm.equity, _spread)

            # Only scan during sessions
            if not _in_session():
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            can_trade, reason = rm.can_trade()
            if not can_trade:
                log.info(f"Risk block: {reason}")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            # Load live bars from MT4 EA
            df_5m_raw   = _load_bars(BARS_5M_FILE, n=BARS_HISTORY)
            df_d1_raw   = _load_bars(BARS_D1_FILE, n=500)

            if df_5m_raw is None or df_d1_raw is None or len(df_5m_raw) < 200:
                log.warning("Waiting for MT4 bar data...")
                time.sleep(SCAN_INTERVAL_SEC)
                continue

            scan_count += 1
            log.info(f"Scan #{scan_count}  bars={len(df_5m_raw)}  equity=£{rm.equity:,.2f}")

            # Enrich
            df_daily_enr, df_5m_enr = _enrich(df_5m_raw, df_d1_raw)

            # Scan for signals
            signals = scan_for_signals(
                ASSET_LABEL, df_daily_enr, df_5m_enr,
                min_rr=MIN_RR,
                use_news_filter=USE_NEWS_FILTER,
                zone_lookback=ZONE_LOOKBACK,
                cisd_lookback=CISD_LOOKBACK,
                force_neutral_bias=True,
            )

            # Filter by minimum confidence
            signals = [s for s in signals if s.confidence >= MIN_CONFIDENCE]

            # ML filter
            if ml_filter:
                signals = [s for s in signals if ml_filter.accept(s, df_5m_enr)]

            # Find new signals (timestamp after last signal)
            new_signals = [
                s for s in signals
                if last_signal_ts is None or s.timestamp > last_signal_ts
            ]

            for sig in new_signals:
                last_signal_id += 1
                _write_signal(last_signal_id, sig)
                last_signal_ts = sig.timestamp

        except KeyboardInterrupt:
            log.info("Signal engine stopped.")
            break
        except Exception as e:
            log.error(f"Error in scan loop: {e}", exc_info=True)

        time.sleep(SCAN_INTERVAL_SEC)


if __name__ == "__main__":
    main()
