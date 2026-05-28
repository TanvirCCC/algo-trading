"""
Live trading configuration — lnterqo v3 on CMC Markets MT4
Edit SYMBOL to match what appears in your MT4 Market Watch.
"""

from pathlib import Path

# ── MT4 bridge paths (auto-detected for this Mac install) ────────────────────
MT4_FILES_DIR = Path.home() / "Library/Application Support/net.metaquotes.wine.metatrader4/drive_c/users/crossover/AppData/Roaming/MetaQuotes/Terminal/Common/Files"

BRIDGE_DIR    = MT4_FILES_DIR   # where EA reads/writes
SIGNAL_FILE   = BRIDGE_DIR / "lnterqo_signals.csv"
TRADES_FILE   = BRIDGE_DIR / "lnterqo_trades.csv"
BARS_5M_FILE  = BRIDGE_DIR / "gold_5m_live.csv"
BARS_D1_FILE  = BRIDGE_DIR / "gold_d1_live.csv"
STATUS_FILE   = BRIDGE_DIR / "lnterqo_status.csv"

# ── Instrument ────────────────────────────────────────────────────────────────
SYMBOL        = "GOLD"          # CMC Markets MT4 symbol name (case-sensitive)
ASSET_LABEL   = "Gold"
TIMEFRAME     = "M5"

# ── Strategy (v3 best params from rolling WFV) ────────────────────────────────
MIN_RR        = 2.0
ZONE_LOOKBACK = 50
CISD_LOOKBACK = 40
ML_THRESHOLD  = 0.60
MIN_CONFIDENCE = 1
USE_NEWS_FILTER = True
PRE_NEWS_MIN  = 15
POST_NEWS_HRS = 2

# ── Risk (v1 fixed tiers — matched to RiskManager) ────────────────────────────
INITIAL_EQUITY   = 10_000.0
DEFAULT_RISK_PCT = 0.005        # 0.5%
HIGH_CONF_RISK   = 0.010        # 1.0% for confidence >= 4
MAX_CONSEC_LOSS  = 5            # block after 5 consecutive losses
DAILY_LOSS_LIMIT = 0.02         # 2% daily cap

# ── Engine timing ─────────────────────────────────────────────────────────────
SCAN_INTERVAL_SEC = 60          # re-scan every 60s during sessions
BARS_HISTORY      = 2000        # how many 5m bars to load from MT4

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = 30
EQUITY_HISTORY_FILE   = Path("live/equity_history.csv")
