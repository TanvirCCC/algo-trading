"""
Supabase sync layer — mirrors CSV bridge data to cloud database.
Dashboard reads from here when running on Streamlit Cloud.
"""

import os
import logging
from datetime import datetime, timezone

log = logging.getLogger("qgts.supabase")

SUPABASE_URL = "https://uuaqxgugwlqmiujkumqc.supabase.co"
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "sb_publishable_1Xsrk5dAvf64gaxIxRcnWQ_naE3kZod")

_client = None

def _get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from supabase import create_client
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
        return _client
    except Exception as e:
        log.warning(f"Supabase client init failed: {e}")
        return None


def push_status(state: str, symbol: str, equity: float, spread: float) -> None:
    sb = _get_client()
    if not sb:
        return
    try:
        sb.table("status").upsert({
            "id": 1,
            "state": state,
            "symbol": symbol,
            "equity": equity,
            "spread": spread,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception as e:
        log.warning(f"Supabase push_status failed: {e}")


def push_signal(sig_id: int, signal) -> None:
    sb = _get_client()
    if not sb:
        return
    try:
        sb.table("signals").upsert({
            "id": sig_id,
            "timestamp": str(signal.timestamp),
            "direction": signal.direction,
            "entry": round(float(signal.entry), 5),
            "stop": round(float(signal.stop), 5),
            "target": round(float(signal.target), 5),
            "confidence": int(signal.confidence),
            "zone_type": str(signal.zone_type),
            "rr": round(float(signal.risk_reward), 2),
            "status": "NEW",
            "rationale": signal.report_rationale.replace(",", ";"),
        }).execute()
    except Exception as e:
        log.warning(f"Supabase push_signal failed: {e}")


def push_equity(equity: float) -> None:
    sb = _get_client()
    if not sb:
        return
    try:
        sb.table("equity_history").insert({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": equity,
        }).execute()
    except Exception as e:
        log.warning(f"Supabase push_equity failed: {e}")
