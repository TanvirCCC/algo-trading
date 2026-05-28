"""
Quantitative Gold Trading System — Live Dashboard
=====================================
Run with:  streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime, timezone, timedelta

from live.config import *
from data.news_calendar import get_high_impact_events

# ── Supabase client (cloud) ───────────────────────────────────────────────────
def _supabase():
    try:
        from supabase import create_client
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QGTS | Live Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.metric-card { background:#1e1e2e; border-radius:8px; padding:12px 16px; margin:4px 0; }
.green { color: #00e676; } .red { color: #ff5252; } .gold { color: #ffd700; }
div[data-testid="metric-container"] { background:#1e1e2e; border-radius:8px; padding:8px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#1e1e2e; border-radius:6px 6px 0 0; padding: 6px 16px; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=DASHBOARD_REFRESH_SEC)
def load_signals() -> pd.DataFrame:
    # Try local CSV first
    if SIGNAL_FILE.exists():
        try:
            df = pd.read_csv(SIGNAL_FILE, parse_dates=["timestamp"])
            return df.sort_values("timestamp", ascending=False).reset_index(drop=True)
        except Exception:
            pass
    # Fallback: Supabase
    try:
        sb = _supabase()
        if sb:
            res = sb.table("signals").select("*").order("timestamp", desc=True).limit(200).execute()
            if res.data:
                return pd.DataFrame(res.data)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_REFRESH_SEC)
def load_trades() -> pd.DataFrame:
    if TRADES_FILE.exists():
        try:
            df = pd.read_csv(TRADES_FILE)
            open_rows  = df[df["status"] == "OPEN"].copy()
            close_rows = df[df["status"].isin(["WIN","LOSS"])].copy()
            if open_rows.empty:
                return pd.DataFrame()
            open_rows  = open_rows.set_index("ticket")
            close_rows = close_rows.set_index("ticket") if not close_rows.empty else pd.DataFrame()
            merged = open_rows.join(close_rows, rsuffix="_close", how="left") if not close_rows.empty else open_rows.copy()
            return merged.reset_index()
        except Exception:
            pass
    # Fallback: Supabase
    try:
        sb = _supabase()
        if sb:
            res = sb.table("trades").select("*").order("timestamp", desc=True).limit(100).execute()
            if res.data:
                return pd.DataFrame(res.data)
    except Exception:
        pass
    return pd.DataFrame()


@st.cache_data(ttl=DASHBOARD_REFRESH_SEC)
def load_equity() -> pd.DataFrame:
    if EQUITY_HISTORY_FILE.exists():
        try:
            df = pd.read_csv(EQUITY_HISTORY_FILE, parse_dates=["timestamp"])
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception:
            pass
    # Fallback: Supabase
    try:
        sb = _supabase()
        if sb:
            res = sb.table("equity_history").select("*").order("timestamp").execute()
            if res.data:
                df = pd.DataFrame(res.data)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return df.sort_values("timestamp").reset_index(drop=True)
    except Exception:
        pass
    return pd.DataFrame(columns=["timestamp", "equity"])


@st.cache_data(ttl=DASHBOARD_REFRESH_SEC)
def load_status() -> dict:
    if STATUS_FILE.exists():
        try:
            df = pd.read_csv(STATUS_FILE)
            return df.iloc[-1].to_dict()
        except Exception:
            pass
    # Fallback: Supabase
    try:
        sb = _supabase()
        if sb:
            res = sb.table("status").select("*").eq("id", 1).execute()
            if res.data:
                return res.data[0]
    except Exception:
        pass
    return {"state": "OFFLINE", "equity": INITIAL_EQUITY, "spread": 0, "timestamp": "—"}


@st.cache_data(ttl=300)
def load_historical_bars() -> pd.DataFrame:
    p = Path("data/historical/gold_5m_combined.parquet")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(p)
        df.columns = [c.lower() for c in df.columns]
        return df[["open", "high", "low", "close", "volume"]]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_backtest_trades() -> pd.DataFrame:
    p = Path("backtest/lnterqo_v3_oos_trades.csv")
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p, parse_dates=["entry_time"])
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_upcoming_news() -> list:
    now = datetime.now(timezone.utc)
    events = get_high_impact_events(now, now + timedelta(days=3))
    return [e for e in events if pd.Timestamp(e) >= pd.Timestamp(now)]


# ── Stats helpers ─────────────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame, r_col="r_multiple", outcome_col="outcome") -> dict:
    if df.empty or outcome_col not in df.columns:
        return {}
    closed = df[df[outcome_col].isin(["win","loss"])]
    if closed.empty:
        return {}
    r    = closed[r_col].dropna() if r_col in closed.columns else pd.Series(dtype=float)
    wins = (closed[outcome_col] == "win").sum()
    n    = len(closed)
    pf   = r[r>0].sum() / abs(r[r<=0].sum()) if len(r[r<=0]) > 0 and abs(r[r<=0].sum()) > 0 else float("inf")
    # Sharpe-like (mean/std of R)
    sharpe = float(r.mean() / r.std()) * np.sqrt(252) if r.std() > 0 else 0
    # Max drawdown on cumulative R
    cum_r  = r.cumsum()
    peak   = cum_r.cummax()
    dd_r   = (cum_r - peak)
    max_dd_r = float(dd_r.min()) if len(dd_r) > 0 else 0
    return {
        "n_trades":      n,
        "win_rate":      wins / n * 100,
        "ev_r":          float(r.mean()) if len(r) else 0,
        "profit_factor": pf,
        "sharpe":        sharpe,
        "max_dd_r":      max_dd_r,
        "total_r":       float(r.sum()),
        "avg_win_r":     float(r[r>0].mean()) if len(r[r>0]) else 0,
        "avg_loss_r":    float(r[r<=0].mean()) if len(r[r<=0]) else 0,
    }


def current_streak(df: pd.DataFrame) -> tuple[int, str]:
    if df.empty or "outcome" not in df.columns:
        return 0, "—"
    outcomes = df.sort_values("entry_time")["outcome"].tolist()
    if not outcomes:
        return 0, "—"
    last = outcomes[-1]
    streak = 0
    for o in reversed(outcomes):
        if o == last:
            streak += 1
        else:
            break
    label = "win" if last == "win" else "loss"
    return streak, label


# ── Chart helpers ─────────────────────────────────────────────────────────────

def equity_curve_chart(eq_df: pd.DataFrame, bt_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not bt_df.empty and "entry_time" in bt_df.columns and "pnl" in bt_df.columns:
        bt_sorted = bt_df.sort_values("entry_time")
        bt_eq     = INITIAL_EQUITY + bt_sorted["pnl"].cumsum().values
        fig.add_trace(go.Scatter(
            x=bt_sorted["entry_time"], y=bt_eq,
            name="v3 Backtest (OOS)", line=dict(color="#546e7a", width=1, dash="dot"), opacity=0.7,
        ))
    if not eq_df.empty:
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=eq_df["equity"],
            name="Live Equity", line=dict(color="#ffd700", width=2.5),
            fill="tozeroy", fillcolor="rgba(255,215,0,0.05)",
        ))
        fig.add_hline(y=INITIAL_EQUITY, line_dash="dash", line_color="#546e7a",
                      annotation_text="Start £10k", annotation_position="right")
    fig.update_layout(
        title="Equity Curve — Live vs Backtest",
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12),
        xaxis_title="", yaxis_title="Equity (£)",
    )
    return fig


def cumulative_r_chart(bt_df: pd.DataFrame, live_trades: pd.DataFrame) -> go.Figure:
    """Cumulative R-multiple — backtest vs live on the same normalized trade-number axis."""
    fig = go.Figure()
    if not bt_df.empty and "r_multiple" in bt_df.columns and "outcome" in bt_df.columns:
        closed = bt_df[bt_df["outcome"].isin(["win","loss"])].sort_values("entry_time")
        cum_r  = closed["r_multiple"].cumsum().values
        fig.add_trace(go.Scatter(
            x=list(range(1, len(cum_r)+1)), y=cum_r,
            name="v3 Backtest", line=dict(color="#546e7a", width=1.5, dash="dot"), opacity=0.8,
        ))
    if not live_trades.empty and "r_multiple" in live_trades.columns:
        live_r = live_trades["r_multiple"].dropna().cumsum().values
        fig.add_trace(go.Scatter(
            x=list(range(1, len(live_r)+1)), y=live_r,
            name="Live", line=dict(color="#00e676", width=2.5),
        ))
    fig.add_hline(y=0, line_dash="solid", line_color="#546e7a", line_width=0.5)
    fig.update_layout(
        title="Cumulative R-Multiple (Backtest vs Live)",
        template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Trade #", yaxis_title="Cumulative R",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def drawdown_chart(eq_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not eq_df.empty and len(eq_df) > 1:
        eq   = eq_df["equity"].values
        peak = np.maximum.accumulate(eq)
        dd   = (eq - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=eq_df["timestamp"], y=dd,
            name="Drawdown %", line=dict(color="#ff5252", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,82,82,0.15)",
        ))
        fig.add_hline(y=-5.37, line_dash="dash", line_color="#ff8a65",
                      annotation_text="v3 MaxDD (−5.37%)", annotation_position="right")
        fig.add_hline(y=-DAILY_LOSS_LIMIT*100, line_dash="dot", line_color="#ff1744",
                      annotation_text=f"Daily limit ({-DAILY_LOSS_LIMIT*100:.0f}%)", annotation_position="right")
    fig.update_layout(
        title="Live Drawdown",
        template="plotly_dark", height=220,
        margin=dict(l=0, r=0, t=36, b=0),
        yaxis_title="%", xaxis_title="",
    )
    return fig


def rolling_metrics_chart(bt_df: pd.DataFrame, window: int = 20) -> go.Figure:
    """Rolling win rate and EV/R on dual-axis."""
    if bt_df.empty or "outcome" not in bt_df.columns:
        return go.Figure()
    df = bt_df[bt_df["outcome"].isin(["win","loss"])].sort_values("entry_time").copy()
    df["win"] = (df["outcome"] == "win").astype(float)
    df["rolling_wr"] = df["win"].rolling(window).mean() * 100
    df["rolling_ev"] = df["r_multiple"].rolling(window).mean() if "r_multiple" in df.columns else 0

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(
        x=df["entry_time"], y=df["rolling_wr"],
        name=f"{window}-trade WR %", line=dict(color="#29b6f6", width=2),
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df["entry_time"], y=df["rolling_ev"],
        name=f"{window}-trade EV/R", line=dict(color="#ffd700", width=1.5, dash="dot"),
    ), secondary_y=True)
    fig.add_hline(y=39.4, line_dash="dash", line_color="#546e7a",
                  annotation_text="avg WR 39.4%", secondary_y=False)
    fig.add_hline(y=2.129, line_dash="dash", line_color="#9e9e9e",
                  annotation_text="avg EV +2.13R", secondary_y=True)
    fig.update_layout(
        title=f"Rolling {window}-Trade Win Rate & EV/R",
        template="plotly_dark", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Win Rate %", secondary_y=False)
    fig.update_yaxes(title_text="EV/R", secondary_y=True)
    return fig


def rolling_sharpe_chart(bt_df: pd.DataFrame, window: int = 30) -> go.Figure:
    if bt_df.empty or "r_multiple" not in bt_df.columns or "outcome" not in bt_df.columns:
        return go.Figure()
    df = bt_df[bt_df["outcome"].isin(["win","loss"])].sort_values("entry_time").copy()
    r  = df["r_multiple"].dropna()
    roll_sharpe = r.rolling(window).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["entry_time"], y=roll_sharpe.values,
        name=f"{window}-trade Rolling Sharpe", line=dict(color="#ab47bc", width=2),
    ))
    fig.add_hline(y=0, line_color="#546e7a", line_width=0.5)
    fig.add_hline(y=1.0, line_dash="dash", line_color="#00e676",
                  annotation_text="Sharpe=1.0", annotation_position="right")
    fig.update_layout(
        title=f"Rolling {window}-Trade Sharpe Ratio",
        template="plotly_dark", height=240,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Sharpe", xaxis_title="",
    )
    return fig


def monthly_pnl_heatmap(bt_df: pd.DataFrame, live_trades: pd.DataFrame) -> go.Figure:
    """Monthly PnL heatmap — backtest + live merged."""
    frames = []
    if not bt_df.empty and "entry_time" in bt_df.columns and "pnl" in bt_df.columns:
        tmp = bt_df[["entry_time","pnl"]].copy()
        tmp["source"] = "backtest"
        frames.append(tmp)
    if not live_trades.empty and "entry_time" in live_trades.columns and "pnl" in live_trades.columns:
        tmp = live_trades[["entry_time","pnl"]].copy()
        tmp["source"] = "live"
        frames.append(tmp)
    if not frames:
        return go.Figure()
    df = pd.concat(frames, ignore_index=True)
    df["year"]  = df["entry_time"].dt.year
    df["month"] = df["entry_time"].dt.month
    pivot = df.groupby(["year","month"])["pnl"].sum().unstack(fill_value=0)
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[months[m-1] for m in pivot.columns],
        y=[str(y) for y in pivot.index],
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="PnL (£)"),
        hovertemplate="Year: %{y}<br>Month: %{x}<br>PnL: £%{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Monthly PnL Heatmap (Backtest + Live)",
        template="plotly_dark", height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def return_distribution_chart(bt_df: pd.DataFrame, live_trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not bt_df.empty and "r_multiple" in bt_df.columns and "outcome" in bt_df.columns:
        r = bt_df[bt_df["outcome"].isin(["win","loss"])]["r_multiple"].dropna()
        fig.add_trace(go.Histogram(x=r[r<=0], name="BT Loss", marker_color="#ff5252", opacity=0.5, nbinsx=30))
        fig.add_trace(go.Histogram(x=r[r>0],  name="BT Win",  marker_color="#29b6f6", opacity=0.5, nbinsx=30))
        fig.add_vline(x=float(r.mean()), line_dash="dash", line_color="#546e7a",
                      annotation_text=f"BT EV={r.mean():.2f}R")
    if not live_trades.empty and "r_multiple" in live_trades.columns:
        r2 = live_trades["r_multiple"].dropna()
        if len(r2) > 0:
            fig.add_trace(go.Histogram(x=r2, name="Live", marker_color="#ffd700", opacity=0.7, nbinsx=20))
            fig.add_vline(x=float(r2.mean()), line_dash="dot", line_color="#ffd700",
                          annotation_text=f"Live EV={r2.mean():.2f}R")
    fig.update_layout(
        title="R-Multiple Distribution — Backtest vs Live",
        template="plotly_dark", barmode="overlay", height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="R Multiple", yaxis_title="Count",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def autocorrelation_chart(hist_df: pd.DataFrame, lags: int = 48) -> go.Figure:
    """ACF of 5-minute returns — shows whether there is serial autocorrelation (edge persistence)."""
    if hist_df.empty:
        return go.Figure()
    ret = hist_df["close"].pct_change().dropna()
    acf_vals = [ret.autocorr(lag=i) for i in range(1, lags + 1)]
    ci = 1.96 / np.sqrt(len(ret))
    fig = go.Figure()
    colors = ["#ff5252" if abs(v) > ci else "#546e7a" for v in acf_vals]
    fig.add_trace(go.Bar(
        x=list(range(1, lags + 1)), y=acf_vals,
        marker_color=colors, name="ACF",
    ))
    fig.add_hline(y=ci,  line_dash="dash", line_color="#ffd700", line_width=1,
                  annotation_text="95% CI", annotation_position="right")
    fig.add_hline(y=-ci, line_dash="dash", line_color="#ffd700", line_width=1)
    fig.add_hline(y=0,   line_color="#546e7a", line_width=0.5)
    fig.update_layout(
        title=f"Return Autocorrelation (ACF) — Gold 5m, {lags} lags",
        template="plotly_dark", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Lag (5m bars)", yaxis_title="Correlation",
        annotations=[dict(
            x=0.01, y=0.95, xref="paper", yref="paper",
            text="Red = statistically significant", showarrow=False,
            font=dict(color="#ff5252", size=11),
        )],
    )
    return fig


def yearly_performance_chart(bt_df: pd.DataFrame) -> go.Figure:
    """Yearly PnL bar chart — in-sample and OOS side by side."""
    if bt_df.empty or "entry_time" not in bt_df.columns or "pnl" not in bt_df.columns:
        return go.Figure()
    df = bt_df.copy()
    df["year"] = pd.to_datetime(df["entry_time"]).dt.year
    grp = df.groupby("year").agg(
        pnl=("pnl","sum"),
        trades=("pnl","count"),
        wr=("outcome", lambda x: (x=="win").mean()*100) if "outcome" in df.columns else ("pnl","count"),
    ).reset_index()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grp["year"].astype(str), y=grp["pnl"],
        name="Annual PnL (£)",
        marker_color=["#00e676" if v >= 0 else "#ff5252" for v in grp["pnl"]],
        text=[f"£{v:,.0f}" for v in grp["pnl"]], textposition="outside",
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grp["year"].astype(str), y=grp["trades"],
        name="# Trades", mode="markers+lines",
        marker=dict(color="#ffd700", size=8), line=dict(dash="dot"),
    ), secondary_y=True)
    fig.update_layout(
        title="Yearly Performance — v3 OOS Backtest",
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="PnL (£)", secondary_y=False)
    fig.update_yaxes(title_text="# Trades", secondary_y=True)
    return fig


def intraday_return_profile(hist_df: pd.DataFrame) -> go.Figure:
    """Average return by hour of day — bar chart showing which hours have edge."""
    if hist_df.empty:
        return go.Figure()
    df = hist_df.copy()
    df["return"] = df["close"].pct_change() * 100
    df["hour"]   = df.index.hour
    grp = df.groupby("hour")["return"].agg(["mean","std","count"]).reset_index()
    grp.columns = ["hour","mean","std","count"]
    # t-statistic: mean / (std / sqrt(n)) — shows significance
    grp["tstat"] = grp["mean"] / (grp["std"] / np.sqrt(grp["count"]))
    ci_color = ["#00e676" if abs(t) > 1.96 else "#546e7a" for t in grp["tstat"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grp["hour"], y=grp["mean"],
        marker_color=ci_color, name="Avg Return %",
        hovertemplate="Hour: %{x}:00<br>Avg Return: %{y:.4f}%<extra></extra>",
    ))
    for h_start, h_end, label, col in [
        (7, 10, "London", "rgba(30,144,255,0.12)"),
        (12, 15, "NY",    "rgba(255,165,0,0.12)"),
    ]:
        fig.add_vrect(x0=h_start-0.5, x1=h_end-0.5,
                      fillcolor=col, layer="below", line_width=0,
                      annotation_text=label, annotation_position="top left")
    fig.add_hline(y=0, line_color="#546e7a", line_width=0.5)
    fig.update_layout(
        title="Intraday Return Profile (Avg % Return by Hour UTC) — Green = Statistically Significant",
        template="plotly_dark", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Hour (UTC)", yaxis_title="Avg Return %",
        xaxis=dict(tickmode="linear", tick0=0, dtick=1),
    )
    return fig


def win_rate_by_breakdown(bt_df: pd.DataFrame, col: str, title: str) -> go.Figure:
    """Bar chart of win rate + trade count grouped by a column."""
    if bt_df.empty or col not in bt_df.columns or "outcome" not in bt_df.columns:
        return go.Figure()
    df = bt_df[bt_df["outcome"].isin(["win","loss"])].copy()
    df["win"] = (df["outcome"] == "win").astype(int)
    grp = df.groupby(col).agg(
        n=("win","count"),
        wr=("win","mean"),
        ev=("r_multiple","mean") if "r_multiple" in df.columns else ("win","mean"),
    ).reset_index()
    grp["wr"] *= 100

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=grp[col].astype(str), y=grp["wr"],
        name="Win Rate %", marker_color="#29b6f6", opacity=0.8,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=grp[col].astype(str), y=grp["n"],
        name="# Trades", mode="markers+lines",
        marker=dict(color="#ffd700", size=8), line=dict(dash="dot"),
    ), secondary_y=True)
    fig.update_layout(
        title=title, template="plotly_dark", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=1.12),
    )
    fig.update_yaxes(title_text="Win Rate %", secondary_y=False)
    fig.update_yaxes(title_text="# Trades", secondary_y=True)
    return fig


def direction_pie(bt_df: pd.DataFrame) -> go.Figure:
    """Donut chart — long vs short win rate."""
    if bt_df.empty or "direction" not in bt_df.columns or "outcome" not in bt_df.columns:
        return go.Figure()
    df = bt_df[bt_df["outcome"].isin(["win","loss"])].copy()
    grp = df.groupby("direction")["outcome"].apply(lambda x: (x=="win").mean()*100).reset_index()
    grp.columns = ["direction","win_rate"]
    fig = go.Figure(go.Bar(
        x=grp["direction"], y=grp["win_rate"],
        marker_color=["#00e676" if d=="long" else "#ff5252" for d in grp["direction"]],
        text=[f"{v:.1f}%" for v in grp["win_rate"]], textposition="outside",
    ))
    fig.update_layout(
        title="Win Rate by Direction",
        template="plotly_dark", height=250,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Win Rate %", yaxis_range=[0, 100],
    )
    return fig


def trade_duration_chart(bt_df: pd.DataFrame) -> go.Figure:
    """Trade duration distribution if entry_time + exit_time exist."""
    if bt_df.empty:
        return go.Figure()
    cols = bt_df.columns.tolist()
    exit_col = next((c for c in ["exit_time","close_time"] if c in cols), None)
    if exit_col is None or "entry_time" not in cols:
        fig = go.Figure()
        fig.update_layout(
            title="Trade Duration (no exit_time column)",
            template="plotly_dark", height=240,
            margin=dict(l=0, r=0, t=40, b=0),
        )
        return fig
    df = bt_df.dropna(subset=["entry_time", exit_col]).copy()
    df["duration_h"] = (pd.to_datetime(df[exit_col]) - pd.to_datetime(df["entry_time"])).dt.total_seconds() / 3600
    df = df[df["duration_h"] > 0]
    wins   = df[df["outcome"]=="win"]["duration_h"]  if "outcome" in df.columns else pd.Series()
    losses = df[df["outcome"]=="loss"]["duration_h"] if "outcome" in df.columns else df["duration_h"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=losses, name="Losses", marker_color="#ff5252", opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=wins,   name="Wins",   marker_color="#00e676", opacity=0.7, nbinsx=30))
    fig.update_layout(
        title="Trade Duration Distribution (hours)",
        template="plotly_dark", barmode="overlay", height=240,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="Duration (hrs)", yaxis_title="Count",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def pnl_by_weekday(bt_df: pd.DataFrame) -> go.Figure:
    if bt_df.empty or "entry_time" not in bt_df.columns or "pnl" not in bt_df.columns:
        return go.Figure()
    df = bt_df[bt_df["outcome"].isin(["win","loss"])].copy() if "outcome" in bt_df.columns else bt_df.copy()
    df["weekday"] = pd.to_datetime(df["entry_time"]).dt.day_name()
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
    grp = df.groupby("weekday")["pnl"].sum().reindex(order, fill_value=0)
    fig = go.Figure(go.Bar(
        x=grp.index, y=grp.values,
        marker_color=["#00e676" if v >= 0 else "#ff5252" for v in grp.values],
        text=[f"£{v:,.0f}" for v in grp.values], textposition="outside",
    ))
    fig.update_layout(
        title="Total PnL by Day of Week (Backtest OOS)",
        template="plotly_dark", height=260,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Total PnL (£)", xaxis_title="",
    )
    return fig


def hourly_heatmap(hist_df: pd.DataFrame) -> go.Figure:
    if hist_df.empty:
        return go.Figure()
    df = hist_df.copy()
    df["return"]  = df["close"].pct_change() * 100
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.dayofweek
    pivot = df.groupby(["weekday","hour"])["return"].mean().unstack(fill_value=0)
    days  = ["Mon","Tue","Wed","Thu","Fri"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[days[i] for i in pivot.index if i < 5],
        colorscale="RdYlGn", zmid=0,
        colorbar=dict(title="Avg Return %"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg Return: %{z:.3f}%<extra></extra>",
    ))
    for h_start, h_end, label, col in [
        (7, 10, "London", "rgba(30,144,255,0.15)"),
        (12, 15, "NY",    "rgba(255,165,0,0.15)"),
    ]:
        fig.add_vrect(x0=f"{h_start:02d}:00", x1=f"{h_end:02d}:00",
                      fillcolor=col, layer="below", line_width=0,
                      annotation_text=label, annotation_position="top left")
    fig.update_layout(
        title="Weekday × Hour Return Heatmap (Gold 5m, 2016–2026)",
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def volume_heatmap(hist_df: pd.DataFrame) -> go.Figure:
    if hist_df.empty:
        return go.Figure()
    df = hist_df.copy()
    df["hour"]    = df.index.hour
    df["weekday"] = df.index.dayofweek
    pivot = df.groupby(["weekday","hour"])["volume"].mean().unstack(fill_value=0)
    days  = ["Mon","Tue","Wed","Thu","Fri"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=[days[i] for i in pivot.index if i < 5],
        colorscale="Blues",
        colorbar=dict(title="Avg Volume"),
        hovertemplate="Day: %{y}<br>Hour: %{x}<br>Avg Vol: %{z:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        title="Weekday × Hour Volume Heatmap",
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def signal_scatter(signals_df: pd.DataFrame) -> go.Figure:
    if signals_df.empty:
        return go.Figure()
    df = signals_df.copy().head(100)
    colours = {"long": "#00e676", "short": "#ff5252"}
    symbols = {"long": "triangle-up", "short": "triangle-down"}
    fig = go.Figure()
    for direction, grp in df.groupby("direction"):
        fig.add_trace(go.Scatter(
            x=grp["timestamp"], y=grp["entry"],
            mode="markers",
            marker=dict(
                color=colours.get(direction, "#aaa"),
                size=grp["confidence"].clip(1,5) * 4 if "confidence" in grp.columns else 8,
                symbol=symbols.get(direction, "circle"),
                line=dict(width=1, color="#fff"),
            ),
            name=direction.capitalize(),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Entry: %{y:.2f}<br>"
                "RR: %{customdata[1]:.1f}<br>"
                "Conf: %{customdata[2]}<br>"
                "Zone: %{customdata[3]}<extra></extra>"
            ),
            customdata=grp[["direction","rr","confidence","zone_type"]].values
            if all(c in grp.columns for c in ["rr","confidence","zone_type"])
            else grp[["direction"]].values,
        ))
    fig.update_layout(
        title="Signal History (size = confidence, ▲ long / ▼ short)",
        template="plotly_dark", height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        xaxis_title="", yaxis_title="Entry Price",
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def open_position_chart(signals_df: pd.DataFrame, current_price: float) -> go.Figure:
    """Latest signal — show entry / SL / TP levels as horizontal lines."""
    if signals_df.empty:
        return go.Figure()
    latest = signals_df.iloc[0]
    fig = go.Figure()
    entry  = float(latest.get("entry", 0))
    sl     = float(latest.get("stop",  0))
    tp     = float(latest.get("target",0))
    direction = str(latest.get("direction",""))

    levels = [
        (tp,    "#00e676", "Take Profit"),
        (entry, "#ffd700", "Entry"),
        (sl,    "#ff5252", "Stop Loss"),
    ]
    for price, color, label in levels:
        fig.add_hline(y=price, line_color=color, line_dash="dash",
                      annotation_text=f"{label} {price:.2f}",
                      annotation_position="right")
    if current_price > 0:
        fig.add_hline(y=current_price, line_color="#ffffff", line_width=1,
                      annotation_text=f"Current {current_price:.2f}",
                      annotation_position="left")
    # shade zone between entry and TP
    if entry and tp:
        fig.add_hrect(
            y0=min(entry,tp), y1=max(entry,tp),
            fillcolor="rgba(0,230,118,0.05)", line_width=0,
        )
    if entry and sl:
        fig.add_hrect(
            y0=min(entry,sl), y1=max(entry,sl),
            fillcolor="rgba(255,82,82,0.05)", line_width=0,
        )
    rr  = float(latest.get("rr", 0))
    conf= latest.get("confidence", "—")
    zt  = latest.get("zone_type", "—")
    fig.update_layout(
        title=f"Latest Signal — {direction.upper()}  |  RR {rr:.1f}  |  Conf {conf}  |  {zt}",
        template="plotly_dark", height=280,
        margin=dict(l=0, r=0, t=40, b=0),
        yaxis_title="Price", xaxis_visible=False,
        yaxis=dict(range=[min(sl,tp)*0.999, max(sl,tp)*1.001] if sl and tp else None),
    )
    return fig


# ── Main dashboard layout ─────────────────────────────────────────────────────

def main():
    st.title("📊 Quantitative Gold Trading System — Live Dashboard")
    st.caption(f"Strategy: Quantitative Gold Trading System (Rolling WFV · Fixed Risk · ML Filter) · {SYMBOL} · Auto-refresh {DASHBOARD_REFRESH_SEC}s")

    st.markdown(
        f'<meta http-equiv="refresh" content="{DASHBOARD_REFRESH_SEC}">',
        unsafe_allow_html=True,
    )

    # ── Load all data ─────────────────────────────────────────────────────────
    status     = load_status()
    eq_df      = load_equity()
    signals    = load_signals()
    trades     = load_trades()
    hist_df    = load_historical_bars()
    bt_trades  = load_backtest_trades()
    news       = load_upcoming_news()

    now_utc    = datetime.now(timezone.utc)
    in_session = 7 <= now_utc.hour < 10 or 12 <= now_utc.hour < 15
    equity     = float(status.get("equity", INITIAL_EQUITY))
    pnl_total  = equity - INITIAL_EQUITY
    pnl_pct    = pnl_total / INITIAL_EQUITY * 100
    spread     = float(status.get("spread", 0))
    bt_stats   = compute_stats(bt_trades)

    # Derive live closed trades from trades file
    live_closed = pd.DataFrame()
    if not trades.empty and "status_close" in trades.columns:
        live_closed = trades[trades["status_close"].isin(["WIN","LOSS"])].copy()
        if "r_multiple" not in live_closed.columns and "r_multiple_close" in live_closed.columns:
            live_closed["r_multiple"] = live_closed["r_multiple_close"]
        if "outcome" not in live_closed.columns and "status_close" in live_closed.columns:
            live_closed["outcome"] = live_closed["status_close"].str.lower()

    live_stats = compute_stats(live_closed) if not live_closed.empty else {}
    streak_n, streak_lbl = current_streak(live_closed) if not live_closed.empty else (0, "—")

    # ── Top status bar ────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("EA Status",  status.get("state", "OFFLINE"),
              delta="LIVE" if status.get("state") == "RUNNING" else None)
    c2.metric("Equity",     f"£{equity:,.2f}",
              delta=f"£{pnl_total:+,.2f} ({pnl_pct:+.2f}%)")
    c3.metric("Session",    "🟢 ACTIVE" if in_session else "⚫ CLOSED")
    c4.metric("Spread",     f"{spread:.2f}")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Signals",     len(signals) if not signals.empty else 0)
    d2.metric("Live Trades", len(live_closed) if not live_closed.empty else 0)
    d3.metric("Streak",      f"{streak_n}× {streak_lbl}" if streak_n > 0 else "—",
              delta=f"{'🟢' if streak_lbl=='win' else '🔴'}" if streak_n > 0 else None)
    d4.metric("Last Update", str(status.get("timestamp", "—"))[-8:])

    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_overview, tab_perf, tab_analytics, tab_signals, tab_market = st.tabs([
        "📈 Overview", "⚖️ Performance", "🔬 Analytics", "🎯 Signals", "🌍 Market"
    ])

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────────────────────────────────────────
    with tab_overview:
        col_main, col_side = st.columns([3, 1])

        with col_main:
            st.plotly_chart(equity_curve_chart(eq_df, bt_trades), use_container_width=True, key="ov_equity")
            st.plotly_chart(drawdown_chart(eq_df), use_container_width=True, key="ov_drawdown")

        with col_side:
            # Live vs Backtest stats table
            st.subheader("Live vs Backtest")
            def _fmt(val, fmt=".2f", suffix=""):
                return f"{val:{fmt}}{suffix}" if val else "—"
            st.markdown(f"""
| Metric | Backtest | Live |
|--------|----------|------|
| Trades | 622 | {live_stats.get('n_trades', '—')} |
| Win Rate | 39.4% | {_fmt(live_stats.get('win_rate'), '.1f', '%')} |
| EV/R | +2.129R | {_fmt(live_stats.get('ev_r'), '+.3f', 'R')} |
| Profit Factor | 4.74 | {_fmt(live_stats.get('profit_factor'), '.2f')} |
| Sharpe | — | {_fmt(live_stats.get('sharpe'), '.2f')} |
| Max DD (R) | — | {_fmt(live_stats.get('max_dd_r'), '.2f', 'R')} |
""")
            st.divider()

            # Upcoming news
            st.subheader("Upcoming News 🔴")
            if news:
                for ev in news[:5]:
                    ts    = pd.Timestamp(ev)
                    delta = ts - pd.Timestamp(now_utc)
                    hrs   = int(delta.total_seconds() / 3600)
                    st.markdown(f"🔴 `{ts.strftime('%a %d %b %H:%M')} UTC`  ({hrs}h)")
            else:
                st.caption("No high-impact events in next 3 days.")

            st.divider()
            st.subheader("Kill Zones (UTC)")
            st.markdown(f"""
| Session | Window | |
|---------|--------|--|
| London | 07:00–10:00 | {"🟢" if 7<=now_utc.hour<10 else "⚫"} |
| NY | 12:00–15:00 | {"🟢" if 12<=now_utc.hour<15 else "⚫"} |
""")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 2 — PERFORMANCE
    # ─────────────────────────────────────────────────────────────────────────
    with tab_perf:
        p1, p2 = st.columns(2)
        with p1:
            st.plotly_chart(cumulative_r_chart(bt_trades, live_closed), use_container_width=True, key="pf_cumr")
            st.plotly_chart(rolling_sharpe_chart(bt_trades), use_container_width=True, key="pf_sharpe")
        with p2:
            st.plotly_chart(return_distribution_chart(bt_trades, live_closed), use_container_width=True, key="pf_rdist")
            st.plotly_chart(monthly_pnl_heatmap(bt_trades, live_closed), use_container_width=True, key="pf_monthly")

        st.divider()
        st.plotly_chart(rolling_metrics_chart(bt_trades), use_container_width=True, key="pf_rolling")

        # Key metrics comparison row
        st.subheader("Detailed Stats Comparison")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("BT Win Rate",      "39.4%",   delta=f"{live_stats.get('win_rate',0)-39.4:+.1f}% live" if live_stats else None)
        mc2.metric("BT EV/R",          "+2.129R",  delta=f"{live_stats.get('ev_r',0)-2.129:+.3f}R live" if live_stats else None)
        mc3.metric("BT Profit Factor", "4.74",     delta=f"{live_stats.get('profit_factor',0)-4.74:+.2f} live" if live_stats else None)
        mc4.metric("BT Max DD",        "−5.37%",   delta=None)

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 3 — ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_analytics:
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(win_rate_by_breakdown(bt_trades, "zone_type",   "Win Rate by Zone Type"), use_container_width=True, key="an_wr_zone")
            st.plotly_chart(win_rate_by_breakdown(bt_trades, "confidence",  "Win Rate by Confidence Level"), use_container_width=True, key="an_wr_conf")
        with a2:
            st.plotly_chart(direction_pie(bt_trades), use_container_width=True, key="an_direction")
            st.plotly_chart(pnl_by_weekday(bt_trades), use_container_width=True, key="an_weekday")

        st.divider()
        a3, a4 = st.columns(2)
        with a3:
            st.plotly_chart(trade_duration_chart(bt_trades), use_container_width=True, key="an_duration")
        with a4:
            if not bt_trades.empty and "zone_type" in bt_trades.columns and "r_multiple" in bt_trades.columns:
                df_zt = bt_trades[bt_trades["outcome"].isin(["win","loss"])].groupby("zone_type")["r_multiple"].mean().reset_index()
                df_zt.columns = ["zone_type","avg_r"]
                fig_zt = go.Figure(go.Bar(
                    x=df_zt["zone_type"], y=df_zt["avg_r"],
                    marker_color=["#00e676" if v >= 0 else "#ff5252" for v in df_zt["avg_r"]],
                    text=[f"{v:+.2f}R" for v in df_zt["avg_r"]], textposition="outside",
                ))
                fig_zt.update_layout(
                    title="Average R by Zone Type",
                    template="plotly_dark", height=240,
                    margin=dict(l=0, r=0, t=40, b=0),
                    yaxis_title="Avg R", xaxis_title="",
                )
                st.plotly_chart(fig_zt, use_container_width=True, key="an_zone_r")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 4 — SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    with tab_signals:
        s1, s2 = st.columns([2, 1])
        with s1:
            st.plotly_chart(signal_scatter(signals), use_container_width=True, key="sig_scatter")
            st.subheader("Signal Log")
            if not signals.empty:
                display_cols = ["id","timestamp","direction","entry","stop","target","rr","confidence","zone_type","status"]
                show_cols    = [c for c in display_cols if c in signals.columns]
                st.dataframe(
                    signals[show_cols].head(50).style.map(
                        lambda v: "color: #00e676" if v=="long" else ("color: #ff5252" if v=="short" else ""),
                        subset=["direction"] if "direction" in show_cols else []
                    ),
                    use_container_width=True, height=320,
                )
            else:
                st.info("No signals yet — scanning starts at London open (07:00 UTC).")

        with s2:
            # Latest signal panel
            st.subheader("Latest Signal")
            if not signals.empty:
                latest = signals.iloc[0]
                direction = str(latest.get("direction",""))
                color     = "#00e676" if direction=="long" else "#ff5252"
                st.markdown(f"<h3 style='color:{color}'>{direction.upper()}</h3>", unsafe_allow_html=True)
                st.metric("Entry",  f"{float(latest.get('entry',0)):.2f}")
                st.metric("Stop",   f"{float(latest.get('stop',0)):.2f}")
                st.metric("Target", f"{float(latest.get('target',0)):.2f}")
                st.metric("RR",     f"{float(latest.get('rr',0)):.1f}R")
                st.metric("Confidence", str(latest.get("confidence","—")))
                st.metric("Zone",   str(latest.get("zone_type","—")))
                st.metric("Time",   str(latest.get("timestamp","—")))
                st.plotly_chart(
                    open_position_chart(signals, float(status.get("equity", 0))),
                    use_container_width=True, key="sig_levels",
                )
            else:
                st.info("No signals generated yet.")

            st.subheader("Live Trade Results")
            if not trades.empty:
                st.dataframe(trades.tail(15), use_container_width=True, height=250)
            else:
                st.info("No live trades yet.")

    # ─────────────────────────────────────────────────────────────────────────
    # TAB 5 — MARKET
    # ─────────────────────────────────────────────────────────────────────────
    with tab_market:
        m1, m2 = st.columns(2)
        with m1:
            st.plotly_chart(hourly_heatmap(hist_df), use_container_width=True, key="mkt_hourly")
        with m2:
            st.plotly_chart(volume_heatmap(hist_df), use_container_width=True, key="mkt_volume")

        st.divider()
        st.plotly_chart(intraday_return_profile(hist_df), use_container_width=True, key="mkt_intraday")

        st.divider()
        st.plotly_chart(autocorrelation_chart(hist_df), use_container_width=True, key="mkt_acf")

        st.divider()
        mk1, mk2 = st.columns(2)
        with mk1:
            st.plotly_chart(yearly_performance_chart(bt_trades), use_container_width=True, key="mkt_yearly")
        with mk2:
            st.plotly_chart(pnl_by_weekday(bt_trades), use_container_width=True, key="mkt_weekday")

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("QGTS")
        st.markdown(f"""
**Strategy:** Rolling WFV + ML
**Symbol:** `{SYMBOL}`
**Risk:** {DEFAULT_RISK_PCT*100:.1f}% / {HIGH_CONF_RISK*100:.1f}% conf≥4
**Min RR:** {MIN_RR}
**Zone LB:** {ZONE_LOOKBACK}
**ML Threshold:** {ML_THRESHOLD}
**Sessions:** London · NY
""")
        st.divider()
        st.markdown("**Bridge paths**")
        st.code(str(BRIDGE_DIR), language=None)
        ea_ok  = BARS_5M_FILE.exists()
        sig_ok = SIGNAL_FILE.exists()
        st.markdown(f"MT4 data feed: {'✅' if ea_ok  else '❌ waiting'}")
        st.markdown(f"Signal file:   {'✅' if sig_ok else '❌ not yet'}")
        st.divider()
        st.markdown("**v3 Backtest Summary**")
        st.markdown("""
| | |
|---|---|
| OOS span | 2020–2026 |
| Trades | 622 |
| Win rate | 39.4% |
| EV/R | +2.129R |
| Profit Factor | 4.74 |
| Max DD | −5.37% |
| Barrier | 100% |
""")
        if st.button("🔄 Clear cache"):
            st.cache_data.clear()
            st.rerun()


if __name__ == "__main__":
    main()
