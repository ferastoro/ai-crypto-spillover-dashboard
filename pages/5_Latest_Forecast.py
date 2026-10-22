import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
from sklearn.metrics import mean_squared_error

from src.data_loader import (
    load_latest_forecasts,
    load_live_scorecard,
    load_btcidr_history,
)
from src.config import COLORS

st.set_page_config(
    page_title="Latest HAR Forecast | RSIC Dashboard",
    page_icon="⚡",
    layout="wide"
)

# Custom card styling
st.markdown("""
<style>
    .forecast-card {
        background-color: #1E222D;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 12px;
    }
    .card-title {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }
    .card-main-val {
        color: #10B981;
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .card-sub-val {
        color: #F8FAFC;
        font-size: 1.1rem;
        font-weight: 500;
        margin-top: 4px;
    }
    .card-meta {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 8px;
    }
    .disclaimer-banner {
        background-color: rgba(239, 68, 68, 0.08);
        border-left: 4px solid #EF4444;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
        color: #FCA5A5;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Latest HAR Volatility Forecast (Live Tracking)")

st.markdown("""
Daily automated out-of-sample volatility forecast for **BTC-IDR (Indodax)** using the econometric 
Heterogeneous Autoregressive (**HAR**) model. Updated daily at 01:00 UTC once the full daily candle has completed.
""")

# Prominent Notice & Disclaimer
st.markdown("""
<div class="disclaimer-banner">
    <b>Important Operational Notice:</b> HAR only. Live forecasts have no validated out-of-sample record yet. Not investment advice.
</div>
""", unsafe_allow_html=True)

# Load data
df_fc = load_latest_forecasts()
df_hist = load_btcidr_history()
df_scorecard = load_live_scorecard()

if df_fc.empty or df_hist.empty:
    st.error("Live forecast data has not been generated yet. Please run `live/update_forecast.py`.")
    st.stop()

# 1. Check Data Freshness & Stale Alert
latest_as_of = df_fc["as_of_date"].max()
now_utc = datetime.now(timezone.utc)
days_stale = (now_utc.date() - latest_as_of.date()).days

if days_stale > 3:
    st.error(f"⚠️ **Stale Data Warning:** Latest completed candle is from `{latest_as_of.strftime('%Y-%m-%d')}` ({days_stale} days ago). The automated daily update may have encountered an API issue or pause.")

# 2. Extract Latest Forecast Cards (h=1 and h=10)
sub_latest = df_fc[df_fc["as_of_date"] == latest_as_of].copy()
fc_h1 = sub_latest[sub_latest["horizon"] == 1]
fc_h10 = sub_latest[sub_latest["horizon"] == 10]

col_h1, col_h10 = st.columns(2)

with col_h1:
    if not fc_h1.empty:
        var_1 = float(fc_h1["forecast"].values[0])
        vol_pct_1 = np.sqrt(max(0, var_1)) * 100.0
        tgt_date_1 = fc_h1["target_date"].iloc[0].strftime("%b %d, %Y")
        gen_time = fc_h1["generated_at_utc"].iloc[0]
        st.markdown(f"""
        <div class="forecast-card">
            <div class="card-title">Latest 1-Day Ahead Forecast (h = 1)</div>
            <div class="card-main-val">±{vol_pct_1:.2f}% <span style="font-size: 0.85rem; color: #94A3B8;">expected daily move size (std dev)</span></div>
            <div class="card-sub-val">Variance: {var_1:.4e}</div>
            <div class="card-meta">
                <b>Target Date:</b> {tgt_date_1}<br>
                <b>As-of Date:</b> {latest_as_of.strftime('%b %d, %Y')} | <b>Updated:</b> {gen_time}
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_h10:
    if not fc_h10.empty:
        var_10 = float(fc_h10["forecast"].values[0])
        vol_pct_10 = np.sqrt(max(0, var_10)) * 100.0
        tgt_date_10 = fc_h10["target_date"].iloc[0].strftime("%b %d, %Y")
        gen_time = fc_h10["generated_at_utc"].iloc[0]
        st.markdown(f"""
        <div class="forecast-card">
            <div class="card-title">Latest 10-Days Ahead Forecast (h = 10)</div>
            <div class="card-main-val">±{vol_pct_10:.2f}% <span style="font-size: 0.85rem; color: #94A3B8;">expected daily move size (std dev)</span></div>
            <div class="card-sub-val">Variance: {var_10:.4e}</div>
            <div class="card-meta">
                <b>Target Date:</b> {tgt_date_10}<br>
                <b>As-of Date:</b> {latest_as_of.strftime('%b %d, %Y')} | <b>Updated:</b> {gen_time}
            </div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# 3. Chart: 90-Day Realized Variance + Forecast Overlays
st.subheader("Realized Volatility & Forecast Trajectory (Last 90 Days)")

# Filter last 90 calendar days of realized history
cutoff_90d = pd.to_datetime((now_utc - timedelta(days=90)).date())
df_hist_90 = df_hist[df_hist["Date"] >= cutoff_90d].sort_values("Date").copy()

# Filter forecasts within the range
df_fc_recent = df_fc[df_fc["as_of_date"] >= cutoff_90d].copy()

fig = go.Figure()

# Realized variance line
fig.add_trace(go.Scatter(
    x=df_hist_90["Date"],
    y=df_hist_90["r2"],
    mode="lines",
    name="Realized Variance (r²)",
    line=dict(color=COLORS["y_true"], width=2),
    hovertemplate="<b>Realized r²</b>: %{y:.6e}<extra></extra>"
))

# Backfill forecasts (h=1)
bf_h1 = df_fc_recent[(df_fc_recent["kind"] == "backfill") & (df_fc_recent["horizon"] == 1)]
if not bf_h1.empty:
    fig.add_trace(go.Scatter(
        x=bf_h1["target_date"],
        y=bf_h1["forecast"],
        mode="markers",
        name="Backfill Forecast (h=1)",
        marker=dict(color="#6366F1", size=6, symbol="circle"),
        hovertemplate="<b>Backfill h=1</b>: %{y:.6e}<extra></extra>"
    ))

# Live forecasts (h=1 and h=10)
live_fc = df_fc_recent[df_fc["kind"] == "live"]
if not live_fc.empty:
    fig.add_trace(go.Scatter(
        x=live_fc["target_date"],
        y=live_fc["forecast"],
        mode="markers+text",
        name="Live Forecast Point",
        marker=dict(color="#10B981", size=12, symbol="star"),
        text=[f"h={h}" for h in live_fc["horizon"]],
        textposition="top center",
        textfont=dict(color="#10B981", size=11),
        hovertemplate="<b>Live Forecast</b>: %{y:.6e} (Target: %{x})<extra></extra>"
    ))

fig.update_layout(
    template="plotly_dark",
    title=dict(text="<b>BTC-IDR Realized Variance vs Out-of-Sample HAR Predictions</b>", font=dict(size=15, color="#F8FAFC")),
    paper_bgcolor="rgba(15, 23, 42, 0.6)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    xaxis=dict(title="Date", gridcolor="rgba(255, 255, 255, 0.07)"),
    yaxis=dict(title="Log-Variance Proxy (r²)", gridcolor="rgba(255, 255, 255, 0.07)"),
    legend=dict(bgcolor="rgba(15, 23, 42, 0.8)", bordercolor="rgba(255, 255, 255, 0.1)"),
    margin=dict(l=50, r=30, t=50, b=40),
    hovermode="closest"
)

st.plotly_chart(fig, width="stretch")

st.divider()

# 4. Live Scorecard Panel
st.subheader("Live Out-of-Sample Scorecard")
st.markdown("""
Statistical tracking of real-time out-of-sample prediction error. As live forecast target dates are reached, 
realized returns are recorded to benchmark HAR against Historical Expansion and Rolling Mean baselines.
""")

live_rows = df_fc[df_fc["kind"] == "live"]
n_live_realized = int(live_rows["realized"].notna().sum())

col_sc1, col_sc2 = st.columns([1, 2])

with col_sc1:
    st.metric("Live Forecasts Realized", f"{n_live_realized} observations")
    if n_live_realized < 30:
        st.info("ℹ️ **Status**: Too few observations to evaluate (< 30 observations). Statistical significance metrics will activate once 30 live predictions have realized.")

with col_sc2:
    if not df_scorecard.empty:
        st.dataframe(
            df_scorecard.style.format({
                "mse_har": "{:.4e}",
                "mse_hist_exp": "{:.4e}",
                "mse_roll_20": "{:.4e}"
            }),
            width="stretch",
            hide_index=True
        )

# Historical Backfill Track Record Context
with st.expander("View Retrospective Backfill Performance (Gap Period: Sep 2026 – Present)"):
    bf_eval = df_fc[(df_fc["kind"] == "backfill") & df_fc["realized"].notna()]
    if not bf_eval.empty:
        bf_metrics = []
        for h_val in [1, 10]:
            sub = bf_eval[bf_eval["horizon"] == h_val]
            if len(sub) > 0:
                mse_bf = mean_squared_error(sub["realized"], sub["forecast"])
                mae_bf = np.mean(np.abs(sub["realized"] - sub["forecast"]))
                bf_metrics.append({
                    "Horizon": f"h = {h_val}",
                    "N Realized": len(sub),
                    "MSE (×10⁻⁶)": mse_bf * 1e6,
                    "MAE (×10⁻⁴)": mae_bf * 1e4
                })
        st.dataframe(pd.DataFrame(bf_metrics), width="stretch", hide_index=True)
