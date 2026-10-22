import streamlit as st
import pandas as pd
from src.data_loader import (
    load_tvp_features,
    load_rq1_table,
    load_metrics_summary,
    load_forecasts,
)

st.set_page_config(
    page_title="AI-Crypto Spillover Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .metric-label {
        color: #94A3B8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .metric-value {
        color: #F8FAFC;
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-sub {
        color: #64748B;
        font-size: 0.8rem;
        margin-top: 4px;
    }
    .disclaimer-box {
        background-color: rgba(245, 158, 11, 0.08);
        border-left: 4px solid #F59E0B;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 20px 0;
        color: #E2E8F0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Description
st.title("AI Tech Stocks & Crypto Volatility Spillover Dashboard")

st.markdown("""
This dashboard presents empirical results investigating volatility transmission from major US Artificial Intelligence 
technology stocks (NVDA, MSFT, TSLA, GOOGL, META) to cryptocurrency markets, focusing on BTC-IDR traded on Indodax. 
It examines whether incorporating dynamic TVP-VAR causal spillover features enhances out-of-sample volatility forecasting 
accuracy across multi-horizon horizons ($h=1$ and $h=10$) using deep learning architectures (TFT, NBEATSx) compared to 
econometric and statistical baselines (HAR, GARCH, EGARCH, Historical Expansion, Rolling Mean).
""")

# Load core datasets
tvp_df = load_tvp_features()
rq1_df = load_rq1_table()
metrics_df = load_metrics_summary()
forecasts_df = load_forecasts()

# Calculate summary figures
avg_tsi = float(tvp_df["tvp_tsi"].mean())
start_date = tvp_df["Date"].min().strftime("%b %d, %Y")
end_date = tvp_df["Date"].max().strftime("%b %d, %Y")
num_obs = len(tvp_df)
num_cutoffs = forecasts_df["cutoff"].nunique()

best_model_h1 = metrics_df[(metrics_df["horizon"] == 1)].sort_values("mse").iloc[0]
best_model_name = best_model_h1["model"]
best_model_mse = best_model_h1["mse"]

# Metric Cards (4 Columns)
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Average Total Spillover</div>
        <div class="metric-value">{avg_tsi:.2f}%</div>
        <div class="metric-sub">Mean TVP-VAR TSI index across sample</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Sample Period</div>
        <div class="metric-value">{num_obs} Days</div>
        <div class="metric-sub">{start_date} – {end_date}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Rolling Evaluation Windows</div>
        <div class="metric-value">{num_cutoffs} Cutoffs</div>
        <div class="metric-sub">344 unique target dates (h=1 to h=10)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Lowest MSE (h=1)</div>
        <div class="metric-value">{best_model_name}</div>
        <div class="metric-sub">MSE = {best_model_mse:.4e} (Baseline)</div>
    </div>
    """, unsafe_allow_html=True)

# Academic Research Notice
st.markdown("""
<div class="disclaimer-box">
    <b>Research Output Notice:</b> All statistics, spillover indices, and model forecasts presented in this application are 
    academic research results derived from historical data. They do not constitute investment advice, trading signals, 
    or operational buy/sell recommendations.
</div>
""", unsafe_allow_html=True)

st.divider()

# Navigation Overview
st.subheader("Dashboard Navigation")
nav1, nav2 = st.columns(2)

with nav1:
    st.markdown("""
    - **[1. Spillover Analysis](pages/1_Spillover.py)**: Dynamic TVP-VAR Total Spillover Index (TSI) over time, average net spillover transmitters/receivers, and directional transmission to BTC-IDR.
    - **[2. Out-of-Sample Forecasts](pages/2_Forecast.py)**: Interactive multi-model time series predictions vs. realized volatility across $h=1$ and $h=10$ horizons with dynamic MSE/MAE recalculation for selected date ranges.
    """)

with nav2:
    st.markdown("""
    - **[3. Model Comparison](pages/3_Model_Comparison.py)**: Comparative MSE/MAE rankings across all deep learning architectures and baselines, accompanied by Diebold-Mariano test matrices and Bonferroni significance checks.
    - **[4. Empirical Limitations](pages/4_Limitations.py)**: Objective assessment of baseline persistence superiority, low prediction correlations, proxy sensitivity, and institutional data considerations.
    """)
