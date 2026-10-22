import streamlit as st
import pandas as pd
from src.data_loader import load_tvp_features, load_rq1_table
from src.charts import create_tsi_chart, create_net_spillover_bar
from src.config import INSTRUMENT_NAMES

st.set_page_config(
    page_title="Spillover Analysis | RSIC Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("Volatility Spillover Dynamics (TVP-VAR)")
st.markdown("""
This page examines the connectedness and transmission of volatility between US AI technology stocks 
(NVDA, MSFT, TSLA, GOOGL, META) and crypto markets (BTC-USD, ETH-USD, BTC-IDR), estimated using a 
Time-Varying Parameter Vector Autoregressive (TVP-VAR) model based on daily log realized variance proxies.
""")

# Load data
tvp_df = load_tvp_features()
rq1_df = load_rq1_table()

# Sidebar: Date range filter for dynamic features
st.sidebar.header("Date Range Filter")
min_date = tvp_df["Date"].min().to_pydatetime()
max_date = tvp_df["Date"].max().to_pydatetime()

selected_dates = st.sidebar.slider(
    "Select observation window:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

filtered_tvp = tvp_df[(tvp_df["Date"] >= pd.Timestamp(selected_dates[0])) & (tvp_df["Date"] <= pd.Timestamp(selected_dates[1]))]

# Section 1: Total Spillover Index (TSI) Line Chart
st.subheader("1. Dynamic Total Spillover Index (TSI)")
st.markdown("""
The Total Spillover Index measures the percentage of forecast error variance across all 8 system assets 
that is explained by cross-market spillovers rather than own shocks.
""")

fig_tsi = create_tsi_chart(filtered_tvp)
st.plotly_chart(fig_tsi, width="stretch")

# Regime Indicator Note
tsi_p80 = float(tvp_df["tvp_tsi"].quantile(0.80))
curr_avg = float(filtered_tvp["tvp_tsi"].mean())
st.caption(f"**Descriptive Note:** The 80th percentile threshold ({tsi_p80:.2f}%) represents a descriptive statistical boundary within this historical sample, not a validated trading or early-warning signal.")

st.divider()

# Section 2 & 3: Net Spillovers and Directional Transmission
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("2. Net Volatility Spillovers")
    st.markdown("""
    Net spillover is calculated as $S_{i}^{\\text{TO}} - S_{i}^{\\text{FROM}}$. 
    Positive values indicate **net transmitters** of systemic volatility; negative values represent **net receivers**.
    """)
    fig_net = create_net_spillover_bar(rq1_df)
    st.plotly_chart(fig_net, width="stretch")

with col_right:
    st.subheader("3. Directional Spillover to BTC-IDR")
    st.markdown("""
    Decomposition of volatility transmission entering **BTC-IDR (Indodax)** from each individual asset, 
    derived from the static average spillover table (RQ1).
    """)

    # Extract BTC-IDR row from RQ1 table
    btc_row = rq1_df[rq1_df["Source"] == "CRYPTO_BTC_IDR_INDODAX"]
    if not btc_row.empty:
        dir_data = []
        for col in ["SAHAM_NVDA", "SAHAM_MSFT", "SAHAM_TSLA", "SAHAM_GOOGL", "SAHAM_META", "CRYPTO_BTC_USD", "CRYPTO_ETH_USD"]:
            if col in rq1_df.columns:
                val = float(btc_row[col].values[0])
                asset_label = INSTRUMENT_NAMES.get(col, col)
                dir_data.append({
                    "Transmitter Asset": asset_label,
                    "Spillover to BTC-IDR (%)": val,
                    "Category": "US AI Stock" if "SAHAM" in col else "Global Crypto"
                })
        
        dir_df = pd.DataFrame(dir_data).sort_values("Spillover to BTC-IDR (%)", ascending=False)
        st.dataframe(
            dir_df.style.format({"Spillover to BTC-IDR (%)": "{:.2f}%"}),
            width="stretch",
            hide_index=True
        )

        own_shock = float(btc_row["CRYPTO_BTC_IDR_INDODAX"].values[0])
        total_from_others = float(btc_row["FROM Others"].values[0])
        st.info(f"**Own Variance Share**: {own_shock:.2f}% | **Total Received from Others**: {total_from_others:.2f}%")
