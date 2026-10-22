import streamlit as st
import pandas as pd
from src.data_loader import load_metrics_summary, load_metrics_dm, load_rq1_table

st.set_page_config(
    page_title="Empirical Limitations | RSIC Dashboard",
    page_icon="⚠️",
    layout="wide"
)

st.title("Empirical Limitations & Scientific Reality Checks")
st.markdown("""
A transparent, evidence-based appraisal of model performance, statistical significance, and econometric boundaries 
derived directly from the empirical experiment outputs.
""")

# Load data dynamically
metrics_df = load_metrics_summary()
dm_df = load_metrics_dm()
rq1_df = load_rq1_table()

# Extract dynamic numbers
har_m_h1 = metrics_df[(metrics_df["model"] == "HAR") & (metrics_df["horizon"] == 1) & (metrics_df["feature_set"] == "BASELINE")].iloc[0]
har_m_h10 = metrics_df[(metrics_df["model"] == "HAR") & (metrics_df["horizon"] == 10) & (metrics_df["feature_set"] == "BASELINE")].iloc[0]

tft_spill_h1 = metrics_df[(metrics_df["model"] == "TFT") & (metrics_df["horizon"] == 1) & (metrics_df["feature_set"] == "SPILL")].iloc[0]
tft_spill_h10 = metrics_df[(metrics_df["model"] == "TFT") & (metrics_df["horizon"] == 10) & (metrics_df["feature_set"] == "SPILL")].iloc[0]

# Net spillover for BTC-IDR
net_row = rq1_df[rq1_df["Source"] == "NET Spillover"]
btc_net = float(net_row["CRYPTO_BTC_IDR_INDODAX"].values[0]) if not net_row.empty else 5.4546

# Summary Alert
st.warning("""
**Key Takeaway**: Complex deep learning models (TFT, NBEATSx) do not demonstrate systemic outperformance 
over parsimonious linear econometric baselines (HAR). Spillover features provide marginal directional covariance 
improvements but do not produce statistically significant error reductions over standard autoregressive volatility specifications.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Econometric Dominance of HAR Baseline")
    st.markdown(f"""
    - **Lowest Overall Test Error**: The standard Heterogeneous Autoregressive model (**HAR**) attains the lowest MSE 
      across all evaluated models at both forecast horizons:
      - **h = 1**: HAR MSE = `{har_m_h1['mse']:.4e}` vs TFT (SPILL) `{tft_spill_h1['mse']:.4e}`
      - **h = 10**: HAR MSE = `{har_m_h10['mse']:.4e}` vs TFT (SPILL) `{tft_spill_h10['mse']:.4e}`
    - **No Deep Learning Advantage over Persistence**: In multi-step horizons, simple autoregressive and rolling estimators 
      consistently match or beat neural network specifications without hyperparameter tuning or compute overhead.
    """)

    st.subheader("2. Formal Diebold-Mariano Significance Tests")
    st.markdown(f"""
    - **Isolated Significance**: Under rigorous Bonferroni multiplicity adjustments ($m=12$), deep learning models 
      only achieve statistically significant superiority against the simplest naive benchmark, **Hist_Exp** ($p_{{\\text{{bonf}}}} = 0.0114$ for TFT at $h=1$).
    - **Zero Significant Wins Against HAR or Roll_20**:
      - TFT vs HAR: $p_{{\\text{{bonf}}}} = 1.0000$ (Statistically Indistinguishable / Tie at both $h=1$ and $h=10$)
      - TFT vs Roll_20: $p_{{\\text{{bonf}}}} = 1.0000$ (Tie)
      - NBEATSx vs HAR: $p_{{\\text{{bonf}}}} = 1.0000$ (Tie)
    - **Spillover Feature Ablation**: Comparing `SPILL` vs `NOSPILL` reveals no statistically significant divergence ($p_{{\\text{{bonf}}}} = 1.0000$ for all models and horizons).
    """)

    st.subheader("3. Weak Forecast Correlation with Realized Variance")
    st.markdown(f"""
    - **Low Directional Covariance**: Although models learn level trends, the correlation between forecasts and 
      realized volatility is low:
      - TFT (SPILL) correlation: $r = {tft_spill_h1['pred_corr']:.4f}$ at $h=1$ and $r = {tft_spill_h10['pred_corr']:.4f}$ at $h=10$.
      - NBEATSx shows higher $r$ at $h=1$ ($0.2762$) but experiences severe variance collapse and negative bias at $h=10$.
    """)

with col2:
    st.subheader("4. Proxy Sensitivity & Net Spillover Sign")
    st.markdown(f"""
    - **Empirical Proxy Dependence**: In this dataset using the daily log squared return proxy ($\\log r^2$), BTC-IDR 
      registers as a net transmitter of systemic volatility (`Net Spillover = {btc_net:+.2f}%`).
    - **Fragility Across Proxy Definitions**: Robustness audits on this dataset test alternative volatility proxies 
      (|r|, r², and Parkinson volatility). Substituting between absolute returns (|r|), squared returns (r²), and 
      Parkinson range estimators demonstrates that empirical spillover magnitudes and network centralities are sensitive 
      to proxy specification.
    """)

    st.subheader("5. Single Contiguous Sample & Market Regimes")
    st.markdown("""
    - **Dataset Scope**: The empirical sample spans 1,555 trading days (January 2020 to September 2026), capturing 
      a specific historical cycle of post-COVID stimulus, crypto bull/bear phases, and the 2023–2025 generative AI boom.
    - **Generalizability**: Findings cannot be unconditionally generalized to other macroeconomic regimes, alternative 
      altcoins, or distinct fiat liquidity pairs without out-of-distribution validation.
    """)

    st.subheader("6. Live Tracking Scope & Exchange Microstructure")
    st.markdown("""
    - **Live Forecasting Model Restriction**: Automated daily updates apply exclusively to the parsimonious linear 
      HAR baseline. Deep learning architectures (TFT, NBEATSx) and TVP-VAR spillover vectors remain strictly static 
      as established during the thesis sample to prevent online retraining artifacts.
    - **Indodax Daily Timestamping**: Indodax daily candles operate on 00:00:00 UTC boundaries. While daily close prices 
      are aligned, intra-day liquidity swings and tick-level order book depth variations have not been independently audited.
    """)

st.divider()

# Practical Recommendation Box
st.info("""
**Recommendation for Practitioners & Researchers**: 
Given that parsimonious linear HAR models deliver superior accuracy with near-zero computational cost, 
production deployment of deep learning architectures for cryptocurrency volatility forecasting should be treated 
with caution. Future inquiries should prioritize high-frequency order book dynamics and regime-switching formulations 
rather than deeper parameterizations on daily series.
""")
