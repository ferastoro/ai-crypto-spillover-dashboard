import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.data_loader import load_forecasts
from src.charts import create_forecast_plot

st.set_page_config(
    page_title="Out-of-Sample Forecasts | RSIC Dashboard",
    page_icon="🔮",
    layout="wide"
)

st.title("Out-of-Sample Volatility Forecasts")
st.markdown("""
Interactive evaluation of realized log-variance ($y_{\\text{true}}$) versus model predictions ($y_{\\text{pred}}$) 
across 335 rolling cutoff windows in the out-of-sample test period. 
Deep learning architectures (TFT, NBEATSx) represent 5-seed ensemble averages.
""")

# Load forecasts data
df_forecasts = load_forecasts()

# Sidebar Controls
st.sidebar.header("Forecast Settings")

# 1. Horizon Selection
horizon_choice = st.sidebar.radio(
    "Forecast Horizon (h):",
    options=[1, 10],
    format_func=lambda x: f"h = {x} ({'1-day ahead' if x == 1 else '10-days ahead'})",
    index=0
)

# 2. Feature Set Selection
feature_set_choice = st.sidebar.radio(
    "Deep Learning Feature Set:",
    options=["SPILL", "NOSPILL"],
    help="SPILL includes 9 TVP-VAR causal features; NOSPILL uses returns, weekend effects, and calendar features.",
    index=0
)

# 3. Model Multi-Select
all_models = ["TFT", "NBEATSx", "HAR", "GARCH11", "EGARCH", "Roll_20", "Hist_Exp"]
default_models = ["TFT", "HAR", "GARCH11"]

selected_models = st.sidebar.multiselect(
    "Select Models to Compare:",
    options=all_models,
    default=default_models
)

if not selected_models:
    st.warning("Please select at least one model from the sidebar.")
    st.stop()

# 4. Date Range Filter
min_date = df_forecasts["real_date"].min().to_pydatetime()
max_date = df_forecasts["real_date"].max().to_pydatetime()

selected_dates = st.sidebar.slider(
    "Evaluation Target Date Range:",
    min_value=min_date,
    max_value=max_date,
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)

# Filter data
sub_h = df_forecasts[df_forecasts["horizon"] == horizon_choice].copy()

# Filter models & feature sets:
# For models with SPILL/NOSPILL (TFT, NBEATSx, HAR), match selected feature_set;
# For strictly BASELINE models (Hist_Exp, Roll_20, GARCH11, EGARCH), include their BASELINE spec.
filtered_rows = []
for m in selected_models:
    if m in ["TFT", "NBEATSx"]:
        m_df = sub_h[(sub_h["model"] == m) & (sub_h["feature_set"] == feature_set_choice)]
    elif m == "HAR":
        # Check if HAR has selected feature set, otherwise fallback to BASELINE
        har_fset = sub_h[(sub_h["model"] == "HAR") & (sub_h["feature_set"] == feature_set_choice)]
        if not har_fset.empty:
            m_df = har_fset
        else:
            m_df = sub_h[(sub_h["model"] == "HAR") & (sub_h["feature_set"] == "BASELINE")]
    else:
        m_df = sub_h[(sub_h["model"] == m) & (sub_h["feature_set"] == "BASELINE")]
    filtered_rows.append(m_df)

plot_df = pd.concat(filtered_rows, ignore_index=True)

# Date filter
plot_df = plot_df[(plot_df["real_date"] >= pd.Timestamp(selected_dates[0])) & (plot_df["real_date"] <= pd.Timestamp(selected_dates[1]))]

if plot_df.empty:
    st.error("No data found for the selected configuration and date range.")
    st.stop()

# Plot Time Series
st.subheader(f"Forecast Trajectories (Horizon h = {horizon_choice})")
fig_fc = create_forecast_plot(plot_df, selected_models)
st.plotly_chart(fig_fc, use_container_width=True)

st.divider()

# Range-Specific Metrics Table
st.subheader("Performance Metrics on Selected Evaluation Range")
st.markdown(f"Evaluated on **{plot_df['real_date'].nunique()} target observations** from `{selected_dates[0].strftime('%Y-%m-%d')}` to `{selected_dates[1].strftime('%Y-%m-%d')}`.")

perf_data = []
for (m_name, fset), grp in plot_df.groupby(["model", "feature_set"]):
    yt = grp["y_true"].values
    yp = grp["y_pred"].values
    mse = mean_squared_error(yt, yp)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mse)
    pred_std = float(np.std(yp))
    corr = float(np.corrcoef(yt, yp)[0, 1]) if pred_std > 0 and np.std(yt) > 0 else np.nan
    bias_ratio = float(np.mean(yp) / np.mean(yt)) if np.mean(yt) > 0 else np.nan

    perf_data.append({
        "Model": m_name,
        "Feature Set": fset,
        "MSE (×10⁻⁶)": mse * 1e6,
        "MAE (×10⁻⁴)": mae * 1e4,
        "RMSE (×10⁻³)": rmse * 1e3,
        "Pred Std (×10⁻⁴)": pred_std * 1e4,
        "Correlation (r)": corr,
        "Bias Ratio (Pred/True)": bias_ratio
    })

perf_df = pd.DataFrame(perf_data).sort_values("MSE (×10⁻⁶)")

st.dataframe(
    perf_df.style.format({
        "MSE (×10⁻⁶)": "{:.3f}",
        "MAE (×10⁻⁴)": "{:.3f}",
        "RMSE (×10⁻³)": "{:.3f}",
        "Pred Std (×10⁻⁴)": "{:.3f}",
        "Correlation (r)": "{:.4f}",
        "Bias Ratio (Pred/True)": "{:.3f}"
    }).highlight_min(subset=["MSE (×10⁻⁶)", "MAE (×10⁻⁴)"], color="#1E3A8A"),
    use_container_width=True,
    hide_index=True
)

st.caption("Note: Lower MSE/MAE indicates superior forecast precision. Correlation measures directional covariance with realized proxy.")
