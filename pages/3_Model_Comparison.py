import streamlit as st
import pandas as pd
from src.data_loader import load_metrics_summary, load_metrics_dm
from src.charts import create_model_comparison_bar

st.set_page_config(
    page_title="Model Comparison & Statistical Tests | RSIC Dashboard",
    page_icon="⚖️",
    layout="wide"
)

st.title("Model Performance & Diebold-Mariano Tests")
st.markdown("""
Comprehensive benchmark ranking and formal pairwise statistical hypothesis testing using the 
**Diebold-Mariano (DM)** test with Newey-West variance estimation, small-sample Harvey-Leybourne-Newbold (HLN) adjustments, 
and family-wise Bonferroni correction.
""")

metrics_df = load_metrics_summary()
dm_df = load_metrics_dm()

# Horizon Selector Tabs
tab_h1, tab_h10 = st.tabs(["Horizon h = 1 (1-Day Ahead)", "Horizon h = 10 (10-Days Ahead)"])

for tab, h_val in [(tab_h1, 1), (tab_h10, 10)]:
    with tab:
        st.subheader(f"1. Out-of-Sample Accuracy Rankings (h = {h_val})")
        
        # Comparison Bar Chart
        metric_col1, metric_col2 = st.columns([3, 1])
        with metric_col2:
            chosen_metric = st.selectbox(
                "Select Ranking Metric:",
                options=["mse", "mae"],
                format_func=lambda x: "MSE (Mean Squared Error)" if x == "mse" else "MAE (Mean Absolute Error)",
                key=f"metric_choice_{h_val}"
            )
            st.info("""
            **Bar Color Legend:**
            - 🟦 **Blue**: Deep Learning Models (TFT, NBEATSx)
            - ⬜ **Gray**: Econometric & Persistence Baselines (HAR, GARCH, EGARCH, Roll_20, Hist_Exp)
            """)
        
        with metric_col1:
            fig_bar = create_model_comparison_bar(metrics_df, horizon=h_val, metric=chosen_metric)
            st.plotly_chart(fig_bar, use_container_width=True)

        # Full Metrics Table
        st.subheader(f"2. Comprehensive Metrics Summary (h = {h_val})")
        sub_m = metrics_df[metrics_df["horizon"] == h_val].copy()
        
        # Display Table formatting
        disp_m = sub_m[[
            "model", "feature_set", "is_baseline", "mse", "mae", "rmse", "r2_oos_histexp", "r2_oos_roll20", "pred_corr"
        ]].copy()
        
        disp_m["mse"] = disp_m["mse"] * 1e6
        disp_m["mae"] = disp_m["mae"] * 1e4
        disp_m["rmse"] = disp_m["rmse"] * 1e3
        
        disp_m.columns = [
            "Model", "Feature Set", "Is Baseline", "MSE (×10⁻⁶)", "MAE (×10⁻⁴)", "RMSE (×10⁻³)",
            "R²_OOS (vs Hist_Exp)", "R²_OOS (vs Roll_20)", "Correlation (r)"
        ]
        
        st.dataframe(
            disp_m.style.format({
                "MSE (×10⁻⁶)": "{:.4f}",
                "MAE (×10⁻⁴)": "{:.4f}",
                "RMSE (×10⁻³)": "{:.4f}",
                "R²_OOS (vs Hist_Exp)": "{:+.4f}",
                "R²_OOS (vs Roll_20)": "{:+.4f}",
                "Correlation (r)": "{:.4f}"
            }).highlight_min(subset=["MSE (×10⁻⁶)", "MAE (×10⁻⁴)"], color="#1E3A8A"),
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # Section 3: Diebold-Mariano Tests
        st.subheader(f"3. Pairwise Diebold-Mariano Tests (h = {h_val})")
        st.markdown("""
        The null hypothesis ($H_0$) states that the candidate model and benchmark model have equal forecast accuracy. 
        Negative DM statistics indicate lower loss for the candidate model. Statistical significance is assessed 
        at $\\alpha=0.05$ after **Bonferroni multiplicity correction** ($m=12$ for baseline comparisons, $m=4$ for ablation tests).
        """)

        h_str = f"h={h_val}"
        sub_dm = dm_df[dm_df["horizon"] == h_str].copy()

        # Split into Ensemble vs Baselines & Ablation
        comp_type = st.radio(
            "Comparison Category:",
            options=["Ensemble vs Baselines (m=12)", "Spillover Ablation: SPILL vs NOSPILL (m=4)"],
            key=f"dm_filter_{h_val}",
            horizontal=True
        )

        if "Baselines" in comp_type:
            view_dm = sub_dm[sub_dm["comparison_type"] == "Ensemble_vs_Baseline"].copy()
        else:
            view_dm = sub_dm[sub_dm["comparison_type"] == "Ablation_SPILL_vs_NOSPILL"].copy()

        # Highlight function for Winner / Significance
        def highlight_dm(row):
            if row["Winner"] != "Tie":
                return ["background-color: rgba(16, 185, 129, 0.25)"] * len(row)
            elif row["p-value (Raw)"] < 0.05:
                return ["background-color: rgba(245, 158, 11, 0.20)"] * len(row)
            return [""] * len(row)

        table_cols = ["config", "benchmark", "stat", "p_raw", "p_bonf", "m_family", "md_mse", "winner"]
        display_dm = view_dm[table_cols].copy()
        display_dm.columns = ["Candidate Config", "Benchmark Reference", "DM Statistic", "p-value (Raw)", "p-value (Bonferroni)", "Family Size (m)", "Mean Loss Diff (md_MSE)", "Winner"]

        st.dataframe(
            display_dm.style.format({
                "DM Statistic": "{:+.4f}",
                "p-value (Raw)": "{:.4f}",
                "p-value (Bonferroni)": "{:.4f}",
                "Mean Loss Diff (md_MSE)": "{:+.3e}"
            }).apply(highlight_dm, axis=1),
            use_container_width=True,
            hide_index=True
        )

        st.caption("🟢 Green highlight indicates statistically significant superiority after Bonferroni correction (Winner != Tie).")
