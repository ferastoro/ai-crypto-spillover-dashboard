import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from src.config import COLORS, INSTRUMENT_NAMES

# Dark theme layout defaults for Plotly charts
PLOTLY_TEMPLATE = "plotly_dark"
FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif"


def format_chart_layout(fig: go.Figure, title: str, xaxis_title: str = "", yaxis_title: str = "") -> go.Figure:
    """Apply consistent styling, clean gridlines, and typography to a Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": f"<b>{title}</b>",
            "font": {"family": FONT_FAMILY, "size": 16, "color": "#F8FAFC"},
            "x": 0.02,
            "xanchor": "left"
        },
        font={"family": FONT_FAMILY, "color": "#94A3B8"},
        paper_bgcolor="rgba(15, 23, 42, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        xaxis={
            "title": xaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.07)",
            "zerolinecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11}
        },
        yaxis={
            "title": yaxis_title,
            "gridcolor": "rgba(255, 255, 255, 0.07)",
            "zerolinecolor": "rgba(255, 255, 255, 0.15)",
            "tickfont": {"size": 11}
        },
        legend={
            "bgcolor": "rgba(15, 23, 42, 0.8)",
            "bordercolor": "rgba(255, 255, 255, 0.1)",
            "borderwidth": 1,
            "font": {"size": 12}
        },
        margin={"l": 50, "r": 30, "t": 60, "b": 40},
        hovermode="x unified"
    )
    return fig


def create_tsi_chart(df: pd.DataFrame) -> go.Figure:
    """Create Total Spillover Index (TSI) time series chart with descriptive 80th percentile line."""
    fig = go.Figure()

    # TSI Line
    fig.add_trace(go.Scatter(
        x=df["Date"],
        y=df["tvp_tsi"],
        mode="lines",
        name="Total Spillover Index (%)",
        line={"color": "#3B82F6", "width": 2},
        hovertemplate="<b>TSI</b>: %{y:.2f}%<extra></extra>"
    ))

    # 80th percentile threshold (Descriptive only)
    p80 = float(df["tvp_tsi"].quantile(0.80))
    fig.add_hline(
        y=p80,
        line_dash="dash",
        line_color="#F59E0B",
        annotation_text=f"80th Percentile ({p80:.1f}%) — Descriptive threshold, not a validated signal",
        annotation_position="top right",
        annotation_font_color="#F59E0B",
        annotation_font_size=11
    )

    format_chart_layout(
        fig,
        title="Dynamic Total Spillover Index (TSI) Over Time",
        xaxis_title="Date",
        yaxis_title="Total Spillover Index (%)"
    )
    return fig


def create_net_spillover_bar(rq1_df: pd.DataFrame) -> go.Figure:
    """Create horizontal bar chart of average net spillover per instrument."""
    # Row 9 is NET Spillover
    net_row = rq1_df[rq1_df["Source"] == "NET Spillover"].copy()
    if net_row.empty:
        # Fallback to last row
        net_row = rq1_df.tail(1).copy()

    asset_cols = [c for c in rq1_df.columns if c not in ["Source", "FROM Others", "TO Others"] and not pd.isna(net_row[c].values[0])]
    
    data = []
    for col in asset_cols:
        val = float(net_row[col].values[0])
        clean_name = INSTRUMENT_NAMES.get(col, col.replace("SAHAM_", "").replace("CRYPTO_", ""))
        data.append({
            "asset": clean_name,
            "net_spillover": val,
            "direction": "Net Transmitter (>0)" if val > 0 else "Net Receiver (<0)"
        })

    plot_df = pd.DataFrame(data).sort_values("net_spillover", ascending=True)

    colors = [COLORS["positive_spill"] if v > 0 else COLORS["negative_spill"] for v in plot_df["net_spillover"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_df["net_spillover"],
        y=plot_df["asset"],
        orientation="h",
        marker={"color": colors, "line": {"width": 1, "color": "rgba(255,255,255,0.2)"}},
        text=[f"{v:+.2f}%" for v in plot_df["net_spillover"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Net Spillover: %{x:+.2f}%<extra></extra>"
    ))

    format_chart_layout(
        fig,
        title="Average Net Spillover by Instrument (Transmitter vs Receiver)",
        xaxis_title="Net Spillover (%) [Positive = Transmitter, Negative = Receiver]",
        yaxis_title=""
    )
    fig.update_layout(hovermode="closest")
    return fig


def create_forecast_plot(df: pd.DataFrame, selected_models: list[str]) -> go.Figure:
    """Plot y_true vs model predictions over selected evaluation dates."""
    fig = go.Figure()

    # Ground truth (y_true)
    # y_true is the same for all models on each date; pick the first model group
    first_model = df["model"].iloc[0]
    first_fset = df["feature_set"].iloc[0]
    base_sub = df[(df["model"] == first_model) & (df["feature_set"] == first_fset)].sort_values("real_date")

    fig.add_trace(go.Scatter(
        x=base_sub["real_date"],
        y=base_sub["y_true"],
        mode="lines",
        name="Realized Log-Variance (y_true)",
        line={"color": COLORS["y_true"], "width": 2.5},
        hovertemplate="<b>y_true</b>: %{y:.6e}<extra></extra>"
    ))

    # Distinct model lines
    palette = [
        COLORS["TFT"],
        COLORS["NBEATSx"],
        COLORS["HAR"],
        COLORS["GARCH11"],
        COLORS["EGARCH"],
        COLORS["Roll_20"],
        COLORS["Hist_Exp"],
        "#06B6D4", "#A855F7", "#EC4899"
    ]

    color_idx = 0
    for (m_name, fset), grp in df.groupby(["model", "feature_set"]):
        grp_sorted = grp.sort_values("real_date")
        line_color = COLORS.get(m_name, palette[color_idx % len(palette)])
        color_idx += 1

        label = f"{m_name} ({fset})" if fset != "BASELINE" else m_name
        fig.add_trace(go.Scatter(
            x=grp_sorted["real_date"],
            y=grp_sorted["y_pred"],
            mode="lines",
            name=label,
            line={"color": line_color, "width": 1.7},
            hovertemplate=f"<b>{label}</b>: %{{y:.6e}}<extra></extra>"
        ))

    format_chart_layout(
        fig,
        title="Out-of-Sample Predictions vs Realized Volatility",
        xaxis_title="Target Evaluation Date",
        yaxis_title="Log Variance Proxy"
    )
    return fig


def create_model_comparison_bar(df: pd.DataFrame, horizon: int, metric: str = "mse") -> go.Figure:
    """Create sorted bar chart comparing models on a specific metric for given horizon."""
    sub = df[df["horizon"] == horizon].copy()
    sub.sort_values(metric, ascending=True, inplace=True)

    # Format model label
    sub["label"] = sub.apply(
        lambda r: f"{r['model']} ({r['feature_set']})" if r["feature_set"] != "BASELINE" else f"{r['model']} (Baseline)",
        axis=1
    )

    colors = [COLORS["baseline"] if r["is_baseline"] else COLORS["dl"] for _, r in sub.iterrows()]

    metric_name = "MSE (Mean Squared Error)" if metric == "mse" else "MAE (Mean Absolute Error)"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=sub["label"],
        y=sub[metric],
        marker={"color": colors, "line": {"width": 1, "color": "rgba(255,255,255,0.2)"}},
        text=[f"{v:.4e}" for v in sub[metric]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>" + f"{metric.upper()}: %{{y:.6e}}<extra></extra>"
    ))

    format_chart_layout(
        fig,
        title=f"{metric_name} Comparison at h={horizon} (Sorted: Lower is Better)",
        xaxis_title="Model Specification",
        yaxis_title=metric.upper()
    )
    fig.update_layout(xaxis={"tickangle": -30}, hovermode="closest")
    return fig
