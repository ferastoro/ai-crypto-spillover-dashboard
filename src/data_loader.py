import streamlit as st
import pandas as pd
from src.config import (
    TVP_FEATURES_FILE,
    RQ1_TABLE_FILE,
    FORECASTS_FILE,
    METRICS_SUMMARY_FILE,
    METRICS_DM_FILE,
    BTCIDR_HISTORY_FILE,
    LATEST_FORECAST_FILE,
    LIVE_SCORECARD_FILE,
)


@st.cache_data
def load_tvp_features() -> pd.DataFrame:
    """Load dynamic TVP-VAR features including TSI and net spillovers."""
    df = pd.read_csv(TVP_FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_rq1_table() -> pd.DataFrame:
    """Load the static RQ1 average spillover table."""
    df = pd.read_csv(RQ1_TABLE_FILE)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "Source"})
    return df


@st.cache_data
def load_forecasts() -> pd.DataFrame:
    """Load combined forecasts for DL models and baselines."""
    df = pd.read_csv(FORECASTS_FILE)
    df["cutoff"] = pd.to_datetime(df["cutoff"])
    df["real_date"] = pd.to_datetime(df["real_date"])
    return df


@st.cache_data
def load_metrics_summary() -> pd.DataFrame:
    """Load model evaluation metrics (MSE, MAE, RMSE, R2_OOS, etc.)."""
    df = pd.read_csv(METRICS_SUMMARY_FILE)
    return df


@st.cache_data
def load_metrics_dm() -> pd.DataFrame:
    """Load Diebold-Mariano test results."""
    df = pd.read_csv(METRICS_DM_FILE)
    return df


@st.cache_data(ttl=3600)
def load_latest_forecasts() -> pd.DataFrame:
    """Load live and backfilled daily HAR forecasts."""
    if not LATEST_FORECAST_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(LATEST_FORECAST_FILE)
    df["as_of_date"] = pd.to_datetime(df["as_of_date"])
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df


@st.cache_data(ttl=3600)
def load_live_scorecard() -> pd.DataFrame:
    """Load live scorecard containing running MSE of HAR vs baselines."""
    if not LIVE_SCORECARD_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(LIVE_SCORECARD_FILE)


@st.cache_data(ttl=3600)
def load_btcidr_history() -> pd.DataFrame:
    """Load full historical daily BTC/IDR prices and realized variances."""
    if not BTCIDR_HISTORY_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(BTCIDR_HISTORY_FILE)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
