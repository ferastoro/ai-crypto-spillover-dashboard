from pathlib import Path

# Base project root directory (relative to this file)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
DATA_LIVE_DIR = DATA_DIR / "live"

# Static historical data file paths
TVP_FEATURES_FILE = DATA_DIR / "tvp_var_dynamic_features.csv"
RQ1_TABLE_FILE = DATA_DIR / "RQ1_Average_Spillover_Table_vol.csv"
FORECASTS_FILE = DATA_DIR / "forecasts_v6.csv"
METRICS_SUMMARY_FILE = DATA_DIR / "metrics_summary.csv"
METRICS_DM_FILE = DATA_DIR / "metrics_dm_summary.csv"

# Live pipeline data file paths
BTCIDR_HISTORY_FILE = DATA_LIVE_DIR / "btcidr_history.csv"
LATEST_FORECAST_FILE = DATA_LIVE_DIR / "latest_forecast.csv"
LIVE_SCORECARD_FILE = DATA_LIVE_DIR / "live_scorecard.csv"

# Color Palette for Charts & UI
COLORS = {
    "TFT": "#3B82F6",         # Modern Blue
    "NBEATSx": "#8B5CF6",     # Modern Purple
    "HAR": "#10B981",         # Emerald Green
    "Roll_20": "#F59E0B",     # Amber/Orange
    "Hist_Exp": "#6B7280",    # Cool Gray
    "GARCH11": "#EC4899",     # Pink/Rose
    "EGARCH": "#14B8A6",      # Teal
    "y_true": "#F8FAFC",      # Bright White
    "live_forecast": "#10B981", # Emerald for live forecast
    "backfill_forecast": "#6366F1", # Indigo for backfill forecast
    "baseline": "#94A3B8",    # Slate
    "dl": "#3B82F6",
    "positive_spill": "#EF4444", # Red (Net transmitter of risk)
    "negative_spill": "#10B981", # Green (Net receiver of risk)
}

# Instrument display name map
INSTRUMENT_NAMES = {
    "saham_nvda": "NVDA (Nvidia)",
    "saham_msft": "MSFT (Microsoft)",
    "saham_tsla": "TSLA (Tesla)",
    "saham_googl": "GOOGL (Alphabet)",
    "saham_meta": "META (Meta Platforms)",
    "crypto_btc_usd": "BTC-USD",
    "crypto_eth_usd": "ETH-USD",
    "crypto_btc_idr_indodax": "BTC-IDR (Indodax)",
    "crypto_eth_idr_indodax": "ETH-IDR (Indodax)",
    "SAHAM_NVDA": "NVDA",
    "SAHAM_MSFT": "MSFT",
    "SAHAM_TSLA": "TSLA",
    "SAHAM_GOOGL": "GOOGL",
    "SAHAM_META": "META",
    "CRYPTO_BTC_USD": "BTC-USD",
    "CRYPTO_ETH_USD": "ETH-USD",
    "CRYPTO_BTC_IDR_INDODAX": "BTC-IDR (Indodax)"
}
