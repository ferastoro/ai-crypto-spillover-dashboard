# AI Tech Stocks & Crypto Volatility Spillover Dashboard

An interactive analytical dashboard exploring the transmission of volatility from major US Artificial Intelligence technology stocks (NVDA, MSFT, TSLA, GOOGL, META) to cryptocurrency markets (BTC-IDR on Indodax, BTC-USD, ETH-USD), based on Time-Varying Parameter Vector Autoregressive (TVP-VAR) modeling and deep learning out-of-sample forecasting architectures.

## Project Overview

- **Spillover Dynamics**: Dynamic Total Spillover Index (TSI), directional transmission networks, and net spillover asymmetries between AI technology equities and digital assets.
- **Out-of-Sample Forecasting**: Empirical evaluation of multi-horizon forecasts ($h=1$ and $h=10$) comparing deep learning models (Temporal Fusion Transformer, NBEATSx) with econometric and statistical baselines (HAR, GARCH, EGARCH, Rolling Mean, Historical Expansion).
- **Statistical Significance**: Robust model comparison using Diebold-Mariano tests with Newey-West variance estimation, small-sample Harvey-Leybourne-Newbold (HLN) adjustments, and Bonferroni family-wise error rate corrections.
- **Critical Limitations**: Objective assessment of baseline persistence dominance, forecast correlation bounds, volatility proxy sensitivity, and market microstructure considerations.
- **⚡ Live HAR Forecast Tracking**: Real-time daily out-of-sample volatility forecasting for BTC-IDR using the econometric Heterogeneous Autoregressive (HAR) model, automated daily via GitHub Actions.

## Data Source & Live Update Pipeline

- **Academic Thesis Benchmark**: Historical datasets encompass 1,555 observations (January 2020 through September 2026) and 335 rolling evaluation cutoffs. Deep learning models and TVP-VAR vectors remain static as established in the thesis.
- **Automated Live Pipeline**: A scheduled GitHub Actions workflow runs daily at 01:00 UTC (after the 00:00:00–23:59:59 UTC daily candle closes), pulling completed BTC/IDR daily candles from the Indodax Public API, updating `data/live/btcidr_history.csv`, generating $h=1$ and $h=10$ HAR forecasts, populating realized target variances, and maintaining the out-of-sample scorecard in `data/live/live_scorecard.csv`.

## Local Installation & Execution

### Prerequisites

- Python 3.10 or higher
- `pip` package manager

### Steps

1. Clone or download this repository.
2. Navigate to the repository root:
   ```bash
   cd rsic-dashboard
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Run the live forecast update manually:
   ```bash
   python live/update_forecast.py
   ```
5. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
6. Open your web browser at `http://localhost:8501`.

## Project Structure

```text
├── .github/
│   └── workflows/
│       └── daily_forecast.yml # Automated daily HAR forecast update workflow
├── .streamlit/
│   └── config.toml          # Streamlit dark theme and server configurations
├── data/
│   ├── RQ1_Average_Spillover_Table_vol.csv # Static average spillover matrix
│   ├── forecasts_v6.csv     # Compiled out-of-sample predictions and ground truth
│   ├── metrics_dm_summary.csv # Diebold-Mariano test results with Bonferroni correction
│   ├── metrics_summary.csv  # Model evaluation metrics (MSE, MAE, RMSE, R2_OOS, r)
│   ├── tvp_var_dynamic_features.csv # Dynamic TVP-VAR TSI and net spillover features
│   └── live/
│       ├── btcidr_history.csv # Continuous BTC/IDR daily history and returns
│       ├── latest_forecast.csv # Append-only live & backfilled HAR forecasts
│       └── live_scorecard.csv # Real-time out-of-sample MSE scorecard
├── live/
│   ├── __init__.py
│   ├── har_model.py         # Self-contained HAR econometric model & replication tests
│   └── update_forecast.py   # Daily Indodax data ingestion and forecast pipeline
├── pages/
│   ├── 1_Spillover.py       # Dynamic TSI time series and directional spillover tables
│   ├── 2_Forecast.py        # Interactive forecast inspection and range metrics
│   ├── 3_Model_Comparison.py # Model rankings and Diebold-Mariano hypothesis tests
│   ├── 4_Limitations.py     # Critical review of baseline dominance and data caveats
│   └── 5_Latest_Forecast.py # ⚡ Live daily HAR volatility forecast tracking
├── src/
│   ├── __init__.py
│   ├── charts.py            # Plotly visualization components
│   ├── config.py            # Relative path definitions and styling tokens
│   └── data_loader.py       # Cached data access functions (st.cache_data)
├── .gitignore               # Ignored local files, cache, and virtual environments
├── README.md                # Project documentation
├── app.py                   # Main dashboard landing page
└── requirements.txt         # Pinned production dependencies
```

## Disclaimer

This dashboard and its underlying calculations represent **academic research output** designed solely for educational and empirical analysis. Nothing contained herein constitutes financial advice, investment recommendations, endorsement of specific securities or digital assets, or operational trading signals. Past statistical associations do not guarantee future market behavior.
