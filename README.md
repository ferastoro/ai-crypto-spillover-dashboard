# AI Tech Stocks & Crypto Volatility Spillover Dashboard

An interactive analytical dashboard exploring the transmission of volatility from major US Artificial Intelligence technology stocks (NVDA, MSFT, TSLA, GOOGL, META) to cryptocurrency markets (BTC-IDR on Indodax, BTC-USD, ETH-USD), based on Time-Varying Parameter Vector Autoregressive (TVP-VAR) modeling and deep learning out-of-sample forecasting architectures.

## Project Overview

- **Spillover Dynamics**: Dynamic Total Spillover Index (TSI), directional transmission networks, and net spillover asymmetries between AI technology equities and digital assets.
- **Out-of-Sample Forecasting**: Empirical evaluation of multi-horizon forecasts ($h=1$ and $h=10$) comparing deep learning models (Temporal Fusion Transformer, NBEATSx) with econometric and statistical baselines (HAR, GARCH, EGARCH, Rolling Mean, Historical Expansion).
- **Statistical Significance**: Robust model comparison using Diebold-Mariano tests with Newey-West variance estimation, small-sample Harvey-Leybourne-Newbold (HLN) adjustments, and Bonferroni family-wise error rate corrections.
- **Critical Limitations**: Objective discussion of benchmark persistence dominance, forecast correlation bounds, volatility proxy sensitivity, and market microstructure considerations.

## Data Source

The empirical datasets in this repository are derived from academic research analyzing daily market observations from January 2020 through September 2026. The dataset encompasses 1,555 historical observations and 335 rolling out-of-sample forecast cutoff windows. All inputs represent final compiled results, ensuring that the application operates in a strict read-only mode without online model training or parameter refitting.

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
4. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
5. Open your web browser at `http://localhost:8501`.

## Project Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit dark theme and server configurations
├── data/
│   ├── RQ1_Average_Spillover_Table_vol.csv # Static average spillover matrix
│   ├── forecasts_v6.csv     # Compiled out-of-sample predictions and ground truth
│   ├── metrics_dm_summary.csv # Diebold-Mariano test results with Bonferroni correction
│   ├── metrics_summary.csv  # Model evaluation metrics (MSE, MAE, RMSE, R2_OOS, r)
│   └── tvp_var_dynamic_features.csv # Dynamic TVP-VAR TSI and net spillover features
├── pages/
│   ├── 1_Spillover.py       # Dynamic TSI time series and directional spillover tables
│   ├── 2_Forecast.py        # Interactive forecast inspection and range metrics
│   ├── 3_Model_Comparison.py # Model rankings and Diebold-Mariano hypothesis tests
│   └── 4_Limitations.py     # Critical review of baseline dominance and data caveats
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
