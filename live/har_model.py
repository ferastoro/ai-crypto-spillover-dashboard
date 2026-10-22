"""
HAR (Heterogeneous Autoregressive) Model for BTC-IDR Realized Volatility.
Self-contained implementation using numpy, pandas, and scikit-learn.
Strictly replicates RSIC Checkpoint 2g / v5 / v6 benchmark methodology.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def compute_har_components(df: pd.DataFrame, r2_col: str = "r2") -> pd.DataFrame:
    """
    Computes standard HAR daily, weekly, and monthly realized volatility components:
      - rv1: daily realized variance (lag 1)
      - rv5: weekly realized variance (5-day rolling average)
      - rv22: monthly realized variance (22-day rolling average)
    """
    df_out = df.copy()
    r2_series = df_out[r2_col]
    df_out["rv1"] = r2_series
    df_out["rv5"] = r2_series.rolling(window=5, min_periods=1).mean()
    df_out["rv22"] = r2_series.rolling(window=22, min_periods=1).mean()
    return df_out


def fit_predict_har_step(
    df_trade: pd.DataFrame,
    origin_idx: int,
    h: int,
    window_size: int = 1174,
    r2_col: str = "r2",
    feat_cols: Optional[list] = None
) -> float:
    """
    Fits standard OLS HAR on rolling window of 1174 observation pairs and predicts h-step ahead.
    
    Training alignment:
      - At origin t (origin_idx), target to predict is y_{t+h}.
      - Latest known target during training is y_{(t-h)+h} = y_t.
      - Training indices tau run from (t - h - window_size + 1) to (t - h).
      - X_train is at tau; Y_train is at tau + h.
      - Test regressor x_test is at t.
    """
    if feat_cols is None:
        feat_cols = ["rv1", "rv5", "rv22"]

    train_s = np.arange(origin_idx - h - window_size + 1, origin_idx - h + 1)
    train_s = train_s[train_s >= 0]
    
    if len(train_s) < 100:
        raise ValueError(f"Insufficient training observations: {len(train_s)} (need >= 100)")

    X_train = df_trade.loc[train_s, feat_cols].values
    Y_train = df_trade.loc[train_s + h, r2_col].values
    
    valid_mask = ~(np.isnan(X_train).any(axis=1) | np.isnan(Y_train))
    X_train_clean = X_train[valid_mask]
    Y_train_clean = Y_train[valid_mask]

    model = LinearRegression(fit_intercept=True)
    model.fit(X_train_clean, Y_train_clean)

    x_test = df_trade.loc[[origin_idx], feat_cols].values
    pred = float(model.predict(x_test)[0])
    return pred


def forecast_latest(
    df_trade: pd.DataFrame,
    h: int,
    window_size: int = 1174
) -> float:
    """
    Generates out-of-sample forecast for the latest observation in df_trade.
    """
    df_feat = compute_har_components(df_trade)
    latest_idx = len(df_feat) - 1
    return fit_predict_har_step(df_feat, origin_idx=latest_idx, h=h, window_size=window_size)


def verify_historical_replication(
    history_file: str,
    cv_file: str,
    har1_ref_file: str,
    har10_ref_file: str,
    window_size: int = 1174,
    tolerance: float = 1e-6
) -> Dict[str, Any]:
    """
    Re-runs the exact 335 cutoff evaluations against historical benchmarks and validates target MSEs.
      - Target h=1: 2.2346806e-06
      - Target h=10: 2.4397431e-06
    """
    df_raw = pd.read_csv(history_file)
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    
    # Filter to trading days
    if "is_trade_day" in df_raw.columns:
        df_trade = df_raw[df_raw["is_trade_day"] == 1].copy().reset_index(drop=True)
    else:
        df_trade = df_raw.dropna(subset=["r2"]).copy().reset_index(drop=True)
        
    df_trade = compute_har_components(df_trade)

    cv_df = pd.read_csv(cv_file)
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    cv_df["real_date"] = pd.to_datetime(cv_df["real_date"])

    hist_series = pd.Series(df_trade["r2"].values, index=df_trade["Date"])
    cutoff_obs_date = {}
    cutoffs = cv_df["cutoff"].unique()
    for c in cutoffs:
        h1_real_date = cv_df.loc[cv_df["cutoff"] == c, "real_date"].min()
        past_dates = hist_series.index[hist_series.index < h1_real_date]
        if len(past_dates) > 0:
            cutoff_obs_date[c] = past_dates.max()

    ref_dfs = {
        1: (pd.read_csv(har1_ref_file), 2.2346806e-06),
        10: (pd.read_csv(har10_ref_file), 2.4397431e-06)
    }

    results = {}
    for h, (ref_df, target_mse) in ref_dfs.items():
        preds = []
        trues = []
        for c in cutoffs:
            c_obs = cutoff_obs_date.get(c)
            origin_idx = df_trade[df_trade["Date"] == c_obs].index[0]
            target_idx = origin_idx + h
            trues.append(df_trade.loc[target_idx, "r2"])
            
            p = fit_predict_har_step(df_trade, origin_idx=origin_idx, h=h, window_size=window_size)
            preds.append(p)

        actual_mse = mean_squared_error(trues, preds)
        rel_diff = abs(actual_mse - target_mse) / target_mse
        max_abs_diff = float(np.abs(np.array(preds) - ref_df["HAR"].values).max())
        passed = bool(rel_diff < tolerance)

        results[f"h={h}"] = {
            "actual_mse": actual_mse,
            "target_mse": target_mse,
            "rel_diff": rel_diff,
            "max_abs_diff": max_abs_diff,
            "passed": passed
        }
        
    return results


if __name__ == "__main__":
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    hist_p = str(base / "data" / "live" / "btcidr_history.csv")
    cv_p = str(base / "data" / "forecasts_v6.csv")  # can also use forecasts_v6 or CV
    # Let's verify with the official files
    print("har_model.py ready.")
