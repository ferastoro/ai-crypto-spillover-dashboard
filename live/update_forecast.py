"""
Automated live forecast update pipeline for BTC-IDR realized volatility using HAR.
Runs daily to ingest finished Indodax daily candles, append to btcidr_history.csv,
compute HAR out-of-sample forecasts (h=1 and h=10), update realized targets,
and maintain live_scorecard.csv.
"""

import sys
import os
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, List, Dict, Any

import requests
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay,
    USLaborDay, USThanksgivingDay, GoodFriday
)

# Add parent directory to sys.path for relative imports
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from live.har_model import compute_har_components, fit_predict_har_step

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S UTC"
)
logger = logging.getLogger("update_forecast")

# File Paths
DATA_LIVE_DIR = BASE_DIR / "data" / "live"
HISTORY_FILE = DATA_LIVE_DIR / "btcidr_history.csv"
FORECAST_FILE = DATA_LIVE_DIR / "latest_forecast.csv"
SCORECARD_FILE = DATA_LIVE_DIR / "live_scorecard.csv"

# Indodax API configuration
INDODAX_ENDPOINT = "https://indodax.com/tradingview/history_v2"
INDODAX_SYMBOL = "BTCIDR"
INDODAX_TF = "1D"


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """
    NYSE trading holiday calendar matching RSIC dataset_deep_learning / df_har definitions.
    Includes standard US market holidays plus historical special closures (e.g. Jimmy Carter mourning day).
    """
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=lambda dt: dt + pd.Timedelta(days=1) if dt.weekday() == 6 else (dt if dt.weekday() != 5 else pd.NaT)),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('Juneteenth', month=6, day=19, start_date='2022-01-01', observance=nearest_workday),
        Holiday('July4th', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday),
        Holiday('PresMourningCarter', year=2025, month=1, day=9)
    ]


_nyse_cal = NYSEHolidayCalendar()
_nyse_holidays_set = set(_nyse_cal.holidays(start='2020-01-01', end='2035-12-31').dropna().strftime('%Y-%m-%d'))


def is_us_trading_day(dt: pd.Timestamp) -> bool:
    """
    Returns True if dt is a US stock market trading day (Mon-Fri, non-NYSE holiday).
    """
    dt_str = dt.strftime("%Y-%m-%d")
    if dt_str in ("2020-01-01", "2020-01-02"):
        # 2020-01-01 was New Year holiday; 2020-01-02 was baseline t0 day with no prior return
        return False
    return (dt.weekday() < 5) and (dt_str not in _nyse_holidays_set)


def get_future_trading_date(start_dt: pd.Timestamp, n_trade_days: int) -> str:
    """
    Advances exactly n_trade_days US trading days from start_dt (skipping weekends and NYSE holidays).
    """
    curr = start_dt
    added = 0
    while added < n_trade_days:
        curr += timedelta(days=1)
        if is_us_trading_day(curr):
            added += 1
    return curr.strftime("%Y-%m-%d")


def fetch_indodax_completed_candles(start_date: str) -> pd.DataFrame:
    """
    Fetch daily candles from Indodax TradingView API.
    Timezone Convention:
      - Indodax daily candles use 00:00:00 UTC to 23:59:59 UTC boundaries.
      - Any candle starting at today 00:00:00 UTC is still ongoing and is strictly discarded.
    """
    logger.info("Fetching Indodax daily candles since %s...", start_date)
    dt_start = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    t_from = int(dt_start.timestamp())
    t_to = int(datetime.now(timezone.utc).timestamp())

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json"
    }
    params = {
        "symbol": INDODAX_SYMBOL,
        "tf": INDODAX_TF,
        "from": t_from,
        "to": t_to
    }

    try:
        resp = requests.get(INDODAX_ENDPOINT, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error("Failed to connect to Indodax API: %s", e)
        raise RuntimeError(f"Indodax API request failed: {e}")

    rows: List[Dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            t_dt = datetime.fromtimestamp(item["Time"], tz=timezone.utc)
            rows.append({
                "Date": t_dt.strftime("%Y-%m-%d"),
                "Open": float(item["Open"]),
                "High": float(item["High"]),
                "Low": float(item["Low"]),
                "Close": float(item["Close"]),
                "Volume": float(item["Volume"]),
                "timestamp": int(item["Time"])
            })
    elif isinstance(data, dict) and "t" in data and len(data["t"]) > 0:
        times = data["t"]
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        vols = data.get("v", [])
        for i in range(len(times)):
            t_dt = datetime.fromtimestamp(times[i], tz=timezone.utc)
            rows.append({
                "Date": t_dt.strftime("%Y-%m-%d"),
                "Open": float(opens[i]),
                "High": float(highs[i]),
                "Low": float(lows[i]),
                "Close": float(closes[i]),
                "Volume": float(vols[i]),
                "timestamp": int(times[i])
            })
    else:
        logger.error("Invalid or empty response format received from Indodax API.")
        raise ValueError("Indodax API returned invalid/empty candle payload.")

    df = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Discard ongoing / unfinished candle
    today_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    df_completed = df[df["Date"] < today_utc_str].copy()
    logger.info("Discarded ongoing candle for %s (UTC). Completed candles up to: %s",
                today_utc_str, df_completed["Date"].max() if not df_completed.empty else "None")
    return df_completed


def update_history_dataset(df_completed: pd.DataFrame) -> pd.DataFrame:
    """
    Merges newly completed daily candles with existing btcidr_history.csv.
    Verifies date continuity (no gaps), computes 1-day log returns (r_t = ln(Close_t / Close_{t-1})),
    and assigns is_trade_day=1 strictly if US trading day (weekends and NYSE holidays get is_trade_day=0).
    """
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"History file not found at {HISTORY_FILE}")

    df_hist = pd.read_csv(HISTORY_FILE)
    df_hist["Date"] = pd.to_datetime(df_hist["Date"])

    # Convert new candles Date to datetime
    df_new = df_completed.copy()
    df_new["Date"] = pd.to_datetime(df_new["Date"])

    # Combine with existing history, deduplicating by Date
    merged = pd.concat([df_hist, df_new], ignore_index=True)
    merged = merged.drop_duplicates(subset=["Date"], keep="first").sort_values("Date").reset_index(drop=True)

    # Continuity check: ensure no missing calendar days
    min_date = merged["Date"].min()
    max_date = merged["Date"].max()
    expected_calendar = pd.date_range(min_date, max_date, freq="D")
    missing_dates = set(expected_calendar) - set(merged["Date"])
    if missing_dates:
        err_msg = f"Data continuity broken: found {len(missing_dates)} missing dates (e.g. {sorted(list(missing_dates))[:3]})."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # Calculate returns, r2, and is_trade_day
    for i in range(1, len(merged)):
        dt_i = merged.loc[i, "Date"]
        # Update is_trade_day strictly matching US trading days
        merged.loc[i, "is_trade_day"] = 1 if is_us_trading_day(dt_i) else 0

        if pd.isna(merged.loc[i, "ret"]) or pd.isna(merged.loc[i, "r2"]):
            prev_close = merged.loc[i - 1, "Close"]
            curr_close = merged.loc[i, "Close"]
            ret = float(np.log(curr_close / prev_close))
            merged.loc[i, "ret"] = ret
            merged.loc[i, "r2"] = ret ** 2

    # Save updated btcidr_history.csv
    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    merged.to_csv(HISTORY_FILE, index=False)
    logger.info("Successfully updated %s (Total rows: %d, latest date: %s).",
                HISTORY_FILE.name, len(merged), merged["Date"].iloc[-1])
    return merged


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executes the full daily update pipeline:
      1. Ingests Indodax data.
      2. Appends to btcidr_history.csv with is_trade_day aligned to US stock market calendar.
      3. Generates HAR forecasts for h=1 and h=10 (with h in trading days).
      4. Appends to latest_forecast.csv (backfill vs live, with causal Hist_Exp & Roll_20 baselines).
      5. Updates realized y_true targets.
      6. Computes live_scorecard.csv comparing HAR vs Hist_Exp vs Roll_20.
    """
    DATA_LIVE_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load history to check last date
    if not HISTORY_FILE.exists():
        raise FileNotFoundError(f"Base historical dataset missing: {HISTORY_FILE}")

    df_hist = pd.read_csv(HISTORY_FILE)
    last_hist_date = df_hist["Date"].max()
    logger.info("Current historical dataset ends on: %s", last_hist_date)

    # Start 7 days prior to ensure overlap for continuity check
    start_fetch = (pd.to_datetime(last_hist_date) - timedelta(days=7)).strftime("%Y-%m-%d")
    df_completed = fetch_indodax_completed_candles(start_fetch)

    # 2. Update btcidr_history
    updated_hist = update_history_dataset(df_completed)
    updated_hist["Date"] = pd.to_datetime(updated_hist["Date"])

    # 3. Determine trading sequence for HAR model (only is_trade_day == 1)
    df_har_seq = updated_hist[(updated_hist["is_trade_day"] == 1) & updated_hist["r2"].notna()].copy().reset_index(drop=True)
    df_feat = compute_har_components(df_har_seq)

    # 4. Load or initialize latest_forecast.csv
    forecast_cols = [
        "as_of_date", "target_date", "horizon", "forecast",
        "pred_hist_exp", "pred_roll_20", "realized", "generated_at_utc", "kind"
    ]
    if FORECAST_FILE.exists():
        df_forecasts = pd.read_csv(FORECAST_FILE)
        # Ensure new baseline columns exist if migrating
        for col in ["pred_hist_exp", "pred_roll_20"]:
            if col not in df_forecasts.columns:
                df_forecasts[col] = np.nan
    else:
        df_forecasts = pd.DataFrame(columns=forecast_cols)

    # Determine existing pairs
    existing_pairs = set(zip(df_forecasts["as_of_date"].astype(str), df_forecasts["horizon"].astype(int)))
    now_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    thesis_cutoff_date = pd.to_datetime("2026-09-04")
    latest_avail_date = df_feat["Date"].max()

    # Identify trading day index range to forecast (from 2026-09-04 to latest_avail_date)
    start_idx = df_feat[df_feat["Date"] >= thesis_cutoff_date].index[0]

    new_rows: List[Dict[str, Any]] = []

    for idx in range(start_idx, len(df_feat)):
        as_of_dt = df_feat.loc[idx, "Date"]
        as_of_str = as_of_dt.strftime("%Y-%m-%d")

        # Define kind: "live" if it is the latest completed date, else "backfill"
        kind = "live" if as_of_dt == latest_avail_date else "backfill"

        # Compute strictly causal baselines using data up to and including as_of_dt
        past_r2 = df_feat.loc[:idx, "r2"]
        pred_hexp = float(past_r2.mean()) if len(past_r2) > 0 else np.nan
        pred_r20 = float(past_r2.tail(20).mean()) if len(past_r2) >= 20 else np.nan

        for h in [1, 10]:
            if (as_of_str, h) in existing_pairs:
                continue

            target_idx = idx + h
            if target_idx < len(df_feat):
                target_str = df_feat.loc[target_idx, "Date"].strftime("%Y-%m-%d")
            else:
                # Advance h trading days (skipping weekends and NYSE holidays)
                remaining_steps = target_idx - (len(df_feat) - 1)
                last_dt = df_feat["Date"].iloc[-1]
                target_str = get_future_trading_date(last_dt, remaining_steps)

            try:
                fc_val = fit_predict_har_step(df_feat, origin_idx=idx, h=h, window_size=1174)
            except Exception as e:
                logger.error("Forecast failed at %s h=%d: %s", as_of_str, h, e)
                continue

            new_rows.append({
                "as_of_date": as_of_str,
                "target_date": target_str,
                "horizon": h,
                "forecast": fc_val,
                "pred_hist_exp": pred_hexp,
                "pred_roll_20": pred_r20,
                "realized": np.nan,
                "generated_at_utc": now_utc_str,
                "kind": kind
            })

    if new_rows:
        df_new_fc = pd.DataFrame(new_rows)
        df_forecasts = pd.concat([df_forecasts, df_new_fc], ignore_index=True)
        logger.info("Generated %d new forecast rows.", len(new_rows))
    else:
        logger.info("No new forecast origins required generation.")

    # Populate missing pred_hist_exp / pred_roll_20 in existing backfill rows if needed
    for i in range(len(df_forecasts)):
        if pd.isna(df_forecasts.loc[i, "pred_hist_exp"]) or pd.isna(df_forecasts.loc[i, "pred_roll_20"]):
            as_of_str = str(df_forecasts.loc[i, "as_of_date"])
            match_idx = df_feat[df_feat["Date"] == as_of_str].index
            if len(match_idx) > 0:
                pos = match_idx[0]
                past_r2 = df_feat.loc[:pos, "r2"]
                df_forecasts.loc[i, "pred_hist_exp"] = float(past_r2.mean())
                df_forecasts.loc[i, "pred_roll_20"] = float(past_r2.tail(20).mean())

    # 5. Fill realized targets on historical rows where target_date is now observed
    r2_lookup = dict(zip(updated_hist["Date"].dt.strftime("%Y-%m-%d"), updated_hist["r2"]))
    filled_count = 0
    for i in range(len(df_forecasts)):
        if pd.isna(df_forecasts.loc[i, "realized"]):
            t_date = str(df_forecasts.loc[i, "target_date"])
            if t_date in r2_lookup and not pd.isna(r2_lookup[t_date]):
                df_forecasts.loc[i, "realized"] = float(r2_lookup[t_date])
                filled_count += 1

    if filled_count > 0:
        logger.info("Updated realized variance for %d past forecast targets.", filled_count)

    # Save latest_forecast.csv
    df_forecasts = df_forecasts[forecast_cols]
    df_forecasts.to_csv(FORECAST_FILE, index=False)
    logger.info("Saved %s (Total rows: %d).", FORECAST_FILE.name, len(df_forecasts))

    # 6. Update live_scorecard.csv (evaluated on kind == 'live' rows where target has realized)
    scorecard_rows = []
    live_eval = df_forecasts[(df_forecasts["kind"] == "live") & df_forecasts["realized"].notna()].copy()

    for h in [1, 10]:
        sub_h = live_eval[live_eval["horizon"] == h]
        n_eval = len(sub_h)
        if n_eval > 0:
            yt = sub_h["realized"].values
            yp_har = sub_h["forecast"].values
            yp_hexp = sub_h["pred_hist_exp"].values
            yp_roll20 = sub_h["pred_roll_20"].values

            mse_har = float(mean_squared_error(yt, yp_har))
            mse_hist_exp = float(mean_squared_error(yt, yp_hexp))
            mse_roll20 = float(mean_squared_error(yt, yp_roll20))
        else:
            mse_har = np.nan
            mse_hist_exp = np.nan
            mse_roll20 = np.nan

        scorecard_rows.append({
            "horizon": h,
            "n_eval": n_eval,
            "mse_har": mse_har,
            "mse_hist_exp": mse_hist_exp,
            "mse_roll_20": mse_roll20,
            "last_updated_utc": now_utc_str
        })

    df_scorecard = pd.DataFrame(scorecard_rows)
    df_scorecard.to_csv(SCORECARD_FILE, index=False)
    logger.info("Updated live scorecard at %s.", SCORECARD_FILE.name)

    return df_forecasts, df_scorecard


if __name__ == "__main__":
    try:
        run_pipeline()
        print("Live update pipeline executed successfully.")
    except Exception as exc:
        logger.error("Pipeline aborted with error: %s", exc)
        sys.exit(1)
