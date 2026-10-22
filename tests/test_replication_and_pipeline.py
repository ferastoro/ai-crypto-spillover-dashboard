"""
Comprehensive test suite verifying historical replication and pipeline consistency.
Validates:
1. Exact replication of all 335 test set cutoffs against thesis benchmarks.
2. Machine-precision match of the final historical cutoff (origin 2026-08-21).
3. Truncation test at 2026-09-04 ensuring zero definition drift in calendar and HAR estimation.
"""

import os
import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from live.har_model import (
    compute_har_components,
    fit_predict_har_step,
    verify_historical_replication
)
from live.update_forecast import (
    NYSEHolidayCalendar,
    is_us_trading_day,
    get_future_trading_date
)


def test_historical_replication_335_cutoffs():
    """
    Validates exact MSE targets on 335 cutoffs from RSIC thesis:
      h=1: 2.2346806e-06
      h=10: 2.4397431e-06
    """
    hist_file = str(BASE_DIR / "data" / "live" / "btcidr_history.csv")
    cv_file = str(BASE_DIR / "data" / "forecasts_v6.csv")
    rsic_dir_env = os.getenv("RSIC_DIR", "/Users/admin/Developer/coding_backup/SEM 5/RSIC/rsic-v5-inputs")
    rsic_dir = Path(rsic_dir_env)

    if not rsic_dir.exists():
        pytest.skip("RSIC directory not mounted, skipping cross-repo reference verification")

    har1_ref = str(rsic_dir / "har_preds_T0_h1_v2.csv")
    har10_ref = str(rsic_dir / "har_preds_T0_h10_v2.csv")

    res = verify_historical_replication(hist_file, cv_file, har1_ref, har10_ref, tolerance=1e-6)

    assert res["h=1"]["passed"], f"h=1 replication failed: {res['h=1']}"
    assert res["h=10"]["passed"], f"h=10 replication failed: {res['h=10']}"
    assert res["h=1"]["rel_diff"] < 1e-6
    assert res["h=10"]["rel_diff"] < 1e-6
    print(f"\n[PASSED] 335 Cutoff Replication: h=1 rel_diff={res['h=1']['rel_diff']:.2e}, h=10 rel_diff={res['h=10']['rel_diff']:.2e}")


def test_final_cutoff_machine_precision():
    """
    Validates cutoff index 334 (origin 2026-08-21) against har_preds_T0_v2.csv last row.
    """
    rsic_dir_env = os.getenv("RSIC_DIR", "/Users/admin/Developer/coding_backup/SEM 5/RSIC/rsic-v5-inputs")
    rsic_dir = Path(rsic_dir_env)
    if not rsic_dir.exists():
        pytest.skip("RSIC directory not mounted")

    df_h1_ref = pd.read_csv(rsic_dir / "har_preds_T0_h1_v2.csv")
    df_h10_ref = pd.read_csv(rsic_dir / "har_preds_T0_h10_v2.csv")

    last_h1 = df_h1_ref.iloc[-1].to_dict()
    last_h10 = df_h10_ref.iloc[-1].to_dict()

    hist_df = pd.read_csv(BASE_DIR / "data" / "live" / "btcidr_history.csv")
    hist_trade = hist_df[(hist_df["is_trade_day"] == 1) & hist_df["r2"].notna()].copy().reset_index(drop=True)
    df_feat = compute_har_components(hist_trade)

    origin_idx = df_feat[df_feat["Date"] == "2026-08-21"].index[0]

    p1 = fit_predict_har_step(df_feat, origin_idx=origin_idx, h=1, window_size=1174)
    p10 = fit_predict_har_step(df_feat, origin_idx=origin_idx, h=10, window_size=1174)

    past_r2 = df_feat.loc[:origin_idx, "r2"]
    r20 = float(past_r2.tail(20).mean())

    # Assert matching down to 1e-10 relative error
    np.testing.assert_allclose(p1, last_h1["HAR"], rtol=1e-10)
    np.testing.assert_allclose(p10, last_h10["HAR"], rtol=1e-10)
    np.testing.assert_allclose(r20, last_h1["Roll_20"], rtol=1e-10)
    np.testing.assert_allclose(df_feat.loc[origin_idx + 1, "r2"], last_h1["True"], rtol=1e-10)
    np.testing.assert_allclose(df_feat.loc[origin_idx + 10, "r2"], last_h10["True"], rtol=1e-10)

    print(f"\n[PASSED] Final Cutoff (2026-08-21) match: p1 diff={abs(p1 - last_h1['HAR']):.2e}, p10 diff={abs(p10 - last_h10['HAR']):.2e}")


def test_pipeline_truncated_at_2026_09_04():
    """
    Truncates data at 2026-09-04 and executes forecast generation.
    Confirms:
      - Origin is 2026-09-04.
      - h=1 target date advances to 2026-09-08 (Tuesday, skipping weekend and Labor Day 2026-09-07).
      - h=10 target date advances to 2026-09-21 (10 trading days).
      - Predictions match the first backfill row in latest_forecast.csv.
    """
    hist_df = pd.read_csv(BASE_DIR / "data" / "live" / "btcidr_history.csv")
    # Truncate at 2026-09-04
    hist_trunc = hist_df[hist_df["Date"] <= "2026-09-04"].copy()

    trade_trunc = hist_trunc[(hist_trunc["is_trade_day"] == 1) & hist_trunc["r2"].notna()].copy().reset_index(drop=True)
    assert len(trade_trunc) == 1677, f"Expected 1677 trading days at 2026-09-04, got {len(trade_trunc)}"

    df_feat = compute_har_components(trade_trunc)
    last_idx = len(df_feat) - 1
    as_of_dt = df_feat.loc[last_idx, "Date"]
    assert as_of_dt == "2026-09-04"

    # Future trading dates from 2026-09-04
    t_dt = pd.to_datetime(as_of_dt)
    target_h1 = get_future_trading_date(t_dt, 1)
    target_h10 = get_future_trading_date(t_dt, 10)

    assert target_h1 == "2026-09-08", f"Expected 2026-09-08 for h=1, got {target_h1}"
    assert target_h10 == "2026-09-21", f"Expected 2026-09-21 for h=10, got {target_h10}"

    fc_1 = fit_predict_har_step(df_feat, origin_idx=last_idx, h=1, window_size=1174)
    fc_10 = fit_predict_har_step(df_feat, origin_idx=last_idx, h=10, window_size=1174)

    # Check against generated backfill row in latest_forecast.csv
    fc_csv = pd.read_csv(BASE_DIR / "data" / "live" / "latest_forecast.csv")
    row_h1 = fc_csv[(fc_csv["as_of_date"] == "2026-09-04") & (fc_csv["horizon"] == 1)].iloc[0]
    row_h10 = fc_csv[(fc_csv["as_of_date"] == "2026-09-04") & (fc_csv["horizon"] == 10)].iloc[0]

    np.testing.assert_allclose(fc_1, row_h1["forecast"], rtol=1e-12)
    np.testing.assert_allclose(fc_10, row_h10["forecast"], rtol=1e-12)
    assert row_h1["target_date"] == "2026-09-08"
    assert row_h10["target_date"] == "2026-09-21"

    print(f"\n[PASSED] Truncation test at 2026-09-04: h=1 -> {target_h1} ({fc_1:.6e}), h=10 -> {target_h10} ({fc_10:.6e})")


if __name__ == "__main__":
    test_historical_replication_335_cutoffs()
    test_final_cutoff_machine_precision()
    test_pipeline_truncated_at_2026_09_04()
    print("\nAll historical and pipeline tests passed successfully!")
