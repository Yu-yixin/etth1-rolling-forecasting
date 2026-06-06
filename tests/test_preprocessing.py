from __future__ import annotations

import pandas as pd

from etth_forecast.config import FEATURE_COLUMNS
from etth_forecast.preprocessing import build_scaler, load_raw_dataframe


def test_standardization_uses_only_reference_period(project_root):
    frame = load_raw_dataframe(project_root / "data" / "raw" / "ETTh1.csv")
    scaler = build_scaler(frame, "2016-07-01 00:00:00", "2017-06-30 23:00:00")

    modified = frame.copy()
    modified.loc[modified["date"] >= pd.Timestamp("2017-07-01 00:00:00"), FEATURE_COLUMNS] += 9999.0
    modified_scaler = build_scaler(modified, "2016-07-01 00:00:00", "2017-06-30 23:00:00")

    assert scaler.to_serializable() == modified_scaler.to_serializable()
