from __future__ import annotations

import numpy as np
import pandas as pd

from etth_forecast.config import FEATURE_COLUMNS, WindowSpec
from etth_forecast.windowing import (
    build_window_metadata,
    extract_window_arrays,
    split_window_metadata_for_month,
)


def test_window_shapes_and_order():
    dates = pd.date_range("2020-01-01 00:00:00", periods=12, freq="h")
    frame = pd.DataFrame({"date": dates})
    for index, column in enumerate(FEATURE_COLUMNS):
        frame[column] = np.arange(len(frame), dtype=float) + index

    spec = WindowSpec(input_length=4, prediction_length=2)
    metadata = build_window_metadata(frame, spec)
    inputs, targets = extract_window_arrays(frame, metadata.head(1), spec)

    assert inputs.shape == (1, 4, 7)
    assert targets.shape == (1, 2, 7)
    assert metadata.iloc[0]["input_end_time"] < metadata.iloc[0]["target_start_time"]


def test_monthly_split_has_no_leakage(project_root):
    frame = pd.read_parquet(project_root / "data" / "processed" / "ETTh1_scaled.parquet")
    metadata = build_window_metadata(frame, WindowSpec())
    train_meta, eval_meta = split_window_metadata_for_month(metadata, "2017-07")

    month_start = pd.Timestamp("2017-07-01 00:00:00")
    month_end = pd.Timestamp("2017-08-01 00:00:00")

    assert (train_meta["target_end_time"] < month_start).all()
    assert (eval_meta["target_start_time"] >= month_start).all()
    assert (eval_meta["target_end_time"] < month_end).all()
