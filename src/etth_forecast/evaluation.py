from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    error = y_pred - y_true
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
    }


def compute_per_variable_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, dict[str, float]]:
    metrics = {}
    for index, variable in enumerate(FEATURE_COLUMNS):
        metrics[variable] = compute_metrics(y_true[:, :, index], y_pred[:, :, index])
    return metrics


def build_prediction_frame(
    metadata: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    prediction_month: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for window_index, meta_row in enumerate(metadata.itertuples(index=False)):
        target_times = pd.date_range(
            meta_row.target_start_time,
            periods=y_true.shape[1],
            freq="h",
        )
        for horizon_idx, timestamp in enumerate(target_times):
            for feature_idx, variable in enumerate(FEATURE_COLUMNS):
                rows.append(
                    {
                        "model": model_name,
                        "prediction_month": prediction_month,
                        "window_start_idx": int(meta_row.window_start_idx),
                        "history_start_time": meta_row.input_start_time,
                        "history_end_time": meta_row.input_end_time,
                        "target_time": timestamp,
                        "horizon_step": horizon_idx + 1,
                        "variable": variable,
                        "y_true": float(y_true[window_index, horizon_idx, feature_idx]),
                        "y_pred": float(y_pred[window_index, horizon_idx, feature_idx]),
                    }
                )
    return pd.DataFrame(rows)


def save_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
