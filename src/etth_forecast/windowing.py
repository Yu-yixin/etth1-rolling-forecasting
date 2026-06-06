from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .config import DATE_COLUMN, FEATURE_COLUMNS, WindowSpec


@dataclass(frozen=True)
class WindowMetadata:
    frame: pd.DataFrame


class ArrayWindowDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = torch.as_tensor(inputs, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.targets[index]


def build_window_metadata(frame: pd.DataFrame, spec: WindowSpec) -> pd.DataFrame:
    max_start = len(frame) - spec.input_length - spec.prediction_length + 1
    rows: list[dict] = []
    for start_idx in range(max_start):
        input_start_idx = start_idx
        input_end_idx = start_idx + spec.input_length - 1
        target_start_idx = input_end_idx + 1
        target_end_idx = target_start_idx + spec.prediction_length - 1
        rows.append(
            {
                "window_start_idx": start_idx,
                "input_start_idx": input_start_idx,
                "input_end_idx": input_end_idx,
                "target_start_idx": target_start_idx,
                "target_end_idx": target_end_idx,
                "input_start_time": frame.iloc[input_start_idx][DATE_COLUMN],
                "input_end_time": frame.iloc[input_end_idx][DATE_COLUMN],
                "target_start_time": frame.iloc[target_start_idx][DATE_COLUMN],
                "target_end_time": frame.iloc[target_end_idx][DATE_COLUMN],
            }
        )
    return pd.DataFrame(rows)


def month_key(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%Y-%m")


def list_prediction_months(
    metadata: pd.DataFrame,
    first_prediction_month: str,
    max_prediction_months: int | None = None,
) -> list[str]:
    first_month = pd.Timestamp(first_prediction_month).strftime("%Y-%m")
    months = sorted(
        month
        for month in metadata["target_start_time"].dt.strftime("%Y-%m").unique().tolist()
        if month >= first_month
    )
    if max_prediction_months is not None:
        months = months[:max_prediction_months]
    return months


def split_window_metadata_for_month(
    metadata: pd.DataFrame,
    prediction_month: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    month_start = pd.Timestamp(f"{prediction_month}-01 00:00:00")
    month_end_exclusive = month_start + pd.offsets.MonthBegin(1)

    train_mask = metadata["target_end_time"] < month_start
    eval_mask = (
        (metadata["target_start_time"] >= month_start)
        & (metadata["target_end_time"] < month_end_exclusive)
    )

    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    eval_meta = metadata.loc[eval_mask].reset_index(drop=True)
    return train_meta, eval_meta


def maybe_limit_windows(metadata: pd.DataFrame, max_windows: int | None) -> pd.DataFrame:
    if max_windows is None or len(metadata) <= max_windows:
        return metadata
    return metadata.iloc[-max_windows:].reset_index(drop=True)


def extract_window_arrays(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    spec: WindowSpec,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = []
    targets = []
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    for row in metadata.itertuples(index=False):
        start_idx = row.window_start_idx
        inputs.append(values[start_idx : start_idx + spec.input_length])
        targets.append(
            values[
                start_idx + spec.input_length : start_idx + spec.input_length + spec.prediction_length
            ]
        )
    if not inputs:
        return (
            np.empty((0, spec.input_length, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0, spec.prediction_length, len(FEATURE_COLUMNS)), dtype=np.float32),
        )
    return np.stack(inputs), np.stack(targets)
