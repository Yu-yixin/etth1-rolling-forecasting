from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import DATE_COLUMN, FEATURE_COLUMNS
from .utils import save_json


@dataclass
class ZScoreScaler:
    mean_: pd.Series
    std_: pd.Series

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        scaled = frame.copy()
        scaled[FEATURE_COLUMNS] = (frame[FEATURE_COLUMNS] - self.mean_) / self.std_
        return scaled

    def to_serializable(self) -> dict:
        return {
            "mean": {column: float(self.mean_[column]) for column in FEATURE_COLUMNS},
            "std": {column: float(self.std_[column]) for column in FEATURE_COLUMNS},
        }


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    return frame.sort_values(DATE_COLUMN).reset_index(drop=True)


def build_scaler(
    frame: pd.DataFrame,
    normalization_start: str,
    normalization_end: str,
) -> ZScoreScaler:
    start = pd.Timestamp(normalization_start)
    end = pd.Timestamp(normalization_end)
    mask = (frame[DATE_COLUMN] >= start) & (frame[DATE_COLUMN] <= end)
    reference = frame.loc[mask, FEATURE_COLUMNS]
    if reference.empty:
        raise ValueError("No rows found inside the normalization date range.")
    if len(reference) != 8760:
        raise ValueError(
            f"Expected 8760 hourly rows for normalization, found {len(reference)}."
        )

    std = reference.std(ddof=0)
    if (std == 0).any():
        zero_columns = std[std == 0].index.tolist()
        raise ValueError(f"Zero standard deviation for columns: {zero_columns}")

    return ZScoreScaler(mean_=reference.mean(), std_=std)


def save_processed_outputs(
    raw_frame: pd.DataFrame,
    scaled_frame: pd.DataFrame,
    scaler: ZScoreScaler,
    processed_dir: Path,
) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_frame.to_csv(processed_dir / "ETTh1.csv", index=False)
    scaled_frame.to_parquet(processed_dir / "ETTh1_scaled.parquet", index=False)
    save_json(scaler.to_serializable(), processed_dir / "standardization_params.json")
    joblib.dump(scaler.to_serializable(), processed_dir / "scaler.pkl")
