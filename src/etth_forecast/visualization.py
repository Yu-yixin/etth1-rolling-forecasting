from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, WindowSpec


def render_monthly_figures(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    inputs: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    prediction_month: str,
    output_dir: Path,
    spec: WindowSpec,
    sample_count: int = 1,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(metadata) == 0:
        return []

    chosen_rows = np.linspace(0, len(metadata) - 1, num=min(sample_count, len(metadata)), dtype=int)
    created_paths: list[str] = []

    for sample_rank, row_index in enumerate(chosen_rows, start=1):
        meta = metadata.iloc[row_index]
        history_times = pd.date_range(meta["input_start_time"], periods=spec.input_length, freq="h")
        future_times = pd.date_range(meta["target_start_time"], periods=spec.prediction_length, freq="h")
        for variable_idx, variable in enumerate(FEATURE_COLUMNS):
            fig, axis = plt.subplots(figsize=(12, 4))
            axis.plot(history_times, inputs[row_index, :, variable_idx], label="History", color="#4C6EF5")
            axis.plot(future_times, y_true[row_index, :, variable_idx], label="Ground Truth", color="#2B8A3E")
            axis.plot(future_times, y_pred[row_index, :, variable_idx], label="Prediction", color="#D9480F")
            axis.axvline(future_times[0], color="#212529", linestyle="--", linewidth=1.5, label="Forecast Start")
            axis.set_title(f"{model_name} | {prediction_month} | {variable} | sample-{sample_rank}")
            axis.set_xlabel("Timestamp")
            axis.set_ylabel("Scaled value")
            axis.legend(loc="best")
            axis.grid(alpha=0.2)
            fig.autofmt_xdate()

            file_path = output_dir / f"{variable}_sample_{sample_rank}.png"
            fig.tight_layout()
            fig.savefig(file_path, dpi=150)
            plt.close(fig)
            created_paths.append(str(file_path))
    return created_paths
