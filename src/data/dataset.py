from __future__ import annotations

from pathlib import Path

from etth_forecast.config import ExperimentConfig
from etth_forecast.preprocessing import load_raw_dataframe
from etth_forecast.windowing import build_window_metadata, extract_window_arrays


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig(project_root=project_root)
    frame = load_raw_dataframe(config.processed_dir / "ETTh1_scaled.parquet")
    metadata = build_window_metadata(frame, config.window)
    inputs, targets = extract_window_arrays(frame, metadata.head(1), config.window)

    print("window count:", len(metadata))
    print("input shape:", inputs.shape)
    print("target shape:", targets.shape)


if __name__ == "__main__":
    main()
