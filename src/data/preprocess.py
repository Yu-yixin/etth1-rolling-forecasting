from __future__ import annotations

from pathlib import Path

from etth_forecast.config import ExperimentConfig
from etth_forecast.preprocessing import build_scaler, load_raw_dataframe, save_processed_outputs


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig(project_root=project_root)
    config.ensure_directories()

    raw_frame = load_raw_dataframe(config.raw_data_path)
    scaler = build_scaler(
        raw_frame,
        normalization_start=config.normalization_start,
        normalization_end=config.normalization_end,
    )
    scaled_frame = scaler.transform(raw_frame)
    save_processed_outputs(raw_frame, scaled_frame, scaler, config.processed_dir)


if __name__ == "__main__":
    main()
