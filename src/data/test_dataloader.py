from __future__ import annotations

from pathlib import Path

from etth_forecast.config import ExperimentConfig
from etth_forecast.preprocessing import load_raw_dataframe
from etth_forecast.windowing import ArrayWindowDataset, build_window_metadata, extract_window_arrays
from torch.utils.data import DataLoader


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    config = ExperimentConfig(project_root=project_root)
    frame = load_raw_dataframe(config.processed_dir / "ETTh1_scaled.parquet")
    metadata = build_window_metadata(frame, config.window).head(128)
    inputs, targets = extract_window_arrays(frame, metadata, config.window)
    loader = DataLoader(ArrayWindowDataset(inputs, targets), batch_size=32, shuffle=False)
    batch_inputs, batch_targets = next(iter(loader))
    print("batch input shape:", tuple(batch_inputs.shape))
    print("batch target shape:", tuple(batch_targets.shape))


if __name__ == "__main__":
    main()
