from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
DATE_COLUMN = "date"
NORMALIZATION_START = "2016-07-01 00:00:00"
NORMALIZATION_END = "2017-06-30 23:00:00"
FIRST_PREDICTION_MONTH = "2017-07-01 00:00:00"
DEFAULT_SEED = 2026


@dataclass(frozen=True)
class WindowSpec:
    input_length: int = 336
    prediction_length: int = 24


@dataclass
class TrainingConfig:
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 5
    patience: int = 2
    validation_fraction: float = 0.1
    min_validation_windows: int = 64
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    torch_num_threads: int = 1


@dataclass
class ExperimentConfig:
    project_root: Path
    raw_data_path: Path = field(init=False)
    processed_dir: Path = field(init=False)
    outputs_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)
    audits_dir: Path = field(init=False)
    report_path: Path = field(init=False)
    final_closeout_path: Path = field(init=False)
    window: WindowSpec = field(default_factory=WindowSpec)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    seed: int = DEFAULT_SEED
    normalization_start: str = NORMALIZATION_START
    normalization_end: str = NORMALIZATION_END
    first_prediction_month: str = FIRST_PREDICTION_MONTH
    sample_plots_per_month: int = 1
    max_train_windows: int | None = None
    max_eval_windows_per_month: int | None = None
    max_prediction_months: int | None = None

    def __post_init__(self) -> None:
        self.raw_data_path = self.project_root / "data" / "raw" / "ETTh1.csv"
        self.processed_dir = self.project_root / "data" / "processed"
        self.outputs_dir = self.project_root / "outputs"
        self.figures_dir = self.outputs_dir / "figures"
        self.metrics_dir = self.outputs_dir / "metrics"
        self.predictions_dir = self.outputs_dir / "predictions"
        self.checkpoints_dir = self.outputs_dir / "checkpoints"
        self.audits_dir = self.outputs_dir / "audits"
        self.report_path = self.project_root / "report.md"
        self.final_closeout_path = self.outputs_dir / "final_closeout.md"

    def ensure_directories(self) -> None:
        for path in [
            self.processed_dir,
            self.outputs_dir,
            self.figures_dir,
            self.metrics_dir,
            self.predictions_dir,
            self.checkpoints_dir,
            self.audits_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
