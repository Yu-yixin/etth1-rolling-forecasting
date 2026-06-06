from __future__ import annotations

from etth_forecast.config import ExperimentConfig
from etth_forecast.pipeline import run_experiment


def test_smoke_pipeline(tmp_path, project_root):
    config = ExperimentConfig(project_root=project_root)
    config.outputs_dir = tmp_path / "outputs"
    config.figures_dir = config.outputs_dir / "figures"
    config.metrics_dir = config.outputs_dir / "metrics"
    config.predictions_dir = config.outputs_dir / "predictions"
    config.checkpoints_dir = config.outputs_dir / "checkpoints"
    config.audits_dir = config.outputs_dir / "audits"
    config.report_path = tmp_path / "report.md"
    config.processed_dir = tmp_path / "processed"
    config.sample_plots_per_month = 1
    config.max_prediction_months = 1
    config.max_train_windows = 256
    config.max_eval_windows_per_month = 32
    config.training.epochs = 1
    config.training.patience = 1
    config.training.batch_size = 64
    config.training.torch_num_threads = 1

    result = run_experiment(config)

    assert not result["summary_metrics"].empty
    assert (config.metrics_dir / "summary_metrics.csv").exists()
    assert config.report_path.exists()
    assert any(path.endswith(".png") for path in result["figure_paths"])
