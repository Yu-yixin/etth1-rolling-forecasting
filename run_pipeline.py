from __future__ import annotations

import argparse
import logging
from pathlib import Path

from etth_forecast.config import ExperimentConfig
from etth_forecast.pipeline import run_experiment


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ETTh1 rolling forecasting pipeline.")
    parser.add_argument("--project-root", default=Path(__file__).resolve().parent, type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-eval-windows-per-month", type=int, default=None)
    parser.add_argument("--max-prediction-months", type=int, default=None)
    parser.add_argument("--sample-plots-per-month", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    return parser


def main() -> None:
    configure_logging()
    parser = build_argument_parser()
    args = parser.parse_args()

    config = ExperimentConfig(project_root=args.project_root)
    config.seed = args.seed
    config.sample_plots_per_month = args.sample_plots_per_month
    config.max_train_windows = args.max_train_windows
    config.max_eval_windows_per_month = args.max_eval_windows_per_month
    config.max_prediction_months = args.max_prediction_months
    config.training.epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.patience = args.patience
    config.training.hidden_size = args.hidden_size
    config.training.torch_num_threads = args.torch_num_threads

    run_experiment(config)


if __name__ == "__main__":
    main()
