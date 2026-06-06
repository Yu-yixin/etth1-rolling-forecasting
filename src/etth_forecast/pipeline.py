from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path

import pandas as pd

from .config import ExperimentConfig, FEATURE_COLUMNS
from .evaluation import build_prediction_frame, compute_metrics, compute_per_variable_metrics, save_prediction_frame
from .models import LSTMForecaster, RepeatLastValueBaseline
from .preprocessing import build_scaler, load_raw_dataframe, save_processed_outputs
from .reporting import write_report
from .training import predict_with_model, train_lstm_model
from .utils import save_json, set_global_seed
from .visualization import render_monthly_figures
from .windowing import build_window_metadata, extract_window_arrays, list_prediction_months, maybe_limit_windows, split_window_metadata_for_month


LOGGER = logging.getLogger(__name__)


def _build_monthly_metric_rows(
    model_name: str,
    prediction_month: str,
    metrics: dict[str, float],
    window_count: int,
) -> dict:
    return {
        "model": model_name,
        "prediction_month": prediction_month,
        "window_count": window_count,
        "mse": metrics["mse"],
        "mae": metrics["mae"],
    }


def _build_variable_metric_rows(
    model_name: str,
    prediction_month: str,
    per_variable_metrics: dict[str, dict[str, float]],
) -> list[dict]:
    rows = []
    for variable, metrics in per_variable_metrics.items():
        rows.append(
            {
                "model": model_name,
                "prediction_month": prediction_month,
                "variable": variable,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
            }
        )
    return rows


def _write_audits(
    config: ExperimentConfig,
    scaled_frame: pd.DataFrame,
    window_metadata: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    per_variable_metrics: pd.DataFrame,
    figure_paths: list[str],
) -> None:
    summary = {
        "round_01_task_consistency": {
            "input_length": config.window.input_length,
            "prediction_length": config.window.prediction_length,
            "feature_count": len(FEATURE_COLUMNS),
            "first_prediction_month": config.first_prediction_month[:7],
        },
        "round_02_standardization_audit": {
            "normalization_start": config.normalization_start,
            "normalization_end": config.normalization_end,
            "scaled_rows": int(len(scaled_frame)),
        },
        "round_03_leakage_audit": {
            "first_train_window_end": str(window_metadata.iloc[0]["target_end_time"]),
            "last_window_end": str(window_metadata.iloc[-1]["target_end_time"]),
        },
        "round_04_shape_audit": {
            "input_shape": [config.window.input_length, len(FEATURE_COLUMNS)],
            "target_shape": [config.window.prediction_length, len(FEATURE_COLUMNS)],
        },
        "round_05_baseline_audit": {
            "baseline_month_count": int((monthly_metrics["model"] == "baseline").sum()),
        },
        "round_06_lstm_audit": {
            "lstm_month_count": int((monthly_metrics["model"] == "lstm").sum()),
        },
        "round_07_rolling_evaluation_audit": {
            "monthly_metric_rows": int(len(monthly_metrics)),
        },
        "round_08_visualization_audit": {
            "figure_count": len(figure_paths),
        },
        "round_09_report_audit": {
            "report_path": str(config.report_path),
        },
        "round_10_final_delivery_audit": {
            "per_variable_metric_rows": int(len(per_variable_metrics)),
        },
    }

    for round_key, payload in summary.items():
        round_number = round_key.split("_")[1]
        content = [f"# Audit Round {round_number}", ""]
        for key, value in payload.items():
            content.append(f"- {key}: {value}")
        (config.audits_dir / f"audit_round_{round_number}.md").write_text(
            "\n".join(content) + "\n",
            encoding="utf-8",
        )


def _write_final_closeout(config: ExperimentConfig, summary_metrics: pd.DataFrame, figure_paths: list[str]) -> None:
    final_closeout_path = config.outputs_dir / "final_closeout.md"
    lines = [
        "# Final Closeout",
        "",
        "## Delivery Checklist",
        "",
        f"- report_exists: {config.report_path.exists()}",
        f"- figure_count: {len(figure_paths)}",
        f"- monthly_metrics_exists: {(config.metrics_dir / 'monthly_metrics.csv').exists()}",
        f"- per_variable_metrics_exists: {(config.metrics_dir / 'per_variable_metrics.csv').exists()}",
        f"- summary_metrics_exists: {(config.metrics_dir / 'summary_metrics.csv').exists()}",
    ]
    final_closeout_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_experiment(config: ExperimentConfig) -> dict:
    config.ensure_directories()
    set_global_seed(config.seed)
    LOGGER.info("Loading raw data from %s", config.raw_data_path)

    raw_frame = load_raw_dataframe(config.raw_data_path)
    LOGGER.info(
        "Loaded %d rows spanning %s to %s",
        len(raw_frame),
        raw_frame["date"].min(),
        raw_frame["date"].max(),
    )
    scaler = build_scaler(
        raw_frame,
        normalization_start=config.normalization_start,
        normalization_end=config.normalization_end,
    )
    scaled_frame = scaler.transform(raw_frame)
    save_processed_outputs(raw_frame, scaled_frame, scaler, config.processed_dir)
    LOGGER.info("Saved processed data and standardization parameters to %s", config.processed_dir)

    window_metadata = build_window_metadata(scaled_frame, config.window)
    prediction_months = list_prediction_months(
        window_metadata,
        first_prediction_month=config.first_prediction_month,
        max_prediction_months=config.max_prediction_months,
    )
    LOGGER.info(
        "Built %d rolling windows; forecasting %d month(s): %s",
        len(window_metadata),
        len(prediction_months),
        ", ".join(prediction_months),
    )
    save_json(
        {
            "config": asdict(config),
            "prediction_months": prediction_months,
        },
        config.outputs_dir / "experiment_config.json",
    )

    monthly_metric_rows: list[dict] = []
    per_variable_metric_rows: list[dict] = []
    figure_paths: list[str] = []
    prediction_frames: list[pd.DataFrame] = []

    for prediction_month in prediction_months:
        LOGGER.info("=== Processing month %s ===", prediction_month)
        train_meta, eval_meta = split_window_metadata_for_month(window_metadata, prediction_month)
        train_meta = maybe_limit_windows(train_meta, config.max_train_windows)
        eval_meta = maybe_limit_windows(eval_meta, config.max_eval_windows_per_month)
        if eval_meta.empty:
            LOGGER.info("Skipping month %s because no evaluation windows are available", prediction_month)
            continue

        train_inputs, train_targets = extract_window_arrays(scaled_frame, train_meta, config.window)
        eval_inputs, eval_targets = extract_window_arrays(scaled_frame, eval_meta, config.window)
        LOGGER.info(
            "[%s] train_windows=%d eval_windows=%d",
            prediction_month,
            len(train_meta),
            len(eval_meta),
        )

        baseline = RepeatLastValueBaseline(config.window.prediction_length)
        LOGGER.info("[%s][baseline] generating predictions", prediction_month)
        baseline_predictions = baseline.predict(eval_inputs)
        baseline_metrics = compute_metrics(eval_targets, baseline_predictions)
        baseline_variable_metrics = compute_per_variable_metrics(eval_targets, baseline_predictions)
        monthly_metric_rows.append(
            _build_monthly_metric_rows("baseline", prediction_month, baseline_metrics, len(eval_meta))
        )
        per_variable_metric_rows.extend(
            _build_variable_metric_rows("baseline", prediction_month, baseline_variable_metrics)
        )
        baseline_frame = build_prediction_frame(
            eval_meta,
            eval_targets,
            baseline_predictions,
            model_name="baseline",
            prediction_month=prediction_month,
        )
        save_prediction_frame(
            baseline_frame,
            config.predictions_dir / "baseline" / f"{prediction_month}.parquet",
        )
        LOGGER.info(
            "[%s][baseline] mse=%.6f mae=%.6f saved predictions to %s",
            prediction_month,
            baseline_metrics["mse"],
            baseline_metrics["mae"],
            config.predictions_dir / "baseline" / f"{prediction_month}.parquet",
        )
        prediction_frames.append(baseline_frame)
        figure_paths.extend(
            render_monthly_figures(
                frame=scaled_frame,
                metadata=eval_meta,
                inputs=eval_inputs,
                y_true=eval_targets,
                y_pred=baseline_predictions,
                model_name="baseline",
                prediction_month=prediction_month,
                output_dir=config.figures_dir / "baseline" / prediction_month,
                spec=config.window,
                sample_count=config.sample_plots_per_month,
            )
        )
        LOGGER.info("[%s][baseline] saved figures to %s", prediction_month, config.figures_dir / "baseline" / prediction_month)

        lstm_model = LSTMForecaster(
            hidden_size=config.training.hidden_size,
            num_layers=config.training.num_layers,
            prediction_length=config.window.prediction_length,
            dropout=config.training.dropout,
        )
        LOGGER.info("[%s][lstm] training started", prediction_month)
        train_info = train_lstm_model(
            model=lstm_model,
            train_inputs=train_inputs,
            train_targets=train_targets,
            config=config.training,
            checkpoint_path=config.checkpoints_dir / f"lstm_{prediction_month}.pt",
            progress_label=f"{prediction_month}/lstm",
        )
        LOGGER.info("[%s][lstm] training finished; running prediction", prediction_month)
        lstm_predictions = predict_with_model(
            lstm_model,
            eval_inputs,
            batch_size=config.training.batch_size,
        )
        lstm_metrics = compute_metrics(eval_targets, lstm_predictions)
        lstm_variable_metrics = compute_per_variable_metrics(eval_targets, lstm_predictions)
        monthly_metric_rows.append(
            _build_monthly_metric_rows("lstm", prediction_month, lstm_metrics, len(eval_meta))
        )
        per_variable_metric_rows.extend(
            _build_variable_metric_rows("lstm", prediction_month, lstm_variable_metrics)
        )
        lstm_frame = build_prediction_frame(
            eval_meta,
            eval_targets,
            lstm_predictions,
            model_name="lstm",
            prediction_month=prediction_month,
        )
        lstm_frame["best_val_loss"] = train_info["best_val_loss"]
        save_prediction_frame(
            lstm_frame,
            config.predictions_dir / "lstm" / f"{prediction_month}.parquet",
        )
        LOGGER.info(
            "[%s][lstm] mse=%.6f mae=%.6f best_val_loss=%.6f saved predictions to %s",
            prediction_month,
            lstm_metrics["mse"],
            lstm_metrics["mae"],
            train_info["best_val_loss"],
            config.predictions_dir / "lstm" / f"{prediction_month}.parquet",
        )
        prediction_frames.append(lstm_frame)
        figure_paths.extend(
            render_monthly_figures(
                frame=scaled_frame,
                metadata=eval_meta,
                inputs=eval_inputs,
                y_true=eval_targets,
                y_pred=lstm_predictions,
                model_name="lstm",
                prediction_month=prediction_month,
                output_dir=config.figures_dir / "lstm" / prediction_month,
                spec=config.window,
                sample_count=config.sample_plots_per_month,
            )
        )
        LOGGER.info("[%s][lstm] saved figures to %s", prediction_month, config.figures_dir / "lstm" / prediction_month)

    monthly_metrics = pd.DataFrame(monthly_metric_rows).sort_values(
        ["model", "prediction_month"]
    ).reset_index(drop=True)
    per_variable_metrics = pd.DataFrame(per_variable_metric_rows).sort_values(
        ["model", "prediction_month", "variable"]
    ).reset_index(drop=True)
    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    summary_rows = []
    variable_summary_rows = []
    for model_name, model_frame in combined_predictions.groupby("model"):
        metrics = compute_metrics(
            model_frame["y_true"].to_numpy(),
            model_frame["y_pred"].to_numpy(),
        )
        summary_rows.append({"model": model_name, **metrics})
        for variable, variable_frame in model_frame.groupby("variable"):
            var_metrics = compute_metrics(
                variable_frame["y_true"].to_numpy(),
                variable_frame["y_pred"].to_numpy(),
            )
            variable_summary_rows.append(
                {
                    "model": model_name,
                    "variable": variable,
                    **var_metrics,
                }
            )
    summary_metrics = pd.DataFrame(summary_rows).sort_values("model").reset_index(drop=True)
    per_variable_summary_metrics = (
        pd.DataFrame(variable_summary_rows)
        .sort_values(["model", "variable"])
        .reset_index(drop=True)
    )
    monthly_pivot = monthly_metrics.pivot(index="prediction_month", columns="model", values="mse")
    worse_months = (
        monthly_pivot.index[monthly_pivot["lstm"] > monthly_pivot["baseline"]].tolist()
        if {"baseline", "lstm"}.issubset(monthly_pivot.columns)
        else []
    )
    variable_pivot = per_variable_summary_metrics.pivot(index="variable", columns="model", values="mse")
    worse_variables = (
        variable_pivot.index[variable_pivot["lstm"] > variable_pivot["baseline"]].tolist()
        if {"baseline", "lstm"}.issubset(variable_pivot.columns)
        else []
    )

    monthly_metrics.to_csv(config.metrics_dir / "monthly_metrics.csv", index=False)
    per_variable_metrics.to_csv(config.metrics_dir / "per_variable_metrics.csv", index=False)
    summary_metrics.to_csv(config.metrics_dir / "summary_metrics.csv", index=False)
    per_variable_summary_metrics.to_csv(
        config.metrics_dir / "per_variable_summary_metrics.csv",
        index=False,
    )

    write_report(
        report_path=config.report_path,
        summary_metrics=summary_metrics,
        monthly_metrics=monthly_metrics,
        per_variable_summary_metrics=per_variable_summary_metrics,
        figure_paths=figure_paths,
        training_config=asdict(config.training),
        total_figure_count=len(figure_paths),
        worse_months=worse_months,
        worse_variables=worse_variables,
    )
    _write_audits(
        config=config,
        scaled_frame=scaled_frame,
        window_metadata=window_metadata,
        monthly_metrics=monthly_metrics,
        per_variable_metrics=per_variable_metrics,
        figure_paths=figure_paths,
    )
    _write_final_closeout(config, summary_metrics, figure_paths)
    LOGGER.info("Saved summary metrics to %s", config.metrics_dir / "summary_metrics.csv")
    LOGGER.info("Saved report to %s", config.report_path)
    LOGGER.info("Experiment complete")
    return {
        "monthly_metrics": monthly_metrics,
        "per_variable_metrics": per_variable_metrics,
        "per_variable_summary_metrics": per_variable_summary_metrics,
        "summary_metrics": summary_metrics,
        "figure_paths": figure_paths,
    }
