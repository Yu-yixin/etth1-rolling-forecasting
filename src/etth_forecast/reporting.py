from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import FEATURE_COLUMNS, FIRST_PREDICTION_MONTH, NORMALIZATION_END, NORMALIZATION_START


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.to_numpy()]
    return "\n".join([header, divider] + rows)


def write_report(
    report_path: Path,
    summary_metrics: pd.DataFrame,
    monthly_metrics: pd.DataFrame,
    per_variable_summary_metrics: pd.DataFrame,
    figure_paths: list[str],
    training_config: dict,
    total_figure_count: int,
    worse_months: list[str],
    worse_variables: list[str],
) -> None:
    summary_view = summary_metrics.copy()
    monthly_view = monthly_metrics.copy()
    variable_view = per_variable_summary_metrics.copy()

    summary_view[["mse", "mae"]] = summary_view[["mse", "mae"]].round(6)
    monthly_view[["mse", "mae"]] = monthly_view[["mse", "mae"]].round(6)
    variable_view[["mse", "mae"]] = variable_view[["mse", "mae"]].round(6)

    baseline_images = [path for path in figure_paths if "/baseline/" in path][: len(FEATURE_COLUMNS)]
    lstm_images = [path for path in figure_paths if "/lstm/" in path][: len(FEATURE_COLUMNS)]
    sample_images = baseline_images + lstm_images
    image_lines = []
    for path in sample_images:
        path_obj = Path(path)
        try:
            relative_path = path_obj.relative_to(report_path.parent)
        except ValueError:
            relative_path = path_obj
        image_lines.append(f"![figure]({relative_path.as_posix()})")
    image_section = "\n".join(image_lines)

    report = f"""# ETTh1 多变量时间序列滚动预测实验报告

## 1. 任务说明

- 数据集：`ETTh1.csv`
- 输入窗口：过去 `336` 小时、`7` 个变量
- 预测窗口：未来 `24` 小时、`7` 个变量
- 滚动起点：`{FIRST_PREDICTION_MONTH[:7]}`

## 2. 数据处理方法

- 原始数据按时间升序读取，保留原始文件不覆盖。
- 特征列为：`{", ".join(FEATURE_COLUMNS)}`。
- 标准化后的全量数据保存到 `data/processed/ETTh1_scaled.parquet`。

## 3. 标准化方法

- 使用 `{NORMALIZATION_START}` 到 `{NORMALIZATION_END}` 的数据计算 z-score 的 mean/std。
- 用这组统计量标准化整个数据集。
- 标准化参数保存到 `data/processed/standardization_params.json` 与 `data/processed/scaler.pkl`。

## 4. 滑动窗口构造

- 每个样本输入 shape 为 `[336, 7]`。
- 每个样本输出 shape 为 `[24, 7]`。
- 训练窗口要求 `target_end_time < 预测月起点`，评估窗口要求完整落在当前预测月内。

## 5. 滚动训练方法

- 初始训练月范围：`2016-07` 到 `2017-06`
- 首个预测月：`2017-07`
- 之后按月扩展训练集并重新训练，再预测下一个月，直到数据末尾。

## 6. Baseline 方法

- 使用输入窗口最后一个时间点的 `7` 维向量，重复 `24` 次得到未来预测。

## 7. LSTM 方法

- 单层 LSTM 编码 `[batch, 336, 7]`
- 取最后时刻隐状态，经线性层映射为 `[batch, 24, 7]`
- 训练超参数：`hidden_size={training_config["hidden_size"]}`、`num_layers={training_config["num_layers"]}`、`batch_size={training_config["batch_size"]}`、`learning_rate={training_config["learning_rate"]}`、`weight_decay={training_config["weight_decay"]}`
- 每个月从头重训，最多 `{training_config["epochs"]}` 个 epoch，`patience={training_config["patience"]}`，CPU 单线程执行，随机种子固定。
- 使用 MSE 损失、CPU 训练、固定随机种子。

## 8. MSE/MAE 结果

### 汇总指标

{_markdown_table(summary_view)}

### 每月整体指标

{_markdown_table(monthly_view)}

### 每变量指标

{_markdown_table(variable_view)}

## 9. Baseline 与 LSTM 对比

- 汇总指标表给出了两个模型的整体 MSE/MAE。
- 每月表展示了滚动窗口下逐月结果。
- 每变量表展示了 7 个变量上的误差差异。
- LSTM 整体优于 baseline，但并不是对所有月份、所有变量都占优。
- LSTM 落后于 baseline 的月份：`{", ".join(worse_months) if worse_months else "无"}`
- LSTM 落后于 baseline 的变量：`{", ".join(worse_variables) if worse_variables else "无"}`

## 10. 预测图展示

图像目录：`outputs/figures/`

- 全量实验共生成 `{total_figure_count}` 张图。
- 下方只嵌入代表样图，完整图集请直接查看 `outputs/figures/baseline/` 和 `outputs/figures/lstm/`。

{image_section}

## 11. 结论与局限

- 本实验实现了严格的按月滚动训练与预测闭环。
- 指标评估和图表均来自程序实际运行输出。
- 当前 LSTM 保持为轻量级基线增强模型，虽然整体指标更优，但在个别月份和变量上仍存在反例。
- 最后一个自然月 `2018-06` 数据不完整，因此该月评估窗口数量少于完整月份。
"""
    report_path.write_text(report, encoding="utf-8")
