# ETTh1 Rolling Forecasting

这个项目实现了一个可复现、可测试、可交付的 ETTh1 多变量时间序列滚动预测流程：

- 使用 `2016-07-01 00:00:00` 到 `2017-06-30 23:00:00` 计算 z-score 标准化参数
- 使用过去 `336` 小时的 `7` 维输入预测未来 `24` 小时的 `7` 维输出
- 按月滚动训练与评估，从 `2017-07` 开始直到数据结束
- 同时输出 baseline 与 LSTM 的预测结果、指标、图表、报告和审计记录

## 运行方式

```bash
PYTHONPATH=src python3 run_pipeline.py
```

快速 smoke test：

```bash
PYTHONPATH=src python3 run_pipeline.py --epochs 1 --max-prediction-months 1 --max-train-windows 256 --max-eval-windows-per-month 32
PYTHONPATH=src python3 -m pytest
```

## 主要输出

- `data/processed/ETTh1_scaled.parquet`
- `data/processed/standardization_params.json`
- `outputs/metrics/*.csv`
- `outputs/predictions/**/*.parquet`
- `outputs/figures/**/*`
- `outputs/audits/*`
- `outputs/final_closeout.md`
- `report.md`

这些运行产物默认保留在本地，但不建议纳入版本控制；仓库中的 `.gitignore` 已经忽略 `data/processed/`、`outputs/` 和 `report.md`，这样重复运行实验时不会污染 `git status`。
