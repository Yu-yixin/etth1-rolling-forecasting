from __future__ import annotations

import numpy as np
import torch

from etth_forecast.evaluation import compute_metrics
from etth_forecast.models import LSTMForecaster, RepeatLastValueBaseline


def test_baseline_output_shape():
    inputs = np.random.randn(5, 336, 7).astype(np.float32)
    model = RepeatLastValueBaseline(prediction_length=24)
    predictions = model.predict(inputs)
    assert predictions.shape == (5, 24, 7)


def test_lstm_output_shape():
    model = LSTMForecaster()
    inputs = torch.randn(3, 336, 7)
    predictions = model(inputs)
    assert tuple(predictions.shape) == (3, 24, 7)


def test_metrics_match_manual_values():
    y_true = np.array([[[1.0], [3.0]]])
    y_pred = np.array([[[2.0], [1.0]]])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mse"] == 2.5
    assert metrics["mae"] == 1.5
