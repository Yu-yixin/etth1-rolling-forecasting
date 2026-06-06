from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .config import FEATURE_COLUMNS, WindowSpec


class RepeatLastValueBaseline:
    def __init__(self, prediction_length: int) -> None:
        self.prediction_length = prediction_length

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        last_step = inputs[:, -1:, :]
        return np.repeat(last_step, self.prediction_length, axis=1)


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = len(FEATURE_COLUMNS),
        hidden_size: int = 64,
        num_layers: int = 1,
        prediction_length: int = WindowSpec().prediction_length,
        output_dim: int = len(FEATURE_COLUMNS),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.prediction_length = prediction_length
        self.output_dim = output_dim
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
        )
        self.readout = nn.Linear(hidden_size, prediction_length * output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(inputs)
        final_state = encoded[:, -1, :]
        raw_output = self.readout(final_state)
        return raw_output.view(-1, self.prediction_length, self.output_dim)
