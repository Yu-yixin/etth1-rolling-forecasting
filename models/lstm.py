from __future__ import annotations

import torch

from etth_forecast.models import LSTMForecaster


def main() -> None:
    batch_size = 8
    seq_len = 336
    input_dim = 7
    pred_len = 24
    inputs = torch.randn(batch_size, seq_len, input_dim)
    model = LSTMForecaster(
        input_dim=input_dim,
        hidden_size=64,
        prediction_length=pred_len,
        output_dim=input_dim,
    )
    predictions = model(inputs)
    print("prediction shape:", tuple(predictions.shape))


if __name__ == "__main__":
    main()
