from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .utils import save_json
from .windowing import ArrayWindowDataset


LOGGER = logging.getLogger(__name__)


def split_train_validation(
    inputs: np.ndarray,
    targets: np.ndarray,
    validation_fraction: float,
    min_validation_windows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = len(inputs)
    if total == 0:
        raise ValueError("Cannot split an empty training set.")

    validation_size = max(int(total * validation_fraction), min_validation_windows)
    validation_size = min(validation_size, max(total // 5, 1))
    if total <= validation_size:
        validation_size = max(total // 10, 1)
    if total <= 2:
        validation_size = 1

    split_idx = max(total - validation_size, 1)
    train_inputs = inputs[:split_idx]
    train_targets = targets[:split_idx]
    val_inputs = inputs[split_idx:]
    val_targets = targets[split_idx:]
    return train_inputs, train_targets, val_inputs, val_targets


def _build_loader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = ArrayWindowDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_lstm_model(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    config: TrainingConfig,
    checkpoint_path: Path,
    progress_label: str = "lstm",
) -> dict:
    torch.set_num_threads(config.torch_num_threads)
    device = torch.device("cpu")
    model = model.to(device)

    split = split_train_validation(
        train_inputs,
        train_targets,
        validation_fraction=config.validation_fraction,
        min_validation_windows=config.min_validation_windows,
    )
    tr_x, tr_y, val_x, val_y = split
    train_loader = _build_loader(tr_x, tr_y, batch_size=config.batch_size, shuffle=True)
    val_loader = _build_loader(val_x, val_y, batch_size=config.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_inputs)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_inputs)
                loss = loss_fn(predictions, batch_targets)
                val_loss += loss.item() * len(batch_inputs)

        mean_train_loss = train_loss / max(len(tr_x), 1)
        mean_val_loss = val_loss / max(len(val_x), 1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "val_loss": mean_val_loss,
            }
        )
        LOGGER.info(
            "[%s] epoch %d/%d train_loss=%.6f val_loss=%.6f",
            progress_label,
            epoch,
            config.epochs,
            mean_train_loss,
            mean_val_loss,
        )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                LOGGER.info(
                    "[%s] early stopping triggered after epoch %d",
                    progress_label,
                    epoch,
                )
                break

    if best_state is None:
        best_state = model.state_dict()

    model.load_state_dict(best_state)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)
    history_path = checkpoint_path.with_suffix(".history.json")
    save_json(
        {
            "training_config": asdict(config),
            "history": history,
            "best_val_loss": best_val_loss,
            "train_windows": int(len(train_inputs)),
            "validation_windows": int(len(val_x)),
        },
        history_path,
    )
    LOGGER.info(
        "[%s] saved checkpoint to %s",
        progress_label,
        checkpoint_path,
    )
    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "checkpoint_path": str(checkpoint_path),
    }


def predict_with_model(
    model: nn.Module,
    inputs: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(inputs) == 0:
        raise ValueError("No inputs provided for prediction.")

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    loader = _build_loader(
        inputs,
        np.zeros((len(inputs), 1, 1), dtype=np.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    outputs = []
    with torch.no_grad():
        for batch_inputs, _ in loader:
            predictions = model(batch_inputs.to(device))
            outputs.append(predictions.cpu().numpy())
    return np.concatenate(outputs, axis=0)
