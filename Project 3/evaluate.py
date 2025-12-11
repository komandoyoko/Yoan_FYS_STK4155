from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import cfg, CHECKPOINT_DIR
from datasets import (
    create_ppg_sequence_datasets,
    create_sleepiness_datasets,
    SplitDatasets,
)
from models import build_model
from train import make_dataloaders, set_seed


def load_best_model(device: str | None = None) -> nn.Module:
    """
    Build a fresh model and load weights from checkpoints/best_model.pt.
    """
    if device is None:
        device = cfg.device

    ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    # same sequence_output logic as in train.py
    if cfg.data.label_type == "sleepiness":
        sequence_output = False
    else:
        sequence_output = (cfg.data.pred_len > 1)

    model = build_model(sequence_output=sequence_output).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run model on dataloader and return (preds, targets) as numpy arrays.
    For sleepiness:
        preds: (N, 7)  logits
        targets: (N,)  class indices
    For regression:
        preds, targets: shapes compatible with MSE.
    """
    model.eval()
    preds_list = []
    targets_list = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        out = model(X_batch)
        preds_list.append(out.detach().cpu().numpy())
        targets_list.append(y_batch.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return preds, targets


def evaluate_model() -> Dict[str, Any]:
    set_seed(cfg.training.seed)

    # 1. Datasets & loaders (same logic as train.py)
    if cfg.data.label_type == "sleepiness":
        splits = create_sleepiness_datasets()
    else:
        splits = create_ppg_sequence_datasets()

    loaders = make_dataloaders(splits)

    # 2. Model
    device = cfg.device
    model = load_best_model(device=device)

    # 3. Collect predictions on test set
    preds, targets = collect_predictions(
        model,
        loaders["test"],
        device=device,
    )

    if cfg.data.label_type == "sleepiness":
    # preds: (N, 7)
        if preds.ndim != 2:
            preds_flat = preds.reshape(preds.shape[0], -1)
        else:
            preds_flat = preds

        y_pred = np.argmax(preds_flat, axis=1)
        y_true = targets.reshape(-1).astype(int)

        acc = float((y_pred == y_true).mean())
        print(f"Test accuracy (sleepiness): {acc:.4f}")

        metric_name = "accuracy"
        metric_value = acc
    else:
        mse = float(np.mean((preds - targets) ** 2))
        print(f"Test MSE (forecasting): {mse:.6f}")
        metric_name = "mse"
        metric_value = mse

    return {
        f"test_{metric_name}": metric_value,
        "preds": preds,
        "targets": targets,
    }

if __name__ == "__main__":
    print("Using device:", cfg.device)
    results = evaluate_model()
    print("Done.")
    print(results.keys())
