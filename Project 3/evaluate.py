from datasets import create_ppg_sequence_datasets, create_sleepiness_datasets


from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import cfg, CHECKPOINT_DIR
from datasets import create_ppg_sequence_datasets, SplitDatasets
from models import build_model


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    splits: SplitDatasets,
) -> Dict[str, DataLoader]:
    batch_size = cfg.training.batch_size

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def load_best_model(
    ckpt_path: Optional[Path] = None,
) -> torch.nn.Module:
    """
    Load the best model saved by train.py.
    """
    if ckpt_path is None:
        ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found at {ckpt_path}. "
            f"Make sure you ran train.py with save_best_model=True."
        )

    device = cfg.device
    # Build a fresh model using current cfg (same architecture as during training)
    model = build_model(sequence_output=(cfg.data.pred_len > 1))
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run the model on all batches in dataloader and return stacked
    (preds, targets) as numpy arrays.

    Works for:
        - pred_len = 1  -> shapes (N, 1)
        - pred_len > 1  -> shapes (N, pred_len, 1)
    """
    preds_list = []
    targets_list = []

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(X_batch)

        preds_list.append(y_pred.cpu().numpy())
        targets_list.append(y_batch.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    return preds, targets


# -----------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------

def plot_next_step_predictions(
    preds: np.ndarray,
    targets: np.ndarray,
    num_points: int = 200,
    title: str = "Next-step prediction (test set)",
):
    """
    For pred_len = 1: plot predicted vs true values over time.
    """
    # Flatten to (N,)
    preds_flat = preds.reshape(-1)
    targets_flat = targets.reshape(-1)

    N = len(preds_flat)
    num_points = min(num_points, N)

    plt.figure(figsize=(10, 4))
    plt.plot(targets_flat[:num_points], label="True", linewidth=1)
    plt.plot(preds_flat[:num_points], label="Predicted", linestyle="--", linewidth=1)
    plt.xlabel("Time step (test samples)")
    plt.ylabel("Normalized PPG value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_sequence_prediction_example(
    preds: np.ndarray,
    targets: np.ndarray,
    index: int = 0,
    title: str = "Sequence prediction example (test set)",
):
    """
    For pred_len > 1: plot one example predicted sequence vs true sequence.

    preds:   (N, pred_len, 1)
    targets: (N, pred_len, 1)
    """
    if preds.ndim != 3 or targets.ndim != 3:
        raise ValueError(
            f"Expected preds/targets with ndim=3 for sequence predictions, "
            f"got preds.shape={preds.shape}, targets.shape={targets.shape}"
        )

    N = preds.shape[0]
    if index < 0 or index >= N:
        raise IndexError(f"index must be between 0 and {N-1}, got {index}")

    pred_seq = preds[index, :, 0]    # (pred_len,)
    true_seq = targets[index, :, 0]  # (pred_len,)

    timesteps = np.arange(len(pred_seq))

    plt.figure(figsize=(8, 4))
    plt.plot(timesteps, true_seq, label="True", linewidth=1)
    plt.plot(timesteps, pred_seq, label="Predicted", linestyle="--", linewidth=1)
    plt.xlabel("Prediction step")
    plt.ylabel("Normalized PPG value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------
# High-level evaluate() function
# -----------------------------------------------------------

def evaluate_model() -> Dict[str, Any]:
    set_seed(cfg.training.seed)

    # 1. Datasets & loaders
    if cfg.data.label_type == "sleepiness":
        splits = create_sleepiness_datasets()
    else:
        splits = create_ppg_sequence_datasets()

    loaders = make_dataloaders(splits)

    # 2. Model
    model = load_best_model()
    device = cfg.device

    # 3. Collect predictions on test set
    preds, targets = collect_predictions(
        model,
        loaders["test"],
        device=device,
    )

    if cfg.data.label_type == "sleepiness":
        # classification: preds are logits (N,7), targets are class indices (N,)
        # Convert to class predictions
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
        # regression: MSE
        mse = float(np.mean((preds - targets) ** 2))
        print(f"Test MSE (from evaluate.py): {mse:.6f}")
        metric_name = "mse"
        metric_value = mse

    return {
        f"test_{metric_name}": metric_value,
        "preds": preds,
        "targets": targets,
    }



# -----------------------------------------------------------
# Script entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    print("Using device:", cfg.device)
    results = evaluate_model()

    preds = results["preds"]
    targets = results["targets"]

    # Choose plotting based on pred_len
    if cfg.data.pred_len == 1:
        plot_next_step_predictions(preds, targets, num_points=300)
    else:
        plot_sequence_prediction_example(preds, targets, index=0)
