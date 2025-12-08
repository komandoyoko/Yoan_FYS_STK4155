from datasets import (
    create_ppg_sequence_datasets,
    create_sleepiness_datasets,
    SplitDatasets,
)


from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from config import cfg, CHECKPOINT_DIR
from datasets import create_ppg_sequence_datasets, SplitDatasets
from models import build_model


# -----------------------------------------------------------
# Utilities
# -----------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloaders(
    splits: SplitDatasets,
) -> Dict[str, DataLoader]:
    """
    Given SplitDatasets(train, val, test), create DataLoaders.
    """
    batch_size = cfg.training.batch_size

    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=True,
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

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }


# -----------------------------------------------------------
# Training & evaluation loops
# -----------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> float:
    """
    Train for a single epoch and return average loss.
    """
    model.train()
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    avg_loss = running_loss / max(n_samples, 1)
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> float:
    """
    Evaluate on a dataset and return average loss.
    """
    model.eval()
    running_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        preds = model(X_batch)
        loss = criterion(preds, y_batch)

        batch_size = X_batch.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

    avg_loss = running_loss / max(n_samples, 1)
    return avg_loss


# -----------------------------------------------------------
# Top-level train() function
# -----------------------------------------------------------

def train() -> Dict[str, Any]:
    """
    Full training pipeline:

    - set seeds
    - create datasets & dataloaders
    - build model
    - train for num_epochs with validation
    - early stopping + save best model

    Returns:
        stats dict with training & validation losses.
    """
    set_seed(cfg.training.seed)

    # 1. Datasets & loaders
    if cfg.data.label_type == "sleepiness":
        splits = create_sleepiness_datasets()

    else:
        splits = create_ppg_sequence_datasets()

    loaders = make_dataloaders(splits)

    # 2. Model, loss, optimizer
    device = cfg.device
    sequence_output = (
        cfg.data.pred_len > 1 and cfg.data.label_type != "sleepiness"
    )
    model = build_model(sequence_output=(cfg.data.pred_len > 1))

    if cfg.data.label_type == "sleepiness":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # 3. Training loop with early stopping
    num_epochs = cfg.training.num_epochs
    print_every = cfg.training.print_every
    patience = cfg.training.early_stopping_patience
    save_best = cfg.training.save_best_model

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
        )
        val_loss = evaluate(
            model,
            loaders["val"],
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % print_every == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"Epoch {epoch:3d}/{num_epochs:3d} "
                f"| train loss: {train_loss:.6f} "
                f"| val loss: {val_loss:.6f}"
            )

        # Early stopping & checkpointing
        if val_loss < best_val_loss - 1e-6:  # tiny margin to avoid float noise
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0

            if save_best:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg.to_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best val loss: {best_val_loss:.6f} (epoch {best_epoch})"
            )
            break

    # 4. Evaluate best model on test set (if saved)
    test_loss = None
    if save_best and ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_loss = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
        )
        print(f"Test loss (best model @ epoch {best_epoch}): {test_loss:.6f}")
    else:
        # Evaluate current model instead
        test_loss = evaluate(
            model,
            loaders["test"],
            criterion,
            device,
        )
        print(f"Test loss (last model): {test_loss:.6f}")

    history["best_val_loss"] = best_val_loss
    history["best_epoch"] = best_epoch
    history["test_loss"] = test_loss

    return history


# -----------------------------------------------------------
# Script entry point
# -----------------------------------------------------------

if __name__ == "__main__":
    print("Using device:", cfg.device)
    stats = train()
    print("Training complete.")
    print("Best validation loss:", stats["best_val_loss"])
    print("Test loss:", stats["test_loss"])
