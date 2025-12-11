from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

from config import cfg, PLOTS_DIR, CHECKPOINT_DIR
from datasets import (
    create_sleepiness_datasets,
    create_ppg_sequence_datasets,
    SplitDatasets,
)
from models import build_model
from train import make_dataloaders, set_seed


# ------------------------------------------------------------
# Helper: load best model and data
# ------------------------------------------------------------

def load_best_model(device: str | None = None) -> torch.nn.Module:
    if device is None:
        device = cfg.device

    ckpt_path = Path(CHECKPOINT_DIR) / "best_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    if cfg.data.label_type == "sleepiness":
        sequence_output = False
    else:
        sequence_output = (cfg.data.pred_len > 1)

    model = build_model(sequence_output=sequence_output).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_splits_and_loaders() -> tuple[SplitDatasets, Dict[str, DataLoader]]:
    set_seed(cfg.training.seed)
    if cfg.data.label_type == "sleepiness":
        splits = create_sleepiness_datasets()
    else:
        splits = create_ppg_sequence_datasets()
    loaders = make_dataloaders(splits)
    return splits, loaders


# ------------------------------------------------------------
# Figure 1: class distribution
# ------------------------------------------------------------

def plot_label_distribution(splits: SplitDatasets):
    # Collect all labels from train+val+test
    def collect_y(ds):
        ys = []
        for _, y in DataLoader(ds, batch_size=1024, shuffle=False):
            ys.append(y.numpy())
        return np.concatenate(ys, axis=0)

    from torch.utils.data import DataLoader
    y_all = np.concatenate(
        [
            collect_y(splits.train),
            collect_y(splits.val),
            collect_y(splits.test),
        ],
        axis=0,
    )
    # labels are 0..6 -> convert to 1..7 for plotting
    y_all_scores = y_all + 1

    unique, counts = np.unique(y_all_scores, return_counts=True)

    plt.figure()
    plt.bar(unique, counts)
    plt.xticks(unique)
    plt.xlabel("Sleepiness score (1–7)")
    plt.ylabel("Count")
    plt.title("Distribution of sleepiness scores (all gamers)")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "label_distribution.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Figure 2–3: example PPG windows per class
# ------------------------------------------------------------

def plot_example_windows_per_class(splits: SplitDatasets, n_classes: int = 7):
    # We assume X is (N, seq_len, 1), y is (N,)
    # We'll sample from the *training* set.
    train_ds = splits.train
    # Pull everything into memory (fine here)
    Xs = []
    ys = []
    for X_batch, y_batch in DataLoader(train_ds, batch_size=2048, shuffle=False):
        Xs.append(X_batch.numpy())
        ys.append(y_batch.numpy())
    X_all = np.concatenate(Xs, axis=0)  # (N, seq_len, 1)
    y_all = np.concatenate(ys, axis=0)  # (N,)

    seq_len = X_all.shape[1]

    plt.figure(figsize=(12, 8))
    classes_shown = 0
    for cls in range(n_classes):
        idx = np.where(y_all == cls)[0]
        if len(idx) == 0:
            continue
        i = np.random.choice(idx)
        signal = X_all[i, :, 0]  # (seq_len,)
        classes_shown += 1

        plt.subplot(3, 3, classes_shown)
        plt.plot(signal)
        plt.title(f"Example window, sleepiness={cls+1}")
        plt.xlabel("Time (samples)")
        plt.ylabel("Normalized PPG")
        if classes_shown >= 9:
            break

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "example_windows_per_class.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Figure 4: learning curves
# ------------------------------------------------------------

def plot_learning_curves():
    hist_path = PLOTS_DIR / "training_history.npz"
    if not hist_path.exists():
        raise FileNotFoundError(f"No training history at {hist_path}, run train.py first.")

    data = np.load(hist_path)
    train_loss = data["train_loss"]
    val_loss = data["val_loss"]

    epochs = np.arange(1, len(train_loss) + 1)

    plt.figure()
    plt.plot(epochs, train_loss, label="Train loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Learning curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "learning_curves.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Figures 5–6: confusion matrix & per-class accuracy
# ------------------------------------------------------------

@torch.no_grad()
def get_test_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    preds_list = []
    targets_list = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        out = model(X_batch)
        preds_list.append(out.detach().cpu().numpy())
        targets_list.append(y_batch.detach().cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    return preds, targets


def plot_confusion_and_class_acc(loaders: Dict[str, DataLoader]):
    device = cfg.device
    model = load_best_model(device=device)

    preds, targets = get_test_predictions(model, loaders["test"], device=device)
    logits = preds
    y_true = targets.reshape(-1).astype(int)
    y_pred = np.argmax(logits, axis=1)

    n_classes = 7
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    ticks = np.arange(n_classes)
    tick_labels = ticks + 1
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Confusion matrix (test set)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    # per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    plt.figure()
    plt.bar(tick_labels, per_class_acc)
    plt.ylim(0, 1)
    plt.xlabel("Sleepiness score (true)")
    plt.ylabel("Accuracy")
    plt.title("Per-class accuracy (test set)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "per_class_accuracy.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Figure 7: true vs predicted over time for one gamer
# ------------------------------------------------------------

def plot_predictions_over_time_for_one_gamer(splits: SplitDatasets, loaders: Dict[str, DataLoader], gamer_id: int = 1):
    """
    Simple version: use *all* test samples, but color those that belong
    to a given gamer id if you stored gamer ids. If you didn't store
    gamer ids separately, you can just show all test points sorted by time.
    For now we'll just plot all test predictions in index order.
    """
    device = cfg.device
    model = load_best_model(device=device)

    preds, targets = get_test_predictions(model, loaders["test"], device=device)
    y_true = targets.reshape(-1).astype(int) + 1      # back to 1..7
    y_pred = np.argmax(preds, axis=1) + 1

    idx = np.arange(len(y_true))

    plt.figure(figsize=(10, 4))
    plt.plot(idx, y_true, label="True", marker="o", linestyle="-", alpha=0.7)
    plt.plot(idx, y_pred, label="Predicted", marker="x", linestyle="--", alpha=0.7)
    plt.xlabel("Test sample index")
    plt.ylabel("Sleepiness score")
    plt.title("True vs predicted sleepiness (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "true_vs_predicted_over_test_index.png", dpi=200)
    plt.close()


# ------------------------------------------------------------
# Main: generate all plots
# ------------------------------------------------------------

if __name__ == "__main__":
    print("Using device:", cfg.device)
    splits, loaders = get_splits_and_loaders()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Plotting label distribution...")
    plot_label_distribution(splits)

    print("Plotting example windows per class...")
    plot_example_windows_per_class(splits)

    print("Plotting learning curves...")
    plot_learning_curves()

    print("Plotting confusion matrix and per-class accuracy...")
    plot_confusion_and_class_acc(loaders)

    print("Plotting true vs predicted over test index...")
    plot_predictions_over_time_for_one_gamer(splits, loaders)

    print("All plots saved to", PLOTS_DIR)
