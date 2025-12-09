# datasets.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset

from config import cfg
from data_loading import load_all_gamers_ppg


# -----------------------------------------------------------
# Sequence construction helper
# -----------------------------------------------------------

def build_sequences_from_signal(
    signal: np.ndarray,
    seq_len: int,
    pred_len: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a 1D signal into sliding window sequences for supervised learning.

    For signal x[0..T-1], we build:

        X[i] = [x[i], ..., x[i + seq_len - 1]]            (length seq_len)
        y[i] = [x[i + seq_len], ..., x[i + seq_len + pred_len - 1]]

    for i = 0 .. T - seq_len - pred_len

    Args:
        signal: np.ndarray, shape (T,)
        seq_len: int, length of input sequence
        pred_len: int, length of target sequence (1 = next step)

    Returns:
        X: np.ndarray, shape (N, seq_len, 1)
        y: np.ndarray, shape (N, pred_len, 1) if pred_len > 1, else (N, 1)
    """
    x = signal.astype("float32")
    T = len(x)
    N = T - seq_len - pred_len + 1

    if N <= 0:
        raise ValueError(
            f"Signal too short for seq_len={seq_len}, pred_len={pred_len}, "
            f"length={T}"
        )

    X = []
    Y = []

    for i in range(N):
        seq_x = x[i : i + seq_len]
        seq_y = x[i + seq_len : i + seq_len + pred_len]
        X.append(seq_x)
        Y.append(seq_y)

    X = np.stack(X, axis=0)  # (N, seq_len)
    Y = np.stack(Y, axis=0)  # (N, pred_len)

    # Add feature dimension = 1
    X = X[..., None]         # (N, seq_len, 1)
    if pred_len == 1:
        Y = Y[:, 0:1]        # (N, 1)
    else:
        Y = Y[..., None]     # (N, pred_len, 1)

    return X, Y


# -----------------------------------------------------------
# Dataset class
# -----------------------------------------------------------

class PPGSequenceDataset(Dataset):
    """
    Generic sequence dataset for PPG data.

    X: (N, seq_len, 1)
    y: (N, pred_len, 1) or (N, 1) for pred_len=1
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: Optional[str] = None,
    ):
        super().__init__()

        if X.ndim != 3:
            raise ValueError(f"X must have shape (N, seq_len, 1), got {X.shape}")
        if y.ndim not in (2, 3):
            raise ValueError(f"y must have shape (N, 1) or (N, pred_len, 1), got {y.shape}")

        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("float32"))

        self.device = device or cfg.device

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        # We don't move to device here; DataLoader + training loop will do .to(device)
        return self.X[idx], self.y[idx]


# -----------------------------------------------------------
# High-level: build datasets from all gamers
# -----------------------------------------------------------

@dataclass
class SplitDatasets:
    train: PPGSequenceDataset
    val: PPGSequenceDataset
    test: PPGSequenceDataset


def create_ppg_sequence_datasets(
    gamer_ids: Optional[List[int]] = None,
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    max_hours_per_gamer: Optional[float] = None,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> SplitDatasets:
    """
    Load PPG data for selected gamers, build sequences, and split into
    train/val/test datasets.

    This function essentially glues together:
      - load_all_gamers_ppg (from data_loading)
      - build_sequences_from_signal (above)
      - train/val/test splitting

    Returns:
        SplitDatasets(train, val, test), each a PPGSequenceDataset.
    """
    # Use defaults from config if not provided
    if gamer_ids is None:
        gamer_ids = list(cfg.data.gamer_ids)
    if seq_len is None:
        seq_len = cfg.data.seq_len
    if pred_len is None:
        pred_len = cfg.data.pred_len
    if max_hours_per_gamer is None:
        max_hours_per_gamer = cfg.data.max_hours_per_gamer
    if train_frac is None:
        train_frac = cfg.data.train_frac
    if val_frac is None:
        val_frac = cfg.data.val_frac
    if seed is None:
        seed = cfg.training.seed

    # 1. Load PPG signals for all gamers (limited to max_hours_per_gamer)
    all_data = load_all_gamers_ppg(
        gamer_ids=gamer_ids,
        max_hours_per_gamer=max_hours_per_gamer,
    )

    # 2. Build sequences per gamer, then concatenate
    X_list = []
    Y_list = []

    for gid in gamer_ids:
        sig = all_data[gid]["signal"]  # 1D array
        Xg, Yg = build_sequences_from_signal(
            sig,
            seq_len=seq_len,
            pred_len=pred_len,
        )
        X_list.append(Xg)
        Y_list.append(Yg)

    X = np.concatenate(X_list, axis=0)
    Y = np.concatenate(Y_list, axis=0)

    N = X.shape[0]

    # 3. Build train/val/test indices
    rng = np.random.RandomState(seed)
    indices = np.arange(N)
    if shuffle:
        rng.shuffle(indices)

    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], Y[train_idx]
    X_val, y_val = X[val_idx], Y[val_idx]
    X_test, y_test = X[test_idx], Y[test_idx]

    train_ds = PPGSequenceDataset(X_train, y_train)
    val_ds = PPGSequenceDataset(X_val, y_val)
    test_ds = PPGSequenceDataset(X_test, y_test)

    return SplitDatasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
    )
class PPGLabeledSleepinessDataset(Dataset):
    """
    PPG window -> sleepiness label (classification).

    X: (N, seq_len, 1)
    y: (N,) integer in {0..6}
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        super().__init__()

        if X.ndim != 3:
            raise ValueError(f"X must have shape (N, seq_len, 1), got {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must have shape (N,), got {y.shape}")

        self.X = torch.from_numpy(X.astype("float32"))
        self.y = torch.from_numpy(y.astype("int64"))  # for CrossEntropyLoss

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

from data_loading import build_ppg_windows_with_sleepiness_for_gamer

def create_sleepiness_datasets(
    gamer_ids: Optional[List[int]] = None,
    seq_len: Optional[int] = None,
    max_hours_per_gamer: Optional[float] = None,
    train_frac: Optional[float] = None,
    val_frac: Optional[float] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> SplitDatasets:
    """
    Build sleepiness-labeled datasets:

    Each sample:
        X: PPG window of length seq_len before an annotation
        y: sleepiness class 0..6

    Splits into train/val/test.
    """
    if gamer_ids is None:
        gamer_ids = list(cfg.data.gamer_ids)
    if seq_len is None:
        seq_len = cfg.data.seq_len
    if max_hours_per_gamer is None:
        max_hours_per_gamer = cfg.data.max_hours_per_gamer
    if train_frac is None:
        train_frac = cfg.data.train_frac
    if val_frac is None:
        val_frac = cfg.data.val_frac
    if seed is None:
        seed = cfg.training.seed

    X_list = []
    y_list = []

    for gid in gamer_ids:
        try:
            Xg, yg = build_ppg_windows_with_sleepiness_for_gamer(
                gamer_id=gid,
                seq_len=seq_len,
                max_hours=max_hours_per_gamer,
            )
            print(f"INFO: gamer {gid} -> {Xg.shape[0]} samples")
            X_list.append(Xg)
            y_list.append(yg)
        except RuntimeError as e:
            # If a gamer has no usable windows, we just skip them
            print(f"WARNING: skipping gamer {gid}: {e}")
            continue

    if not X_list:
        raise RuntimeError("No valid sleepiness samples found for any gamer.")

    X = np.concatenate(X_list, axis=0)  # (N, seq_len, 1)
    y = np.concatenate(y_list, axis=0)  # (N,)

    N = X.shape[0]

    rng = np.random.RandomState(seed)
    indices = np.arange(N)
    if shuffle:
        rng.shuffle(indices)

    n_train = int(train_frac * N)
    n_val = int(val_frac * N)
    n_test = N - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_ds = PPGLabeledSleepinessDataset(X_train, y_train)
    val_ds = PPGLabeledSleepinessDataset(X_val, y_val)
    test_ds = PPGLabeledSleepinessDataset(X_test, y_test)

    return SplitDatasets(
        train=train_ds,
        val=val_ds,
        test=test_ds,
    )
