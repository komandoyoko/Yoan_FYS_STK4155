# data_loading.py

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from config import DATA_DIR, cfg


# -----------------------------------------------------------
# Helpers to find files
# -----------------------------------------------------------

def list_ppg_files_for_gamer(
    gamer_id: int,
    data_dir: Path = DATA_DIR,
) -> List[Path]:
    """
    Return a sorted list of PPG CSV files for a given gamer.

    Uses the pattern from cfg.data.ppg_pattern, e.g. "gamer{gamer_id}-ppg-*.csv".
    """
    pattern = cfg.data.ppg_pattern.format(gamer_id=gamer_id)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No PPG files found for gamer {gamer_id} in {data_dir} "
            f"with pattern '{pattern}'"
        )
    return files


def annotations_file_for_gamer(
    gamer_id: int,
    data_dir: Path = DATA_DIR,
) -> Path:
    """
    Return the annotations CSV path for a given gamer.
    """
    pattern = cfg.data.annotations_pattern.format(gamer_id=gamer_id)
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No annotations file found for gamer {gamer_id} in {data_dir} "
            f"with pattern '{pattern}'"
        )
    if len(files) > 1:
        # Shouldn't normally happen, but be explicit
        raise RuntimeError(
            f"Multiple annotations files found for gamer {gamer_id}: {files}"
        )
    return files[0]


# -----------------------------------------------------------
# Core loading functions
# -----------------------------------------------------------

def _extract_date_from_ppg_filename(path: Path) -> str:
    """
    From 'gamer1-ppg-2000-01-01.csv' -> '2000-01-01'
    """
    parts = path.stem.split("-")  # ['gamer1', 'ppg', '2000', '01', '01']
    if len(parts) < 5:
        raise ValueError(f"Unexpected PPG filename format: {path.name}")
    return "-".join(parts[-3:])   # '2000-01-01'


def load_ppg_dataframe_for_gamer(
    gamer_id: int,
    data_dir: Path = DATA_DIR,
    max_hours: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load and concatenate PPG CSV files for a gamer, add a datetime column,
    optionally trim to a maximum of `max_hours` hours from the start.
    """
    ppg_files = list_ppg_files_for_gamer(gamer_id, data_dir=data_dir)

    dfs = []
    for path in ppg_files:
        df = pd.read_csv(path)

        if "Time" not in df.columns or "Red_Signal" not in df.columns:
            raise ValueError(
                f"Expected columns 'Time' and 'Red_Signal' in {path}, "
                f"got {df.columns.tolist()}"
            )

        # Drop rows with missing PPG
        df = df.dropna(subset=["Red_Signal"]).reset_index(drop=True)

        date_str = _extract_date_from_ppg_filename(path)
        dt_str = date_str + " " + df["Time"].astype(str)
        df["datetime"] = pd.to_datetime(dt_str)

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values("datetime").reset_index(drop=True)

    if max_hours is not None:
        t0 = full_df["datetime"].min()
        t_end = t0 + pd.Timedelta(hours=max_hours)
        full_df = full_df[full_df["datetime"] <= t_end].reset_index(drop=True)

    return full_df

def load_annotations_for_gamer(
    gamer_id: int,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load annotations (e.g., Stanford Sleepiness scores) for a gamer.

    Returns DataFrame with at least:
        ['Datetime', 'Event', 'Value', 'datetime']
    """
    path = annotations_file_for_gamer(gamer_id, data_dir=data_dir)
    df = pd.read_csv(path)

    if "Datetime" not in df.columns:
        raise ValueError(
            f"Expected column 'Datetime' in {path}, got {df.columns.tolist()}"
        )

    df["datetime"] = pd.to_datetime(df["Datetime"])
    return df


# -----------------------------------------------------------
# Convenience: get PPG signal as numpy (with optional normalization)
# -----------------------------------------------------------

def normalize_signal_array(
    x: np.ndarray,
    mode: str = "zscore",
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Normalize a 1D signal according to the given mode.

    Returns:
        normalized_x, stats_dict

    NaNs in x are handled with nan-safe statistics and then replaced by 0
    after normalization.
    """
    x = x.astype("float32")
    stats: Dict[str, float] = {}

    # If everything is NaN, bail out with zeros
    if np.all(np.isnan(x)):
        x_norm = np.zeros_like(x, dtype="float32")
        stats = {"mean": 0.0, "std": 1.0}
        return x_norm, stats

    if mode == "zscore":
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        if std == 0.0 or np.isnan(std):
            std = 1.0
        if np.isnan(mean):
            mean = 0.0
        x_norm = (x - mean) / std
        stats = {"mean": mean, "std": std}

    elif mode == "minmax":
        min_val = float(np.nanmin(x))
        max_val = float(np.nanmax(x))
        if np.isnan(min_val) or np.isnan(max_val) or max_val == min_val:
            x_norm = np.zeros_like(x, dtype="float32")
        else:
            x_norm = (x - min_val) / (max_val - min_val)
        stats = {"min": min_val, "max": max_val}

    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    # Replace any remaining NaNs (from x) with 0
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")

    return x_norm, stats


def load_ppg_signal_for_gamer(
    gamer_id: int,
    max_hours: Optional[float] = None,
    normalize: Optional[bool] = None,
    data_dir: Path = DATA_DIR,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, float]]]:
    """
    High-level helper:

    - Loads PPG dataframe for the gamer (optionally limited to `max_hours`)
    - Extracts the 'Red_Signal' column as a 1D float32 array.
    - Optionally normalizes it according to cfg.data.normalization_mode.
    - Returns (signal, timestamps, stats)

    Where:
        signal    : np.ndarray of shape (T,)
        timestamps: np.ndarray of dtype datetime64[ns] of shape (T,)
        stats     : dict with normalization parameters, or None
    """
    if normalize is None:
        normalize = cfg.data.normalize_signal

    df = load_ppg_dataframe_for_gamer(
        gamer_id=gamer_id,
        max_hours=max_hours,
        data_dir=data_dir,
    )

    signal = df["Red_Signal"].astype("float32").to_numpy()
    timestamps = df["datetime"].to_numpy()
    stats: Optional[Dict[str, float]] = None

    if normalize:
        signal, stats = normalize_signal_array(
            signal,
            mode=cfg.data.normalization_mode,
        )

    return signal, timestamps, stats


# -----------------------------------------------------------
# Convenience: load all gamers at once
# -----------------------------------------------------------

def load_all_gamers_ppg(
    gamer_ids: Optional[List[int]] = None,
    max_hours_per_gamer: Optional[float] = None,
    data_dir: Path = DATA_DIR,
) -> Dict[int, Dict[str, object]]:
    """
    Load PPG for multiple gamers.

    Returns a dict:
        {
          gamer_id: {
            "signal": np.ndarray (T,),
            "timestamps": np.ndarray (T,),
            "stats": dict or None,
          },
          ...
        }
    """
    if gamer_ids is None:
        gamer_ids = list(cfg.data.gamer_ids)

    if max_hours_per_gamer is None:
        max_hours_per_gamer = cfg.data.max_hours_per_gamer

    out: Dict[int, Dict[str, object]] = {}

    for gid in gamer_ids:
        sig, ts, stats = load_ppg_signal_for_gamer(
            gamer_id=gid,
            max_hours=max_hours_per_gamer,
            data_dir=data_dir,
        )
        out[gid] = {
            "signal": sig,
            "timestamps": ts,
            "stats": stats,
        }

    return out

def build_ppg_windows_with_sleepiness_for_gamer(
    gamer_id: int,
    seq_len: int,
    max_hours: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For one gamer:

    - Load PPG (optionally trimmed, but we will usually use full series).
    - Load annotations.
    - For each numeric sleepiness annotation (1–7) at time t_ann:
        * take PPG in [t_ann - backward_minutes, t_ann]
        * from that segment, build MANY windows of length seq_len
          with some stride
        * each window gets label = score (1–7) -> class 0..6

    Returns:
        X: (N, seq_len, 1)  normalized PPG windows
        y: (N,)             int labels in {0..6}
    """
    # Use config default if not provided
    if max_hours is None:
        max_hours = cfg.data.max_hours_per_gamer

    # 1) Load PPG for this gamer
    #    If max_hours_per_gamer is None, this loads full 2 days.
    ppg_df = load_ppg_dataframe_for_gamer(
        gamer_id=gamer_id,
        max_hours=max_hours,
    )

    if ppg_df.empty:
        raise RuntimeError(
            f"No PPG data loaded for gamer {gamer_id}."
        )

    # Normalize PPG
    sig = ppg_df["Red_Signal"].astype("float32").to_numpy()
    if cfg.data.normalize_signal:
        sig, _ = normalize_signal_array(
            sig,
            mode=cfg.data.normalization_mode,
        )

    ppg_df = ppg_df.copy()
    ppg_df["ppg_norm"] = sig

    # 2) Load annotations
    ann_df = load_annotations_for_gamer(gamer_id)

    # Collect (time, score) for sleepiness events (Value in 1..7)
    valid_anns = []
    for _, row in ann_df.iterrows():
        try:
            score = int(row["Value"])
        except (ValueError, TypeError):
            continue
        if 1 <= score <= 7:
            valid_anns.append((row["datetime"], score))

    if not valid_anns:
        raise RuntimeError(
            f"No numeric sleepiness annotations (1–7) found for gamer {gamer_id}."
        )

    backward_minutes = getattr(cfg.data, "backward_minutes", 10.0)
    stride_fraction = getattr(cfg.data, "window_stride_fraction", 0.5)
    stride = max(1, int(seq_len * stride_fraction))

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for t_ann, score in valid_anns:
        # PPG segment in the backward window
        t_start = t_ann - pd.Timedelta(minutes=backward_minutes)
        seg = ppg_df[
            (ppg_df["datetime"] >= t_start) &
            (ppg_df["datetime"] <= t_ann)
        ]

        values = seg["ppg_norm"].to_numpy().astype("float32")

        if len(values) < seq_len:
            # Not enough samples in this backward window
            continue

        # Build many overlapping windows from this segment
        for start in range(0, len(values) - seq_len + 1, stride):
            x_win = values[start:start + seq_len]  # (seq_len,)
            X_list.append(x_win)
            y_list.append(score - 1)  # 1..7 -> 0..6

    if not X_list:
        raise RuntimeError(
            f"No valid PPG windows found for gamer {gamer_id} with "
            f"seq_len={seq_len}, backward_minutes={backward_minutes}."
        )

    X = np.stack(X_list, axis=0)        # (N, seq_len)
    y = np.array(y_list, dtype="int64") # (N,)

    # Add feature dimension
    X = X[..., None]  # (N, seq_len, 1)

    return X, y
