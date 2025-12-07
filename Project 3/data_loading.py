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
    max_hours: Optional[float] = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """
    Load and concatenate all PPG CSVs for a gamer into a single DataFrame.

    - Parses 'Time' strings and attaches the date inferred from filename.
    - Combines all days, sorts by datetime.
    - Optionally trims to the first `max_hours` of data starting from the
      earliest timestamp.

    Returns a DataFrame with at least columns:
        ['datetime', 'Time', 'Red_Signal', ...]
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

        date_str = _extract_date_from_ppg_filename(path)
        # Combine date from filename with time-of-day in the CSV
        dt_str = date_str + " " + df["Time"].astype(str)
        df["datetime"] = pd.to_datetime(dt_str)

        dfs.append(df)

    full = pd.concat(dfs, axis=0, ignore_index=True)
    full = full.sort_values("datetime").reset_index(drop=True)

    # Trim to first `max_hours` if requested
    if max_hours is not None:
        start = full["datetime"].min()
        cutoff = start + pd.Timedelta(hours=max_hours)
        full = full[full["datetime"] <= cutoff].copy().reset_index(drop=True)

    return full


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
    """
    x = x.astype("float32")
    stats: Dict[str, float] = {}

    if mode == "zscore":
        mean = float(x.mean())
        std = float(x.std())
        if std == 0.0:
            std = 1.0
        x_norm = (x - mean) / std
        stats = {"mean": mean, "std": std}

    elif mode == "minmax":
        min_val = float(x.min())
        max_val = float(x.max())
        if max_val == min_val:
            x_norm = x - min_val
        else:
            x_norm = (x - min_val) / (max_val - min_val)
        stats = {"min": min_val, "max": max_val}

    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    return x_norm.astype("float32"), stats


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
