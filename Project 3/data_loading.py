from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

from config import config, RAW_CSV_PATH


def _infer_sample_rate_from_timestamps(
    timestamps: pd.Series,
) -> float:
    """
    Infer sample rate in Hz from a datetime-like pandas Series.

    We compute the median difference between consecutive timestamps and
    invert it to get samples per second. This assumes fairly regular sampling.
    """
    # Convert to ns (int64), then differences
    diffs = timestamps.sort_values().view("int64").diff().dropna()
    if diffs.empty:
        raise ValueError("Not enough timestamps to infer sample rate.")

    median_diff_ns = diffs.median()
    # seconds between samples
    seconds_per_sample = median_diff_ns / 1e9
    if seconds_per_sample <= 0:
        raise ValueError(f"Invalid time differences; got {seconds_per_sample} seconds per sample.")

    sample_rate_hz = 1.0 / seconds_per_sample
    return float(sample_rate_hz)

def _limit_by_hours_with_timestamps(
    df: pd.DataFrame,
    timestamp_col: str,
    limit_hours: float,
) -> pd.DataFrame:
    """
    Keep only the first `limit_hours` hours of data based on the timestamp column.
    """
    df = df.sort_values(timestamp_col)
    start_time = df[timestamp_col].iloc[0]
    end_time = start_time + pd.Timedelta(hours=limit_hours)
    mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] < end_time)
    return df.loc[mask].copy()

def _limit_by_hours_with_fixed_sr(
    df: pd.DataFrame,
    limit_hours: float,
    sample_rate_hz: float,
) -> pd.DataFrame:
    """
    Keep only the first N samples that correspond to `limit_hours`, assuming
    a known and constant sample rate.
    """
    total_seconds = limit_hours * 60.0 * 60.0
    n_samples = int(sample_rate_hz * total_seconds)
    return df.iloc[:n_samples].copy()

def load_ppg(
    csv_path: Path = RAW_CSV_PATH,
) -> Tuple[np.ndarray, Optional[pd.Series], float, Dict[str, Any]]:

    data_cfg = config.data

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"[data_loading] Reading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Basic checks
    if data_cfg.ppg_col not in df.columns:
        raise ValueError(
            f"PPG column '{data_cfg.ppg_col}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    has_timestamp = data_cfg.timestamp_col is not None and data_cfg.timestamp_col in df.columns

    timestamps: Optional[pd.Series] = None
    sample_rate_hz: float

    if has_timestamp:
        # Parse timestamps
        print(f"[data_loading] Using timestamp column: {data_cfg.timestamp_col}")
        df[data_cfg.timestamp_col] = pd.to_datetime(df[data_cfg.timestamp_col], errors="coerce")
        # Drop rows with invalid timestamps
        df = df.dropna(subset=[data_cfg.timestamp_col])

        if df.empty:
            raise ValueError("All timestamps are NaT after parsing. Check the timestamp format.")

        # Limit by hours
        df = _limit_by_hours_with_timestamps(df, data_cfg.timestamp_col, data_cfg.limit_hours)

        if df.empty:
            raise ValueError("No data remains after limiting to the given number of hours.")

        # Infer sample rate
        sample_rate_hz = _infer_sample_rate_from_timestamps(df[data_cfg.timestamp_col])
        timestamps = df[data_cfg.timestamp_col].reset_index(drop=True)

        print(f"[data_loading] Inferred sample rate: {sample_rate_hz:.2f} Hz")

    else:
        # No timestamp column: we assume a fixed sample rate from config
        sample_rate_hz = float(data_cfg.sample_rate_hz)
        print(
            "[data_loading] No valid timestamp column found. "
            f"Using fixed sample rate from config: {sample_rate_hz:.2f} Hz"
        )

        df = _limit_by_hours_with_fixed_sr(df, data_cfg.limit_hours, sample_rate_hz)

        if df.empty:
            raise ValueError("No data remains after limiting to the given number of hours.")

    # Extract PPG signal
    ppg = df[data_cfg.ppg_col].astype("float32").to_numpy()

    if ppg.ndim != 1:
        raise ValueError("Expected PPG column to be 1D.")

    info: Dict[str, Any] = {
        "csv_path": str(csv_path),
        "n_samples": int(len(ppg)),
        "limit_hours": float(data_cfg.limit_hours),
        "timestamp_used": has_timestamp,
    }

    print(f"[data_loading] Loaded {info['n_samples']} samples after trimming.")

    return ppg, timestamps, sample_rate_hz, info
