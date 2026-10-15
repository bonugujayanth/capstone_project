#!/usr/bin/env python3
"""Add rolling-window temporal features to a telemetry dataset.

This keeps the model in a standard tabular LightGBM setup while giving it recent
history from the last 5/10/20 seconds. All rolling computations are done within
session or VM groups, ordered by timestamp, without using future rows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WINDOWS = [5, 10, 20]
DEFAULT_BASE_FEATURES = [
    "write_bytes_per_s",
    "inplace_ratio",
    "mean_write_entropy",
    "mean_entropy_delta",
    "files_modified",
    "distinct_dirs_touched",
    "cpu_pct",
]
SESSION_HINTS = [
    "session_id",
    "vm_id",
    "vm_name",
    "session",
    "capture_id",
    "source_session",
]


def resolve_session_column(df):
    for column in SESSION_HINTS:
        if column in df.columns:
            return column

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
        if timestamps.notna().any():
            sessions = pd.Series("session_0", index=df.index, dtype=object)
            current = 0
            previous = None
            for idx, value in timestamps.items():
                if pd.isna(value):
                    sessions.at[idx] = f"session_{current}"
                    continue
                if previous is not None and (value - previous) > pd.Timedelta(seconds=300):
                    current += 1
                sessions.at[idx] = f"session_{current}"
                previous = value
            return sessions

    return pd.Series("session_0", index=df.index, dtype=object)


def ensure_required_columns(df):
    """Fill in missing feature columns with defaults so the pipeline stays robust."""
    data = df.copy()
    if "inplace_ratio" not in data.columns:
        data["inplace_ratio"] = data.get("write_iops_fraction", 0.0).clip(0, 1)
    if "mean_write_entropy" not in data.columns:
        data["mean_write_entropy"] = 0.0
    if "mean_entropy_delta" not in data.columns:
        data["mean_entropy_delta"] = 0.0
    if "distinct_dirs_touched" not in data.columns:
        data["distinct_dirs_touched"] = data.get("directories_affected", 0)
    if "files_modified" not in data.columns:
        data["files_modified"] = data.get("files_touched_total", 0)
    if "cpu_pct" not in data.columns:
        data["cpu_pct"] = 0.0
    if "write_bytes_per_s" not in data.columns:
        data["write_bytes_per_s"] = 0.0
    return data


def _rolling_slope(series, window):
    if len(series) == 0:
        return pd.Series([], dtype=float)

    result = pd.Series(index=series.index, dtype=float)
    for pos in range(len(series)):
        start = max(0, pos - window + 1)
        window_values = series.iloc[start:pos + 1]
        if len(window_values) <= 1:
            slope = 0.0
        else:
            x = np.arange(len(window_values), dtype=float)
            slope, _ = np.polyfit(x, window_values.to_numpy(dtype=float), 1)
            slope = float(slope)
        result.iloc[pos] = slope
    return result


def _inplace_streak(series):
    streak = pd.Series(0.0, index=series.index, dtype=float)
    current = 0
    for idx, value in series.items():
        if value > 0.5:
            current += 1
        else:
            current = 0
        streak.at[idx] = float(current)
    return streak


def _cumulative_files_modified(series):
    cumulative = series.clip(lower=0).cumsum()
    idle_reset = series.eq(0).rolling(window=3, min_periods=1).sum().ge(3)
    return cumulative.mask(idle_reset.shift(fill_value=False), 0.0)


def add_rolling_features(df, windows=None, base_features=None, session_column=None):
    """Add rolling temporal features while preserving the original rows and labels."""
    if windows is None:
        windows = WINDOWS

    data = ensure_required_columns(df.copy())
    if session_column is None:
        session_series = resolve_session_column(data)
        if isinstance(session_series, pd.Series):
            data["session_id"] = session_series.to_numpy()
        else:
            data["session_id"] = session_series
    else:
        if session_column not in data.columns:
            raise ValueError(f"session column {session_column!r} not found")
        data["session_id"] = data[session_column]

    if "timestamp" in data.columns:
        data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")

    data = data.sort_values(["session_id", "timestamp"], kind="mergesort").reset_index(drop=True)

    first_index = data.groupby("session_id", sort=False).head(1).index
    if len(first_index):
        data = data.drop(index=first_index).reset_index(drop=True)

    selected = list(base_features) if base_features else DEFAULT_BASE_FEATURES
    feature_pool = [feature for feature in selected if feature in data.columns]
    feature_pool = list(dict.fromkeys(feature_pool))

    for feature in feature_pool:
        for window in windows:
            grouped = data.groupby("session_id", sort=False)[feature]
            data[f"{feature}_mean_{window}s"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).mean()
            )
            data[f"{feature}_max_{window}s"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).max()
            )
            data[f"{feature}_std_{window}s"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).std(ddof=0).fillna(0.0)
            )
            data[f"{feature}_sum_{window}s"] = grouped.transform(
                lambda s: s.rolling(window=window, min_periods=1).sum()
            )
            data[f"{feature}_delta_{window}s"] = grouped.transform(
                lambda s: s - s.shift(window)
            ).fillna(0.0)
            data[f"{feature}_slope_{window}s"] = grouped.transform(
                lambda s: _rolling_slope(s, window)
            ).fillna(0.0)

    if "inplace_ratio" in data.columns:
        data["inplace_streak"] = data.groupby("session_id", sort=False)["inplace_ratio"].transform(
            lambda s: _inplace_streak(s)
        )

    if "files_modified" in data.columns:
        data["cumulative_files_modified"] = data.groupby("session_id", sort=False)["files_modified"].transform(
            lambda s: _cumulative_files_modified(s)
        )

    return data


def main():
    parser = argparse.ArgumentParser(description="Add rolling temporal features to a telemetry dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/labeled_training.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled_training_rolling.csv"))
    parser.add_argument("--windows", nargs="*", type=int, default=[5, 10, 20])
    parser.add_argument("--session-column", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    enriched = add_rolling_features(
        df,
        windows=args.windows,
        session_column=args.session_column,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(args.output, index=False)

    base_count = len(df.columns)
    added = len(enriched.columns) - base_count
    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(enriched)}")
    print(f"Added rolling features: {added}")
    print(f"Final feature count: {len(enriched.columns)}")
    print(f"Saved enriched dataset to {args.output}")
    print("Comment: these rolling features give LightGBM temporal memory over the last 5/10/20 seconds so it responds to sustained ransomware build-up instead of single-window spikes.")


if __name__ == "__main__":
    main()
