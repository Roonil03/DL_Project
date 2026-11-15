"""Chronological expanding and rolling walk-forward split generation."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd


def walk_forward_splits(n_samples: int, initial_train_size: int, test_size: int,
                        step_size: int | None = None, mode: str = "expanding",
                        train_window: int | None = None, gap: int = 0) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield non-overlapping train/test indices without future leakage.

    ``gap`` omits observations between the end of training and test start, for
    labels whose forecast horizon extends beyond the input timestamp.
    """
    if n_samples <= 0 or initial_train_size <= 0 or test_size <= 0 or gap < 0:
        raise ValueError("sizes must be positive and gap must be non-negative")
    if mode not in {"expanding", "rolling"}:
        raise ValueError("mode must be 'expanding' or 'rolling'")
    if mode == "rolling" and (train_window is None or train_window <= 0):
        train_window = initial_train_size
    step_size = test_size if step_size is None else step_size
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    train_end = initial_train_size
    while train_end + gap + test_size <= n_samples:
        train_start = 0 if mode == "expanding" else max(0, train_end - int(train_window))
        yield np.arange(train_start, train_end), np.arange(train_end + gap, train_end + gap + test_size)
        train_end += step_size


def iter_walk_forward(*args, **kwargs):
    """Compatibility-friendly alias for :func:`walk_forward_splits`."""
    return walk_forward_splits(*args, **kwargs)


def calendar_crisis_folds(decision_dates, target_dates, test_years=(2008, 2020), train_years=5):
    """Return expanding-free train/validation/test indices for complete test years.

    The preceding calendar year is validation and the five preceding years
    are training by default. Assign samples by *target* date and require the
    decision to be inside the same partition to avoid boundary leakage.
    """
    decisions, targets = pd.DatetimeIndex(decision_dates), pd.DatetimeIndex(target_dates)
    if len(decisions) != len(targets) or not len(decisions):
        raise ValueError("Decision and target dates must have equal nonzero length")
    if (not decisions.is_monotonic_increasing or not targets.is_monotonic_increasing
            or decisions.has_duplicates or targets.has_duplicates or (targets <= decisions).any()):
        raise ValueError("Decision and target dates must be unique, chronological, and target after decision")
    if train_years < 1:
        raise ValueError("train_years must be positive")
    folds = []
    for year in test_years:
        start = pd.Timestamp(year=int(year), month=1, day=1)
        end = start + pd.DateOffset(years=1)
        val_start = start - pd.DateOffset(years=1)
        train_start = val_start - pd.DateOffset(years=train_years)
        if decisions.min() > train_start + pd.Timedelta(days=7) or targets.max() < end - pd.Timedelta(days=8):
            raise ValueError(f"Insufficient coverage for complete {year} test fold and training history")
        def indices(left, right):
            return np.flatnonzero((decisions >= left) & (targets < right) & (targets >= left))
        fold = tuple(indices(left, right) for left, right in ((train_start, val_start), (val_start, start), (start, end)))
        if any(len(part) == 0 for part in fold):
            raise ValueError(f"Empty train, validation, or test partition for {year}")
        folds.append(fold)
    return folds
