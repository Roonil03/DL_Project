"""Chronological expanding and rolling walk-forward split generation."""
from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np


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
