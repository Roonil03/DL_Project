"""Predictive, crisis, rank-stability, and regime-analysis helpers."""
from __future__ import annotations

from typing import Dict, Mapping, Sequence
import numpy as np
import pandas as pd

from .metrics import calculate_sharpe_ratio


def predictive_metrics(predictions, targets) -> Dict[str, float]:
    """MSE, directional accuracy and Pearson signal/return correlation."""
    prediction = np.asarray(predictions, dtype=float).reshape(-1)
    target = np.asarray(targets, dtype=float).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction, target = prediction[valid], target[valid]
    if not len(target):
        return {"mse": 0.0, "directional_accuracy": 0.0, "signal_return_correlation": 0.0, "observations": 0}
    correlation = 0.0 if len(target) < 2 or np.std(prediction) < 1e-12 or np.std(target) < 1e-12 else float(np.corrcoef(prediction, target)[0, 1])
    return {"mse": float(np.mean((prediction - target) ** 2)),
            "directional_accuracy": float(np.mean(np.sign(prediction) == np.sign(target))),
            "signal_return_correlation": correlation, "observations": int(len(target))}


def crisis_window_metrics(returns: pd.Series, crisis_start=None, crisis_end=None, crisis_mask=None) -> Dict[str, float]:
    """Summarize return behavior over an explicit crisis date range or mask."""
    series = pd.Series(returns).dropna()
    if crisis_mask is not None:
        selected = series.loc[np.asarray(crisis_mask, dtype=bool)]
    elif crisis_start is not None or crisis_end is not None:
        selected = series.loc[crisis_start:crisis_end]
    else:
        raise ValueError("provide crisis_start/crisis_end or crisis_mask")
    return {"observations": int(len(selected)), "cumulative_return": float((1 + selected).prod() - 1) if len(selected) else 0.0,
            "sharpe": calculate_sharpe_ratio(selected), "worst_daily_return": float(selected.min()) if len(selected) else 0.0}


def architecture_rank_stability(scores: pd.DataFrame | Mapping[str, Sequence[float]]) -> Dict[str, object]:
    """Ranks models per fold and reports mean rank plus cross-fold stability."""
    frame = pd.DataFrame(scores)
    if frame.empty:
        return {"mean_rank": {}, "rank_std": {}, "rank_correlation": 0.0}
    ranks = frame.rank(axis=1, ascending=False, method="average")
    correlations = ranks.T.corr().to_numpy()
    upper = correlations[np.triu_indices_from(correlations, 1)] if len(frame) > 1 else np.array([1.0])
    return {"mean_rank": ranks.mean().to_dict(), "rank_std": ranks.std(ddof=0).to_dict(),
            "rank_correlation": float(np.nanmean(upper)) if len(upper) else 1.0, "ranks": ranks}
