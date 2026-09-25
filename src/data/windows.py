import numpy as np
import pandas as pd
from typing import Tuple, Optional

def create_rolling_windows(
    features: np.ndarray, 
    targets: np.ndarray, 
    lookback: int,
    dates: Optional[pd.Index] = None
) -> Tuple[np.ndarray, np.ndarray, Optional[pd.Index]]:
    """
    Creates (X, y) pairs with shape [batch, lookback, features].
    
    Args:
        features: shape (T, num_features)
        targets: shape (T, num_assets)
        lookback: sequence length L
        dates: optional DatetimeIndex of length T
        
    Returns:
        X: shape (T - lookback, lookback, features)
        y: shape (T - lookback, num_assets) (realized returns at t+1)
        decision_dates: DatetimeIndex corresponding to decision time t
    """
    T = len(features)
    sample_count = T - lookback

    if sample_count <= 0:
        # Preserve the historical empty-output behavior for short inputs.
        X_arr = np.array([], dtype=np.float32)
        y_arr = np.array([], dtype=np.float32)
        decision_dates = pd.DatetimeIndex([]) if dates is not None else None
        return X_arr, y_arr, decision_dates

    # sliding_window_view avoids constructing every window in Python. Its
    # output layout is [sample, feature, lookback], so move the window axis to
    # retain the public [sample, lookback, feature] contract. The final source
    # window has no next-period target and is therefore excluded.
    windows = np.lib.stride_tricks.sliding_window_view(
        features, window_shape=lookback, axis=0
    )[:-1]
    X_arr = np.ascontiguousarray(np.moveaxis(windows, -1, 1), dtype=np.float32)
    y_arr = np.ascontiguousarray(targets[lookback:T], dtype=np.float32)

    decision_dates = None
    if dates is not None:
        decision_dates = pd.DatetimeIndex(dates[lookback - 1 : T - 1])
    
    return X_arr, y_arr, decision_dates

