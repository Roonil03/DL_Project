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
    X, y = [], []
    valid_dates = []
    
    T = len(features)
    for i in range(T - lookback):
        # Window covers time [i : i + lookback]
        X.append(features[i : i + lookback])
        # Target return is at time i + lookback (t+1 return after decision at t)
        y.append(targets[i + lookback])
        if dates is not None:
            # Decision time t is at index i + lookback - 1
            valid_dates.append(dates[i + lookback - 1])
            
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.float32)
    decision_dates = pd.DatetimeIndex(valid_dates) if dates is not None else None
    
    return X_arr, y_arr, decision_dates

