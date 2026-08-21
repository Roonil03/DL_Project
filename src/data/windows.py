import numpy as np

def create_rolling_windows(features: np.ndarray, targets: np.ndarray, lookback: int):
    """
    Creates (X, y) pairs with shape [batch, lookback, features].
    
    Args:
        features: shape (T, num_assets * num_features)
        targets: shape (T, num_assets)
        lookback: sequence length L
        
    Returns:
        X: shape (T - lookback, lookback, features)
        y: shape (T - lookback, num_assets)
    """
    X, y = [], []
    for i in range(len(features) - lookback):
        X.append(features[i : i + lookback])
        # Target is at t+1 (which corresponds to index i + lookback)
        y.append(targets[i + lookback])
        
    return np.array(X), np.array(y)
