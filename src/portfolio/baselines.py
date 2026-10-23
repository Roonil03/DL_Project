import numpy as np
import pandas as pd

def get_zero_signal_baseline(returns: np.ndarray) -> np.ndarray:
    """
    1. Zero-Signal Baseline: Flat position (0.0 allocation).
    """
    return np.zeros_like(returns)


def get_long_only_baseline(returns: np.ndarray) -> np.ndarray:
    """
    2. Long-Only Baseline: Constant +1.0 fully invested position.
    """
    return np.ones_like(returns)


def get_momentum_baseline(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    3. Simple Trend / Momentum Baseline: 
       Goes long (+1) if trailing rolling sum of returns is > 0, else short (-1).
    """
    T, N = returns.shape
    signals = np.zeros_like(returns)
    
    for t in range(lookback, T):
        # Trailing sum of returns over the lookback window
        trailing_ret = np.sum(returns[t - lookback : t], axis=0)
        signals[t] = np.sign(trailing_ret)
        
    return signals


def get_volatility_managed_baseline(returns: np.ndarray, target_vol: float = 0.10, lookback: int = 20) -> np.ndarray:
    """
    4. Volatility-Managed Baseline:
       Scales position inversely proportional to rolling realized volatility.
    """
    T, N = returns.shape
    signals = np.ones_like(returns)  # Base long direction
    ex_ante_vol = np.ones_like(returns)
    
    for t in range(lookback, T):
        window_ret = returns[t - lookback : t]
        # Annualized volatility estimate (assuming daily returns, sqrt(252))
        vol = np.std(window_ret, axis=0) * np.sqrt(252)
        ex_ante_vol[t] = np.where(vol < 1e-6, 1e-6, vol)
        
    # Scale signal by target vol / ex-ante vol ratio
    managed_signals = signals * (target_vol / ex_ante_vol)
    return np.clip(managed_signals, -3.0, 3.0)  # Capped at max leverage 3.0