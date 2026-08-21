import pandas as pd
import numpy as np

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily returns from prices."""
    return prices.pct_change().fillna(0)

def calculate_rolling_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates rolling annualized volatility."""
    return returns.rolling(window=window, min_periods=1).std() * np.sqrt(252)

def calculate_momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates price momentum."""
    return (prices / prices.shift(window) - 1).fillna(0)

def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs all required features (Returns, Volatility, Momentum).
    Does NOT use future information.
    """
    rets = calculate_returns(prices)
    vol_20 = calculate_rolling_volatility(rets, 20)
    mom_20 = calculate_momentum(prices, 20)
    
    # Combine (for a real implementation, we'd use MultiIndex or concatenate columns)
    # Placeholder for structure
    return rets, vol_20, mom_20
