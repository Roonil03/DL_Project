import pandas as pd
import numpy as np

def detect_regimes(prices: pd.DataFrame, vol_window=60, trend_window=120) -> pd.Series:
    """
    Classifies market environments using exclusively historical information at time t.
    Creates 4 primary regimes:
    1: Low Volatility + Positive Trend
    2: High Volatility + Positive Trend
    3: Low Volatility + Negative Trend
    4: High Volatility + Negative Trend
    """
    # Calculate rolling metrics
    returns = prices.pct_change()
    volatility = returns.rolling(window=vol_window).std() * np.sqrt(252)
    trend = (prices / prices.shift(trend_window) - 1)
    
    # We need a market proxy. If prices is a dataframe of multiple assets, 
    # we can use an equal-weight average or a specific index if provided.
    # For simplicity, assuming 'prices' here represents the market index.
    market_vol = volatility.mean(axis=1) if isinstance(volatility, pd.DataFrame) else volatility
    market_trend = trend.mean(axis=1) if isinstance(trend, pd.DataFrame) else trend
    
    # Define thresholds (expanding window median to prevent look-ahead bias)
    vol_threshold = market_vol.expanding().median()
    
    regimes = pd.Series(index=prices.index, dtype=str)
    
    # Assign labels
    regimes[(market_vol <= vol_threshold) & (market_trend > 0)] = "LowVol_Bull"
    regimes[(market_vol > vol_threshold) & (market_trend > 0)] = "HighVol_Bull"
    regimes[(market_vol <= vol_threshold) & (market_trend <= 0)] = "LowVol_Bear"
    regimes[(market_vol > vol_threshold) & (market_trend <= 0)] = "HighVol_Bear"
    
    # Optional 5th regime: Extreme Stress (e.g. > 95th percentile of expanding vol)
    stress_threshold = market_vol.expanding().quantile(0.95)
    regimes[market_vol > stress_threshold] = "Extreme_Stress"
    
    return regimes.ffill()
