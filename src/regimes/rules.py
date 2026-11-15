import pandas as pd
import numpy as np

def detect_regimes(prices: pd.DataFrame, vol_window=60, trend_window=120, train_end=None) -> pd.Series:
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
    
    # Fix thresholds from training only for a benchmark fold. The expanding
    # historical mode is retained for callers without a fitted train boundary.
    if train_end is None:
        vol_threshold = market_vol.expanding().median()
        stress_threshold = market_vol.expanding().quantile(0.95)
    else:
        training_vol = market_vol.loc[:train_end].dropna()
        if training_vol.empty:
            raise ValueError('no finite training volatility to fit regime thresholds')
        vol_threshold = training_vol.median()
        stress_threshold = training_vol.quantile(0.95)
    
    regimes = pd.Series(index=prices.index, dtype=str)
    
    # Assign labels
    regimes[(market_vol <= vol_threshold) & (market_trend > 0)] = "LowVol_Bull"
    regimes[(market_vol > vol_threshold) & (market_trend > 0)] = "HighVol_Bull"
    regimes[(market_vol <= vol_threshold) & (market_trend <= 0)] = "LowVol_Bear"
    regimes[(market_vol > vol_threshold) & (market_trend <= 0)] = "HighVol_Bear"
    
    # Optional 5th regime: Extreme Stress (e.g. > 95th percentile of expanding vol)
    regimes[market_vol > stress_threshold] = "Extreme_Stress"
    
    return regimes.ffill()
