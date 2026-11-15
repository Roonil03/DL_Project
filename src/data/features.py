import pandas as pd
import numpy as np
from typing import Tuple, Dict

def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculates daily returns from prices strictly backward-looking."""
    return prices.pct_change().fillna(0.0)

def calculate_rolling_volatility(returns: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates rolling annualized volatility strictly using past window."""
    # Keep the warm-up period missing. Backfilling here would copy a statistic
    # calculated with future observations into earlier timestamps.
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)

def calculate_momentum(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculates price momentum strictly using past window."""
    mom = (prices / prices.shift(window) - 1.0)
    return mom.fillna(0.0)

def build_features(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Constructs leak-free multivariate feature matrices for all assets.
    
    Returns:
        feature_df: DataFrame with shape (T, num_assets * num_features)
        returns_df: Daily returns DataFrame (T, num_assets)
        vol_df: 20-day rolling annualized volatility DataFrame (T, num_assets)
        feature_dict: Dictionary of individual feature DataFrames
    """
    rets = calculate_returns(prices)
    vol_20 = calculate_rolling_volatility(rets, 20)
    vol_60 = calculate_rolling_volatility(rets, 60)
    mom_5 = calculate_momentum(prices, 5)
    mom_20 = calculate_momentum(prices, 20)
    mom_60 = calculate_momentum(prices, 60)
    
    # Assemble feature columns per asset
    feature_dfs = []
    assets = prices.columns.tolist()
    
    for asset in assets:
        asset_feats = pd.DataFrame({
            f"{asset}_ret": rets[asset],
            f"{asset}_vol20": vol_20[asset],
            f"{asset}_vol60": vol_60[asset],
            f"{asset}_mom5": mom_5[asset],
            f"{asset}_mom20": mom_20[asset],
            f"{asset}_mom60": mom_60[asset],
        }, index=prices.index)
        feature_dfs.append(asset_feats)
        
    full_features = pd.concat(feature_dfs, axis=1)
    full_features = full_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove only the initial rolling warm-up rows, keeping every returned
    # object synchronized with the leakage-free feature index.
    valid_index = full_features.index
    rets = rets.loc[valid_index]
    vol_20 = vol_20.loc[valid_index]
    
    feature_dict = {
        'returns': rets,
        'vol_20': vol_20,
        'vol_60': vol_60.loc[valid_index],
        'mom_5': mom_5.loc[valid_index],
        'mom_20': mom_20.loc[valid_index],
        'mom_60': mom_60.loc[valid_index]
    }
    
    return full_features, rets, vol_20, feature_dict


def build_features_from_returns(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Build the same features from decimal daily returns without inventing prices.

    Missing rows are rejected instead of dropped: dropping one would turn the
    next available observation into a false next-trading-day target. Select a
    complete contiguous period or explicitly implement missing-data masks.
    """
    rets = returns.astype(float)
    if rets.empty or not rets.index.is_unique or not rets.index.is_monotonic_increasing:
        raise ValueError("Returns require a nonempty, unique, chronological index")
    if not np.isfinite(rets.to_numpy()).all() or (rets < -1).any().any():
        raise ValueError("Returns must be finite and at least -100%; do not forward-fill missing returns")
    vol_20 = calculate_rolling_volatility(rets, 20)
    vol_60 = calculate_rolling_volatility(rets, 60)
    momenta = {window: (1 + rets).rolling(window, min_periods=window).apply(np.prod, raw=True) - 1
               for window in (5, 20, 60)}
    feature_frames = []
    for asset in rets.columns:
        feature_frames.append(pd.DataFrame({
            f"{asset}_ret": rets[asset], f"{asset}_vol20": vol_20[asset],
            f"{asset}_vol60": vol_60[asset], f"{asset}_mom5": momenta[5][asset],
            f"{asset}_mom20": momenta[20][asset], f"{asset}_mom60": momenta[60][asset],
        }, index=rets.index))
    features = pd.concat(feature_frames, axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    idx = features.index
    parts = {'returns': rets.loc[idx], 'vol_20': vol_20.loc[idx], 'vol_60': vol_60.loc[idx],
             'mom_5': momenta[5].loc[idx], 'mom_20': momenta[20].loc[idx], 'mom_60': momenta[60].loc[idx]}
    return features, parts['returns'], parts['vol_20'], parts

