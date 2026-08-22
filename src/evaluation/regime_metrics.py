import pandas as pd
import numpy as np
from typing import Dict, Any
from .metrics import calculate_sharpe_ratio, calculate_sortino_ratio, calculate_max_drawdown, calculate_annualized_return

def evaluate_by_regime(returns: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """
    Groups returns by regime and calculates metrics for each.
    """
    aligned = pd.DataFrame({'returns': returns, 'regime': regimes}).dropna()
    
    results = []
    for regime, group in aligned.groupby('regime'):
        if len(group) < 2:
            continue
            
        r = group['returns']
        results.append({
            'regime': regime,
            'observations': len(r),
            'sharpe': calculate_sharpe_ratio(r),
            'sortino': calculate_sortino_ratio(r),
            'max_drawdown': calculate_max_drawdown(r),
            'annual_return': calculate_annualized_return(r)
        })
        
    if len(results) == 0:
        return pd.DataFrame(columns=['observations', 'sharpe', 'sortino', 'max_drawdown', 'annual_return'])
        
    df_res = pd.DataFrame(results).set_index('regime')
    return df_res

def calculate_regime_robustness(returns: pd.Series, regimes: pd.Series) -> Dict[str, float]:
    """
    Calculates Regime Robustness Score and Stress Degradation.
    Degradation = Sharpe(Normal/LowVol_Bull) - Sharpe(Stress)
    Robustness Score = Minimum Sharpe across all regimes / Maximum Sharpe across all regimes
    """
    regime_df = evaluate_by_regime(returns, regimes)
    if regime_df.empty:
        return {'robustness_score': 0.0, 'stress_degradation': 0.0}
        
    sharpes = regime_df['sharpe']
    min_sharpe = sharpes.min()
    max_sharpe = sharpes.max()
    
    # Stress Sharpe vs Normal Sharpe
    stress_sharpe = sharpes.get('Extreme_Stress', sharpes.get('HighVol_Bear', 0.0))
    normal_sharpe = sharpes.get('LowVol_Bull', sharpes.mean())
    
    degradation = float(normal_sharpe - stress_sharpe)
    robustness_score = float(min_sharpe / (abs(max_sharpe) + 1e-6))
    
    return {
        'robustness_score': robustness_score,
        'stress_degradation': degradation,
        'min_regime_sharpe': float(min_sharpe),
        'max_regime_sharpe': float(max_sharpe)
    }

