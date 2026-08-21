import pandas as pd
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
        
    return pd.DataFrame(results).set_index('regime')
