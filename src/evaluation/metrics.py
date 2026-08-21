import numpy as np
import pandas as pd

def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    # Use sum of log returns for geometric approximation or compound
    return float(np.mean(returns) * periods_per_year)

def calculate_annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    vol = calculate_annualized_volatility(returns, periods_per_year)
    if vol == 0:
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return (ann_ret - risk_free_rate) / vol

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    downside_returns = returns[returns < 0]
    downside_vol = np.std(downside_returns, ddof=1) * np.sqrt(periods_per_year)
    if downside_vol == 0 or np.isnan(downside_vol):
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return (ann_ret - risk_free_rate) / downside_vol

def calculate_max_drawdown(returns: pd.Series) -> float:
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.expanding(min_periods=1).max()
    drawdown = (cum_returns / peak) - 1
    return float(drawdown.min())
