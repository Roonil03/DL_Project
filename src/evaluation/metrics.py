import numpy as np
import pandas as pd
from typing import Dict, Union

def calculate_annualized_return(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    ret_arr = np.asarray(returns)
    if len(ret_arr) == 0:
        return 0.0
    return float(np.nanmean(ret_arr) * periods_per_year)

def calculate_annualized_volatility(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    ret_arr = np.asarray(returns)
    if len(ret_arr) < 2:
        return 0.0
    return float(np.nanstd(ret_arr, ddof=1) * np.sqrt(periods_per_year))

def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    vol = calculate_annualized_volatility(returns, periods_per_year)
    if vol <= 1e-8 or np.isnan(vol):
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / vol)

def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    ret_arr = np.asarray(returns)
    downside = ret_arr[ret_arr < 0]
    if len(downside) < 2:
        return 0.0
    downside_vol = float(np.nanstd(downside, ddof=1) * np.sqrt(periods_per_year))
    if downside_vol <= 1e-8 or np.isnan(downside_vol):
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / downside_vol)

def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    ret_arr = np.asarray(returns)
    if len(ret_arr) == 0:
        return 0.0
    cum_returns = np.cumprod(1.0 + np.nan_to_num(ret_arr))
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(drawdown))

def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    mdd = abs(calculate_max_drawdown(returns))
    if mdd <= 1e-8:
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float(ann_ret / mdd)

def calculate_win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    ret_arr = np.asarray(returns)
    valid = ret_arr[~np.isnan(ret_arr)]
    if len(valid) == 0:
        return 0.0
    return float(np.sum(valid > 0) / len(valid))

def calculate_profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    ret_arr = np.asarray(returns)
    gains = np.sum(ret_arr[ret_arr > 0])
    losses = np.abs(np.sum(ret_arr[ret_arr < 0]))
    if losses <= 1e-8:
        return 10.0 if gains > 0 else 1.0
    return float(gains / losses)

def compute_all_metrics(returns: Union[pd.Series, np.ndarray], turnover: Union[pd.Series, np.ndarray] = None) -> Dict[str, float]:
    metrics = {
        'Annualized Return': calculate_annualized_return(returns),
        'Annualized Volatility': calculate_annualized_volatility(returns),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(returns),
        'Calmar Ratio': calculate_calmar_ratio(returns),
        'Win Rate': calculate_win_rate(returns),
        'Profit Factor': calculate_profit_factor(returns)
    }
    if turnover is not None:
        metrics['Annualized Turnover'] = float(np.nanmean(turnover) * 252)
    return metrics

