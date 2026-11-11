import numpy as np
import pandas as pd
from typing import Dict, Union

def _finite_returns(returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError('returns must be a one-dimensional series')
    return values[np.isfinite(values)]

def _periods_per_year(value: int) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError('periods_per_year must be positive and finite')
    return float(value)

def calculate_annualized_return(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    ret_arr = _finite_returns(returns)
    if ret_arr.size == 0:
        return 0.0
    return float(np.mean(ret_arr) * _periods_per_year(periods_per_year))

def calculate_annualized_volatility(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    ret_arr = _finite_returns(returns)
    if ret_arr.size < 2:
        return 0.0
    return float(np.std(ret_arr, ddof=1) * np.sqrt(_periods_per_year(periods_per_year)))

def calculate_sharpe_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    vol = calculate_annualized_volatility(returns, periods_per_year)
    if vol <= 1e-8 or np.isnan(vol):
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / vol)

def calculate_sortino_ratio(returns: Union[pd.Series, np.ndarray], risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    ret_arr = _finite_returns(returns)
    if ret_arr.size == 0:
        return 0.0
    # Zero minimum acceptable return; lower partial second moment over every
    # observation, including those above the target.
    target = risk_free_rate / _periods_per_year(periods_per_year)
    downside_vol = float(np.sqrt(np.mean(np.minimum(ret_arr - target, 0.0) ** 2)) * np.sqrt(periods_per_year))
    if downside_vol <= 1e-8 or np.isnan(downside_vol):
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float((ann_ret - risk_free_rate) / downside_vol)

def calculate_max_drawdown(returns: Union[pd.Series, np.ndarray]) -> float:
    ret_arr = _finite_returns(returns)
    if ret_arr.size == 0:
        return 0.0
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + ret_arr)))
    peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - peak) / np.where(peak > 0, peak, 1.0)
    return float(np.min(drawdown))

def calculate_calmar_ratio(returns: Union[pd.Series, np.ndarray], periods_per_year: int = 252) -> float:
    mdd = abs(calculate_max_drawdown(returns))
    if mdd <= 1e-8:
        return 0.0
    ann_ret = calculate_annualized_return(returns, periods_per_year)
    return float(ann_ret / mdd)

def calculate_win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    valid = _finite_returns(returns)
    if valid.size == 0:
        return 0.0
    return float(np.sum(valid > 0) / len(valid))

def calculate_profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    ret_arr = _finite_returns(returns)
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
        turnover_values = np.asarray(turnover, dtype=float)
        if turnover_values.ndim != 1:
            raise ValueError('turnover must be a one-dimensional series')
        finite_turnover = turnover_values[np.isfinite(turnover_values)]
        metrics['Annualized Turnover'] = (
            float(np.mean(finite_turnover) * 252) if finite_turnover.size else 0.0
        )
    return metrics

