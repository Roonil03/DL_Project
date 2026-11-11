import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from src.portfolio.volatility_targeting import apply_volatility_target
from src.portfolio.transaction_costs import (
    calculate_position_turnover,
    calculate_transaction_costs,
)

class Backtester:
    def __init__(self, target_vol: float = 0.10, max_leverage: float = 3.0, tc_bps: float = 5.0):
        if not np.isfinite(target_vol) or target_vol <= 0:
            raise ValueError('target_vol must be positive and finite')
        if not np.isfinite(max_leverage) or max_leverage <= 0:
            raise ValueError('max_leverage must be positive and finite')
        if not np.isfinite(tc_bps) or tc_bps < 0:
            raise ValueError('tc_bps must be non-negative and finite')
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.tc_bps = tc_bps

    def run(
        self, 
        signals: np.ndarray, 
        ex_ante_vol: np.ndarray, 
        next_period_returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None
    ) -> Dict[str, Any]:
        """
        Executes a vectorized chronological backtest.
        
        Args:
            signals: shape (T, assets) at decision time t
            ex_ante_vol: shape (T, assets) realized volatility up to t
            next_period_returns: shape (T, assets) asset returns realized at t+1
            dates: Optional DatetimeIndex for timestamp alignment
            
        Returns:
            Dictionary with positions, gross_returns, net_returns, turnover, cumulative_returns
        """
        signals = np.asarray(signals, dtype=float)
        ex_ante_vol = np.asarray(ex_ante_vol, dtype=float)
        next_period_returns = np.asarray(next_period_returns, dtype=float)
        if signals.ndim != 2 or ex_ante_vol.ndim != 2 or next_period_returns.ndim != 2:
            raise ValueError('signals, ex_ante_vol, and next_period_returns must be 2D arrays')
        if signals.shape != ex_ante_vol.shape or signals.shape != next_period_returns.shape:
            raise ValueError('signals, ex_ante_vol, and next_period_returns must have matching shapes')
        if signals.shape[0] == 0 or signals.shape[1] == 0:
            raise ValueError('backtest inputs must contain at least one time step and asset')
        if not all(np.isfinite(values).all() for values in (signals, ex_ante_vol, next_period_returns)):
            raise ValueError('backtest inputs must contain only finite values')
        if dates is not None and len(dates) != signals.shape[0]:
            raise ValueError('dates must have the same time length as backtest inputs')

        # 1. Size positions based strictly on information available at t
        positions = apply_volatility_target(signals, ex_ante_vol, self.target_vol, self.max_leverage)
        
        # 2. Compute gross returns at t+1
        gross_asset_returns = positions * next_period_returns
        
        # 3. Compute transaction costs incurred when adjusting positions from t-1 to t
        asset_turnover = calculate_position_turnover(positions)
        costs = calculate_transaction_costs(
            positions, self.tc_bps, turnover=asset_turnover
        )
        
        # 4. Compute net returns
        net_asset_returns = gross_asset_returns - costs
        
        # 5. Equal capital-weighted portfolio aggregation across assets
        port_gross = np.nanmean(gross_asset_returns, axis=1)
        port_net = np.nanmean(net_asset_returns, axis=1)
        
        # Turnover calculation: mean absolute change in leverage
        turnover_per_step = np.nanmean(asset_turnover, axis=1)
        
        if dates is not None:
            gross_series = pd.Series(port_gross, index=dates, name='Gross_Return')
            net_series = pd.Series(port_net, index=dates, name='Net_Return')
            turnover_series = pd.Series(turnover_per_step, index=dates, name='Turnover')
            pos_df = pd.DataFrame(positions, index=dates)
        else:
            gross_series = pd.Series(port_gross)
            net_series = pd.Series(port_net)
            turnover_series = pd.Series(turnover_per_step)
            pos_df = pd.DataFrame(positions)
            
        cum_net = (1.0 + net_series).cumprod()
        
        return {
            "positions": pos_df,
            "gross_returns": gross_series,
            "net_returns": net_series,
            "turnover": turnover_series,
            "cumulative_returns": cum_net
        }

