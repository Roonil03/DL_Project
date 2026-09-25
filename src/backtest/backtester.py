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
        
        if dates is not None and len(dates) == len(port_net):
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

