import numpy as np
from src.portfolio.volatility_targeting import apply_volatility_target
from src.portfolio.transaction_costs import calculate_transaction_costs

class Backtester:
    def __init__(self, target_vol=0.10, max_leverage=3.0, tc_bps=0.0):
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.tc_bps = tc_bps

    def run(self, signals, ex_ante_vol, next_period_returns):
        """
        Executes a vectorized backtest enforcing chronology.
        
        signals: shape (T, assets) at time t
        ex_ante_vol: shape (T, assets) at time t
        next_period_returns: shape (T, assets) realized at time t+1
        """
        # 1. Size positions based on data available at t
        positions = apply_volatility_target(signals, ex_ante_vol, self.target_vol, self.max_leverage)
        
        # 2. Compute gross returns at t+1
        gross_returns = positions * next_period_returns
        
        # 3. Compute transaction costs incurred to enter positions at t
        costs = calculate_transaction_costs(positions, self.tc_bps)
        
        # 4. Compute net returns
        net_returns = gross_returns - costs
        
        # 5. Portfolio aggregation (Equal capital weighting across valid assets)
        # Using nanmean to ignore missing assets
        port_gross_returns = np.nanmean(gross_returns, axis=1)
        port_net_returns = np.nanmean(net_returns, axis=1)
        
        return {
            "positions": positions,
            "gross_returns": port_gross_returns,
            "net_returns": port_net_returns,
            "turnover": np.nanmean(np.abs(positions - np.roll(positions, 1, axis=0)), axis=1)
        }
