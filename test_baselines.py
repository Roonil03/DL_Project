import numpy as np
import pandas as pd
from src.backtest.backtester import Backtester
from src.portfolio.baselines import (
    get_zero_signal_baseline,
    get_long_only_baseline,
    get_momentum_baseline,
    get_volatility_managed_baseline
)

# Dummy test data (T=100 days, 3 assets)
np.random.seed(42)
mock_returns = np.random.normal(0.0005, 0.01, size=(100, 3))
mock_vol = np.full((100, 3), 0.15)

backtester = Backtester(target_vol=0.10, tc_bps=5.0)

# 1. Zero-Signal Baseline
sig_zero = get_zero_signal_baseline(mock_returns)
res_zero = backtester.run(sig_zero, mock_vol, mock_returns)

# 2. Long-Only Baseline
sig_long = get_long_only_baseline(mock_returns)
res_long = backtester.run(sig_long, mock_vol, mock_returns)

# 3. Momentum Baseline
sig_mom = get_momentum_baseline(mock_returns, lookback=10)
res_mom = backtester.run(sig_mom, mock_vol, mock_returns)

# 4. Volatility-Managed Baseline
sig_vol_managed = get_volatility_managed_baseline(mock_returns, target_vol=0.10, lookback=20)
res_vol_managed = backtester.run(sig_vol_managed, mock_vol, mock_returns)

print("All 4 Strategy Baselines Executed Successfully!")
print("Zero-Signal Cumulative Return:     ", res_zero["cumulative_returns"].iloc[-1])
print("Long-Only Cumulative Return:       ", res_long["cumulative_returns"].iloc[-1])
print("Momentum Cumulative Return:        ", res_mom["cumulative_returns"].iloc[-1])
print("Volatility-Managed Cumulative Ret:", res_vol_managed["cumulative_returns"].iloc[-1])