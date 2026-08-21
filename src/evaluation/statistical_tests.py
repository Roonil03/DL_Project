import numpy as np

def block_bootstrap_sharpe(returns: np.ndarray, block_size: int = 20, num_bootstraps: int = 1000):
    """
    Block bootstrap for dependent time-series returns to generate confidence intervals for Sharpe Ratio.
    """
    n = len(returns)
    num_blocks = n // block_size
    
    sharpes = []
    
    for _ in range(num_bootstraps):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n - block_size, size=num_blocks)
        
        boot_returns = []
        for idx in block_indices:
            boot_returns.extend(returns[idx : idx + block_size])
            
        boot_returns = np.array(boot_returns)
        
        mean_ret = np.mean(boot_returns) * 252
        vol_ret = np.std(boot_returns, ddof=1) * np.sqrt(252)
        
        if vol_ret > 0:
            sharpes.append(mean_ret / vol_ret)
            
    return np.percentile(sharpes, 2.5), np.percentile(sharpes, 97.5)
