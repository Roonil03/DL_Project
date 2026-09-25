import numpy as np
from typing import Tuple, Dict

def block_bootstrap_sharpe(returns: np.ndarray, block_size: int = 20, num_bootstraps: int = 1000, random_seed: int = 42) -> Tuple[float, float, float]:
    """
    Stationary block bootstrap for time-series returns to generate 95% confidence intervals for Sharpe Ratio.
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    ret_arr = np.asarray(returns)
    ret_arr = ret_arr[~np.isnan(ret_arr)]
    n = len(ret_arr)
    if n < block_size * 2:
        return 0.0, 0.0, 0.0
        
    point_mean = np.mean(ret_arr) * 252.0
    point_vol = np.std(ret_arr, ddof=1) * np.sqrt(252.0)
    point_sharpe = float(point_mean / point_vol) if point_vol > 1e-8 else 0.0
    
    num_blocks = max(1, n // block_size)
    block_offsets = np.arange(block_size)
    sharpes = []
    
    for _ in range(num_bootstraps):
        start_indices = np.random.randint(0, n - block_size + 1, size=num_blocks)
        block_indices = start_indices[:, None] + block_offsets
        boot_arr = ret_arr[block_indices.ravel()]
        mean_ret = np.mean(boot_arr) * 252.0
        vol_ret = np.std(boot_arr, ddof=1) * np.sqrt(252.0)
        
        if vol_ret > 1e-8:
            sharpes.append(mean_ret / vol_ret)
            
    if len(sharpes) == 0:
        return point_sharpe, point_sharpe, point_sharpe
        
    ci_lower = float(np.percentile(sharpes, 2.5))
    ci_upper = float(np.percentile(sharpes, 97.5))
    return point_sharpe, ci_lower, ci_upper

def paired_sharpe_difference_test(returns_a: np.ndarray, returns_b: np.ndarray, num_bootstraps: int = 1000) -> Dict[str, float]:
    """
    Paired bootstrap test for difference in Sharpe ratio: H0: Sharpe(A) <= Sharpe(B) vs H1: Sharpe(A) > Sharpe(B).
    """
    r_a = np.asarray(returns_a)
    r_b = np.asarray(returns_b)
    min_len = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:min_len], r_b[:min_len]
    
    s_a, _, _ = block_bootstrap_sharpe(r_a, num_bootstraps=100)
    s_b, _, _ = block_bootstrap_sharpe(r_b, num_bootstraps=100)
    observed_diff = s_a - s_b
    
    # Bootstrap p-value
    boot_diffs = []
    for _ in range(num_bootstraps):
        indices = np.random.randint(0, min_len, size=min_len)
        boot_a = r_a[indices]
        boot_b = r_b[indices]
        sa = (np.mean(boot_a) * 252.0) / (np.std(boot_a, ddof=1) * np.sqrt(252.0) + 1e-8)
        sb = (np.mean(boot_b) * 252.0) / (np.std(boot_b, ddof=1) * np.sqrt(252.0) + 1e-8)
        boot_diffs.append(sa - sb)
        
    p_value = float(np.mean(np.array(boot_diffs) <= 0))
    return {
        'sharpe_a': s_a,
        'sharpe_b': s_b,
        'sharpe_diff': observed_diff,
        'p_value': p_value
    }

