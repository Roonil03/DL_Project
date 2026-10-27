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

def paired_block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample paired time indices in contiguous circular blocks."""
    if n <= 0 or block_size <= 0:
        raise ValueError("n and block_size must be positive")
    blocks = int(np.ceil(n / block_size))
    starts = rng.integers(0, n, size=blocks)
    return (starts[:, None] + np.arange(block_size)[None, :]).reshape(-1)[:n] % n


def _sharpe(values: np.ndarray) -> float:
    volatility = np.std(values, ddof=1) * np.sqrt(252.0) if len(values) > 1 else 0.0
    return float(np.mean(values) * 252.0 / volatility) if volatility > 1e-8 else 0.0


def paired_sharpe_difference_test(returns_a: np.ndarray, returns_b: np.ndarray, num_bootstraps: int = 1000,
                                  block_size: int = 20, random_seed: int = 42,
                                  alternative: str = 'greater') -> Dict[str, float]:
    """
    Paired bootstrap test for difference in Sharpe ratio: H0: Sharpe(A) <= Sharpe(B) vs H1: Sharpe(A) > Sharpe(B).
    """
    r_a = np.asarray(returns_a, dtype=float).reshape(-1)
    r_b = np.asarray(returns_b, dtype=float).reshape(-1)
    min_len = min(len(r_a), len(r_b))
    r_a, r_b = r_a[:min_len], r_b[:min_len]
    valid = np.isfinite(r_a) & np.isfinite(r_b)
    r_a, r_b = r_a[valid], r_b[valid]
    min_len = len(r_a)
    if alternative not in {'greater', 'less', 'two-sided'}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    if min_len < 2:
        return {'sharpe_a': 0.0, 'sharpe_b': 0.0, 'sharpe_diff': 0.0, 'p_value': 1.0,
                'alternative': alternative, 'block_size': block_size, 'observations': min_len}
    
    s_a = _sharpe(r_a)
    s_b = _sharpe(r_b)
    observed_diff = s_a - s_b
    
    # Bootstrap p-value
    # Resample the same time blocks for both series: pairing preserves their
    # contemporaneous dependence and blocks preserve serial dependence.
    rng = np.random.default_rng(random_seed)
    boot_diffs = []
    for _ in range(num_bootstraps):
        indices = paired_block_bootstrap_indices(min_len, min(block_size, min_len), rng)
        boot_a = r_a[indices]
        boot_b = r_b[indices]
        sa, sb = _sharpe(boot_a), _sharpe(boot_b)
        boot_diffs.append(sa - sb)
    boot_diffs = np.asarray(boot_diffs)
    # Centering implements the null while the resampling retains paired blocks.
    null_diffs = boot_diffs - np.mean(boot_diffs)
    if alternative == 'greater':
        p_value = float((1 + np.sum(null_diffs >= observed_diff)) / (len(null_diffs) + 1))
    elif alternative == 'less':
        p_value = float((1 + np.sum(null_diffs <= observed_diff)) / (len(null_diffs) + 1))
    else:
        p_value = float((1 + np.sum(np.abs(null_diffs) >= abs(observed_diff))) / (len(null_diffs) + 1))
    return {
        'sharpe_a': s_a,
        'sharpe_b': s_b,
        'sharpe_diff': observed_diff,
        'p_value': p_value,
        'alternative': alternative,
        'block_size': min(block_size, min_len),
        'observations': min_len,
    }

