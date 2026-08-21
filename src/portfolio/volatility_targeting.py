import numpy as np

def apply_volatility_target(signal: np.ndarray, ex_ante_vol: np.ndarray, target_vol: float = 0.10, max_leverage: float = 3.0):
    """
    Scales directional signal into a position using ex-ante volatility.
    
    position = signal * (target_vol / ex_ante_vol)
    """
    # Prevent division by zero
    safe_vol = np.where(ex_ante_vol < 1e-6, 1e-6, ex_ante_vol)
    
    desired_position = signal * (target_vol / safe_vol)
    
    # Cap leverage
    capped_position = np.clip(desired_position, -max_leverage, max_leverage)
    
    return capped_position
