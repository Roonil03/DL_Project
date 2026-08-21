import numpy as np

def calculate_transaction_costs(positions: np.ndarray, bps: float = 5.0) -> np.ndarray:
    """
    Calculates transaction costs based on change in position.
    positions: shape (time, assets)
    bps: basis points of cost (e.g., 5.0 = 0.05%)
    """
    rate = bps / 10000.0
    
    # Shift positions to find change (t minus t-1)
    # At t=0, cost is on the whole initial position
    shifted_pos = np.roll(positions, shift=1, axis=0)
    shifted_pos[0] = 0.0 
    
    turnover = np.abs(positions - shifted_pos)
    costs = turnover * rate
    
    return costs
