import numpy as np


def calculate_position_turnover(positions: np.ndarray) -> np.ndarray:
    """Return absolute position changes without allocating a shifted copy."""
    turnover = np.empty_like(positions)
    turnover[0] = np.abs(positions[0])
    turnover[1:] = np.abs(positions[1:] - positions[:-1])
    return turnover


def calculate_transaction_costs(
    positions: np.ndarray,
    bps: float = 5.0,
    turnover: np.ndarray = None,
) -> np.ndarray:
    """
    Calculates transaction costs based on change in position.
    positions: shape (time, assets)
    bps: basis points of cost (e.g., 5.0 = 0.05%)
    """
    if turnover is None:
        turnover = calculate_position_turnover(positions)
    return turnover * (bps / 10000.0)
