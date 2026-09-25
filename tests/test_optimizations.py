import numpy as np
import pandas as pd

from src.backtest.backtester import Backtester
from src.data.windows import create_rolling_windows
from src.evaluation.statistical_tests import block_bootstrap_sharpe
from src.portfolio.transaction_costs import (
    calculate_position_turnover,
    calculate_transaction_costs,
)


def _legacy_windows(features, targets, lookback, dates=None):
    x_values, y_values, valid_dates = [], [], []
    for index in range(len(features) - lookback):
        x_values.append(features[index : index + lookback])
        y_values.append(targets[index + lookback])
        if dates is not None:
            valid_dates.append(dates[index + lookback - 1])
    return (
        np.array(x_values, dtype=np.float32),
        np.array(y_values, dtype=np.float32),
        pd.DatetimeIndex(valid_dates) if dates is not None else None,
    )


def _legacy_block_bootstrap(
    returns, block_size=20, num_bootstraps=1000, random_seed=42
):
    np.random.seed(random_seed)
    ret_arr = np.asarray(returns)
    ret_arr = ret_arr[~np.isnan(ret_arr)]
    sample_count = len(ret_arr)
    point_mean = np.mean(ret_arr) * 252.0
    point_vol = np.std(ret_arr, ddof=1) * np.sqrt(252.0)
    point_sharpe = float(point_mean / point_vol) if point_vol > 1e-8 else 0.0
    block_count = max(1, sample_count // block_size)
    sharpes = []

    for _ in range(num_bootstraps):
        starts = np.random.randint(
            0, sample_count - block_size + 1, size=block_count
        )
        boot_returns = []
        for start in starts:
            boot_returns.extend(ret_arr[start : start + block_size])
        boot_arr = np.array(boot_returns)
        mean_return = np.mean(boot_arr) * 252.0
        volatility = np.std(boot_arr, ddof=1) * np.sqrt(252.0)
        if volatility > 1e-8:
            sharpes.append(mean_return / volatility)

    return (
        point_sharpe,
        float(np.percentile(sharpes, 2.5)),
        float(np.percentile(sharpes, 97.5)),
    )


def test_vectorized_windows_match_legacy_output_exactly():
    generator = np.random.default_rng(7)
    features = generator.normal(size=(40, 6))
    targets = generator.normal(size=(40, 3))
    dates = pd.date_range("2021-01-01", periods=40, freq="D")

    expected = _legacy_windows(features, targets, lookback=8, dates=dates)
    actual = create_rolling_windows(features, targets, lookback=8, dates=dates)

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert actual[2].equals(expected[2])
    assert actual[0].flags.c_contiguous


def test_reused_turnover_matches_previous_backtest_calculation():
    positions = np.array(
        [[0.5, -0.25], [0.75, -0.10], [-0.20, 0.30]], dtype=float
    )
    shifted = np.roll(positions, shift=1, axis=0)
    shifted[0] = 0.0
    expected_turnover = np.abs(positions - shifted)

    turnover = calculate_position_turnover(positions)
    costs = calculate_transaction_costs(positions, bps=5.0, turnover=turnover)

    np.testing.assert_array_equal(turnover, expected_turnover)
    np.testing.assert_array_equal(costs, expected_turnover * 0.0005)

    signals = positions
    volatility = np.full_like(signals, 0.10)
    returns = np.full_like(signals, 0.01)
    result = Backtester(target_vol=0.10, max_leverage=3.0, tc_bps=5.0).run(
        signals, volatility, returns
    )
    np.testing.assert_array_equal(
        result["turnover"].to_numpy(), expected_turnover.mean(axis=1)
    )


def test_vectorized_bootstrap_matches_legacy_result_exactly():
    returns = np.linspace(-0.02, 0.025, 120)
    expected = _legacy_block_bootstrap(
        returns, block_size=12, num_bootstraps=50, random_seed=19
    )
    actual = block_bootstrap_sharpe(
        returns, block_size=12, num_bootstraps=50, random_seed=19
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
