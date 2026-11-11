import numpy as np
import pandas as pd
import pytest

from src.backtest.backtester import Backtester
from src.data.windows import create_rolling_windows
from src.evaluation.metrics import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_max_drawdown,
    compute_all_metrics,
)
from src.portfolio.baselines import get_momentum_baseline


def test_metrics_ignore_nonfinite_observations_consistently():
    returns = np.array([0.01, np.nan, 0.03, np.inf, -0.01])

    assert calculate_annualized_return(returns) == pytest.approx(2.52)
    assert calculate_annualized_volatility(returns) == pytest.approx(
        np.std([0.01, 0.03, -0.01], ddof=1) * np.sqrt(252)
    )
    assert calculate_max_drawdown(returns) == pytest.approx(
        ((1.01 * 1.03 * 0.99) - (1.01 * 1.03)) / (1.01 * 1.03)
    )
    metrics = compute_all_metrics(returns, turnover=np.array([0.1, np.nan, np.inf, 0.3]))
    assert metrics['Annualized Turnover'] == pytest.approx(50.4)
    assert np.isfinite(list(metrics.values())).all()


def test_metrics_reject_non_vector_inputs():
    with pytest.raises(ValueError, match='one-dimensional'):
        calculate_annualized_return(np.ones((3, 2)))
    with pytest.raises(ValueError, match='one-dimensional'):
        compute_all_metrics(np.ones(3), turnover=np.ones((3, 1)))


def test_rolling_windows_validate_alignment_and_keep_empty_shapes():
    features = np.ones((3, 2))
    targets = np.ones((3, 1))
    dates = pd.date_range('2024-01-01', periods=3)

    x, y, decision_dates = create_rolling_windows(features, targets, 4, dates)
    assert x.shape == (0, 4, 2)
    assert y.shape == (0, 1)
    assert len(decision_dates) == 0
    with pytest.raises(ValueError, match='positive integer'):
        create_rolling_windows(features, targets, 0)
    with pytest.raises(ValueError, match='same time length'):
        create_rolling_windows(features, targets[:-1], 2)
    with pytest.raises(ValueError, match='dates'):
        create_rolling_windows(features, targets, 2, dates[:2])


def test_backtester_rejects_misaligned_inputs_and_dates():
    backtester = Backtester()
    signals = np.ones((4, 2))
    vol = np.ones_like(signals)
    returns = np.ones_like(signals)

    with pytest.raises(ValueError, match='matching shapes'):
        backtester.run(signals, vol[:, :1], returns)
    with pytest.raises(ValueError, match='dates'):
        backtester.run(signals, vol, returns, pd.date_range('2024-01-01', periods=3))
    with pytest.raises(ValueError, match='finite'):
        backtester.run(signals, vol, np.full_like(returns, np.nan))


def test_momentum_baseline_requires_integer_lookback():
    with pytest.raises(ValueError, match='positive integer'):
        get_momentum_baseline(np.ones((5, 2)), lookback=1.5)
