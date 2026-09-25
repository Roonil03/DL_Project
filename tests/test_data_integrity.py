import numpy as np
import pandas as pd
import pytest

from src.data.features import build_features, calculate_rolling_volatility
from src.data.windows import create_rolling_windows
from src.models.mamba_model import MAMBA_AVAILABLE, MambaSignalModel


def test_rolling_volatility_does_not_backfill_warmup_period():
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    returns = pd.DataFrame({"asset": np.linspace(-0.01, 0.01, 30)}, index=dates)

    volatility = calculate_rolling_volatility(returns, window=20)

    assert volatility.iloc[:19].isna().all().all()
    assert volatility.iloc[19:].notna().all().all()


def test_feature_outputs_are_aligned_after_warmup_removal():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.DataFrame(
        {
            "asset_a": 100.0 * np.cumprod(1.0 + np.linspace(-0.002, 0.003, 100)),
            "asset_b": 80.0 * np.cumprod(1.0 + np.linspace(0.001, -0.001, 100)),
        },
        index=dates,
    )

    features, returns, vol_20, feature_dict = build_features(prices)

    assert features.index.equals(returns.index)
    assert features.index.equals(vol_20.index)
    assert not features.isna().any().any()
    assert features.index[0] == dates[59]
    assert all(frame.index.equals(features.index) for frame in feature_dict.values())


def test_rolling_windows_keep_next_return_alignment():
    dates = pd.date_range("2020-01-01", periods=6, freq="D")
    features = np.arange(12, dtype=float).reshape(6, 2)
    targets = np.arange(6, dtype=float).reshape(6, 1)

    x, y, decision_dates = create_rolling_windows(features, targets, lookback=3, dates=dates)

    np.testing.assert_array_equal(x[0], features[:3])
    np.testing.assert_array_equal(y[:, 0], targets[3:, 0])
    assert decision_dates[0] == dates[2]


def test_mamba_cannot_silently_become_lstm():
    if MAMBA_AVAILABLE:
        pytest.skip("The real mamba_ssm dependency is installed.")

    with pytest.raises(ImportError, match="mamba_ssm is required"):
        MambaSignalModel(lookback=20, num_features=4, num_assets=2)
