"""Prepare French-industry data and calendar folds for notebook benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import build_features_from_returns
from src.data.scaling import TimeSeriesScaler
from src.data.walk_forward import calendar_crisis_folds
from src.data.windows import create_rolling_windows
from src.regimes.rules import detect_regimes


def prepare_french_benchmark(raw_returns, assets, fold, lookback):
    missing = sorted(set(assets) - set(raw_returns.columns))
    if missing or not assets or len(set(assets)) != len(assets):
        raise ValueError(f"Invalid French industry asset selection: {missing}")
    test_years = (2008, 2020)
    if fold < 0 or fold >= len(test_years):
        raise ValueError(f"FOLD={fold} outside available 0..{len(test_years)-1}")
    if not isinstance(raw_returns.index, pd.DatetimeIndex) or not raw_returns.index.is_unique or not raw_returns.index.is_monotonic_increasing:
        raise ValueError("Returns require a unique, chronological DatetimeIndex")
    # Each fold has five train years, one validation year and one test year.
    # Include a fixed observation warmup for 60-day features, model lookback,
    # and the regime classifier's 120-day trend, without using unrelated history.
    test_start = pd.Timestamp(year=test_years[fold], month=1, day=1)
    train_start = test_start - pd.DateOffset(years=6)
    test_end = test_start + pd.DateOffset(years=1)
    start_pos = raw_returns.index.searchsorted(train_start)
    end_pos = raw_returns.index.searchsorted(test_end)
    warmup = 60 + lookback + 121
    returns = raw_returns.iloc[max(0, start_pos - warmup):end_pos].loc[:, assets]
    features, returns, vol, _ = build_features_from_returns(returns)
    # Each decision at the US close uses returns through that close; the target
    # is the next observed trading day's return.
    _, _, decisions = create_rolling_windows(features.to_numpy(), returns.to_numpy(), lookback, features.index)
    target_dates = features.index[lookback:]
    indices = calendar_crisis_folds(decisions, target_dates, test_years=(test_years[fold],))[0]
    masks = {}
    for name, idx in zip(('train', 'val', 'test'), indices):
        masks[name] = np.zeros(len(decisions), dtype=bool)
        masks[name][idx] = True
    train_decisions = decisions[indices[0]]
    scaler = TimeSeriesScaler()
    scaler.fit(features.loc[train_decisions].to_numpy())
    X, y, decisions_scaled = create_rolling_windows(scaler.transform(features.to_numpy()), returns.to_numpy(), lookback, features.index)
    assert decisions_scaled.equals(decisions)
    # These are cumulative research-return indices solely for regime labels.
    # Their levels do not represent actual traded prices.
    wealth = (1 + raw_returns.iloc[max(0, start_pos - warmup):end_pos].loc[:, assets]).cumprod()
    first_train = wealth.index.get_indexer([train_decisions.min()])[0]
    regime_wealth = wealth.iloc[max(0, first_train - 121):]
    regimes = detect_regimes(regime_wealth, train_end=train_decisions.max()).reindex(decisions)
    bounds = {name: [str(decisions[idx].min()), str(target_dates[idx].max())]
              for name, idx in zip(('train', 'val', 'test'), indices)}
    return {'prices': wealth, 'returns_full': returns, 'X': X, 'y': y, 'dates': decisions,
            'target_dates': target_dates, 'vol': vol.reindex(decisions).to_numpy(),
            'regimes': regimes, 'masks': masks, 'assets': assets, 'fold_bounds': bounds}
