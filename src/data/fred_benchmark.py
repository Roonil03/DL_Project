"""Prepare FRED sector data and calendar folds for notebook benchmarks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.features import build_features_from_returns
from src.data.scaling import TimeSeriesScaler
from src.data.walk_forward import calendar_crisis_folds
from src.data.windows import create_rolling_windows
from src.regimes.rules import detect_regimes
from src.data.fred_sectors import ASSETS


def validate_fred_config(config):
    if not isinstance(config, dict) or config.get('dataset') != 'fred_sectors':
        raise ValueError('FRED config must name dataset fred_sectors')
    if config.get('source_file') != 'sector_indices.csv' or config.get('returns') != 'index_total_return':
        raise ValueError('Unsupported FRED source file or return type')
    assets = config.get('assets')
    if (not isinstance(assets, list) or not assets or any(not isinstance(a, str) for a in assets)
            or len(set(assets)) != len(assets) or not set(assets).issubset(ASSETS)):
        raise ValueError('FRED config assets must be a unique nonempty list of known industries')
    folds = config.get('folds')
    if not isinstance(folds, dict):
        raise ValueError('FRED config must contain folds')
    years = folds.get('test_years')
    if (not isinstance(years, list) or not years or
            any(not isinstance(y, int) or isinstance(y, bool) or not 1926 <= y <= 2100 for y in years) or
            years != sorted(set(years))):
        raise ValueError('test_years must be unique, increasing integer years between 1926 and 2100')
    if any(not isinstance(folds.get(k), int) or isinstance(folds[k], bool) or folds[k] < 1 for k in ('train_years', 'validation_years')):
        raise ValueError('train_years and validation_years must be positive integers')
    return config


def prepare_fred_benchmark(raw_returns, assets, fold, lookback, *, config):
    config = validate_fred_config(config)
    missing = sorted(set(assets) - set(raw_returns.columns))
    if missing or not assets or len(set(assets)) != len(assets) or not set(assets).issubset(config['assets']):
        raise ValueError(f"Invalid FRED sector asset selection: {missing}")
    fold_config = config['folds']
    test_years = fold_config['test_years']
    if not isinstance(fold, int) or isinstance(fold, bool) or fold < 0 or fold >= len(test_years):
        raise ValueError(f"FOLD={fold} outside available 0..{len(test_years)-1}")
    if not isinstance(raw_returns.index, pd.DatetimeIndex) or not raw_returns.index.is_unique or not raw_returns.index.is_monotonic_increasing:
        raise ValueError("Returns require a unique, chronological DatetimeIndex")
    # Each fold has configured train and validation years and one test year.
    # Include a fixed observation warmup for 60-day features, model lookback,
    # and the regime classifier's 120-day trend, without using unrelated history.
    test_start = pd.Timestamp(year=test_years[fold], month=1, day=1)
    train_start = test_start - pd.DateOffset(years=fold_config['train_years'] + fold_config['validation_years'])
    test_end = test_start + pd.DateOffset(years=1)
    start_pos = raw_returns.index.searchsorted(train_start)
    end_pos = raw_returns.index.searchsorted(test_end)
    warmup = 60 + lookback + 121
    returns = raw_returns.iloc[max(0, start_pos - warmup):end_pos].loc[:, assets]
    features, returns, vol, _ = build_features_from_returns(returns)
    # Lag signals and sizing by one observed session: data at the decision
    # close cannot be used to execute at that same close.
    features = features.shift(1).dropna()
    vol = vol.shift(1).loc[features.index]
    returns = returns.loc[features.index]
    _, _, decisions = create_rolling_windows(features.to_numpy(), returns.to_numpy(), lookback, features.index)
    target_dates = features.index[lookback:]
    indices = calendar_crisis_folds(decisions, target_dates, test_years=(test_years[fold],),
                                    train_years=fold_config['train_years'],
                                    validation_years=fold_config['validation_years'])[0]
    masks = {}
    for name, idx in zip(('train', 'val', 'test'), indices):
        masks[name] = np.zeros(len(decisions), dtype=bool)
        masks[name][idx] = True
    train_decisions = decisions[indices[0]]
    scaler = TimeSeriesScaler()
    scaler.fit(features.loc[train_decisions].to_numpy())
    X, y, decisions_scaled = create_rolling_windows(scaler.transform(features.to_numpy()), returns.to_numpy(), lookback, features.index)
    assert decisions_scaled.equals(decisions)
    # These reconstructed index levels are solely for regime labels.
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
