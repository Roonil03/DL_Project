import hashlib
import zipfile

import numpy as np
import pandas as pd
import pytest

from src.data.french_industry import ASSETS, FrenchIndustryAdapter
from src.data.features import build_features_from_returns
from src.data.walk_forward import calendar_crisis_folds
from src.data.windows import create_rolling_windows


def make_zip(path, rows):
    header = ',' + ','.join(ASSETS)
    text = '\n'.join(['Source description', '  Average Value Weighted Returns -- Daily', header,
                      *rows, '', '  Average Equal Weighted Returns -- Daily', header,
                      '20080102,' + ','.join(['9'] * 10)])
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('10_Industry_Portfolios_daily.csv', text)


def test_french_parses_only_value_weighted_and_records_hash(tmp_path):
    path = tmp_path / 'source.zip'
    make_zip(path, ['20080102,' + ','.join(['1.25', '-99.99'] + ['0'] * 8),
                    '20080103,' + ','.join(['-999'] + ['0'] * 9)])
    adapter = FrenchIndustryAdapter(path)
    returns = adapter.load()
    assert list(returns.columns) == list(ASSETS)
    assert returns.shape == (2, 10)
    assert returns.iloc[0, 0] == pytest.approx(.0125)
    assert pd.isna(returns.iloc[0, 1]) and pd.isna(returns.iloc[1, 0])
    manifest = adapter.manifest(retrieved_at_utc='2026-01-01T00:00:00Z')
    assert manifest['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['retrieved_at_utc'] == '2026-01-01T00:00:00Z'
    assert manifest['missing_by_asset']['NoDur'] == 1


def test_french_rejects_missing_and_duplicate_data(tmp_path):
    with pytest.raises(FileNotFoundError, match='Download the official file'):
        FrenchIndustryAdapter(tmp_path / 'missing.zip').load()
    path = tmp_path / 'duplicate.zip'
    row = '20080102,' + ','.join(['1'] * 10)
    make_zip(path, [row, row])
    with pytest.raises(ValueError, match='unique and increasing'):
        FrenchIndustryAdapter(path).load()


def test_direct_returns_features_compound_and_preserve_next_target():
    dates = pd.bdate_range('2007-01-01', periods=100)
    returns = pd.DataFrame({'asset': np.linspace(-.01, .01, 100)}, index=dates)
    features, aligned, _, _ = build_features_from_returns(returns)
    assert features.index[0] == dates[59]
    assert features.iloc[0]['asset_mom20'] == pytest.approx(np.prod(1 + returns.iloc[40:60, 0]) - 1)
    x, y, decision = create_rolling_windows(features.to_numpy(), aligned.to_numpy(), 5, aligned.index)
    assert decision[0] == aligned.index[4]
    assert y[0, 0] == pytest.approx(aligned.iloc[5, 0])
    returns.iloc[70, 0] = np.nan
    with pytest.raises(ValueError, match='do not forward-fill'):
        build_features_from_returns(returns)


def test_crisis_folds_respect_boundaries_and_require_full_coverage():
    dates = pd.bdate_range('2001-01-01', '2021-01-08')
    decisions, targets = dates[:-1], dates[1:]
    folds = calendar_crisis_folds(decisions, targets)
    assert len(folds) == 2
    for year, (train, val, test) in zip((2008, 2020), folds):
        assert targets[train].max() < decisions[val].min()
        assert targets[val].max() < decisions[test].min()
        assert set(targets[test].year) == {year}
    with pytest.raises(ValueError, match='Insufficient coverage'):
        calendar_crisis_folds(decisions[:-260], targets[:-260], test_years=[2020])
