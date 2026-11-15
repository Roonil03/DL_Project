import hashlib
import numpy as np
import pandas as pd
import pytest
import src.data.india_indices as source
from src.data.india_benchmark import prepare_india_benchmark

CONFIG={'dataset':'india_indices','source_file':'index_list.csv','returns':'index_price_return', 'assets':list(source.ASSETS), 'folds':{'test_years':[2020,2023],'train_years':5,'validation_years':1}}

def test_real_snapshot_when_available():
    import os
    path=os.getenv('INDIA_TEST_DATA')
    if not path: pytest.skip('Set INDIA_TEST_DATA to the pinned, locally downloaded snapshot')
    a=source.IndiaIndicesAdapter(path); levels=a.levels()
    assert len(levels)==3675 and str(levels.index[0].date())=='2011-01-31' and str(levels.index[-1].date())=='2025-12-31'
    for fold,year in [(0,2020),(1,2023)]:
        data=prepare_india_benchmark(a.load(),list(source.ASSETS),fold,20,config=CONFIG)
        assert set(data['target_dates'][data['masks']['test']].year)=={year}

def test_lagged_signals_and_sizing():
    dates=pd.bdate_range('2012-01-01','2024-01-10'); rng=np.random.default_rng(4)
    returns=pd.DataFrame(rng.normal(0,.01,(len(dates),3)),index=dates,columns=source.ASSETS)
    changed=returns.copy(); date=pd.Timestamp('2020-06-01'); changed.loc[date,'NIFTY_BANK']+=.1
    a=prepare_india_benchmark(returns,list(source.ASSETS),0,20,config=CONFIG)
    b=prepare_india_benchmark(changed,list(source.ASSETS),0,20,config=CONFIG)
    i=a['dates'].get_loc(date)
    np.testing.assert_allclose(a['X'][i],b['X'][i]); np.testing.assert_allclose(a['vol'][i],b['vol'][i])
    assert a['target_dates'][i]>a['dates'][i]


def test_pinned_panel_rejects_partial_missing_and_duplicate_dates(tmp_path, monkeypatch):
    dates = pd.to_datetime(['2020-01-02','2020-01-03','2020-01-06'])
    for asset in source.ASSETS:
        pd.DataFrame({'Close':[100.,101.,102.]}, index=dates).to_parquet(tmp_path / (asset+'.parquet'))
    (tmp_path/'index_list.csv').write_text('catalog')
    def pin():
        monkeypatch.setattr(source, 'HASHES', {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()})
    pin(); assert len(source.IndiaIndicesAdapter(tmp_path).levels()) == 3
    pd.DataFrame({'Close':[100.,102.]}, index=dates[[0,2]]).to_parquet(tmp_path/'NIFTY_IT.parquet')
    pin()
    with pytest.raises(ValueError, match='align'): source.IndiaIndicesAdapter(tmp_path).levels()
    pd.DataFrame({'Close':[100.,101.,102.]}, index=dates[[0,0,2]]).to_parquet(tmp_path/'NIFTY_IT.parquet')
    pin()
    with pytest.raises(ValueError, match='duplicate'): source.IndiaIndicesAdapter(tmp_path).levels()
