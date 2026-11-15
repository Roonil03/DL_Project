import hashlib
import numpy as np
import pandas as pd
import pytest
from src.data.fred_sectors import ASSETS, FredSectorsAdapter
from src.data.fred_benchmark import prepare_fred_benchmark

CONFIG = {'dataset':'fred_sectors', 'source_file':'sector_indices.csv', 'returns':'index_total_return',
          'assets':list(ASSETS), 'folds':{'test_years':[2008,2020], 'train_years':5, 'validation_years':1}}

def test_fred_schema_returns_missing_and_hash(tmp_path):
    path=tmp_path/'sector_indices.csv'
    path.write_text('observation_date,'+','.join(ASSETS)+'\n2008-01-02,100,100,100,100,100\n2008-01-03,,,,,\n2008-01-04,101,102,103,104,105\n')
    a=FredSectorsAdapter(path); result=a.load()
    assert len(result)==1 and result.iloc[0,0]==pytest.approx(.01)
    assert a.manifest()['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_text('observation_date,'+','.join(ASSETS)+'\n2008-01-02,100,,100,100,100\n')
    with pytest.raises(ValueError, match='Partially missing'): a.load()
    path.write_text('observation_date,'+','.join(ASSETS)+'\n2008-01-02,0,1,1,1,1\n')
    with pytest.raises(ValueError, match='positive'): a.load()

def test_fred_crisis_folds_and_one_session_signal_lag():
    dates=pd.bdate_range('2001-01-01','2021-01-08')
    rng=np.random.default_rng(3)
    r=pd.DataFrame(rng.normal(0,.01,(len(dates),len(ASSETS))),index=dates,columns=ASSETS)
    for fold,year in [(0,2008),(1,2020)]:
        result=prepare_fred_benchmark(r,list(ASSETS),fold,20,config=CONFIG)
        assert set(result['target_dates'][result['masks']['test']].year)=={year}
        assert (result['target_dates']>result['dates']).all()
    date=pd.Timestamp('2020-06-01'); changed=r.copy(); changed.loc[date,ASSETS[0]]+=.1
    a=prepare_fred_benchmark(r,list(ASSETS),1,20,config=CONFIG)
    b=prepare_fred_benchmark(changed,list(ASSETS),1,20,config=CONFIG)
    i=a['dates'].get_loc(date)
    np.testing.assert_allclose(a['X'][i],b['X'][i])
    np.testing.assert_allclose(a['vol'][i],b['vol'][i])
