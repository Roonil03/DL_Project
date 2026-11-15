import hashlib
import json
import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.data import india_indices, fred_sectors

@pytest.mark.parametrize('mode', ['india_indices', 'fred_sectors', 'global_indian'])
def test_validation_notebook_accepts_audited_modes_without_training(tmp_path, monkeypatch, mode):
    root=tmp_path; data=root/'data'; data.mkdir(); run=root/'outputs'/'test'; meta=run/'metadata'; meta.mkdir(parents=True)
    monkeypatch.setenv('DATASET_MODE',mode); monkeypatch.setenv('RUN_ID','test'); monkeypatch.setenv('OUTPUT_ROOT',str(root/'outputs'))
    monkeypatch.delenv('FIN_ASSETS',raising=False); monkeypatch.delenv('FIN_DATA_PATH',raising=False); monkeypatch.delenv('FIN_MACRO_PATH',raising=False)
    if mode == 'india_indices':
        directory=data/'india_indices'; directory.mkdir()
        for asset in india_indices.ASSETS:
            pd.DataFrame({'Close':[100.,101.]},index=pd.to_datetime(['2020-01-02','2020-01-03'])).to_parquet(directory/(asset+'.parquet'))
        (directory/'index_list.csv').write_text('catalog')
        monkeypatch.setattr(india_indices,'HASHES',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()})
        adapter=india_indices.IndiaIndicesAdapter(directory)
        audit={'schema_version':'1.0','dataset_mode':mode,'source_snapshot':adapter.manifest(), 'selected_assets':list(india_indices.ASSETS),'asset_candidates':list(india_indices.ASSETS)}
    elif mode == 'fred_sectors':
        directory=data/'fred'; directory.mkdir(); p=directory/'sector_indices.csv'
        p.write_text('observation_date,'+','.join(fred_sectors.ASSETS)+'\n2020-01-02,100,100,100,100,100\n2020-01-03,101,101,101,101,101\n')
        adapter=fred_sectors.FredSectorsAdapter(p)
        audit={'schema_version':'1.0','dataset_mode':mode,'source_snapshot':adapter.manifest(), 'selected_assets':list(fred_sectors.ASSETS),'asset_candidates':list(fred_sectors.ASSETS)}
    else:
        directory=data/'legacy'; directory.mkdir()
        daily=directory/'daily_market_data.csv'; daily.write_text('date,nifty50_close\n2020-01-02,100\n2020-01-03,101\n')
        monthly=directory/'monthly_macro_data.csv'; monthly.write_text('date,value\n2020-01-01,1\n')
        sources=directory/'codes_and_sources.md'; sources.write_text('source')
        def record(p): return {'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'columns':['date'],'dtypes':{'date':'object'},'missing_by_column':{'date':0}}
        audit={'schema_version':'1.0','selected_assets':[],'files':{'daily':record(daily),'monthly':record(monthly)},'source_mapping':{'sha256':hashlib.sha256(sources.read_bytes()).hexdigest()}}
        monkeypatch.setenv('FIN_DATA_PATH',str(daily)); monkeypatch.setenv('FIN_MACRO_PATH',str(monthly))
    monkeypatch.setenv('DATA_DIR',str(directory))
    (meta/'data_manifest.json').write_text(json.dumps(audit)); (meta/'run_manifest.json').write_text(json.dumps({'dataset_mode':mode}))
    for name in ('dependency_lock.txt','config_snapshot.json'): (meta/name).write_text('{}')
    for name in ('00_core','10_data_audit','20_benchmark','30_validation','40_report'):
        p=root/'notebooks'/f'{name}.ipynb'; p.parent.mkdir(exist_ok=True); p.write_text('{}')
    table=run/'tables'/'dlinear'/'seed_42'/'fold_0'; table.mkdir(parents=True)
    fields={'metrics.csv':['cost_bps','Sharpe Ratio','mse','directional_accuracy'], 'regimes.csv':['regime','sharpe','sharpe_ci_lower','sharpe_ci_upper'], 'returns.csv':['decision_date','target_date','gross_return'], 'predictions.csv':['asset','signal','target_return']}
    for name,columns in fields.items():
        row={'model':'dlinear','seed':42,'fold':0}; row.update({col:1 for col in columns});pd.DataFrame([row]).to_csv(table/name,index=False)
    checkpoint=run/'checkpoints'/'dlinear'/'seed_42'/'fold_0';checkpoint.mkdir(parents=True)
    for name in ('latest.pt','best.pt'): (checkpoint/name).write_bytes(b'checkpoint')
    (run/'telemetry').mkdir();(run/'telemetry'/'dlinear__seed_42__fold_0.done').write_text('done')
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(load=lambda *a,**kw:{'unit_identity':{'model':'dlinear','seed':42,'fold':0}}))
    import subprocess
    monkeypatch.setattr(subprocess,'run',lambda *a,**kw:None)
    notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/30_validation.ipynb').read_text(encoding='utf-8'))
    env={'Path':Path,'ROOT':root,'os':__import__('os'),'sys':sys,'pd':pd,'np':np}
    exec(''.join(notebook['cells'][7]['source']),env)
    exec(''.join(notebook['cells'][10]['source']),env)
    # Audit checks must fail if a selected input was changed after audit.
    if mode == 'fred_sectors':
        (directory/'sector_indices.csv').write_text((directory/'sector_indices.csv').read_text().replace('101,101','102,101',1))
        with pytest.raises(ValueError,match='audited manifest'): exec(''.join(notebook['cells'][7]['source']),env)
