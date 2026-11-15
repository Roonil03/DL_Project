"""Validate the audited input for the selected benchmark mode."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import pandas as pd


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_audited_data(root, run_dir, mode, data_dir=None, data_path=None, macro_path=None, selected=None):
    root, run_dir = Path(root), Path(run_dir)
    audit_path = run_dir / 'metadata' / 'data_manifest.json'
    if not audit_path.is_file():
        raise FileNotFoundError(f'Missing audited data manifest: {audit_path}')
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit.get('schema_version') != '1.0':
        raise ValueError('Unsupported data manifest schema')
    selected = list(selected or [])
    if mode == 'india_indices':
        from src.data.india_indices import DEFAULT_PATH, HASHES, IndiaIndicesAdapter
        directory = Path(data_dir) if data_dir else root / DEFAULT_PATH
        adapter = IndiaIndicesAdapter(directory)
        snapshot = adapter.manifest()
        required = [directory / name for name in HASHES]
        if audit.get('dataset_mode') != mode or audit.get('source_snapshot', {}).get('files') != snapshot['files']:
            raise ValueError('Indian snapshot does not match audited manifest')
    elif mode == 'fred_sectors':
        from src.data.fred_sectors import DEFAULT_PATH, FredSectorsAdapter
        path = (Path(data_dir) / 'sector_indices.csv') if data_dir else root / DEFAULT_PATH
        adapter = FredSectorsAdapter(path)
        snapshot = adapter.manifest()
        required = [path]
        if audit.get('dataset_mode') != mode or audit.get('source_snapshot', {}).get('sha256') != snapshot['sha256']:
            raise ValueError('FRED snapshot does not match audited manifest')
    elif mode == 'global_indian':
        directory = root / 'data' / 'kaggle_global_indian_markets' / 'global-india-markets-macro'
        daily = Path(data_path) if data_path else directory / 'daily_market_data.csv'
        monthly = Path(macro_path) if macro_path else directory / 'monthly_macro_data.csv'
        sources = daily.parent / 'codes_and_sources.md'
        required = [daily, monthly, sources]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError('Missing legacy mixed-market input')
        records = audit.get('files', {})
        if set(records) != {'daily', 'monthly'}:
            raise ValueError('Legacy audit must contain daily and monthly files')
        for name, path in (('daily', daily), ('monthly', monthly)):
            record = records[name]
            if (record.get('sha256') != _digest(path) or not record.get('columns') or
                    not record.get('dtypes') or record.get('missing_by_column') is None):
                raise ValueError(f'Legacy {name} file does not match audited manifest')
        if audit.get('source_mapping', {}).get('sha256') != _digest(sources):
            raise ValueError('Legacy source mapping does not match audited manifest')
        raw = pd.read_csv(daily)
        date_col = next((c for c in raw if c.lower() == 'date'), None)
        if date_col is None: raise ValueError('Legacy source has no date column')
        dates = pd.to_datetime(raw[date_col], errors='raise')
        if dates.isna().any() or not dates.is_monotonic_increasing or dates.duplicated().any():
            raise ValueError('Legacy dates must be valid, unique and increasing')
        if not set(selected).issubset({'NIFTY50','SP500','USDINR','GOLD','BRENT'}):
            raise ValueError('Unknown legacy asset selection')
        snapshot = None
    else:
        raise ValueError(f'Unsupported DATASET_MODE: {mode}')
    if snapshot is not None:
        expected = selected or audit.get('asset_candidates', [])
        if not expected or len(set(expected)) != len(expected) or not set(expected).issubset(snapshot['assets']):
            raise ValueError('Invalid audited asset selection')
        if audit.get('selected_assets') != expected:
            raise ValueError('Selected assets do not match audited manifest')
    elif selected != audit.get('selected_assets', []):
        raise ValueError('Selected assets do not match audited manifest')
    return required
