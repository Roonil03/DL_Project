"""Pinned FRED Nasdaq US sector total-return index snapshot."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ASSETS = ("NASDAQNQUSB10LMT", "NASDAQNQUSB20LMT", "NASDAQNQUSB30LMT", "NASDAQNQUSB50LMT", "NASDAQNQUSB60LMT")
SOURCE_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=" + ",".join(ASSETS)
SOURCE_PAGE = "https://fred.stlouisfed.org/series/NASDAQNQUSB10LMT"
DEFAULT_PATH = "data/fred_nasdaq_sectors/sector_indices.csv"

class FredSectorsAdapter:
    """Read positive index levels, discarding only entirely missing calendar rows."""
    def __init__(self, data_path=DEFAULT_PATH):
        self.path = Path(data_path)

    def levels(self):
        if not self.path.is_file():
            raise FileNotFoundError(f"FRED CSV not found: {self.path}. Run scripts/prepare_fred_sectors.py or download {SOURCE_URL}")
        frame = pd.read_csv(self.path, dtype=str, keep_default_na=False)
        if list(frame.columns) != ['observation_date', *ASSETS]:
            raise ValueError('Unexpected FRED CSV schema')
        dates = pd.to_datetime(frame.pop('observation_date'), format='%Y-%m-%d', errors='raise')
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            raise ValueError('Index dates must be valid, unique and increasing')
        frame = frame.replace({'': np.nan, '.': np.nan}).apply(pd.to_numeric, errors='raise')
        frame.index = pd.DatetimeIndex(dates, name='date')
        missing = frame.isna()
        if (missing.any(axis=1) & ~missing.all(axis=1)).any():
            raise ValueError('Partially missing index row; do not forward-fill')
        frame = frame.loc[~missing.all(axis=1)]
        if frame.empty or not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
            raise ValueError('Index levels must be finite and positive')
        return frame

    def load(self):
        return self.levels().pct_change(fill_method=None).iloc[1:]

    def manifest(self, *, retrieved_at_utc=None):
        levels = self.levels()
        return {'dataset': 'fred_nasdaq_us_large_mid_sectors', 'source_url': SOURCE_URL,
                'source_page': SOURCE_PAGE, 'path': str(self.path),
                'sha256': hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'retrieved_at_utc': retrieved_at_utc,
                'source_vintage': 'mutable download; exact snapshot identified by SHA-256',
                'methodology': 'Nasdaq US large/mid-cap sector total-return indices; not tradable assets; historical backfill status unverified',
                'units': 'index levels; decimal daily total returns computed from levels',
                'assets': list(levels.columns), 'first_date': str(levels.index.min().date()),
                'last_date': str(levels.index.max().date()), 'rows': len(levels),
                'missing_by_asset': levels.isna().sum().to_dict(),
                'license': 'Nasdaq copyright; FRED labels data pre-approval required for sharing'}
