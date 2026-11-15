"""Pinned NSE sector price indices from an independent research dataset."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

REVISION = "1f4aff7c3e53a9f2bfb6a167b0d80f4d0c8f8bf9"
BASE_URL = f"https://raw.githubusercontent.com/aswinv90/indian-market-historical-data/{REVISION}/data/INDICES/"
SOURCE_PAGE = "https://github.com/aswinv90/indian-market-historical-data"
DEFAULT_PATH = "data/india_indices"
ASSETS = ("NIFTY_BANK", "NIFTY_IT", "NIFTY_PHARMA")
# The snapshot contains a missing Pharma session in 2026 and an all-missing final row.
END_DATE = "2025-12-31"
HASHES = {
    "NIFTY_BANK.parquet": "cb19fc783ac88fd0f57be97dab2b615a5be5bb86659925315170bb47deb7795c",
    "NIFTY_IT.parquet": "281106003c13169a69f91c9cb0709169bea5663cf04b2c733e6abafceb751aab",
    "NIFTY_PHARMA.parquet": "71e517166f82e12a0a2740222d34d47d10d4a2b104c940636762bd09a6bd3ded",
    "index_list.csv": "c164b2e1ce9cef2df67f98ee695b25684cc6be360a0999465f0a283260ba5292",
}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

class IndiaIndicesAdapter:
    """Read pinned same-exchange index close levels; they are not ETFs."""
    def __init__(self, data_path=DEFAULT_PATH):
        self.path = Path(data_path)

    def levels(self):
        for name, expected in HASHES.items():
            path = self.path / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; run scripts/prepare_india_indices.py")
            if digest(path) != expected:
                raise ValueError(f"Snapshot checksum mismatch for {name}")
        series = []
        for asset in ASSETS:
            frame = pd.read_parquet(self.path / (asset + '.parquet'))
            if 'Close' not in frame.columns:
                raise ValueError(f"Missing Close in {asset}")
            dates = pd.DatetimeIndex(pd.to_datetime(frame.index, errors='raise')).tz_localize(None).normalize()
            if dates.hasnans or dates.has_duplicates or not dates.is_monotonic_increasing:
                raise ValueError(f"Invalid or duplicate chronological dates in {asset}")
            values = pd.to_numeric(frame['Close'], errors='raise').to_numpy(dtype=float)
            series.append(pd.Series(values, index=dates, name=asset))
        start = max(x.index.min() for x in series)
        end = min(pd.Timestamp(END_DATE), *(x.index.max() for x in series))
        levels = pd.concat(series, axis=1).loc[start:end]
        if levels.empty or not np.isfinite(levels.to_numpy()).all() or (levels <= 0).any().any():
            raise ValueError("Sector sessions do not align; missing closes cannot be forward-filled")
        levels.index.name = 'date'
        return levels

    def load(self):
        return self.levels().pct_change(fill_method=None).iloc[1:]

    def manifest(self, *, retrieved_at_utc=None):
        levels = self.levels()
        return {'dataset': 'india_nifty_sectors', 'source_url': BASE_URL, 'source_page': SOURCE_PAGE,
                'revision': REVISION, 'tickers': {'NIFTY_BANK': '^NSEBANK', 'NIFTY_IT': '^CNXIT', 'NIFTY_PHARMA': '^CNXPHARMA'}, 'files': {name: digest(self.path / name) for name in HASHES},
                'retrieved_at_utc': retrieved_at_utc, 'assets': list(levels.columns),
                'first_date': str(levels.index.min().date()), 'last_date': str(levels.index.max().date()),
                'rows': len(levels), 'alignment': 'common overlapping period; partial missing rows rejected, no forward fill',
                'excluded_source_dates': {'2026-01-01': 'missing Pharma', '2026-11-12': 'all three closes missing'},
                'cutoff_reason': '2026 source has missing session and missing final row; truncated at 2025-12-31', 'units': 'index close; decimal close-to-close price returns',
                'methodology': 'NSE sector price indices from third-party dataset, not directly tradable; dividends excluded; historical source/vintage limitations apply',
                'license': 'Non-commercial research and personal use; attribution required; underlying provider rights unverified'}
