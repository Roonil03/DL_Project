"""Pinned-file access to Kenneth French's daily industry research returns."""
from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/10_Industry_Portfolios_daily_CSV.zip"
SOURCE_PAGE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html"
DEFAULT_PATH = "data/french_industry/10_Industry_Portfolios_daily_CSV.zip"
ASSETS = ("NoDur", "Durbl", "Manuf", "Enrgy", "HiTec", "Telcm", "Shops", "Hlth", "Utils", "Other")


class FrenchIndustryAdapter:
    """Read the value-weighted daily block; returns are decimal, not prices.

    The official ZIP is mutable. Keep the original bytes and record its hash;
    the source does not provide a point-in-time historical vintage guarantee.
    """
    def __init__(self, data_path=DEFAULT_PATH):
        self.path = Path(data_path)

    def load(self):
        if not self.path.is_file():
            raise FileNotFoundError(
                f"French industry ZIP not found: {self.path}. Download the official file "
                f"from {SOURCE_URL} to that path; see RunningNotebooks.md."
            )
        raw = self.path.read_bytes()
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = [n for n in archive.namelist() if n.lower().endswith('.csv') and not n.endswith('/')]
                if len(members) != 1:
                    raise ValueError("Expected exactly one CSV in the French industry ZIP")
                text = archive.read(members[0]).decode("utf-8-sig")
        except (zipfile.BadZipFile, UnicodeError) as exc:
            raise ValueError("Invalid French industry ZIP or CSV encoding") from exc
        lines = text.splitlines()
        starts = [i for i, line in enumerate(lines) if 'average value weighted returns' in line.lower() and 'daily' in line.lower()]
        if len(starts) != 1:
            raise ValueError("Expected one daily value-weighted return section")
        header_idx = starts[0] + 1
        while header_idx < len(lines) and not lines[header_idx].strip():
            header_idx += 1
        header = [field.strip() for field in next(csv.reader([lines[header_idx]]))]
        if tuple(header[1:]) != ASSETS:
            raise ValueError(f"Unexpected industry columns: {header[1:]}")
        rows = []
        for line in lines[header_idx + 1:]:
            fields = [field.strip() for field in next(csv.reader([line]))]
            if not fields or not re.fullmatch(r"\d{8}", fields[0]):
                if rows:
                    break
                continue
            if len(fields) != len(ASSETS) + 1:
                raise ValueError("Malformed daily industry row")
            rows.append(fields)
        if not rows:
            raise ValueError("No daily industry returns found")
        dates = pd.to_datetime([row[0] for row in rows], format="%Y%m%d", errors="raise")
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Daily return dates must be unique and increasing")
        frame = pd.DataFrame([row[1:] for row in rows], index=dates, columns=ASSETS).apply(pd.to_numeric, errors="raise")
        frame = frame.replace({-99.99: float('nan'), -999.0: float('nan')}) / 100.0
        if (frame < -1).any().any():
            raise ValueError("Industry return below -100%; check the input")
        frame.index.name = "date"
        return frame

    def manifest(self, *, retrieved_at_utc=None):
        frame = self.load()
        return {
            "dataset": "french_10_industry_daily", "source_url": SOURCE_URL,
            "source_page": SOURCE_PAGE, "path": str(self.path),
            "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "retrieved_at_utc": retrieved_at_utc,
            "source_vintage": "mutable download; exact vintage identified by SHA-256",
            "methodology": "value-weighted daily US industry portfolio returns; current CRSP-based reconstruction",
            "units": "decimal daily total return", "assets": list(frame.columns),
            "first_date": str(frame.index.min().date()), "last_date": str(frame.index.max().date()),
            "rows": len(frame), "missing_by_asset": frame.isna().sum().to_dict(),
            "license": "not verified; source page declares copyright",
        }
