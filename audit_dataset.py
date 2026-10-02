#!/usr/bin/env python
"""Dataset audit script for the Financial DL Benchmark.

Provenance: restored as valid Python on 2026-10-02. The previous
`audit_dataset.py` revision accidentally contained Jupyter notebook JSON
instead of Python source. This file restores the executable script logic
(identical audit behaviour) extracted from that embedded cell.

Notebook mirror: `notebooks/03_dataset_audit.ipynb` (unexecuted,
outputs cleared).

Usage (inside Docker venv, host remains untouched):
    python audit_dataset.py

Expected input (git-ignored, not committed):
    data/global_indian_markets.csv
"""

import hashlib
import pandas as pd


def audit_data(path: str = "data/global_indian_markets.csv") -> None:
    """Print a lightweight integrity report for the primary dataset."""
    if not pd.io.common.file_exists(path):
        print(f"Dataset not found at {path}. Please place the Kaggle CSV there.")
        return

    df = pd.read_csv(path)
    print("--- DATASET AUDIT REPORT ---")
    print(f"File Path: {path}")
    print(f"Total Records: {len(df):,}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data Types:\n{df.dtypes}")
    print(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"Duplicate Rows: {df.duplicated().sum()}")

    # Generate SHA-256 Hash for reproducibility
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"Dataset SHA-256 Hash: {file_hash}")


if __name__ == "__main__":
    audit_data()
