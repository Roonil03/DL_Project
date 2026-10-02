# Notebooks

Unexecuted, outputs-cleared mirrors — safe to open without triggering training.
Run them inside the Docker container on kernel **Financial DL Benchmark**
(`fin-dl-benchmark`); nothing is installed on the host.

| Notebook | Source script (kept) | Purpose |
|---|---|---|
| `financial_time_series_benchmark.ipynb` | — (original scaffold, 19 cells) | Early pipeline scaffold with pre-audit outputs/conclusions; retained for traceability, not report-ready |
| `02_run_benchmark.ipynb` | `../run_benchmark.py` | Full 7-stage pipeline mirror; training cell gated by `RUN_TRAINING = False` (`OPTIONAL — EXECUTE TO TRAIN`) |
| `03_dataset_audit.ipynb` | `../audit_dataset.py` | Dataset integrity check for `data/global_indian_markets.csv` |

Conventions: `execution_count: null`, no stored outputs, `RUN_TRAINING` guard on
expensive cells, `CODE VALIDATION ONLY` labeling where synthetic checks apply.
See `../DOCKER.md` for container run steps and `../remaining_work.md` for why
`results/` must be regenerated before any claim.
