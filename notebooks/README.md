# Notebooks

Unexecuted, outputs-cleared mirror — safe to open without triggering training.
Run inside the Docker container on kernel **Financial DL Benchmark**
(`fin-dl-benchmark`); nothing is installed on the host.

| Notebook | Source | Purpose |
|---|---|---|
| `benchmark.ipynb` | retired `audit_dataset.py` (Part A) + `run_benchmark.py` (Part B), live `src/` | Single consolidated notebook: §0 Kaggle-API dataset download, Part A integrity check, then the full 7-stage pipeline. Training cell gated by `RUN_TRAINING = False` (`OPTIONAL — EXECUTE TO TRAIN`) |
| `full_and_tests.ipynb` | same as above + live `../tests/*.py` mirrors | Everything in `benchmark.ipynb` plus the test-suite mirrors for inspection |

History: this folder previously held `02_run_benchmark.ipynb`,
`03_dataset_audit.ipynb`, and the original `financial_time_series_benchmark.ipynb`
(19 cells with pre-Sept-2026-audit outputs/conclusions). They were consolidated
into `benchmark.ipynb`; the originals remain recoverable via git history. The
original's stale conclusions were intentionally not carried over — see
`../remaining_work.md` for why `results/` must be regenerated before any claim.

Conventions: `execution_count: null`, no stored outputs, `RUN_TRAINING` guard on
expensive cells, `CODE VALIDATION ONLY` labeling where synthetic checks apply.
