# Notebooks

Executed notebook records from the 5 October 2026 Docker run. Re-running them
uses the **Financial DL Benchmark** (`fin-dl-benchmark`) kernel; nothing is
installed on the host. `RUN_TRAINING` defaults to false unless explicitly set
in the container environment.

| Notebook | Source | Purpose |
|---|---|---|
| `benchmark.ipynb` | retired `audit_dataset.py` (Part A) + `run_benchmark.py` (Part B), live `src/` | Executed full benchmark on the verified public Kaggle CSV; DLinear, LSTM, and PatchTST trained across five seeds. Mamba was truthfully skipped because `mamba_ssm` is unavailable. |
| `full_and_tests.ipynb` | same as above + live `../tests/*.py` mirrors | Executed in inspection mode after the full run; includes test-suite mirrors for inspection. |

History: this folder previously held `02_run_benchmark.ipynb`,
`03_dataset_audit.ipynb`, and the original `financial_time_series_benchmark.ipynb`
(19 cells with pre-Sept-2026-audit outputs/conclusions). They were consolidated
into `benchmark.ipynb`; the originals remain recoverable via git history. The
original's stale conclusions were intentionally not carried over. Current
results and limitations are recorded in `../Final_Metrics.md`.

Conventions: stored outputs document the verified run; `RUN_TRAINING` guards
expensive cells and defaults to false; synthetic checks remain labelled
`CODE VALIDATION ONLY`.
