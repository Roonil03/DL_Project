# Notebooks

Clean notebook templates for the CUDA benchmark. Executed copies are written
under `outputs/<run_id>/notebooks/`; templates keep empty outputs.

| Notebook | Source | Purpose |
|---|---|---|
| `00_core.ipynb` | Shared configuration, logging, telemetry, checkpoint, model, and plotting helpers. |
| `10_data_audit.ipynb` | Exact Kaggle source and schema/integrity manifest. |
| `20_benchmark.ipynb` | Ordered 33-section model, baseline, and walk-forward benchmark. |
| `30_validation.ipynb` | Tests, schemas, recovery, GPU, and artifact checks. |
| `40_report.ipynb` | Final metrics, stability, timing, failure, and report plots. |

The retired broad notebooks remain recoverable through Git history. Expensive
training is guarded by `RUN_TRAINING=1`; `run_all_models.sh` supplies the full
unit environment and is the supported execution path.
