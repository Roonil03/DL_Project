# Notebooks

> **Global Indian data restriction:** The benchmark rejects selected price series with
> missing values and rejects multiple market series until close timestamps and
> execution availability have been verified. The legacy mode's default five-asset selection
> therefore cannot currently complete. Review the data and select an eligible
> series before executing. Indian sector mode is described in
> [RunningNotebooks.md](../RunningNotebooks.md#indian-sector-data).

Clean notebook templates for the CUDA benchmark. Executed copies are written
under `outputs/<run_id>/notebooks/`; templates keep empty outputs.

- `00_core.ipynb` provides configuration, logging, telemetry, checkpoints,
  models and plotting helpers.
- `10_data_audit.ipynb` records the selected dataset's provenance and integrity.
- `20_benchmark.ipynb` trains one model/seed/fold unit and evaluates baselines.
- `30_validation.ipynb` checks schemas, recovery, GPU and artifact integrity;
  it runs an optional local test suite when present and records its absence otherwise.
- `40_report.ipynb` generates metrics, stability, timing and report plots.

The completed [India study review](../docs/analysis-2026-11-15.md) distinguishes
the archived run's test results from future checkouts, which do not ship tests.

The retired broad notebooks remain recoverable through Git history. Expensive
training is guarded by `RUN_TRAINING=1`; `run_all_models.sh` supplies the full
unit environment and is the supported execution path.
