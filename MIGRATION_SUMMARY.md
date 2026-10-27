# Notebook conversion summary

The `notebook-conversion` branch replaces the two broad orchestration notebooks
with a clean five-notebook suite while preserving `src/` and `tests/` as the
canonical reusable Python library.

- `00_core.ipynb`: shared runtime, telemetry, checkpoint, model, and plotting helpers.
- `10_data_audit.ipynb`: exact Kaggle data provenance and schema manifest.
- `20_benchmark.ipynb`: ordered 33-section walk-forward benchmark.
- `30_validation.ipynb`: test and artifact acceptance gates.
- `40_report.ipynb`: aggregate metrics, figures, and `Final_Metrics.md`.

The Docker runtime now uses a pinned CUDA development image, an uv-managed
Python 3.11 interpreter, an isolated `/opt/venv`, pip3-installed CUDA PyTorch,
and strict real-Mamba dependencies. `run_all_models.sh` is the supported host
and container entrypoint for preflight, recovery, execution, validation,
reporting, and packaging.

Raw data, run outputs, and ZIP artifacts remain git-ignored. A successful run
records source/config/dependency/data identities and produces the only
publishable result archive; incomplete runs remain unbundled.
