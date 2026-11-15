# CUDA Docker runtime

> **Global Indian data restriction:** The benchmark rejects selected price series with
> missing values and rejects multiple market series until close timestamps and
> execution availability have been verified. The default five-asset selection
> therefore cannot currently complete. Review the data and select an eligible
> series before executing. French notebook mode is described in
> [RunningNotebooks.md](RunningNotebooks.md#french-industry-data-and-notebook-mode).

The benchmark image is pinned to CUDA 12.4, Python 3.11, and a CUDA-enabled
PyTorch build. `uv` installs the interpreter and creates `/opt/venv`; all
project packages are installed with `/opt/venv/bin/pip3`. Conda and the Mamba
environment manager are not used (`mamba-ssm` is the model library).

## Host requirements

- NVIDIA driver compatible with CUDA 12.4.
- Docker Desktop/Engine with Compose v2 and NVIDIA Container Toolkit support.
- The selected dataset under the repository's `data/` directory; see
  [RunningNotebooks.md](RunningNotebooks.md) for source-specific files.

## Run

```bash
DATASET_MODE=french_industry FOLDS= ./run_all_models.sh
```

When invoked on the host, the script checks the Docker daemon, reuses the
local image or builds it if missing, starts the Jupyter
service, and re-enters itself inside the container. It then runs CUDA/Mamba
preflight, data audit, one benchmark notebook per model/seed/fold, validation,
report generation, checksums, and ZIP integrity checks.

Common overrides:

```bash
DATASET_MODE=french_industry RUN_ID=pilot ESTIMATE_ONLY=1 ./run_all_models.sh
DATASET_MODE=french_industry RUN_ID=smoke MODELS=dlinear,lstm SEEDS=42 FOLDS=0 EPOCHS=2 ./run_all_models.sh
DATASET_MODE=french_industry RUN_ID=full DEVICE=gpu RESUME_MODE=auto MAX_RETRIES=2 ./run_all_models.sh
```

`DEVICE=gpu` fails fast when CUDA is unavailable. CPU use must be explicit with
`DEVICE=cpu`. `RESUME_MODE` is `auto`, `restart`, or `fail`. Raw data is omitted
from the final ZIP unless `INCLUDE_DATA=1` is set. Checkpoints are included
by default; `PACKAGE_CHECKPOINTS=0` excludes them.

JupyterLab is available separately with:

```bash
docker compose up --build jupyter
```

Open <http://localhost:8888> and select **Financial DL Benchmark (CUDA)**.
Clean templates are under `notebooks/`; executed copies are always stored under
the selected run directory.

## Output contract

Successful full runs produce `outputs/<run_id>/`, root `Final_Metrics.md`, and
`artifacts/<run_id>.zip`. A timing-only pilot reports `pilot_complete` and
does not create a ZIP. Failed runs preserve partial telemetry and
checkpoints, return a nonzero status, and never create a ZIP.
