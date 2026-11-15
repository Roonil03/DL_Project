# Running the notebook benchmark

> **Global Indian data restriction:** The benchmark rejects selected price series with
> missing values and rejects multiple market series until close timestamps and
> execution availability have been verified. The default five-asset selection
> therefore cannot currently complete. Review the data and select an eligible
> series before executing. French industry mode is described below.

This document contains the execution instructions for the notebook suite. The
recommended path is the pinned CUDA Docker image; it keeps the host Python
environment out of the audited run and fails fast when a requested GPU or real
Mamba implementation is unavailable.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2.
- NVIDIA Container Toolkit and a driver compatible with the CUDA version in
  [`DOCKER.md`](DOCKER.md) for `DEVICE=gpu`.
- For `global_indian` mode, the Kaggle files under:

  ```text
  data/kaggle_global_indian_markets/global-india-markets-macro/
  ├── daily_market_data.csv
  ├── monthly_macro_data.csv
  └── codes_and_sources.md
  ```

- Enough disk space for checkpoints, figures, executed notebooks, and the ZIP
  archive. The raw dataset is not included in the archive by default.

## French industry data and notebook mode

The [official Kenneth R. French data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_10_ind_port.html)
provides daily value-weighted US industry research portfolio returns. These are
constructed portfolios, not directly tradable assets, and have no macro series.
The local ZIP is currently absent: the attempted official download failed due
to DNS. The source declares copyright; verify use and redistribution terms.

With Docker and the GPU-enabled Compose service available, set `JUPYTER_TOKEN`
as described below, build the image, and run the preparation script in its
virtual environment. From PowerShell at the repository root:

```powershell
docker compose build jupyter
docker compose run --rm jupyter /opt/venv/bin/python scripts/prepare_french_industry.py
```

The script downloads and validates the ZIP, records its SHA-256 and retrieval
time in a sidecar manifest, and refuses to overwrite an existing snapshot. If
DNS still fails, obtain the official `10_Industry_Portfolios_daily_CSV.zip`
separately and place it at `data/french_industry/10_Industry_Portfolios_daily_CSV.zip`.
The audit notebook validates it and records its hash; retain the exact ZIP
because the library can reconstruct historical returns on later updates.

For interactive notebook use, start Jupyter with `DATASET_MODE=french_industry`
and the same `RUN_ID` for the audit and benchmark. For example, after setting
the token, run:

```powershell
docker compose run --rm --service-ports -e DATASET_MODE=french_industry `
  -e RUN_ID=french-study jupyter
```

Run `10_data_audit.ipynb` before `20_benchmark.ipynb`; the benchmark verifies
the source hash, selection, and config against the audit manifest. The default
directory is `data/french_industry/`. `configs/french_industry.yaml` defines
the ten industry columns and folds; optional `FIN_ASSETS` selects a
comma-separated subset and must match at audit and training. Fold 0 tests
2008 and fold 1 tests 2020, each using five prior training years and an
intervening validation year. Signals use information through the decision
close and target the next US trading day's return. Regimes use cumulative
research returns as indices. Live execution and actual costs are not established.
Training is disabled by default. To execute a unit, provide `RUN_TRAINING=1`,
`MODEL`, `SEED`, `FOLD`, `DEVICE`, and the audited `RUN_ID` in the notebook
environment before running the benchmark cells. Keep the same data selection
and config used for the audit.
The runners also support French mode; the PowerShell example below uses both
configured folds unless `-Folds` is supplied.

## Docker runner (recommended)

Set `JUPYTER_TOKEN` to a private, nonempty value before starting Compose. The
service binds to host loopback and requires this token. Native Windows does
not provide the supported CUDA extension build path for genuine Mamba in this
project; use WSL2 with NVIDIA Container Toolkit or the CUDA container.

For PowerShell 5.1 or 7, use `scripts/run_dl_benchmark.ps1`. For example:

```powershell
$env:JUPYTER_TOKEN = Read-Host 'Jupyter token'
.\scripts\run_dl_benchmark.ps1 -DatasetMode french_industry -Device gpu -WhatIf
.\scripts\run_dl_benchmark.ps1 -DatasetMode french_industry -Device gpu -RunId french-study
```

The script validates local files and hashes inputs, then runs the notebook
runner in the image's `/opt/venv`. `-WhatIf` prints the plan without training.
The PowerShell wrapper supports GPU mode; the Bash runner also accepts an
explicit CPU selection.
Checkpoint files use PyTorch serialization; load only checkpoints from trusted
authors and keep raw market data and credentials outside the result archive.

From Git Bash or WSL at the repository root:

```bash
./run_all_models.sh
```

The host entrypoint checks the Docker daemon, reuses the local image or builds
it when missing, starts the Compose Jupyter service, and
re-enters the same script inside the container. The container then performs
the CUDA tensor and genuine-Mamba preflight, audits the data, executes one
benchmark notebook per model/seed/fold, runs validation and reporting, writes
the final metrics, and creates `artifacts/<run_id>.zip` only when every required
unit succeeds.

To run the container directly from PowerShell (without invoking Bash), set the
run parameters and execute the mounted script inside Compose:

```powershell
$env:RUN_ID = "benchmark-$(Get-Date -Format yyyyMMdd-HHmmss)"
$env:MODELS = "dlinear,lstm,mamba,patchtst"
$env:SEEDS = "42,52,62,72,82"
$env:FOLDS = "0"
$env:DEVICE = "gpu"
$env:RESUME_MODE = "auto"
$env:EPOCHS = "100"
$env:MAX_RETRIES = "2"
$env:INCLUDE_DATA = "0"

docker compose build jupyter
docker compose run --rm `
  -e "RUN_ID=$env:RUN_ID" `
  -e "MODELS=$env:MODELS" `
  -e "SEEDS=$env:SEEDS" `
  -e "FOLDS=$env:FOLDS" `
  -e "DEVICE=$env:DEVICE" `
  -e "RESUME_MODE=$env:RESUME_MODE" `
  -e "EPOCHS=$env:EPOCHS" `
  -e "MAX_RETRIES=$env:MAX_RETRIES" `
  -e "INCLUDE_DATA=$env:INCLUDE_DATA" `
  jupyter ./run_all_models.sh
```

For a CPU smoke test, use `DEVICE=cpu` explicitly. GPU mode never silently
falls back to CPU.

## Runner parameters

| Variable | Default | Purpose |
|---|---|---|
| `RUN_ID` | UTC timestamp | Output directory and archive name. |
| `MODELS` | `dlinear,lstm,mamba,patchtst` | Comma-separated model list. |
| `SEEDS` | `42,52,62,72,82` | Comma-separated random seeds. |
| `DATASET_MODE` | `global_indian` | Select `global_indian` or `french_industry`. |
| `DATA_DIR` | Dataset-specific directory | Input directory under the repository data mount. |
| `FIN_ASSETS` | Dataset-specific selection | Comma-separated assets; French defaults to its configured list. |
| `FOLDS` | Global: `0`; French: configured folds | Comma-separated fold identifiers. |
| `DEVICE` | `gpu` | `gpu` or explicit `cpu`. |
| `RESUME_MODE` | `auto` | `auto` resumes compatible units; `restart` archives prior state; `fail` refuses an existing run. |
| `EPOCHS` | `100` | Positive training epoch count. |
| `MAX_RETRIES` | `2` | Retries after a failure only when a compatible checkpoint exists. |
| `OUTPUT_ROOT` | `outputs` | Root directory for run outputs. |
| `INCLUDE_DATA` | `0` | Set to `1` to include raw data in the ZIP. |
| `PACKAGE_CHECKPOINTS` | `1` | Set to `0` to omit checkpoints from the ZIP; PowerShell uses `-ExcludeCheckpoints`. |
| `ESTIMATE_ONLY` | `0` | Set to `1` for a measured one-epoch GPU timing pilot. |

Examples from Git Bash/WSL:

```bash
DATASET_MODE=french_industry RUN_ID=pilot ESTIMATE_ONLY=1 ./run_all_models.sh
DATASET_MODE=french_industry RUN_ID=smoke MODELS=dlinear,lstm SEEDS=42 FOLDS=0 EPOCHS=2 DEVICE=cpu ./run_all_models.sh
DATASET_MODE=french_industry RUN_ID=full RESUME_MODE=auto MAX_RETRIES=2 ./run_all_models.sh
```

## Interactive JupyterLab

To inspect or edit the clean templates interactively:

```bash
docker compose up --build jupyter
```

Open <http://localhost:8888> and select **Financial DL Benchmark (CUDA)**.
Templates remain under `notebooks/`; executed copies are written into the
selected run directory, so the source notebooks stay clean.

## Notebook order

The runner executes the notebooks in this order:

1. `00_core.ipynb` — runtime helpers, telemetry, checkpoints, and plots.
2. `10_data_audit.ipynb` — source files, hashes, schema, dates, missingness,
   duplicates, and license metadata.
3. `20_benchmark.ipynb` — model training, walk-forward evaluation, baselines,
   regime analysis, statistical tests, and figures.
4. `30_validation.ipynb` — integrity, failure-injection, checkpoint/resume,
   GPU, schema, and archive checks.
5. `40_report.ipynb` — reproducible tables, rankings, timing, and final plots.

Training is disabled in clean templates. The runner sets `RUN_TRAINING=1` for
each requested model/seed/fold unit and saves the executed notebook under
`outputs/<run_id>/notebooks/`.

## Resume, failure, and timing behaviour

Each unit writes atomic checkpoints under
`outputs/<run_id>/checkpoints/<model>/seed_<seed>/`. In `auto` mode, completed
units are skipped and failed units resume only from a compatible checkpoint.
Corrupt, incompatible, or missing checkpoints are recorded as failures rather
than silently restarting. `ESTIMATE_ONLY=1` measures pilot timing on the
actual selected device and writes `telemetry/timing.json`.
The pilot ends with status `pilot_complete` and does not produce a ZIP.

## Outputs and portable archive

A successful run produces:

```text
outputs/<run_id>/
├── metadata/       # run, system, data, config, dependency manifests
├── logs/           # run.log and structured events.jsonl
├── telemetry/      # system and epoch metrics, timing, failures, and run state
├── checkpoints/    # latest and best model checkpoints
├── figures/        # PNG figures and exported plot data
├── tables/         # per-unit metrics and regime tables
├── predictions/    # predictions and returns
└── notebooks/      # executed notebook copies
```

The runner copies the source code and execution files, including `LICENSE` and
`requirements.cpu.txt`, into the run tree,
creates `bundle_manifest.json` with SHA-256 checksums, writes
`Final_Metrics.md`, creates `artifacts/<run_id>.zip`, and runs `unzip -t` before
reporting success. The ZIP includes checkpoints by default; opt out with
`PACKAGE_CHECKPOINTS=0` or PowerShell's `-ExcludeCheckpoints`. Raw data is
excluded unless `INCLUDE_DATA=1` is passed to the Bash runner. Failed runs preserve partial logs, telemetry, and
checkpoints but do not create a publishable ZIP.

## Replaying a completed run

Extract the archive into a separate directory and inspect
`Final_Metrics.md`, `metadata/run_manifest.json`, and `bundle_manifest.json`.
The archive is a record of the completed run; rerunning is only necessary when
changing the data, configuration, image, model set, or seed set.
