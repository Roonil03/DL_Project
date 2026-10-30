# Running the notebook benchmark

This document contains the execution instructions for the notebook suite. The
recommended path is the pinned CUDA Docker image; it keeps the host Python
environment out of the audited run and fails fast when a requested GPU or real
Mamba implementation is unavailable.

## Prerequisites

- Docker Desktop or Docker Engine with Compose v2.
- NVIDIA Container Toolkit and a driver compatible with the CUDA version in
  [`DOCKER.md`](DOCKER.md) for `DEVICE=gpu`.
- The audited Kaggle files under:

  ```text
  data/kaggle_global_indian_markets/global-india-markets-macro/
  ├── daily_market_data.csv
  ├── monthly_macro_data.csv
  └── codes_and_sources.md
  ```

- Enough disk space for checkpoints, figures, executed notebooks, and the ZIP
  archive. The raw dataset is not included in the archive by default.

## Docker runner (recommended)

From Git Bash or WSL at the repository root:

```bash
./run_all_models.sh
```

The host entrypoint builds the image, starts the Compose Jupyter service, and
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
| `FOLDS` | `0` | Comma-separated walk-forward fold identifiers. |
| `DEVICE` | `gpu` | `gpu` or explicit `cpu`. |
| `RESUME_MODE` | `auto` | `auto` resumes compatible units; `restart` archives prior state; `fail` refuses an existing run. |
| `EPOCHS` | `100` | Positive training epoch count. |
| `MAX_RETRIES` | `2` | Retries after a failure only when a compatible checkpoint exists. |
| `OUTPUT_ROOT` | `outputs` | Root directory for run outputs. |
| `INCLUDE_DATA` | `0` | Set to `1` to include raw data in the ZIP. |
| `ESTIMATE_ONLY` | `0` | Set to `1` for a measured one-epoch GPU timing pilot. |

Examples from Git Bash/WSL:

```bash
RUN_ID=pilot ESTIMATE_ONLY=1 ./run_all_models.sh
RUN_ID=smoke MODELS=dlinear,lstm SEEDS=42 FOLDS=0 EPOCHS=2 DEVICE=cpu ./run_all_models.sh
RUN_ID=full RESUME_MODE=auto MAX_RETRIES=2 ./run_all_models.sh
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

## Outputs and portable archive

A successful run produces:

```text
outputs/<run_id>/
├── metadata/       # run, system, data, config, dependency manifests
├── logs/           # run.log and structured events.jsonl
├── telemetry/      # metrics, timing, failures, and run state
├── checkpoints/    # latest and best model checkpoints
├── figures/        # PNG figures and exported plot data
├── tables/         # per-unit metrics and regime tables
├── predictions/    # predictions and returns
└── notebooks/      # executed notebook copies
```

The runner copies the source code and execution files into the run tree,
creates `bundle_manifest.json` with SHA-256 checksums, writes
`Final_Metrics.md`, creates `artifacts/<run_id>.zip`, and runs `unzip -t` before
reporting success. Failed runs preserve partial logs, telemetry, and
checkpoints but do not create a publishable ZIP.

## Replaying a completed run

Extract the archive into a separate directory and inspect
`Final_Metrics.md`, `metadata/run_manifest.json`, and `bundle_manifest.json`.
The archive is a record of the completed run; rerunning is only necessary when
changing the data, configuration, image, model set, or seed set.
