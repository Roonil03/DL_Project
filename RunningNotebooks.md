# Running the notebook benchmark

> **Global Indian data restriction:** The benchmark rejects selected price series with
> missing values and rejects multiple market series until close timestamps and
> execution availability have been verified. The legacy mode's default five-asset selection
> therefore cannot currently complete. Review the data and select an eligible
> series before executing. The Indian primary and separate US datasets are
> described below.

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
  archive. Raw datasets are excluded from archives.

## Indian sector data

The primary study uses the pinned third-party [Indian market historical dataset](https://github.com/aswinv90/indian-market-historical-data).
Its NIFTY BANK, IT and PHARMA daily parquet files provide Close prices; the
common panel begins in 2011 and covers the 2020 crisis. Returns calculated from
those closes exclude dividends. The source permits noncommercial research and
does not permit raw data redistribution. No new benchmark has been completed.
The pinned source commit is `1f4aff7c3e53a9f2bfb6a167b0d80f4d0c8f8bf9`.
The local directory is `data/india_indices/`; it contains
`NIFTY_BANK.parquet`, `NIFTY_IT.parquet`, `NIFTY_PHARMA.parquet`, and
`index_list.csv`. The configured test years are 2020 and 2023, each with five
training years and one validation year. The fixed source cutoff is 31 December
2025. The pinned panel contains 3,675 complete sessions from 31 January 2011
through that cutoff. Input features and sizing volatility are lagged one
observed session; exact close-time execution is not established. Adjusted
Close has not been verified as dividend-adjusted.

The PowerShell and Bash runners download missing pinned files and validate
their SHA-256 hashes. To prepare them explicitly in the container, use
`docker compose run --rm jupyter /opt/venv/bin/python scripts/prepare_india_indices.py`.
The preparation script validates existing files and refuses hash mismatches.

## FRED US sector data

The planned benchmark uses the [official FRED combined CSV](https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQNQUSB10LMT,NASDAQNQUSB20LMT,NASDAQNQUSB30LMT,NASDAQNQUSB50LMT,NASDAQNQUSB60LMT)
for Nasdaq US Large/Mid Cap total-return indices in technology, health,
financials, industrials, and energy. These are indices, not tradable ETFs.
Nasdaq claims copyright and requires pre-approval for redistribution.

The reviewed local snapshot at `data/fred_nasdaq_sectors/sector_indices.csv`
spans 30 March 2001 through 13 November 2026, with 6,518 complete trading rows
and 168 all-missing holiday rows shared by all five series; there are no
partially missing rows. It has 253 complete observations in 2008 and 257 in
2020. Portfolio returns are derived from percentage changes in index levels.
Its SHA-256 is `b3be7fb4980c2794d338b06eb150aa4cabd4e4de9978b02e8231a09b92e93e2c`.
Keep the exact input snapshot and manifest for reproducibility. No new sector
benchmark results are available.
The runners download the CSV if it is absent; alternatively, run
`docker compose run --rm jupyter /opt/venv/bin/python scripts/prepare_fred_sectors.py`.
The preparation script refuses to replace an existing file. FRED mode uses
2008 and 2020 test folds, five training years, and one validation year; input
features and sizing volatility are lagged one observed session. Run the
separate comparison with `-DatasetMode fred_sectors` in PowerShell or
`DATASET_MODE=fred_sectors` in Bash.

Treat the Indian primary study, US sector comparison, and legacy Kaggle
mixed-market analysis as separate scopes. The US panel is not a worldwide
market sample; do not pool rankings across datasets.

## Docker runner (recommended)

Set `JUPYTER_TOKEN` to a private, nonempty value before starting Compose. The
service binds to host loopback and requires this token. Native Windows does
not provide the supported CUDA extension build path for genuine Mamba in this
project; use WSL2 with NVIDIA Container Toolkit or the CUDA container.

For PowerShell 5.1 or 7, use `scripts/run_dl_benchmark.ps1`. For example:

```powershell
$env:JUPYTER_TOKEN = Read-Host 'Jupyter token'
.\scripts\run_dl_benchmark.ps1 -DatasetMode india_indices -Device gpu -WhatIf
.\scripts\run_dl_benchmark.ps1 -DatasetMode india_indices -Device gpu -RunId india-study
```

The script validates local files and hashes inputs, then runs the notebook
runner in the image's `/opt/venv`. `-WhatIf` does not download data or train:
if a required file is missing it reports the planned download and returns
before the full plan; with files present it checks them and prints the plan.
The PowerShell wrapper supports GPU mode; the Bash runner also accepts an
explicit CPU selection.
Checkpoint files use PyTorch serialization; load only checkpoints from trusted
authors and keep raw market data and credentials outside the result archive.

With the Indian sector data prepared as described above, run from Git Bash or WSL at
the repository root:

```bash
DATASET_MODE=india_indices ./run_all_models.sh
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
$env:DATASET_MODE = "india_indices"
$env:MODELS = "dlinear,lstm,mamba,patchtst"
$env:SEEDS = "42,52,62,72,82"
Remove-Item Env:FOLDS -ErrorAction SilentlyContinue
$env:DEVICE = "gpu"
$env:RESUME_MODE = "auto"
$env:EPOCHS = "100"
$env:MAX_RETRIES = "2"
$env:INCLUDE_DATA = "0"

docker compose build jupyter
docker compose run --rm `
  -e "RUN_ID=$env:RUN_ID" `
  -e "DATASET_MODE=$env:DATASET_MODE" `
  -e "MODELS=$env:MODELS" `
  -e "SEEDS=$env:SEEDS" `
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
| `DATASET_MODE` | `india_indices` | Select `india_indices`, `fred_sectors`, or legacy `global_indian`. |
| `DATA_DIR` | Dataset-specific directory | Input directory under the repository data mount. |
| `FIN_ASSETS` | Dataset-specific selection | Comma-separated assets; the index modes use their configured lists by default. |
| `FOLDS` | Dataset-specific | Comma-separated fold identifiers; index modes use configured folds by default. |
| `DEVICE` | `gpu` | `gpu` or explicit `cpu`. |
| `RESUME_MODE` | `auto` | `auto` resumes compatible units; `restart` archives prior state; `fail` refuses an existing run. |
| `EPOCHS` | `100` | Positive training epoch count. |
| `MAX_RETRIES` | `2` | Retries after a failure only when a compatible checkpoint exists. |
| `OUTPUT_ROOT` | `outputs` | Root directory for run outputs. |
| `INCLUDE_DATA` | `0` | Raw source data cannot be included; `1` is rejected. |
| `PACKAGE_CHECKPOINTS` | `1` | Set to `0` to omit checkpoints from the ZIP; PowerShell uses `-ExcludeCheckpoints`. |
| `ESTIMATE_ONLY` | `0` | Set to `1` for a measured one-epoch GPU timing pilot. |

Examples from Git Bash/WSL (the CPU smoke test is for development checks,
not the full GPU benchmark):

```bash
DATASET_MODE=india_indices RUN_ID=pilot ESTIMATE_ONLY=1 ./run_all_models.sh
DATASET_MODE=india_indices RUN_ID=smoke MODELS=dlinear,lstm SEEDS=42 FOLDS=0 EPOCHS=2 DEVICE=cpu ./run_all_models.sh
DATASET_MODE=india_indices RUN_ID=full RESUME_MODE=auto MAX_RETRIES=2 ./run_all_models.sh
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
`outputs/<run_id>/checkpoints/<model>/seed_<seed>/fold_<fold>/`. In `auto` mode, completed
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
`Final_Metrics.md`, creates `artifacts/<run_id>.zip`, and verifies it with
Python's `ZipFile.testzip()` and SHA-256 checks before reporting success. The ZIP includes checkpoints by default; opt out with
`PACKAGE_CHECKPOINTS=0` or PowerShell's `-ExcludeCheckpoints`. Raw data is
excluded and `INCLUDE_DATA=1` is rejected. Failed runs preserve partial logs, telemetry, and
checkpoints but do not create a publishable ZIP.

## Replaying a completed run

Extract the archive into a separate directory and inspect
`Final_Metrics.md`, `metadata/run_manifest.json`, and `bundle_manifest.json`.
The archive is a record of the completed run; rerunning is only necessary when
changing the data, configuration, image, model set, or seed set.
