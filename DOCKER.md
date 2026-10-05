# Docker usage — Financial DL Benchmark

All Python dependencies live in an isolated venv **inside the image** (`/opt/venv`).
Nothing is installed on the host and no host root paths are touched. Host files
are only accessed through explicit mounts at `docker compose` run time.

## Prerequisites (host)

- Docker Engine + Docker Compose v2.
- Optional for GPU training: a CUDA-enabled PyTorch base/build plus NVIDIA
  Container Toolkit. The pinned base image must be checked because some tags
  ship CPU-only PyTorch.
- Place the Kaggle daily-market CSV at `data/global_indian_markets.csv`
  (git-ignored), or let notebook `§0` fetch the public Kaggle archive and extract
  `daily_market_data.csv` to that canonical path. Credentials are not required
  for this public download.
- No `pip install`, no `python -m venv` on the host.

## Build (manual — not run during migration)

```bash
docker build -t fin-dl-benchmark .
# or
docker compose build jupyter
```

## Run Jupyter (manual)

```bash
docker compose up jupyter
# open http://localhost:8888
```

Inside JupyterLab, select kernel **Financial DL Benchmark** (`fin-dl-benchmark`),
then open:

- `notebooks/benchmark.ipynb` — single notebook: data integrity check (Part A, run first), then the full pipeline (Part B, inspection-safe with
  `RUN_TRAINING = False`). Set `True` only for the expensive 4-model x 5-seed run.
- `notebooks/full_and_tests.ipynb` — same as above plus the `tests/` suite mirrors for inspection.

Tests run inside the container (the old root runners were retired in favor of
the notebooks; recoverable via git history):

```bash
docker compose run --rm jupyter /opt/venv/bin/python -m pytest tests/ -v
```

## Mounts

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `./data` | `/home/jovyan/work/data` | `rw` | Git-ignored Kaggle dataset cache |
| `./results` | `/home/jovyan/work/results` | `rw` | regenerated tables/figures |
| `./notebooks` | `/home/jovyan/work/notebooks` | `rw` | live notebook editing |
| `./configs` | `/home/jovyan/work/configs` | `ro` | experiment configs |

## Notes

- Base image: `quay.io/jupyter/pytorch-notebook:python-3.11`. The project venv
  reuses its in-image PyTorch/scientific packages to avoid installing a second
  multi-gigabyte framework stack. CPU-only hosts work unchanged. GPU execution
  additionally requires a CUDA-enabled base/build and NVIDIA Container Toolkit;
  `runtime: nvidia` alone cannot turn a CPU-only PyTorch build into a GPU build.
- `mamba-ssm` is intentionally **not** baked in (platform-sensitive). Install it
  in a derived image before claiming any Mamba result:
  `pip install mamba-ssm>=2.2` (see `remaining_work.md` Priority 1).
- The committed `results/` files and `Final_Metrics.md` were regenerated on
  5 October 2026 from the verified public Kaggle daily-market CSV. Mamba remains
  excluded until a compatible real `mamba_ssm` environment is used.
