# Docker usage — Financial DL Benchmark

All Python dependencies live in an isolated venv **inside the image** (`/opt/venv`).
Nothing is installed on the host and no host root paths are touched. Host files
are only accessed through explicit mounts at `docker compose` run time.

## Prerequisites (host)

- Docker Engine + Docker Compose v2.
- Optional for GPU training: NVIDIA Container Toolkit.
- Place the Kaggle CSV at `data/global_indian_markets.csv` (git-ignored).
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

- `notebooks/03_dataset_audit.ipynb` — data integrity check (run first).
- `notebooks/02_run_benchmark.ipynb` — full pipeline; inspection-safe with
  `RUN_TRAINING = False`. Set `True` only for the expensive 4-model x 5-seed run.
- `notebooks/financial_time_series_benchmark.ipynb` — original 19-cell scaffold
  (pre-audit conclusions; retained for traceability).

Equivalent script runs inside the container:

```bash
docker compose run --rm jupyter /opt/venv/bin/python audit_dataset.py
docker compose run --rm jupyter /opt/venv/bin/python run_benchmark.py
```

## Mounts

| Host path | Container path | Mode | Purpose |
|---|---|---|---|
| `./data` | `/home/jovyan/work/data` | `ro` | Kaggle CSV input |
| `./results` | `/home/jovyan/work/results` | `rw` | regenerated tables/figures |
| `./notebooks` | `/home/jovyan/work/notebooks` | `rw` | live notebook editing |
| `./configs` | `/home/jovyan/work/configs` | `ro` | experiment configs |

## Notes

- Base image: `quay.io/jupyter/pytorch-notebook:python-3.11` (GPU-capable torch).
  CPU-only hosts work unchanged; for NVIDIA GPUs uncomment `runtime: nvidia`.
- `mamba-ssm` is intentionally **not** baked in (platform-sensitive). Install it
  in a derived image before claiming any Mamba result:
  `pip install mamba-ssm>=2.2` (see `remaining_work.md` Priority 1).
- Committed `results/` files predate the Sept 2026 audit — regenerate them from
  the verified dataset before citing any ranking.
