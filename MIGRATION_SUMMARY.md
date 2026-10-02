# Migration Summary — Notebook Conversion + Docker Containerization

Date: 2026-10-02. Scope: only this repo and subdirectories. No push performed.
Nothing was executed: no notebook ran, no `docker build/run`, no model training,
no host `pip install` or host venv was created.

## 1. Repository inventory (read before changing)

- Root scripts: `run_benchmark.py` (366 lines, valid Python, 7-stage pipeline),
  `audit_dataset.py` (93 lines, anomaly: contained notebook JSON, not Python).
- Library (kept as `.py` so `import src.*` / `pytest` keep working): 20 files in
  `src/data|models|losses|portfolio|regimes|training|evaluation|backtest|visualization`
  (~1200 lines) + 3 test files (~213 lines).
- Existing notebook: `notebooks/financial_time_series_benchmark.ipynb` (19 cells,
  executed outputs + pre-Sept-2026-audit conclusions; spec wants 33 sections).
- Configs: `data|models|training|backtest|regime.yaml`; `requirements.txt` (13 loose
  names), `environment.yml` (python 3.11 vs host 3.12.3).
- No Docker files, no `src/**/__init__.py`, incomplete `.gitignore`.
- `results/` (10 figures + 8 tables) predates the audit — preserved untouched.

## 2. Python → notebook conversion (scripts kept + notebooks added)

| Source `.py` (kept, canonical) | New `.ipynb` (unexecuted, outputs cleared) | Notes |
|---|---|---|
| `run_benchmark.py` | `notebooks/02_run_benchmark.ipynb` | Split into staged markdown/code cells (configs → ingest → features/regimes → splits/scaling/windows → 4 model factories → OPTIONAL training → metrics/regimes/costs/stats → figures). Expensive multi-seed cell gated by `RUN_TRAINING = False`. Mamba `allow_fallback` contradiction flagged in markdown, not silently fixed. Kernel `fin-dl-benchmark`. |
| `audit_dataset.py` (restored, see below) | `notebooks/03_dataset_audit.ipynb` | Integrity check for git-ignored `data/global_indian_markets.csv`. Run line left commented; manifest step points to `remaining_work.md` P0. |
| — | `notebooks/README.md` | Index of 01 (original) vs 02/03 (new) + conventions. |
| `audit_dataset.py` itself | fixed in place | Was notebook JSON; restored as valid Python (`audit_data()` + `__main__`), identical behaviour, plus provenance docstring. |

`src/` and `tests/` were intentionally **not** converted: notebook-izing them would
break imports and the `instructions.txt` requirement to keep both script and
notebook versions. Only the runnable top-level scripts got notebook mirrors.

## 3. Containerization (Jupyter image + in-image venv)

- `Dockerfile` (`FROM quay.io/jupyter/pytorch-notebook:python-3.11`):
  `root` only for `apt-get install git` + `/opt/venv` creation, then `USER jovyan`;
  `python -m venv /opt/venv`, `pip install -r requirements.docker.txt`,
  `ipykernel install --name fin-dl-benchmark`. `WORKDIR /home/jovyan/work`,
  `EXPOSE 8888`, non-root `CMD start-notebook.sh`. No host paths written at build.
- `requirements.docker.txt`: pinned floors (`torch>=2.4,<3`, `numpy>=1.26,<3`,
  `pandas>=2.2,<3`, …) + CUDA-12.1 swap comment + `mamba-ssm` deliberately excluded
  (install in a derived image only). Original `requirements.txt` untouched.
- `docker-compose.yml`: `jupyter` service, `8888:8888`, mounts
  `./data:ro`, `./results:rw`, `./notebooks:rw`, `./configs:ro`; `runtime: nvidia`
  left commented for optional GPU.
- `.dockerignore`: excludes `.git/`, venvs, `__pycache__`, CSVs/parquet, `.pt/.pth`,
  predictions/backtests, logs, so secrets/weights never enter the build context.
- `DOCKER.md`: build/run instructions, kernel selection, in-container script
  equivalents, mount table, Mamba + stale-`results/` warnings.
- `.gitignore`: fixed quoted `"venv/"` → `venv/`, added `*.log`, `.docker/`.

Host isolation: the only host writes at run time are the explicit `./data`
(read-only input) and `./results`/`./notebooks` mounts. No root-level installs.

## 4. Verification (static only, per constraint)

- `git status --short`, `git diff --check` before each commit; file reads to
  confirm valid notebook JSON shape (`nbformat` 4.5, `execution_count: null`).
- Deliberately **not** run: `jupyter nbconvert --execute`, `python run_benchmark.py`,
  `python audit_dataset.py`, `pytest`, `docker build`, `docker compose up`,
  `pip install`. User runs those inside the container when ready (see `DOCKER.md`).

## 5. Version control — commits made (no push) + reuse pattern

Commits on `main` (run `git log --oneline -6` to verify):

1. `fix(audit): restore audit_dataset.py as valid python extracted from embedded notebook JSON`
2. `feat(notebooks): add 02_run_benchmark and 03_dataset_audit mirrors with cleared outputs, keep .py sources`
3. `chore(deps): add pinned requirements.docker.txt for reproducible in-image venv builds`
4. `feat(docker): add pytorch-notebook Dockerfile with in-image venv, compose mounts, dockerignore`
5. `docs(docker): add DOCKER.md and notebooks index, harden gitignore for venv and docker`
6. `docs(migration): add MIGRATION_SUMMARY.md report of conversion and containerization` (this file)

Reuse pattern for future work (from this migration):

```bash
git status --short          # must show only intended paths under this repo
git diff --check            # whitespace/format gate
git add <scoped paths>      # never `git add -A` with data/ outputs present
git commit -m "<type>(<area>): <what + why>"
git log --oneline -3        # confirm; do not push unless asked
```

## 6. Carryovers / safe next steps (not done here)

- Resolve `mamba.allow_fallback` (`configs/models.yaml: false` vs runner `True`);
  install/pin `mamba_ssm` before any Mamba claim.
- Acquire the exact Kaggle CSV into git-ignored `data/`, record the manifest
  (hashes, columns, coverage, license) per `remaining_work.md` P0.
- Regenerate `results/` from the audited pipeline; expand the notebook toward the
  33-section spec; add walk-forward, baselines, block-bootstrap, and seed-robustness
  work per `remaining_work.md` P1–P3.

## 7. Addendum — notebook consolidation (same day, after the migration commits)

- `02_run_benchmark.ipynb` (19 cells) + `03_dataset_audit.ipynb` (3 cells) were
  merged into a single `notebooks/benchmark.ipynb` (23 cells: 1 header + audit +
  pipeline), then the three previous notebooks (`02_`, `03_`, and the original
  `financial_time_series_benchmark.ipynb` with pre-audit conclusions) were removed
  via `git rm`. The originals remain recoverable via git history; stale conclusions
  were intentionally not carried over.
- `notebooks/full_and_tests.ipynb` (31 cells) = `benchmark.ipynb` + byte-identical
  mirrors of `tests/test_data_integrity.py`, `tests/test_shapes.py`, and
  `tests/test_optimizations.py` for inspection only (run via
  `docker compose run --rm jupyter /opt/venv/bin/python -m pytest tests/ -v`).
  Note: the requested name `full_and_tests.ipyb` was saved with the correct
  `.ipynb` extension.
- `notebooks/README.md` and `DOCKER.md` were updated to the new layout. The `.py`
  scripts and `tests/` remain canonical; nothing was executed.

## 8. Addendum — Kaggle import, review fixes, script retirement

- New `§0` cell in both notebooks downloads the primary dataset via the Kaggle API
  (`import kaggle` + `dataset_download_files(.../global-and-indian-financial-markets-dataset-2025)`,
  unzip to `data/`), unexecuted like everything else. `kaggle>=1.6,<2` added to
  `requirements.docker.txt`; `docker-compose.yml` gained a commented `~/.kaggle`
  credentials mount and `DOCKER.md` documents it.
- Review fixes: `SP500Adapter` Close-column check disambiguated for
  MultiIndex vs flat columns; notebook Mamba factory now reads `allow_fallback`
  from `configs/models.yaml` (truthful-by-default `false`) instead of hardcoding
  `True`; audit cells use `os.path.exists` instead of `pd.io.common.file_exists`.
- Retired `run_benchmark.py` and `audit_dataset.py` (fully mirrored by the
  notebooks; recoverable via git history). `src/` and `tests/` stay as the
  canonical `.py` code. `Dockerfile`, `DOCKER.md`, notebook headers, and this
  report were updated; nothing was executed.
