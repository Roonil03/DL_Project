# Remaining work and recommended order

Audit date: 25 September 2026

The synopsis work plan places the project at the interim-report and preliminary
verification milestone. The next formal milestone is the full multi-market
evaluation, hyperparameter tuning, and regime-specific backtest. Do not start
expensive training until the blocking items below are complete.

## Priority 0 - evidence and data integrity

1. Obtain the exact Global and Indian Financial Markets Dataset files from the
   cited Kaggle source and place them under `data/` without committing them.
2. Record a data manifest containing file names, hashes, columns, data types,
   date coverage, frequency, missing values, duplicate timestamps, asset
   identifiers, source fields, and the verified license.
3. Replace or explicitly separate the current Yahoo Finance download fallback.
   It is not proof that the named Kaggle dataset was used and it omits parts of
   the synopsis universe, including India 10-year yield and macro fields.
4. Regenerate every table and figure after the audited pipeline is used. The
   existing `results/` artifacts predate leakage and model-label safeguards and
   are preserved only for traceability.

## Priority 1 - model and experiment validity

1. Install and pin a compatible `mamba_ssm` version, then verify that the model
   reports `implementation_name == "Mamba"`. Never publish fallback output as
   Mamba.
2. Decide and document the Sharpe training unit. Chronological mini-batches are
   safer than shuffled batches but still estimate Sharpe locally; full-sequence
   epochs or contiguous large blocks should be compared.
3. Implement expanding-window and rolling-window walk-forward evaluation. The
   current runner performs one chronological 60/20/20 holdout only.
4. Add validation-only hyperparameter selection with a small documented search
   space and keep the test split untouched until the final comparison.
5. Add required baselines beyond equal-weight buy-and-hold: zero-signal,
   long-only, simple trend/momentum, and volatility-managed baselines.

## Priority 2 - analysis required by the specification

1. Add predictive MSE and signal/return correlation to the existing directional
   accuracy so predictive and economic performance can be compared properly.
2. Add crisis-window analysis, architecture-rank stability, explicit sample
   counts by regime, and confidence intervals for regime metrics.
3. Replace the pairwise IID resampling in
   `src/evaluation/statistical_tests.py` with a paired block bootstrap and
   document whether each hypothesis is one- or two-sided.
4. Review the regime robustness formula. The current minimum-to-maximum Sharpe
   ratio can become difficult to interpret when signs differ.
5. Add architecture-specific error and failure-mode analysis, especially the
   synopsis questions about memory limits, volatility shifts, overfitting, and
   computational efficiency.

## Priority 3 - notebook, documentation, and tests

1. Rebuild the notebook to follow the 33-section order in `instructions.txt`.
   Clear stale outputs and keep expensive training behind an explicit flag.
2. Remove the current notebook's result conclusions until audited experiments
   have been rerun.
3. Expand tests for split chronology, train-only scaling, no future leakage,
   transaction costs, leverage caps, missing assets, regime construction, loss
   stability, and statistical tests.
4. Pin dependency versions and document CPU/GPU and Mamba installation paths.
5. Add a reproducibility manifest for seeds, configuration snapshots, package
   versions, source commit, data hashes, and output directory.

## Immediate next action

The safest next task is dataset acquisition plus a non-training data audit. It
unblocks every later experiment without risking misleading or expensive model
runs.
