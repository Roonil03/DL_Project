# Work completed in the September 2026 audit

Date: 25 September 2026

## Review performed

- Read all four pages of the synopsis and visually checked the rendered pages.
- Audited the repository structure, Git history, source modules, configuration,
  notebook, tests, and existing result tables.
- Compared the implementation against both the synopsis and the more detailed
  master specification in `instructions.txt`.
- Created `already_done.md` and `remaining_work.md` as the persistent project
  memory requested for future work.

## Safe code changes made

- Removed future-data leakage from rolling volatility features by eliminating
  backward filling and dropping only the initial feature warm-up rows.
- Kept feature, return, volatility, and feature-dictionary indexes aligned
  after warm-up removal.
- Preserved chronological training order by disabling DataLoader shuffling for
  the batch Sharpe objective.
- Prevented an unavailable Mamba dependency from silently producing an LSTM
  result labelled as Mamba. The fallback now requires explicit opt-in and is
  marked debug-only.
- Added `allow_fallback: false` to the Mamba configuration.
- Added a dedicated PatchTST `d_model` configuration and corrected the runner
  so it no longer reads the LSTM hidden size for that setting.
- Fixed `BaseTrainer.predict()` so an empty loader can return a NumPy array
  instead of raising because NumPy was not imported.

## Tests and documentation added

- Added tests proving rolling volatility does not backfill its warm-up period.
- Added tests proving all feature outputs remain aligned after warm-up removal.
- Added a rolling-window target/date alignment test.
- Updated the model-shape test to make its Mamba fallback explicit and limited
  to interface validation.
- Added a results provenance warning without deleting or overwriting any prior
  figures or tables.
- Updated the README with the audited status and links to the three project
  memory files.

## Verification

- Python source compilation completed successfully.
- Test result: `5 passed` using the available Python 3.10 environment with
  PyTorch 2.7.1 CPU.
- No expensive training, external downloads, or result regeneration was run.
- Existing prior results were preserved unchanged.

## Result interpretation

The old result files are not considered final evidence. They were generated
before the leakage fix, chronological-batch safeguard, and truthful Mamba
dependency enforcement. Follow `remaining_work.md` before using any model
ranking or conclusion in the interim/final report.
