# Results provenance

The existing tables and figures were generated before the September 2026
methodology audit. They are preserved to avoid destroying prior work, but they
must not be used as final report evidence yet.

Reasons regeneration is required:

- the earlier feature pipeline backfilled rolling volatility into the warm-up
  period, which introduced look-ahead leakage;
- the earlier runner shuffled samples while optimizing a batch Sharpe loss;
- when `mamba_ssm` was unavailable, the earlier `MambaSignalModel` silently used
  LSTM layers while downstream output still carried the label `Mamba`;
- the repository does not contain the primary Kaggle dataset or a data-quality
  manifest proving the exact files, fields, coverage, missingness, duplicates,
  or license used for these outputs.

Do not delete these artifacts. Regenerate them into a new timestamped results
directory after completing the checklist in `../remaining_work.md`, then compare
the audited outputs with this preserved baseline.
