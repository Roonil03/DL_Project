# Methodology audit and verification

The methodology audit merged at commit `02be460`. Its changes apply to future
runs; they do not retroactively change the historical `full-validation` run.

The audit aligned training and validation with volatility-scaled portfolio
positions, corrected and distinguished strategy baselines, isolated
checkpoints by fold, froze benchmark regime thresholds using training data,
and expanded bounded model-specific tuning candidates. It also added guards
against missing prices and unverified mixed-market inputs. These are code
changes, not evidence of improved investment performance.

The mixed-market restriction remains: the source close timestamps and the
time at which each market's data can be used for execution have not been
verified. Calendar-date alignment alone does not establish that information
was available at the decision time. The default five-asset selection is
therefore rejected. The notebooks support the Kenneth R. French 10 Industry
Portfolios daily value-weighted US return series as a separate benchmark; its official
download was unavailable during this review, its raw data is absent, and its
copyright is declared and redistribution terms remain unverified. Its configured
folds test 2008 and 2020; no benchmark on it has been completed.

For the audit merge, Python compilation, notebook syntax parsing, and diff
checks passed. Runtime tests could not be run in the local environment because
PyTorch was unavailable offline and the Docker daemon was down. These checks do
not establish end-to-end training, GPU behavior, or research validity.

The earlier [run review](analysis-2026-11-11.md) records the actual 2013 test
dates, provenance and archive limitations. The still earlier
[results archive](../results_archive/README.md) documents known leakage and
model-labeling problems and must not be used as final evidence.
