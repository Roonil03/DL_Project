# Methodology audit and verification
Its changes apply to future
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
therefore rejected. The completed primary `india-study` benchmark uses daily Close prices for
NIFTY BANK, IT and PHARMA from a pinned third-party Indian dataset. The common
panel starts in 2011, covers the 2020 crisis, and excludes dividends from
derived returns. A separate planned comparison uses the FRED Nasdaq US
Large/Mid Cap sector total-return index panel, with 2008 and 2020 test periods.
The reviewed local snapshot covers 30 March 2001 to 13 November 2026; it has
6,518 complete trading rows and 168 all-missing holiday rows shared across
series, with no partial missingness. Nasdaq claims
copyright and requires pre-approval for redistribution. These indices are not
tradable ETFs; no US sector benchmark has been completed.

For the audit merge, Python compilation, notebook syntax parsing, and diff
checks passed. Runtime tests could not be run in the local environment because
PyTorch was unavailable offline and the Docker daemon was down. These checks do
not establish end-to-end training, GPU behavior, or research validity.

The completed India run on 15 November records 40 trained units, 2020 and 2023
test years, 58 tests passed and 4 skipped, and a fully hash-verified ZIP. Read
the [current review](analysis-2026-11-15.md) for recovery history and research
limits. The neural models did not outperform the long/cash trend baseline;
methodology corrections are not evidence of investment improvement.

The earlier [run review](archive/full-validation/analysis-2026-11-11.md) records
the actual 2013 test dates, provenance and archive limitations. The still earlier
[results archive](../results_archive/README.md) documents known leakage and
model-labeling problems and must not be used as final evidence.
