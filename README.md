# Regime-Robust Financial Deep Learning Benchmark

This project benchmarks four neural architectures for financial time-series
signal generation and portfolio construction: DLinear, LSTM, Mamba, and
PatchTST. The notebooks provide an audited, chronological, leakage-safe
workflow that evaluates both predictive quality and risk-adjusted trading
performance across market regimes.

## Research gap

Many financial deep-learning comparisons report one aggregate forecasting or
Sharpe score. That can hide failures during high-volatility, bearish, or crisis
periods. This project addresses the gap by comparing architectures on an
independent multi-market dataset with regime-specific risk, trading-cost,
uncertainty, and compute analysis rather than relying on a single aggregate
ranking.

## Models and contributors

| Model | Worked by |
|---|---|
| DLinear | [Lakshmihollat](https://github.com/Lakshmihollat) |
| LSTM | [Roonil03](https://github.com/Roonil03) |
| Mamba | [Aryan Soni](https://github.com/AryanSoni1609) |
| PatchTST | [tanay](https://github.com/electro-glitch) |

## Dataset

The benchmark uses the **Global and Indian Financial Markets Dataset 2025**
from [Kaggle](https://www.kaggle.com/datasets/muzaddidafaque/global-and-indian-financial-markets-dataset-2025).
The audited input directory is:

```text
data/kaggle_global_indian_markets/global-india-markets-macro/
```

It contains the daily market file, the monthly macro file, and the source/code
mapping used to verify provenance. The runner records file names, SHA-256
hashes, schema, data types, date coverage, frequency, missing values,
duplicate timestamps, asset identifiers, source fields, and the declared
dataset license in `metadata/data_manifest.json`. Raw data is not committed to
Git and is excluded from result archives unless explicitly requested.

## Evaluation metrics

Metrics are computed on untouched chronological test data after validation-only
configuration selection. Unless stated otherwise, model statistics are
aggregated over five seeds (`42, 52, 62, 72, 82`) and reported with the mean,
standard deviation, and paired block-bootstrap confidence intervals where
applicable.

| Dimension | Metrics and statistics |
|---|---|
| Economic performance | Annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and downside deviation. |
| Predictive quality | Mean squared error (MSE), directional accuracy, and signal/return correlation. |
| Trading behaviour | Turnover, average and maximum leverage, transaction-cost attribution, and cost sensitivity in basis points. |
| Regime robustness | Sharpe, return, volatility, drawdown, sample count, and confidence interval for each volatility/trend regime; stress-regime degradation and rank stability. |
| Compute and reliability | Parameter count, epoch duration, throughput, GPU memory/utilization, retry count, checkpoint/resume outcome, and failure timeline. |

### Observed benchmark snapshot

The following values come from the completed `full-validation` run of
11 November 2026. They represent the first recorded walk-forward fold. They
use a 5 bps transaction-cost assumption, one evaluation fold, and five seeds;
the Sharpe standard deviation is across seeds.

| Model | Mean Sharpe | SD Sharpe | Mean turnover | Directional accuracy | MSE |
|---|---:|---:|---:|---:|---:|
| DLinear | -0.4568 | 0.9983 | 28.0627 | 0.4989 | 0.130059 |
| LSTM | -0.6172 | 1.1431 | 8.5427 | 0.4826 | 0.240953 |
| Mamba | 0.5511 | 0.6568 | 76.6328 | 0.4774 | 0.262322 |
| PatchTST | 0.3800 | 0.7999 | 43.1207 | 0.5098 | 0.370780 |

These are historical backtest statistics, not investment advice. Interpret
them together with regime sample sizes, confidence intervals, costs, signal
timing, data quality, and the reproducibility manifest. The full generated
report is available in [`Final_Metrics.md`](Final_Metrics.md).
See [`docs/analysis-2026-11-11.md`](docs/analysis-2026-11-11.md) for the
actual evaluation dates, provenance checks, cost sensitivity, and archive
limitations. The date-split settings in the config do not describe fold 0.

## Information documents

The flow is: audited Kaggle files → past-only features → chronological
train/validation/test folds → one bounded signal per asset → one portfolio
sizing and fee pass → matched-date metrics and regime reports. DLinear is a
DLinear-inspired signal model, and the PatchTST implementation is a compact
patch Transformer rather than a faithful reproduction. The optional
`ahat_fin` model is inspired by dual-axis attention research; its results are
not part of the published four-model benchmark.

For a synthetic CPU development environment, install pinned packages from
`requirements.cpu.txt` in a local virtual environment and run only targeted
tests. The full audited benchmark uses the pinned container. Reproduce any
claim with the data hash, Git commit, resolved configuration, seeds, folds,
fees, dependency lock, and complete run archive.

- [RunningNotebooks.md](RunningNotebooks.md) — Docker-first instructions for
  building the CUDA image, running or resuming notebooks, estimating runtime,
  validating outputs, and creating the portable ZIP archive.
- [DOCKER.md](DOCKER.md) — pinned CUDA image, dependency, and GPU preflight
  details.
- [Final_Metrics.md](Final_Metrics.md) — metrics from the completed benchmark
  run.
