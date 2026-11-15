# Benchmarking Deep Learning Architectures for Financial Time-Series Prediction and Risk Optimization under Market Regimes

This project benchmarks four neural architectures for financial time-series
signal generation and portfolio construction: DLinear, LSTM, Mamba, and
PatchTST. This group project compares predictive quality and risk-adjusted
trading performance across market regimes: periods distinguished by volatility
and trend. The workflow uses chronological evaluation and safeguards that must
be checked for each dataset and execution setup.

## Research gap

Many financial deep-learning comparisons report one aggregate forecasting or
Sharpe score. That can hide failures during high-volatility, bearish, or crisis
periods. This project studies architecture comparisons with regime-specific
risk, trading-cost, uncertainty, and compute analysis rather than relying on
a single aggregate ranking. The primary study focuses on Indian sectors; a
separate US sector comparison and the historical multi-market run have
different datasets and limitations.

## Models and contributors

The existing model credits for this group project are:

| Model | Worked by |
|---|---|
| DLinear | [Lakshmihollat](https://github.com/Lakshmihollat) |
| LSTM | [Roonil03](https://github.com/Roonil03) |
| Mamba | [Aryan Soni](https://github.com/AryanSoni1609) |
| PatchTST | [tanay](https://github.com/electro-glitch) |

## Dataset and current status

The original benchmark uses the Kaggle data described below. In that dataset
mode the current runner rejects selected price series with missing values and selections of multiple
market series because source close times and execution availability across
markets remain unverified. Its default five-asset selection therefore cannot
currently complete a new run.

The planned primary study uses the [Indian market historical dataset](https://github.com/aswinv90/indian-market-historical-data)
at pinned commit `1f4aff7c3e53a9f2bfb6a167b0d80f4d0c8f8bf9`
for NIFTY BANK, IT and PHARMA. Its common daily panel begins in 2011 and
has 3,675 complete sessions from 31 January 2011 through 31 December 2025,
including the 2020 crisis. Returns are derived from daily Close prices and exclude
dividends. The source permits noncommercial research and does not permit raw
redistribution. No new benchmark results are available.

An optional separate US comparison uses [FRED's Nasdaq US Large/Mid Cap sector
total-return index panel](https://fred.stlouisfed.org/graph/fredgraph.csv?id=NASDAQNQUSB10LMT,NASDAQNQUSB20LMT,NASDAQNQUSB30LMT,NASDAQNQUSB50LMT,NASDAQNQUSB60LMT):
technology, health, financials, industrials and
energy. The reviewed local snapshot spans 30 March 2001 to 13 November 2026,
with 6,518 complete trading rows and 168 all-missing holiday rows shared by
all series, with no partially missing rows. It contains 253 complete rows in
2008 and 257 in 2020. Returns are derived from percentage changes in index
levels. These are indices, not tradable ETFs. Nasdaq
claims copyright and requires pre-approval for redistribution. No sector
benchmark has been run or results reported.

These are separate studies: the US panel is not a worldwide market sample.
Keep Indian, US, and legacy mixed-market results separate rather than pooling
their rankings across datasets.

The historical benchmark used the **Global and Indian Financial Markets Dataset 2025**
from [Kaggle](https://www.kaggle.com/datasets/muzaddidafaque/global-and-indian-financial-markets-dataset-2025).
The audited input directory is:

```text
data/kaggle_global_indian_markets/global-india-markets-macro/
```

It contains the daily market file, the monthly macro file, and the source/code
mapping used to verify provenance. The runner records file names, SHA-256
hashes, schema, data types, date coverage, frequency, missing values,
duplicate timestamps, asset identifiers, source fields, and license information
in `metadata/data_manifest.json`. Verify the source license independently before
using or redistributing the data. Raw data is not committed to
Git and is excluded from result archives.

## Evaluation metrics

The evaluation is designed to use chronological test data after selecting
configurations on validation data. Sharpe ratio measures return relative to
volatility; mean squared error (MSE) measures prediction error. Transaction
costs are expressed in basis points (one basis point is 0.01%). Each run must
report its own seeds, dates, sample sizes, and uncertainty.

| Dimension | Metrics and statistics |
|---|---|
| Economic performance | Annualized return, annualized volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and downside deviation. |
| Predictive quality | Mean squared error (MSE), directional accuracy, and signal/return correlation. |
| Trading behaviour | Turnover, average and maximum leverage, transaction-cost attribution, and cost sensitivity in basis points. |
| Regime robustness | Sharpe, return, volatility, drawdown, sample count, and confidence interval for each volatility/trend regime; stress-regime degradation and rank stability. |
| Compute and reliability | Parameter count, epoch duration, throughput, GPU memory/utilization, retry count, checkpoint/resume outcome, and failure timeline. |

### Historical benchmark snapshot

The following values come from the completed `full-validation` run of
11 November 2026, before the later methodology audit. They represent one
walk-forward fold, with 125 decision dates from 18 February to 9 August 2013. They
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

The intended flow is: reviewed data → past-only features → chronological
train/validation/test folds → one bounded signal per asset → one portfolio
sizing and fee pass → matched-date metrics and regime reports. DLinear is a
DLinear-inspired signal model, and the PatchTST implementation is a compact
patch Transformer rather than a faithful reproduction. The optional
`ahat_fin` model is inspired by dual-axis attention research; its results are
not part of the published four-model benchmark.

For a synthetic CPU development environment, install pinned packages from
`requirements.cpu.txt` in a local virtual environment and run only targeted
tests. The benchmark runner uses the pinned container. Reproduce any
claim with the data hash, Git commit, resolved configuration, seeds, folds,
fees, dependency lock, and complete run archive.

- [RunningNotebooks.md](RunningNotebooks.md) — Docker-first instructions for
  building the CUDA image, running or resuming notebooks, estimating runtime,
  validating outputs, and creating the portable ZIP archive.
- [DOCKER.md](DOCKER.md) — pinned CUDA image, dependency, and GPU preflight
  details.
- [Final_Metrics.md](Final_Metrics.md) — metrics from the completed benchmark
  run.
- [docs/audit-summary.md](docs/audit-summary.md) — methodology changes,
  remaining market-timing restriction, and verification limits.
