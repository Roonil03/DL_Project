# Final Metrics — india-study

## Aggregate economic and predictive metrics

| Model | Cost bps | Seeds | Folds | Mean Sharpe | SD Sharpe | Mean turnover | Directional accuracy | MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_equal_weight | 5 | 5 | 2 | 1.7026 | 0.4997 | 1.0224 | 0.5475 | 7.304416 |
| baseline_long_only | 5 | 5 | 2 | 2.3317 | 0.1692 | 18.3798 | 0.3525 | 0.631804 |
| baseline_momentum | 5 | 5 | 2 | 1.5186 | 0.0647 | 35.5642 | 0.5240 | 0.998529 |
| baseline_volatility_managed | 5 | 5 | 2 | 1.7957 | 0.5191 | 7.1486 | 0.5475 | 0.997977 |
| baseline_zero | 5 | 5 | 2 | 0.0000 | 0.0000 | 0.0000 | 0.0007 | 0.000300 |
| dlinear | 5 | 5 | 2 | 0.3801 | 1.2975 | 22.9383 | 0.5196 | 0.163603 |
| lstm | 5 | 5 | 2 | -0.2994 | 1.1678 | 5.7618 | 0.5045 | 0.233724 |
| mamba | 5 | 5 | 2 | -0.1809 | 1.0148 | 14.6974 | 0.5210 | 0.323786 |
| patchtst | 5 | 5 | 2 | -0.6360 | 0.6702 | 15.7267 | 0.5100 | 0.678600 |

## Provenance

Inputs and hashes: metadata/data_manifest.json and metadata/run_manifest.json.
Per-unit metrics, regimes, returns and predictions: tables/<model>/seed_<seed>/fold_<fold>/.
Training events and checkpoints: telemetry/ and checkpoints/.

## Interpretation limits

These are historical backtests. Review regime sample sizes, costs, signal timing, uncertainty, data quality, macro release timing, and licensing before drawing conclusions.
Unavailable architectures must never be ranked as trained models.
