# Final Metrics — final-image-smoke-20261025

Generated from observed notebook unit outputs. No metrics are imputed.

## Aggregate economic and predictive metrics

| Model | Cost bps | Seeds | Folds | Mean Sharpe | SD Sharpe | Mean turnover | Directional accuracy | MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_equal_weight | 5 | 1 | 1 | -1.5327 | nan | 7.2317 | 0.4992 | 0.513771 |
| baseline_long_only | 5 | 1 | 1 | -0.2242 | nan | 11.9110 | 0.4992 | 1.000333 |
| baseline_momentum | 5 | 1 | 1 | -1.1534 | nan | 61.3236 | 0.4672 | 1.000309 |
| baseline_volatility_managed | 5 | 1 | 1 | -0.1398 | nan | 12.7349 | 0.4992 | 0.598755 |
| baseline_zero | 5 | 1 | 1 | 0.0000 | nan | 0.0000 | 0.0416 | 0.000111 |
| dlinear | 5 | 1 | 1 | -0.0341 | nan | 12.1208 | 0.5072 | 0.172036 |
| lstm | 5 | 1 | 1 | -1.8680 | nan | 1.0646 | 0.4672 | 0.003670 |
| mamba | 5 | 1 | 1 | 2.1082 | nan | 0.8023 | 0.5008 | 0.005949 |
| patchtst | 5 | 1 | 1 | -1.1821 | nan | 15.3396 | 0.4544 | 0.628090 |

## Provenance

Inputs and hashes: metadata/data_manifest.json and metadata/run_manifest.json.
Per-unit metrics, regimes, returns and predictions: tables/<model>/seed_<seed>/fold_<fold>/.
Training events and checkpoints: telemetry/ and checkpoints/.

## Interpretation limits

These are historical backtests. Review regime sample sizes, costs, signal timing, uncertainty, data quality, macro release timing, and licensing before drawing conclusions.
Unavailable architectures must never be ranked as trained models.
