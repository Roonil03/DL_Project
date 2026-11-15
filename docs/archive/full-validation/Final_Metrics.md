# Final Metrics — full-validation

This local user run covers walk-forward fold 0 only. The recorded test
decision interval is in 2013, despite calendar split values of 2018/2020 in
the resolved config. Read `docs/analysis-2026-11-11.md` before comparing or
citing these scores.

Generated from observed notebook unit outputs. No metrics are imputed.

## Aggregate economic and predictive metrics

| Model | Cost bps | Seeds | Folds | Mean Sharpe | SD Sharpe | Mean turnover | Directional accuracy | MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_equal_weight | 5 | 5 | 1 | -1.5327 | 0.0000 | 7.2317 | 0.4992 | 0.513771 |
| baseline_long_only | 5 | 5 | 1 | -0.2242 | 0.0000 | 11.9110 | 0.4992 | 1.000333 |
| baseline_momentum | 5 | 5 | 1 | -1.1534 | 0.0000 | 61.3236 | 0.4672 | 1.000309 |
| baseline_volatility_managed | 5 | 5 | 1 | -0.1398 | 0.0000 | 12.7349 | 0.4992 | 0.598755 |
| baseline_zero | 5 | 5 | 1 | 0.0000 | 0.0000 | 0.0000 | 0.0416 | 0.000111 |
| dlinear | 5 | 5 | 1 | -0.4568 | 0.9983 | 28.0627 | 0.4989 | 0.130059 |
| lstm | 5 | 5 | 1 | -0.6172 | 1.1431 | 8.5427 | 0.4826 | 0.240953 |
| mamba | 5 | 5 | 1 | 0.5511 | 0.6568 | 76.6328 | 0.4774 | 0.262322 |
| patchtst | 5 | 5 | 1 | 0.3800 | 0.7999 | 43.1207 | 0.5098 | 0.370780 |

## Provenance

The provenance files are in the local, gitignored
`outputs/full-validation/metadata/` directory. Per-unit tables and
predictions are under `outputs/full-validation/tables/<model>/seed_<seed>/fold_<fold>/`;
telemetry and checkpoints are in sibling directories. They are not included
in this Git checkout or PR.

## Interpretation limits

These are historical backtests. Review regime sample sizes, costs, signal timing, uncertainty, data quality, macro release timing, and licensing before drawing conclusions.
Unavailable architectures must never be ranked as trained models.
