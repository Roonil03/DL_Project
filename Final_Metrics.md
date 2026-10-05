# Final Metrics

Generated on 5 October 2026 by executing `notebooks/benchmark.ipynb` in the
Docker Compose environment. The companion `notebooks/full_and_tests.ipynb` was
also executed in inspection mode. These are experimental backtest results, not
evidence of future profitability.

## Data and evaluation scope

- Source: public Kaggle archive `muzaddidafaque/global-and-indian-financial-markets-dataset-2025`.
- Verified daily CSV SHA-256: `e18fe803a85e1045584c8f1609247126114d89c4d8e061214614646d02641def`.
- Raw data: 4,172 rows and 26 columns, covering 1 January 2010 through 30 December 2025.
- Model assets: NIFTY 50, S&P 500, USD/INR, gold, and Brent close prices.
- Aligned price data: 4,171 rows, from 4 January 2010 through 30 December 2025.
- Chronological split: 60% training, 20% validation, 20% test; scaler fitted on training data only.
- Rolling windows: 2,447 training, 802 validation, and 803 test samples with a 20-day lookback.
- Out-of-sample decision dates: 30 November 2022 through 29 December 2025.
- Portfolio settings: 10% target volatility, 3.0 maximum leverage, and 5 bps transaction costs for the primary comparison.
- Seeds: 42, 52, 62, 72, and 82. Primary detailed metrics use seed 42; the seed mean and standard deviation use all five runs.

## Primary economic metrics

### Equal-weight buy-and-hold baseline

- Annualized return: 2.0093%.
- Annualized volatility: 1.1296%.
- Sharpe ratio: 1.7788; block-bootstrap 95% interval `[0.85, 2.77]`.
- Sortino ratio: 2.3905.
- Maximum drawdown: -1.1241%.
- Calmar ratio: 1.7874.
- Win rate: 57.7833%.
- Profit factor: 1.3684.

### DLinear

- Annualized return: 2.5949%.
- Annualized volatility: 2.7712%.
- Sharpe ratio: 0.9364; block-bootstrap 95% interval `[-0.16, 1.82]`.
- Sortino ratio: 1.3923.
- Maximum drawdown: -2.6946%.
- Calmar ratio: 0.9630.
- Win rate: 50.8095%; profit factor: 1.1824.
- Annualized turnover: 163.15; parameters: 1,587; mean training time: 4.2 seconds.
- Five-seed Sharpe: 0.73 ± 0.71.

### LSTM

- Annualized return: 2.8783%.
- Annualized volatility: 2.1573%.
- Sharpe ratio: 1.3342; block-bootstrap 95% interval `[0.48, 2.41]`.
- Sortino ratio: 1.6880.
- Maximum drawdown: -2.8571%.
- Calmar ratio: 1.0074.
- Win rate: 57.7833%; profit factor: 1.3402.
- Annualized turnover: 6.88; parameters: 58,181; mean training time: 3.3 seconds.
- Five-seed Sharpe: 1.34 ± 0.43.

### PatchTST

- Annualized return: 5.7355%.
- Annualized volatility: 4.1110%.
- Sharpe ratio: 1.3952; block-bootstrap 95% interval `[0.24, 2.31]`.
- Sortino ratio: 1.9207.
- Maximum drawdown: -3.4279%.
- Calmar ratio: 1.6732.
- Win rate: 56.0399%; profit factor: 1.2650.
- Annualized turnover: 118.74; parameters: 111,173; mean training time: 4.4 seconds.
- Five-seed Sharpe: 0.45 ± 0.49.

## Regime Sharpe ratios

- DLinear: HighVol_Bear 1.8025; HighVol_Bull -0.8736; LowVol_Bear 5.6544; LowVol_Bull 0.9637.
- LSTM: HighVol_Bear 1.8085; HighVol_Bull 2.8691; LowVol_Bear -0.6448; LowVol_Bull 1.3021.
- PatchTST: HighVol_Bear 3.5688; HighVol_Bull 1.8562; LowVol_Bear 1.6521; LowVol_Bull 1.1204.

## Transaction-cost sensitivity

- DLinear Sharpe at 0/1/5/10/20 bps: 3.9080 / 3.3159 / 0.9364 / -1.9534 / -6.9409.
- LSTM Sharpe at 0/1/5/10/20 bps: 1.4944 / 1.4623 / 1.3342 / 1.1741 / 0.8541.
- PatchTST Sharpe at 0/1/5/10/20 bps: 2.8467 / 2.5570 / 1.3952 / -0.0487 / -2.8192.

## Pairwise block-bootstrap tests

- DLinear versus LSTM: Sharpe difference -0.398; p-value 0.6820.
- DLinear versus PatchTST: Sharpe difference -0.459; p-value 0.7690.
- LSTM versus PatchTST: Sharpe difference -0.061; p-value 0.5630.

None of these pairwise differences is statistically significant at the 5%
level under the implemented test.

## Important limitation

`mamba_ssm` is not available in the CPU Docker image. Mamba was therefore
excluded from training and reporting. No LSTM fallback was labelled or reported
as Mamba. A four-architecture comparison requires a separate compatible
CUDA/Mamba image and a fresh run of the same notebook.
