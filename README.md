# Regime Robust Finance Benchmark

*Benchmarking Deep Learning Architectures for Financial Time-Series Prediction and Risk Optimization under Market Regimes*

## Abstract
This study performs an independent, controlled benchmark of four architectural families for financial time-series signal generation on a multi-market dataset, with particular emphasis on the stability of risk-adjusted performance across market regimes. Rather than evaluating architectures only by prediction error or aggregate Sharpe ratio, the study examines regime-specific Sharpe, downside risk, drawdown, turnover, computational cost, and predictive-economic alignment.

## Introduction — What the Code Does

The codebase implements an end-to-end, leakage-safe benchmark that turns market
history into volatility-scaled trading positions and scores architectures on
risk-adjusted, regime-conditioned performance rather than raw prediction error:

- **Data audit**: `notebooks/10_data_audit.ipynb` verifies the exact Kaggle
  daily and monthly source files, source mapping, license, schemas, dates,
  missingness, duplicates, and SHA-256 hashes before training.
- **Part A — Dataset audit**: integrity check (records, columns, dtypes, missing
  values, duplicates, SHA-256) before anything is trusted.
- **Benchmark**: `notebooks/20_benchmark.ipynb` follows the 33-section research
  protocol: load configs → ingest multi-asset prices →
  engineer backward-looking features → classify Volatility × Trend regimes →
  chronological 60/20/20 split with train-only scaling → rolling lookback windows
  → instantiate DLinear / LSTM / Mamba / PatchTST under one common
  signal interface → Sharpe-loss training across 5 seeds → volatility-targeted
  backtest with transaction-cost sensitivity → regime-stratified metrics,
  bootstrap statistics, and publication figures under `results/`.
- **`src/` library**: importable modules for data, models, Sharpe loss,
  portfolio construction, regimes, training, evaluation, backtesting, and
  visualization — the notebooks orchestrate, they don't reimplement.
- **Notebook suite**: `00_core` provides shared runtime helpers, `30_validation`
  runs integrity checks, and `40_report` generates report tables and figures.
- **`tests/` suite**: shape contracts, leakage/alignment guards, Mamba identity,
  walk-forward splits, telemetry, and checkpoint correctness.

## Research Gap
Existing recent benchmarks, including Saly-Kaufmann et al. (2026), are based on particular asset universes and backtesting protocols. It remains valuable to determine whether conclusions about architectural robustness generalize to an independent publicly available multi-market dataset. Furthermore, aggregate Sharpe ratio can conceal differences in performance across market regimes (low/high volatility, bull/bear, crisis).

## Research Questions
1. Which architecture has the best aggregate Sharpe?
2. Which architecture has the best Sharpe in high-volatility/stress conditions?
3. Which architecture has the smallest degradation from normal to stress regime?
4. Which architecture delivers the best Sharpe relative to parameter count/training time?
5. Does the model with the best forecasting score also have the highest Sharpe?

## Objectives
- Build a generic PyTorch framework to train, backtest, and evaluate deep learning architectures under a Sharpe optimization objective.
- Implement DLinear, LSTM, PatchTST, and Mamba models.
- Apply volatility targeting and transaction cost modelling.
- Detect market regimes and evaluate robustness.

## Dataset
- **Primary Dataset**: Global & Indian Financial Markets Dataset 2025 (Kaggle)
- **Dataset Caveats**: Financial market data is prone to look-ahead bias, non-stationarity, and survivorship bias (especially in indices). The codebase attempts to strictly prevent chronological leakage.

## Models
1. **DLinear**: Simple learned temporal mapping baseline.
2. **LSTM**: Recurrent temporal context baseline.
3. **Mamba (State Space)**: Selective state space representation.
4. **PatchTST**: Attention-based temporal patching.

## Experimental Protocol
- 60% Train, 20% Validation, 20% Test (Chronological splits).
- Inputs are historical lookback windows (e.g., 20 days).
- Models predict a bounded directional signal \([-1, 1]\).
- Positions scaled by ex-ante volatility (Target volatility: 10%).
- Training objective: Negative Annualized Sharpe Ratio.

## Metrics
- **Economic**: Annualized Sharpe, Sortino, Max Drawdown, Annualized Return.
- **Trading**: Turnover, average leverage.
- **Predictive**: Directional accuracy, MSE, Correlation.

## Regime Analysis
- Volatility × Trend Regime classification based purely on available historical data at time `t`.

## Notebook Instructions — How to Run

### Option A — Docker (recommended, host untouched)

All dependencies live in the image's uv-created `/opt/venv`; no Conda or
Mamba environment manager is used. See `DOCKER.md` for the pinned matrix.

```bash
./run_all_models.sh
```

The runner builds and starts Compose, performs CUDA and genuine-Mamba
preflight, audits the data, executes each model/seed/fold unit, validates the
run, writes `Final_Metrics.md`, and creates `artifacts/<run_id>.zip` only after
all required units succeed. Set `DEVICE=cpu` explicitly for a CPU smoke run.
Useful overrides include `RUN_ID`, `MODELS`, `SEEDS`, `FOLDS`, `EPOCHS`,
`RESUME_MODE`, `MAX_RETRIES`, `ESTIMATE_ONLY`, and `INCLUDE_DATA`.

### Option B — Local virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name fin-dl-benchmark --display-name "Financial DL Benchmark"
```

Local environments are intended for lightweight development only; audited
benchmark results must come from the pinned container.

## Reproducibility
- Chronological train/val/test constraints are enforced.
- Configurable settings via `configs/`.
- Tested across multiple random seeds `[42, 52, 62, 72, 82]`.

## Result Status

Only an archive produced by a successful `run_all_models.sh` execution is a
publishable result. The runner rejects missing CUDA for `DEVICE=gpu` and rejects
any Mamba implementation whose identity is not exactly `Mamba`.

## Academic Integrity
*Note: External AI assistance may have been used during the development of this codebase, subject to the university/course policy.*

## References
1. Saly-Kaufmann, A., Wood, K., Peter-Calliess, J., & Zohren, S. (2026). Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance. arXiv:2603.01820.
