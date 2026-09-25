# Project state already completed

Audit date: 25 September 2026

Reference reviewed: `C:\Users\tanrc\Downloads\dl_synopsis.pdf` (4 pages).

This file records what was present before the current audit and what can be
reused. "Implemented" means code exists; it does not automatically mean that
the associated experimental result is scientifically verified.

## Synopsis coverage already present

- The repository uses the synopsis topic: benchmarking DLinear, LSTM, Mamba,
  and PatchTST for financial time-series signal generation under volatility
  regimes.
- A PyTorch project structure exists with separate modules for data handling,
  models, training, portfolio construction, backtesting, evaluation, regimes,
  and visualization.
- DLinear is implemented with temporal trend/remainder decomposition and a
  bounded `tanh` signal head.
- LSTM is implemented with configurable hidden size, layers, and dropout.
- PatchTST is implemented as temporal patches followed by a Transformer
  encoder and bounded signal head.
- A Mamba wrapper exists and supports the external `mamba_ssm` package. The
  pre-audit implementation also had an LSTM fallback; its reporting risk is
  documented in `remaining_work.md`.
- Rolling-window sample creation and next-period return alignment exist.
- Chronological 60/20/20 splitting and a train-only scaler exist.
- A differentiable negative annualized Sharpe loss exists, including an
  optional directional-MSE blend.
- Volatility targeting, leverage caps, transaction costs, equal-capital
  portfolio aggregation, and a vectorized backtester exist.
- Aggregate economic metrics, regime metrics, bootstrap utilities, and ten
  report-oriented plotting functions exist.
- Configuration files exist for data, model, training, backtest, and regime
  parameters.
- A Python runner and Jupyter notebook both exist.
- A `.gitignore`, `requirements.txt`, `environment.yml`, README, and initial
  model-shape test exist.
- Prior tables and figures exist under `results/`; they have been preserved.

## Repository history already present

- Git history contains incremental commits for the financial benchmark
  scaffold, core pipeline, runner, results, and notebook.
- The working tree was clean at the start of this audit.
- No primary dataset file is tracked by Git, which is appropriate for size and
  licensing but means the exact dataset used cannot currently be reproduced
  from this checkout alone.

## Important qualification

The notebook has 19 cells (9 code cells), all code cells carry execution
counts, and it embeds conclusions based on the pre-audit results. The master
specification asks for a much more detailed 33-section notebook. Therefore the
notebook is a useful scaffold, not yet the final reproducible research record.
