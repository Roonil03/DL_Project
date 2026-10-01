#!/usr/bin/env python
"""
Master Execution Script for Financial DL Benchmark
Benchmarking Deep Learning Architectures for Financial Time-Series Prediction and Risk Optimization under Market Regimes
"""

import os
import sys
import yaml
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure root path in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data.adapters import GlobalIndianAdapter
from src.data.features import build_features
from src.data.splits import chronological_split
from src.data.scaling import TimeSeriesScaler
from src.data.windows import create_rolling_windows
from src.regimes.rules import detect_regimes
from src.models.dlinear import DLinear
from src.models.lstm import LSTMSignalModel
from src.models.patchtst import PatchTSTSignalModel
from src.models.mamba_model import MambaSignalModel
from src.losses.sharpe_loss import NegativeSharpeLoss
from src.training.trainer import BaseTrainer
from src.backtest.backtester import Backtester
from src.evaluation.metrics import compute_all_metrics, calculate_sharpe_ratio
from src.evaluation.regime_metrics import evaluate_by_regime, calculate_regime_robustness
from src.evaluation.statistical_tests import block_bootstrap_sharpe, paired_sharpe_difference_test
from src.visualization.performance import (
    plot_market_overview,
    plot_regime_classification,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_sharpe_by_regime,
    plot_turnover_vs_sharpe,
    plot_transaction_cost_sensitivity,
    plot_predictive_vs_economic,
    plot_complexity_tradeoff,
    plot_seed_robustness
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_yaml(filepath: str) -> dict:
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)

def run_experiment():
    print("=" * 80)
    print(" FINANCIAL DEEP LEARNING BENCHMARK: EXECUTION PIPELINE")
    print("=" * 80)
    
    # 1. Load Configurations
    config_data = load_yaml('configs/data.yaml')
    config_models = load_yaml('configs/models.yaml')
    config_train = load_yaml('configs/training.yaml')
    config_backtest = load_yaml('configs/backtest.yaml')
    config_regime = load_yaml('configs/regime.yaml')
    
    lookback = config_data.get('lookback', 20)
    seeds = config_train.get('seeds', [42, 52, 62, 72, 82])
    tc_bps_list = config_backtest.get('transaction_costs_bps', [0, 1, 5, 10, 20])
    target_vol = config_backtest.get('target_volatility', 0.10)
    max_lev = config_backtest.get('max_leverage', 3.0)
    
    for output_dir in ('results/figures', 'results/tables', 'results/predictions'):
        os.makedirs(output_dir, exist_ok=True)
    
    # 2. Data Ingestion
    print("\n[1/7] Loading & Preparing Full Multi-Asset Dataset (2010–2025)...")
    adapter = GlobalIndianAdapter(data_path="data/global_indian_markets.csv")
    prices = adapter.load()
    print(f"-> Successfully loaded dataset: {prices.shape[0]} daily records across {prices.shape[1]} assets: {list(prices.columns)}")
    print(f"-> Date Range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    
    # Generate Overview Plot
    plot_market_overview(prices, save_path="results/figures/01_market_data_overview.png")
    
    # 3. Feature Engineering & Regime Detection
    print("\n[2/7] Engineering Features & Classifying Market Regimes...")
    feature_df, returns_df, vol_df, _ = build_features(prices)
    regimes = detect_regimes(
        prices, 
        vol_window=config_regime.get('volatility_window', 60), 
        trend_window=config_regime.get('trend_window', 120)
    )
    plot_regime_classification(prices, regimes, save_path="results/figures/02_regime_classification.png")
    
    regime_counts = regimes.value_counts()
    print(f"-> Regime Distribution across dataset:\n{regime_counts.to_string()}")
    
    # 4. Chronological Splitting & Leakage-Free Normalization
    print("\n[3/7] Creating Chronological Splits & Scaling...")
    train_feat_raw, val_feat_raw, test_feat_raw = chronological_split(feature_df, train_ratio=0.6, val_ratio=0.2)
    train_ret_raw, val_ret_raw, test_ret_raw = chronological_split(returns_df, train_ratio=0.6, val_ratio=0.2)
    train_vol_raw, val_vol_raw, test_vol_raw = chronological_split(vol_df, train_ratio=0.6, val_ratio=0.2)
    
    print(f"-> Train Split: {train_feat_raw.index[0].strftime('%Y-%m-%d')} to {train_feat_raw.index[-1].strftime('%Y-%m-%d')} ({len(train_feat_raw)} days)")
    print(f"-> Val Split:   {val_feat_raw.index[0].strftime('%Y-%m-%d')} to {val_feat_raw.index[-1].strftime('%Y-%m-%d')} ({len(val_feat_raw)} days)")
    print(f"-> Test Split:  {test_feat_raw.index[0].strftime('%Y-%m-%d')} to {test_feat_raw.index[-1].strftime('%Y-%m-%d')} ({len(test_feat_raw)} days)")
    
    scaler = TimeSeriesScaler()
    scaler.fit(train_feat_raw.values)
    train_feat_scaled = scaler.transform(train_feat_raw.values)
    val_feat_scaled = scaler.transform(val_feat_raw.values)
    test_feat_scaled = scaler.transform(test_feat_raw.values)
    
    # Rolling Windows
    X_train, y_train, train_dates = create_rolling_windows(train_feat_scaled, train_ret_raw.values, lookback, train_feat_raw.index)
    X_val, y_val, val_dates = create_rolling_windows(val_feat_scaled, val_ret_raw.values, lookback, val_feat_raw.index)
    X_test, y_test, test_dates = create_rolling_windows(test_feat_scaled, test_ret_raw.values, lookback, test_feat_raw.index)
    
    # Align ex-ante vol for test set backtesting
    test_vol_aligned = test_vol_raw.iloc[lookback-1 : lookback-1 + len(y_test)].values
    test_regimes_aligned = regimes.loc[test_dates]
    
    num_features = X_train.shape[-1]
    num_assets = y_train.shape[-1]
    print(f"-> Windowed Shapes: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"-> Input Features={num_features}, Target Assets={num_assets}")
    
    batch_size = config_train.get('batch_size', 64)
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
    
    # Preserve chronological ordering. Shuffling is inappropriate for a
    # batch-level Sharpe objective because it destroys local return structure.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. Model Definitions
    model_factories = {
        'DLinear': lambda: DLinear(
            lookback=lookback,
            num_features=num_features,
            num_assets=num_assets,
            trend_window=config_models.get('dlinear', {}).get('trend_window', 5)
        ),
        'LSTM': lambda: LSTMSignalModel(
            lookback=lookback,
            num_features=num_features,
            num_assets=num_assets,
            hidden_dim=config_models.get('lstm', {}).get('hidden_dim', 64),
            num_layers=config_models.get('lstm', {}).get('layers', 2),
            dropout=config_models.get('lstm', {}).get('dropout', 0.1)
        ),
        'Mamba': lambda: MambaSignalModel(
            lookback=lookback,
            num_features=num_features,
            num_assets=num_assets,
            d_model=config_models.get('mamba', {}).get('d_model', 64),
            n_layers=config_models.get('mamba', {}).get('layers', 2),
            allow_fallback=True  # <-- Make sure this is explicitly True!
        ),
        'PatchTST': lambda: PatchTSTSignalModel(
            lookback=lookback,
            num_features=num_features,
            num_assets=num_assets,
            patch_length=config_models.get('patchtst', {}).get('patch_length', 5),
            d_model=config_models.get('patchtst', {}).get('d_model', 64),
            n_heads=config_models.get('patchtst', {}).get('heads', 4),
            n_layers=config_models.get('patchtst', {}).get('layers', 2),
            dropout=0.1
        )
    }
    
    # 6. Training & Multi-Seed Benchmarking
    print("\n[4/7] Training Models across Multiple Random Seeds & Evaluating...")
    
    model_params = {}
    model_train_times = {m: [] for m in model_factories.keys()}
    seed_sharpe_results = {m: [] for m in model_factories.keys()}
    primary_test_returns = {}
    primary_test_turnover = {}
    primary_test_signals = {}
    
    # Also evaluate Buy & Hold Equal Weight baseline
    ew_positions = np.ones_like(y_test) * (1.0 / num_assets)
    ew_backtester = Backtester(target_vol=target_vol, max_leverage=max_lev, tc_bps=5.0)
    ew_res = ew_backtester.run(ew_positions, test_vol_aligned, y_test, test_dates)
    primary_test_returns['Buy & Hold (EW)'] = ew_res['net_returns']
    primary_test_turnover['Buy & Hold (EW)'] = 0.0
    
    for model_name, factory in model_factories.items():
        sample_model = factory()
        param_count = sample_model.count_parameters()
        model_params[model_name] = param_count
        print(f"\n--- Benchmark Architecture: {model_name} ({param_count:,} parameters) ---")
        
        for seed_idx, seed in enumerate(seeds):
            set_seed(seed)
            model = factory()
            criterion = NegativeSharpeLoss(eps=1e-6, ann_factor=252.0, alpha=config_train.get('loss_alpha', 1.0))
            optimizer = optim.AdamW(model.parameters(), lr=config_train.get('learning_rate', 0.001), weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config_train.get('epochs', 50))
            
            trainer = BaseTrainer(model, criterion, optimizer, config_train, scheduler=scheduler)
            history, duration = trainer.train(train_loader, val_loader)
            model_train_times[model_name].append(duration)
            
            # Predict test signals
            signals = trainer.predict(test_loader)
            
            # Run test backtest at 5 bps
            backtester = Backtester(target_vol=target_vol, max_leverage=max_lev, tc_bps=5.0)
            bt_res = backtester.run(signals, test_vol_aligned, y_test, test_dates)
            net_ret = bt_res['net_returns']
            net_sharpe = calculate_sharpe_ratio(net_ret)
            seed_sharpe_results[model_name].append(net_sharpe)
            
            # Save the primary run (seed 42) for comprehensive reporting
            if seed_idx == 0:
                primary_test_returns[model_name] = net_ret
                primary_test_turnover[model_name] = float(np.nanmean(bt_res['turnover']) * 252.0)
                primary_test_signals[model_name] = signals
                
            print(f"  Seed {seed} -> Validation Loss: {history['val_loss'][-1]:.4f}, Out-of-Sample Net Sharpe: {net_sharpe:.3f} ({duration:.1f}s)")
            
    # 7. Comprehensive Metric Analysis
    print("\n[5/7] Computing Economic, Predictive, and Regime Metrics...")
    
    # A. Primary Overall Metrics Table
    metrics_records = []
    for name, ret_series in primary_test_returns.items():
        m_dict = compute_all_metrics(ret_series)
        # Block bootstrap 95% CI
        point_s, ci_low, ci_high = block_bootstrap_sharpe(ret_series.values, block_size=20, num_bootstraps=1000)
        m_dict['Model'] = name
        m_dict['95% Sharpe CI'] = f"[{ci_low:.2f}, {ci_high:.2f}]"
        m_dict['Turnover (ann.)'] = f"{primary_test_turnover.get(name, 0.0):.2f}"
        if name in model_params:
            m_dict['Parameters'] = f"{model_params[name]:,}"
            m_dict['Mean Train Time (s)'] = f"{np.mean(model_train_times[name]):.1f}"
            m_dict['Seed Sharpe Mean ± Std'] = f"{np.mean(seed_sharpe_results[name]):.2f} ± {np.std(seed_sharpe_results[name]):.2f}"
        else:
            m_dict['Parameters'] = "N/A"
            m_dict['Mean Train Time (s)'] = "N/A"
            m_dict['Seed Sharpe Mean ± Std'] = "N/A"
        metrics_records.append(m_dict)
        
    df_metrics = pd.DataFrame(metrics_records)
    # Reorder columns
    cols_order = ['Model', 'Sharpe Ratio', '95% Sharpe CI', 'Seed Sharpe Mean ± Std', 'Annualized Return', 'Annualized Volatility', 'Sortino Ratio', 'Max Drawdown', 'Calmar Ratio', 'Win Rate', 'Profit Factor', 'Turnover (ann.)', 'Parameters', 'Mean Train Time (s)']
    df_metrics = df_metrics[[c for c in cols_order if c in df_metrics.columns]]
    df_metrics.to_csv('results/tables/01_primary_economic_metrics.csv', index=False)
    
    # Save as Markdown
    with open('results/tables/01_primary_economic_metrics.md', 'w') as f:
        f.write("# Primary Economic Performance Summary (Test Split: 2022–2025)\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n")
        
    print("\n--- Summary Performance Table ---")
    print(df_metrics.to_string(index=False))
    
    # B. Regime-Stratified Performance
    regime_results = {}
    regime_robustness_records = []
    for model_name in model_factories.keys():
        ret_s = primary_test_returns[model_name]
        reg_eval = evaluate_by_regime(ret_s, test_regimes_aligned)
        regime_results[model_name] = reg_eval['sharpe']
        
        rob_dict = calculate_regime_robustness(ret_s, test_regimes_aligned)
        rob_dict['Model'] = model_name
        regime_robustness_records.append(rob_dict)
        
    df_regime_sharpes = pd.DataFrame(regime_results).T.fillna(0.0)
    df_regime_sharpes.to_csv('results/tables/02_regime_sharpes.csv')
    
    df_robustness = pd.DataFrame(regime_robustness_records).set_index('Model')
    df_robustness.to_csv('results/tables/03_regime_robustness_scores.csv')
    
    with open('results/tables/02_regime_performance.md', 'w') as f:
        f.write("# Regime-Stratified Sharpe Ratios\n\n")
        f.write(df_regime_sharpes.to_markdown())
        f.write("\n\n# Regime Robustness & Stress Degradation\n\n")
        f.write(df_robustness.to_markdown())
        f.write("\n")
        
    # C. Transaction Cost Sensitivity
    print("\n[6/7] Evaluating Transaction Cost Sensitivity...")
    tc_sensitivity_data = {}
    for model_name, sigs in primary_test_signals.items():
        tc_sensitivity_data[model_name] = {}
        for bps in tc_bps_list:
            bt = Backtester(target_vol=target_vol, max_leverage=max_lev, tc_bps=bps)
            r = bt.run(sigs, test_vol_aligned, y_test, test_dates)['net_returns']
            tc_sensitivity_data[model_name][bps] = calculate_sharpe_ratio(r)
            
    df_tc = pd.DataFrame(tc_sensitivity_data).T
    df_tc.columns = [f"{b} bps" for b in df_tc.columns]
    df_tc.to_csv('results/tables/04_transaction_cost_sensitivity.csv')
    
    # D. Predictive vs Economic Alignment
    pred_acc_dict = {}
    for model_name, sigs in primary_test_signals.items():
        # Directional accuracy: sign(signals) == sign(next_period_returns)
        dir_match = (np.sign(sigs) == np.sign(y_test))
        acc = float(np.mean(dir_match))
        pred_acc_dict[model_name] = acc
        
    # E. Statistical Hypothesis Testing
    stat_test_records = []
    model_names_list = list(model_factories.keys())
    for i in range(len(model_names_list)):
        for j in range(i + 1, len(model_names_list)):
            m_a = model_names_list[i]
            m_b = model_names_list[j]
            test_res = paired_sharpe_difference_test(
                primary_test_returns[m_a].values, 
                primary_test_returns[m_b].values, 
                num_bootstraps=1000
            )
            stat_test_records.append({
                'Comparison': f"{m_a} vs {m_b}",
                'Sharpe (A)': f"{test_res['sharpe_a']:.3f}",
                'Sharpe (B)': f"{test_res['sharpe_b']:.3f}",
                'Difference (A - B)': f"{test_res['sharpe_diff']:.3f}",
                'Bootstrap p-value': f"{test_res['p_value']:.4f}",
                'Significant at 5%': "Yes" if test_res['p_value'] < 0.05 or test_res['p_value'] > 0.95 else "No"
            })
    df_stats = pd.DataFrame(stat_test_records)
    df_stats.to_csv('results/tables/05_hypothesis_tests.csv', index=False)
    
    with open('results/tables/05_hypothesis_tests.md', 'w') as f:
        f.write("# Paired Bootstrap Hypothesis Tests (Sharpe Ratio Differences)\n\n")
        f.write(df_stats.to_markdown(index=False))
        f.write("\n")
        
    # 8. Publication Visualizations
    print("\n[7/7] Generating High-Resolution Report Figures...")
    plot_cumulative_returns(primary_test_returns, save_path="results/figures/03_cumulative_returns.png")
    plot_drawdowns(primary_test_returns, save_path="results/figures/04_drawdown_curves.png")
    plot_sharpe_by_regime(df_regime_sharpes, save_path="results/figures/05_sharpe_by_regime.png")
    
    primary_sharpes = {m: df_metrics.set_index('Model').loc[m, 'Sharpe Ratio'] for m in model_factories.keys()}
    plot_turnover_vs_sharpe(primary_test_turnover, primary_sharpes, save_path="results/figures/06_turnover_vs_sharpe.png")
    plot_transaction_cost_sensitivity(tc_sensitivity_data, save_path="results/figures/07_transaction_cost_sensitivity.png")
    plot_predictive_vs_economic(pred_acc_dict, primary_sharpes, save_path="results/figures/08_predictive_vs_economic.png")
    plot_complexity_tradeoff(model_params, primary_sharpes, save_path="results/figures/09_complexity_tradeoff.png")
    plot_seed_robustness(seed_sharpe_results, save_path="results/figures/10_seed_robustness.png")
    
    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETED SUCCESSFULLY!")
    print(" Results stored in 'results/tables/' and 'results/figures/'")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
