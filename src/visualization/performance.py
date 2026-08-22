import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from typing import Dict, List

# Set modern aesthetic styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

def plot_market_overview(prices: pd.DataFrame, save_path="results/figures/01_market_data_overview.png"):
    """Normalized asset price paths."""
    plt.figure(figsize=(12, 6))
    norm_prices = prices / prices.iloc[0] * 100.0
    for col in norm_prices.columns:
        plt.plot(norm_prices.index, norm_prices[col], label=col, lw=1.5)
    plt.title('Normalized Multi-Asset Historical Prices (Base = 100)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Normalized Level', fontsize=11)
    plt.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_regime_classification(prices: pd.DataFrame, regimes: pd.Series, save_path="results/figures/02_regime_classification.png"):
    """Plots market proxy with colored regime background."""
    plt.figure(figsize=(14, 6))
    market_proxy = prices.iloc[:, 0]
    plt.plot(market_proxy.index, market_proxy, color='#1f77b4', lw=1.5, label='Market Benchmark')
    
    unique_regimes = regimes.unique()
    palette = {'LowVol_Bull': '#2ca02c', 'HighVol_Bull': '#98df8a', 'LowVol_Bear': '#ff7f0e', 'HighVol_Bear': '#d62728', 'Extreme_Stress': '#9467bd'}
    
    for r_name in unique_regimes:
        if pd.isna(r_name): continue
        mask = (regimes == r_name)
        plt.fill_between(regimes.index, market_proxy.min(), market_proxy.max(), where=mask, color=palette.get(r_name, '#cccccc'), alpha=0.25, label=f'Regime: {r_name}')
        
    plt.title('Market Trajectory and Volatility × Trend Regime Classification', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Price', fontsize=11)
    # Deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cumulative_returns(returns_dict: Dict[str, pd.Series], save_path="results/figures/03_cumulative_returns.png"):
    """Plots cumulative return equity curves."""
    plt.figure(figsize=(12, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (name, returns) in enumerate(returns_dict.items()):
        cum_ret = (1.0 + returns.fillna(0.0)).cumprod()
        plt.plot(cum_ret.index, cum_ret, label=name, lw=2.0, color=colors[idx % len(colors)])
        
    plt.title('Cumulative Out-of-Sample Portfolio Growth', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Growth Factor (Base = 1.0)', fontsize=11)
    plt.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdowns(returns_dict: Dict[str, pd.Series], save_path="results/figures/04_drawdown_curves.png"):
    """Plots underwater drawdown curves."""
    plt.figure(figsize=(12, 5))
    for name, returns in returns_dict.items():
        cum = (1.0 + returns.fillna(0.0)).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        plt.plot(dd.index, dd * 100.0, label=name, lw=1.5)
        
    plt.title('Underwater Portfolio Drawdowns (%)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Date', fontsize=11)
    plt.ylabel('Drawdown (%)', fontsize=11)
    plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_by_regime(regime_df: pd.DataFrame, save_path="results/figures/05_sharpe_by_regime.png"):
    """Heatmap of Sharpe Ratio across models and regimes."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(regime_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0, cbar_kws={'label': 'Sharpe Ratio'}, linewidths=0.5)
    plt.title('Risk-Adjusted Performance (Sharpe Ratio) by Market Regime', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Model Architecture', fontsize=11)
    plt.xlabel('Market Regime', fontsize=11)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_turnover_vs_sharpe(turnover_dict: Dict[str, float], sharpe_dict: Dict[str, float], save_path="results/figures/06_turnover_vs_sharpe.png"):
    """Scatter plot of Annualized Turnover vs Sharpe Ratio."""
    plt.figure(figsize=(8, 6))
    for model in turnover_dict.keys():
        t = turnover_dict[model]
        s = sharpe_dict.get(model, 0.0)
        plt.scatter(t, s, s=120, label=model)
        plt.annotate(model, (t, s), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10, fontweight='bold')
    plt.title('Trading Turnover vs. Risk-Adjusted Return (Sharpe)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Annualized Turnover', fontsize=11)
    plt.ylabel('Sharpe Ratio (Net)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_transaction_cost_sensitivity(tc_results: Dict[str, Dict[int, float]], save_path="results/figures/07_transaction_cost_sensitivity.png"):
    """Sharpe decay as transaction costs increase (bps)."""
    plt.figure(figsize=(9, 6))
    for model, bps_map in tc_results.items():
        bps_levels = sorted(bps_map.keys())
        sharpes = [bps_map[b] for b in bps_levels]
        plt.plot(bps_levels, sharpes, marker='o', lw=2, label=model)
    plt.title('Transaction Cost Sensitivity: Sharpe Ratio vs Fee (bps)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Transaction Cost (basis points)', fontsize=11)
    plt.ylabel('Net Annualized Sharpe Ratio', fontsize=11)
    plt.legend(frameon=True, facecolor='white')
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_predictive_vs_economic(pred_acc: Dict[str, float], sharpes: Dict[str, float], save_path="results/figures/08_predictive_vs_economic.png"):
    """Scatter of Directional Accuracy vs Sharpe."""
    plt.figure(figsize=(8, 6))
    for model in pred_acc.keys():
        acc = pred_acc[model]
        s = sharpes.get(model, 0.0)
        plt.scatter(acc * 100.0, s, s=130, label=model)
        plt.annotate(model, (acc * 100.0, s), textcoords="offset points", xytext=(6, 6), fontweight='bold')
    plt.title('Predictive Accuracy vs. Economic Performance (Sharpe)', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Directional Accuracy (%)', fontsize=11)
    plt.ylabel('Net Sharpe Ratio', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_complexity_tradeoff(params_dict: Dict[str, int], sharpes: Dict[str, float], save_path="results/figures/09_complexity_tradeoff.png"):
    """Model Parameters vs Sharpe Ratio."""
    plt.figure(figsize=(9, 6))
    for model, p in params_dict.items():
        s = sharpes.get(model, 0.0)
        plt.scatter(p, s, s=140, label=model)
        plt.annotate(f"{model} ({p:,} params)", (p, s), textcoords="offset points", xytext=(8, 5), fontweight='bold')
    plt.title('Model Complexity Trade-Off: Parameter Count vs. Sharpe', fontsize=13, fontweight='bold', pad=12)
    plt.xlabel('Trainable Parameters (log scale)', fontsize=11)
    plt.ylabel('Net Sharpe Ratio', fontsize=11)
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_seed_robustness(seed_results: Dict[str, List[float]], save_path="results/figures/10_seed_robustness.png"):
    """Boxplot of Sharpe ratios across random seeds."""
    plt.figure(figsize=(9, 6))
    df = pd.DataFrame(seed_results)
    sns.boxplot(data=df, palette="Set2", width=0.4)
    sns.stripplot(data=df, color='black', alpha=0.7, size=7, jitter=0.1)
    plt.title('Stability Across Random Seeds [42, 52, 62, 72, 82]', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Out-of-Sample Net Sharpe Ratio', fontsize=11)
    plt.xlabel('Architecture', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

