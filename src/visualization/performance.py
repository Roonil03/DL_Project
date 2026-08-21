import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_cumulative_returns(returns_dict: dict, save_path="results/figures/08_cumulative_returns.png"):
    plt.figure(figsize=(12, 6))
    for name, returns in returns_dict.items():
        cum_ret = (1 + returns).cumprod()
        plt.plot(cum_ret.index, cum_ret, label=name)
        
    plt.title('Cumulative Strategy Returns')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_sharpe_by_regime(regime_df: pd.DataFrame, save_path="results/figures/12_sharpe_by_regime.png"):
    """
    Heatmap of Sharpe Ratio across models and regimes.
    regime_df index = models, columns = regimes
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(regime_df, annot=True, fmt=".2f", cmap="RdYlGn", center=0)
    plt.title('Sharpe Ratio by Market Regime')
    plt.ylabel('Model Architecture')
    plt.xlabel('Regime')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
