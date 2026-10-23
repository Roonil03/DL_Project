import pandas as pd
import torch.nn as nn
from src.training.tune import run_validation_tuning
from src.models.dlinear import DLinear

print("Loading dataset for tuning smoke test...")
# Load a subset of your audited daily data for a fast test run
df = pd.read_csv("data/daily_market_data.csv").head(500)

# Define feature and target columns based on your dataset schema
feature_cols = ['nifty50_close', 'sp500_close', 'usd_inr_close', 'gold_close', 'brent_close']
target_cols = ['nifty50_close']

# Simple MSE loss criterion factory for testing
criterion_factory = lambda: nn.MSELoss()

# Base configuration template (keep epochs low for a fast smoke test)
config_template = {
    "epochs": 2,
    "early_stopping_patience": 2,
    "grad_clip": 1.0,
    "num_features": len(feature_cols),
    "num_assets": len(target_cols)
}

print("Executing validation-only hyperparameter search...")
best_params, best_state = run_validation_tuning(
    full_train_val_df=df,
    model_class=DLinear,
    criterion_factory=criterion_factory,
    config_template=config_template,
    feature_cols=feature_cols,
    target_cols=target_cols
)

print("\n--- TEST RESULTS ---")
print("Best Hyperparameters Selected:", best_params)
print("Task 4 Tuning Test Passed Successfully!")