import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import inspect
from torch.utils.data import TensorDataset, DataLoader
from src.data.splits import chronological_split
from src.data.windows import create_rolling_windows
from src.training.trainer import BaseTrainer

# Documented small search space for financial time-series model validation
HYPERPARAMETER_SEARCH_SPACE = {
    "lr": [1e-3, 1e-4],
    "lookback": [30, 60],
    "d_model": [32, 64],
    "batch_size": [32]
}

def run_validation_tuning(full_train_val_df, model_class, criterion_factory, config_template, feature_cols, target_cols):
    """
    Performs validation-only hyperparameter tuning using sliding windows.
    The test dataset is strictly excluded from this entire loop.
    """
    # Split available training data into training and validation folds chronologically
    train_df, val_df, _ = chronological_split(full_train_val_df, train_ratio=0.75, val_ratio=0.25)
    
    best_val_loss = float('inf')
    best_hyperparameters = None
    best_model_state = None
    
    print("=" * 50)
    print("Starting Validation-Only Hyperparameter Search (Task 4)...")
    print("=" * 50)
    
    # Grid search over the documented search space
    for lr in HYPERPARAMETER_SEARCH_SPACE["lr"]:
        for lookback in HYPERPARAMETER_SEARCH_SPACE["lookback"]:
            for d_model in HYPERPARAMETER_SEARCH_SPACE["d_model"]:
                for batch_size in HYPERPARAMETER_SEARCH_SPACE["batch_size"]:
                    
                    current_config = config_template.copy()
                    current_config["lr"] = lr
                    current_config["lookback"] = lookback
                    current_config["d_model"] = d_model
                    current_config["batch_size"] = batch_size
                    
                    # 1. Build rolling windows for training subset
                    train_features = train_df[feature_cols].values
                    train_targets = train_df[target_cols].values
                    X_train, y_train, _ = create_rolling_windows(train_features, train_targets, lookback)
                    
                    # 2. Build rolling windows for validation subset
                    val_features = val_df[feature_cols].values
                    val_targets = val_df[target_cols].values
                    X_val, y_val, _ = create_rolling_windows(val_features, val_targets, lookback)
                    
                    if len(X_train) == 0 or len(X_val) == 0:
                        continue # Skip configurations where lookback exceeds data split length
                        
                    # 3. Create PyTorch DataLoaders
                    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
                    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
                    
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                    # Dynamically inspect model constructor arguments to prevent TypeErrors
                    sig = inspect.signature(model_class.__init__)
                    potential_kwargs = {
                        "lookback": lookback,
                        "seq_len": lookback, # Support models expecting seq_len
                        "num_features": len(feature_cols),
                        "enc_in": len(feature_cols), # Support DLinear convention
                        "num_assets": len(target_cols),
                        "pred_len": len(target_cols), # Support forecasting convention
                        "d_model": d_model
                    }
                    # Filter kwargs to match only what model_class.__init__ accepts
                    model_kwargs = {k: v for k, v in potential_kwargs.items() if k in sig.parameters}
                    
                    # Instantiate model safely
                    model = model_class(**model_kwargs)
                    
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = criterion_factory()
                    
                    trainer = BaseTrainer(
                        model=model, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        config=current_config
                    )
                    
                    # Train and evaluate using validation feedback only
                    history, _ = trainer.train(train_loader, val_loader=val_loader)
                    final_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
                    
                    print(f"Tested Config -> lr: {lr}, lookback: {lookback}, d_model: {d_model} | Val Loss: {final_val_loss:.4f}")
                    
                    # Track best hyperparameters based strictly on validation loss
                    if final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        best_hyperparameters = {"lr": lr, "lookback": lookback, "d_model": d_model, "batch_size": batch_size}
                        best_model_state = model.state_dict()

    print("=" * 50)
    print(f"Hyperparameter tuning completed using validation feedback only.")
    print(f"Best Configuration Found: {best_hyperparameters}")
    print("Test dataset remains untouched and secured.")
    print("=" * 50)
    
    return best_hyperparameters, best_model_state

def bounded_model_candidates(model_name, models_config, learning_rates, max_trials=2):
    """Return at most two comparable validation candidates, with one architecture change.

    Candidate zero uses the configured architecture. Candidate one halves one
    model-specific width (subject to its attention head count) and uses the
    second learning rate. Test data must not be passed to selection.
    """
    import copy
    if max_trials < 1 or not learning_rates:
        raise ValueError('max_trials and learning_rates must be nonempty')
    fields = {'dlinear': 'trend_window', 'lstm': 'hidden_dim',
              'mamba': 'd_model', 'patchtst': 'd_model', 'ahat_fin': 'd_model'}
    if model_name not in fields:
        raise ValueError(f'unknown model: {model_name}')
    baseline = copy.deepcopy(models_config)
    candidates = [{'learning_rate': float(learning_rates[0]), 'models': baseline}]
    if max_trials > 1:
        variant = copy.deepcopy(models_config)
        cfg = variant[model_name]
        field = fields[model_name]
        value = int(cfg[field])
        if field == 'trend_window':
            # Moving-average decomposition requires a positive odd kernel.
            candidate = max(1, value // 2)
            cfg[field] = candidate if candidate % 2 else candidate + 1
        elif model_name in ('patchtst', 'ahat_fin'):
            heads = int(cfg['heads'])
            cfg[field] = max(heads, (value // (2 * heads)) * heads)
        else:
            cfg[field] = max(1, value // 2)
        candidates.append({'learning_rate': float(learning_rates[min(1, len(learning_rates)-1)]), 'models': variant})
    return candidates


def select_validation_candidate(trials):
    """Select the smallest finite negative validation Sharpe, never test metrics."""
    valid = [trial for trial in trials if np.isfinite(trial['negative_validation_sharpe'])]
    if not valid:
        raise ValueError('no finite validation scores')
    return min(valid, key=lambda trial: trial['negative_validation_sharpe'])
