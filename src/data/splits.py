import pandas as pd
from typing import Generator, Tuple

def chronological_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset chronologically to prevent leakage (Baseline single split).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df


def expanding_window_split(
    df: pd.DataFrame, 
    initial_train_size: int, 
    test_size: int, 
    step_size: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generates expanding-window train and test splits for walk-forward evaluation.
    - initial_train_size: Number of rows for the first training window.
    - test_size: Number of rows for the evaluation test fold.
    - step_size: Number of rows to roll forward in each iteration.
    """
    n = len(df)
    current_train_end = initial_train_size
    
    while current_train_end + test_size <= n:
        train_df = df.iloc[:current_train_end]
        test_df = df.iloc[current_train_end : current_train_end + test_size]
        
        yield train_df, test_df
        current_train_end += step_size


def rolling_window_split(
    df: pd.DataFrame, 
    window_size: int, 
    test_size: int, 
    step_size: int
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
    """
    Generates rolling-window (fixed-size training window) train and test splits.
    - window_size: Fixed number of rows for the training window.
    - test_size: Number of rows for the evaluation test fold.
    - step_size: Number of rows to roll forward in each iteration.
    """
    n = len(df)
    start_idx = 0
    
    while start_idx + window_size + test_size <= n:
        train_df = df.iloc[start_idx : start_idx + window_size]
        test_df = df.iloc[start_idx + window_size : start_idx + window_size + test_size]
        
        yield train_df, test_df
        start_idx += step_size