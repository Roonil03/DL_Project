import pandas as pd

def chronological_split(df: pd.DataFrame, train_ratio=0.6, val_ratio=0.2):
    """
    Splits the dataset chronologically to prevent leakage.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    return train_df, val_df, test_df
