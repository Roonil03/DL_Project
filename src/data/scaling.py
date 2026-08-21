import numpy as np

class TimeSeriesScaler:
    """
    Scales features strictly using training data to prevent look-ahead bias.
    """
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, train_data: np.ndarray):
        """Fits scaler on training data. Assumes shape (time, features) or similar."""
        self.mean = np.nanmean(train_data, axis=0)
        self.std = np.nanstd(train_data, axis=0)
        # Prevent division by zero
        self.std[self.std == 0] = 1e-8

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise ValueError("Scaler must be fitted before transform.")
        return (data - self.mean) / self.std

    def fit_transform(self, train_data: np.ndarray) -> np.ndarray:
        self.fit(train_data)
        return self.transform(train_data)
