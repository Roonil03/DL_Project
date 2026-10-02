import os
import pandas as pd
import yfinance as yf


def _ensure_parent_directory(path: str) -> None:
    """Create the cache directory when a parent path is present."""
    parent = os.path.dirname(os.fspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


class BaseDatasetAdapter:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement load()")

class GlobalIndianAdapter(BaseDatasetAdapter):
    """
    Adapter for the Global & Indian Financial Markets Dataset (2010–2025).
    Includes:
      - NIFTY 50 (^NSEI)
      - S&P 500 (^GSPC)
      - USD/INR (INR=X)
      - Gold (GC=F)
      - Brent Crude (BZ=F)
      - US 10-Year Treasury Yield (^TNX)
    Handles date alignment across international market holidays with forward-filling.
    """
    TICKER_MAP = {
        '^NSEI': 'NIFTY50',
        '^GSPC': 'SP500',
        'INR=X': 'USDINR',
        'GC=F': 'GOLD',
        'BZ=F': 'BRENT',
        '^TNX': 'US10Y'
    }

    def __init__(self, data_path: str = "data/global_indian_markets.csv", start_date: str = "2010-01-01", end_date: str = "2025-12-31"):
        super().__init__(data_path)
        self.start_date = start_date
        self.end_date = end_date

    def load(self, force_download: bool = False) -> pd.DataFrame:
        if os.path.exists(self.data_path) and not force_download:
            df = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            return df

        _ensure_parent_directory(self.data_path)
        tickers = list(self.TICKER_MAP.keys())
        raw_data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
        
        # Extract Close prices
        if 'Close' in raw_data.columns.levels[0]:
            df_close = raw_data['Close'].copy()
        else:
            df_close = raw_data.copy()

        # Rename tickers to clean asset names
        df_close = df_close.rename(columns=self.TICKER_MAP)
        
        # Ensure standard column order
        cols = [self.TICKER_MAP[t] for t in tickers if self.TICKER_MAP[t] in df_close.columns]
        df_close = df_close[cols]

        # Forward fill missing dates (for different international market trading calendars)
        df_close = df_close.ffill().dropna()

        # Save cached CSV
        df_close.to_csv(self.data_path)
        return df_close

class SP500Adapter(BaseDatasetAdapter):
    """
    Adapter for S&P 500 constituents or single index sanity check.
    """
    def __init__(self, data_path: str = "data/sp500_daily.csv", start_date: str = "2010-01-01", end_date: str = "2025-12-31"):
        super().__init__(data_path)
        self.start_date = start_date
        self.end_date = end_date

    def load(self, force_download: bool = False) -> pd.DataFrame:
        if os.path.exists(self.data_path) and not force_download:
            return pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            
        _ensure_parent_directory(self.data_path)
        df = yf.download('^GSPC', start=self.start_date, end=self.end_date, progress=False)
        is_multi = isinstance(df.columns, pd.MultiIndex)
        has_close = ('Close' in df.columns.levels[0]) if is_multi else ('Close' in df.columns)
        if has_close:
            close_s = df['Close']
        else:
            close_s = df
        close_df = pd.DataFrame({'SP500': close_s.squeeze()}).dropna()
        close_df.to_csv(self.data_path)
        return close_df

