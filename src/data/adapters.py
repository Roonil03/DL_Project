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
    Loads explicitly from the audited local data files.
    """
    def __init__(
        self,
        data_path: str = "data/daily_market_data.csv",
        macro_path: str = "data/monthly_macro_data.csv",
        start_date: str = "2010-01-01",
        end_date: str = "2025-12-31",
        allow_yfinance_fallback: bool = False,
    ):
        super().__init__(data_path)
        self.macro_path = macro_path
        self.start_date = start_date
        self.end_date = end_date
        self.allow_yfinance_fallback = allow_yfinance_fallback

    def load(self, force_download: bool = False) -> pd.DataFrame:
        if os.path.exists(self.data_path) and not force_download:
            # Load the main daily market data CSV
            df = pd.read_csv(self.data_path)
            
            # Standardize date column (handle common variations like 'Date' or 'date')
            date_col = next((c for c in df.columns if c.lower() == 'date'), None)
            if date_col:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.set_index(date_col)
            elif 'Unnamed: 0' in df.columns:
                df = df.set_index('Unnamed: 0')
                df.index = pd.to_datetime(df.index, errors='coerce')
            elif isinstance(df.index, pd.DatetimeIndex) is False:
                # Try setting first column as index if it looks like dates
                try:
                    df.index = pd.to_datetime(df.iloc[:, 0], errors='raise')
                    df = df.iloc[:, 1:]
                except Exception:
                    pass

            df = df.apply(pd.to_numeric, errors='coerce')
            df = df[~df.index.duplicated(keep='last')].sort_index()
            
            # Filter by start and end date bounds
            df = df.loc[self.start_date:self.end_date]
            return df.ffill().dropna()

        if not self.allow_yfinance_fallback:
            raise FileNotFoundError(
                f"Audited dataset not found at {self.data_path!r}. "
                "Ensure 'daily_market_data.csv' is placed inside the 'data/' directory."
            )

        raise FileNotFoundError("Yahoo Finance fallback is disabled by project audit policy.")

    def load_macro(self) -> pd.DataFrame:
        """Loads the monthly macroeconomic indicators dataset."""
        if os.path.exists(self.macro_path):
            macro_df = pd.read_csv(self.macro_path)
            date_col = next((c for c in macro_df.columns if c.lower() == 'date'), None)
            if date_col:
                macro_df[date_col] = pd.to_datetime(macro_df[date_col], errors='coerce')
                macro_df = macro_df.set_index(date_col)
            macro_df = macro_df.apply(pd.to_numeric, errors='coerce')
            return macro_df.sort_index().ffill()
        return pd.DataFrame()


class SP500Adapter(BaseDatasetAdapter):
    """
    Adapter for S&P 500 benchmark checks.
    """
    def __init__(self, data_path: str = "data/sp500_daily.csv", start_date: str = "2010-01-01", end_date: str = "2025-12-31"):
        super().__init__(data_path)
        self.start_date = start_date
        self.end_date = end_date

    def load(self, force_download: bool = False) -> pd.DataFrame:
        if os.path.exists(self.data_path) and not force_download:
            return pd.read_csv(self.data_path, index_col=0, parse_dates=True)
            
        # Fallback to local daily data mapping if available
        daily_file = "data/daily_market_data.csv"
        if os.path.exists(daily_file):
            df = pd.read_csv(daily_file, parse_dates=True, index_col=0)
            sp_col = next((c for c in df.columns if 'sp500' in c.lower() or 'gspc' in c.lower()), None)
            if sp_col:
                return pd.DataFrame({'SP500': df[sp_col]}).dropna()
                
        raise FileNotFoundError("S&P 500 data source could not be resolved locally.")