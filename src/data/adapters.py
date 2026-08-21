import pandas as pd

class BaseDatasetAdapter:
    def __init__(self, data_path):
        self.data_path = data_path

    def load(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement load()")

class GlobalIndianAdapter(BaseDatasetAdapter):
    """
    Adapter for the Global & Indian Financial Markets Dataset.
    Must handle parsing the CSV, aligning dates, and forward-filling missing values logically.
    """
    def load(self) -> pd.DataFrame:
        # Placeholder for actual parsing
        # df = pd.read_csv(self.data_path, parse_dates=['Date'])
        # df.set_index('Date', inplace=True)
        # df.fillna(method='ffill', inplace=True)
        # return df
        pass

class SP500Adapter(BaseDatasetAdapter):
    """
    Adapter for S&P 500 Daily Historical Data.
    """
    def load(self) -> pd.DataFrame:
        pass
