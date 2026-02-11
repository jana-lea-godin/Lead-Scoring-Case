import pandas as pd
from pathlib import Path


class LeadDataLoader:
    """
    Responsible only for loading raw data.
    No cleaning, no transformation.
    """

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"File not found: {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        return self.df

    def basic_info(self) -> None:
        """Print basic dataset info."""
        if self.df is None:
            raise ValueError("Data not loaded yet. Call load_data() first.")

        print("Shape:", self.df.shape)
        print("\nMissing values:")
        print(self.df.isna().sum().sort_values(ascending=False).head(10))
        print("\nData types:")
        print(self.df.dtypes)