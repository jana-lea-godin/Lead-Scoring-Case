from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import LeadScoringCaseConfig


@dataclass
class LoadedData:
    """Container for loaded datasets (raw + optional processed)."""
    raw: pd.DataFrame
    cleaned: Optional[pd.DataFrame] = None
    with_tracking: Optional[pd.DataFrame] = None
    no_tracking: Optional[pd.DataFrame] = None


class LeadDataLoader:
    """
    Loads raw + processed artifacts from the project folder structure.

    Responsibility:
    - reading CSVs from disk
    - validating paths exist
    - NO cleaning / NO transformations
    """

    def __init__(self, cfg: LeadScoringCaseConfig, project_root: str | Path):
        self.cfg = cfg
        self.project_root = Path(project_root)

        self.raw_path = self.project_root / self.cfg.paths.data_raw_dir / self.cfg.data.raw_filename
        self.cleaned_path = self.project_root / self.cfg.paths.data_processed_dir / self.cfg.data.processed_cleaned_filename
        self.with_tracking_path = self.project_root / self.cfg.paths.data_processed_dir / self.cfg.data.processed_with_tracking_filename
        self.no_tracking_path = self.project_root / self.cfg.paths.data_processed_dir / self.cfg.data.processed_no_tracking_filename

    def load_raw(self) -> pd.DataFrame:
        """Load raw dataset from data/raw."""
        return self._read_csv(self.raw_path)

    def load_processed(self) -> LoadedData:
        """
        Load raw + whatever processed files exist.
        Returns LoadedData with None where files are missing.
        """
        raw = self.load_raw()

        cleaned = self._read_csv_if_exists(self.cleaned_path)
        with_tracking = self._read_csv_if_exists(self.with_tracking_path)
        no_tracking = self._read_csv_if_exists(self.no_tracking_path)

        return LoadedData(
            raw=raw,
            cleaned=cleaned,
            with_tracking=with_tracking,
            no_tracking=no_tracking,
        )

    # -------------------------
    # internal helpers
    # -------------------------

    @staticmethod
    def _read_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        return pd.read_csv(path)

    @staticmethod
    def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        return pd.read_csv(path)