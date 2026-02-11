from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Sequence

@dataclass(frozen=True)
class CasePaths:
    """Project-relative paths used by the pipeline."""
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    results_dir: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"


@dataclass(frozen=True)
class DataConfig:
    """Data loading + basic schema configuration."""
    dataset_filename: str = "Lead Scoring.csv"
    target_col: str = "Converted"

    # If your dataset has a date/time column we can use for drift checks.
    # Set to None if not available.
    time_col: Optional[str] = None

    # Columns that are IDs / free text / high-cardinality that we typically exclude from segment tests.
    drop_cols: Sequence[str] = field(default_factory=lambda: (
        "Prospect ID",
        "Lead Number",
    ))


@dataclass(frozen=True)
class CleaningConfig:
    """Rules for cleaning + level grouping BEFORE analysis."""
    # Minimum segment size to consider a category level as its own segment.
    min_segment_n: int = 50

    # Replace common placeholders with NaN. (We will apply this to object columns.)
    missing_tokens: Sequence[str] = field(default_factory=lambda: (
        "Select", "select",
        "Not Provided", "not provided",
        "None", "none",
        "", " ",
    ))

    # For rare categories below min_segment_n we can group into "__OTHER__"
    other_label: str = "__OTHER__"

    # Optional: cap number of levels per categorical feature (after grouping) to keep reports readable.
    max_levels_per_feature: int = 25


@dataclass(frozen=True)
class SignificanceConfig:
    """Gate 2: significance testing configuration."""
    alpha: float = 0.05
    # Multiple testing correction method for many tests.
    # We'll implement BH-FDR first.
    multiple_testing_method: str = "bh_fdr"

    # Ignore tests for segments smaller than this even if min_segment_n is lower (safety).
    hard_min_n: int = 20


@dataclass(frozen=True)
class RobustnessConfig:
    """Gate 3: robustness checks."""
    # Bootstrap / subsample settings
    bootstrap_iterations: int = 200
    bootstrap_sample_frac: float = 0.7
    random_state: int = 42

    # A segment effect is considered stable if this fraction of bootstraps keeps same direction.
    min_stability_fraction: float = 0.80

    # If time_col exists: number of buckets for time splits (e.g., monthly/quantiles)
    time_buckets: int = 6


@dataclass(frozen=True)
class MultivariateConfig:
    """Multivariate explanatory model (logistic regression)."""
    enabled: bool = True
    # Regularization keeps coefficients stable with many one-hot variables
    penalty: str = "l2"
    C: float = 1.0
    max_iter: int = 2000

    # Rare one-hot columns can explode; we’ll reduce via grouping in cleaning anyway.
    # This flag just controls whether we drop features with extremely low variance.
    drop_low_variance: bool = True


@dataclass(frozen=True)
class DecisionConfig:
    """How to convert evidence into SCALE / INVESTIGATE / STOP."""
    # Minimum absolute lift to consider meaningful (e.g., +3 percentage points)
    min_abs_lift: float = 0.03

    # Require significance + robustness for SCALE
    require_significant_for_scale: bool = True
    require_robust_for_scale: bool = True

    # Leakage handling
    # If a feature is flagged leakage, force "DO_NOT_USE"
    force_do_not_use_if_leakage: bool = True


@dataclass(frozen=True)
class LeadScoringCaseConfig:
    """Top-level config bundling everything for Case 1."""
    paths: CasePaths = CasePaths()
    data: DataConfig = DataConfig()
    cleaning: CleaningConfig = CleaningConfig()
    significance: SignificanceConfig = SignificanceConfig()
    robustness: RobustnessConfig = RobustnessConfig()
    multivariate: MultivariateConfig = MultivariateConfig()
    decision: DecisionConfig = DecisionConfig()

    # Feature governance: explicitly mark suspect/leakage columns (you can refine after EDA)
    leakage_suspects: Sequence[str] = field(default_factory=lambda: (
        # Examples (we’ll validate against actual columns once loaded):
        "Last Activity",
        "Last Notable Activity",
        "Tags",
        "Lead Quality",
        "Last Email Open Date",
        "Last Email Click Date",
    ))

    # Optional: whitelist categorical features for segment analysis.
    # If empty, we auto-detect object/category columns excluding drop_cols/target.
    categorical_whitelist: Sequence[str] = field(default_factory=tuple)