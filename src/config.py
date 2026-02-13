from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping


# ======================================================================================
# PATHS
# ======================================================================================

@dataclass(frozen=True)
class CasePaths:
    """Project-relative paths used by the pipeline."""
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"
    results_dir: str = "results"
    tables_dir: str = "results/tables"
    figures_dir: str = "results/figures"


# ======================================================================================
# DATA + SCHEMA
# ======================================================================================

@dataclass(frozen=True)
class DataConfig:
    """Data loading + basic schema configuration."""
    raw_filename: str = "Lead Scoring.csv"

    # processed artifacts
    processed_cleaned_filename: str = "lead_scoring_cleaned.csv"
    processed_with_tracking_filename: str = "lead_scoring_with_tracking.csv"
    processed_no_tracking_filename: str = "lead_scoring_no_tracking.csv"

    # (optional but useful) dataset after leakage/id removal
    processed_model_filename: str = "lead_scoring_model.csv"

    target_col: str = "Converted"


    time_col: Optional[str] = None

    # IDs that should never be used as model features / segment tests,
    # but are kept in data for joining recommendations later.
    id_cols: Sequence[str] = field(default_factory=lambda: (
        "Prospect ID",
        "Lead Number",
    ))


# ======================================================================================
# CLEANING RULES
# ======================================================================================

@dataclass(frozen=True)
class CleaningConfig:
    """
    Rules for cleaning BEFORE analysis.

    - pseudo-missing tokens -> NaN
    - create tracking split if desired
    """
    # Replace placeholders with NaN in object columns
    missing_tokens: Sequence[str] = field(default_factory=lambda: (
        "Select", "select",
        "Not Provided", "not provided",
        "None", "none",
        "", " ",
    ))

    # If True: strip whitespace for all object columns
    strip_object_whitespace: bool = True

    # If True: normalize some common casing issues in selected columns
    normalize_case: bool = True

    # Rare category grouping (used later in segment profiler / model prep)
    min_segment_n: int = 50
    other_label: str = "__OTHER__" 
    max_levels_per_feature: int = 25

    # Tracking columns used to define "no tracking" rows (NaN)
    # (Im Notebook war TotalVisits entscheidend; hier bleiben wir kompatibel + erweitern sinnvoll.)
    tracking_numeric_cols: Sequence[str] = field(default_factory=lambda: (
        "TotalVisits",
        "Total Time Spent on Website",
        "Page Views Per Visit",
    ))

    # If True: create a separate dataset where tracking cols are missing (e.g., 137 rows)
    split_tracking_missing_rows: bool = True

    # Name for the missing-tracking group (informal label)
    tracking_missing_label: str = "TRACKING_MISSING"

    # Normalizations you explicitly used in the notebook
    normalize_lead_source_titlecase: bool = True
    unknown_country_to_na: bool = True


# ======================================================================================
# FEATURE GOVERNANCE
# ======================================================================================

@dataclass(frozen=True)
class FeatureGovernance:
    """
    Governance for 'Structural/Causal' vs 'Predictive' setups.
    """
    # Process / funnel proximity columns: exclude for Structural/Causal case.
    process_proximity_cols: Sequence[str] = field(default_factory=lambda: (
        "Last Activity",
        "Last Notable Activity",
        "Tags",
        "Lead Quality",
    ))

    # Columns to drop entirely from the project (low value / unclear origin)
    drop_always_cols: Sequence[str] = field(default_factory=lambda: (
        "Asymmetrique Activity Index",
        "Asymmetrique Profile Index",
        "Asymmetrique Activity Score",
        "Asymmetrique Profile Score",
        "I agree to pay the amount through cheque",
        "A free copy of Mastering The Interview",
    ))

    # Leakage columns you dropped before modeling (keine IDs! IDs stehen in DataConfig.id_cols)
    leakage_cols: Sequence[str] = field(default_factory=lambda: (
        "Lead Quality",
        "Tags",
        "Lead Profile",
        "Last Notable Activity",
    ))

    # Geo columns (optional)
    geo_cols: Sequence[str] = field(default_factory=lambda: (
        "Country",
        "City",
    ))

    # Structural blocks used in the block contribution analysis
    structural_blocks: Mapping[str, Sequence[str]] = field(default_factory=lambda: {
        "A_Marketing": ("Lead Origin", "Lead Source"),
        "B_Profile": ("What is your current occupation", "Specialization"),
        "C_Behavior": ("TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"),
        "D_Geo": ("Country", "City"),
    })


# ======================================================================================
# FEATURE CATALOG BRIDGE (for feature_catalog.py / multivariate_explainer.py)
# ======================================================================================

@dataclass(frozen=True)
class FeatureCatalogConfig:
    """
    Single source of truth for the catalogs used by the OOP pipeline.
    This prevents duplicated lists across feature_catalog.py and config.py.
    """
    # pseudo-missing tokens (object cols -> NaN)
    pseudo_missing_tokens: Sequence[str] = field(default_factory=lambda: (
        "Select", "select",
        "Not Provided", "not provided",
        "None", "none",
        "", " ",
    ))

    # drop early (raw cleanup)
    drop_columns_raw: Sequence[str] = field(default_factory=lambda: (
        "Asymmetrique Activity Index",
        "Asymmetrique Profile Index",
        "Asymmetrique Activity Score",
        "Asymmetrique Profile Score",
        "I agree to pay the amount through cheque",
        "A free copy of Mastering The Interview",
    ))

    # drop before modeling: IDs + leakage/proximity
    # (IDs kommen aus DataConfig.id_cols; leakage kommt aus FeatureGovernance.leakage_cols)
    # => wird im Code zusammengeführt, hier KEINE Duplikation.

    # structural blocks (for block analysis)
    blocks: Mapping[str, Sequence[str]] = field(default_factory=lambda: {
        "A_Marketing": ("Lead Origin", "Lead Source"),
        "B_Profile": ("What is your current occupation", "Specialization"),
        "C_Behavior": ("TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"),
        "D_Geo": ("Country", "City"),
    })


# ======================================================================================
# MODELING (Logistic Regression explanatory model)
# ======================================================================================

@dataclass(frozen=True)
class MultivariateConfig:
    """Multivariate explanatory model (logistic regression)."""
    enabled: bool = True

    # Train/test
    test_size: float = 0.2
    random_state: int = 42

    # Logistic regression
    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 5000

    # Preprocessing
    # Structural/Causal view: neutralize missingness signal
    categorical_impute_strategy: str = "most_frequent"   # structural default
    numeric_impute_strategy: str = "median"
    scale_numeric: bool = True

    # Predictive view: allow missingness as explicit category if desired
    predictive_categorical_impute_strategy: str = "constant"
    predictive_missing_label: str = "Missing"
    predictive_scale_numeric: bool = False

    # For readable numeric effects (since we scale)
    numeric_effect_increments: Mapping[str, float] = field(default_factory=lambda: {
        "TotalVisits": 5.0,
        "Total Time Spent on Website": 100.0,
        "Page Views Per Visit": 1.0,
    })


# ======================================================================================
# STATS GATES (later)
# ======================================================================================

@dataclass(frozen=True)
class SignificanceConfig:
    """Gate 2: significance testing configuration."""
    alpha: float = 0.05
    
    multiple_testing_method: str = "bh_fdr"
    
    hard_min_n: int = 20


@dataclass(frozen=True)
class RobustnessConfig:
    """Gate 3: robustness checks."""
    
    bootstrap_iterations: int = 200
    bootstrap_sample_frac: float = 0.7
    random_state: int = 42
    
    min_stability_fraction: float = 0.80
    
    time_buckets: int = 6
    

@dataclass(frozen=True)

class DecisionConfig:
    """How to convert evidence into SCALE / INVESTIGATE / STOP."""
    
    min_abs_lift: float = 0.03
    
    require_significant_for_scale: bool = True
    require_robust_for_scale: bool = True
    
    force_do_not_use_if_leakage: bool = True


# ======================================================================================
# TOP-LEVEL CONFIG
# ======================================================================================

@dataclass(frozen=True)
class LeadScoringCaseConfig:
    """
    Top-level config bundling everything for Case 1.
    """
    paths: CasePaths = field(default_factory=CasePaths)
    data: DataConfig = field(default_factory=DataConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    governance: FeatureGovernance = field(default_factory=FeatureGovernance)
    catalog: FeatureCatalogConfig = field(default_factory=FeatureCatalogConfig)
    multivariate: MultivariateConfig = field(default_factory=MultivariateConfig)
    significance: SignificanceConfig = field(default_factory=SignificanceConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    # Optional whitelist for segment analysis (empty => auto-detect)
    categorical_whitelist: Sequence[str] = field(default_factory=tuple)