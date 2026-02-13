from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .config import LeadScoringCaseConfig


@dataclass(frozen=True)

class FeatureCatalog:
    """
    Single source of truth for feature lists used by preprocessing + multivariate explainer.
    Built FROM config.py to avoid duplicated lists across files.
    """
    target_col: str

    # Cleaning / early drop
    drop_columns_raw: Sequence[str]
    pseudo_missing_tokens: Sequence[str]

    # Before modeling: remove IDs + leakage columns
    drop_columns_before_modeling: Sequence[str]

    # Structural/Causal: process proximity cols
    process_cols: Sequence[str]

    # Blocks used in block contribution analysis
    blocks: Mapping[str, Sequence[str]]

    # Optional (used in geo-drop tests)
    geo_cols: Sequence[str]


def build_feature_catalog(cfg: LeadScoringCaseConfig) -> FeatureCatalog:
    """
    Build catalog from config.py (no duplication).
    """
    # IDs + leakage are removed before modeling
    drop_before_model = tuple(cfg.data.id_cols) + tuple(cfg.governance.leakage_cols)

    return FeatureCatalog(
        target_col=cfg.data.target_col,
        drop_columns_raw=tuple(cfg.governance.drop_always_cols),
        pseudo_missing_tokens=tuple(cfg.cleaning.missing_tokens),
        drop_columns_before_modeling=drop_before_model,
        process_cols=tuple(cfg.governance.process_proximity_cols),
        blocks=dict(cfg.governance.structural_blocks),
        geo_cols=tuple(cfg.governance.geo_cols),
    )