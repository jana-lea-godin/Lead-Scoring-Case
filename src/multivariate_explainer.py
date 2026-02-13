from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import LeadScoringCaseConfig
from .feature_catalog import FeatureCatalog, build_feature_catalog

# ======================================================================================
# SMALL PARAM OBJECT
# ======================================================================================

@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    auc: float
    report: str
    categorical_cols: List[str]
    numeric_cols: List[str]


# ======================================================================================
# HELPERS
# ======================================================================================

def _safe_drop(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing).copy()


def _available_cols(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def _split_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target '{target_col}' not found in DataFrame.")
    y = df[target_col].copy()
    X = df.drop(columns=[target_col]).copy()
    return X, y


# ======================================================================================
# PREPROCESSOR (Notebook cleaning logic, but without EDA/prints)
# ======================================================================================

class LeadScoringPreprocessor:
    def __init__(self, cfg: LeadScoringCaseConfig, catalog: Optional[FeatureCatalog] = None):
        self.cfg = cfg
        self.catalog = catalog or build_feature_catalog(cfg)

    def reduce_raw(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        # Drop always cols early
        return _safe_drop(df_raw, self.catalog.drop_columns_raw)

    def convert_pseudo_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pseudo-missing tokens to np.nan (NOT pd.NA),
        because sklearn can't reliably handle pd.NA.
        """
        df_out = df.copy()
        tokens = list(self.catalog.pseudo_missing_tokens)

        for col in df_out.columns:
            if df_out[col].dtype == "object":
                df_out[col] = df_out[col].replace(tokens, np.nan)

        return df_out

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize string columns without turning them into pandas StringDtype
        (StringDtype can introduce pd.NA, which sklearn dislikes).
        """
        df_out = df.copy()

        # optional strip whitespace for all object columns
        if self.cfg.cleaning.strip_object_whitespace:
            obj_cols = df_out.select_dtypes(include=["object"]).columns
            for c in obj_cols:
                # keep as object; strip safely
                df_out[c] = df_out[c].astype(object)
                df_out[c] = df_out[c].where(df_out[c].isna(), df_out[c].astype(str).str.strip())

        # notebook: Lead Source -> title case
        if self.cfg.cleaning.normalize_lead_source_titlecase and "Lead Source" in df_out.columns:
            s = df_out["Lead Source"].astype(object)
            df_out["Lead Source"] = s.where(s.isna(), s.astype(str).str.title())

        # notebook: Country "unknown" -> NA (use np.nan)
        if self.cfg.cleaning.unknown_country_to_na and "Country" in df_out.columns:
            df_out["Country"] = df_out["Country"].replace("unknown", np.nan)

        # final safety: ensure no pd.NA remains
        df_out = df_out.replace({pd.NA: np.nan})

        return df_out

    def build_clean(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = self.reduce_raw(df_raw)
        df = self.convert_pseudo_missing(df)
        df = self.normalize(df)
        return df

    def tracking_split(self, df_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split into (no_tracking, with_tracking).
        Decision rule: "TotalVisits is NaN" (same as notebook).
        """
        if "TotalVisits" not in df_clean.columns:
            raise KeyError("Column 'TotalVisits' not found; cannot create tracking split.")
        mask = df_clean["TotalVisits"].isna()
        return df_clean.loc[mask].copy(), df_clean.loc[~mask].copy()

    def build_model_df(self, df_clean: pd.DataFrame) -> pd.DataFrame:
        """
        Remove IDs + leakage columns before modeling.
        """
        return _safe_drop(df_clean, self.catalog.drop_columns_before_modeling)


# ======================================================================================
# MODEL BUILDING
# ======================================================================================

def _build_preprocessor(
    X: pd.DataFrame,
    *,
    cat_impute_strategy: str,
    cat_fill_value: str,
    num_impute_strategy: str,
    scale_numeric: bool,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()

    # categorical pipeline
    if cat_impute_strategy == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    elif cat_impute_strategy == "constant":
        cat_imputer = SimpleImputer(strategy="constant", fill_value=cat_fill_value)
    else:
        raise ValueError("cat_impute_strategy must be 'most_frequent' or 'constant'.")

    cat_pipeline = Pipeline(steps=[
        ("imputer", cat_imputer),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ])

    # numeric pipeline
    num_steps = [("imputer", SimpleImputer(strategy=num_impute_strategy))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipeline = Pipeline(steps=num_steps)

    pre = ColumnTransformer(
        transformers=[
            ("cat", cat_pipeline, categorical_cols),
            ("num", num_pipeline, numeric_cols),
        ]
    )
    return pre, categorical_cols, numeric_cols


def train_logit(
    df_model: pd.DataFrame,
    cfg: LeadScoringCaseConfig,
    *,
    structural: bool = True,
) -> TrainResult:
    """
    One clean training function.
    - structural=True: uses cfg.multivariate structural defaults (most_frequent + scaling)
    - structural=False: predictive defaults (constant 'Missing' + no scaling)
    """
    target = cfg.data.target_col
    X, y = _split_xy(df_model, target_col=target)

    # --- SAFETY: sklearn cannot handle pd.NA reliably ---
    X = X.replace({pd.NA: np.nan})

    # If any pandas StringDtype columns slipped in, force them back to object
    for c in X.columns:
        if pd.api.types.is_string_dtype(X[c]):
            X[c] = X[c].astype("object")

    mv = cfg.multivariate

    if structural:
        cat_impute = mv.categorical_impute_strategy
        cat_fill = "Missing"  # not used for most_frequent, but harmless
        scale_num = mv.scale_numeric
    else:
        cat_impute = mv.predictive_categorical_impute_strategy
        cat_fill = mv.predictive_missing_label
        scale_num = mv.predictive_scale_numeric

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mv.test_size,
        random_state=mv.random_state,
        stratify=y,
    )

    pre, cat_cols, num_cols = _build_preprocessor(
        X_train,
        cat_impute_strategy=cat_impute,
        cat_fill_value=cat_fill,
        num_impute_strategy=mv.numeric_impute_strategy,
        scale_numeric=scale_num,
    )

    clf = LogisticRegression(
        penalty=mv.penalty,
        C=mv.C,
        solver=mv.solver,
        max_iter=mv.max_iter,
    )

    model = Pipeline(steps=[
        ("preprocessing", pre),
        ("classifier", clf),
    ])

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    auc = float(roc_auc_score(y_test, proba))
    report = classification_report(y_test, pred, zero_division=0)

    return TrainResult(
        model=model,
        auc=auc,
        report=report,
        categorical_cols=cat_cols,
        numeric_cols=num_cols,
    )


# ======================================================================================
# INTERPRETATION
# ======================================================================================

def top_coefficients(
    trained: TrainResult,
    *,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Return top coefficients with odds ratios (by absolute effect).
    """
    model = trained.model
    pre: ColumnTransformer = model.named_steps["preprocessing"]
    clf: LogisticRegression = model.named_steps["classifier"]

    cat_pipe: Pipeline = pre.named_transformers_["cat"]
    ohe: OneHotEncoder = cat_pipe.named_steps["ohe"]

    cat_feature_names = ohe.get_feature_names_out(trained.categorical_cols)
    num_feature_names = np.array(trained.numeric_cols, dtype=object)
    feature_names = np.concatenate([cat_feature_names, num_feature_names])

    coefs = clf.coef_.ravel()

    df = pd.DataFrame({
        "feature": feature_names,
        "coef_logit": coefs,
        "odds_ratio": np.exp(coefs),
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return df.head(top_n).reset_index(drop=True)


# ======================================================================================
# STRUCTURAL / BLOCK CONTRIBUTION
# ======================================================================================

def build_structural_df(df_model: pd.DataFrame, cfg: LeadScoringCaseConfig) -> pd.DataFrame:
    """
    Structural view: remove process proximity columns (from governance).
    """
    target = cfg.data.target_col
    df = _safe_drop(df_model, cfg.governance.process_proximity_cols)

    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found after structural drop.")
    return df


def auc_for_features(
    df_struct: pd.DataFrame,
    cfg: LeadScoringCaseConfig,
    feature_cols: List[str],
) -> float:
    """
    Structural AUC for a subset of features.
    """
    target = cfg.data.target_col
    cols = [target] + feature_cols
    df_sub = df_struct[[c for c in cols if c in df_struct.columns]].copy()
    res = train_logit(df_sub, cfg, structural=True)
    return float(res.auc)


def block_contribution_analysis(
    df_model: pd.DataFrame,
    cfg: LeadScoringCaseConfig,
) -> Dict[str, object]:
    """
    Returns:
    - full structural auc
    - geo drop deltas
    - auc per block alone
    - incremental auc across blocks
    """
    catalog = build_feature_catalog(cfg)
    target = cfg.data.target_col

    df_struct = build_structural_df(df_model, cfg)
    all_feats = [c for c in df_struct.columns if c != target]

    auc_full = auc_for_features(df_struct, cfg, all_feats)

    # geo drop tests
    geo = list(catalog.geo_cols)
    cols_no_geo = [c for c in all_feats if c not in geo]
    auc_no_geo = auc_for_features(df_struct, cfg, cols_no_geo)

    # blocks (available only)
    blocks_clean = {k: _available_cols(df_struct, v) for k, v in catalog.blocks.items()}

    # block alone
    alone = []
    for name, cols in blocks_clean.items():
        if not cols:
            continue
        alone.append({
            "block": name,
            "auc": auc_for_features(df_struct, cfg, cols),
            "n_features": len(cols),
            "cols": cols,
        })
    alone = sorted(alone, key=lambda d: d["auc"], reverse=True)

    # incremental A -> A+B -> ...
    order = ["A_Marketing", "B_Profile", "C_Behavior", "D_Geo"]
    cum_cols: List[str] = []
    incremental = []
    prev = None

    for blk in order:
        cum_cols += blocks_clean.get(blk, [])
        if not cum_cols:
            continue
        auc = auc_for_features(df_struct, cfg, cum_cols)
        incremental.append({
            "up_to_block": blk,
            "auc": auc,
            "delta_vs_prev": None if prev is None else (auc - prev),
            "total_features": len(cum_cols),
        })
        prev = auc

    return {
        "structural_shape": df_struct.shape,
        "auc_struct_full": auc_full,
        "auc_no_geo": auc_no_geo,
        "delta_drop_geo": auc_full - auc_no_geo,
        "blocks_available": blocks_clean,
        "block_alone": alone,
        "incremental": incremental,
    }