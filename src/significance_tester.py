from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class EffectTable:
    table: pd.DataFrame


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction; returns q-values in original order."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _get_feature_names_from_model(model: Pipeline) -> np.ndarray:
    """
    Extract OHE + numeric feature names from our sklearn pipeline.
    Assumes steps: preprocessing (ColumnTransformer), classifier (LogisticRegression).
    """
    pre: ColumnTransformer = model.named_steps["preprocessing"]
    clf: LogisticRegression = model.named_steps["classifier"]

    # cat transformer
    cat_pipe = pre.named_transformers_["cat"]
    ohe = cat_pipe.named_steps["ohe"]

    # columns used for cat/num
    cat_cols = pre.transformers_[0][2]
    num_cols = pre.transformers_[1][2]

    cat_names = ohe.get_feature_names_out(cat_cols)
    num_names = np.array(list(num_cols), dtype=object)

    names = np.concatenate([cat_names, num_names])

    if names.shape[0] != clf.coef_.ravel().shape[0]:
        raise ValueError("Feature name length does not match coefficient length.")
    return names


def _extract_coefs(model: Pipeline) -> Tuple[np.ndarray, float]:
    clf: LogisticRegression = model.named_steps["classifier"]
    coefs = clf.coef_.ravel().astype(float)
    intercept = float(clf.intercept_[0])
    return coefs, intercept


def build_effect_table_from_trained_model(
    trained_model: Pipeline,
    cfg,
    *,
    bootstrap_coefs: Optional[np.ndarray] = None,
) -> EffectTable:
    """
    Create a human-readable effect table for a trained sklearn logistic regression pipeline.

    If bootstrap_coefs is provided, it must have shape (B, n_features) aligned with the
    trained model's coefficient vector (fixed feature space).
    """
    feature_names = _get_feature_names_from_model(trained_model)
    coefs, _ = _extract_coefs(trained_model)

    df = pd.DataFrame({
        "feature": feature_names,
        "coef_logit": coefs,
        "odds_ratio": np.exp(coefs),
    })

    if bootstrap_coefs is not None:
        b = np.asarray(bootstrap_coefs, dtype=float)

        if b.ndim != 2 or b.shape[1] != coefs.shape[0]:
            raise ValueError(
                f"bootstrap_coefs must be (B, {coefs.shape[0]}). Got {b.shape}."
            )

        lo = np.quantile(b, 0.025, axis=0)
        hi = np.quantile(b, 0.975, axis=0)

        df["ci_low_or"] = np.exp(lo)
        df["ci_high_or"] = np.exp(hi)

        # Approx. two-sided p-value via bootstrap mass around 0
        p_left = np.mean(b <= 0.0, axis=0)
        p_right = np.mean(b >= 0.0, axis=0)
        pvals = 2.0 * np.minimum(p_left, p_right)
        pvals = np.clip(pvals, 0.0, 1.0)

        df["p_value"] = pvals
        df["q_value"] = _bh_fdr(pvals)

        alpha = cfg.significance.alpha
        df["significant"] = df["q_value"] < alpha
    else:
        df["ci_low_or"] = np.nan
        df["ci_high_or"] = np.nan
        df["p_value"] = np.nan
        df["q_value"] = np.nan
        df["significant"] = False

    # convenience: sort by absolute effect
    df["abs_coef"] = np.abs(df["coef_logit"])
    df = df.sort_values("abs_coef", ascending=False).reset_index(drop=True)

    return EffectTable(table=df)


def bootstrap_logit_coefficients_fixed_preprocessing(
    trained_model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_bootstrap: int,
    sample_frac: float,
    random_state: int,
) -> np.ndarray:
    """
    Bootstrap coefficients while keeping the one-hot space FIXED.

    We reuse the already-fitted preprocessing pipeline from trained_model
    and refit only LogisticRegression on the transformed matrix.

    Returns array (B, n_features) aligned with trained_model.coef_.
    """
    rng = np.random.default_rng(random_state)

    pre = trained_model.named_steps["preprocessing"]  # fitted ColumnTransformer
    base_clf: LogisticRegression = trained_model.named_steps["classifier"]

    n = len(X)
    n_draw = max(1, int(n * sample_frac))

    # Transform FULL X once? (optional). We'll transform per bootstrap to keep memory low.
    coefs_list = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n_draw, replace=True)
        Xb = X.iloc[idx]
        yb = y.iloc[idx]

        Xb_t = pre.transform(Xb)

        clf = LogisticRegression(
            penalty=base_clf.penalty,
            C=base_clf.C,
            solver=base_clf.solver,
            max_iter=base_clf.max_iter,
            class_weight=base_clf.class_weight,
            fit_intercept=base_clf.fit_intercept,
        )
        clf.fit(Xb_t, yb)

        coefs_list.append(clf.coef_.ravel().astype(float))

    return np.vstack(coefs_list)