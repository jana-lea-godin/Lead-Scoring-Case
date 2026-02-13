from __future__ import annotations

import numpy as np
import pandas as pd


def stability_from_bootstrap(
    effects_table: pd.DataFrame,
    bootstrap_coefs: np.ndarray,
) -> pd.DataFrame:
    """
    Adds stability metrics based on bootstrap coefficients.

    stability_sign: fraction of bootstrap samples with same sign as trained coef
    stability_nonzero: fraction of bootstrap samples where coef != 0 (usually ~1 for LR)
    """
    if "coef_logit" not in effects_table.columns:
        raise KeyError("effects_table must include 'coef_logit'.")

    coefs = effects_table["coef_logit"].to_numpy(dtype=float)

    b = np.asarray(bootstrap_coefs, dtype=float)
    if b.ndim != 2 or b.shape[1] != coefs.shape[0]:
        raise ValueError(f"bootstrap_coefs must have shape (B, {coefs.shape[0]}). Got {b.shape}.")

    # sign stability: share of samples matching trained sign
    sign_ref = np.sign(coefs)
    sign_b = np.sign(b)

    # if trained coef is exactly 0, define stability as NaN
    same_sign = (sign_b == sign_ref[None, :]).astype(float)
    stability_sign = same_sign.mean(axis=0)

    stability_nonzero = (b != 0.0).mean(axis=0)

    out = effects_table.copy()
    out["stability_sign"] = stability_sign
    out["stability_nonzero"] = stability_nonzero
    return out