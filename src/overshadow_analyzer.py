from __future__ import annotations

import numpy as np
import pandas as pd


def _as_bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """
    Future-proof conversion to bool Series.
    Handles missing column, NaNs, object dtype, pandas BooleanDtype.
    """
    s = df.get(col, pd.Series(default, index=df.index))
    s = s.infer_objects(copy=False)

    # If it's already boolean-like, keep it. Otherwise coerce carefully.
    # Using BooleanDtype first avoids FutureWarning edge cases.
    try:
        s = s.astype("boolean")
    except Exception:
        # Fallback for weird objects: map common truthy strings/numbers
        s = s.map(lambda v: True if str(v).strip().lower() in {"true", "1", "yes"} else False)

    s = s.fillna(default).astype(bool)
    return s


def _as_str_series(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """Safe string series (no FutureWarning)."""
    s = df.get(col, pd.Series(default, index=df.index))
    s = s.infer_objects(copy=False).fillna(default).astype(str)
    return s


def compare_structural_vs_predictive(
    structural_effects: pd.DataFrame,
    predictive_effects: pd.DataFrame,
    *,
    feature_col: str = "feature",
) -> pd.DataFrame:
    """
    Join two effect tables (typically evidence tables after Gate 3) and compute deltas + flags.

    Output columns include:
    - coef_struct / coef_pred
    - or_struct / or_pred
    - q_struct / q_pred
    - sig_struct / sig_pred
    - stab_struct / stab_pred
    - dec_struct / dec_pred
    - delta_* and overestimated flag

    Comparison is done on expanded (OHE) feature names.
    """

    s = structural_effects.copy()
    p = predictive_effects.copy()

    # Keep only columns that might exist (robust to schema changes)
    keep = [feature_col, "coef_logit", "odds_ratio", "q_value", "significant", "stability_sign", "decision"]

    s_cols = [c for c in keep if c in s.columns]
    p_cols = [c for c in keep if c in p.columns]

    if feature_col not in s_cols:
        raise KeyError(f"structural_effects must contain '{feature_col}' column.")
    if feature_col not in p_cols:
        raise KeyError(f"predictive_effects must contain '{feature_col}' column.")

    s = s[s_cols].rename(
        columns={
            "coef_logit": "coef_struct",
            "odds_ratio": "or_struct",
            "q_value": "q_struct",
            "significant": "sig_struct",
            "stability_sign": "stab_struct",
            "decision": "dec_struct",
        }
    )

    p = p[p_cols].rename(
        columns={
            "coef_logit": "coef_pred",
            "odds_ratio": "or_pred",
            "q_value": "q_pred",
            "significant": "sig_pred",
            "stability_sign": "stab_pred",
            "decision": "dec_pred",
        }
    )

    df = s.merge(p, on=feature_col, how="outer")

    # ---------- Safe numeric conversion ----------
    for c in ["coef_struct", "coef_pred", "or_struct", "or_pred", "q_struct", "q_pred", "stab_struct", "stab_pred"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---------- Deltas (logit & log(OR)) ----------
    df["delta_coef_pred_minus_struct"] = df.get("coef_pred") - df.get("coef_struct")

    eps = 1e-12
    or_struct = pd.to_numeric(df.get("or_struct"), errors="coerce").to_numpy(dtype=float, copy=False)
    or_pred = pd.to_numeric(df.get("or_pred"), errors="coerce").to_numpy(dtype=float, copy=False)

    log_or_struct = np.log(np.clip(or_struct, eps, np.inf))
    log_or_pred = np.log(np.clip(or_pred, eps, np.inf))

    df["log_or_struct"] = log_or_struct
    df["log_or_pred"] = log_or_pred
    df["delta_log_or_pred_minus_struct"] = df["log_or_pred"] - df["log_or_struct"]

    # ---------- Overestimated heuristic ----------
    # predictive strong, structural weak, and meaningful gap
    sig_pred = _as_bool_series(df, "sig_pred", default=False)
    sig_struct = _as_bool_series(df, "sig_struct", default=False)

    dec_pred = _as_str_series(df, "dec_pred", default="")
    dec_struct = _as_str_series(df, "dec_struct", default="")

    pred_strong = sig_pred | dec_pred.isin(["SCALE", "INVESTIGATE"])
    struct_weak = (~sig_struct) | dec_struct.isin(["STOP"])

    # big gap threshold in log-OR space: 0.25 ~ OR 1.28
    big_gap = df["delta_log_or_pred_minus_struct"].abs() >= 0.25

    df["overestimated"] = (pred_strong & struct_weak & big_gap).fillna(False).astype(bool)

    # Nice ordering: most overestimated first by gap size
    df["abs_gap"] = df["delta_log_or_pred_minus_struct"].abs()
    df = df.sort_values(["overestimated", "abs_gap"], ascending=[False, False]).reset_index(drop=True)

    return df
