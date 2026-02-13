from __future__ import annotations

import numpy as np
import pandas as pd


def compare_structural_vs_predictive(
    structural_effects: pd.DataFrame,
    predictive_effects: pd.DataFrame,
    *,
    feature_col: str = "feature",
) -> pd.DataFrame:
    """
    Join two effect tables (from build_effect_table_from_trained_model).
    Returns a comparison table with deltas and flags.

    We compare on the expanded feature names (OHE feature names).
    """

    s = structural_effects.copy()
    p = predictive_effects.copy()

    # keep only relevant columns, but don't fail if some are missing
    s_cols = [
        c for c in ["feature", "coef_logit", "odds_ratio", "q_value", "significant", "stability_sign", "decision"]
        if c in s.columns
    ]
    p_cols = [
        c for c in ["feature", "coef_logit", "odds_ratio", "q_value", "significant", "stability_sign", "decision"]
        if c in p.columns
    ]

    s = s[s_cols].rename(columns={
        "coef_logit": "coef_struct",
        "odds_ratio": "or_struct",
        "q_value": "q_struct",
        "significant": "sig_struct",
        "stability_sign": "stab_struct",
        "decision": "dec_struct",
    })

    p = p[p_cols].rename(columns={
        "coef_logit": "coef_pred",
        "odds_ratio": "or_pred",
        "q_value": "q_pred",
        "significant": "sig_pred",
        "stability_sign": "stab_pred",
        "decision": "dec_pred",
    })

    df = s.merge(p, on=feature_col, how="outer")

    # ---------- Safe numeric conversion ----------
    # (avoids dtype surprises and keeps NaN where missing)
    df["coef_struct"] = pd.to_numeric(df.get("coef_struct"), errors="coerce")
    df["coef_pred"] = pd.to_numeric(df.get("coef_pred"), errors="coerce")
    df["or_struct"] = pd.to_numeric(df.get("or_struct"), errors="coerce")
    df["or_pred"] = pd.to_numeric(df.get("or_pred"), errors="coerce")

    # deltas (work in log-odds space)
    df["delta_coef_pred_minus_struct"] = df["coef_pred"] - df["coef_struct"]

    # Also compare OR in log space (more symmetric)
    # protect log from 0 / negative / NaN
    eps = 1e-12
    log_or_struct = np.log(np.clip(df["or_struct"].to_numpy(dtype=float, copy=False), eps, np.inf))
    log_or_pred = np.log(np.clip(df["or_pred"].to_numpy(dtype=float, copy=False), eps, np.inf))

    df["log_or_struct"] = log_or_struct
    df["log_or_pred"] = log_or_pred
    df["delta_log_or_pred_minus_struct"] = df["log_or_pred"] - df["log_or_struct"]

    # ---------- Overestimated heuristic ----------
    # - predictive says strong + (sig or SCALE/INVESTIGATE)
    # - structural says NOT significant and/or STOP
    # - AND absolute delta is non-trivial
    df["overestimated"] = False

    # IMPORTANT: avoid FutureWarning by infer_objects(copy=False) before astype(bool)
    tmp_sig_pred = df.get("sig_pred", pd.Series(False, index=df.index)).fillna(False)
    sig_pred = tmp_sig_pred.infer_objects(copy=False).astype(bool)

    tmp_sig_struct = df.get("sig_struct", pd.Series(False, index=df.index)).fillna(False)
    sig_struct = tmp_sig_struct.infer_objects(copy=False).astype(bool)

    # decisions as strings (safe)
    dec_pred = df.get("dec_pred", pd.Series("", index=df.index)).fillna("").astype(str)
    dec_struct = df.get("dec_struct", pd.Series("", index=df.index)).fillna("").astype(str)

    pred_strong = sig_pred | dec_pred.isin(["SCALE", "INVESTIGATE"])
    struct_weak = (~sig_struct) | dec_struct.isin(["STOP"])

    # big gap threshold in log-OR space: 0.25 ~ OR 1.28
    big_gap = df["delta_log_or_pred_minus_struct"].abs() >= 0.25

    df.loc[pred_strong & struct_weak & big_gap, "overestimated"] = True

    # Nice ordering: most overestimated first by gap size
    df["abs_gap"] = df["delta_log_or_pred_minus_struct"].abs()
    df = df.sort_values(["overestimated", "abs_gap"], ascending=[False, False]).reset_index(drop=True)

    return df
