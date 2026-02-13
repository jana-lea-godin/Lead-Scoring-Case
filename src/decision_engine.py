from __future__ import annotations

import pandas as pd


def decide_actions(
    effects: pd.DataFrame,
    *,
    alpha: float,
    min_abs_lift: float,
    min_stability_fraction: float,
) -> pd.DataFrame:
    """
    Convert evidence into SCALE / INVESTIGATE / STOP.

    We use:
    - significant (q_value < alpha)
    - abs effect size via odds_ratio distance from 1
    - stability_sign >= min_stability_fraction
    """

    df = effects.copy()

    # Safety: if missing columns, create them
    if "q_value" not in df.columns:
        df["q_value"] = pd.NA
    if "significant" not in df.columns:
        df["significant"] = False
    if "stability_sign" not in df.columns:
        df["stability_sign"] = pd.NA

    # effect size proxy: absolute lift in odds ratio space
    # (e.g. OR=1.2 -> +0.2, OR=0.8 -> +0.2)
    df["abs_or_lift"] = (df["odds_ratio"] - 1.0).abs()

    is_big = df["abs_or_lift"] >= float(min_abs_lift)
    is_sig = df["q_value"].astype(float) < float(alpha)
    is_stable = df["stability_sign"].astype(float) >= float(min_stability_fraction)

    # Decision rules
    df["decision"] = "STOP"
    df.loc[is_big & (is_sig | is_stable), "decision"] = "INVESTIGATE"
    df.loc[is_big & is_sig & is_stable, "decision"] = "SCALE"

    # Nice ordering
    keep_cols = [
        "feature",
        "odds_ratio",
        "ci_low_or",
        "ci_high_or",
        "p_value",
        "q_value",
        "significant",
        "stability_sign",
        "abs_or_lift",
        "decision",
        "coef_logit",
    ]
    cols = [c for c in keep_cols if c in df.columns] + [c for c in df.columns if c not in keep_cols]
    df = df[cols]

    return df.sort_values(["decision", "abs_or_lift"], ascending=[True, False]).reset_index(drop=True)