from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import LeadScoringCaseConfig


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _predict_proba_1(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Return P(y=1) as 1D numpy array for a variety of model APIs.
    Supports sklearn-like predict_proba, statsmodels-like predict, etc.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        arr = np.asarray(proba)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1]
        if arr.ndim == 1:
            return arr
        return arr[:, -1]

    if hasattr(model, "predict"):
        pred = model.predict(X)
        arr = np.asarray(pred)
        if arr.ndim == 1:
            return arr
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, 1]
        return arr.reshape(-1)

    raise TypeError("Model has neither predict_proba nor predict")


def _quantiles(scores: pd.Series, qs: Iterable[float]) -> Dict[float, float]:
    qv = scores.quantile(list(qs))
    out = qv.to_dict()
    return {float(k): float(v) for k, v in out.items()}


def _assign_score_segment(score: float, q: Dict[float, float]) -> str:
    # expected keys: 0.95, 0.90, 0.80, 0.20
    if score >= q[0.95]:
        return f"score >= p95 ({q[0.95]:.3f})"
    if score >= q[0.90]:
        return f"score >= p90 ({q[0.90]:.3f})"
    if score >= q[0.80]:
        return f"score >= p80 ({q[0.80]:.3f})"
    if score <= q[0.20]:
        return f"score <= p20 ({q[0.20]:.3f})"
    return "mid (p20-p80)"


def _safe_bool_series(s: pd.Series) -> pd.Series:
    # avoids dtype warnings and handles missing
    return s.fillna(False).infer_objects(copy=False).astype(bool)


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentExportPaths:
    segments_dir: Path
    leads_high_priority: Path
    leads_low_priority: Path
    segment_playbook: Path
    segment_profiles_csv: Path


@dataclass(frozen=True)
class SegmentRunResult:
    segments_table: pd.DataFrame
    stats: Dict[str, object]
    paths: SegmentExportPaths


def build_segment_exports(*, project_root: Path, cfg: LeadScoringCaseConfig) -> SegmentExportPaths:
    segments_dir = project_root / "results" / "segments"
    _ensure_dir(segments_dir)

    tables_dir = project_root / cfg.paths.tables_dir
    _ensure_dir(tables_dir)

    return SegmentExportPaths(
        segments_dir=segments_dir,
        leads_high_priority=segments_dir / "leads_high_priority.csv",
        leads_low_priority=segments_dir / "leads_low_priority.csv",
        segment_playbook=tables_dir / "segment_playbook.csv",
        segment_profiles_csv=tables_dir / "segment_profiles.csv",
    )


def attach_scores(
    *,
    df_clean: pd.DataFrame,
    X_pred: pd.DataFrame,
    model: Any,
    score_col: str = "score",
) -> pd.DataFrame:
    """
    Adds model score (P(y=1)) to df_clean. Assumes row order matches X_pred.
    """
    scores = _predict_proba_1(model, X_pred)
    scores = np.asarray(scores, dtype=float)
    scores = np.clip(scores, 0.0, 1.0)

    out = df_clean.copy()
    if len(out) != len(scores):
        raise ValueError(f"Row mismatch: df_clean has {len(out)} rows, scores has {len(scores)} rows.")
    out[score_col] = scores
    return out


def _segment_stats(
    df: pd.DataFrame,
    *,
    target_col: str,
    baseline_rate: float,
    segment_label: str,
    segment_type: str,
) -> Dict[str, object]:
    n = int(len(df))
    if n == 0:
        return {
            "segment": segment_label,
            "n": 0,
            "share": 0.0,
            "conversion_rate": np.nan,
            "lift": np.nan,
            "segment_type": segment_type,
        }
    conv = float(df[target_col].astype(float).mean())
    return {
        "segment": segment_label,
        "n": n,
        "share": np.nan,  # filled later when we know total
        "conversion_rate": conv,
        "lift": conv - baseline_rate,
        "segment_type": segment_type,
    }


def build_segment_profiles(
    *,
    df_scored: pd.DataFrame,
    cfg: LeadScoringCaseConfig,
    score_col: str = "score",
    single_feature_segments: Sequence[Tuple[str, str, object]] = (
        ("What is your current occupation", "Working Professional", "single_feature"),
        ("What is your current occupation", "Student", "single_feature"),
        ("Lead Origin", "Landing Page Submission", "single_feature"),
        ("Lead Origin", "API", "single_feature"),
    ),
) -> pd.DataFrame:
    """
    Builds a compact segment profile table and returns it.
    Also writes nothing here (writing is done by run_segment_exports).

    Segments:
      - score >= p95 / p90 / p80 / score <= p20
      - a few single-feature slices (if columns exist)
    """
    target_col = cfg.data.target_col
    if target_col not in df_scored.columns:
        raise ValueError(f"df_scored missing target column '{target_col}'")

    if score_col not in df_scored.columns:
        raise ValueError(f"df_scored missing score column '{score_col}'")

    total_n = int(len(df_scored))
    baseline = float(df_scored[target_col].astype(float).mean()) if total_n > 0 else float("nan")

    scores = df_scored[score_col].astype(float)
    q = _quantiles(scores, qs=[0.95, 0.90, 0.80, 0.20])

    rows: List[Dict[str, object]] = []

    # Score buckets (match your console output style)
    mask_p95 = scores >= q[0.95]
    mask_p90 = scores >= q[0.90]
    mask_p80 = scores >= q[0.80]
    mask_p20 = scores <= q[0.20]

    rows.append(_segment_stats(df_scored.loc[mask_p95], target_col=target_col, baseline_rate=baseline,
                              segment_label=f"score >= p95 ({q[0.95]:.3f})", segment_type="score"))
    rows.append(_segment_stats(df_scored.loc[mask_p90], target_col=target_col, baseline_rate=baseline,
                              segment_label=f"score >= p90 ({q[0.90]:.3f})", segment_type="score"))
    rows.append(_segment_stats(df_scored.loc[mask_p80], target_col=target_col, baseline_rate=baseline,
                              segment_label=f"score >= p80 ({q[0.80]:.3f})", segment_type="score"))
    rows.append(_segment_stats(df_scored.loc[mask_p20], target_col=target_col, baseline_rate=baseline,
                              segment_label=f"score <= p20 ({q[0.20]:.3f})", segment_type="score"))

    # Single feature segments
    for col, val, seg_type in single_feature_segments:
        if col not in df_scored.columns:
            continue
        seg_df = df_scored.loc[df_scored[col].astype(str) == str(val)]
        rows.append(_segment_stats(
            seg_df,
            target_col=target_col,
            baseline_rate=baseline,
            segment_label=f"{col} == {val}",
            segment_type=str(seg_type),
        ))

    out = pd.DataFrame(rows)

    # fill share
    if total_n > 0 and "n" in out.columns:
        out["share"] = out["n"].astype(float) / float(total_n)

    # order + sort: show high lift first, but keep a readable order
    cols_pref = ["segment", "n", "share", "conversion_rate", "lift", "segment_type"]
    cols = [c for c in cols_pref if c in out.columns] + [c for c in out.columns if c not in cols_pref]
    out = out[cols].copy()

    if "lift" in out.columns:
        out = out.sort_values("lift", ascending=False, na_position="last").reset_index(drop=True)

    return out


def export_lead_lists(
    *,
    df_scored: pd.DataFrame,
    cfg: LeadScoringCaseConfig,
    out_paths: SegmentExportPaths,
    id_col_candidates: Tuple[str, ...] = ("Lead Number", "Prospect ID"),
    score_col: str = "score",
    keep_cols: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Writes:
      - leads_high_priority.csv (p90+)
      - leads_low_priority.csv  (p20-)
    Returns quantiles and counts.
    """
    if score_col not in df_scored.columns:
        raise ValueError(f"Missing score column '{score_col}' in df_scored")

    id_col = next((c for c in id_col_candidates if c in df_scored.columns), None)

    scores = df_scored[score_col].astype(float)
    q = _quantiles(scores, qs=[0.95, 0.90, 0.80, 0.20])

    tmp = df_scored.copy()

    # columns to keep
    if keep_cols is None:
        keep_cols = []
        if id_col is not None:
            keep_cols.append(id_col)

        for c in [
            "Lead Origin",
            "Lead Source",
            "What is your current occupation",
            "Total Time Spent on Website",
            "TotalVisits",
            "Do Not Email",
            cfg.data.target_col,
        ]:
            if c in tmp.columns and c not in keep_cols:
                keep_cols.append(c)

        keep_cols += [score_col]

    high = tmp.loc[tmp[score_col] >= q[0.90]].copy().sort_values(score_col, ascending=False)
    low = tmp.loc[tmp[score_col] <= q[0.20]].copy().sort_values(score_col, ascending=True)

    high.to_csv(out_paths.leads_high_priority, index=False, columns=[c for c in keep_cols if c in high.columns])
    low.to_csv(out_paths.leads_low_priority, index=False, columns=[c for c in keep_cols if c in low.columns])

    return {
        "id_col": id_col,
        "q": q,
        "n_high": int(len(high)),
        "n_low": int(len(low)),
    }


def build_segment_playbook(
    *,
    segment_profiles: pd.DataFrame,
    compare_df: Optional[pd.DataFrame],
    out_path: Path,
) -> pd.DataFrame:
    """
    Turns segment_profiles into a business playbook table.
    Adds:
      - recommended_action
      - caution (if segment mentions any 'overestimated' feature)
    """
    sp: pd.DataFrame = segment_profiles.copy()
    if "segment" not in sp.columns:
        raise ValueError("segment_profiles needs a 'segment' column")

    # lift column (ensure float series)
    lift: pd.Series
    if "lift" in sp.columns:
        lift = sp["lift"].astype(float)
    else:
        lift = pd.Series(np.nan, index=sp.index, dtype=float)

    def _action_from_lift(x: object) -> str:
        xf = float(x) if x is not None else float("nan")
        if np.isnan(xf):
            return "INVESTIGATE"
        if xf >= 0.15:
            return "SCALE"
        if xf >= 0.05:
            return "INVESTIGATE"
        if xf <= -0.10:
            return "STOP"
        return "INVESTIGATE"

    sp["recommended_action"] = lift.apply(_action_from_lift)

    # caution: segment text mentions an overestimated feature name
    caution = pd.Series("", index=sp.index, dtype=str)
    if compare_df is not None and "overestimated" in compare_df.columns and "feature" in compare_df.columns:
        over_mask = _safe_bool_series(compare_df["overestimated"])
        over_feats = compare_df.loc[over_mask, "feature"].dropna().astype(str).tolist()

        def _caution(seg: object) -> str:
            s = str(seg)
            for f in over_feats:
                if f and f in s:
                    return "contains_overestimated_feature"
            return ""

        caution = sp["segment"].astype(str).apply(_caution)

    sp["caution"] = caution

    cols_pref = ["segment", "segment_type", "n", "share", "conversion_rate", "lift", "recommended_action", "caution"]
    cols = [c for c in cols_pref if c in sp.columns] + [c for c in sp.columns if c not in cols_pref]
    sp = sp[cols].copy()

    sp.to_csv(out_path, index=False)
    return sp


def run_segment_exports(
    *,
    project_root: Path,
    cfg: LeadScoringCaseConfig,
    df_clean: pd.DataFrame,
    X_pred: pd.DataFrame,
    pred_model: Any,
    compare_df: Optional[pd.DataFrame] = None,
    export_lists: bool = True,
) -> SegmentRunResult:
    """
    One-call convenience:
      - attach score to df_clean (via pred_model + X_pred)
      - build & write segment_profiles.csv
      - export lead lists (optional)
      - write segment_playbook.csv
    """
    paths = build_segment_exports(project_root=project_root, cfg=cfg)

    df_scored = attach_scores(df_clean=df_clean, X_pred=X_pred, model=pred_model, score_col="score")

    segments_table = build_segment_profiles(df_scored=df_scored, cfg=cfg, score_col="score")
    segments_table.to_csv(paths.segment_profiles_csv, index=False)

    stats: Dict[str, object] = {
        "segment_profiles_path": str(paths.segment_profiles_csv),
    }

    if export_lists:
        stats_lists = export_lead_lists(df_scored=df_scored, cfg=cfg, out_paths=paths, score_col="score")
        stats.update(stats_lists)

    # always write playbook (it’s cheap) if segment table exists
    build_segment_playbook(segment_profiles=segments_table, compare_df=compare_df, out_path=paths.segment_playbook)
    stats.update({
        "leads_high_priority_path": str(paths.leads_high_priority),
        "leads_low_priority_path": str(paths.leads_low_priority),
        "segment_playbook_path": str(paths.segment_playbook),
    })

    return SegmentRunResult(segments_table=segments_table, stats=stats, paths=paths)
