# src/report_writer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

from .config import LeadScoringCaseConfig

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ModuleNotFoundError:
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False


@dataclass(frozen=True)
class ReportPaths:
    report_md: Path
    figures_dir: Path


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _try_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _fmt_pct(x: object) -> str:
    try:
        if x is None:
            return "n/a"
        xf = float(x)
        if np.isnan(xf):
            return "n/a"
        return f"{100.0 * xf:.1f}%"
    except Exception:
        return "n/a"


def _fmt_float(x: object, nd: int = 4) -> str:
    try:
        if x is None:
            return "n/a"
        xf = float(x)
        if np.isnan(xf):
            return "n/a"
        return f"{xf:.{nd}f}"
    except Exception:
        return "n/a"


def _save_fig(fig, path: Path) -> None:
    if not _HAS_MPL:
        return
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)  # type: ignore[arg-type]


def _plot_top_effects(
    effects_struct: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    """Bar chart of abs_coef for strongest (structural) effects."""
    if not _HAS_MPL:
        return

    df = effects_struct.copy()

    if "significant" in df.columns:
        sig_mask = df["significant"].astype("boolean").fillna(False).astype(bool)
        df_sig = df.loc[sig_mask].copy()
        if len(df_sig) >= 5:
            df = df_sig

    if "abs_coef" not in df.columns and "coef_logit" in df.columns:
        df["abs_coef"] = df["coef_logit"].astype(float).abs()

    if "feature" not in df.columns or "abs_coef" not in df.columns:
        return

    df = df.sort_values("abs_coef", ascending=False).head(top_n).iloc[::-1]

    fig = plt.figure(figsize=(10, 6))  # type: ignore[union-attr]
    ax = fig.add_subplot(111)
    ax.barh(df["feature"].astype(str), df["abs_coef"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("|logit coef|")
    _save_fig(fig, out_path)


def _plot_overestimated(
    compare_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    if not _HAS_MPL:
        return
    df = compare_df.copy()
    if "overestimated" not in df.columns:
        return

    mask = df["overestimated"].astype("boolean").fillna(False).astype(bool)
    df = df.loc[mask].copy()
    if df.empty:
        return

    if "abs_gap" not in df.columns and "delta_log_or_pred_minus_struct" in df.columns:
        df["abs_gap"] = df["delta_log_or_pred_minus_struct"].astype(float).abs()

    if "feature" not in df.columns or "abs_gap" not in df.columns:
        return

    df = df.sort_values("abs_gap", ascending=False).head(top_n).iloc[::-1]

    fig = plt.figure(figsize=(10, 6))  # type: ignore[union-attr]
    ax = fig.add_subplot(111)
    ax.barh(df["feature"].astype(str), df["abs_gap"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("|Δ log(OR) predictive - structural|")
    _save_fig(fig, out_path)


def _plot_underestimated(
    compare_df: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 15,
) -> None:
    """Underestimated = structural strong, predictive weak (heuristic)."""
    if not _HAS_MPL:
        return
    df = compare_df.copy()
    if "overestimated" not in df.columns:
        return

    over = df["overestimated"].astype("boolean").fillna(False).astype(bool)
    mask = ~over

    if "sig_struct" in df.columns:
        sigs = df["sig_struct"].astype("boolean").fillna(False).astype(bool)
        mask = mask & sigs

    if "abs_gap" not in df.columns and "delta_log_or_pred_minus_struct" in df.columns:
        df["abs_gap"] = df["delta_log_or_pred_minus_struct"].astype(float).abs()

    df = df.loc[mask].copy()
    if df.empty or "abs_gap" not in df.columns or "feature" not in df.columns:
        return

    df = df.sort_values("abs_gap", ascending=False).head(top_n).iloc[::-1]

    fig = plt.figure(figsize=(10, 6))  # type: ignore[union-attr]
    ax = fig.add_subplot(111)
    ax.barh(df["feature"].astype(str), df["abs_gap"].astype(float))
    ax.set_title(title)
    ax.set_xlabel("|Δ log(OR)| (large gap, structural stronger)")
    _save_fig(fig, out_path)


def _plot_segment_lift(
    segment_profiles: pd.DataFrame,
    out_path: Path,
    title: str,
    top_n: int = 20,
) -> None:
    if not _HAS_MPL:
        return
    df = segment_profiles.copy()

    if "lift" not in df.columns or "segment" not in df.columns:
        return

    df = df.dropna(subset=["lift"]).copy()
    if df.empty:
        return

    df = df.sort_values("lift", ascending=False).head(top_n).iloc[::-1]

    fig = plt.figure(figsize=(10, 7))  # type: ignore[union-attr]
    ax = fig.add_subplot(111)
    ax.barh(df["segment"].astype(str), df["lift"].astype(float))
    ax.axvline(0.0, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Lift vs baseline conversion rate")
    _save_fig(fig, out_path)


def _md_table(df: Optional[pd.DataFrame], n: int = 15) -> str:
    """Render a small markdown table WITHOUT requiring tabulate."""
    if df is None or df.empty:
        return "_(keine Daten)_\n\n"

    head = df.head(n).copy()

    # stringify + avoid ugly 'nan'
    for c in head.columns:
        head[c] = head[c].map(lambda v: "" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v))

    cols = [str(c) for c in head.columns]

    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in head.iterrows():
        vals = [str(row[c]) for c in head.columns]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n\n"


def _pick_true_drivers(evidence_struct: Optional[pd.DataFrame], effects_struct: Optional[pd.DataFrame], k: int = 5) -> Optional[pd.DataFrame]:
    """
    Prefer Gate-3 evidence if available; fall back to Gate-2 effects.
    Picks strongest drivers by effect size (abs_or_lift if present else abs_coef).
    """
    df = None
    if evidence_struct is not None and not evidence_struct.empty:
        df = evidence_struct.copy()
        # If decisions exist, pick SCALE first, otherwise INVESTIGATE, otherwise strongest significant.
        if "decision" in df.columns:
            # Order: SCALE -> INVESTIGATE -> STOP (best first)
            priority = {"SCALE": 0, "INVESTIGATE": 1, "STOP": 2}
            df["_prio"] = df["decision"].astype(str).map(priority).fillna(3).astype(int)
        else:
            df["_prio"] = 1

        if "abs_or_lift" in df.columns:
            df["_score"] = pd.to_numeric(df["abs_or_lift"], errors="coerce")
        elif "odds_ratio" in df.columns:
            df["_score"] = (pd.to_numeric(df["odds_ratio"], errors="coerce") - 1.0).abs()
        elif "abs_coef" in df.columns:
            df["_score"] = pd.to_numeric(df["abs_coef"], errors="coerce")
        elif "coef_logit" in df.columns:
            df["_score"] = pd.to_numeric(df["coef_logit"], errors="coerce").abs()
        else:
            df["_score"] = np.nan

        # Prefer significant if available (Gate 2)
        if "significant" in df.columns:
            sig = df["significant"].astype("boolean").fillna(False).astype(bool)
            df["_sig"] = sig.astype(int)
        else:
            df["_sig"] = 0

        df = df.sort_values(["_prio", "_sig", "_score"], ascending=[True, False, False])
        cols_keep = [c for c in ["feature", "odds_ratio", "ci_low_or", "ci_high_or", "q_value", "stability_sign", "decision"] if c in df.columns]
        out = df[cols_keep].head(k).copy()
        return out.reset_index(drop=True)

    if effects_struct is not None and not effects_struct.empty:
        df = effects_struct.copy()
        if "odds_ratio" in df.columns:
            df["_score"] = (pd.to_numeric(df["odds_ratio"], errors="coerce") - 1.0).abs()
        elif "abs_coef" in df.columns:
            df["_score"] = pd.to_numeric(df["abs_coef"], errors="coerce")
        elif "coef_logit" in df.columns:
            df["_score"] = pd.to_numeric(df["coef_logit"], errors="coerce").abs()
        else:
            df["_score"] = np.nan

        if "significant" in df.columns:
            df["_sig"] = df["significant"].astype("boolean").fillna(False).astype(bool).astype(int)
        else:
            df["_sig"] = 0

        df = df.sort_values(["_sig", "_score"], ascending=[False, False])
        cols_keep = [c for c in ["feature", "odds_ratio", "ci_low_or", "ci_high_or", "q_value", "abs_coef"] if c in df.columns]
        out = df[cols_keep].head(k).copy()
        return out.reset_index(drop=True)

    return None


def _pick_overestimated(compare_df: Optional[pd.DataFrame], k: int = 3) -> Optional[pd.DataFrame]:
    if compare_df is None or compare_df.empty or "overestimated" not in compare_df.columns:
        return None
    df = compare_df.copy()
    mask = df["overestimated"].astype("boolean").fillna(False).astype(bool)
    df = df.loc[mask].copy()
    if df.empty:
        return None
    if "abs_gap" not in df.columns and "delta_log_or_pred_minus_struct" in df.columns:
        df["abs_gap"] = pd.to_numeric(df["delta_log_or_pred_minus_struct"], errors="coerce").abs()
    sort_col = "abs_gap" if "abs_gap" in df.columns else None
    if sort_col:
        df = df.sort_values(sort_col, ascending=False)
    cols_keep = [c for c in ["feature", "abs_gap", "dec_struct", "dec_pred", "sig_struct", "sig_pred"] if c in df.columns]
    return df[cols_keep].head(k).reset_index(drop=True)


def _pick_underestimated(under_table: Optional[pd.DataFrame], compare_df: Optional[pd.DataFrame], k: int = 3) -> Optional[pd.DataFrame]:
    if under_table is not None and not under_table.empty:
        df = under_table.copy()
        if "abs_gap" in df.columns:
            df = df.sort_values("abs_gap", ascending=False)
        cols_keep = [c for c in ["feature", "abs_gap", "dec_struct", "dec_pred", "sig_struct", "sig_pred"] if c in df.columns]
        if cols_keep:
            return df[cols_keep].head(k).reset_index(drop=True)
        return df.head(k).reset_index(drop=True)

    # fallback: derive from compare
    if compare_df is None or compare_df.empty:
        return None
    df = compare_df.copy()
    if "overestimated" in df.columns:
        over = df["overestimated"].astype("boolean").fillna(False).astype(bool)
        df = df.loc[~over].copy()
    if "sig_struct" in df.columns:
        sigs = df["sig_struct"].astype("boolean").fillna(False).astype(bool)
        df = df.loc[sigs].copy()
    if df.empty:
        return None
    if "abs_gap" not in df.columns and "delta_log_or_pred_minus_struct" in df.columns:
        df["abs_gap"] = pd.to_numeric(df["delta_log_or_pred_minus_struct"], errors="coerce").abs()
    if "abs_gap" in df.columns:
        df = df.sort_values("abs_gap", ascending=False)
    cols_keep = [c for c in ["feature", "abs_gap", "dec_struct", "dec_pred", "sig_struct", "sig_pred"] if c in df.columns]
    return df[cols_keep].head(k).reset_index(drop=True)


def _pick_segment_actions(segment_playbook: Optional[pd.DataFrame], segment_profiles: Optional[pd.DataFrame], k: int = 8) -> Optional[pd.DataFrame]:
    df = None
    if segment_playbook is not None and not segment_playbook.empty:
        df = segment_playbook.copy()
        # order: SCALE first, then INVESTIGATE, then STOP
        prio = {"SCALE": 0, "INVESTIGATE": 1, "STOP": 2}
        if "recommended_action" in df.columns:
            df["_prio"] = df["recommended_action"].astype(str).map(prio).fillna(3).astype(int)
        else:
            df["_prio"] = 1
        if "lift" in df.columns:
            df["_lift"] = pd.to_numeric(df["lift"], errors="coerce")
        else:
            df["_lift"] = np.nan
        df = df.sort_values(["_prio", "_lift"], ascending=[True, False])
        cols_keep = [c for c in ["segment", "segment_type", "share", "conversion_rate", "lift", "recommended_action", "caution"] if c in df.columns]
        return df[cols_keep].head(k).reset_index(drop=True)

    if segment_profiles is not None and not segment_profiles.empty:
        df = segment_profiles.copy()
        if "lift" in df.columns:
            df["_lift"] = pd.to_numeric(df["lift"], errors="coerce")
            df = df.sort_values("_lift", ascending=False)
        cols_keep = [c for c in ["segment", "segment_type", "share", "conversion_rate", "lift"] if c in df.columns]
        return df[cols_keep].head(k).reset_index(drop=True)

    return None


def build_executive_summary(
    *,
    project_root: Path,
    cfg: LeadScoringCaseConfig,
    summary: Dict[str, object],
    evidence_struct: Optional[pd.DataFrame],
    effects_struct: Optional[pd.DataFrame],
    compare: Optional[pd.DataFrame],
    under_table: Optional[pd.DataFrame],
    segment_playbook: Optional[pd.DataFrame],
    segment_profiles: Optional[pd.DataFrame],
) -> Tuple[Path, str]:
    """
    Build a 1-page, C-level markdown summary from pipeline artifacts.
    Writes results/executive_summary.md and returns (path, markdown_text).
    """
    results_dir = project_root / cfg.paths.results_dir
    _ensure_dir(results_dir)

    auc_struct = summary.get("auc_struct")
    auc_pred = summary.get("auc_pred")

    true_drivers = _pick_true_drivers(evidence_struct, effects_struct, k=5)
    over = _pick_overestimated(compare, k=3)
    under = _pick_underestimated(under_table, compare, k=3)
    seg = _pick_segment_actions(segment_playbook, segment_profiles, k=8)

    alpha = cfg.significance.alpha
    min_abs_lift = cfg.decision.min_abs_lift
    min_stab = cfg.robustness.min_stability_fraction

    md: List[str] = []
    md.append("# Executive Summary – Lead Scoring Case\n\n")

    md.append("## Ziel\n")
    md.append("Conversion steigern – aber nur über **echte Hebel** (strukturell/robust), nicht über Proxy-Signale, die nur gut vorhersagen.\n\n")

    md.append("## Modell-Setup (Dualität)\n")
    md.append(f"- Predictive Logit (Performance): AUC **{_fmt_float(auc_pred, 4)}**\n")
    md.append(f"- Structural Logit (Actionability): AUC **{_fmt_float(auc_struct, 4)}**\n")
    md.append("\n")

    md.append("## Decision Gates\n")
    md.append(f"- Gate 2 (Evidenz): Bootstrap + BH-FDR, signifikant bei **q < {alpha}**\n")
    md.append(f"- Gate 3 (Robustness → Entscheidung): Effektgröße **|OR−1| ≥ {min_abs_lift}** und Stabilität **stability_sign ≥ {min_stab}**\n")
    md.append("\n")

    md.append("## 5 echte Treiber (Structural Truth)\n\n")
    md.append(_md_table(true_drivers, 5) if true_drivers is not None else "_(keine Daten)_\n\n")

    md.append("## 3 überschätzte Features (Predictive >> Structural)\n\n")
    md.append(_md_table(over, 3) if over is not None else "_(keine Daten)_\n\n")

    md.append("## 3 unterschätzte Features (Structural >> Predictive)\n\n")
    md.append(_md_table(under, 3) if under is not None else "_(keine Daten)_\n\n")

    md.append("## Segment Playbook (Focus & Routing)\n\n")
    md.append(_md_table(seg, 8) if seg is not None else "_(keine Daten)_\n\n")

    md.append("## Empfohlene Aktionen (kurz)\n")
    md.append("- **Scale:** ICP/High-Intent Sources + Top Score Segmente (p80/p90/p95)\n")
    md.append("- **Investigate:** Landing Page Submission (Qualifizierung/Offer), API Leads (separates Handling), Chat (Experiment)\n")
    md.append("- **Stop:** Motivation-Narratives als Budget-Hebel, Sales-Zeit für Bottom-Score Segmente\n\n")

    out_path = results_dir / "executive_summary.md"
    text = "".join(md)
    out_path.write_text(text, encoding="utf-8")
    return out_path, text


def build_report(
    *,
    project_root: Path,
    cfg: LeadScoringCaseConfig,
    summary: Dict[str, object],
) -> Tuple[Path, Path]:
    """
    Loads known artifacts from results/tables, writes a Markdown report and (optional) figures.
    Also generates a 1-page executive summary (results/executive_summary.md).
    """
    results_dir = project_root / "results"
    tables_dir = project_root / cfg.paths.tables_dir
    figures_dir = project_root / cfg.paths.figures_dir

    _ensure_dir(results_dir)
    _ensure_dir(tables_dir)
    _ensure_dir(figures_dir)

    # ---- load tables (optional) ----
    effects_struct = _try_read_csv(tables_dir / "effects_structural_full.csv")
    effects_struct_sig = _try_read_csv(tables_dir / "effects_structural_significant.csv")
    evidence_struct = _try_read_csv(tables_dir / "evidence_structural.csv")

    effects_pred = _try_read_csv(tables_dir / "effects_predictive_full.csv")
    evidence_pred = _try_read_csv(tables_dir / "evidence_predictive.csv")

    compare = _try_read_csv(tables_dir / "compare_structural_vs_predictive.csv")
    segment_profiles = _try_read_csv(tables_dir / "segment_profiles.csv")
    segment_playbook = _try_read_csv(tables_dir / "segment_playbook.csv")
    under_table = _try_read_csv(tables_dir / "underestimated_features.csv")

    # ---- figures ----
    fig_paths: Dict[str, Path] = {}

    if _HAS_MPL and effects_struct is not None:
        p = figures_dir / "fig_top_structural_effects.png"
        _plot_top_effects(effects_struct, p, title="Top Structural Effects (|coef|)")
        if p.exists():
            fig_paths["Top Structural Effects"] = p

    if _HAS_MPL and compare is not None:
        p = figures_dir / "fig_overestimated_features.png"
        _plot_overestimated(compare, p, title="Overestimated Features (Predictive >> Structural)")
        if p.exists():
            fig_paths["Overestimated Features"] = p

        p2 = figures_dir / "fig_underestimated_features.png"
        _plot_underestimated(compare, p2, title="Underestimated Features (Structural >> Predictive)")
        if p2.exists():
            fig_paths["Underestimated Features"] = p2

    if _HAS_MPL and segment_profiles is not None:
        p = figures_dir / "fig_segment_lift.png"
        _plot_segment_lift(segment_profiles, p, title="Best Segments by Lift")
        if p.exists():
            fig_paths["Segment Lift"] = p

    # ---- narrative numbers ----
    auc_struct = summary.get("auc_struct")
    auc_pred = summary.get("auc_pred")

    # ---- Executive Summary (auto) ----
    exec_path, exec_text = build_executive_summary(
        project_root=project_root,
        cfg=cfg,
        summary=summary,
        evidence_struct=evidence_struct,
        effects_struct=effects_struct,
        compare=compare,
        under_table=under_table,
        segment_playbook=segment_playbook,
        segment_profiles=segment_profiles,
    )

    report_path = results_dir / "report.md"

    md: list[str] = []
    md.append("# Lead Scoring Case – Explainability Report\n\n")

    # Embed executive summary at the top (optional but recommended)
    md.append("## Executive Summary (auto-generated)\n\n")
    md.append(f"> Export: `{exec_path.relative_to(project_root).as_posix()}`\n\n")
    # Keep it readable by embedding the content below; comment out if you prefer separate file only.
    md.append(exec_text + "\n\n---\n\n")

    md.append("## Ziel\n")
    md.append("- **Welche Merkmale erklären Conversion wirklich?** (strukturell, robust, signifikant)\n")
    md.append("- **Welche werden überschätzt?** (predictive stark, strukturell schwach)\n")
    md.append("- **Echt oder Zufall?** (Bootstrap-CI, q-values, Stabilität)\n\n")

    md.append("## Modell-Übersicht\n")
    if auc_struct is not None:
        md.append(f"- Structural Logit AUC: **{_fmt_float(auc_struct, 4)}**\n")
    if auc_pred is not None:
        md.append(f"- Predictive Logit AUC: **{_fmt_float(auc_pred, 4)}**\n")
    md.append("\n")

    md.append("## 1) Structural Effects – Was erklärt Conversion wirklich?\n\n")
    if effects_struct_sig is not None and not effects_struct_sig.empty:
        md.append("**Top signifikante Structural-Effekte (Gate 2):**\n\n")
        md.append(_md_table(effects_struct_sig, 20))
    elif effects_struct is not None:
        md.append("**Structural-Effekte (Gate 2):**\n\n")
        md.append(_md_table(effects_struct, 20))
    else:
        md.append("_(Keine effects_structural Tabelle gefunden)_\n\n")

    md.append("## 2) Evidence / Entscheidungen (Gate 3)\n\n")
    if evidence_struct is not None:
        md.append("**Structural Evidence (Entscheidungen):**\n\n")
        md.append(_md_table(evidence_struct, 25))
    else:
        md.append("_(Keine evidence_structural Tabelle gefunden)_\n\n")

    md.append("## 3b) Predictive Effects (Gate 2)\n\n")
    if effects_pred is not None and not effects_pred.empty:
        md.append("**Top Predictive-Effekte (Gate 2):**\n\n")
        md.append(_md_table(effects_pred, 20))
    else:
        md.append("_(Keine effects_predictive Tabelle gefunden)_\n\n")
        

    md.append("## 3) Predictive Evidence (nur zur Einordnung)\n\n")
    if evidence_pred is not None:
        md.append("**Predictive Evidence:**\n\n")
        md.append(_md_table(evidence_pred, 25))
    else:
        md.append("_(Keine evidence_predictive Tabelle gefunden)_\n\n")

    md.append("## 4) Overestimated – Predictive >> Structural\n\n")
    if compare is not None and "overestimated" in compare.columns:
        over_mask = compare["overestimated"].astype("boolean").fillna(False).astype(bool)
        over = compare.loc[over_mask].copy()
        md.append(f"- Anzahl overestimated: **{len(over)}**\n\n")
        md.append(_md_table(over, 25))
    else:
        md.append("_(Keine compare_structural_vs_predictive Tabelle gefunden)_\n\n")

    md.append("## 5) Underestimated – Structural >> Predictive\n\n")
    if under_table is not None and not under_table.empty:
        md.append(_md_table(under_table, 25))
    elif compare is not None and "abs_gap" in compare.columns and "overestimated" in compare.columns:
        over_mask = compare["overestimated"].astype("boolean").fillna(False).astype(bool)
        tmp2 = compare.loc[~over_mask].sort_values("abs_gap", ascending=False)
        md.append(_md_table(tmp2, 25))
    else:
        md.append("_(Keine underestimated Tabelle verfügbar)_\n\n")

    md.append("## 6) Segmente (Profiling)\n\n")
    if segment_profiles is not None and not segment_profiles.empty:
        md.append("**Top Segmente nach Lift:**\n\n")
        md.append(_md_table(segment_profiles, 25))
        md.append("> Export: `results/tables/segment_profiles.csv`\n\n")
    else:
        md.append("_(Keine segment_profiles Tabelle gefunden)_\n\n")

    md.append("## Grafiken\n\n")
    if not fig_paths:
        md.append("_(Noch keine Grafiken erzeugt oder matplotlib nicht verfügbar)_\n\n")
    else:
        for label, p in fig_paths.items():
            rel = p.relative_to(project_root)
            md.append(f"### {label}\n\n")
            md.append(f"![{label}]({rel.as_posix()})\n\n")

    md.append("## Artefakte\n")
    md.append("- Tabellen: `results/tables/`\n")
    md.append("- Grafiken: `results/figures/`\n")
    md.append("- Report: `results/report.md`\n")
    md.append("- Executive Summary: `results/executive_summary.md`\n")

    report_path.write_text("".join(md), encoding="utf-8")
    return report_path, figures_dir
