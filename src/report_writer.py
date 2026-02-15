# src/report_writer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def build_report(
    *,
    project_root: Path,
    cfg: LeadScoringCaseConfig,
    summary: Dict[str, object],
) -> Tuple[Path, Path]:
    """
    Loads known artifacts from results/tables, writes a Markdown report and (optional) figures.
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

    report_path = results_dir / "report.md"

    md: list[str] = []
    md.append("# Lead Scoring Case – Explainability Report\n\n")

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

    report_path.write_text("".join(md), encoding="utf-8")
    return report_path, figures_dir
