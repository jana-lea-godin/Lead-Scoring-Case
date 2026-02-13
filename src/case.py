from __future__ import annotations

from pathlib import Path

from .config import LeadScoringCaseConfig
from .data_loader import LeadDataLoader
from .multivariate_explainer import (
    LeadScoringPreprocessor,
    build_structural_df,
    train_logit,
    top_coefficients,
    block_contribution_analysis,
)


def _project_root() -> Path:
    # src/case.py -> src -> project root
    return Path(__file__).resolve().parents[1]


def run() -> None:
    cfg = LeadScoringCaseConfig()
    root = _project_root()

    # Ensure results/tables exists early (used by multiple blocks)
    tables_dir = root / cfg.paths.tables_dir
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 1) Load data
    # ----------------------------
    loader = LeadDataLoader(cfg=cfg, project_root=root)
    df_raw = loader.load_raw()

    print("=" * 100)
    print("RAW LOADED")
    print("=" * 100)
    print("Shape:", df_raw.shape)

    # ----------------------------
    # 2) Clean + (optional) tracking split
    # ----------------------------
    prep = LeadScoringPreprocessor(cfg)
    df_clean = prep.build_clean(df_raw)

    print("\n" + "=" * 100)
    print("CLEAN BUILT")
    print("=" * 100)
    print("Shape:", df_clean.shape)

    if cfg.cleaning.split_tracking_missing_rows:
        df_no_tracking, df_with_tracking = prep.tracking_split(df_clean)
        print("\nTracking split:")
        print(" - no_tracking:", df_no_tracking.shape)
        print(" - with_tracking:", df_with_tracking.shape)
    else:
        df_no_tracking, df_with_tracking = None, None

    # ----------------------------
    # 3) Model dataset (drop IDs + leakage cols)
    # ----------------------------
    df_model = prep.build_model_df(df_clean)

    print("\n" + "=" * 100)
    print("MODEL DF BUILT (IDs + leakage removed)")
    print("=" * 100)
    print("Shape:", df_model.shape)

    # ----------------------------
    # 4) Train structural model (drops process/proximity cols)
    # ----------------------------
    df_struct = build_structural_df(df_model, cfg)
    res = train_logit(df_struct, cfg, structural=True)

    print("\n" + "=" * 100)
    print("STRUCTURAL LOGIT RESULTS")
    print("=" * 100)
    print("AUC:", round(res.auc, 4))
    print(res.report)

    # ----------------------------
    # 5) Gate 2 (Structural): Bootstrap CIs + q-values
    # ----------------------------
    from .significance_tester import (
        bootstrap_logit_coefficients_fixed_preprocessing,
        build_effect_table_from_trained_model,
    )

    X_struct = df_struct.drop(columns=[cfg.data.target_col]).copy()
    y_struct = df_struct[cfg.data.target_col].copy()

    boot = bootstrap_logit_coefficients_fixed_preprocessing(
        res.model,
        X_struct,
        y_struct,
        n_bootstrap=cfg.robustness.bootstrap_iterations,
        sample_frac=cfg.robustness.bootstrap_sample_frac,
        random_state=cfg.robustness.random_state,
    )

    effects = build_effect_table_from_trained_model(
        res.model,
        cfg,
        bootstrap_coefs=boot,
    ).table

    print("\n" + "=" * 100)
    print("GATE 2: EFFECTS (OR + 95% CI + q-values)")
    print("=" * 100)
    print(effects.head(30).to_string(index=False))

    # ----------------------------
    # 5b) Write Gate 2 artifacts (Structural)
    # ----------------------------
    full_effects_path = tables_dir / "effects_structural_full.csv"
    effects.to_csv(full_effects_path, index=False)

    if "q_value" in effects.columns:
        effects_sig = effects.loc[effects["q_value"] < cfg.significance.alpha].copy()
    else:
        effects_sig = effects.copy()

    sig_effects_path = tables_dir / "effects_structural_significant.csv"
    effects_sig.to_csv(sig_effects_path, index=False)

    print("\nWROTE GATE 2 TABLES")
    print(" -", full_effects_path)
    print(" -", sig_effects_path)

    # ----------------------------
    # Gate 3 (Structural): Robustness + Decision table
    # ----------------------------
    from .robustness_checker import stability_from_bootstrap
    from .decision_engine import decide_actions

    effects_with_stability = stability_from_bootstrap(effects, boot)

    evidence = decide_actions(
        effects_with_stability,
        alpha=cfg.significance.alpha,
        min_abs_lift=cfg.decision.min_abs_lift,
        min_stability_fraction=cfg.robustness.min_stability_fraction,
    )

    evidence_path = tables_dir / "evidence_structural.csv"
    evidence.to_csv(evidence_path, index=False)

    print("\nWROTE EVIDENCE TABLE")
    print(" -", evidence_path)

    # ==================================================================================
    # A) Predictive train + Gate 2 + Gate 3  (for "overestimated features")
    # ==================================================================================
    res_pred = train_logit(df_model, cfg, structural=False)

    print("\n" + "=" * 100)
    print("PREDICTIVE LOGIT RESULTS")
    print("=" * 100)
    print("AUC:", round(res_pred.auc, 4))
    print(res_pred.report)

    X_pred = df_model.drop(columns=[cfg.data.target_col]).copy()
    y_pred = df_model[cfg.data.target_col].copy()

    boot_pred = bootstrap_logit_coefficients_fixed_preprocessing(
        res_pred.model,
        X_pred,
        y_pred,
        n_bootstrap=cfg.robustness.bootstrap_iterations,
        sample_frac=cfg.robustness.bootstrap_sample_frac,
        random_state=cfg.robustness.random_state,
    )

    effects_pred = build_effect_table_from_trained_model(
        res_pred.model,
        cfg,
        bootstrap_coefs=boot_pred,
    ).table

    effects_pred_with_stability = stability_from_bootstrap(effects_pred, boot_pred)

    evidence_pred = decide_actions(
        effects_pred_with_stability,
        alpha=cfg.significance.alpha,
        min_abs_lift=cfg.decision.min_abs_lift,
        min_stability_fraction=cfg.robustness.min_stability_fraction,
    )

    effects_pred_path = tables_dir / "effects_predictive_full.csv"
    evidence_pred_path = tables_dir / "evidence_predictive.csv"

    effects_pred_with_stability.to_csv(effects_pred_path, index=False)
    evidence_pred.to_csv(evidence_pred_path, index=False)

    print("\nWROTE PREDICTIVE TABLES")
    print(" -", effects_pred_path)
    print(" -", evidence_pred_path)

    # ==================================================================================
    # B) Structural vs Predictive Compare (write compare table + show top overestimated)
    # ==================================================================================
    from .overshadow_analyzer import compare_structural_vs_predictive

    compare = compare_structural_vs_predictive(
        structural_effects=evidence,       # structural decisions
        predictive_effects=evidence_pred,  # predictive decisions
    )

    compare_path = tables_dir / "compare_structural_vs_predictive.csv"
    compare.to_csv(compare_path, index=False)

    print("\nWROTE COMPARISON TABLE")
    print(" -", compare_path)

    # ---- sanitize types for filtering (kills yellow warnings) ----
    overestimated = compare["overestimated"].fillna(False).astype(bool)
    sig_struct = compare["sig_struct"].fillna(False).infer_objects(copy=False).astype(bool)
    sig_pred = compare["sig_pred"].fillna(False).infer_objects(copy=False).astype(bool)
    overestimated = compare["overestimated"].fillna(False).infer_objects(copy=False).astype(bool)
    dec_pred = compare["dec_pred"].fillna("").astype(str)

    print("\n" + "=" * 100)
    print("TOP OVER-ESTIMATED FEATURES (Predictive >> Structural)")
    print("=" * 100)
    over = compare.loc[overestimated].head(25)
    if len(over) == 0:
        print("(none flagged by current heuristic)")
    else:
        print(over.to_string(index=False))

    # -------------------------------------------------
    # Underestimated Features (Structural > Predictive)
    # -------------------------------------------------
    pred_weak = (~sig_pred) | (dec_pred.isin(["STOP"]))
    under = compare.loc[(~overestimated) & sig_struct & pred_weak].copy()

    if "abs_gap" in under.columns:
        under = under.sort_values("abs_gap", ascending=False)

    under_path = tables_dir / "underestimated_features.csv"
    under.to_csv(under_path, index=False)

    print("\nUNDER-ESTIMATED FEATURES (Structural strong, Predictive weak)")
    print("=" * 100)
    print(under.head(25).to_string(index=False))
    print("\nWROTE UNDER-ESTIMATION TABLE")
    print(" -", under_path)

    # ----------------------------
    # 6) Top coefficients (interpretation) - Structural
    # ----------------------------
    coef_top = top_coefficients(res, top_n=25)
    print("\n" + "=" * 100)
    print("TOP COEFFICIENTS (ABS EFFECT)")
    print("=" * 100)
    print(coef_top.to_string(index=False))

    # ----------------------------
    # 7) Block contribution analysis
    # ----------------------------
    blocks = block_contribution_analysis(df_model, cfg)
    print("\n" + "=" * 100)
    print("BLOCK CONTRIBUTION")
    print("=" * 100)
    print("Structural AUC (full):", round(blocks["auc_struct_full"], 4))
    print("Δ drop geo:", round(blocks["delta_drop_geo"], 4))

    print("\nAUC per block alone:")
    for row in blocks["block_alone"]:
        print(f" - {row['block']}: AUC={row['auc']:.4f} | n={row['n_features']}")

    print("\nIncremental AUC:")
    for row in blocks["incremental"]:
        delta = row["delta_vs_prev"]
        delta_str = "" if delta is None else f" (Δ {delta:+.4f})"
        print(f" - up to {row['up_to_block']}: AUC={row['auc']:.4f}{delta_str}")

    # ----------------------------
    # 8) Write processed artifacts
    # ----------------------------
    processed_dir = root / cfg.paths.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = processed_dir / cfg.data.processed_cleaned_filename
    df_clean.to_csv(cleaned_path, index=False)

    if df_with_tracking is not None:
        df_with_tracking.to_csv(
            processed_dir / cfg.data.processed_with_tracking_filename,
            index=False
        )

    if df_no_tracking is not None:
        df_no_tracking.to_csv(
            processed_dir / cfg.data.processed_no_tracking_filename,
            index=False
        )

    model_path = processed_dir / cfg.data.processed_model_filename
    df_model.to_csv(model_path, index=False)

    print("\n" + "=" * 100)
    print("WROTE PROCESSED FILES")
    print("=" * 100)
    print(" -", cleaned_path)
    print(" -", model_path)
    if df_with_tracking is not None:
        print(" -", processed_dir / cfg.data.processed_with_tracking_filename)
    if df_no_tracking is not None:
        print(" -", processed_dir / cfg.data.processed_no_tracking_filename)


if __name__ == "__main__":
    run()