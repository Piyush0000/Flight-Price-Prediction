"""
Model explainability module using SHAP (TreeExplainer) and Permutation Importance.
Generates summary plots, dependence plots, interaction analyses, and feature ranking correlation tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import shap
from sklearn.inspection import permutation_importance

from src.config import ProjectConfig
from src.features import build_sklearn_preprocessor
from src.utils import save_table, setup_logger

logger = setup_logger("explainability")


def run_explainability_pipeline(
    best_model: Any,
    catboost_model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Execute full explainability analysis:
    1. Preprocess feature subset for SHAP
    2. Compute TreeExplainer SHAP values on best LightGBM model
    3. Generate beeswarm summary plot, bar plot, and dependence plots
    4. Compute SHAP interaction values for (days_left x class) and (stops x duration)
    5. Compare feature rankings across LightGBM SHAP, CatBoost feature importance, and Permutation Importance
    6. Compute Spearman rank correlation table
    """
    logger.info("Starting TreeExplainer and feature attribution pipeline...")

    cat_sub = [c for c in config.categorical_features if c in X_train.columns]
    num_sub = [c for c in config.numerical_features if c in X_train.columns]
    bin_sub = [c for c in config.binary_features if c in X_train.columns]

    preprocessor = build_sklearn_preprocessor(cat_sub, num_sub, bin_sub, use_one_hot=False)
    X_tr_proc = preprocessor.fit_transform(X_train)
    X_te_proc = preprocessor.transform(X_test)

    # Feature names from preprocessor
    feature_names = num_sub + cat_sub + bin_sub

    # Sample for fast, accurate SHAP computation
    sample_size = min(config.shap_sample_size, len(X_te_proc))
    sample_idx = np.random.RandomState(config.seed).choice(len(X_te_proc), size=sample_size, replace=False)
    X_sample = X_te_proc[sample_idx]
    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    # 1. SHAP TreeExplainer on best tree model (extract estimator from pipeline if needed)
    tree_regressor = best_model.named_steps["regressor"] if hasattr(best_model, "named_steps") else best_model

    logger.info("Computing SHAP values for sample size %d...", sample_size)
    explainer = shap.TreeExplainer(tree_regressor)
    shap_values = explainer.shap_values(X_sample)

    # A. Beeswarm Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample_df, show=False, max_display=12)
    plt.title("SHAP Beeswarm Summary Plot (Feature Impact on Log Price)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(config.figures_dir / "shap_summary_beeswarm.png", dpi=300)
    plt.close()

    # B. Bar Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False, max_display=12)
    plt.title("Mean Absolute SHAP Values (Global Feature Importance)", fontsize=13, pad=15)
    plt.tight_layout()
    plt.savefig(config.figures_dir / "shap_summary_bar.png", dpi=300)
    plt.close()

    # C. Dependence Plots
    target_dep_features = ["days_left", "duration", "stops", "class"]
    for feat in target_dep_features:
        if feat in feature_names:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.dependence_plot(
                feat,
                shap_values,
                X_sample_df,
                ax=ax,
                show=False,
                interaction_index="auto",
            )
            ax.set_title(f"SHAP Dependence Plot: {feat}", fontsize=12)
            plt.tight_layout()
            plt.savefig(config.figures_dir / f"shap_dependence_{feat}.png", dpi=300)
            plt.close(fig)

    # D. Interaction Values
    # Use smaller sample of 150 rows for interaction computation speed
    logger.info("Computing SHAP Interaction values for selected feature pairs...")
    int_sample_size = min(150, sample_size)
    X_int_sample = X_sample[:int_sample_size]
    X_int_df = pd.DataFrame(X_int_sample, columns=feature_names)
    interaction_values = explainer.shap_interaction_values(X_int_sample)

    # Interaction 1: days_left x class
    if "days_left" in feature_names and "class" in feature_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        d_idx = feature_names.index("days_left")
        c_idx = feature_names.index("class")
        ax.scatter(
            X_int_df["days_left"],
            interaction_values[:, d_idx, c_idx],
            c=X_int_df["class"],
            cmap="coolwarm",
            alpha=0.7,
        )
        ax.set_xlabel("Days Left (Booking Lead Time)")
        ax.set_ylabel("SHAP Interaction Value (Days Left x Class)")
        ax.set_title("Interaction Dynamics: Days Left x Travel Class")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(config.figures_dir / "shap_interaction_days_left_class.png", dpi=300)
        plt.close(fig)

    # Interaction 2: stops x duration
    if "stops" in feature_names and "duration" in feature_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        s_idx = feature_names.index("stops")
        dur_idx = feature_names.index("duration")
        ax.scatter(
            X_int_df["duration"],
            interaction_values[:, s_idx, dur_idx],
            c=X_int_df["stops"],
            cmap="viridis",
            alpha=0.7,
        )
        ax.set_xlabel("Flight Duration (Hours)")
        ax.set_ylabel("SHAP Interaction Value (Stops x Duration)")
        ax.set_title("Interaction Dynamics: Stops x Flight Duration")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.savefig(config.figures_dir / "shap_interaction_stops_duration.png", dpi=300)
        plt.close(fig)

    # 2. Compare Feature Rankings: LightGBM SHAP, CatBoost Importances, Permutation Importance
    logger.info("Comparing feature rankings across explainability approaches...")

    # LightGBM SHAP importance
    lgb_shap_imp = np.mean(np.abs(shap_values), axis=0)

    # CatBoost native feature importance (mapped to shared feature subset)
    if hasattr(catboost_model, "get_feature_importance"):
        cb_imp_raw = catboost_model.get_feature_importance()
        cb_feat_names = catboost_model.feature_names_
        cb_dict = dict(zip(cb_feat_names, cb_imp_raw))
        cb_imp = np.array([cb_dict.get(col, 0.0) for col in feature_names])
    else:
        cb_imp = lgb_shap_imp

    # Permutation importance
    logger.info("Computing Permutation Importance on test sample...")
    perm_res = permutation_importance(
        tree_regressor,
        X_sample,
        y_test[sample_idx],
        n_repeats=5,
        random_state=config.seed,
        n_jobs=4,
    )
    perm_imp = perm_res.importances_mean

    # Save feature importances table
    feat_imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "lgbm_shap_mean_abs": lgb_shap_imp,
            "catboost_importance": cb_imp,
            "permutation_importance": perm_imp,
        }
    ).sort_values(by="lgbm_shap_mean_abs", ascending=False).reset_index(drop=True)
    save_table(feat_imp_df, config.tables_dir / "feature_importances.csv", index=False)

    # Compute pairwise Spearman Rank Correlations
    rho_lgb_cb, p_lgb_cb = spearmanr(feat_imp_df["lgbm_shap_mean_abs"], feat_imp_df["catboost_importance"])
    rho_lgb_perm, p_lgb_perm = spearmanr(feat_imp_df["lgbm_shap_mean_abs"], feat_imp_df["permutation_importance"])
    rho_cb_perm, p_cb_perm = spearmanr(feat_imp_df["catboost_importance"], feat_imp_df["permutation_importance"])

    corr_data = [
        {
            "Comparison_Pair": "LightGBM SHAP vs. CatBoost Importance",
            "Spearman_Rho": round(rho_lgb_cb, 4),
            "P_Value": round(p_lgb_cb, 6),
        },
        {
            "Comparison_Pair": "LightGBM SHAP vs. Permutation Importance",
            "Spearman_Rho": round(rho_lgb_perm, 4),
            "P_Value": round(p_lgb_perm, 6),
        },
        {
            "Comparison_Pair": "CatBoost Importance vs. Permutation Importance",
            "Spearman_Rho": round(rho_cb_perm, 4),
            "P_Value": round(p_cb_perm, 6),
        },
    ]

    corr_df = pd.DataFrame(corr_data)
    save_table(corr_df, config.tables_dir / "feature_ranking_correlation.csv", index=False)
    logger.info("Feature ranking correlation table saved.")

    return corr_df
