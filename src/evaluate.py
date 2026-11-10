"""
Comprehensive evaluation, cross-validation, leak-free vs plain KFold comparisons,
error breakdown analyses, ablation studies, and diagnostic plotting.
All reported metrics are strictly converted back to the original price scale.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold

from src.config import ProjectConfig
from src.features import build_sklearn_preprocessor
from src.utils import evaluate_predictions, save_table, setup_logger

logger = setup_logger("evaluation")


def evaluate_cv(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    is_catboost: bool = False,
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray]:
    """
    Evaluate an estimator across cross-validation splits.
    Returns (mean_metrics, std_metrics, oof_predictions).
    """
    fold_metrics: List[Dict[str, float]] = []
    oof_preds = np.zeros(len(X), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]

        if is_catboost:
            from catboost import CatBoostRegressor
            fold_model = CatBoostRegressor(
                iterations=model.get_param("iterations"),
                learning_rate=model.get_param("learning_rate"),
                depth=model.get_param("depth"),
                random_seed=model.get_param("random_seed") + fold,
                verbose=0,
                cat_features=model.get_param("cat_features"),
                thread_count=4,
            )
            fold_model.fit(X_tr, y_tr)
            preds_log = fold_model.predict(X_va)
        else:
            fold_model = clone(model)
            fold_model.fit(X_tr, y_tr)
            preds_log = fold_model.predict(X_va)

        oof_preds[val_idx] = preds_log
        metrics = evaluate_predictions(y_va, preds_log, is_log_target=True)
        fold_metrics.append(metrics)

    # Compute mean and standard deviation
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean().round(2).to_dict()
    std_metrics = metrics_df.std().round(2).to_dict()

    return mean_metrics, std_metrics, oof_preds


def compare_kfold_vs_groupkfold(
    best_pipe: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Compare GroupKFold vs plain random KFold to demonstrate the degree
    of performance inflation caused by data leakage.
    """
    logger.info("Evaluating GroupKFold (leakage-safe) vs random KFold (leaking)...")

    # GroupKFold
    gkf = GroupKFold(n_splits=config.cv_folds)
    gkf_splits = list(gkf.split(X, y, groups=groups))
    gkf_mean, gkf_std, _ = evaluate_cv(best_pipe, X, y, gkf_splits)

    # Random KFold
    kf = KFold(n_splits=config.cv_folds, shuffle=True, random_state=config.seed)
    kf_splits = list(kf.split(X, y))
    kf_mean, kf_std, _ = evaluate_cv(best_pipe, X, y, kf_splits)

    comparison_data = [
        {
            "Split_Strategy": "GroupKFold (Leakage-Safe)",
            "MAE_Mean": gkf_mean["MAE"],
            "MAE_Std": gkf_std["MAE"],
            "RMSE_Mean": gkf_mean["RMSE"],
            "RMSE_Std": gkf_std["RMSE"],
            "R2_Mean": gkf_mean["R2"],
            "R2_Std": gkf_std["R2"],
            "MAPE_Mean": gkf_mean["MAPE"],
            "sMAPE_Mean": gkf_mean["sMAPE"],
        },
        {
            "Split_Strategy": "Random KFold (Leaking)",
            "MAE_Mean": kf_mean["MAE"],
            "MAE_Std": kf_std["MAE"],
            "RMSE_Mean": kf_mean["RMSE"],
            "RMSE_Std": kf_std["RMSE"],
            "R2_Mean": kf_mean["R2"],
            "R2_Std": kf_std["R2"],
            "MAPE_Mean": kf_mean["MAPE"],
            "sMAPE_Mean": kf_mean["sMAPE"],
        },
    ]

    comp_df = pd.DataFrame(comparison_data)
    save_table(comp_df, config.tables_dir / "kfold_vs_groupkfold.csv", index=False)
    logger.info("Saved KFold vs GroupKFold leakage comparison.")
    return comp_df


def compute_error_breakdowns(
    y_test_true: np.ndarray,
    y_test_pred: np.ndarray,
    test_df: pd.DataFrame,
    config: ProjectConfig,
) -> Dict[str, pd.DataFrame]:
    """
    Compute fine-grained error metrics (MAE, RMSE, MAPE, sMAPE)
    broken down by: class, stops, airline, top routes, and days_left bucket.
    """
    df = test_df.copy()
    df["y_true_real"] = np.expm1(y_test_true)
    df["y_pred_real"] = np.clip(np.expm1(y_test_pred), a_min=0, a_max=None)
    df["abs_err"] = np.abs(df["y_true_real"] - df["y_pred_real"])
    df["sq_err"] = (df["y_true_real"] - df["y_pred_real"]) ** 2
    df["ape"] = np.abs(df["y_true_real"] - df["y_pred_real"]) / (df["y_true_real"] + 1e-8) * 100.0

    breakdown_tables: Dict[str, pd.DataFrame] = {}

    dimensions = [
        ("class", "error_breakdown_class.csv"),
        ("stops", "error_breakdown_stops.csv"),
        ("airline", "error_breakdown_airline.csv"),
        ("days_left_bucket", "error_breakdown_days_left.csv"),
    ]

    for dim, filename in dimensions:
        if dim in df.columns:
            grouped = df.groupby(dim).agg(
                count=("abs_err", "count"),
                MAE=("abs_err", "mean"),
                RMSE=("sq_err", lambda x: np.sqrt(np.mean(x))),
                MAPE=("ape", "mean"),
            ).round(2).reset_index()
            save_table(grouped, config.tables_dir / filename, index=False)
            breakdown_tables[dim] = grouped

    # Route breakdown for top routes
    if "route" in df.columns:
        top_routes = df["route"].value_counts().head(10).index
        df_top_routes = df[df["route"].isin(top_routes)]
        route_grouped = df_top_routes.groupby("route").agg(
            count=("abs_err", "count"),
            MAE=("abs_err", "mean"),
            RMSE=("sq_err", lambda x: np.sqrt(np.mean(x))),
            MAPE=("ape", "mean"),
        ).round(2).reset_index()
        save_table(route_grouped, config.tables_dir / "error_breakdown_route.csv", index=False)
        breakdown_tables["route"] = route_grouped

    logger.info("Error breakdown tables generated and saved.")
    return breakdown_tables


def run_ablation_study(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Perform feature ablation study:
    1. Full features
    2. Without days_left (days_left, days_left_bucket, is_last_minute)
    3. Without route features (route, airline_route)
    4. Without airline (airline, airline_route)
    5. Trivial features only (airline, class)
    """
    from lightgbm import LGBMRegressor
    logger.info("Running feature ablation experiments...")

    ablation_definitions = [
        ("Full Feature Set", None),
        ("Without days_left", ["days_left", "days_left_bucket", "is_last_minute"]),
        ("Without route features", ["route", "airline_route"]),
        ("Without airline", ["airline", "airline_route"]),
        ("Trivial features only (airline + class)", "trivial_only"),
    ]

    results = []

    for name, dropped_cols in ablation_definitions:
        if dropped_cols == "trivial_only":
            keep_cols = ["airline", "class"]
            X_tr_sub = X_train[keep_cols].copy()
            X_te_sub = X_test[keep_cols].copy()
            cat_sub = ["airline", "class"]
            num_sub = []
            bin_sub = []
        elif dropped_cols is not None:
            X_tr_sub = X_train.drop(columns=[c for c in dropped_cols if c in X_train.columns])
            X_te_sub = X_test.drop(columns=[c for c in dropped_cols if c in X_test.columns])
            cat_sub = [c for c in config.categorical_features if c not in dropped_cols and c in X_tr_sub.columns]
            num_sub = [c for c in config.numerical_features if c not in dropped_cols and c in X_tr_sub.columns]
            bin_sub = [c for c in config.binary_features if c not in dropped_cols and c in X_tr_sub.columns]
        else:
            X_tr_sub = X_train.copy()
            X_te_sub = X_test.copy()
            cat_sub = [c for c in config.categorical_features if c in X_tr_sub.columns]
            num_sub = [c for c in config.numerical_features if c in X_tr_sub.columns]
            bin_sub = [c for c in config.binary_features if c in X_tr_sub.columns]

        preprocessor = build_sklearn_preprocessor(cat_sub, num_sub, bin_sub, use_one_hot=False)
        X_tr_proc = preprocessor.fit_transform(X_tr_sub)
        X_te_proc = preprocessor.transform(X_te_sub)

        model = LGBMRegressor(n_estimators=150, learning_rate=0.08, num_leaves=63, random_state=config.seed, n_jobs=4, verbose=-1)
        model.fit(X_tr_proc, y_train)
        preds_log = model.predict(X_te_proc)
        metrics = evaluate_predictions(y_test, preds_log, is_log_target=True)

        results.append({
            "Ablation_Variant": name,
            "Test_MAE": metrics["MAE"],
            "Test_RMSE": metrics["RMSE"],
            "Test_R2": metrics["R2"],
            "Test_MAPE": metrics["MAPE"],
            "Test_sMAPE": metrics["sMAPE"],
        })

    ablation_df = pd.DataFrame(results)
    save_table(ablation_df, config.tables_dir / "ablation_study.csv", index=False)
    logger.info("Ablation study completed and saved.")
    return ablation_df


def plot_residuals_and_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: ProjectConfig,
) -> None:
    """Generate and save residual plots and predicted-vs-actual plots."""
    y_true_real = np.expm1(y_true)
    y_pred_real = np.clip(np.expm1(y_pred), a_min=0, a_max=None)
    residuals = y_true_real - y_pred_real

    # 1. Residual Plots (Scatter + Histogram)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residuals vs Predicted
    axes[0].scatter(y_pred_real, residuals, alpha=0.15, color="#1f77b4", s=10)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Price (INR)")
    axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residuals vs. Predicted Price")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    # Residual Distribution
    sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color="#2ca02c")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Residual (INR)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Error Distribution")
    axes[1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(config.figures_dir / "residual_plots.png", dpi=300)
    plt.close(fig)

    # 2. Predicted vs. Actual Plot
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true_real, y_pred_real, alpha=0.15, color="#3470a3", s=10)
    max_val = max(np.max(y_true_real), np.max(y_pred_real)) * 1.05
    ax.plot([0, max_val], [0, max_val], color="crimson", linestyle="--", linewidth=2, label="Identity (y = x)")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel("Actual Price (INR)")
    ax.set_ylabel("Predicted Price (INR)")
    ax.set_title("Predicted vs. Actual Flight Price")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(config.figures_dir / "predicted_vs_actual.png", dpi=300)
    plt.close(fig)

    logger.info("Saved diagnostic residual and predicted-vs-actual figures.")


def plot_learning_curve(
    model: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: ProjectConfig,
) -> None:
    """Generate and save learning curve across increasing training subsets."""
    train_sizes = np.linspace(0.2, 1.0, 5)
    train_maes = []
    val_maes = []

    cat_sub = [c for c in config.categorical_features if c in X_train.columns]
    num_sub = [c for c in config.numerical_features if c in X_train.columns]
    bin_sub = [c for c in config.binary_features if c in X_train.columns]

    preprocessor = build_sklearn_preprocessor(cat_sub, num_sub, bin_sub, use_one_hot=False)
    X_tr_proc = preprocessor.fit_transform(X_train)
    X_va_proc = preprocessor.transform(X_val)

    from lightgbm import LGBMRegressor
    for frac in train_sizes:
        n_samples = int(frac * len(X_tr_proc))
        subset_idx = np.random.RandomState(config.seed).choice(len(X_tr_proc), size=n_samples, replace=False)

        sub_model = LGBMRegressor(n_estimators=100, learning_rate=0.08, num_leaves=63, random_state=config.seed, n_jobs=4, verbose=-1)
        sub_model.fit(X_tr_proc[subset_idx], y_train[subset_idx])

        tr_preds = sub_model.predict(X_tr_proc[subset_idx])
        va_preds = sub_model.predict(X_va_proc)

        tr_mae = evaluate_predictions(y_train[subset_idx], tr_preds, is_log_target=True)["MAE"]
        va_mae = evaluate_predictions(y_val, va_preds, is_log_target=True)["MAE"]

        train_maes.append(tr_mae)
        val_maes.append(va_mae)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([int(f * len(X_train)) for f in train_sizes], train_maes, marker="o", label="Training MAE", color="#1f77b4")
    ax.plot([int(f * len(X_train)) for f in train_sizes], val_maes, marker="s", label="Validation MAE", color="#d62728")
    ax.set_xlabel("Number of Training Samples")
    ax.set_ylabel("MAE (INR)")
    ax.set_title("Learning Curve (Best Gradient Boosting Model)")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(config.figures_dir / "learning_curve.png", dpi=300)
    plt.close(fig)
    logger.info("Saved learning curve figure.")

