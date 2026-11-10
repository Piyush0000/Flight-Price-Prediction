"""
End-to-end execution script for Explainable Flight Price Prediction research pipeline.
Orchestrates data ingestion, validation, feature engineering, Optuna tuning, 7-model evaluation,
conformal prediction intervals, SHAP explainability, and booking-timing analysis.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.data import load_dataset
from src.evaluate import (
    compare_kfold_vs_groupkfold,
    compute_error_breakdowns,
    evaluate_cv,
    plot_learning_curve,
    plot_residuals_and_diagnostics,
    run_ablation_study,
)
from src.explain import run_explainability_pipeline
from src.features import engineer_features, get_feature_column_names
from src.intervals import run_prediction_interval_pipeline, train_quantile_lightgbm
from src.models import (
    GroupedStackingRegressor,
    create_baseline_models,
    create_catboost_model,
    create_hist_gradient_boosting_pipeline,
    create_lightgbm_pipeline,
    create_linear_pipelines,
    create_tree_pipelines,
)
from src.timing import analyze_booking_timing
from src.tuning import run_hyperparameter_tuning
from src.utils import evaluate_predictions, save_model, save_table, set_seed, setup_logger

logger = setup_logger("run_all", log_file=PROJECT_ROOT / "pipeline.log")


def main() -> None:
    start_time = time.time()
    logger.info("================================================================================")
    logger.info("Starting Explainable Flight Price Prediction Pipeline")
    logger.info("================================================================================")

    # 1. Load Configuration
    config = load_config(PROJECT_ROOT / "config.yaml")
    set_seed(config.seed)

    # 2. Ingestion & Diagnostics
    logger.info("[Step 1/8] Ingesting and validating flight price dataset...")
    try:
        df_clean = load_dataset(config)
    except FileNotFoundError as e:
        logger.error("Execution stopped: Raw CSV dataset missing from data/raw/.")
        sys.exit(1)

    # 3. Feature Engineering
    logger.info("[Step 2/8] Engineering domain features and target transformations...")
    df_feat = engineer_features(df_clean)
    cat_cols, num_cols, bin_cols = get_feature_column_names(config)
    all_feature_cols = cat_cols + num_cols + bin_cols

    # 4. Leakage-Safe Holdout Split (Grouped by Flight Identifier)
    logger.info("[Step 3/8] Performing leakage-safe group split on '%s'...", config.group_column)
    groups = df_feat[config.group_column].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=config.holdout_fraction, random_state=config.seed)
    train_idx, test_idx = next(gss.split(df_feat, groups=groups))

    train_df = df_feat.iloc[train_idx].reset_index(drop=True)
    test_df = df_feat.iloc[test_idx].reset_index(drop=True)

    X_train = train_df[all_feature_cols]
    y_train = train_df[config.log_target_column].to_numpy()
    g_train = train_df[config.group_column].to_numpy()

    X_test = test_df[all_feature_cols]
    y_test = test_df[config.log_target_column].to_numpy()
    g_test = test_df[config.group_column].to_numpy()

    logger.info(
        "Split summary: %d Train/Val rows (%d unique groups), %d Test rows (%d unique groups)",
        len(train_df),
        len(np.unique(g_train)),
        len(test_df),
        len(np.unique(g_test)),
    )

    # 5. Hyperparameter Tuning (Inside Training Groups Only)
    logger.info("[Step 4/8] Running Optuna hyperparameter optimization on training groups...")
    tuned_params = run_hyperparameter_tuning(
        X_train, y_train, g_train, cat_cols, num_cols, bin_cols, config
    )

    # 6. CV Setup and Leakage Comparison
    logger.info("[Step 5/8] Setting up 5-fold GroupKFold cross-validation...")
    gkf = GroupKFold(n_splits=config.cv_folds)
    cv_splits = list(gkf.split(X_train, y_train, groups=g_train))

    # Evaluate GroupKFold vs plain KFold leakage comparison
    lgb_proto = create_lightgbm_pipeline(
        cat_cols, num_cols, bin_cols, seed=config.seed, params=tuned_params.get("lightgbm")
    )
    compare_kfold_vs_groupkfold(lgb_proto, X_train, y_train, g_train, config)

    # 7. Model Benchmarking & Evaluation (7 Models)
    logger.info("[Step 6/8] Benchmarking 7 model architectures...")

    # A. Baselines
    baselines = create_baseline_models(route_col="route")

    # B. Linear Models
    linear_models = create_linear_pipelines(
        cat_cols, num_cols, bin_cols, seed=config.seed, params=tuned_params
    )

    # C. Tree Models
    tree_models = create_tree_pipelines(cat_cols, num_cols, bin_cols, seed=config.seed)

    # D. LightGBM
    lgbm_model = create_lightgbm_pipeline(
        cat_cols, num_cols, bin_cols, seed=config.seed, params=tuned_params.get("lightgbm")
    )

    # E. CatBoost
    catboost_model = create_catboost_model(
        cat_cols, seed=config.seed, params=tuned_params.get("catboost")
    )

    # F. HistGradientBoosting
    hist_gb_model = create_hist_gradient_boosting_pipeline(
        cat_cols, num_cols, bin_cols, seed=config.seed
    )

    # Assemble models dictionary
    models_dict: Dict[str, Any] = {
        "Median_Baseline": baselines["median_baseline"],
        "Route_Mean_Baseline": baselines["route_mean_baseline"],
        "Ridge": linear_models["ridge"],
        "ElasticNet": linear_models["elastic_net"],
        "RandomForest": tree_models["random_forest"],
        "ExtraTrees": tree_models["extra_trees"],
        "LightGBM": lgbm_model,
        "CatBoost": catboost_model,
        "HistGradientBoosting": hist_gb_model,
    }

    # Cross-validation results collector
    model_comparison_records = []
    fitted_models: Dict[str, Any] = {}
    oof_predictions_dict: Dict[str, np.ndarray] = {}

    for name, model in models_dict.items():
        logger.info("Evaluating model: %s ...", name)
        is_cb = name == "CatBoost"
        mean_cv, std_cv, oof_preds = evaluate_cv(model, X_train, y_train, cv_splits, is_catboost=is_cb)
        oof_predictions_dict[name] = oof_preds

        # Fit on full training data
        if is_cb:
            model.fit(X_train, y_train)
            test_preds_log = model.predict(X_test)
        else:
            model.fit(X_train, y_train)
            test_preds_log = model.predict(X_test)

        fitted_models[name] = model
        save_model(model, config.models_dir / f"{name.lower()}.joblib")

        # Evaluate on holdout test set (converted back to real price scale)
        test_metrics = evaluate_predictions(y_test, test_preds_log, is_log_target=True)

        record = {
            "Model": name,
            "CV_MAE_Mean": mean_cv["MAE"],
            "CV_MAE_Std": std_cv["MAE"],
            "CV_RMSE_Mean": mean_cv["RMSE"],
            "CV_RMSE_Std": std_cv["RMSE"],
            "CV_R2_Mean": mean_cv["R2"],
            "CV_R2_Std": std_cv["R2"],
            "CV_MAPE_Mean": mean_cv["MAPE"],
            "CV_sMAPE_Mean": mean_cv["sMAPE"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
            "Test_MAPE": test_metrics["MAPE"],
            "Test_sMAPE": test_metrics["sMAPE"],
        }
        model_comparison_records.append(record)

    # G. Stacking Ensemble
    logger.info("Building Stacking Ensemble with Ridge Meta-Learner...")
    stacking_base_models = {
        "lightgbm": create_lightgbm_pipeline(
            cat_cols, num_cols, bin_cols, seed=config.seed, params=tuned_params.get("lightgbm")
        ),
        "catboost": create_catboost_model(
            cat_cols, seed=config.seed, params=tuned_params.get("catboost")
        ),
        "random_forest": tree_models["random_forest"],
        "hist_gb": create_hist_gradient_boosting_pipeline(
            cat_cols, num_cols, bin_cols, seed=config.seed
        ),
    }

    stacker = GroupedStackingRegressor(stacking_base_models, cat_cols=cat_cols)
    stacker.fit_with_oof(X_train, y_train, g_train, cv_splits)
    fitted_models["Stacking_Ensemble"] = stacker
    save_model(stacker, config.models_dir / "stacking_ensemble.joblib")

    # Evaluate stacker on test set
    stacker_test_preds = stacker.predict(X_test)
    stacker_test_metrics = evaluate_predictions(y_test, stacker_test_preds, is_log_target=True)

    # Stacker CV metrics from meta-learner OOF predictions
    stacker_cv_metrics = evaluate_predictions(y_train, stacker.oof_meta_pred_, is_log_target=True)

    model_comparison_records.append(
        {
            "Model": "Stacking_Ensemble",
            "CV_MAE_Mean": stacker_cv_metrics["MAE"],
            "CV_MAE_Std": 0.0,
            "CV_RMSE_Mean": stacker_cv_metrics["RMSE"],
            "CV_RMSE_Std": 0.0,
            "CV_R2_Mean": stacker_cv_metrics["R2"],
            "CV_R2_Std": 0.0,
            "CV_MAPE_Mean": stacker_cv_metrics["MAPE"],
            "CV_sMAPE_Mean": stacker_cv_metrics["sMAPE"],
            "Test_MAE": stacker_test_metrics["MAE"],
            "Test_RMSE": stacker_test_metrics["RMSE"],
            "Test_R2": stacker_test_metrics["R2"],
            "Test_MAPE": stacker_test_metrics["MAPE"],
            "Test_sMAPE": stacker_test_metrics["sMAPE"],
        }
    )

    comparison_df = pd.DataFrame(model_comparison_records).sort_values(by="Test_MAE").reset_index(drop=True)
    save_table(comparison_df, config.tables_dir / "model_comparison.csv", index=False)
    logger.info("Saved model comparison table.")

    # Select best model for diagnostics and interpretability
    best_model_name = comparison_df.iloc[0]["Model"]
    logger.info("Best performing model identified: %s", best_model_name)
    best_model = fitted_models[best_model_name]

    # Generate best model predictions on test set
    best_test_preds = (
        best_model.predict(X_test)
        if best_model_name != "CatBoost"
        else fitted_models["CatBoost"].predict(X_test)
    )

    # Diagnostic Plots & Residuals
    plot_residuals_and_diagnostics(y_test, best_test_preds, config)
    plot_learning_curve(fitted_models["LightGBM"], X_train, y_train, X_test, y_test, config)

    # Error Breakdowns & Ablations
    compute_error_breakdowns(y_test, best_test_preds, test_df, config)
    run_ablation_study(X_train, y_train, X_test, y_test, config)

    # 8. Prediction Intervals & CQR Calibration
    logger.info("[Step 7/8] Estimating Prediction Intervals and applying CQR Conformal Calibration...")
    # Split test set into calibration and test evaluation groups
    gss_cal = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=config.seed)
    cal_idx, eval_idx = next(gss_cal.split(test_df, groups=g_test))

    X_cal, y_cal = X_test.iloc[cal_idx], y_test[cal_idx]
    X_eval, y_eval = X_test.iloc[eval_idx], y_test[eval_idx]

    run_prediction_interval_pipeline(
        X_train, y_train, X_cal, y_cal, X_eval, y_eval, config
    )

    # 9. Explainability & SHAP Feature Attributions
    logger.info("[Step 8/8] Computing SHAP TreeExplainer, interaction values, and importance correlations...")
    run_explainability_pipeline(
        fitted_models["LightGBM"],
        fitted_models["CatBoost"],
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )

    # 10. Booking-Timing Dynamics
    logger.info("Analyzing Booking-Timing Dynamics across Top Flight Routes...")
    # Train dedicated quantile LightGBM models for the timing curve bands
    cat_sub = [c for c in config.categorical_features if c in X_train.columns]
    num_sub = [c for c in config.numerical_features if c in X_train.columns]
    bin_sub = [c for c in config.binary_features if c in X_train.columns]
    from src.features import build_sklearn_preprocessor
    timing_prep = build_sklearn_preprocessor(cat_sub, num_sub, bin_sub, use_one_hot=False)
    X_tr_proc = timing_prep.fit_transform(X_train)
    q_models = train_quantile_lightgbm(X_tr_proc, y_train, alphas=[0.10, 0.90], seed=config.seed)

    analyze_booking_timing(
        fitted_models["LightGBM"],
        q_models[0.10],
        q_models[0.90],
        df_feat,
        config,
    )

    elapsed_time = round(time.time() - start_time, 2)
    logger.info("================================================================================")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY IN %.2f SECONDS!", elapsed_time)
    logger.info("================================================================================")

    # Print summary table to console
    print("\n" + "=" * 95)
    print("FINAL MODEL BENCHMARK SUMMARY (Sorted by Test MAE - Lower is Better)")
    print("=" * 95)
    print(comparison_df.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
