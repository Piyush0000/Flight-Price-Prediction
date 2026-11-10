"""
Hyperparameter optimization module using Optuna and leakage-safe Grouped Cross-Validation.
Tuning operates strictly within the training set, optimizing original-scale MAE.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from src.config import ProjectConfig
from src.features import build_sklearn_preprocessor
from src.utils import save_table, setup_logger

logger = setup_logger("hyperparameter_tuning")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    config: ProjectConfig,
) -> Dict[str, Any]:
    """Tune Ridge regularization alpha using grouped CV."""
    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=True)
    X_proc = preprocessor.fit_transform(X_train)

    gkf = GroupKFold(n_splits=config.tuning_cv_folds)
    splits = list(gkf.split(X_proc, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        alpha = trial.suggest_float("alpha", 1e-3, 1e2, log=True)
        cv_maes = []
        for tr_idx, va_idx in splits:
            reg = Ridge(alpha=alpha, random_state=config.seed)
            reg.fit(X_proc[tr_idx], y_train[tr_idx])
            pred_log = reg.predict(X_proc[va_idx])
            # evaluate on real price scale
            mae = mean_absolute_error(np.expm1(y_train[va_idx]), np.expm1(pred_log))
            cv_maes.append(mae)
        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=min(15, config.n_trials), timeout=config.tuning_timeout)
    logger.info("Best Ridge MAE: %.2f with params: %s", study.best_value, study.best_params)
    return study.best_params


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    config: ProjectConfig,
) -> Dict[str, Any]:
    """Tune LightGBM hyperparameters with grouped CV."""
    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=False)
    X_proc = preprocessor.fit_transform(X_train)

    gkf = GroupKFold(n_splits=config.tuning_cv_folds)
    splits = list(gkf.split(X_proc, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 300, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "random_state": config.seed,
            "n_jobs": 4,
            "verbose": -1,
        }
        cv_maes = []
        for tr_idx, va_idx in splits:
            reg = LGBMRegressor(**params)
            reg.fit(X_proc[tr_idx], y_train[tr_idx])
            pred_log = reg.predict(X_proc[va_idx])
            mae = mean_absolute_error(np.expm1(y_train[va_idx]), np.expm1(pred_log))
            cv_maes.append(mae)
        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=config.n_trials, timeout=config.tuning_timeout)
    logger.info("Best LightGBM MAE: %.2f with params: %s", study.best_value, study.best_params)
    return study.best_params


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    cat_cols: List[str],
    config: ProjectConfig,
) -> Dict[str, Any]:
    """Tune CatBoost hyperparameters with native categoricals and grouped CV."""
    gkf = GroupKFold(n_splits=config.tuning_cv_folds)
    splits = list(gkf.split(X_train, y_train, groups=groups))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 200, 350, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.04, 0.12, log=True),
            "depth": trial.suggest_int("depth", 4, 7),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_seed": config.seed,
            "verbose": 0,
            "thread_count": 4,
            "cat_features": cat_cols,
        }
        cv_maes = []
        for tr_idx, va_idx in splits:
            reg = CatBoostRegressor(**params)
            reg.fit(X_train.iloc[tr_idx], y_train[tr_idx])
            pred_log = reg.predict(X_train.iloc[va_idx])
            mae = mean_absolute_error(np.expm1(y_train[va_idx]), np.expm1(pred_log))
            cv_maes.append(mae)
        return float(np.mean(cv_maes))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.seed))
    study.optimize(objective, n_trials=min(15, config.n_trials), timeout=config.tuning_timeout)
    logger.info("Best CatBoost MAE: %.2f with params: %s", study.best_value, study.best_params)
    return study.best_params


def run_hyperparameter_tuning(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    config: ProjectConfig,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute hyperparameter tuning for Ridge, LightGBM, and CatBoost.
    Returns dictionary of best parameters and saves a summary table.
    """
    if not config.tuning_enabled:
        logger.info("Hyperparameter tuning is disabled in config. Using defaults.")
        return {}

    logger.info("Starting hyperparameter optimization with Optuna...")

    # Subsample if dataset is large to maintain fast, responsive tuning
    n_samples = len(X_train)
    if n_samples > 35000:
        logger.info("Subsampling %d -> 30000 rows for hyperparameter tuning to ensure efficiency.", n_samples)
        sample_indices = np.random.RandomState(config.seed).choice(n_samples, size=30000, replace=False)
        X_sub = X_train.iloc[sample_indices].reset_index(drop=True)
        y_sub = y_train[sample_indices]
        g_sub = groups[sample_indices]
    else:
        X_sub, y_sub, g_sub = X_train, y_train, groups

    best_ridge = tune_ridge(X_sub, y_sub, g_sub, cat_cols, num_cols, bin_cols, config)
    best_lgb = tune_lightgbm(X_sub, y_sub, g_sub, cat_cols, num_cols, bin_cols, config)
    best_cb = tune_catboost(X_sub, y_sub, g_sub, cat_cols, config)

    tuning_summary = [
        {"model": "ridge", "best_params": str(best_ridge)},
        {"model": "lightgbm", "best_params": str(best_lgb)},
        {"model": "catboost", "best_params": str(best_cb)},
    ]
    save_table(pd.DataFrame(tuning_summary), config.tables_dir / "tuning_results.csv", index=False)

    return {
        "ridge": best_ridge,
        "lightgbm": best_lgb,
        "catboost": best_cb,
    }

