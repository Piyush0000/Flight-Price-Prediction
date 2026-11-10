"""
Model definitions, pipelines, and stacking ensemble for flight price prediction.
Zero XGBoost dependencies. Implements baselines, regularized linear models, tree ensembles,
LightGBM, CatBoost (native categoricals), HistGradientBoosting, and Stacking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline

from src.features import RouteTargetEncoder, build_sklearn_preprocessor
from src.utils import setup_logger

logger = setup_logger("models")


# ==============================================================================
# Baseline Estimators
# ==============================================================================


class MedianBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the overall median target observed during fit."""

    def __init__(self):
        self.median_val_: float = 0.0

    def fit(self, X: Any, y: np.ndarray | pd.Series) -> "MedianBaselineRegressor":
        self.median_val_ = float(np.median(y))
        return self

    def predict(self, X: Any) -> np.ndarray:
        n_samples = len(X)
        return np.full(shape=(n_samples,), fill_value=self.median_val_)


class RouteMeanBaselineRegressor(BaseEstimator, RegressorMixin):
    """Predicts the target mean for each route, falling back to global mean."""

    def __init__(self, route_col: str = "route"):
        self.route_col = route_col
        self.route_means_: Dict[str, float] = {}
        self.global_mean_: float = 0.0

    def fit(self, X: pd.DataFrame, y: np.ndarray | pd.Series) -> "RouteMeanBaselineRegressor":
        y_series = pd.Series(y, index=X.index) if not isinstance(y, pd.Series) else y
        self.global_mean_ = float(y_series.mean())
        self.route_means_ = y_series.groupby(X[self.route_col]).mean().to_dict()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (
            X[self.route_col]
            .map(self.route_means_)
            .fillna(self.global_mean_)
            .to_numpy(dtype=float)
        )


# ==============================================================================
# Model Factory Functions
# ==============================================================================


def create_baseline_models(route_col: str = "route") -> Dict[str, BaseEstimator]:
    """Return dictionary of baseline estimators."""
    return {
        "median_baseline": MedianBaselineRegressor(),
        "route_mean_baseline": RouteMeanBaselineRegressor(route_col=route_col),
    }


def create_linear_pipelines(
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    seed: int = 42,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Pipeline]:
    """Create Ridge and ElasticNet regression pipelines with scaling and one-hot encoding."""
    params = params or {}
    ridge_params = params.get("ridge", {"alpha": 1.0, "random_state": seed})
    enet_params = params.get(
        "elastic_net",
        {"alpha": 0.05, "l1_ratio": 0.5, "max_iter": 2000, "random_state": seed},
    )

    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=True)

    ridge_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", Ridge(**ridge_params)),
        ]
    )

    elastic_net_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ElasticNet(**enet_params)),
        ]
    )

    return {"ridge": ridge_pipe, "elastic_net": elastic_net_pipe}


def create_tree_pipelines(
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    seed: int = 42,
    params: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Pipeline]:
    """Create RandomForest and ExtraTrees pipelines with ordinal encoding for scalability."""
    params = params or {}
    rf_params = params.get(
        "random_forest",
        {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "n_jobs": 4,
            "random_state": seed,
        },
    )
    et_params = params.get(
        "extra_trees",
        {
            "n_estimators": 100,
            "max_depth": 16,
            "min_samples_leaf": 4,
            "n_jobs": 4,
            "random_state": seed,
        },
    )

    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=False)

    rf_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(**rf_params)),
        ]
    )

    et_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", ExtraTreesRegressor(**et_params)),
        ]
    )

    return {"random_forest": rf_pipe, "extra_trees": et_pipe}


def create_lightgbm_pipeline(
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create LightGBM regressor pipeline with ordinal categorical encoding."""
    lgb_params = {
        "n_estimators": 250,
        "learning_rate": 0.08,
        "num_leaves": 63,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 4,
        "verbose": -1,
    }
    if params:
        lgb_params.update(params)

    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=False)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LGBMRegressor(**lgb_params)),
        ]
    )
    return pipe


def create_catboost_model(
    cat_cols: List[str],
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> CatBoostRegressor:
    """Create CatBoost regressor with native categorical features support."""
    cb_params = {
        "iterations": 350,
        "learning_rate": 0.08,
        "depth": 6,
        "random_seed": seed,
        "verbose": 0,
        "cat_features": cat_cols,
        "thread_count": 4,
    }
    if params:
        cb_params.update(params)
    return CatBoostRegressor(**cb_params)


def create_hist_gradient_boosting_pipeline(
    cat_cols: List[str],
    num_cols: List[str],
    bin_cols: List[str],
    seed: int = 42,
    params: Optional[Dict[str, Any]] = None,
) -> Pipeline:
    """Create sklearn HistGradientBoostingRegressor pipeline."""
    hgb_params = {
        "max_iter": 200,
        "learning_rate": 0.08,
        "max_leaf_nodes": 63,
        "min_samples_leaf": 20,
        "random_state": seed,
    }
    if params:
        hgb_params.update(params)

    preprocessor = build_sklearn_preprocessor(cat_cols, num_cols, bin_cols, use_one_hot=False)

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", HistGradientBoostingRegressor(**hgb_params)),
        ]
    )
    return pipe


# ==============================================================================
# Grouped Stacking Regressor
# ==============================================================================


class GroupedStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble that trains a meta-regressor (Ridge) on out-of-fold predictions
    generated by base learners (LightGBM, CatBoost, RandomForest, HistGradientBoosting).
    """

    def __init__(
        self,
        base_models: Dict[str, Any],
        meta_regressor: Optional[BaseEstimator] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        self.base_models = base_models
        self.meta_regressor = meta_regressor or Ridge(alpha=1.0, positive=True)
        self.cat_cols = cat_cols or []
        self.fitted_base_models_: Dict[str, Any] = {}

    def fit_with_oof(
        self,
        X: pd.DataFrame,
        y: np.ndarray | pd.Series,
        groups: np.ndarray | pd.Series,
        cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    ) -> "GroupedStackingRegressor":
        """
        Train base models across CV folds to collect out-of-fold predictions,
        fit the meta-learner on OOF predictions, and fit final base models on full training data.
        """
        n_samples = len(X)
        model_names = list(self.base_models.keys())
        oof_predictions = {name: np.zeros(n_samples) for name in model_names}

        logger.info("Computing grouped Out-Of-Fold predictions for Stacking Ensemble...")
        for fold, (train_idx, val_idx) in enumerate(cv_splits, 1):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va = X.iloc[val_idx]

            for name, model in self.base_models.items():
                # Clone/instantiate or re-fit for this fold
                from sklearn.base import clone
                if name == "catboost":
                    fold_model = CatBoostRegressor(
                        iterations=model.get_param("iterations"),
                        learning_rate=model.get_param("learning_rate"),
                        depth=model.get_param("depth"),
                        random_seed=model.get_param("random_seed") + fold,
                        verbose=0,
                        cat_features=self.cat_cols,
                        thread_count=4,
                    )
                    fold_model.fit(X_tr, y_tr)
                    preds = fold_model.predict(X_va)
                else:
                    fold_model = clone(model)
                    fold_model.fit(X_tr, y_tr)
                    preds = fold_model.predict(X_va)

                oof_predictions[name][val_idx] = preds

        # Build meta feature matrix
        meta_X = np.column_stack([oof_predictions[name] for name in model_names])
        self.meta_regressor.fit(meta_X, y)
        logger.info(
            "Meta-learner weights: %s",
            dict(zip(model_names, np.round(getattr(self.meta_regressor, "coef_", [1.0] * len(model_names)), 4))),
        )

        # Refit base models on full training data
        logger.info("Refitting base models on full training set...")
        for name, model in self.base_models.items():
            if name == "catboost":
                model.fit(X, y)
                self.fitted_base_models_[name] = model
            else:
                from sklearn.base import clone
                m = clone(model)
                m.fit(X, y)
                self.fitted_base_models_[name] = m

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions using fitted base models and meta-learner."""
        preds = []
        for name in self.base_models.keys():
            model = self.fitted_base_models_[name]
            p = model.predict(X)
            preds.append(p)

        meta_X = np.column_stack(preds)
        return self.meta_regressor.predict(meta_X)
