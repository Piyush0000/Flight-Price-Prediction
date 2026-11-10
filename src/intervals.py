"""
Uncertainty quantification via Quantile Regression and Conformalized Quantile Regression (CQR).
Trains pinball-loss models (alphas 0.1, 0.5, 0.9) and calibrates prediction intervals on a held-out group set
to guarantee the 80% coverage target.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from src.config import ProjectConfig
from src.features import build_sklearn_preprocessor
from src.utils import save_table, setup_logger

logger = setup_logger("prediction_intervals")


class ConformalizedQuantileRegressor:
    """
    Applies Conformalized Quantile Regression (CQR) on top of lower and upper quantile estimators.
    Computes non-conformity scores on a calibration set to adjust interval bounds for exact coverage.
    """

    def __init__(self, target_coverage: float = 0.80):
        self.target_coverage = target_coverage
        self.conformal_offset_: float = 0.0

    def calibrate(
        self,
        y_cal_real: np.ndarray,
        q_low_real: np.ndarray,
        q_high_real: np.ndarray,
    ) -> float:
        """
        Compute non-conformity score E_i = max(q_low - y, y - q_high) and its empirical quantile.
        """
        scores = np.maximum(q_low_real - y_cal_real, y_cal_real - q_high_real)
        n = len(scores)
        # Quantile level with finite sample correction: ceil((n + 1) * target_coverage) / n
        q_level = min(1.0, np.ceil((n + 1) * self.target_coverage) / n)
        self.conformal_offset_ = float(np.quantile(scores, q_level))
        logger.info(
            "CQR Calibration: target coverage=%.2f, offset=%.2f INR across %d samples.",
            self.target_coverage,
            self.conformal_offset_,
            n,
        )
        return self.conformal_offset_

    def predict_intervals(
        self,
        q_low_real: np.ndarray,
        q_high_real: np.ndarray,
        apply_calibration: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate calibrated or uncalibrated lower and upper bounds."""
        offset = self.conformal_offset_ if apply_calibration else 0.0
        low = np.clip(q_low_real - offset, a_min=0.0, a_max=None)
        high = np.clip(q_high_real + offset, a_min=0.0, a_max=None)
        return low, high


def compute_interval_metrics(
    y_real: np.ndarray,
    low_real: np.ndarray,
    high_real: np.ndarray,
) -> Dict[str, float]:
    """Compute PICP (coverage) and MPIW (mean interval width)."""
    covered = (y_real >= low_real) & (y_real <= high_real)
    picp = float(np.mean(covered) * 100.0)
    mpiw = float(np.mean(high_real - low_real))
    return {
        "PICP": round(picp, 2),
        "MPIW": round(mpiw, 2),
    }


def train_quantile_lightgbm(
    X_tr_proc: np.ndarray,
    y_tr: np.ndarray,
    alphas: List[float],
    seed: int = 42,
) -> Dict[float, LGBMRegressor]:
    """Train pinball quantile loss regressors using LightGBM."""
    models = {}
    for alpha in alphas:
        logger.info("Training Quantile LightGBM (alpha=%.2f)...", alpha)
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=150,
            learning_rate=0.08,
            num_leaves=63,
            random_state=seed,
            n_jobs=4,
            verbose=-1,
        )
        model.fit(X_tr_proc, y_tr)
        models[alpha] = model
    return models


def train_quantile_catboost(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    cat_cols: List[str],
    alphas: List[float],
    seed: int = 42,
) -> Dict[float, CatBoostRegressor]:
    """Train pinball quantile loss regressors using CatBoost."""
    models = {}
    for alpha in alphas:
        logger.info("Training Quantile CatBoost (alpha=%.2f)...", alpha)
        loss_fn = f"Quantile:alpha={alpha}"
        model = CatBoostRegressor(
            loss_function=loss_fn,
            iterations=200,
            learning_rate=0.08,
            depth=6,
            random_seed=seed,
            cat_features=cat_cols,
            verbose=0,
            thread_count=4,
        )
        model.fit(X_tr, y_tr)
        models[alpha] = model
    return models


def run_prediction_interval_pipeline(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_cal: pd.DataFrame,
    y_cal: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    config: ProjectConfig,
) -> pd.DataFrame:
    """
    Execute full prediction interval workflow:
    1. Preprocess data
    2. Train Quantile LightGBM and Quantile CatBoost for alphas [0.1, 0.5, 0.9]
    3. Calibrate intervals via CQR on held-out calibration groups
    4. Evaluate PICP and MPIW before vs after calibration on test set
    5. Save results table and visualization plot
    """
    cat_sub = [c for c in config.categorical_features if c in X_train.columns]
    num_sub = [c for c in config.numerical_features if c in X_train.columns]
    bin_sub = [c for c in config.binary_features if c in X_train.columns]

    preprocessor = build_sklearn_preprocessor(cat_sub, num_sub, bin_sub, use_one_hot=False)
    X_tr_proc = preprocessor.fit_transform(X_train)
    X_cal_proc = preprocessor.transform(X_cal)
    X_te_proc = preprocessor.transform(X_test)

    y_cal_real = np.expm1(y_cal)
    y_te_real = np.expm1(y_test)

    # 1. LightGBM Quantile Models
    lgb_models = train_quantile_lightgbm(X_tr_proc, y_train, config.interval_alphas, seed=config.seed)
    lgb_cal_low = np.expm1(lgb_models[0.10].predict(X_cal_proc))
    lgb_cal_high = np.expm1(lgb_models[0.90].predict(X_cal_proc))
    lgb_te_low = np.expm1(lgb_models[0.10].predict(X_te_proc))
    lgb_te_med = np.expm1(lgb_models[0.50].predict(X_te_proc))
    lgb_te_high = np.expm1(lgb_models[0.90].predict(X_te_proc))

    cqr_lgb = ConformalizedQuantileRegressor(target_coverage=config.target_coverage)
    cqr_lgb.calibrate(y_cal_real, lgb_cal_low, lgb_cal_high)
    lgb_uncal_low, lgb_uncal_high = cqr_lgb.predict_intervals(lgb_te_low, lgb_te_high, apply_calibration=False)
    lgb_cal_low_te, lgb_cal_high_te = cqr_lgb.predict_intervals(lgb_te_low, lgb_te_high, apply_calibration=True)

    metrics_lgb_uncal = compute_interval_metrics(y_te_real, lgb_uncal_low, lgb_uncal_high)
    metrics_lgb_cal = compute_interval_metrics(y_te_real, lgb_cal_low_te, lgb_cal_high_te)

    # 2. CatBoost Quantile Models
    cb_models = train_quantile_catboost(X_train, y_train, cat_sub, config.interval_alphas, seed=config.seed)
    cb_cal_low = np.expm1(cb_models[0.10].predict(X_cal))
    cb_cal_high = np.expm1(cb_models[0.90].predict(X_cal))
    cb_te_low = np.expm1(cb_models[0.10].predict(X_test))
    cb_te_high = np.expm1(cb_models[0.90].predict(X_test))

    cqr_cb = ConformalizedQuantileRegressor(target_coverage=config.target_coverage)
    cqr_cb.calibrate(y_cal_real, cb_cal_low, cb_cal_high)
    cb_uncal_low, cb_uncal_high = cqr_cb.predict_intervals(cb_te_low, cb_te_high, apply_calibration=False)
    cb_cal_low_te, cb_cal_high_te = cqr_cb.predict_intervals(cb_te_low, cb_te_high, apply_calibration=True)

    metrics_cb_uncal = compute_interval_metrics(y_te_real, cb_uncal_low, cb_uncal_high)
    metrics_cb_cal = compute_interval_metrics(y_te_real, cb_cal_low_te, cb_cal_high_te)

    # Build summary table
    table_data = [
        {
            "Method": "Quantile LightGBM (Raw)",
            "Target_Coverage_%": config.target_coverage * 100,
            "Empirical_PICP_%": metrics_lgb_uncal["PICP"],
            "MPIW_INR": metrics_lgb_uncal["MPIW"],
            "CQR_Calibrated": "No",
        },
        {
            "Method": "Quantile LightGBM + CQR",
            "Target_Coverage_%": config.target_coverage * 100,
            "Empirical_PICP_%": metrics_lgb_cal["PICP"],
            "MPIW_INR": metrics_lgb_cal["MPIW"],
            "CQR_Calibrated": "Yes",
        },
        {
            "Method": "Quantile CatBoost (Raw)",
            "Target_Coverage_%": config.target_coverage * 100,
            "Empirical_PICP_%": metrics_cb_uncal["PICP"],
            "MPIW_INR": metrics_cb_uncal["MPIW"],
            "CQR_Calibrated": "No",
        },
        {
            "Method": "Quantile CatBoost + CQR",
            "Target_Coverage_%": config.target_coverage * 100,
            "Empirical_PICP_%": metrics_cb_cal["PICP"],
            "MPIW_INR": metrics_cb_cal["MPIW"],
            "CQR_Calibrated": "Yes",
        },
    ]

    intervals_df = pd.DataFrame(table_data)
    save_table(intervals_df, config.tables_dir / "prediction_intervals.csv", index=False)
    logger.info("Prediction interval metrics saved to tables.")

    # Plot sample intervals
    plot_prediction_intervals(
        y_te_real,
        lgb_te_med,
        lgb_cal_low_te,
        lgb_cal_high_te,
        config,
        sample_size=60,
    )

    return intervals_df


def plot_prediction_intervals(
    y_true: np.ndarray,
    y_pred_med: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    config: ProjectConfig,
    sample_size: int = 60,
) -> None:
    """Plot prediction intervals for a representative sample of flights."""
    sample_idx = np.random.RandomState(config.seed).choice(len(y_true), size=sample_size, replace=False)
    order = np.argsort(y_true[sample_idx])
    sorted_idx = sample_idx[order]

    x_axis = np.arange(sample_size)
    y_sampled = y_true[sorted_idx]
    low_sampled = low[sorted_idx]
    high_sampled = high[sorted_idx]
    med_sampled = y_pred_med[sorted_idx]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.fill_between(x_axis, low_sampled, high_sampled, color="#a1c9f4", alpha=0.5, label="80% Calibrated Prediction Interval")
    ax.plot(x_axis, med_sampled, color="#0b559f", linewidth=1.5, linestyle="--", label="Median Estimate (q=0.50)")
    ax.scatter(x_axis, y_sampled, color="#d62728", s=25, zorder=5, label="Actual Price (INR)")

    ax.set_xlabel("Flight Sample Index (Sorted by Price)")
    ax.set_ylabel("Price (INR)")
    ax.set_title("Calibrated 80% Prediction Intervals on Holdout Flights (CQR + LightGBM)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    fig.savefig(config.figures_dir / "prediction_intervals.png", dpi=300)
    plt.close(fig)
    logger.info("Prediction interval figure saved to results/figures/prediction_intervals.png.")

