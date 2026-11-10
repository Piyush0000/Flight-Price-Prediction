"""
Utility functions for logging, metric calculation, reproducible seeding, and artifact caching.
"""

from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def setup_logger(
    name: str = "flight_pricing",
    log_file: Optional[Path | str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and configure a structured logger with both console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """Enforce reproducibility by fixing seeds across random, numpy, and environment."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Symmetric Mean Absolute Percentage Error (sMAPE) in percentage [0, 200].
    Formula: 100/n * sum(2 * |y_true - y_pred| / (|y_true| + |y_pred| + 1e-8))
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + 1e-8
    diff = np.abs(y_true - y_pred)
    return float(np.mean(diff / denominator) * 100.0)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Percentage Error (MAPE) in percentage."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    non_zero = y_true != 0
    if not np.any(non_zero):
        return 0.0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)


def evaluate_predictions(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    is_log_target: bool = False,
) -> Dict[str, float]:
    """
    Calculate regression metrics (MAE, RMSE, R2, MAPE, sMAPE) on the original price scale.
    If is_log_target is True, y_true and y_pred are transformed via expm1 first.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if is_log_target:
        y_true_arr = np.expm1(y_true_arr)
        y_pred_arr = np.expm1(y_pred_arr)

    # Protect against any negative predictions in real price domain
    y_pred_arr = np.clip(y_pred_arr, a_min=0.0, a_max=None)

    mae = float(mean_absolute_error(y_true_arr, y_pred_arr))
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = float(r2_score(y_true_arr, y_pred_arr))
    mape = calculate_mape(y_true_arr, y_pred_arr)
    smape = calculate_smape(y_true_arr, y_pred_arr)

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "sMAPE": round(smape, 2),
    }


def save_table(df: pd.DataFrame, file_path: Path | str, index: bool = True) -> None:
    """Save a DataFrame as CSV with directory creation and clean formatting."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)


def save_model(model: Any, file_path: Path | str) -> None:
    """Persist a trained model or pipeline to disk using joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))


def load_model(file_path: Path | str) -> Any:
    """Load a persisted model or pipeline from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    return joblib.load(str(path))
