"""
Configuration management module for the flight price prediction research project.
Loads settings, paths, and hyperparameters from config.yaml.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


class ProjectConfig:
    """Class to parse, validate, and access configuration parameters."""

    def __init__(self, config_dict: Dict[str, Any], project_root: Optional[Path] = None):
        self._raw_config = config_dict
        self.project_root = project_root or Path(__file__).resolve().parent.parent

        # Project metadata
        self.project_name: str = self._raw_config.get("project", {}).get(
            "name", "Flight Price Prediction"
        )
        self.seed: int = int(self._raw_config.get("project", {}).get("seed", 42))

        # Paths
        paths_cfg = self._raw_config.get("paths", {})
        self.raw_data_dir = self.project_root / paths_cfg.get("raw_data_dir", "data/raw")
        self.processed_data_dir = (
            self.project_root / paths_cfg.get("processed_data_dir", "data/processed")
        )
        self.models_dir = self.project_root / paths_cfg.get("models_dir", "models")
        self.figures_dir = self.project_root / paths_cfg.get("figures_dir", "results/figures")
        self.tables_dir = self.project_root / paths_cfg.get("tables_dir", "results/tables")
        self.raw_csv_filename: str = paths_cfg.get("raw_csv_filename", "Clean_Dataset.csv")
        self.processed_data_file = (
            self.project_root
            / paths_cfg.get("processed_data_file", "data/processed/cleaned_flights.parquet")
        )

        # Data schema & column mapping
        data_cfg = self._raw_config.get("data", {})
        self.column_mapping: Dict[str, str] = data_cfg.get("column_mapping", {})
        self.group_column: str = data_cfg.get("group_column", "flight")
        self.target_column: str = data_cfg.get("target_column", "price")
        self.log_target_column: str = data_cfg.get("log_target_column", "log_price")

        # Features
        feat_cfg = self._raw_config.get("features", {})
        self.categorical_features: List[str] = feat_cfg.get("categorical_features", [])
        self.numerical_features: List[str] = feat_cfg.get("numerical_features", [])
        self.binary_features: List[str] = feat_cfg.get("binary_features", [])

        # Validation
        val_cfg = self._raw_config.get("validation", {})
        self.cv_folds: int = int(val_cfg.get("cv_folds", 5))
        self.holdout_fraction: float = float(val_cfg.get("holdout_fraction", 0.20))
        self.group_col: str = val_cfg.get("group_col", "flight")
        self.run_kfold_leakage_comparison: bool = bool(
            val_cfg.get("run_kfold_leakage_comparison", True)
        )

        # Tuning
        tune_cfg = self._raw_config.get("tuning", {})
        self.tuning_enabled: bool = bool(tune_cfg.get("enabled", True))
        self.n_trials: int = int(tune_cfg.get("n_trials", 25))
        self.tuning_cv_folds: int = int(tune_cfg.get("cv_folds", 3))
        self.tuning_timeout: int = int(tune_cfg.get("timeout", 300))

        # Models
        models_cfg = self._raw_config.get("models", {})
        self.baselines: List[str] = models_cfg.get("baselines", [])
        self.linear_models: List[str] = models_cfg.get("linear_models", [])
        self.tree_models: List[str] = models_cfg.get("tree_models", [])
        self.ensemble: List[str] = models_cfg.get("ensemble", [])

        # Intervals
        int_cfg = self._raw_config.get("intervals", {})
        self.interval_alphas: List[float] = [float(a) for a in int_cfg.get("alphas", [0.1, 0.5, 0.9])]
        self.target_coverage: float = float(int_cfg.get("target_coverage", 0.80))
        self.calibration_size: float = float(int_cfg.get("calibration_size", 0.25))

        # Explainability
        exp_cfg = self._raw_config.get("explainability", {})
        self.shap_sample_size: int = int(exp_cfg.get("sample_size", 1000))
        self.interaction_pairs: List[List[str]] = exp_cfg.get(
            "interaction_pairs", [["days_left", "class"], ["stops", "duration"]]
        )

        # Booking Timing
        timing_cfg = self._raw_config.get("timing_analysis", {})
        self.min_days_left: int = int(timing_cfg.get("min_days_left", 1))
        self.max_days_left: int = int(timing_cfg.get("max_days_left", 50))
        self.top_routes_count: int = int(timing_cfg.get("top_routes_count", 5))

    def ensure_directories(self) -> None:
        """Create output directories if they do not exist."""
        for directory in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.figures_dir,
            self.tables_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


def load_config(config_path: Optional[str | Path] = None) -> ProjectConfig:
    """Load config.yaml from project root or custom path."""
    if config_path is None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path).resolve()
        project_root = config_path.parent

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    cfg = ProjectConfig(config_dict, project_root=project_root)
    cfg.ensure_directories()
    return cfg

