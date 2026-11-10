"""
Data ingestion, validation, and cleaning module for flight price prediction.
Includes defensive checks for raw CSV files, flexible column mapping, and diagnostics logging.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import ProjectConfig
from src.utils import save_table, setup_logger

logger = setup_logger("data_processing")


def find_raw_csv(data_dir: Path, target_filename: Optional[str] = None) -> Path:
    """
    Search for a CSV file in data_dir.
    If no file is found, print explicit user guidance and exit cleanly.
    """
    if not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)

    # Check for target filename first
    if target_filename:
        target_path = data_dir / target_filename
        if target_path.exists() and target_path.is_file():
            return target_path

    # Check for any .csv file in data_dir
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        return csv_files[0]

    # If no CSV exists, halt execution with explicit instructions
    error_msg = (
        "\n"
        + "=" * 80
        + "\n"
        + "[DATA ERROR] No flight price dataset found in data/raw/.\n"
        + f"Expected location: {data_dir.resolve()}\n"
        + "Please place your raw flight price CSV (e.g., Clean_Dataset.csv) in this folder.\n"
        + "Required columns:\n"
        + "  - airline, flight, source_city, departure_time, stops,\n"
        + "  - arrival_time, destination_city, class, duration, days_left, price\n"
        + "=" * 80
        + "\n"
    )
    logger.error(error_msg)
    print(error_msg, file=sys.stderr)
    raise FileNotFoundError(
        f"Missing dataset: please put your flight price CSV in {data_dir.resolve()}"
    )


def inspect_and_log_raw_data(
    df: pd.DataFrame, tables_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate and save initial diagnostics:
    - Schema & Data Types
    - Missing value counts & percentages
    - Target (price) summary statistics
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Schema & Data Types
    schema_df = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(t) for t in df.dtypes],
            "non_null_count": df.notnull().sum().values,
            "unique_count": [df[c].nunique() for c in df.columns],
        }
    )
    save_table(schema_df, tables_dir / "data_schema.csv", index=False)

    # 2. Missing Values
    missing_counts = df.isnull().sum()
    missing_df = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": missing_counts.values,
            "missing_percentage": (missing_counts.values / len(df) * 100).round(2),
        }
    )
    save_table(missing_df, tables_dir / "missing_values.csv", index=False)

    # 3. Price Summary Statistics (if price is present)
    price_cols = [c for c in df.columns if "price" in c.lower()]
    if price_cols:
        p_col = price_cols[0]
        price_stats = df[p_col].describe().to_frame(name="price_summary").reset_index()
        price_stats.rename(columns={"index": "statistic"}, inplace=True)
        save_table(price_stats, tables_dir / "price_summary.csv", index=False)
    else:
        price_stats = pd.DataFrame()

    logger.info("Raw data diagnostics successfully written to %s", tables_dir)
    return schema_df, missing_df, price_stats


def clean_flight_data(df: pd.DataFrame, config: ProjectConfig) -> pd.DataFrame:
    """
    Clean raw dataset according to pipeline specifications:
    1. Drop index/unnamed columns
    2. Apply column mapping
    3. Remove exact duplicates
    4. Validate and drop non-positive or missing prices
    5. Ensure proper data types
    6. Log all decisions to results/tables/data_cleaning_log.csv
    """
    initial_rows = len(df)
    cleaning_log = []

    # Drop unnamed columns (such as Kaggle indices)
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed") or c == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        cleaning_log.append(
            {
                "action": "drop_unnamed_columns",
                "details": f"Removed index columns: {unnamed_cols}",
                "rows_affected": 0,
            }
        )

    # Apply column mapping
    mapping = {k: v for k, v in config.column_mapping.items() if k in df.columns}
    df = df.rename(columns=mapping)
    cleaning_log.append(
        {
            "action": "column_mapping",
            "details": f"Mapped {len(mapping)} columns to standard schema",
            "rows_affected": 0,
        }
    )

    # Check for duplicate rows
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
    cleaning_log.append(
        {
            "action": "drop_exact_duplicates",
            "details": f"Removed exact duplicates",
            "rows_affected": duplicate_count,
        }
    )

    # Validate price
    target_col = config.target_column
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    invalid_price_mask = df[target_col].isnull() | (df[target_col] <= 0)
    invalid_price_count = int(invalid_price_mask.sum())
    if invalid_price_count > 0:
        df = df[~invalid_price_mask].reset_index(drop=True)
    cleaning_log.append(
        {
            "action": "filter_positive_price",
            "details": f"Removed missing or non-positive price records (price <= 0)",
            "rows_affected": invalid_price_count,
        }
    )

    # Validate duration and days_left
    if "duration" in df.columns:
        df["duration"] = pd.to_numeric(df["duration"], errors="coerce")
    if "days_left" in df.columns:
        df["days_left"] = pd.to_numeric(df["days_left"], errors="coerce").astype(int)

    # Add log price target
    df[config.log_target_column] = np.log1p(df[target_col].astype(float))

    final_rows = len(df)
    cleaning_log.append(
        {
            "action": "summary",
            "details": f"Initial rows: {initial_rows}, Final clean rows: {final_rows}",
            "rows_affected": initial_rows - final_rows,
        }
    )

    # Save cleaning log table
    log_df = pd.DataFrame(cleaning_log)
    save_table(log_df, config.tables_dir / "data_cleaning_log.csv", index=False)
    logger.info("Cleaned data: %d -> %d rows (-%d rows)", initial_rows, final_rows, initial_rows - final_rows)

    return df


def load_dataset(config: ProjectConfig) -> pd.DataFrame:
    """
    Complete ingestion pipeline:
    Find CSV -> Load -> Inspect & Save Initial Tables -> Clean -> Persist Cleaned Cache -> Return DataFrame.
    """
    csv_path = find_raw_csv(config.raw_data_dir, config.raw_csv_filename)
    logger.info("Loading raw dataset from %s ...", csv_path)
    df_raw = pd.read_csv(csv_path)

    # Diagnostics on raw data
    inspect_and_log_raw_data(df_raw, config.tables_dir)

    # Clean data
    df_clean = clean_flight_data(df_raw, config)

    # Save processed dataset
    config.processed_data_dir.mkdir(parents=True, exist_ok=True)
    try:
        df_clean.to_parquet(config.processed_data_file, index=False)
        logger.info("Saved processed dataset to %s", config.processed_data_file)
    except Exception as e:
        csv_fallback = config.processed_data_dir / "cleaned_flights.csv"
        df_clean.to_csv(csv_fallback, index=False)
        logger.warning("Parquet save failed (%s), saved to CSV fallback: %s", e, csv_fallback)

    return df_clean

