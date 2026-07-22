"""Validation module for Retail Sales Forecasting ETL pipeline.

This module validates the datasets extracted by extract_data(),
checking for missing values, duplicates, invalid values, and
ensuring date columns are properly typed.
"""

import logging
from typing import Dict

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATE_COLUMN_CANDIDATES = ["date"]


def _check_not_empty(name: str, df: pd.DataFrame) -> None:
    """Check that a DataFrame is not empty.

    Args:
        name: Name of the dataset.
        df: DataFrame to check.
    """
    if df.empty:
        logger.warning("Dataset '%s' is empty.", name)
    else:
        logger.info("Dataset '%s' is not empty.", name)


def _log_shape_and_columns(name: str, df: pd.DataFrame) -> None:
    """Log the shape and column names of a DataFrame.

    Args:
        name: Name of the dataset.
        df: DataFrame to inspect.
    """
    logger.info("Dataset '%s' shape: %s", name, df.shape)
    logger.info("Dataset '%s' columns: %s", name, list(df.columns))


def _count_missing_values(name: str, df: pd.DataFrame) -> None:
    """Count and log missing values per column.

    Args:
        name: Name of the dataset.
        df: DataFrame to inspect.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("Dataset '%s' has no missing values.", name)
    else:
        logger.info(
            "Dataset '%s' missing values per column:\n%s", name, missing
        )


def _count_duplicate_rows(name: str, df: pd.DataFrame) -> None:
    """Count and log duplicate rows in a DataFrame.

    Args:
        name: Name of the dataset.
        df: DataFrame to inspect.
    """
    duplicate_count = df.duplicated().sum()
    logger.info("Dataset '%s' duplicate rows: %d", name, duplicate_count)


def _convert_date_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Convert recognized date columns to datetime dtype.

    Args:
        name: Name of the dataset.
        df: DataFrame to convert.

    Returns:
        DataFrame with date columns converted to datetime, where present.
    """
    for col in DATE_COLUMN_CANDIDATES:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                logger.info(
                    "Converted column '%s' to datetime in dataset '%s'.",
                    col,
                    name,
                )
            except Exception:
                logger.exception(
                    "Failed to convert column '%s' to datetime in "
                    "dataset '%s'.",
                    col,
                    name,
                )
    return df


def _check_non_negative(name: str, df: pd.DataFrame, column: str) -> None:
    """Check that a numeric column contains no negative values.

    Args:
        name: Name of the dataset.
        df: DataFrame to check.
        column: Column name to validate.
    """
    if column not in df.columns:
        logger.warning(
            "Column '%s' not found in dataset '%s'; skipping check.",
            column,
            name,
        )
        return

    negative_count = (df[column] < 0).sum()
    if negative_count > 0:
        logger.warning(
            "Dataset '%s' has %d negative values in column '%s'.",
            name,
            negative_count,
            column,
        )
    else:
        logger.info(
            "Dataset '%s' column '%s' has no negative values.",
            name,
            column,
        )


def validate_data(
    dataframes: Dict[str, pd.DataFrame]
) -> Dict[str, pd.DataFrame]:
    """Validate all datasets extracted by extract_data().

    Performs the following checks on each dataset:
        - Not empty
        - Missing value counts per column
        - Duplicate row counts
        - Shape and column names
        - Date column conversion to datetime
        - Non-negative checks for 'sales' and 'transactions' datasets

    Args:
        dataframes: Dictionary mapping dataset names to DataFrames,
            as returned by extract_data().

    Returns:
        The validated dictionary of DataFrames, with date columns
        converted where applicable.

    Raises:
        Exception: If an unexpected error occurs during validation.
    """
    logger.info("Starting data validation process.")

    try:
        for name, df in dataframes.items():
            logger.info("Validating dataset: '%s'", name)

            _check_not_empty(name, df)
            _log_shape_and_columns(name, df)
            _count_missing_values(name, df)
            _count_duplicate_rows(name, df)

            df = _convert_date_columns(name, df)
            dataframes[name] = df

            if name == "sales" and "sales" in df.columns:
                _check_non_negative(name, df, "sales")

            if name == "transactions" and "transactions" in df.columns:
                _check_non_negative(name, df, "transactions")

        logger.info("Data validation completed successfully.")
        return dataframes

    except Exception:
        logger.exception("Validation failed due to an unexpected error.")
        raise


if __name__ == "__main__":
    from extract import extract_data

    try:
        raw_dataframes = extract_data()
        validated_dataframes = validate_data(raw_dataframes)
        for dataset_name, dataframe in validated_dataframes.items():
            logger.info(
                "Validated dataset '%s' final shape: %s",
                dataset_name,
                dataframe.shape,
            )
    except Exception:
        logger.exception("Validation process terminated due to an error.")
