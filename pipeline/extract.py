"""Extraction module for Retail Sales Forecasting pipeline.

This module reads all raw source datasets required for the forecasting
pipeline and returns them as pandas DataFrames.
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from pandas.errors import EmptyDataError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DATA_DIR = Path("data/raw")

FILE_PATHS = {
    "sales": RAW_DATA_DIR / "train.csv",
    "stores": RAW_DATA_DIR / "stores.csv",
    "transactions": RAW_DATA_DIR / "transactions.csv",
    "oil": RAW_DATA_DIR / "oil.csv",
    "holidays": RAW_DATA_DIR / "holidays_events.csv",
}


def _read_csv_safely(file_path: Path) -> pd.DataFrame:
    """Read a single CSV file into a DataFrame with error handling.

    Args:
        file_path: Path to the CSV file to read.

    Returns:
        A pandas DataFrame containing the file's data.

    Raises:
        FileNotFoundError: If the file does not exist.
        EmptyDataError: If the file exists but contains no data.
        Exception: For any other unexpected error during reading.
    """
    try:
        logger.info("Reading file: %s", file_path)
        df = pd.read_csv(file_path)
        logger.info("Successfully read %s (%d rows)", file_path, len(df))
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except EmptyDataError:
        logger.error("File is empty: %s", file_path)
        raise
    except Exception:
        logger.exception("Unexpected error reading file: %s", file_path)
        raise


def extract_data() -> Dict[str, pd.DataFrame]:
    """Extract all raw source datasets into pandas DataFrames.

    Reads the following files from data/raw/:
        - train.csv
        - stores.csv
        - transactions.csv
        - oil.csv
        - holidays_events.csv

    Returns:
        A dictionary mapping dataset names to their corresponding
        pandas DataFrames:
        {
            "sales": train_df,
            "stores": stores_df,
            "transactions": transactions_df,
            "oil": oil_df,
            "holidays": holidays_df
        }

    Raises:
        FileNotFoundError: If any required file is missing.
        EmptyDataError: If any required file is empty.
        Exception: For any other unexpected error during extraction.
    """
    logger.info("Starting data extraction process.")
    dataframes: Dict[str, pd.DataFrame] = {}

    try:
        for name, path in FILE_PATHS.items():
            dataframes[name] = _read_csv_safely(path)

        logger.info("Data extraction completed successfully.")
        return dataframes

    except FileNotFoundError as exc:
        logger.error("Extraction failed due to missing file: %s", exc)
        raise
    except EmptyDataError as exc:
        logger.error("Extraction failed due to empty file: %s", exc)
        raise
    except Exception:
        logger.exception("Extraction failed due to an unexpected error.")
        raise


if __name__ == "__main__":
    try:
        data = extract_data()
        for dataset_name, dataframe in data.items():
            logger.info(
                "Dataset '%s' shape: %s", dataset_name, dataframe.shape
            )
    except Exception:
        logger.exception("Extraction process terminated due to an error.")