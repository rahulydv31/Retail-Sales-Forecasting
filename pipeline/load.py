"""Load module for Retail Sales Forecasting ETL pipeline.

This module loads validated pandas DataFrames into PostgreSQL tables
using efficient bulk inserts via psycopg2's execute_values. It also
handles dataset-specific column renaming and datetime-to-date
conversion prior to insertion.
"""

import logging
from datetime import date, datetime
from typing import Dict

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database.connection import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TABLE_MAP = {
    "sales": "sales",
    "stores": "stores",
    "transactions": "transactions",
    "oil": "oil_prices",
    "holidays": "holidays",
}

COLUMN_RENAME_MAP: Dict[str, Dict[str, str]] = {
    "stores": {"type": "store_type"},
    "holidays": {"type": "holiday_type"},
}


def _rename_columns(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for a dataset according to COLUMN_RENAME_MAP.

    Args:
        dataset_name: Name of the dataset being processed.
        df: DataFrame whose columns may need renaming.

    Returns:
        DataFrame with columns renamed, if applicable.
    """
    rename_map = COLUMN_RENAME_MAP.get(dataset_name)
    if not rename_map:
        return df

    existing_renames = {
        old: new for old, new in rename_map.items() if old in df.columns
    }

    if existing_renames:
        logger.info(
            "Renaming columns for dataset '%s': %s",
            dataset_name,
            existing_renames,
        )
        df = df.rename(columns=existing_renames)

    return df


def _convert_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert datetime64 columns into native Python date objects.

    Args:
        df: DataFrame to process.

    Returns:
        DataFrame with datetime columns converted to date objects,
        suitable for insertion into PostgreSQL DATE columns.
    """
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].apply(
                lambda x: x.date() if isinstance(x, (pd.Timestamp, datetime))
                else x
            )
    return df


def _prepare_dataframe(dataset_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Apply column renaming and datetime conversion before loading.

    Args:
        dataset_name: Name of the dataset being processed.
        df: DataFrame to prepare.

    Returns:
        Prepared DataFrame ready for insertion.
    """
    df = _rename_columns(dataset_name, df)
    df = _convert_datetime_columns(df)
    return df


def _insert_dataframe(
    conn: psycopg2.extensions.connection,
    df: pd.DataFrame,
    table_name: str,
) -> None:
    """Insert a DataFrame into a PostgreSQL table using bulk insert.

    Args:
        conn: Active psycopg2 database connection.
        df: DataFrame to insert.
        table_name: Name of the target PostgreSQL table.

    Raises:
        Exception: If the insert operation fails.
    """
    if df.empty:
        logger.warning(
            "DataFrame for table '%s' is empty; skipping insert.",
            table_name,
        )
        return

    columns = list(df.columns)
    values = [tuple(row) for row in df.itertuples(index=False, name=None)]

    insert_query = (
        f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES %s"
    )

    try:
        with conn.cursor() as cursor:
            logger.info(
                "Inserting %d rows into table '%s'.", len(values), table_name
            )
            execute_values(cursor, insert_query, values)
        conn.commit()
        logger.info(
            "Successfully committed data into table '%s'.", table_name
        )
    except Exception:
        conn.rollback()
        logger.exception(
            "Failed to insert data into table '%s'. Rolled back.",
            table_name,
        )
        raise


def load_data(dataframes: Dict[str, pd.DataFrame]) -> None:
    """Load validated DataFrames into their corresponding PostgreSQL tables.

    Applies dataset-specific column renaming and datetime conversion
    before inserting into PostgreSQL.
    """
    logger.info("Starting data load process.")
    conn = None

    try:
        conn = get_connection()
        logger.info("Database connection established.")

        # Load parent tables before child tables
        load_order = [
            "stores",
            "oil",
            "holidays",
            "transactions",
            "sales",
        ]

        for dataset_name in load_order:

            if dataset_name not in dataframes:
                logger.warning(
                    "Dataset '%s' not found; skipping.",
                    dataset_name,
                )
                continue

            table_name = TABLE_MAP.get(dataset_name)

            if table_name is None:
                logger.warning(
                    "No table mapping found for dataset '%s'; skipping.",
                    dataset_name,
                )
                continue

            prepared_df = _prepare_dataframe(
                dataset_name,
                dataframes[dataset_name].copy(),
            )

            logger.info(
                "Loading '%s' into table '%s'.",
                dataset_name,
                table_name,
            )

            _insert_dataframe(
                conn,
                prepared_df,
                table_name,
            )

        logger.info("Data load process completed successfully.")

    except Exception:
        logger.exception("Data load process failed due to an error.")
        raise

    finally:
        if conn is not None:
            conn.close()
            logger.info("Database connection closed.")


if __name__ == "__main__":
    from extract import extract_data
    from validator import validate_data

    try:
        raw_dataframes = extract_data()
        validated_dataframes = validate_data(raw_dataframes)
        load_data(validated_dataframes)
    except Exception:
        logger.exception("Load process terminated due to an error.")