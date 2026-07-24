"""
Feature Engineering module for Retail Sales Forecasting.

This module creates machine learning features from the raw sales
dataset including calendar-based features and transaction merging.
"""

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

RAW_DATA = Path("data/raw/train.csv")
STORES_DATA = Path("data/raw/stores.csv")
TRANSACTIONS_DATA = Path("data/raw/transactions.csv")
OIL_DATA = Path("data/raw/oil.csv")
HOLIDAYS_DATA = Path("data/raw/holidays_events.csv")

OUTPUT_PATH = Path("data/featured/featured_sales.csv")


def load_sales_data() -> pd.DataFrame:
    """Load sales dataset."""

    logger.info("Loading sales dataset...")

    df = pd.read_csv(RAW_DATA)

    logger.info("Sales dataset loaded successfully.")

    return df


def load_supporting_data():
    """Load supporting datasets."""

    logger.info("Loading supporting datasets...")

    stores = pd.read_csv(STORES_DATA)

    transactions = pd.read_csv(TRANSACTIONS_DATA)

    oil = pd.read_csv(OIL_DATA)

    holidays = pd.read_csv(HOLIDAYS_DATA)

    logger.info("Supporting datasets loaded successfully.")

    return stores, transactions, oil, holidays


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features."""

    logger.info("Creating calendar features...")

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year

    df["month"] = df["date"].dt.month

    df["day"] = df["date"].dt.day

    df["day_of_week"] = df["date"].dt.dayofweek

    df["week_of_year"] = (
        df["date"]
        .dt
        .isocalendar()
        .week
        .astype(int)
    )

    df["quarter"] = df["date"].dt.quarter

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    logger.info("Calendar features created successfully.")

    return df


def merge_transactions(
    sales: pd.DataFrame,
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Merge transaction data."""

    logger.info("Merging transactions...")

    transactions["date"] = pd.to_datetime(
        transactions["date"]
    )

    sales = sales.merge(
        transactions,
        on=["date", "store_nbr"],
        how="left",
    )

    logger.info("Transactions merged successfully.")

    return sales

def merge_oil_prices(
    sales: pd.DataFrame,
    oil: pd.DataFrame,
) -> pd.DataFrame:
    """Merge oil prices."""

    logger.info("Merging oil prices...")

    oil["date"] = pd.to_datetime(
        oil["date"]
    )

    sales = sales.merge(
        oil,
        on="date",
        how="left",
    )

    logger.info("Oil prices merged successfully.")

    return sales

def merge_holidays(
    sales: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Merge holiday information."""

    logger.info("Merging holidays...")

    holidays["date"] = pd.to_datetime(
        holidays["date"]
    )

    sales = sales.merge(
        holidays,
        on="date",
        how="left",
    )

    logger.info("Holidays merged successfully.")

    return sales

def merge_stores(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge store information."""

    logger.info("Merging store information...")

    sales = sales.merge(
        stores,
        on="store_nbr",
        how="left",
    )

    logger.info("Store information merged successfully.")

    return sales

def create_lag_features(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Create lag features."""

    logger.info("Creating lag features...")

    sales = sales.sort_values(
        by=["store_nbr", "family", "date"]
    )

    sales["lag_1"] = (
        sales.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .shift(1)
    )

    sales["lag_7"] = (
        sales.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .shift(7)
    )

    logger.info("Lag features created successfully.")

    return sales

def create_rolling_features(
    sales: pd.DataFrame,
) -> pd.DataFrame:
    """Create rolling statistics."""

    logger.info("Creating rolling features...")

    sales["rolling_mean_7"] = (
        sales.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    sales["rolling_std_7"] = (
        sales.groupby(
            ["store_nbr", "family"]
        )["sales"]
        .transform(
            lambda x: x.shift(1).rolling(
                window=7,
                min_periods=1,
            ).std()
        )
    )

    logger.info("Rolling features created successfully.")

    return sales

def save_dataset(df: pd.DataFrame) -> None:
    """Save engineered dataset."""

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Featured dataset saved to %s",
        OUTPUT_PATH,
    )


def main():

    sales = load_sales_data()

    sales = create_calendar_features(sales)

    stores, transactions, oil, holidays = load_supporting_data()

    stores = pd.read_csv(STORES_DATA)

    stores = stores.rename(
    columns={"type": "store_type"}
)

    sales = merge_transactions(
        sales,
        transactions,
    )

    sales = merge_oil_prices(
        sales,
        oil,
    )

    sales = merge_holidays(
        sales,
        holidays,
)
    
    sales = merge_stores(
    sales,
    stores,
)
    
    sales = create_lag_features(
    sales,
)

    sales = create_rolling_features(
    sales,
)
    
    save_dataset(sales)

    logger.info("Feature engineering completed successfully.")


if __name__ == "__main__":
    main()