"""
Model Training module for Retail Sales Forecasting.

This module trains multiple machine learning models using the
engineered dataset and selects the best one.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/featured/featured_sales.csv")
MODEL_PATH = Path("models/saved_models")


def load_dataset() -> pd.DataFrame:
    """Load engineered dataset."""

    logger.info("Loading engineered dataset...")

    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully.")

    return df


def preprocess_data(
    df: pd.DataFrame,
):
    """Prepare dataset for model training."""

    logger.info("Preprocessing dataset...")

    df = df.copy()

    df = df.dropna(subset=["sales"])

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    categorical_columns = (
        df.select_dtypes(
            include="object"
        ).columns
    )

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
    )

    X = df.drop(columns=["sales"])
    y = df["sales"]

    logger.info("Dataset preprocessed successfully.")

    return X, y


def main():

    data = load_dataset()

    X, y = preprocess_data(data)

    logger.info("Sampling dataset...")

    X = X.sample(
        n=100000,
        random_state=42,
    )

    y = y.loc[X.index]

    logger.info("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    logger.info("Training models...")

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            random_state=42,
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_model = None
    best_r2 = float("-inf")

    for name, model in models.items():

        logger.info(f"Training {name}...")

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test,
        )

        mae = mean_absolute_error(
            y_test,
            predictions,
        )

        rmse = (
            mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5
        )

        r2 = r2_score(
            y_test,
            predictions,
        )

        results.append(
            [name, mae, rmse, r2]
        )

        if r2 > best_r2:
            best_r2 = r2
            best_model = model

    joblib.dump(
        best_model,
        MODEL_PATH / "best_model.pkl",
    )

    logger.info("Best model saved successfully.")

    results = pd.DataFrame(
        results,
        columns=[
            "Model",
            "MAE",
            "RMSE",
            "R2",
        ],
    )

    print("\nModel Comparison:\n")
    print(results)

    print("\nBest Model:\n")
    print(
        results.sort_values(
            by="R2",
            ascending=False,
        ).head(1)
    )


if __name__ == "__main__":
    main()