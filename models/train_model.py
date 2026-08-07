"""
Model Training module for Retail Sales Forecasting.

This module trains the machine learning model using the
engineered dataset.
"""
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import pandas as pd

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

    # Remove rows with missing target
    df = df.dropna(subset=["sales"])

    # Fill missing numeric values
    numeric_columns = df.select_dtypes(include="number").columns
    df[numeric_columns] = df[numeric_columns].fillna(0)

    # Convert categorical columns
    categorical_columns = df.select_dtypes(include="object").columns

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

    logger.info("Training Random Forest model...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    MODEL_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH / "random_forest_model.pkl",
    )

    logger.info("Model saved successfully.")

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
        )
        ** 0.5
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


if __name__ == "__main__":
    main()
