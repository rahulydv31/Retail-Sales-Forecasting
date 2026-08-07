"""
Prediction module for Retail Sales Forecasting.

This module loads the trained model and generates
sales predictions.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

DATA_PATH = Path("data/featured/featured_sales.csv")
MODEL_PATH = Path("models/saved_models/best_model.pkl")
OUTPUT_PATH = Path("data/processed/predictions.csv")

def load_dataset() -> pd.DataFrame:
    """Load engineered dataset."""

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully.")

    return df

def load_model():
    """Load trained model."""

    logger.info("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")

    return model

def preprocess_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare dataset for prediction."""

    logger.info("Preprocessing dataset...")

    df = df.copy()

    # Keep IDs and actual sales for later
    prediction_data = df[
        ["id", "date", "sales"]
    ].copy()

    # Fill missing numeric values
    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    df[numeric_columns] = df[
        numeric_columns
    ].fillna(0)

    # Convert categorical columns
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
    )

    # Remove target column
    X = df.drop(
        columns=["sales"],
    )

    logger.info("Dataset preprocessed successfully.")

    return X, prediction_data

def make_predictions(
    model,
    X: pd.DataFrame,
    prediction_data: pd.DataFrame,
) -> pd.DataFrame:
    """Generate predictions."""

    logger.info("Generating predictions...")

    prediction_data["predicted_sales"] = (
        model.predict(X)
    )

    logger.info("Predictions generated successfully.")

    return prediction_data

def save_predictions(
    prediction_data: pd.DataFrame,
) -> None:
    """Save predictions."""

    logger.info("Saving predictions...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Predictions saved to %s",
        OUTPUT_PATH,
    )

def main():

    data = load_dataset()

    logger.info("Sampling dataset...")

    data = data.sample(
        n=100000,
        random_state=42,
    )

    model = load_model()

    X, prediction_data = preprocess_data(
        data,
    )

    prediction_data = make_predictions(
        model,
        X,
        prediction_data,
    )

    save_predictions(
        prediction_data,
    )

    print(
        prediction_data.head()
    )

    logger.info(
        "Prediction pipeline completed successfully."
    )


if __name__ == "__main__":
    main()