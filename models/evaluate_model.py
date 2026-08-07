"""
Model Evaluation module for Retail Sales Forecasting.

This module evaluates the trained model and
creates visualizations.
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

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
MODEL_PATH = Path("models/saved_models/best_model.pkl")
OUTPUT_DIR = Path("models/evaluation")

def load_dataset() -> pd.DataFrame:
    """Load engineered dataset."""

    logger.info("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    logger.info("Dataset loaded successfully.")

    return df


def load_model():
    """Load trained model."""

    logger.info("Loading model...")

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")

    return model

def preprocess_data(
    df: pd.DataFrame,
):
    """Prepare dataset."""

    logger.info("Preprocessing dataset...")

    df = df.copy()

    df = df.dropna(
        subset=["sales"],
    )

    numeric_columns = df.select_dtypes(
        include="number",
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .fillna(0)
    )

    categorical_columns = (
        df.select_dtypes(
            include="object",
        ).columns
    )

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
    )

    X = df.drop(
        columns=["sales"],
    )

    y = df["sales"]

    return X, y

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
):
    """Evaluate trained model."""

    logger.info("Evaluating model...")

    predictions = model.predict(X)

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = (
        mean_squared_error(
            y,
            predictions,
        ) ** 0.5
    )

    r2 = r2_score(
        y,
        predictions,
    )

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    return predictions

def plot_actual_vs_predicted(
    y: pd.Series,
    predictions,
):
    """Create Actual vs Predicted graph."""

    logger.info("Creating Actual vs Predicted graph...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.figure(
        figsize=(8, 6),
    )

    plt.scatter(
        y,
        predictions,
        alpha=0.3,
    )

    plt.xlabel("Actual Sales")

    plt.ylabel("Predicted Sales")

    plt.title("Actual vs Predicted Sales")

    plt.savefig(
        OUTPUT_DIR / "actual_vs_predicted.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Graph saved successfully.")

def plot_residuals(
    y: pd.Series,
    predictions,
):
    """Create residual plot."""

    logger.info("Creating residual plot...")

    residuals = y - predictions

    plt.figure(
        figsize=(8, 6),
    )

    plt.scatter(
        predictions,
        residuals,
        alpha=0.3,
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--",
    )

    plt.xlabel("Predicted Sales")

    plt.ylabel("Residuals")

    plt.title("Residual Plot")

    plt.savefig(
        OUTPUT_DIR / "residual_plot.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Residual plot saved successfully.")

def main():

    data = load_dataset()

    logger.info("Sampling dataset...")

    data = data.sample(
        n=100000,
        random_state=42,
    )

    model = load_model()

    X, y = preprocess_data(
        data,
    )

    predictions = evaluate_model(
        model,
        X,
        y,
    )

    plot_actual_vs_predicted(
        y,
        predictions,
    )

    plot_residuals(
        y,
        predictions,
    )

    logger.info(
        "Model evaluation completed successfully."
    )


if __name__ == "__main__":
    main()