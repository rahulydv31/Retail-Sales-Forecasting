"""
Retail Sales Forecasting Dashboard.
"""
from PIL import Image
import pandas as pd
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="Retail Sales Forecasting",
    page_icon="📈",
    layout="wide",
)
DATA_PATH = Path("data/featured/featured_sales.csv")
PREDICTION_PATH = Path("data/processed/predictions.csv")
ACTUAL_PREDICTED_PATH = Path(
    "models/evaluation/actual_vs_predicted.png"
)

RESIDUAL_PLOT_PATH = Path(
    "models/evaluation/residual_plot.png"
)

@st.cache_data
def load_data():
    featured_data = pd.read_csv(DATA_PATH)
    predictions_data = pd.read_csv(PREDICTION_PATH)
    return featured_data, predictions_data


featured, predictions = load_data()
# ---------------- Sidebar ---------------- #

st.sidebar.title("Retail Sales Forecasting")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Dataset",
        "Predictions",
        "Model Performance",
        "About",
    ],
)

# ---------------- Home ---------------- #

if page == "Home":

    st.title("📈 Retail Sales Forecasting Dashboard")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
    "Dataset",
    f"{len(featured):,} Rows",
)

    col2.metric(
    "Features",
    featured.shape[1],
)

    col3.metric(
    "Best Model",
    "Random Forest",
)

    st.markdown("---")

    st.subheader("Project Overview")

    st.write(
        """
This dashboard predicts retail sales using Machine Learning.

The project includes:

- Data Extraction
- Data Validation
- Feature Engineering
- Model Training
- Prediction Pipeline
- Model Evaluation
"""
    )

    st.markdown("---")

    st.subheader("Sales Trend")

    sales_trend = (
        featured.groupby("date")["sales"]
        .sum()
        .reset_index()
    )

    st.line_chart(
        sales_trend.set_index("date")
    )

    st.markdown("---")

    st.subheader("Top Product Families by Sales")

    family_sales = (
        featured.groupby("family")["sales"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(family_sales.head(10))

    st.markdown("---")

    st.subheader("Top Stores by Number of Records")

    store_counts = (
    featured["store_nbr"]
    .value_counts()
    .head(10)
)

    st.bar_chart(store_counts)

    st.markdown("---")

    st.caption(
    "Retail Sales Forecasting Dashboard | Machine Learning Project | Developed using Python & Streamlit"
)

# ---------------- Dataset ---------------- #

# ---------------- Dataset ----------------

elif page == "Dataset":

    st.title("Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Rows",
        f"{len(featured):,}",
    )

    col2.metric(
        "Columns",
        featured.shape[1],
    )

    col3.metric(
        "Stores",
        featured["store_nbr"].nunique(),
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    rows = st.slider(
        "Select number of rows",
        min_value=5,
        max_value=100,
        value=10,
    )

    st.subheader("Filter by Store")

    store = st.selectbox(
        "Select Store",
        sorted(featured["store_nbr"].unique()),
    )

    filtered = featured[
        featured["store_nbr"] == store
    ]

    search = st.text_input(
        "Search Product Family"
    )

    if search:
        filtered = filtered[
            filtered["family"].str.contains(
                search,
                case=False,
                na=False,
            )
        ]

    st.dataframe(
        filtered.head(rows),
        use_container_width=True,
    )

    st.write("Rows:", len(filtered))
    st.write("Columns:", filtered.shape[1])

# ---------------- Predictions ---------------- #

elif page == "Predictions":

    st.title("Predictions")
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predictions",
        f"{len(predictions):,}",
    )

    col2.metric(
        "Average Prediction",
        f"{predictions['predicted_sales'].mean():.2f}",
    )

    col3.metric(
        "Maximum Prediction",
        f"{predictions['predicted_sales'].max():.2f}",
    )

    st.markdown("---")

    st.dataframe(
    predictions.head(),
    use_container_width=True,
)
    st.download_button(
    label="📥 Download Predictions CSV",
    data=predictions.to_csv(index=False),
    file_name="predictions.csv",
    mime="text/csv",
)

    st.write("Total Predictions:", len(predictions))

# ---------------- Model ---------------- #

elif page == "Model Performance":

    st.title("Model Performance")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "R² Score",
        "0.9526",
    )

    col2.metric(
        "RMSE",
        "241.16",
    )

    col3.metric(
        "MAE",
        "52.86",
    )

    st.markdown("---")

    st.subheader("Actual vs Predicted")

    actual_vs_predicted = Image.open(
        ACTUAL_PREDICTED_PATH
    )

    st.image(
        actual_vs_predicted,
        use_container_width=True,
    )

    st.markdown("---")

    st.subheader("Residual Plot")

    residual_plot = Image.open(
        RESIDUAL_PLOT_PATH
    )

    st.image(
        residual_plot,
        use_container_width=True,
    )
# ---------------- About ---------------- #

elif page == "About":

    st.title("About")

st.markdown("""
## Retail Sales Forecasting Project

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Matplotlib
- Joblib

### Machine Learning Workflow

✔ Data Extraction

✔ Data Validation

✔ Feature Engineering

✔ Model Training

✔ Model Comparison

✔ Sales Prediction

✔ Model Evaluation

✔ Interactive Dashboard

---

Developed as an end-to-end Machine Learning project for Retail Sales Forecasting.
""")