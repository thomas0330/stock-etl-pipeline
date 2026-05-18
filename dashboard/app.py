import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATABASE_PATH = PROJECT_ROOT / "data" / "stock_data.db"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ticker_service import add_ticker_and_update_predictions
from src.utils import setup_logging


@st.cache_data(ttl=300)
def read_table(query: str, params: tuple = ()) -> pd.DataFrame:
    """Read a SQLite query into pandas with short caching for dashboard speed."""
    with sqlite3.connect(DATABASE_PATH) as conn:
        return pd.read_sql_query(query, conn, params=params)


def format_probability(value: float | None) -> str:
    if pd.isna(value):
        return "Not available"
    return f"{value:.1%}"


st.set_page_config(page_title="Daily Market Data Dashboard", layout="wide")
st.title("Daily Market Data ETL and Prediction Dashboard")

if not DATABASE_PATH.exists():
    st.warning("Database not found. Run `python main.py` from the project root first.")
    st.stop()

with st.sidebar.form("add_ticker_form"):
    st.subheader("Add Ticker")
    new_ticker = st.text_input("Ticker symbol", placeholder="AMD")
    retrain_model = st.checkbox("Retrain model now", value=True)
    add_ticker_submitted = st.form_submit_button("Add ticker and predict")

if add_ticker_submitted:
    try:
        with st.spinner("Updating market data and predictions..."):
            result = add_ticker_and_update_predictions(
                new_ticker,
                retrain_model=retrain_model,
                logger=setup_logging(),
            )
            read_table.clear()

        st.sidebar.success(
            f"Added {result['ticker']} with {result['price_rows']} price rows."
        )
    except Exception as exc:  # noqa: BLE001 - show friendly dashboard errors.
        st.sidebar.error(str(exc))

tickers = read_table("SELECT DISTINCT ticker FROM stock_features ORDER BY ticker")
if tickers.empty:
    st.warning("No feature data found. Run the pipeline first.")
    st.stop()

selected_ticker = st.sidebar.selectbox("Ticker", tickers["ticker"].tolist())

latest_prediction = read_table(
    """
    SELECT *
    FROM model_predictions
    WHERE ticker = ?
    ORDER BY date DESC, created_at DESC
    LIMIT 1
    """,
    (selected_ticker,),
)

prices = read_table(
    """
    SELECT date, ticker, close, moving_average_7d, moving_average_30d, daily_return
    FROM stock_features
    WHERE ticker = ?
    ORDER BY date
    """,
    (selected_ticker,),
)
prices["date"] = pd.to_datetime(prices["date"])

recent_predictions = read_table(
    """
    SELECT date, ticker, predicted_tomorrow_up, prediction_probability, model_name, created_at
    FROM model_predictions
    WHERE ticker = ?
    ORDER BY date DESC, created_at DESC
    LIMIT 20
    """,
    (selected_ticker,),
)

metrics = read_table(
    """
    SELECT model_name, training_date, test_start_date, test_end_date,
           accuracy, precision, recall, f1_score, roc_auc
    FROM model_metrics
    ORDER BY training_date
    """
)

st.subheader(f"Latest Prediction: {selected_ticker}")

if latest_prediction.empty:
    st.info("No prediction has been generated for this ticker yet.")
else:
    row = latest_prediction.iloc[0]
    direction = "Up" if int(row["predicted_tomorrow_up"]) == 1 else "Down or flat"
    probability = format_probability(row["prediction_probability"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Tomorrow Direction", direction)
    col2.metric("Probability of Up Move", probability)
    col3.metric("Model", row["model_name"])

st.subheader("Recent Price Trend")
price_chart_data = prices.tail(180).melt(
    id_vars=["date"],
    value_vars=["close", "moving_average_7d", "moving_average_30d"],
    var_name="Series",
    value_name="Price",
)
fig_price = px.line(
    price_chart_data,
    x="date",
    y="Price",
    color="Series",
    title=f"{selected_ticker} Close Price with 7-Day and 30-Day Moving Averages",
)
st.plotly_chart(fig_price, use_container_width=True)

st.subheader("Daily Return Trend")
fig_returns = px.line(
    prices.tail(180),
    x="date",
    y="daily_return",
    title=f"{selected_ticker} Daily Return",
)
st.plotly_chart(fig_returns, use_container_width=True)

st.subheader("Recent Predictions")
if recent_predictions.empty:
    st.info("No recent predictions to show yet.")
else:
    display_predictions = recent_predictions.copy()
    display_predictions["prediction_probability"] = display_predictions[
        "prediction_probability"
    ].map(format_probability)
    display_predictions["predicted_tomorrow_up"] = display_predictions[
        "predicted_tomorrow_up"
    ].map({1: "Up", 0: "Down or flat"})
    st.dataframe(display_predictions, use_container_width=True)

st.subheader("Model Performance Over Time")
if metrics.empty:
    st.info("No model metrics recorded yet.")
else:
    metrics["training_date"] = pd.to_datetime(metrics["training_date"])
    metric_choice = st.selectbox(
        "Metric",
        ["accuracy", "precision", "recall", "f1_score", "roc_auc"],
    )
    fig_metrics = px.line(
        metrics,
        x="training_date",
        y=metric_choice,
        color="model_name",
        markers=True,
        title=f"{metric_choice} by Training Run",
    )
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.dataframe(metrics.sort_values("training_date", ascending=False), use_container_width=True)

st.subheader("Metric Guide")
st.markdown(
    """
- **Accuracy**: Share of test predictions that were correct.
- **Precision**: When the model predicted "up", how often it was right.
- **Recall**: Of the actual "up" days, how many the model found.
- **F1 score**: Balance between precision and recall.
- **ROC AUC**: How well the model ranks up days above down days when probabilities are available.
"""
)
