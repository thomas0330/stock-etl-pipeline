import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import DATABASE_PATH, MODEL_PATH
from src.load import create_tables, get_connection


def _load_latest_features(db_path: Path) -> pd.DataFrame:
    """Load the most recent feature row for each ticker."""
    query = """
        SELECT sf.*
        FROM stock_features sf
        INNER JOIN (
            SELECT ticker, MAX(date) AS max_date
            FROM stock_features
            GROUP BY ticker
        ) latest
            ON sf.ticker = latest.ticker
           AND sf.date = latest.max_date
        ORDER BY sf.ticker
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def generate_latest_predictions(
    db_path: Path = DATABASE_PATH,
    model_path: Path = MODEL_PATH,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Predict tomorrow's direction for the latest row of each ticker."""
    create_tables(db_path=db_path, logger=logger)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    package = joblib.load(model_path)
    model = package["model"]
    feature_columns = package["features"]
    model_name = package["model_name"]

    latest_features = _load_latest_features(db_path)
    if latest_features.empty:
        raise ValueError("No feature rows found for prediction.")

    prediction_input = latest_features.dropna(subset=feature_columns).copy()
    if prediction_input.empty:
        raise ValueError("Latest feature rows contain missing values needed by the model.")

    predicted_labels = model.predict(prediction_input[feature_columns])
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(prediction_input[feature_columns])[:, 1]
    else:
        probabilities = [None] * len(prediction_input)

    created_at = datetime.now(timezone.utc).isoformat()
    predictions = pd.DataFrame(
        {
            "date": prediction_input["date"],
            "ticker": prediction_input["ticker"],
            "predicted_tomorrow_up": predicted_labels.astype(int),
            "prediction_probability": probabilities,
            "model_name": model_name,
            "created_at": created_at,
        }
    )

    sql = """
        INSERT INTO model_predictions (
            date,
            ticker,
            predicted_tomorrow_up,
            prediction_probability,
            model_name,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(ticker, date, model_name) DO UPDATE SET
            predicted_tomorrow_up = excluded.predicted_tomorrow_up,
            prediction_probability = excluded.prediction_probability,
            created_at = excluded.created_at;
    """

    with get_connection(db_path) as conn:
        conn.executemany(sql, [tuple(row) for row in predictions.to_numpy()])
        conn.commit()

    if logger:
        logger.info("Saved %s latest predictions", len(predictions))

    return predictions
