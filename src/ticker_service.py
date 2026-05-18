import logging

from src.extract import extract_stock_data
from src.load import load_pipeline_data
from src.predict import generate_latest_predictions
from src.tickers import add_ticker, validate_ticker
from src.train_model import train_and_save_best_model
from src.transform import transform_stock_data


def add_ticker_and_update_predictions(
    ticker: str,
    retrain_model: bool = True,
    logger: logging.Logger | None = None,
) -> dict:
    """Add one ticker, load its history, and refresh predictions.

    This is used by the dashboard. It intentionally reuses the same ETL modules
    as the daily pipeline so the interactive workflow and automated workflow
    stay consistent.
    """
    normalized = validate_ticker(ticker)

    if logger:
        logger.info("Adding ticker %s", normalized)

    prices = extract_stock_data(tickers=[normalized], logger=logger)
    if prices.empty:
        raise ValueError(f"No price data was returned for {normalized}. Check the ticker symbol.")

    features = transform_stock_data(prices, logger=logger)
    add_ticker(normalized, logger=logger)
    load_pipeline_data(prices, features, logger=logger)

    training_result = None
    if retrain_model:
        training_result = train_and_save_best_model(logger=logger)

    predictions = generate_latest_predictions(logger=logger)

    return {
        "ticker": normalized,
        "price_rows": len(prices),
        "feature_rows": len(features),
        "latest_predictions": len(predictions),
        "training_result": training_result,
    }
