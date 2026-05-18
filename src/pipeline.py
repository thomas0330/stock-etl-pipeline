import logging

from src.config import DATABASE_PATH, MODEL_PATH
from src.extract import extract_stock_data
from src.load import load_pipeline_data
from src.predict import generate_latest_predictions
from src.tickers import get_active_tickers
from src.train_model import train_and_save_best_model
from src.transform import transform_stock_data
from src.utils import project_path


def run_full_pipeline(logger: logging.Logger) -> dict:
    """Run extraction, transformation, loading, training, and prediction."""
    logger.info("Loading active ticker list")
    tickers = get_active_tickers(logger=logger)
    logger.info("Active tickers: %s", ", ".join(tickers))

    logger.info("Step 1/5: Extracting stock data")
    prices = extract_stock_data(tickers=tickers, logger=logger)
    if prices.empty:
        raise RuntimeError("Pipeline stopped because no stock prices were downloaded.")

    logger.info("Step 2/5: Transforming stock data into features")
    features = transform_stock_data(prices, logger=logger)

    logger.info("Step 3/5: Loading prices and features into SQLite")
    load_pipeline_data(prices, features, logger=logger)

    logger.info("Step 4/5: Training and saving the best model")
    training_result = train_and_save_best_model(logger=logger)

    logger.info("Step 5/5: Generating latest predictions")
    predictions = generate_latest_predictions(logger=logger)

    return {
        "database": project_path(DATABASE_PATH),
        "model": project_path(MODEL_PATH),
        "best_model": training_result["model_name"],
        "latest_predictions": len(predictions),
        "tickers": tickers,
    }
