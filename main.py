from src.pipeline import run_full_pipeline
from src.utils import setup_logging


def main() -> None:
    """Run the full daily ETL, training, and prediction pipeline."""
    logger = setup_logging()
    result = run_full_pipeline(logger)

    print("\nDaily Market Data ETL and Prediction Pipeline completed successfully.")
    print(f"Tickers: {', '.join(result['tickers'])}")
    print(f"Database: {result['database']}")
    print(f"Model: {result['model']}")
    print(f"Best model: {result['best_model']}")
    print(f"Latest predictions saved: {result['latest_predictions']}")


if __name__ == "__main__":
    main()
