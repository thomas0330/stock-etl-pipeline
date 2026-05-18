from pathlib import Path


# Project paths are defined in one place so every module can find the same files.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
SQL_DIR = PROJECT_ROOT / "sql"

DATABASE_PATH = DATA_DIR / "stock_data.db"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
CREATE_TABLES_SQL_PATH = SQL_DIR / "create_tables.sql"

DEFAULT_TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"]
DOWNLOAD_PERIOD = "3y"
DOWNLOAD_INTERVAL = "1d"

RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.2

FEATURE_COLUMNS = [
    "daily_return",
    "price_change",
    "price_change_pct",
    "volume_change_pct",
    "moving_average_7d",
    "moving_average_30d",
    "volatility_7d",
    "volatility_30d",
    "close_vs_ma7",
    "close_vs_ma30",
]

TARGET_COLUMN = "tomorrow_up"
