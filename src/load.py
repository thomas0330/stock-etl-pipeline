import logging
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.config import CREATE_TABLES_SQL_PATH, DATABASE_PATH


PRICE_TABLE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]

FEATURE_TABLE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
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
    "tomorrow_up",
]


@contextmanager
def get_connection(db_path: Path = DATABASE_PATH) -> Iterator[sqlite3.Connection]:
    """Open a SQLite connection and make sure the data folder exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def create_tables(
    db_path: Path = DATABASE_PATH,
    sql_path: Path = CREATE_TABLES_SQL_PATH,
    logger: logging.Logger | None = None,
) -> None:
    """Create database tables from the SQL schema file."""
    with get_connection(db_path) as conn:
        conn.executescript(sql_path.read_text(encoding="utf-8"))
        conn.commit()

    if logger:
        logger.info("Database tables are ready at %s", db_path)


def _prepare_records(frame: pd.DataFrame, columns: list[str]) -> list[tuple]:
    """Convert DataFrame rows into SQLite-friendly tuples."""
    clean = frame[columns].copy()
    clean = clean.where(pd.notna(clean), None)
    return [tuple(row) for row in clean.to_numpy()]


def upsert_prices(
    prices: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> None:
    """Insert or update daily stock prices using ticker/date as the key."""
    if prices.empty:
        if logger:
            logger.warning("No price rows to load")
        return

    records = _prepare_records(prices, PRICE_TABLE_COLUMNS)
    placeholders = ", ".join(["?"] * len(PRICE_TABLE_COLUMNS))

    sql = f"""
        INSERT INTO stock_prices ({", ".join(PRICE_TABLE_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(ticker, date) DO UPDATE SET
            open = excluded.open,
            high = excluded.high,
            low = excluded.low,
            close = excluded.close,
            adjusted_close = excluded.adjusted_close,
            volume = excluded.volume,
            updated_at = CURRENT_TIMESTAMP;
    """

    with get_connection(db_path) as conn:
        conn.executemany(sql, records)
        conn.commit()

    if logger:
        logger.info("Loaded %s price rows", len(records))


def upsert_features(
    features: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> None:
    """Insert or update feature rows using ticker/date as the key."""
    if features.empty:
        if logger:
            logger.warning("No feature rows to load")
        return

    records = _prepare_records(features, FEATURE_TABLE_COLUMNS)
    placeholders = ", ".join(["?"] * len(FEATURE_TABLE_COLUMNS))

    update_columns = [
        column
        for column in FEATURE_TABLE_COLUMNS
        if column not in {"ticker", "date"}
    ]
    update_clause = ",\n            ".join(
        [f"{column} = excluded.{column}" for column in update_columns]
    )

    sql = f"""
        INSERT INTO stock_features ({", ".join(FEATURE_TABLE_COLUMNS)})
        VALUES ({placeholders})
        ON CONFLICT(ticker, date) DO UPDATE SET
            {update_clause},
            updated_at = CURRENT_TIMESTAMP;
    """

    with get_connection(db_path) as conn:
        conn.executemany(sql, records)
        conn.commit()

    if logger:
        logger.info("Loaded %s feature rows", len(records))


def load_pipeline_data(
    prices: pd.DataFrame,
    features: pd.DataFrame,
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> None:
    """Create tables, then load prices and features."""
    create_tables(db_path=db_path, logger=logger)
    upsert_prices(prices, db_path=db_path, logger=logger)
    upsert_features(features, db_path=db_path, logger=logger)
