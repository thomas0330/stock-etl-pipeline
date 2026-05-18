import logging
import re
from pathlib import Path

from src.config import DATABASE_PATH, DEFAULT_TICKERS
from src.load import create_tables, get_connection


TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=-]{1,12}$")


def normalize_ticker(ticker: str) -> str:
    """Clean user input into the uppercase ticker format yfinance expects."""
    return ticker.strip().upper()


def validate_ticker(ticker: str) -> str:
    """Validate ticker text before it is saved or sent to yfinance."""
    normalized = normalize_ticker(ticker)
    if not normalized:
        raise ValueError("Ticker cannot be empty.")
    if not TICKER_PATTERN.match(normalized):
        raise ValueError(
            "Ticker can only contain letters, numbers, dots, dashes, equals signs, or carets."
        )
    return normalized


def seed_default_tickers(
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> None:
    """Add the starter tickers to the ticker registry if they are missing."""
    create_tables(db_path=db_path, logger=logger)
    with get_connection(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO stock_tickers (ticker, is_active)
            VALUES (?, 1)
            ON CONFLICT(ticker) DO UPDATE SET
                is_active = 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            [(ticker,) for ticker in DEFAULT_TICKERS],
        )
        conn.commit()


def add_ticker(
    ticker: str,
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> str:
    """Save a ticker in the registry so future pipeline runs include it."""
    normalized = validate_ticker(ticker)
    create_tables(db_path=db_path, logger=logger)

    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO stock_tickers (ticker, is_active)
            VALUES (?, 1)
            ON CONFLICT(ticker) DO UPDATE SET
                is_active = 1,
                updated_at = CURRENT_TIMESTAMP
            """,
            (normalized,),
        )
        conn.commit()

    if logger:
        logger.info("Ticker %s is active in the ticker registry", normalized)
    return normalized


def get_active_tickers(
    db_path: Path = DATABASE_PATH,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Return active tickers from SQLite, seeding defaults for a new database."""
    seed_default_tickers(db_path=db_path, logger=logger)

    with get_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ticker
            FROM stock_tickers
            WHERE is_active = 1
            ORDER BY ticker
            """
        ).fetchall()

    return [row[0] for row in rows]
