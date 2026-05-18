import logging
from typing import Iterable

import pandas as pd
import yfinance as yf

from src.config import DEFAULT_TICKERS, DOWNLOAD_INTERVAL, DOWNLOAD_PERIOD


PRICE_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]


def extract_stock_data(
    tickers: Iterable[str] | None = None,
    period: str = DOWNLOAD_PERIOD,
    interval: str = DOWNLOAD_INTERVAL,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Download daily stock prices from Yahoo Finance.

    Each ticker is downloaded separately. This makes error handling beginner
    friendly: one failed ticker is logged and skipped, but the rest continue.
    """
    tickers = list(tickers or DEFAULT_TICKERS)
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        try:
            if logger:
                logger.info("Downloading %s price data", ticker)

            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw.empty:
                if logger:
                    logger.warning("No data returned for %s", ticker)
                continue

            # yfinance can return slightly different column shapes depending on
            # version and arguments, so flattening keeps this code predictable.
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            frame = raw.reset_index()
            frame["ticker"] = ticker
            frame = frame.rename(
                columns={
                    "Date": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adjusted_close",
                    "Volume": "volume",
                }
            )

            if "adjusted_close" not in frame.columns:
                frame["adjusted_close"] = frame["close"]

            frame["date"] = pd.to_datetime(frame["date"]).dt.date.astype(str)
            all_frames.append(frame[PRICE_COLUMNS])

        except Exception as exc:  # noqa: BLE001 - keep pipeline alive per ticker.
            if logger:
                logger.exception("Failed to download %s: %s", ticker, exc)

    if not all_frames:
        if logger:
            logger.error("No stock data was downloaded")
        return pd.DataFrame(columns=PRICE_COLUMNS)

    data = pd.concat(all_frames, ignore_index=True)
    return data.sort_values(["ticker", "date"]).reset_index(drop=True)
