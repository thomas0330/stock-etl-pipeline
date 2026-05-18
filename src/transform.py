import logging

import numpy as np
import pandas as pd

from src.config import FEATURE_COLUMNS, TARGET_COLUMN


def transform_stock_data(
    prices: pd.DataFrame,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    """Clean price data and create model-ready analytical features."""
    if prices.empty:
        if logger:
            logger.warning("No price rows to transform")
        return prices.copy()

    data = prices.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["ticker"] = data["ticker"].astype(str)

    numeric_columns = ["open", "high", "low", "close", "adjusted_close", "volume"]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.sort_values(["ticker", "date"])
    data = data.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Rows without a close price cannot support returns, features, or labels.
    data = data.dropna(subset=["close"])

    # For OHLC gaps, use nearby values within the same ticker only.
    data[numeric_columns] = data.groupby("ticker", group_keys=False)[numeric_columns].apply(
        lambda group: group.ffill().bfill()
    )
    data["volume"] = data["volume"].fillna(0)

    grouped = data.groupby("ticker", group_keys=False)

    data["daily_return"] = grouped["close"].pct_change()
    data["price_change"] = grouped["close"].diff()
    data["price_change_pct"] = data["daily_return"]
    data["volume_change_pct"] = grouped["volume"].pct_change()

    data["moving_average_7d"] = grouped["close"].transform(
        lambda series: series.rolling(window=7, min_periods=1).mean()
    )
    data["moving_average_30d"] = grouped["close"].transform(
        lambda series: series.rolling(window=30, min_periods=1).mean()
    )
    data["volatility_7d"] = grouped["daily_return"].transform(
        lambda series: series.rolling(window=7, min_periods=2).std()
    )
    data["volatility_30d"] = grouped["daily_return"].transform(
        lambda series: series.rolling(window=30, min_periods=2).std()
    )

    data["close_vs_ma7"] = data["close"] / data["moving_average_7d"] - 1
    data["close_vs_ma30"] = data["close"] / data["moving_average_30d"] - 1

    tomorrow_close = grouped["close"].shift(-1)
    data[TARGET_COLUMN] = np.where(
        tomorrow_close.isna(),
        np.nan,
        (tomorrow_close > data["close"]).astype(int),
    )

    data[FEATURE_COLUMNS] = data[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    data["date"] = data["date"].dt.date.astype(str)

    if logger:
        logger.info("Created %s transformed feature rows", len(data))

    return data.reset_index(drop=True)
