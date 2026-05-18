CREATE TABLE IF NOT EXISTS stock_prices (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS stock_tickers (
    ticker TEXT PRIMARY KEY,
    is_active INTEGER NOT NULL DEFAULT 1,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_features (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    adjusted_close REAL,
    volume INTEGER,
    daily_return REAL,
    price_change REAL,
    price_change_pct REAL,
    volume_change_pct REAL,
    moving_average_7d REAL,
    moving_average_30d REAL,
    volatility_7d REAL,
    volatility_30d REAL,
    close_vs_ma7 REAL,
    close_vs_ma30 REAL,
    tomorrow_up INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (ticker, date)
);

CREATE TABLE IF NOT EXISTS model_predictions (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    predicted_tomorrow_up INTEGER NOT NULL,
    prediction_probability REAL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (ticker, date, model_name)
);

CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    training_date TEXT NOT NULL,
    test_start_date TEXT,
    test_end_date TEXT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    roc_auc REAL,
    feature_list TEXT,
    parameters TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
