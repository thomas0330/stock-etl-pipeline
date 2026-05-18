# Daily Market Data ETL and Prediction Dashboard

An end-to-end Python portfolio project that collects daily stock data, transforms it into analytical features, stores it in SQLite, retrains machine learning models, generates daily predictions, and displays results in a Streamlit dashboard.

This project is designed to demonstrate ETL pipeline design, feature engineering, model training, evaluation tracking, automation, and dashboarding. It is not financial advice and does not guarantee stock market prediction accuracy.

## Architecture

```text
yfinance
   |
   v
src/extract.py
   |
   v
src/transform.py
   |
   v
SQLite database: data/stock_data.db
   |
   +--> src/train_model.py --> models/best_model.joblib
   |
   +--> src/predict.py --> model_predictions table
   |
   v
dashboard/app.py
```

## ETL Flow

1. Extract daily stock prices for AAPL, MSFT, NVDA, TSLA, and GOOGL.
2. Clean and transform the raw prices.
3. Create analytical features such as returns, moving averages, volatility, and price-vs-average ratios.
4. Save prices and features into SQLite with upsert logic to avoid duplicate rows.
5. Retrain machine learning models after new data is loaded.
6. Save model metrics and the best model.
7. Generate the latest prediction for each ticker.
8. Visualize results in Streamlit.

## Data Source

Data comes from Yahoo Finance through the `yfinance` Python package. The default download window is the last 3 years of daily prices.

## Feature Engineering

The pipeline creates:

- Daily return
- Price change
- Price change percentage
- Volume change percentage
- 7-day moving average
- 30-day moving average
- 7-day volatility
- 30-day volatility
- Close price versus 7-day moving average
- Close price versus 30-day moving average

The target column is `tomorrow_up`, where:

- `1` means tomorrow's close is greater than today's close.
- `0` means tomorrow's close is less than or equal to today's close.

The target is calculated within each ticker only. Feature columns use current and historical data, not future data.

## Model Training Approach

The training script uses a date-based split:

- Older rows are used for training.
- Newer rows are used for testing.

It trains and compares:

- Logistic Regression
- Random Forest with simple `GridSearchCV` tuning
- Extra Trees
- Gradient Boosting
- XGBoost with simple `GridSearchCV` tuning

Metrics saved to SQLite include:

- Accuracy
- Precision
- Recall
- F1 score
- ROC AUC when possible

Every training run is saved in the `model_metrics` table. This lets you compare performance across runs over time. The model is retrained daily after new stock data is loaded, but the project does not assume the model will improve every day.

## Adding New Tickers

The project stores active ticker symbols in the `stock_tickers` table. The default ticker list is seeded automatically, and new tickers can be added from the Streamlit dashboard sidebar.

When you add a ticker from the dashboard, the app:

1. Validates the ticker symbol.
2. Downloads the latest 3 years of daily data with `yfinance`.
3. Stores the ticker in SQLite.
4. Loads price and feature rows into the database.
5. Optionally retrains the model candidates.
6. Generates refreshed predictions.

Future daily GitHub Actions runs include active tickers from the database.

## Daily Automation

GitHub Actions runs the pipeline daily using:

```yaml
cron: "0 1 * * *"
```

GitHub Actions cron schedules use UTC. This project runs at 1:00 AM UTC, which is 9:00 AM Taiwan time.

The workflow also supports manual runs with `workflow_dispatch`.

After the pipeline finishes, the workflow commits updated versions of:

- `data/stock_data.db`
- `models/best_model.joblib`

## Dashboard Screenshots

Add screenshots here after running the dashboard locally.

```text
docs/dashboard-home.png
docs/dashboard-metrics.png
```

## Installation

```bash
git clone <your-repository-url>
cd stock-etl-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the virtual environment with:

```bash
source .venv/bin/activate
```

## Run The Pipeline

```bash
python main.py
```

This creates or updates:

- `data/stock_data.db`
- `models/best_model.joblib`
- `logs/pipeline.log`

## Run The Dashboard

```bash
streamlit run dashboard/app.py
```

Open the local Streamlit URL shown in the terminal.

Use the sidebar form to add another ticker such as `AMD`, `META`, or `JPM`.

## Limitations

- Stock direction prediction is a difficult problem and this project is for learning, not trading.
- The feature set is intentionally simple.
- Yahoo Finance data availability can vary by ticker and network conditions.
- The model does not include transaction costs, risk management, or portfolio construction.
- Market regimes can change, so past performance may not generalize.

## Future Improvements

- Add tests for each ETL step.
- Add more tickers and sector metadata.
- Add technical indicators such as RSI or MACD.
- Add walk-forward validation.
- Add model explainability charts.
- Add a ticker removal/deactivation workflow.
- Add Docker support.
- Store dashboard screenshots in a `docs/` folder.
