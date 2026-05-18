# AGENTS.md

## Project Goal

Build and maintain a beginner-friendly portfolio project that demonstrates a daily stock market ETL pipeline, feature engineering, SQLite loading, model training, prediction generation, automation, and Streamlit dashboarding.

This project is for learning ETL, ML, and MLOps patterns. Do not present the predictions as financial advice or guaranteed market forecasts.

## Coding Style

- Keep modules small and focused.
- Prefer readable pandas code over clever one-liners.
- Add beginner-friendly comments for ETL, ML, and database concepts.
- Keep configuration in `src/config.py`.
- Keep reusable helpers in `src/utils.py`.
- Avoid hardcoding secrets, API keys, or machine-specific absolute paths.
- Use explicit column lists when writing to SQLite.

## File Structure

- `main.py`: runs the full pipeline.
- `src/extract.py`: downloads daily market data from yfinance.
- `src/transform.py`: cleans data and creates features and labels.
- `src/load.py`: creates SQLite tables and upserts data.
- `src/train_model.py`: trains, evaluates, tunes, and saves the model.
- `src/predict.py`: creates latest ticker predictions.
- `src/tickers.py`: validates, stores, and reads active tickers.
- `src/ticker_service.py`: adds one ticker and refreshes data, models, and predictions.
- `src/pipeline.py`: reusable full-pipeline orchestration.
- `dashboard/app.py`: Streamlit dashboard.
- `sql/create_tables.sql`: SQLite schema.
- `.github/workflows/daily_pipeline.yml`: daily GitHub Actions automation.

## Commands To Run

```bash
pip install -r requirements.txt
python main.py
streamlit run dashboard/app.py
```

## Testing And Validation Checklist

- `python main.py` completes without errors.
- `data/stock_data.db` is created.
- `models/best_model.joblib` is created.
- SQLite tables exist: `stock_prices`, `stock_features`, `model_predictions`, `model_metrics`.
- Latest predictions exist for the default tickers.
- Adding a ticker from the dashboard downloads data and creates predictions.
- `streamlit run dashboard/app.py` starts the dashboard.
- The GitHub Actions YAML remains valid.

## Important Reminders

- Do not hardcode API keys.
- Keep code modular and beginner-friendly.
- Use date-based train/test splits for time series style data.
- Avoid feature leakage from future prices into model features.
- Save model metrics for every training run so performance can be compared over time.
