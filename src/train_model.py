import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATABASE_PATH, FEATURE_COLUMNS, MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE_RATIO
from src.load import create_tables, get_connection

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - only used if xgboost is not installed.
    XGBClassifier = None


def load_training_data(db_path: Path = DATABASE_PATH) -> pd.DataFrame:
    """Read labeled feature rows from SQLite."""
    query = "SELECT * FROM stock_features WHERE tomorrow_up IS NOT NULL ORDER BY date, ticker"
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn)


def _date_based_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by date so test rows happen after training rows."""
    unique_dates = sorted(data["date"].unique())
    split_index = max(1, int(len(unique_dates) * (1 - TEST_SIZE_RATIO)))
    split_index = min(split_index, len(unique_dates) - 1)
    split_date = unique_dates[split_index]

    train_data = data[data["date"] < split_date].copy()
    test_data = data[data["date"] >= split_date].copy()
    return train_data, test_data


def _evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float | None]:
    """Calculate common classification metrics."""
    predictions = model.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": None,
    }

    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(x_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, probabilities)

    return metrics


def _save_metrics(
    model_name: str,
    metrics: dict[str, float | None],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: dict,
    db_path: Path,
) -> None:
    """Store every training run so model performance can be compared over time."""
    sql = """
        INSERT INTO model_metrics (
            model_name,
            training_date,
            test_start_date,
            test_end_date,
            accuracy,
            precision,
            recall,
            f1_score,
            roc_auc,
            feature_list,
            parameters
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    values = (
        model_name,
        datetime.now(timezone.utc).isoformat(),
        test_data["date"].min(),
        test_data["date"].max(),
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1_score"],
        metrics["roc_auc"],
        json.dumps(FEATURE_COLUMNS),
        json.dumps(parameters, default=str),
    )

    with get_connection(db_path) as conn:
        conn.execute(sql, values)
        conn.commit()


def train_and_save_best_model(
    db_path: Path = DATABASE_PATH,
    model_path: Path = MODEL_PATH,
    logger: logging.Logger | None = None,
) -> dict:
    """Train several model types, compare them, then save the best model."""
    create_tables(db_path=db_path, logger=logger)
    data = load_training_data(db_path)

    if data.empty:
        raise ValueError("No labeled feature data found. Run extraction and loading first.")

    data = data.replace([np.inf, -np.inf], np.nan)
    model_data = data.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    model_data[TARGET_COLUMN] = model_data[TARGET_COLUMN].astype(int)

    if len(model_data) < 100:
        raise ValueError("Not enough clean labeled rows to train a useful model.")

    train_data, test_data = _date_based_split(model_data)
    if train_data.empty or test_data.empty:
        raise ValueError("Date split produced empty train or test data.")

    x_train = train_data[FEATURE_COLUMNS]
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data[FEATURE_COLUMNS]
    y_test = test_data[TARGET_COLUMN]

    candidates = {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": GridSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [3, 5, None],
                "min_samples_leaf": [1, 3, 5],
            },
            cv=TimeSeriesSplit(n_splits=3),
            scoring="f1",
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        ),
    }

    if XGBClassifier is not None:
        candidates["XGBoost"] = GridSearchCV(
            estimator=XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                tree_method="hist",
                n_jobs=-1,
            ),
            param_grid={
                "n_estimators": [100, 200],
                "max_depth": [2, 3, 4],
                "learning_rate": [0.03, 0.1],
                "subsample": [0.8, 1.0],
            },
            cv=TimeSeriesSplit(n_splits=3),
            scoring="f1",
            n_jobs=1,
        )
    elif logger:
        logger.warning("XGBoost is not installed, so the XGBoost candidate was skipped")

    results: list[dict] = []

    for model_name, model in candidates.items():
        if logger:
            logger.info("Training %s", model_name)

        model.fit(x_train, y_train)
        fitted_model = model.best_estimator_ if isinstance(model, GridSearchCV) else model
        metrics = _evaluate_model(fitted_model, x_test, y_test)
        parameters = model.best_params_ if isinstance(model, GridSearchCV) else fitted_model.get_params()

        _save_metrics(model_name, metrics, train_data, test_data, parameters, db_path)

        results.append(
            {
                "model_name": model_name,
                "model": fitted_model,
                "metrics": metrics,
                "parameters": parameters,
            }
        )

        if logger:
            logger.info("%s metrics: %s", model_name, metrics)

    best_result = max(results, key=lambda item: item["metrics"]["f1_score"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_package = {
        "model": best_result["model"],
        "model_name": best_result["model_name"],
        "features": FEATURE_COLUMNS,
        "metrics": best_result["metrics"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    joblib.dump(model_package, model_path)

    if logger:
        logger.info("Saved best model: %s to %s", best_result["model_name"], model_path)

    return {
        "model_name": best_result["model_name"],
        "metrics": best_result["metrics"],
        "test_start_date": test_data["date"].min(),
        "test_end_date": test_data["date"].max(),
    }
