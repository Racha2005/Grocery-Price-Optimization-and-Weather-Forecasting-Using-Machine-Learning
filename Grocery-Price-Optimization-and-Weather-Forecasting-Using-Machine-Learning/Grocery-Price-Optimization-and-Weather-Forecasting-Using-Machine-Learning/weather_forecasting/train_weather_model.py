"""Train and evaluate a Random Forest temperature-regression pipeline."""

from pathlib import Path
import re
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    train_test_split, RandomizedSearchCV, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "weather" / "weather_data.xlsx"
OUTPUT_DIR = ROOT / "outputs" / "weather"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "Humidity (%)",
    "Wind Speed (km/h or mph)",
    "Precipitation (%)",
    "Pressure (hPa)",
    "Visibility (km)",
    "UV Index",
]
CATEGORICAL_FEATURES = ["City", "Weather Condition"]


def extract_temperature(value):
    """Extract the first numeric temperature value from mixed text."""
    match = re.search(r"-?\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else np.nan


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Workbook not found: {DATA_PATH}\n"
            "Add your authorized Excel file as data/weather/weather_data.xlsx."
        )

    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    temp_columns = [c for c in df.columns if "temperature" in c.lower()]
    if not temp_columns:
        raise ValueError("Could not find a column containing 'Temperature'.")
    target_column = temp_columns[0]

    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["temp_numeric"] = df[target_column].apply(extract_temperature)
    df = df.dropna(subset=["temp_numeric"]).copy()

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["temp_numeric"]

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ])

    param_dist = {
        "regressor__n_estimators": [50, 100, 200, 300],
        "regressor__max_depth": [None, 5, 10, 20],
        "regressor__min_samples_split": [2, 5, 10],
        "regressor__min_samples_leaf": [1, 2, 4],
        # 'auto' from older sklearn versions is avoided for compatibility.
        "regressor__max_features": [1.0, "sqrt", 0.5],
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        cv=cv,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
        "best_parameters": search.best_params_,
    }
    (OUTPUT_DIR / "metrics.json").write_text(
        __import__("json").dumps(metrics, indent=2), encoding="utf-8"
    )
    print(metrics)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions)
    low = min(float(y_test.min()), float(predictions.min()))
    high = max(float(y_test.max()), float(predictions.max()))
    plt.plot([low, high], [low, high], linestyle="--")
    plt.xlabel("Actual Temperature")
    plt.ylabel("Predicted Temperature")
    plt.title("Actual vs Predicted Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png")
    plt.close()

    residuals = y_test - predictions
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=20)
    plt.title("Residual Distribution")
    plt.xlabel("Actual - Predicted Temperature")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "residuals_histogram.png")
    plt.close()

    joblib.dump(best_model, OUTPUT_DIR / "weather_rf_best_model.joblib")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
