"""Compare classification models for grocery sales performance categories."""

from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "grocery" / "grocery_sales.csv"
OUTPUT_DIR = ROOT / "outputs" / "grocery"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Item_ID", "Category", "Sub_Category", "Price",
    "Quantity_Sold", "Day_of_Week", "Store_ID"
]
NUMERIC_FEATURES = ["Price", "Quantity_Sold", "Day_of_Week"]
CATEGORICAL_FEATURES = [
    "Item_ID", "Category", "Sub_Category", "Store_ID"
]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Add your authorized CSV as data/grocery/grocery_sales.csv."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    if "Sales_Category" in df.columns:
        target = df["Sales_Category"].astype(str).str.strip().str.title()
    elif "Total_Sales" in df.columns:
        # The source report does not state its exact class cut-offs.
        # Quantile grouping is a transparent fallback, not a claim about
        # the original report's label-generation rule.
        total_sales = pd.to_numeric(df["Total_Sales"], errors="coerce")
        labels = ["Low", "Medium", "High"]
        target = pd.qcut(
            total_sales, q=3, labels=labels, duplicates="drop"
        ).astype("object")
    else:
        raise ValueError(
            "CSV must include Sales_Category or Total_Sales."
        )

    valid = target.notna()
    df = df.loc[valid].copy()
    target = target.loc[valid]

    X = df[FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, target


def main() -> None:
    X, y = load_data(DATA_PATH)

    # Stratification preserves class proportions when possible.
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=stratify
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

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "SVM": SVC(kernel="rbf", random_state=42),
    }

    summary = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        safe_name = name.lower().replace(" ", "_")
        report = classification_report(
            y_test, predictions, zero_division=0
        )
        (OUTPUT_DIR / f"{safe_name}_classification_report.txt").write_text(
            report, encoding="utf-8"
        )

        labels = sorted(pd.Series(y_test).unique())
        matrix = confusion_matrix(y_test, predictions, labels=labels)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels
        )
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png")
        plt.close()

        summary.append({"model": name, "accuracy": round(accuracy, 4)})

    results = pd.DataFrame(summary).sort_values(
        "accuracy", ascending=False
    )
    results.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    print(results.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
