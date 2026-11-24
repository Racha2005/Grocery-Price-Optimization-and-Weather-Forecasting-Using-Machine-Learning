# Machine Learning Projects: Grocery Price Optimization & Weather Forecasting

This repository contains two academic machine-learning projects implemented in Python.

## Projects

### 1. Grocery Price Optimization Using Machine Learning
Classifies grocery sales performance into **Low**, **Medium**, and **High** categories using product and sales attributes. The project report compares Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine (SVM).

**Input features described in the report:** `Item_ID`, `Category`, `Sub_Category`, `Price`, `Quantity_Sold`, `Day_of_Week`, and `Store_ID`.

**Target:** `Sales_Category` (Low / Medium / High), derived from `Total_Sales`.

The report records these accuracy results:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 0.85 |
| Decision Tree | 0.97 |
| Random Forest | 0.96 |
| SVM | 0.91 |

The report's conclusion identifies **Decision Tree (97%)** as the highest-accuracy model. A separate model-selection section describes Random Forest (96%) as more consistent. Both statements are retained here as reported; the preferred model depends on the evaluation objective.

### 2. Weather Forecasting Using Machine Learning
Predicts temperature using weather observations from multiple cities. A Random Forest Regressor is trained in a preprocessing pipeline with numerical scaling and categorical one-hot encoding. Hyperparameter tuning uses `RandomizedSearchCV`.

**Input features:** Humidity, Wind Speed, Precipitation, Pressure, Visibility, UV Index, City, and Weather Condition.

**Target:** Temperature.

The report records **MAE = 1.07**, **RMSE = 1.69**, and **R² = 0.92**. These are values reported in the academic report; they are not guaranteed to reproduce unless the original dataset and environment are used.

## Repository structure

```text
.
├── grocery_price_optimization/
│   └── train_grocery_models.py
├── weather_forecasting/
│   └── train_weather_model.py
├── data/
│   ├── grocery/
│   │   └── README.md
│   └── weather/
│       └── README.md
├── requirements.txt
└── README.md
```

## Setup

Python 3.10 or later is recommended.

```bash
python -m venv .venv
```

Activate the environment:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the projects

### Grocery Price Optimization

Place your grocery CSV file at `data/grocery/grocery_sales.csv`, then run:

```bash
python grocery_price_optimization/train_grocery_models.py
```

The CSV must contain the input columns listed above and either a `Sales_Category` column or a `Total_Sales` column. If `Sales_Category` is absent, the script creates Low / Medium / High labels using three quantile groups of `Total_Sales`. This is an explicit implementation assumption because the original report does not specify the exact thresholds used to create its labels.

Outputs are written to `outputs/grocery/`, including model comparison, classification reports, and confusion-matrix images.

### Weather Forecasting

Place the Excel dataset at `data/weather/weather_data.xlsx`, then run:

```bash
python weather_forecasting/train_weather_model.py
```

The workbook must contain these columns (matching the project report):

- `Humidity (%)`
- `Wind Speed (km/h or mph)`
- `Precipitation (%)`
- `Pressure (hPa)`
- `Visibility (km)`
- `UV Index`
- `City`
- `Weather Condition`
- A temperature column whose name contains `Temperature`

Outputs are written to `outputs/weather/`, including evaluation metrics, actual-versus-predicted and residual plots, and the saved model.

## Notes on source material and reproducibility

- The scripts are organized into runnable, reproducible versions based on the methods and feature descriptions in the two academic reports.
- The original datasets were not included with the supplied PDFs. Add the permitted dataset files locally; do not upload data you do not have permission to share.
- The grocery report describes the dataset and models but does not provide exact Low / Medium / High threshold values. The script therefore uses quantile-based labels only when a pre-labelled `Sales_Category` column is not supplied.
- The weather report contains 30 cleaned records and reports the metrics above. The original workbook was not attached, so those metrics cannot be independently regenerated from this repository alone.
