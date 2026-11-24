# Machine Learning Projects: Grocery Price Optimization & Weather Forecasting



## Overview

Grocery Price Optimization and Weather Forecasting are two machine learning projects developed using Python to explore how data and predictive models can help solve real-world problems. The Grocery Price Optimization project focuses on understanding grocery sales patterns and classifying sales performance to support better pricing decisions. The Weather Forecasting project uses historical weather information to predict temperature based on different environmental conditions. Both projects involve working with datasets, preparing data for analysis, applying machine learning algorithms, building predictive models, and evaluating their performance.

---

## Projects

### 1. Grocery Price Optimization Using Machine Learning

#### About the Project
Grocery Price Optimization uses machine learning techniques to study grocery sales data and classify sales performance into three categories: Low, Medium, and High. It considers product details, pricing, quantity sold, store information, and sales-related attributes to understand purchasing patterns and support pricing decisions.

#### Objectives
- Understand grocery sales patterns using historical data.
- Identify factors that influence sales performance.
- Classify sales into Low, Medium, and High categories.
- Compare different machine learning algorithms.
- Identify a suitable model based on classification performance.
- Support better pricing and inventory-related decisions through data insights.

#### Key Features
- Grocery sales data preprocessing and preparation.
- Analysis of product and sales-related attributes.
- Classification of sales performance into three categories.
- Implementation of multiple machine learning algorithms.
- Model comparison using accuracy and classification reports.
- Confusion matrix visualizations for model evaluation.

#### Dataset and Input Features
The project report describes the following input attributes:

| Feature | Description |
|---|---|
| Item_ID | Unique grocery item identifier |
| Category | Main product category |
| Sub_Category | Product subcategory |
| Price | Product selling price |
| Quantity_Sold | Number of units sold |
| Day_of_Week | Day associated with the sales record |
| Store_ID | Store identifier |

**Target Variable:** `Sales_Category` — Low, Medium, or High.

The target can also be derived from `Total_Sales` when a pre-labelled sales category is unavailable.

#### Machine Learning Algorithms
The project compares four classification algorithms:

1. **Logistic Regression** – Used as a classification approach to predict sales categories.
2. **Decision Tree** – Classifies sales performance using decision-based rules.
3. **Random Forest** – Combines multiple decision trees to improve prediction performance.
4. **Support Vector Machine (SVM)** – Classifies sales records by identifying suitable decision boundaries.

#### Reported Model Performance

| Algorithm | Accuracy |
|---|---:|
| Logistic Regression | 85% |
| Decision Tree | 97% |
| Random Forest | 96% |
| SVM | 91% |

The academic report identifies Decision Tree as the highest-accuracy model. It also describes Random Forest as a consistent model. These values are reproduced from the report and have not been independently verified against the original dataset.

---

### 2. Weather Forecasting Using Machine Learning

#### About the Project
Weather Forecasting uses historical weather data and machine learning to predict temperature based on environmental conditions. The project considers information such as humidity, wind speed, precipitation, atmospheric pressure, visibility, UV index, city, and weather condition to build a predictive model.

#### Objectives
- Understand relationships between weather conditions and temperature.
- Prepare historical weather data for machine learning.
- Identify useful numerical and categorical features.
- Build a model to predict temperature.
- Improve model performance through hyperparameter tuning.
- Evaluate predictions using regression evaluation metrics.

#### Key Features
- Weather dataset loading and preprocessing.
- Handling of numerical and categorical features.
- Numerical feature scaling.
- Categorical feature encoding using One-Hot Encoding.
- Random Forest Regression model implementation.
- Hyperparameter tuning using RandomizedSearchCV.
- Model evaluation using MAE, RMSE, and R².
- Actual-versus-predicted temperature visualization.
- Residual distribution visualization.
- Saving the trained machine learning model.

#### Dataset and Input Features

| Feature | Description |
|---|---|
| Humidity (%) | Moisture level in the air |
| Wind Speed | Speed of the wind |
| Precipitation (%) | Precipitation measurement |
| Pressure (hPa) | Atmospheric pressure |
| Visibility (km) | Visibility distance |
| UV Index | Ultraviolet radiation level |
| City | Location associated with the observation |
| Weather Condition | Recorded weather condition |

**Target Variable:** Temperature — the model predicts temperature using the available weather attributes.

#### Machine Learning Algorithm

**Random Forest Regressor**

The project uses Random Forest Regression to predict temperature. The algorithm combines predictions from multiple decision trees to produce a numerical output.

The model uses a preprocessing pipeline:
- Numerical features are scaled before model training.
- Categorical features are transformed using One-Hot Encoding.
- RandomizedSearchCV is used to explore model hyperparameters and improve performance.

#### Model Evaluation

The academic report records the following results:

| Evaluation Metric | Reported Value |
|---|---:|
| Mean Absolute Error (MAE) | 1.07 |
| Root Mean Squared Error (RMSE) | 1.69 |
| R-squared (R²) | 0.92 |

These values are taken from the academic report. They are not independently reproduced because the original dataset was not included with the report.

---

## Project Workflow

Both projects follow a machine learning workflow, with different prediction objectives.

1. **Data Collection:** Load the available grocery sales and historical weather datasets.
2. **Data Preprocessing:** Prepare the data and handle the required input features.
3. **Feature Selection:** Identify the attributes relevant to each prediction task.
4. **Model Building:** Apply suitable machine learning algorithms.
5. **Model Training:** Train models using the prepared datasets.
6. **Model Evaluation:** Compare predictions using appropriate evaluation metrics.
7. **Visualization:** Present model performance through charts and evaluation results.

## Technologies and Tools Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning and model evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Saving trained models |
| Excel / CSV | Dataset storage and handling |

## Repository Structure

```text
.
├── grocery_price_optimization/
│   └── train_grocery_models.py
│
├── weather_forecasting/
│   └── train_weather_model.py
│
├── data/
│   ├── grocery/
│   │   └── README.md
│   └── weather/
│       └── README.md
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### Prerequisites
- Python 3.10 or later
- pip package manager
- Grocery sales dataset for the grocery project
- Weather dataset for the weather project

### Step 1: Clone the Repository

```bash
git clone <your-repository-url>
cd Grocery-Price-Optimization-and-Weather-Forecasting-Using-Machine-Learning
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 3: Activate the Environment

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Projects

### Grocery Price Optimization

Place the grocery dataset at:

`data/grocery/grocery_sales.csv`

Run:

```bash
python grocery_price_optimization/train_grocery_models.py
```

The dataset should contain the required input features and either `Sales_Category` or `Total_Sales`.

If `Sales_Category` is not available, the implementation creates Low, Medium, and High labels using three quantile groups of `Total_Sales`. This is an implementation assumption because the original report does not specify the exact category thresholds.

**Generated Outputs:**
- Model comparison results.
- Classification reports.
- Confusion matrix visualizations.
- Model evaluation results.

### Weather Forecasting

Place the weather dataset at:

`data/weather/weather_data.xlsx`

Run:

```bash
python weather_forecasting/train_weather_model.py
```

The workbook should contain the weather features described in the dataset section and a temperature column whose name contains `Temperature`.

**Generated Outputs:**
- Trained Random Forest model.
- Evaluation metrics in JSON format.
- Actual-versus-predicted temperature graph.
- Residual distribution graph.

---

## Results and Visualizations

The projects use model evaluation and visual outputs to understand prediction performance.

**Grocery Price Optimization**
- Compares the performance of four classification algorithms.
- Uses accuracy and classification reports.
- Displays confusion matrices to understand classification results.

**Weather Forecasting**
- Evaluates temperature predictions using MAE, RMSE, and R².
- Compares actual and predicted temperature values.
- Uses residual distribution to examine prediction errors.

---

## Future Enhancements

- Include larger and more recent datasets for improved model training.
- Explore additional machine learning algorithms.
- Improve feature selection and model tuning.
- Add interactive dashboards to present prediction results.
- Develop user-friendly interfaces for exploring model predictions.
- Evaluate models on unseen data to assess their generalization.

---

## Notes on Reproducibility

- The implementations are organized based on the methods and feature descriptions in the academic project reports.
- The original datasets were not included with the supplied reports.
- The grocery classification thresholds are not specified in the original report; quantile-based labels are used when a pre-labelled target is unavailable.
- The weather report describes 30 cleaned records, but its original workbook was not supplied.
- The reported model metrics are included for reference and are not guaranteed to reproduce without the original datasets and experimental setup.

---

## Conclusion

These projects demonstrate the use of Python and machine learning for two different prediction tasks: understanding grocery sales performance and forecasting weather conditions. They provide hands-on experience with data preprocessing, feature engineering, classification, regression, model training, hyperparameter tuning, evaluation, and visualization.

---
