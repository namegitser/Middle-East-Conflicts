# Middle East Conflict Forecasting System

This project is a full machine-learning pipeline for forecasting weekly conflict fatalities in the Middle East using historical ACLED-style event data. The repository combines data cleaning, feature engineering, model training, evaluation, explainability, risk classification, temporal validation, and a Streamlit dashboard into one end-to-end workflow.

The main idea is simple but important:

- Start from raw conflict event logs.
- Convert them into a consistent weekly panel with one row per country-week.
- Create lag-based and rolling features that capture recent conflict momentum.
- Train forecasting models to predict next-week fatalities.
- Compare model performance using explicit metrics.
- Build a risk layer that flags unusually dangerous weeks.
- Use explainability tools to understand which signals matter most.
- Produce outputs for both analysis and a dashboard.

This repository is not just a single model notebook. It is a structured, reproducible forecasting system designed to answer a specific question: can recent historical conflict activity predict next week’s fatalities well enough to beat a simple baseline?

---

## 1. What this project does

At a high level, the repository does the following:

1. Loads raw event data from ACLED-like CSV files.
2. Aggregates the raw records into a weekly panel at the country level.
3. Fills missing country-week combinations with zeros so that the data remains consistent for time-series modeling.
4. Engineers temporal features such as:
   - lagged fatalities
   - lagged events
   - rolling mean and rolling standard deviation of recent fatalities
5. Builds a supervised learning target:
   - predict next week’s fatalities from the current week’s historical information
6. Trains several models:
   - naive baseline
   - Ridge regression
   - Random Forest regression
7. Evaluates models using MAE and RMSE.
8. Builds a binary risk classifier that labels weeks as high-risk or normal based on a country-specific percentile threshold.
9. Implements optional analysis techniques such as:
   - walk-forward validation
   - out-of-time testing
   - SHAP explainability
   - spatial lag features at ADMIN1 level
10. Produces dashboard-ready parquet data and a Streamlit app.

The project is therefore both a forecasting system and a modeling workflow for structured conflict data.

---

## 2. Why this problem is difficult

Conflict data is difficult because it is:

- sparse
- highly irregular
- zero-inflated
- skewed toward a few extreme spikes
- non-stationary across time
- partially driven by unseen political, economic, and security events

Because of this, a naive model can be surprisingly competitive. The core question is therefore not “can we make a prediction?” but “can we build a model that meaningfully improves over persistence?”

In this repo, the benchmark is the naive model:

$$
\hat{y}_{t+1} = y_t
$$

That means the simplest forecast assumes that next week’s fatalities will be the same as this week’s fatalities.

This is a strong baseline because conflict tends to be persistent over short periods, especially in ongoing conflicts.

---

## 3. Data source and data assumptions

The project expects raw data in a tabular CSV format similar to ACLED event extracts.

The code uses the following minimum required columns:

- WEEK
- COUNTRY
- EVENTS
- FATALITIES

Additional columns such as ADMIN1, CENTROID_LATITUDE, and CENTROID_LONGITUDE are used in the spatial modeling workflow.

### Important data assumption

A raw event log usually contains only rows for weeks or places where something happened. It does not explicitly contain rows for peaceful or inactive weeks.

For time-series forecasting, that creates a major issue: if the pipeline simply tries to build lags from the raw table, the model would silently assume missing weeks are unknown instead of truly zero. That would be a form of leakage or incorrect temporal handling.

To solve that, the project creates a complete panel:

- all weeks in the observed date range
- all countries in the dataset
- all missing country-week combinations are filled with 0 for EVENTS and FATALITIES

This is a key design choice in the code and a core part of the project’s methodology.

---

## 4. System architecture

The repository is organized around a clear pipeline:

1. Raw data ingestion
2. Temporal panel construction
3. Feature engineering
4. Train/test splitting
5. Model training
6. Evaluation
7. Risk classification
8. Explainability
9. Dashboard data generation
10. Optional advanced analysis

The code is split into modules so the workflow is testable and reusable.

### Repository layout

- `src/data_processing.py` — data loading, aggregation, panel construction, zero-filling
- `src/features.py` — lag features, rolling features, target shifting
- `src/models.py` — baseline, Ridge, Random Forest model definitions
- `src/split.py` — chronological train/test split
- `src/evaluations.py` — MAE/RMSE evaluation utilities
- `src/risk_classifier.py` — high-risk classification using country-specific thresholds
- `src/SHAP_explain.py` — TreeSHAP explainability script
- `src/spatial_modeling.py` — ADMIN1 aggregation and spatial lag computation
- `src/validation.py` — walk-forward validation
- `src/out_of_time_test.py` — frozen-model out-of-time evaluation
- `src/gradboost_comparison.py` — broader model comparison including XGBoost and LightGBM
- `src/dashboard_data.py` — generates dashboard parquet output
- `app/app.py` — Streamlit dashboard
- `data/raw-data/` — raw CSV inputs
- `data/processed/` — processed parquet outputs
- `models/` — saved model artifacts
- `notebooks/` — exploratory notebooks
- `tests/` — project test files

---

## 5. Core ML problem formulation

The repository treats this as a supervised time-series regression problem.

### Target variable

The target is:

- next week’s fatalities for each country

In code, the target is named:

- `next_week_fatalities`

This is created by shifting the fatality series forward by one week within each country.

### Forecasting task

Given historical information up to week t, predict:

```math
\(\hat{y}_\){t+1}
```

where $y_t$ is current fatalities and $y_{t+1}$ is next week’s fatalities.

### Why this is a valid supervised setup

The full weekly panel gives one row per country-week. The feature row at time t contains information from the past (lags, rolling windows, and recent events). The target at that same row is the actual number of fatalities in the next week.

This is a standard supervised learning framing for a one-step-ahead forecasting problem.

---

## 6. Data processing pipeline

### 6.1 Loading the raw file

The function `load_raw(path)` reads the CSV and checks that the required columns exist.

It then converts the `WEEK` column to a pandas datetime format.

### 6.2 Aggregation to country-week

The function `aggregate_to_country_week(df)` collapses all raw events into a single row per country-week.

That means multiple event records in the same country during the same week are summed into:

- `EVENTS`
- `FATALITIES`

This is the first major transformation because the raw data is an event log, not a weekly panel.

### 6.3 Complete panel construction

The function `build_complete_panel(df)` creates a full grid:

```math
\([\text{all weeks}] \times [\text{all countries}] \%\%\)MAGIT_PARSER_PROTECT%%```

This is then reindexed and all missing combinations are filled with zero.

This gives a balanced panel with no gaps in the time series and no missing country-week rows.

### 6.4 Why zero-fill matters

Without this step, the lag logic would produce irregular time gaps, and the model would potentially treat “missing in raw data” as “unknown” rather than “zero observed activity.”

The repository explicitly adopts the assumption that:

- missing country-week records mean no recorded conflict activity in that period
- not a missing-value situation

This is a strong and sensible assumption for weekly ACLED-style aggregates, especially for a first-pass forecasting system.

---

## 7. Feature engineering

The repo’s feature engineering is centered around short-term historical structure.

### 7.1 Lag features

In `src/features.py`, the function `add_lag_features(df)` adds:

- `fatalities_lag1`
- `fatalities_lag2`
- `fatalities_lag3`
- `events_lag1`

These are generated within each country group using pandas groupby shifts.

For example:

```math
\(\text{fatalities\_lag1}_\){i,t} = y_{i,t-1}
```

```math
\(\text{fatalities\_lag2}_\){i,t} = y_{i,t-2}
```

```math
\(\text{fatalities\_lag3}_\){i,t} = y_{i,t-3}
```

This gives the model recent memory of how violent the country has been over the last few weeks.

### 7.2 Rolling features

The function `add_rolling_features(df)` adds:

- `fatalities_roll_mean`
- `fatalities_roll_std`

These are built from the lagged fatality signal using a 4-week window over the previous periods.

For a country i:

```math
\(\mu_{i,t} = \frac{1}{4} \sum_\){k=1}^{4} y_{i,t-k}
```

```math
\(\sigma_{i,t} = \sqrt\){\(\frac{1}{4-1} \sum_\){k=1}^{4} (y_{i,t-k} \(- \mu_\){i,t})^2}
```

The rolling mean captures recent average intensity, while the rolling standard deviation captures volatility.

### 7.3 Target creation

The function `add_target(df)` creates:

- `next_week_fatalities`

by shifting fatalities one step forward within each country:

```math
\(\text{next\_week\_fatalities}_\){i,t} = y_{i,t+1}
```

After that, rows with missing target values are dropped, because the target is not available for the final week of the panel.


### 7.4 Feature set used by the main models

The main models in `src/models.py` use these features:

- `fatalities_lag1`
- `fatalities_lag2`
- `fatalities_lag3`
- `events_lag1`
- `fatalities_roll_mean`
- `fatalities_roll_std`

This is a compact, interpretable feature space that focuses on short-term conflict momentum and recent volatility.

---

## 8. Train/test splitting

The project uses a chronological split rather than random shuffling.

The function `chronological_split(df, train_frac=0.8)`:

- sorts unique weeks
- picks a cutoff date
- trains on weeks before the cutoff
- tests on weeks from the cutoff onward

This prevents future data leakage, which would otherwise make the evaluation artificially optimistic.

The importance of this step cannot be overstated: in time-series forecasting, shuffling rows would create data leakage.

---

## 9. Models in this repository

## 9.1 Naive baseline

`naive_forecast(df)` returns the current week’s fatalities as the predicted next-week fatalities.

This implements:

$$
\hat{y}_{t+1} = y_t
$$

This is the benchmark. If the more complex models cannot beat it, the forecasting problem is not yet meaningfully solved.

## 9.2 Ridge regression

`train_ridge(train_df)` fits a `Ridge(alpha=1.0)` model.

Ridge solves:

$$
\min_{\beta} \left\| X\beta - y \right\|^2 + \alpha \left\| \beta \right\|^2
$$

where:

- $X$ is the feature matrix
- $y$ is the target
- $\beta$ are the regression coefficients
- $\alpha$ controls regularization strength

Ridge is useful because the lagged features are strongly correlated. Regularization stabilizes the model and makes the relationship less sensitive to multicollinearity.

## 9.3 Random Forest regressor

`train_random_forest(train_df)` fits a `RandomForestRegressor` with:

- `n_estimators=300`
- `random_state=42`
- `n_jobs=-1`

Random Forests are ensemble methods made of many decision trees. Each tree learns a partition of the feature space, and the final prediction is the average across trees.

This makes them effective for:

- nonlinear relationships
- interactions between lag features and volatility measures
- mixed patterns that are not well captured by a linear model

For conflict forecasting, this is especially attractive because escalation behavior is rarely perfectly linear.

## 9.4 Risk classifier

`src/risk_classifier.py` adds a binary label called `high_risk_next_week`.

The label is defined as:

- next week’s fatalities exceed that country’s own expanding 90th percentile threshold from prior history

This is computed with an expanding quantile, shifted one step back to prevent leakage.

The classifier then trains a `RandomForestClassifier` with `class_weight="balanced"` to handle class imbalance.

This is a practical way to turn the forecasting system into an operational alert tool.

---

## 10. Evaluation metrics

The core evaluation utilities are defined in `src/evaluations.py`.

### 10.1 Mean Absolute Error (MAE)

$$
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

MAE is the primary evaluation metric in this repository because it is easy to interpret and measures average absolute forecasting error in the same units as fatalities.

### 10.2 Root Mean Squared Error (RMSE)

$$
RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

RMSE penalizes large errors more strongly. This is useful because conflict forecasting errors can be very uneven, and the project wants a metric that reflects both typical and severe misses.

### 10.3 Why MAE and RMSE together

- MAE tells you typical forecast error.
- RMSE tells you how much the model struggles on large misses.

That combination gives a more complete view of performance than either metric alone.

---

## 11. Explainability and interpretation

### 11.1 SHAP

`src/SHAP_explain.py` computes exact TreeSHAP values for the trained Random Forest.

SHAP values answer the question:

- how much did each feature contribute to this specific prediction?

In tree models, SHAP provides a principled way to distribute a prediction among its input features.

This is useful for:

- global importance analysis
- local prediction debugging
- understanding which lag or volatility features drive the model

### 11.2 Why SHAP matters here

In a conflict forecasting system, a prediction without explanation is risky. SHAP helps show whether the model is relying on broad recent momentum, volatility, or event counts.

This matters when explaining results to stakeholders or when comparing different model behaviors.

---

## 12. Spatial modeling

The research project also includes a sub-national extension in `src/spatial_modeling.py`.

This version aggregates at the `ADMIN1` level and adds spatial spillover features.

### 12.1 ADMIN1 aggregation

The function `aggregate_to_admin1_week(df)` groups data by:

- WEEK
- COUNTRY
- ADMIN1

This produces a finer-grained panel than the country-only version.

### 12.2 Panel construction for admin regions

`build_admin1_panel(df)` builds a complete panel across weeks and provinces, then fills missing values with zeros where needed.

### 12.3 Spatial lag feature

The function `add_spatial_lag(df, n_neighbors=5)` computes the average fatalities of neighboring admin regions at the previous week.

This is motivated by the idea that conflict does not always stay within one administrative area. Violence in nearby provinces may spill over and affect adjacent regions.

The repo uses a BallTree with the haversine distance metric to find nearby administrative centroids.

The haversine distance is:

$$
 d = 2R\arcsin\left(\sqrt{\sin^2\left(\frac{\phi_2-\phi_1}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\lambda_2-\lambda_1}{2}\right)}\right)
$$

Where:

- $R$ is the Earth radius
- $\phi$ is latitude
- $\lambda$ is longitude

This lets the system measure distance between nearby provinces and compute a spatial neighbor signal.

---

## 13. Validation strategy

This repository includes multiple validation designs.

### 13.1 Chronological split

Used for a basic train/test evaluation.

### 13.2 Walk-forward validation

`src/validation.py` implements rolling-origin validation.

This repeatedly trains on an expanding history and evaluates on the next few weeks. It is more honest than a single holdout because it checks whether the model is stable across different time periods.

### 13.3 Out-of-time testing

`src/out_of_time_test.py` freezes a historical model trained on data through March 7, 2026, then evaluates it on later data from March 8 through August 29, 2026.

This is one of the strongest tests in the repository because it measures whether the model can handle genuinely future, unseen periods.

---

## 14. Advanced experiments included in the codebase

Several advanced scripts exist as experiments or extensions beyond the base pipeline.

### 14.1 Gradient boosting comparison

`src/gradboost_comparison.py` compares:

- naive baseline
- Ridge
- Random Forest
- Random Forest with log-transformed target (`log1p`)
- XGBoost
- LightGBM
- LightGBM with Tweedie objective

This script demonstrates that the repo is not limited to a single model family.

### 14.2 Tweedie objective

The LightGBM Tweedie setup is especially relevant for data with many zeros and a long tail of positive counts.

The Tweedie distribution is useful because it combines aspects of:

- Poisson-like count behavior
- Gamma-like continuous behavior

This makes it a natural choice for zero-inflated, highly skewed count data.

### 14.3 Log-target transformation

The log-target experiment uses:

$$
\log(1 + y)
$$

This helps reduce the impact of extreme fatalities on tree-based regressors.

Predictions are then inverted using:

$$
\exp(\tilde{y}) - 1
$$

This is a standard technique for highly skewed target variables.

---

## 15. Dashboard and outputs

### 15.1 Dashboard data generation

`src/dashboard_data.py` generates a dataset for the app.

It:

- loads cleaned and engineered data
- retrieves a trained model or retrains a Random Forest as needed
- creates `predicted_fatalities`
- writes the result to:

`data/processed/latest_forecast.parquet`

### 15.2 Streamlit app

`app/app.py` is a multi-tab dashboard that loads the parquet file and displays:

- overview metrics
- time-series plots
- country-level forecast comparisons
- mapped latest predictions
- forecast tables

The dashboard enables interactive viewing of the model outputs without needing to rerun the analysis manually.

---

## 16. How to run the project

## 16.1 Environment setup

Create a virtual environment and install dependencies:

```bash
python -m venv ml_env

# Windows
ml_env\Scripts\activate

# macOS / Linux
source ml_env/bin/activate

pip install -r requirements.txt
```

The current `requirements.txt` includes the core scientific stack:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- openpyxl
- jupyter
- pytest

Additional packages such as `streamlit`, `shap`, `lightgbm`, and `xgboost` are used by optional scripts and should be installed separately when needed.

## 16.2 Generate base forecast data

From the project root, run:

```bash
python src/dashboard_data.py
```

This produces the parquet file needed by the dashboard.

## 16.3 Start the dashboard

```bash
streamlit run app/app.py
```

Then open the local Streamlit URL shown in the terminal.

## 16.4 Run the evaluation pipeline

```bash
python src/evaluations.py
```

This loads the data, builds features, splits into train/test, trains the baseline models, and prints a leaderboard of MAE/RMSE results.

## 16.5 Run the risk classifier

```bash
python src/risk_classifier.py
```

This builds the high-risk label and evaluates the classifier using ROC-AUC and classification-report metrics.

## 16.6 Run SHAP analysis

```bash
python src/SHAP_explain.py
```

This trains a Random Forest and computes feature contribution values.

## 16.7 Run walk-forward validation

```bash
python src/validation.py
```

This performs repeated expanding-window validation and reports average MAE across folds.

## 16.8 Run out-of-time testing

```bash
python src/out_of_time_test.py
```

This evaluates a frozen historical model on later data and then retrains a production model on the expanded dataset.

---

## 17. Important modeling caveats

This project is useful, but it also has real limitations.

### 17.1 Extreme spikes are hard to forecast

Conflict fatalities are heavily right-skewed, with many low or zero weeks and occasional enormous spikes. Tree-based methods can struggle to extrapolate beyond the range seen in training.

### 17.2 Non-stationarity

Conflict dynamics change over time due to war escalation, ceasefires, political shocks, and shifts in reporting. A model trained on one period may not generalize perfectly to another.

### 17.3 Missing variables

This system uses only recent historical conflict patterns. It does not directly incorporate:

- political context
- economic indicators
- diplomatic events
- troop movement
- rebel organization changes
- sanctions and aid flows

Because of that, the model is best understood as a pattern-based forecasting system, not a full geopolitical simulator.

### 17.4 Country-level smoothing

The main pipeline aggregates to country-week. That makes the model much easier to build and interpret, but it also hides within-country regional variation.

This is one reason the repository includes the ADMIN1 spatial extension.

---

## 18. What results should you expect

The project is designed to produce a leaderboard showing comparisons such as:

- Naive Baseline
- Ridge Regression
- Random Forest
- optionally XGBoost / LightGBM / Tweedie models

The exact numbers depend on your dataset and split. The expected pattern is that:

- naive is a strong benchmark
- Ridge is useful but often limited
- Random Forest often captures nonlinear structure well
- advanced models may improve over the baseline depending on the dataset and feature design

The repo contains the code needed to compute and compare these results explicitly.

---

## 19. Practical interpretation of the project

This project is best understood as a pipeline for answering a very specific forecasting question:

“Can recent conflict history be used to predict next week’s fatalities at a useful level of accuracy?”

The answer is not assumed up front. The code is built to test that directly.

The repository therefore does two things at once:

1. It provides a practical forecasting workflow.
2. It creates a rigorous comparison between simple and more expressive models.

That makes it a strong learning project and a good foundation for a more advanced portfolio system later.

---

## 20. Recommended reading order

If you are new to this repo, read the files in this order:

1. `src/data_processing.py`
2. `src/features.py`
3. `src/models.py`
4. `src/split.py`
5. `src/evaluations.py`
6. `src/risk_classifier.py`
7. `src/validation.py`
8. `src/spatial_modeling.py`
9. `src/SHAP_explain.py`
10. `app/app.py`

That sequence follows the natural workflow of the repository from raw data to prediction to interpretation.

---

## 21. Project status and notes

This repository contains both:

- the original capstone-style implementation
- additional exploratory and portfolio-style extensions

The current codebase reflects a practical, research-oriented pipeline rather than a single hardcoded notebook.

A few implementation notes:

- the project expects ACLED-like CSV input files in `data/raw-data/`
- the dashboard expects a processed parquet file in `data/processed/`
- the out-of-time workflow is optional and depends on additional post-March-2026 data being present
- some advanced scripts require extra packages beyond the base requirements file

---

## 22. Quick start summary

If you want the shortest possible path:

```bash
python -m venv ml_env
source ml_env/bin/activate   # or ml_env\Scripts\activate on Windows
pip install -r requirements.txt
python src/evaluations.py
python src/risk_classifier.py
python src/dashboard_data.py
streamlit run app/app.py
```

That gives you the core forecasting workflow, the risk layer, and the dashboard in a runnable form.

---

## 23. Final takeaway

This repository is a complete conflict forecasting project that combines:

- data engineering
- temporal feature design
- classical and ensemble machine learning
- risk classification
- spatial spillover features
- explainability
- time-series validation
- dashboard delivery

In short, it is a full end-to-end machine learning system for predicting next-week conflict fatalities and operational risk in the Middle East using historical event data.
