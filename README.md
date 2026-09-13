# Middle East Conflict Forecasting System

A complete, reproducible machine-learning system for forecasting **next-week conflict fatalities** at the country level in the Middle East, using historical ACLED-style event data. The project doesn't just fit a model — it is built to rigorously answer one question:

> **Can recent historical conflict activity predict next week's fatalities well enough to beat a simple "nothing changes" forecast?**

This README documents not just *what* the code does, but *why* every technical decision was made — the statistics, the ML theory, and the math behind each choice — so that the repository is self-contained as both a working system and a study reference.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Why Conflict Forecasting Is Hard](#2-why-conflict-forecasting-is-hard)
3. [Data Source & Core Assumptions](#3-data-source--core-assumptions)
4. [Repository Architecture](#4-repository-architecture)
5. [Data Pipeline: Theory & Implementation](#5-data-pipeline-theory--implementation)
6. [Feature Engineering: Theory & Math](#6-feature-engineering-theory--math)
7. [Target Variable & Supervised Learning Formulation](#7-target-variable--supervised-learning-formulation)
8. [Train/Test Splitting & Data Leakage](#8-traintest-splitting--data-leakage)
9. [Models: Full Theory](#9-models-full-theory)
10. [Evaluation Metrics: Full Theory](#10-evaluation-metrics-full-theory)
11. [Risk Classification Layer](#11-risk-classification-layer)
12. [Explainability: SHAP Theory](#12-explainability-shap-theory)
13. [Spatial Modeling: Theory](#13-spatial-modeling-theory)
14. [Validation Strategy: Theory](#14-validation-strategy-theory)
15. [Results & How to Interpret Them](#15-results--how-to-interpret-them)
16. [Limitations](#16-limitations)
17. [How to Run — Step by Step](#17-how-to-run--step-by-step)
18. [Project Development Timeline](#18-project-development-timeline)
19. [Future Work](#19-future-work)
20. [Further Reading](#20-further-reading)

---

## 1. Problem Statement

**Grain of prediction:** one observation = one `COUNTRY × WEEK`.

**Target:** `next_week_fatalities` — total conflict fatalities in that country during the week immediately following the observation week.

**Formulation in plain notation** (used throughout this README instead of LaTeX, for GitHub rendering stability):

```
Given information known at the end of week t for country i,
predict y(i, t+1) = fatalities in country i during week t+1.
```

This is a **one-step-ahead, panel-structured, supervised regression problem** — not a classification problem, not a generative simulation, and not an attempt to predict specific events, political outcomes, or causes of conflict.

---

## 2. Why Conflict Forecasting Is Hard

Understanding *why* this is a hard problem is what justifies almost every later modeling decision, so it's worth being precise about the statistical properties involved.

### 2.1 Zero-inflation
Most country-weeks have **zero or near-zero** fatalities. A large point mass at zero violates the assumptions of ordinary least squares (which assumes roughly continuous, symmetric residuals) and of naive count models like plain Poisson regression (which assume mean ≈ variance).

### 2.2 Heavy right skew / extreme spikes
Observed target distribution in this project:
```
mean    ≈ 4.75
median  = 0
std     ≈ 39.5
max     = 8102
```
A distribution where the **standard deviation is ~8x the mean**, and the max is ~1700x the mean, is extremely heavy-tailed. This means:
- A model that just predicts near-zero for everything will already achieve a deceptively "good-looking" MAE, because most rows *are* near zero.
- The few extreme rows dominate squared-error metrics disproportionately.
- Standard cross-validation assumptions (i.i.d., similar-variance folds) don't hold cleanly.

### 2.3 Non-stationarity
The statistical properties of conflict (mean fatalities, volatility, spatial spread) **change over time** — escalation phases, ceasefires, and political shocks shift the data-generating process itself. A model is not just interpolating a fixed function; it's forecasting a moving target. This is why simple k-fold cross-validation (which assumes exchangeable, time-independent samples) is invalid here — see Section 8.

### 2.4 Persistence / autocorrelation
Despite the above, conflict has strong **short-term autocorrelation** — an active conflict zone this week is very likely to still be active next week. This single fact is what makes the naive baseline (Section 9.1) so strong, and it's the central benchmark the entire project is built around.

### 2.5 Omitted variables
The available features are purely **endogenous** (derived from the fatalities/events series itself). Real conflict escalation is driven by political negotiations, troop movements, foreign intervention, ceasefire agreements, and economic shocks — none of which are in this dataset. This bounds the theoretical ceiling of any model trained only on lagged fatalities/events.

---

## 3. Data Source & Core Assumptions

Data: ACLED-style weekly aggregated Middle East conflict event extracts, with minimum required columns:

```
WEEK, COUNTRY, EVENTS, FATALITIES
```

Extended columns used for spatial modeling: `ADMIN1`, `CENTROID_LATITUDE`, `CENTROID_LONGITUDE`.

### 3.1 The critical structural fact about the raw data
A raw ACLED-style export is an **event log**, not a panel: it can contain **multiple rows per country-week** — one per admin-region/event-type/sub-event-type combination. This means:

```
df.groupby('COUNTRY')['FATALITIES'].shift(-1)
```
does **not** compute "next week's fatalities." It computes "the next *row* for that country" — which, if a country has several rows in the same week, might still be the *same* week. This was an early, documented mistake in this project and is exactly why Section 5.2 (aggregation) must always happen **before** any lag/shift logic.

### 3.2 The missing-week assumption
ACLED-style exports only contain rows for weeks/places where an event was recorded. There is no explicit "zero activity" row. The project makes an explicit, documented modeling assumption:

> **Absence of a country-week record means zero recorded conflict activity that week — not "unknown" or "missing data."**

This is a **Missing Completely At Random vs. Missing Not At Random** distinction from a statistics standpoint: we are asserting the missingness mechanism is structural (ACLED doesn't log rows for inactivity) rather than informative (data wasn't collected). This assumption is reasonable for a conflict-monitoring dataset like ACLED but is explicitly stated here because it directly determines model behavior — treating gaps as `NaN` vs. `0` produces materially different lag/rolling features.

---

## 4. Repository Architecture

```
Middle-East-Conflicts/
├── app/
│   └── app.py                     Streamlit dashboard
├── data/
│   ├── raw-data/                  Raw ACLED-style CSV input
│   └── processed/                 Generated parquet outputs
├── models/                        Saved model artifacts (.pkl)
├── notebooks/                     Exploratory analysis notebooks
├── src/
│   ├── data_processing.py         Loading, aggregation, zero-filled panel
│   ├── features.py                Lag features, rolling features, target
│   ├── models.py                  Baseline, Ridge, Random Forest
│   ├── split.py                   Chronological train/test split
│   ├── evaluations.py             MAE / RMSE evaluation utilities
│   ├── risk_classifier.py         High-risk binary classification layer
│   ├── SHAP_explain.py            TreeSHAP explainability
│   ├── spatial_modeling.py        ADMIN1-level panel + spatial lag features
│   ├── validation.py              Walk-forward (rolling-origin) validation
│   ├── out_of_time_test.py        Frozen-model genuine future-data test
│   ├── gradboost_comparison.py    XGBoost / LightGBM / Tweedie comparison
│   └── dashboard_data.py          Generates dashboard-ready parquet
└── tests/                         Unit tests
```

**Design rationale for this module split:** each file maps to exactly one pipeline stage (data → features → split → model → eval → risk → explain → spatial → validate → serve). This makes every stage independently testable and independently swappable — e.g., you can replace `models.py`'s Random Forest with a different estimator without touching feature engineering or evaluation code at all. The alternative (one monolithic script) is faster to write but couples every concern together, making it hard to isolate bugs like the shift-before-aggregation mistake in Section 3.1.

---

## 5. Data Pipeline: Theory & Implementation

### 5.1 Loading (`load_raw`)
Reads the CSV, validates required columns exist, parses `WEEK` to `datetime64`. Failing fast on a missing column here prevents a silent `KeyError` deep inside feature engineering, where it would be much harder to trace back to its source.

### 5.2 Aggregation to country-week (`aggregate_to_country_week`)
```
country_week = raw.groupby(['WEEK', 'COUNTRY']).agg(
    EVENTS = sum(EVENTS),
    FATALITIES = sum(FATALITIES)
)
```
This is a **sum aggregation**, not mean or max, because both `EVENTS` and `FATALITIES` are *counts* — additive quantities across sub-regions and event types within the same country-week. This step is what converts an event log into a proper panel-data structure (one row per unit-of-analysis per time period), which is the precondition for any of the lag/rolling logic in Section 6 to be mathematically meaningful.

### 5.3 Complete panel construction (`build_complete_panel`)
Builds the full Cartesian product:
```
all_weeks × all_countries
```
then reindexes the aggregated data onto that grid, filling missing combinations with `0`.

**Why this matters mathematically:** a lag feature like `fatalities_lag1` is only correctly defined if consecutive rows for a country represent *consecutive weeks*. If a country has a gap (no ACLED record for a quiet week), a naive `.shift(1)` would silently pull the value from the *previous recorded week*, which might be several weeks earlier — corrupting every downstream lag/rolling feature. The complete panel eliminates this failure mode entirely by guaranteeing every country has exactly one row per week with no gaps.

---

## 6. Feature Engineering: Theory & Math

### 6.1 Lag features
```
fatalities_lag1(i,t) = FATALITIES(i, t-1)
fatalities_lag2(i,t) = FATALITIES(i, t-2)
fatalities_lag3(i,t) = FATALITIES(i, t-3)
events_lag1(i,t)     = EVENTS(i, t-1)
```
**Statistical justification:** these are **autoregressive features** — they let a model approximate an AR(3)-style process (a linear/nonlinear function of the last 3 lags), which is the simplest way to encode short-term persistence (Section 2.4) into a supervised-learning feature matrix instead of using a dedicated time-series model like ARIMA.

**Why 3 lags and not more?** Alternatives considered: 1 lag (too little memory — can't distinguish a spike from a sustained trend), 8+ lags (risks overfitting with a modest sample size per country, and dilutes the signal since conflict autocorrelation decays quickly week-to-week). **Chosen: 3 lags** — enough to represent short-term trend + momentum without excessive dimensionality.

### 6.2 Rolling features
```
fatalities_roll_mean(i,t) = mean( FATALITIES(i, t-4..t-1) )
fatalities_roll_std(i,t)  = std ( FATALITIES(i, t-4..t-1) )
```
- **Rolling mean** approximates the recent *baseline intensity* of conflict for that country — a smoothed signal that's less noisy than any single lag.
- **Rolling standard deviation** approximates recent *volatility* — two countries can have the same average fatalities but very different risk profiles (one steady, one spiking unpredictably), and this feature lets the model distinguish them.

**Why a 4-week window?** Alternatives: a 2-week window (too noisy, barely smooths anything), an 8+ week window (smooths away exactly the recent escalation signal that matters for a 1-week-ahead forecast), an exponentially-weighted moving average / EWMA (weights recent weeks more heavily — arguably better, but harder to explain and audit than a plain window mean). **Chosen: simple 4-week rolling window** — a defensible trade-off between noise reduction and responsiveness, and trivial to explain to a non-technical stakeholder.

### 6.3 The leakage question, resolved precisely
An earlier draft of this project over-corrected and assumed *any* inclusion of the current week `t` in a feature was leakage. That's not quite right. The precise rule is:

```
A feature used to predict week (t+1) may use any information
available by the END of week t — including week t itself.
It must never use information from week (t+1) or later.
```

Under that rule, a rolling window of `t-3..t` (including the current week) would technically be valid too. This project deliberately still uses `t-4..t-1` (excluding week `t`) anyway, because it produces a feature definition that is unambiguous, trivially auditable, and identical in spirit to "everything you knew as of last week" — a stricter but easier-to-explain standard than the theoretical maximum.

### 6.4 Final feature set
```
[fatalities_lag1, fatalities_lag2, fatalities_lag3,
 events_lag1, fatalities_roll_mean, fatalities_roll_std]
```
A compact, fully endogenous, fully explainable 6-feature space — intentionally small so that every model comparison in Section 9 is testing *modeling capacity*, not feature-engineering luck.

---

## 7. Target Variable & Supervised Learning Formulation

```
next_week_fatalities(i, t) = FATALITIES(i, t+1)
```
Implemented as a **grouped forward shift**: `groupby('COUNTRY')['FATALITIES'].shift(-1)`. The final week for each country has no future week to shift in, producing `NaN` — those rows are dropped before training, since there is no ground truth to learn from or evaluate against.

**Why this is a valid supervised-learning setup:** each row's feature vector contains only information dated ≤ week `t`; its label is the true outcome at week `t+1`. This satisfies the core requirement of any legitimate one-step-ahead forecasting model: **no feature may be a function of the label or of anything chronologically after it.**

---

## 8. Train/Test Splitting & Data Leakage

### 8.1 Why random/shuffled splitting is invalid here
Standard `train_test_split(shuffle=True)` assumes rows are **i.i.d.** (independent and identically distributed). Time-ordered, autocorrelated data violates this directly: a shuffled split would let the model train on week 500 and be tested on week 499 — i.e., train on the *future* and test on the *past*, producing an artificially optimistic, meaningless evaluation.

### 8.2 Why row-order-based splitting (`shuffle=False` after sorting by country) is *also* invalid
If data is sorted `Bahrain → Egypt → Iraq → ... → Yemen` and then split by the last 20% of *rows*, the test set ends up disproportionately representing **alphabetically later countries**, not later **time periods**. This was an actual mistake made during development — worth stating explicitly because it's a subtle, easy-to-repeat error.

### 8.3 The correct approach: split by date, not by row
```
unique_weeks = sorted(all distinct WEEK values)
cutoff = unique_weeks[floor(0.8 * len(unique_weeks))]

train = rows where WEEK <  cutoff
test  = rows where WEEK >= cutoff
```
Every country now has historical data in train and later data in test — the model is evaluated on its ability to generalize **forward in time**, which is the only evaluation that means anything for a forecasting task.

### 8.4 Beyond a single cutoff: walk-forward validation
A single train/test cutoff answers "how good is the model at this one point in history?" It does **not** tell you whether performance is stable across different time regimes (calm periods vs. escalation periods). See Section 14.2 for the walk-forward methodology that addresses this.

---

## 9. Models: Full Theory

All models are compared on the identical feature set and identical split, so any performance difference reflects **modeling capacity**, not feature engineering differences.

### 9.1 Naive baseline (persistence forecast)
```
y_hat(t+1) = y(t)
```
**Theoretical grounding:** this is the optimal 1-step-ahead forecast under a **random walk model** — i.e., if fatalities followed `y(t+1) = y(t) + noise`, this baseline would already be unbeatable in expectation (a classic result from time-series theory: for a true random walk, no feature-based model can systematically outperform the last observed value). The entire point of comparing every other model against this baseline is to test whether conflict fatalities deviate meaningfully from a pure random walk — i.e., whether there's exploitable structure beyond "next week looks like this week."

**Why this baseline and not another:** alternatives considered — seasonal-naive (same week last year: needs 52+ weeks of history per country and assumes annual seasonality that conflict doesn't reliably exhibit), historical country mean (ignores recent trend entirely, weaker for short-horizon forecasts). **Chosen: last-observed-value persistence** — the hardest, most standard baseline for 1-week-ahead time series.

### 9.2 Ridge Regression
**Model:**
```
y_hat = X * beta
```
**Objective function (what `.fit()` minimizes):**
```
minimize over beta:   ||X*beta - y||^2  +  alpha * ||beta||^2
```
where `||X*beta - y||^2` is the sum of squared residuals (ordinary least-squares loss) and `alpha * ||beta||^2` is an **L2 penalty** on the coefficient vector's magnitude.

**Closed-form solution:**
```
beta_ridge = (X^T X + alpha*I)^-1  X^T y
```
Compare to plain OLS: `beta_ols = (X^T X)^-1 X^T y`. The `+ alpha*I` term is what makes Ridge numerically stable even when `X^T X` is close to singular.

**Why this matters here specifically:** `fatalities_lag1`, `fatalities_lag2`, `fatalities_lag3`, and `fatalities_roll_mean` are, by construction, **strongly correlated with each other** (they're all derived from the same underlying series at nearby time offsets). This is **multicollinearity**, and under plain OLS it causes `(X^T X)^-1` to become numerically unstable — small changes in the data produce wildly different coefficient estimates (high variance). The L2 penalty shrinks correlated coefficients toward each other and toward zero, trading a small amount of bias for a large reduction in variance — a direct application of the **bias-variance tradeoff**.

**Why Ridge over alternatives:** plain OLS (unstable here, as above), Lasso/L1 (would zero out some of the lag features entirely — undesirable when we specifically want to see each lag's relative contribution, not a sparse subset), Elastic Net (a reasonable middle ground, adds a second hyperparameter to tune for limited benefit at this feature-space size). **Chosen: Ridge** — handles the known multicollinearity cheaply, keeps all features' contributions visible, one hyperparameter (`alpha`).

### 9.3 Random Forest Regressor
**Base unit — a single regression tree:** at each node, the tree searches over features and split points to find the split that most reduces variance in the target within each resulting child node:
```
Split quality (variance reduction) at node m:
  reduction = Var(y_m) - [ (n_left/n_m)*Var(y_left) + (n_right/n_m)*Var(y_right) ]
```
The tree greedily picks the split maximizing this reduction, recursively, until a stopping condition (max depth, min samples per leaf, etc.).

**Ensembling — bagging (Bootstrap Aggregating):**
1. Draw `B` bootstrap samples (sampling with replacement) from the training data.
2. Fit one regression tree on each bootstrap sample, using a random subset of features at each split (this second randomization step is what distinguishes Random Forest from plain bagged trees).
3. Final prediction = **average** of all `B` trees' predictions:
```
y_hat = (1/B) * sum over b of Tree_b(x)
```
**Why averaging reduces variance:** if each tree has prediction variance `sigma^2` and trees were fully independent, the ensemble variance would be `sigma^2 / B`. In practice trees are correlated (from shared training data), so the reduction is smaller but still substantial — this is the core statistical mechanism that makes Random Forest more stable than any single deep tree.

**Why Random Forest here specifically:** the relationship between recent fatality lags and next-week fatalities is very unlikely to be linear — escalation can behave like a threshold effect (nothing happens until some tipping point, then it spikes) rather than a smooth linear function. Trees naturally model **thresholds and interactions** (e.g., "high `fatalities_lag1` AND high `fatalities_roll_std`" behaving differently than either alone) without requiring the modeler to manually specify interaction terms, and without requiring feature scaling (tree splits are invariant to monotonic transformations of a feature).

**Why Random Forest over alternatives:** plain single decision tree (high variance, overfits badly), Support Vector Regression (requires careful feature scaling and kernel choice, slower at this data volume, less interpretable via feature importances), Gradient Boosting/XGBoost (likely stronger — deliberately introduced as the *next* rung, Section 9.4, rather than the first nonlinear model, so the comparison ladder is legible: linear → bagged-nonlinear → boosted-nonlinear).

### 9.4 Gradient Boosting (XGBoost / LightGBM)
**Core idea — additive modeling via functional gradient descent:** unlike bagging (parallel, variance-reduction), boosting is **sequential, bias-reduction**: each new tree is fit to the *residual errors* of the current ensemble.

```
F_0(x) = initial constant prediction (e.g., mean of y)
For m = 1 to M:
    residual_i = y_i - F_(m-1)(x_i)          [the "gradient" of squared-error loss]
    fit tree h_m(x) to predict residual_i
    F_m(x) = F_(m-1)(x) + learning_rate * h_m(x)
Final prediction: F_M(x)
```
This is literally gradient descent, but performed in **function space** rather than parameter space — each tree is a step in the direction that most reduces the loss.

**Why this can outperform Random Forest:** bagging reduces variance but does nothing about bias (an ensemble of unbiased-but-noisy trees is still centered near the truth on average). Boosting explicitly targets whatever the current ensemble is still getting wrong, which often yields lower bias and, empirically, better accuracy on structured/tabular data — this is well documented in ML benchmarks (gradient-boosted trees are consistently among the strongest performers on tabular regression tasks).

**Why introduced only as a later/optional model here:** it adds a new dependency, more hyperparameters to tune (`learning_rate`, `max_depth`, `n_estimators`, regularization terms), and a higher overfitting risk if used carelessly. The project's guiding principle — don't add complexity until the simpler model's ceiling is understood — places it deliberately after Ridge and Random Forest, not before.

### 9.5 Tweedie objective (LightGBM)
**The problem it targets:** the target distribution (Section 2.2) has a large point mass at zero plus a continuous, heavy-tailed spread of positive values. Squared-error loss (used by Ridge, RF, and plain gradient boosting) implicitly assumes roughly homogeneous, symmetric-around-the-mean residual behavior — a poor match for zero-inflated count-like data.

**What the Tweedie distribution is:** a member of the exponential dispersion family parameterized by a power parameter `p`, where its **variance function** follows:
```
Var(Y) = phi * mean(Y)^p
```
- `p = 0` → Normal distribution (constant variance)
- `p = 1` → Poisson distribution (variance = mean, pure count data)
- `p = 2` → Gamma distribution (continuous, right-skewed, no mass at zero)
- `1 < p < 2` → **Compound Poisson-Gamma**: a genuine mixture that produces an exact point mass at zero *plus* a continuous positive distribution — mechanically, this arises from "a Poisson-distributed number of Gamma-distributed events summed together," which is a very natural generative story for conflict fatalities (a Poisson-ish number of violent incidents in a week, each contributing a Gamma-distributed fatality count).

**Why this is a principled choice, not just a fancier-sounding one:** unlike log-transforming the target and using ordinary squared error (Section 9.6), the Tweedie loss respects the mean-variance relationship of the data directly during training, rather than trying to approximately linearize it through a transform. In practice, this project treats it as a testable hypothesis (per the "let the evaluation decide" principle), compared directly against squared-error models on the same walk-forward folds — not adopted a priori.

### 9.6 Log-target transformation
```
train on:      log_target = log(1 + y)
predict, then invert:   y_hat = exp(log_target_hat) - 1
```
**Why `log(1+y)` and not `log(y)`:** the target contains zeros, and `log(0)` is undefined (`-infinity`). Adding 1 before the log (the `log1p` transform) keeps zero mapped to zero while still compressing the heavy right tail.

**Effect:** squared-error loss on the log scale penalizes *relative* errors rather than *absolute* ones — a miss of 5 fatalities on a base of 2 (150% relative error) is penalized far more than a miss of 5 fatalities on a base of 500 (1% relative error). This is usually a better match for how forecast quality "should" be judged on a heavy-tailed target, since it prevents the handful of extreme-fatality weeks from single-handedly dominating the training loss the way they would under raw squared error.

**Trade-off to be aware of:** back-transforming via `exp(x) - 1` on a squared-error-optimized log-scale prediction is a **biased** estimator of the mean of the original-scale target (a known result — Jensen's inequality: `E[exp(X)] >= exp(E[X])` for any random variable `X`). In practice this project doesn't apply a bias-correction factor (e.g., a smearing estimator), which is worth stating explicitly as a known limitation of this specific technique rather than treating the inverted predictions as unbiased.

---

## 10. Evaluation Metrics: Full Theory

### 10.1 Mean Absolute Error (MAE) — primary metric
```
MAE = (1/n) * sum( |y_i - y_hat_i| )
```
- Same units as the target (fatalities) → directly interpretable ("on average, off by X fatalities").
- The **population median** is the value that minimizes expected MAE — meaning MAE-optimal models are naturally robust to the extreme spikes in this dataset, since they aren't pulled as hard toward outliers as squared-error-optimal models are.
- **Chosen as the primary metric** specifically because of the target's heavy skew (Section 2.2): a metric that doesn't get dominated by a handful of extreme rows gives a more representative picture of "typical" forecast quality.

### 10.2 Root Mean Squared Error (RMSE) — secondary metric
```
RMSE = sqrt( (1/n) * sum( (y_i - y_hat_i)^2 ) )
```
- The **population mean** is the value that minimizes expected squared error — RMSE-optimal predictions are pulled toward the mean, which is heavily influenced by outliers in a skewed distribution.
- RMSE >= MAE always (a consequence of Jensen's inequality applied to the convexity of the square function), and the *gap* between RMSE and MAE is itself informative: a large RMSE-to-MAE ratio signals that a small number of large errors are driving overall error, which is expected and worth explicitly reporting given this project's extreme-spike target distribution.
- **Chosen as a secondary, not primary, metric** — it's the right tool for answering "how bad are our worst misses," which MAE alone can't tell you, but using it as the sole metric would make the score overly sensitive to the rare extreme rows.

### 10.3 Why MAPE is deliberately NOT used
```
MAPE = (1/n) * sum( |y_i - y_hat_i| / |y_i| ) * 100
```
Mathematically undefined (division by zero) whenever `y_i = 0` — which describes the **majority** of rows in this dataset (median = 0). This isn't a minor edge case here; it makes MAPE structurally unusable for this specific target distribution, which is why it's absent from the evaluation code despite being a common regression metric elsewhere.

### 10.4 Metric summary table

| Metric | Optimal point estimate | Sensitive to outliers? | Role here |
|---|---|---|---|
| MAE | Median | Low | Primary — typical error |
| RMSE | Mean | High | Secondary — worst-case sensitivity |
| MAPE | — | Undefined at y=0 | Not used |

---

## 11. Risk Classification Layer

### 11.1 Label definition
```
high_risk_next_week(i,t) = 1  if  next_week_fatalities(i,t) > threshold(i,t)
                          = 0  otherwise

threshold(i,t) = expanding 90th percentile of country i's own
                 historical FATALITIES, computed using only data
                 up to week t (shifted back 1 step to avoid leakage)
```
**Why a country-specific, expanding threshold and not a fixed global number:** a fixed global fatality threshold (e.g., "high-risk if > 50") would be unfair across heterogeneous countries — a number that's an extreme outlier for a historically calm country might be an ordinary week for a country in active, sustained conflict. An **expanding quantile** (recomputed using only data available up to that point in time) self-calibrates per country and grows more precise as more history accumulates, without injecting an arbitrary global constant. This also explicitly avoids an earlier proposed rule (`prediction > 1.5 × recent average`), which the project rejected as an unjustified, arbitrary multiplier with no statistical grounding.

### 11.2 Classifier and class imbalance
```
RandomForestClassifier(class_weight='balanced')
```
By construction, only ~10% of rows are labeled high-risk (the 90th-percentile definition guarantees this). This is a classic **imbalanced classification** setting, where a naive classifier can achieve high accuracy just by always predicting the majority class.

`class_weight='balanced'` reweights the loss function inversely proportional to class frequency:
```
weight(class c) = n_samples / (n_classes * n_samples_in_class_c)
```
This makes misclassifying the rare high-risk class more costly during training, without needing to physically resample the data (compare to SMOTE, which synthesizes new minority-class examples — a heavier-handed and less interpretable fix than reweighting for this scale of imbalance).

### 11.3 Why ROC-AUC and not accuracy
Accuracy is misleading here — a classifier that *always* predicts "not high-risk" would already score ~90% accuracy while being useless. Instead:
```
ROC-AUC = P( score(random positive example) > score(random negative example) )
```
This measures the model's ability to **rank** high-risk weeks above normal weeks across every possible decision threshold, independent of class balance — the appropriate metric for an early-warning/alerting system where the actual operating threshold (how conservative vs. aggressive the alert should be) is a separate downstream decision from model quality itself. Precision/recall at a chosen threshold is reported alongside it, since for an alert system, missed escalations (false negatives) are typically more costly than false alarms (false positives) — a trade-off ROC-AUC alone doesn't communicate, which is why the project also uses a full classification report.

---

## 12. Explainability: SHAP Theory

### 12.1 Foundation: Shapley values (cooperative game theory)
SHAP (SHapley Additive exPlanations) is built on the **Shapley value**, originally from cooperative game theory: given a "game" with `n` players who jointly produce some payoff, the Shapley value fairly distributes that payoff among players based on their **average marginal contribution** across every possible coalition (ordering) of players.

Applied to ML: "players" = features, "payoff" = the difference between the model's prediction for a specific row and the model's average prediction over the dataset. The Shapley value for feature `j` on instance `x` is:
```
phi_j = sum over all subsets S not containing j of:
    [ |S|! * (n - |S| - 1)! / n! ] * [ f(S ∪ {j}) - f(S) ]
```
i.e., the feature's contribution, averaged over every possible order in which features could be "added" to the prediction.

### 12.2 The additivity property
```
f(x) = base_value + sum over all features j of phi_j
```
This is what makes SHAP genuinely useful for this project: every individual prediction can be exactly decomposed into "the average prediction" plus "how much each of `fatalities_lag1`, `fatalities_roll_std`, etc. pushed this specific prediction up or down" — with the contributions guaranteed to sum exactly to the model's actual output (unlike some other feature-attribution heuristics, which don't guarantee this).

### 12.3 TreeSHAP
Computing exact Shapley values naively is exponential in the number of features (`2^n` coalitions). **TreeSHAP** is a polynomial-time algorithm specific to tree-based models (Random Forest, XGBoost, LightGBM) that computes *exact* Shapley values by exploiting the tree structure directly, rather than the approximate sampling-based methods needed for arbitrary black-box models (e.g., KernelSHAP). This is why this project's `SHAP_explain.py` uses `shap.TreeExplainer` specifically — it's exact and fast for exactly the model types used here.

**Why SHAP over simpler alternatives:** built-in `feature_importances_` (Random Forest's mean-decrease-in-impurity importance) only gives a single **global** ranking and is known to be biased toward high-cardinality/continuous features; permutation importance is global-only too. SHAP additionally gives **per-prediction, per-feature** explanations — necessary for answering "why did the model flag Syria as high-risk *this specific week*," not just "which features matter on average."

---

## 13. Spatial Modeling: Theory

### 13.1 Motivation — Tobler's First Law of Geography
> "Everything is related to everything else, but near things are more related than distant things."

Conflict frequently spills across administrative borders — violence in a neighboring province is a meaningful predictor of near-term local risk that a purely country-level, purely temporal model cannot see. The ADMIN1-level extension exists to capture this.

### 13.2 ADMIN1 panel construction
Identical logic to Section 5, but grouped by `[WEEK, COUNTRY, ADMIN1]` instead of `[WEEK, COUNTRY]` — a finer spatial grain, same temporal methodology.

### 13.3 Spatial lag feature via nearest neighbors
```
spatial_lag(region i, week t) = mean( FATALITIES(j, t-1)
                                       for j in the k nearest
                                       neighboring regions to i )
```
Implemented with a **BallTree** — a space-partitioning data structure that supports efficient k-nearest-neighbor queries (`O(log n)` average query time vs. `O(n)` for brute-force distance comparison against every other region) — using the **haversine metric**, since region centroids are given as latitude/longitude on a sphere, not points on a flat Euclidean plane.

### 13.4 Haversine distance — the correct metric for lat/lon data
```
a = sin^2((lat2-lat1)/2) + cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2)
d = 2 * R * arcsin( sqrt(a) )
```
where `R` is Earth's radius, and `lat`/`lon` are in radians.

**Why haversine and not plain Euclidean distance on raw lat/lon values:** latitude and longitude are angular coordinates on a sphere, not a flat grid — one degree of longitude represents a very different physical distance near the equator versus near the poles. Euclidean distance on raw coordinates would badly distort real-world proximity, especially at the range of latitudes the Middle East spans; haversine distance correctly accounts for the Earth's curvature and returns true great-circle distance.

**Why BallTree and not brute-force distance computation:** at a few thousand region-week rows, brute-force is tractable, but BallTree scales far better if the panel grows (more countries, more admin regions, or higher-frequency data) — and it's the standard `scikit-learn` structure that natively supports the haversine metric for exactly this kind of geospatial nearest-neighbor query.

### 13.5 An explicit caution about combining spatial and temporal validation
Testing "does the model generalize to a future *time period*" (Section 14) and testing "does the model generalize to a *held-out region it's never seen*" (a spatial leave-one-region-out split) are different questions with different validation designs. This project does not conflate them — each should be reported as a separate, clearly-labeled result if both are run.

---

## 14. Validation Strategy: Theory

### 14.1 Chronological holdout (baseline validation)
The single train/before-cutoff, test/after-cutoff split from Section 8.3. Answers: "how good is the model at one specific point in history?" Cheap, but a single split can be an unrepresentative sample of one particular regime (e.g., an unusually calm or unusually volatile period).

### 14.2 Walk-forward (rolling-origin) validation
```
for each fold:
    train on all weeks before cutoff_k
    test  on the next block of weeks after cutoff_k
    advance cutoff_k forward
    repeat
```
This produces **multiple** MAE/RMSE estimates across different historical windows rather than one. Reporting both the **average** performance and the **spread/variance across folds** answers a materially different and more useful question than a single holdout: not just "is this model good," but "is this model *consistently* good, or does it fall apart during specific regimes (e.g., escalation periods)?" This is the standard, textbook-correct way to validate a time-series model, analogous to k-fold cross-validation's role for i.i.d. data — but respecting temporal order, unlike standard k-fold.

### 14.3 Out-of-time testing (the strongest test in the project)
```
1. Freeze a model trained ONLY on data through a fixed historical cutoff.
2. Do not touch or retune it.
3. Obtain genuinely later data (collected after the model was frozen).
4. Run the frozen model's predictions on that later data.
5. Compare its error there against its error on the original test set.
```
**Why this is qualitatively stronger than walk-forward validation:** walk-forward validation still evaluates the model *within the same historical dataset* it was developed against — there's always a risk that decisions made while building the pipeline (feature choices, hyperparameters, even which validation scheme to trust) were implicitly, even unconsciously, shaped by repeatedly looking at that same historical data. An out-of-time test on data that plainly did not exist yet when the model was frozen is the closest thing to a genuine, unbiased test of real-world forecasting ability — it directly answers "would this model actually have worked, going forward, in practice?" A materially worse out-of-time MAE than the historical test MAE is itself a valid and important finding (evidence of non-stationarity — Section 2.3), not a failure to hide.

---

## 15. Results & How to Interpret Them

Actual numbers depend on your specific data extract and split, but the pipeline is designed to always produce a leaderboard like:

```
Model                          MAE     RMSE
Naive baseline (persistence)   ?.??    ?.??
Ridge Regression                ?.??    ?.??
Random Forest                   ?.??    ?.??
XGBoost / LightGBM              ?.??    ?.??
LightGBM (Tweedie objective)    ?.??    ?.??
```

**How to read this table honestly:**
- If Naive has the *lowest* MAE, that is a legitimate, reportable finding: **the available features and models don't yet add predictive value beyond simple persistence.** This is common and expected in conflict forecasting and should never be hidden or reframed.
- If a model beats Naive, report the **percentage improvement**, not just the raw numbers — e.g., "Random Forest reduced MAE by X% versus the naive baseline."
- Always report the **out-of-time result** (Section 14.3) alongside the historical-holdout result — a model that wins historically but degrades badly out-of-time tells a more complete and more honest story than either number alone.

---

## 16. Limitations

- **Extreme spikes are structurally hard to forecast** — tree-based models in particular struggle to extrapolate beyond the magnitude of spikes seen in training data; a genuinely unprecedented escalation will likely be under-predicted by any model here.
- **Non-stationarity** — conflict dynamics shift due to political events, ceasefires, and external intervention; a model fit to one period is not guaranteed to generalize to a materially different regime.
- **No exogenous predictors** — the feature set is purely endogenous (derived only from past fatalities/events); no political, economic, diplomatic, or military-movement signals are incorporated, which bounds the ceiling of achievable accuracy regardless of model sophistication.
- **Country-level aggregation hides local structure** — the core pipeline treats a whole country as one unit, which can mask sharply different sub-national dynamics (addressed only partially by the ADMIN1 extension, which introduces its own added complexity and data-sparsity challenges at finer grain).
- **Log-target back-transformation bias** — as noted in Section 9.6, `exp(pred) - 1` is a biased estimator of the mean on the original scale; no smearing/bias correction is currently applied.

---

## 17. How to Run — Step by Step

### 17.1 Environment setup
```bash
git clone https://github.com/namegitser/Middle-East-Conflicts.git
cd Middle-East-Conflicts

python -m venv ml_env

# Windows
ml_env\Scripts\activate
# macOS / Linux
source ml_env/bin/activate

pip install -r requirements.txt
```
Core `requirements.txt`: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `openpyxl`, `jupyter`, `pytest`.
Install separately for the advanced scripts: `pip install streamlit shap lightgbm xgboost plotly`.

### 17.2 Place your data
```
data/raw-data/<your-ACLED-style-extract>.csv
```
Must contain at minimum: `WEEK, COUNTRY, EVENTS, FATALITIES`. For spatial modeling, also `ADMIN1, CENTROID_LATITUDE, CENTROID_LONGITUDE`.

### 17.3 Core version — baseline, Ridge, Random Forest, evaluation
```bash
python src/evaluations.py
```
Runs the full pipeline: load → aggregate → complete panel → features → target → chronological split → naive/Ridge/Random Forest → prints the MAE/RMSE leaderboard (Section 15).

### 17.4 Core version — risk classification
```bash
python src/risk_classifier.py
```
Builds the `high_risk_next_week` label (Section 11.1) and reports ROC-AUC and a full classification report.

### 17.5 Advanced version — gradient boosting comparison
```bash
python src/gradboost_comparison.py
```
Extends the leaderboard with Random Forest (log-target), XGBoost, LightGBM, and LightGBM-Tweedie (Section 9.4–9.6), all evaluated on the same split for a fair comparison.

### 17.6 Advanced version — walk-forward validation
```bash
python src/validation.py
```
Runs the rolling-origin validation described in Section 14.2 and reports average MAE/RMSE across folds plus their spread.

### 17.7 Advanced version — out-of-time test
```bash
python src/out_of_time_test.py
```
Freezes a model on your historical cutoff, evaluates it on genuinely later data (Section 14.3), then retrains a production model on the full expanded dataset. Requires a second, later data extract to be present.

### 17.8 Advanced version — explainability
```bash
python src/SHAP_explain.py
```
Trains (or loads) a tree model and computes exact TreeSHAP values (Section 12) — produces both a global summary plot and per-prediction breakdowns.

### 17.9 Advanced version — spatial modeling
```bash
python src/spatial_modeling.py
```
Builds the ADMIN1×WEEK panel and computes the haversine-based spatial lag feature (Section 13).

### 17.10 Dashboard
```bash
python src/dashboard_data.py     # generates data/processed/latest_forecast.parquet
streamlit run app/app.py         # launches the interactive dashboard
```

### 17.11 Fastest end-to-end path (quick start)
```bash
python -m venv ml_env && source ml_env/bin/activate
pip install -r requirements.txt
python src/evaluations.py
python src/risk_classifier.py
python src/dashboard_data.py
streamlit run app/app.py
```

---

## 18. Project Development Timeline

This project was built in two stages, each intentionally scoped to prove the previous stage was correct before adding complexity:

**Stage 1 — Core capstone system:** raw ACLED data → country-week aggregation and zero-filled panel → lag/rolling feature engineering → chronological (not random) train/test split → naive baseline vs. Ridge vs. Random Forest → honest MAE/RMSE comparison → error analysis → documented README. Two real mistakes were caught and fixed during this stage: shifting the raw event log before aggregation (Section 3.1), and splitting by row order instead of by date (Section 8.2).

**Stage 2 — Portfolio/advanced extensions:** walk-forward validation to test stability across time regimes → a frozen-model out-of-time test against genuinely future data → gradient boosting (XGBoost/LightGBM) and a Tweedie objective to better match the zero-inflated skewed target → a country-calibrated risk classification layer → SHAP explainability for per-prediction transparency → an ADMIN1-level spatial extension with a haversine-based spatial lag feature → a Streamlit dashboard for interactive delivery.

Both stages remain runnable independently (Section 17.3–17.4 for the core system, 17.5–17.9 for the extensions) — the advanced stage does not replace the core pipeline, it builds on top of it.

---

## 19. Future Work

- Bias-correct the log-target back-transformation (smearing estimator).
- Extend the risk classifier into a multi-level severity scale rather than a binary flag.
- Incorporate genuinely exogenous predictors (economic indicators, political-event calendars) where available.
- Formal leave-one-region-out spatial validation, reported separately from temporal validation (Section 13.5).
- Automate scheduled ACLED ingestion — pending verification of ACLED's terms of use for programmatic access.
- Hyperparameter tuning (e.g., Optuna) for the gradient boosting models once the current comparison ladder is stable.

---

## 20. Further Reading

- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1).
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics.
- Lundberg, S. & Lee, S-I. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP). NeurIPS.
- Tweedie, M. C. K. (1984). *An index which distinguishes between some important exponential families.*
- Hyndman, R. J. & Athanasopoulos, G. — *Forecasting: Principles and Practice* (walk-forward/rolling-origin validation reference).
- ACLED (Armed Conflict Location & Event Data Project) — [acleddata.com](https://acleddata.com) — data source and methodology documentation.
