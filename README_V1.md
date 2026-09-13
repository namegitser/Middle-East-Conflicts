# Conflict Forecasting & Risk Analysis: Complete Project Journey

## 1. Project Inception & Objective
The objective of this project was to build an end-to-end machine learning pipeline capable of predicting next-week, country-level conflict fatalities in the Middle East. The core challenge was to determine whether historical conflict activity (recent lags and rolling averages) contains enough predictive signal to mathematically outperform a "Naive Baseline"—the simple assumption that next week's fatality count will be identical to this week's.

## 2. Data Sourcing & Characteristics
The project utilized data from the Armed Conflict Location & Event Data Project (ACLED), specifically focusing on the Middle East region. 

**Raw Data Structure:** The raw dataset was an *event log*. It contained rows only for specific incidents (e.g., battles, explosions, protests), capturing the date, location, event type, and resulting fatalities. 
**Inherent Challenge:** Event logs are sparse. If a country experienced a peaceful week, there was no row in the dataset indicating zero fatalities; the week simply did not exist in the data.

## 3. Data Transformation & Feature Engineering
To apply supervised machine learning to a time-series problem, the raw event log required rigorous structural transformation.

### The "Gridding" Process (Panel Balancing)
**What we did:** We collapsed the daily event rows into weekly aggregates per country. We then generated a complete Cartesian product grid of `[All Unique Dates] × [All Countries]` present in the data. Any Country-Week combination missing from the raw log was explicitly filled with `0` events and `0` fatalities.
**Why we did it:** Time-series algorithms rely on a constant time step ($\Delta t$). If we extract a "previous week" feature ($y_{t-1}$) without zero-filling peaceful weeks, the algorithm silently bridges massive temporal gaps, assuming a conflict from a month ago happened yesterday. Zero-filling enforces the assumption that an absence of records equals an absence of conflict, preventing fatal temporal leakage.

### Temporal Feature Engineering
We engineered the following features to serve as the model's short-term memory and momentum indicators:
*   **Autoregressive Lags ($y_{t-1}, y_{t-2}, y_{t-3}$):** The fatality counts from the preceding three weeks.
*   **Cross-Feature Lag:** The total count of distinct conflict *events* from $t-1$.
*   **Rolling Statistics ($\mu_{t-1 \to t-4}, \sigma_{t-1 \to t-4}$):** The 4-week historical rolling mean and standard deviation.
**Why we did it:** War is highly persistent (an Autoregressive process). These features mathematically encode immediate volatility, mid-term momentum, and the general baseline of violence without leaking data from the target week.

### Target Shifting & Chronological Splitting
We shifted the target variable backwards by one step ($\hat{y}_{t+1}$), mapping the known state of week $t$ to the outcome of week $t+1$. 
To prevent future-data leakage, we split the data strictly by time. Models trained exclusively on data prior to December 9, 2023, and were evaluated solely on data from that date onward.

## 4. Machine Learning Models
We tested three distinct approaches to establish a hierarchy of predictive power:

*   **The Naive Baseline ($\hat{y}_{t+1} = y_t$):** The assumption that next week will mirror this week. This is the gold standard benchmark in conflict forecasting due to the high autocorrelation of violence.
*   **Ridge Regression ($L_2$ Penalty):** A linear model. Because our lag features are highly collinear (Lag 1 is highly correlated with Lag 2 and the Rolling Mean), standard Ordinary Least Squares (OLS) would become unstable. Ridge applies an $L_2$ penalty to shrink redundant coefficients, yielding a stable, interpretable linear baseline.
*   **Random Forest Regressor:** An ensemble of 300 decision trees. Conflict escalation is non-linear (e.g., violence might only spike if rolling variance crosses a specific threshold). Random Forests capture these non-linear interactions, require no feature scaling, and are robust to the extreme right-skew of fatality counts.

## 5. Evaluation Metrics & Results
We evaluated the models using two metrics because the underlying data is heavily right-skewed (mostly zeros, punctuated by massive spikes in the thousands).

*   **Mean Absolute Error (MAE):** Scales linearly. Provides a reliable measure of "everyday" accuracy without being completely hijacked by a single massive outlier.
*   **Root Mean Squared Error (RMSE):** Squares the errors. Acts as a strict penalty metric for models that fail to predict massive, sudden outbreaks of violence.

**Final Leaderboard (Test Set):**

| Model | MAE | RMSE |
| :--- | :--- | :--- |
| **Random Forest Regressor** | **45.73** | **665.94** |
| Naive Baseline | 48.87 | 928.21 |
| Ridge Regression | 58.13 | 773.49 |

**Conclusion:** The Random Forest successfully extracted predictive, non-linear signals from the historical lags, decisively beating the Naive Baseline in both MAE and RMSE. Ridge Regression's failure to beat the baseline MAE proved that linear models are insufficient for modeling conflict escalation dynamics.

## 6. Bottlenecks & Limitations (Error Analysis)
Deep-dive error analysis revealed a massive discrepancy in the Random Forest's performance based on the severity of the conflict. The model exhibits a severe "Spike Blindspot."

**The Bimodal Accuracy Profile:**
*   **Normal Weeks (Bottom 95% of data):** Highly accurate (MAE ~21 fatalities).
*   **Extreme Spikes (Top 5% of data):** Systematically fails (MAE ~513 fatalities).

**Underlying Causes of the Bottleneck:**
1.  **Extrapolation Failure:** A Random Forest operates by averaging the leaves of decision trees. It is mathematically bounded by its training set and cannot output a prediction higher than the maximum historical value it observed. If a new geopolitical shock causes 27,000 fatalities, but the training data maxed out at 2,000, the model cannot predict the true magnitude.
2.  **Mean Reversion:** When faced with an unprecedented spike, ensemble tree methods average predictions across all trees, naturally dampening the extreme values and pulling the prediction toward the mean.
3.  **Autoregressive Limits:** The model relies entirely on historical fatality counts. If a sudden structural break (e.g., a surprise invasion) occurs with no prior escalation in the preceding 4 weeks, the model is completely blind to it. It acts as an excellent tracker of ongoing, grinding conflicts, but not a forecaster of sudden geopolitical shocks.