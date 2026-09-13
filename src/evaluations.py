import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    # Using np.sqrt ensures compatibility across all scikit-learn versions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"model": label, "MAE": mae, "RMSE": rmse}

def comparison_table(results: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)





if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore') # Suppress pandas future warnings for clean output
    
    # Import everything we've built so far
    from data_processing import clean_and_grid
    from features import add_lag_features, add_rolling_features, add_target
    from models import (
        naive_forecast, train_ridge, train_random_forest, 
        FEATURES, TARGET
    )
    from split import chronological_split

    print("1. Loading and cleaning data...")
    # Update this path if your CSV is named differently
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv") 

    print("2. Engineering features...")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    print("3. Splitting into Train/Test (chronological)...")
    train_df, test_df, cutoff = chronological_split(df, train_frac=0.8)
    print(f"   Training on data before {cutoff.date()}")
    print(f"   Testing on data from {cutoff.date()} onwards")

    print("4. Training models...")
    ridge_model = train_ridge(train_df)
    rf_model = train_random_forest(train_df)

    print("5. Evaluating models on Test set...\n")
    
    # Generate predictions
    y_true = test_df[TARGET]
    
    # We fillna(0) on features because the first few weeks of the dataset 
    # will have NaN for their rolling averages/lags.
    X_test = test_df[FEATURES].fillna(0)
    
    # Compile results
    results = [
        evaluate(y_true, naive_forecast(test_df), "Naive Baseline"),
        evaluate(y_true, ridge_model.predict(X_test), "Ridge Regression"),
        evaluate(y_true, rf_model.predict(X_test), "Random Forest"),
    ]

    leaderboard = comparison_table(results)
    print("=== FINAL RESULTS ===")
    print(leaderboard)