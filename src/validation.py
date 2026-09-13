import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import naive_forecast, train_ridge, train_random_forest, FEATURES, TARGET
from evaluations import evaluate

def walk_forward_splits(df: pd.DataFrame, min_train_weeks: int = 100, step: int = 4):
    """
    Yields (train_df, test_df) pairs. 
    Train window expands; test window is the next `step` weeks (4 weeks = 1 month).
    """
    weeks = sorted(df["WEEK"].unique())
    # Ensure we don't go out of bounds
    max_idx = len(weeks) - step
    
    for i in range(min_train_weeks, max_idx, step):
        train_cutoff = weeks[i]
        test_cutoff = weeks[i + step]
        
        train_df = df[df["WEEK"] < train_cutoff].copy()
        test_df = df[(df["WEEK"] >= train_cutoff) & (df["WEEK"] < test_cutoff)].copy()
        
        yield train_df, test_df, train_cutoff, test_cutoff

if __name__ == "__main__":
    print("Loading data and engineering features...")
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv") 
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)
    
    # Store results across all folds
    fold_results = {"Naive": [], "Ridge": [], "Random Forest": []}
    
    print("\nRunning Walk-Forward Validation (Step = 4 weeks)...")
    fold_count = 0
    
    # Start with 200 weeks of history, step forward 4 weeks at a time
    for train_df, test_df, start_date, end_date in walk_forward_splits(df, min_train_weeks=200, step=4):
        fold_count += 1
        X_test = test_df[FEATURES].fillna(0)
        y_true = test_df[TARGET]
        
        # Train models for this specific fold
        ridge_model = train_ridge(train_df)
        rf_model = train_random_forest(train_df)
        
        # Evaluate
        fold_results["Naive"].append(mean_absolute_error(y_true, naive_forecast(test_df)))
        fold_results["Ridge"].append(mean_absolute_error(y_true, ridge_model.predict(X_test)))
        fold_results["Random Forest"].append(mean_absolute_error(y_true, rf_model.predict(X_test)))

    print(f"\n=== WALK-FORWARD RESULTS ({fold_count} FOLDS) ===")
    
    for model_name, metrics in fold_results.items():
        avg_mae = np.mean(metrics)
        std_mae = np.std(metrics)
        print(f"{model_name:15}: Average MAE = {avg_mae:.2f} (± {std_mae:.2f})")