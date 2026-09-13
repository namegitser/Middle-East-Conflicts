import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import train_random_forest, FEATURES, TARGET
from split import chronological_split
from evaluations import evaluate

if __name__ == "__main__":
    # 1. Rebuild and train the winning model
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv") 
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)
    
    train_df, test_df, cutoff = chronological_split(df, train_frac=0.8)
    rf_model = train_random_forest(train_df)
    
    # 2. Build the error analysis dataframe
    error_df = test_df[["COUNTRY", "WEEK", TARGET]].copy()
    error_df["predicted"] = rf_model.predict(test_df[FEATURES].fillna(0))
    # Round predictions to whole numbers (you can't have half a fatality)
    error_df["predicted"] = error_df["predicted"].round(0)
    error_df["abs_error"] = (error_df[TARGET] - error_df["predicted"]).abs()

    # 3. Analyze the worst misses
    print("=== TOP 10 WORST PREDICTIONS ===")
    worst = error_df.sort_values("abs_error", ascending=False).head(10)
    print(worst.to_string(index=False))

    # 4. Analyze performance on extreme spikes (Top 5% most violent weeks)
    threshold = test_df[TARGET].quantile(0.95)
    high_severity = error_df[error_df[TARGET] >= threshold]
    low_severity = error_df[error_df[TARGET] < threshold]
    
    print("\n=== PERFORMANCE BY SEVERITY ===")
    print(f"95th Percentile Threshold: {threshold:.0f} fatalities")
    print(f"MAE on Normal Weeks (Bottom 95%): {low_severity['abs_error'].mean():.2f}")
    print(f"MAE on Spike Weeks (Top 5%): {high_severity['abs_error'].mean():.2f}")