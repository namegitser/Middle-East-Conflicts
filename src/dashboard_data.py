import pandas as pd
import joblib
from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import FEATURES

if __name__ == "__main__":
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    # Load production model (or retrain RF)
    from sklearn.ensemble import RandomForestRegressor
    X = df[FEATURES].fillna(0)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X, df["FATALITIES"]) # Or train on historical subset
    
    df["predicted_fatalities"] = rf.predict(X).round(1)
    
    import os
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/latest_forecast.parquet", index=False)
    print("Dashboard dataset generated successfully.")