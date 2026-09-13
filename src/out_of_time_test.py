import os
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import train_random_forest, FEATURES, TARGET
from split import chronological_split

os.makedirs("models", exist_ok=True)

print("1. Loading historical data through 7 March 2026...")
df_hist = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
df_hist = add_lag_features(df_hist)
df_hist = add_rolling_features(df_hist)
df_hist = add_target(df_hist)

# Split to get exact train set up to cutoff
train_df, _, cutoff = chronological_split(df_hist, train_frac=0.8)

print("2. Training and freezing historical Random Forest model...")
rf_historical = train_random_forest(train_df)
joblib.dump(rf_historical, "models/rf_historical_2026-03-07.pkl")

print("3. Loading out-of-time (OOT) future data (8 Mar → 29 Aug 2026)...")
oot_raw_path = "data/raw-data/MEdata_08-03-26_to_29-08-26.csv"

if not os.path.exists(oot_raw_path):
    print(f"\n[NOTICE] File '{oot_raw_path}' not found.")
    print("If you haven't downloaded the post-March 2026 ACLED export yet, drop it into data/raw-data/ and re-run.")
else:
    df_oot = clean_and_grid(oot_raw_path)
    df_oot = add_lag_features(df_oot)
    df_oot = add_rolling_features(df_oot)
    df_oot = add_target(df_oot)

    print("4. Evaluating frozen model on OOT future data...")
    X_oot = df_oot[FEATURES].fillna(0)
    y_oot = df_oot[TARGET]
    oot_preds = rf_historical.predict(X_oot)

    oot_mae = mean_absolute_error(y_oot, oot_preds)
    oot_rmse = np.sqrt(mean_squared_error(y_oot, oot_preds))

    print(f"\n=== OUT-OF-TIME RESULTS (Mar–Aug 2026) ===")
    print(f"OOT MAE  : {oot_mae:.2f}")
    print(f"OOT RMSE : {oot_rmse:.2f}")

    print("\n5. Retraining production model on full history through August 2026...")
    # Combine historical and OOT data for the new production baseline
    full_df = pd.concat([df_hist, df_oot], ignore_index=True)
    full_df = full_df.drop_duplicates(subset=["WEEK", "COUNTRY"]).sort_values(["COUNTRY", "WEEK"])
    
    # Re-apply target shift across combined timeline
    full_df["next_week_fatalities"] = full_df.groupby("COUNTRY")["FATALITIES"].shift(-1)
    full_df_clean = full_df.dropna(subset=["next_week_fatalities"])
    
    X_full = full_df_clean[FEATURES].fillna(0)
    y_full = full_df_clean[TARGET]
    
    rf_production = train_random_forest(pd.concat([X_full, y_full], axis=1)) # Or direct fit
    joblib.dump(rf_production, "models/rf_production_latest.pkl")
    print("Saved updated production model to 'models/rf_production_latest.pkl'.")