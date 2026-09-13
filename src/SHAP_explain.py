import pandas as pd
import numpy as np
import shap
import warnings
warnings.filterwarnings('ignore')

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import  train_random_forest, FEATURES, TARGET
from split import chronological_split
if __name__ == "__main__":
    print("1. Loading and gridding data...")
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    train_df, test_df, cutoff = chronological_split(df, train_frac=0.8)
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES].fillna(0)

    print("2. Training Random Forest model...")
    rf_model = train_random_forest(train_df)

    print("3. Computing exact TreeSHAP values...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)

    # Global feature importance (mean absolute SHAP value)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        "Feature": FEATURES,
        "Mean_Abs_SHAP": mean_abs_shap
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    print("\n=== GLOBAL FEATURE IMPORTANCE (SHAP) ===")
    print(shap_importance.to_string(index=False))

    # Local sample explanation (inspecting row 0 of test set)
    sample_idx = 0
    sample_data = X_test.iloc[[sample_idx]]
    sample_shap = shap_values[sample_idx]
    
    print(f"\n=== LOCAL EXPLANATION FOR SAMPLE ROW {sample_idx} ===")
    print(f"Actual Target: {test_df[TARGET].iloc[sample_idx]}")
    print(f"Model Prediction: {rf_model.predict(sample_data)[0]:.2f}")
    local_breakdown = pd.DataFrame({
        "Feature": FEATURES,
        "Value": sample_data.values[0],
        "SHAP_Effect": sample_shap
    }).sort_values("SHAP_Effect", key=abs, ascending=False)
    print(local_breakdown.to_string(index=False))