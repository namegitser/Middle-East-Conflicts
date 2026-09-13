import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from data_processing import clean_and_grid
from features import add_lag_features, add_rolling_features, add_target
from models import naive_forecast, FEATURES, TARGET
from split import chronological_split
def evaluate_preds(y_true, y_pred, label: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"Model": label, "MAE": mae, "RMSE": rmse}

if __name__ == "__main__":
    print("1. Loading and gridding data...")
    # Use your complete dataset path here
    df = clean_and_grid("data/raw-data/MEdata_07-03-26.csv")
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_target(df)

    train_df, test_df, cutoff = chronological_split(df, train_frac=0.8)
    X_train, y_train = train_df[FEATURES].fillna(0), train_df[TARGET]
    X_test, y_test = test_df[FEATURES].fillna(0), test_df[TARGET]

    results = []

    # 1. Naive Baseline
    results.append(evaluate_preds(y_test, naive_forecast(test_df), "Naive Baseline"))

    # 2. Ridge Regression
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    results.append(evaluate_preds(y_test, ridge.predict(X_test), "Ridge Regression"))

    # 3. Random Forest (Baseline benchmark)
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, y_train)
    results.append(evaluate_preds(y_test, rf.predict(X_test), "Random Forest"))

    # 4. Random Forest with log1p target transformation
    log_y_train = np.log1p(y_train)
    rf_log = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X_train, log_y_train)
    rf_log_preds = np.expm1(rf_log.predict(X_test))
    results.append(evaluate_preds(y_test, rf_log_preds, "Random Forest (log1p)"))

    # 5. XGBoost Regressor
    xgb = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=-1)
    xgb.fit(X_train, y_train)
    results.append(evaluate_preds(y_test, xgb.predict(X_test), "XGBoost"))

    # 6. LightGBM Standard (MSE objective)
    lgb_std = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1)
    lgb_std.fit(X_train, y_train)
    results.append(evaluate_preds(y_test, lgb_std.predict(X_test), "LightGBM (MSE)"))

    # 7. LightGBM Tweedie objective (Count-data aware)
    lgb_tweedie = lgb.LGBMRegressor(
        objective="tweedie", 
        tweedie_variance_power=1.3, 
        n_estimators=300, 
        learning_rate=0.05, 
        random_state=42, 
        verbose=-1
    )
    lgb_tweedie.fit(X_train, y_train)
    results.append(evaluate_preds(y_test, lgb_tweedie.predict(X_test), "LightGBM (Tweedie)"))

    leaderboard = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    print("\n=== MODEL COMPARISON LEADERBOARD ===")
    print(leaderboard.to_string(index=False))