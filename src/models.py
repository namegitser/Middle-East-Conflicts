
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

FEATURES = [
    "fatalities_lag1", "fatalities_lag2", "fatalities_lag3",
    "events_lag1", "fatalities_roll_mean", "fatalities_roll_std",
]
TARGET = "next_week_fatalities"

def naive_forecast(df):
    """Prediction(t+1) = Fatalities(t). No fitting required."""
    return df["FATALITIES"]

def train_ridge(train_df):
    X, y = train_df[FEATURES].fillna(0), train_df[TARGET]
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model

def train_random_forest(train_df):
    X, y = train_df[FEATURES].fillna(0), train_df[TARGET]
    model = RandomForestRegressor(
        n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
    )
    model.fit(X, y)
    return model